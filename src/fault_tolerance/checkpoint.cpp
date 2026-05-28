// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/fault_tolerance/checkpoint.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace symplex::fault_tolerance {

// ── Constructor ─────────────────────────────────────────────────────────

CheckpointPlanner::CheckpointPlanner(int64_t available_memory_bytes)
    : available_memory_bytes_(available_memory_bytes)
{
    if (available_memory_bytes_ < 0) {
        available_memory_bytes_ = 0;
    }
}

// ── available_memory ────────────────────────────────────────────────────

int64_t CheckpointPlanner::available_memory() const {
    return available_memory_bytes_;
}

// ── decide (single-layer) ───────────────────────────────────────────────

CheckpointDecision CheckpointPlanner::decide(
    int64_t activation_bytes,
    int64_t recompute_ns,
    int64_t remaining_memory
) const {
    // If there is not enough memory, we must recompute.
    if (remaining_memory < activation_bytes) {
        return CheckpointDecision::RECOMPUTE;
    }

    // Compare recompute wall-clock cost against memory transfer cost.
    // Memory transfer cost = activation_bytes * memory_cost_per_byte (HBM write + read).
    // We use a factor of 2 because the activation is written during forward
    // and read back during backward.
    double memory_transfer_cost_ns =
        2.0 * static_cast<double>(activation_bytes) * memory_cost_per_byte_ns_;

    double recompute_cost_ns = static_cast<double>(recompute_ns);

    // SAVE if recomputing is more expensive than the memory transfer;
    // RECOMPUTE if the memory transfer overhead exceeds recomputation.
    if (recompute_cost_ns > memory_transfer_cost_ns) {
        return CheckpointDecision::SAVE;
    }
    return CheckpointDecision::RECOMPUTE;
}

// ── plan (full model) ───────────────────────────────────────────────────

CheckpointPlan CheckpointPlanner::plan(
    const std::vector<int64_t>& layer_activation_bytes,
    const std::vector<int64_t>& layer_recompute_ns
) const {
    CheckpointPlan result;
    const size_t n = layer_activation_bytes.size();

    if (layer_recompute_ns.size() != n) {
        return result; // Mismatched inputs – return empty plan
    }

    if (n == 0) {
        result.total_memory_bytes = 0;
        result.total_recompute_ns = 0;
        result.memory_savings_percent = 0.0;
        return result;
    }

    // ── Step 1: Compute all-SAVE baseline and per-layer saving_ratio ────

    int64_t baseline_memory = 0;
    for (size_t i = 0; i < n; ++i) {
        baseline_memory += layer_activation_bytes[i];
    }

    result.decisions.resize(n);
    for (size_t i = 0; i < n; ++i) {
        // saving_ratio = memory_bytes / recompute_ns
        // Higher ratio → more memory freed per unit of recompute → better
        // recompute candidate.
        double ratio = 0.0;
        if (layer_recompute_ns[i] > 0) {
            ratio = static_cast<double>(layer_activation_bytes[i])
                  / static_cast<double>(layer_recompute_ns[i]);
        }
        // Layers with zero recompute cost (e.g. pure input fetches) get
        // infinite ratio – they are the absolute best recompute candidates.
        else if (layer_activation_bytes[i] > 0) {
            ratio = 1e18; // Practically infinite
        }

        result.decisions[i] = {
            static_cast<int64_t>(i),
            CheckpointDecision::SAVE,
            layer_activation_bytes[i],
            layer_recompute_ns[i],
            ratio
        };
    }

    int64_t total_mem = baseline_memory;

    // If everything fits within the budget, no decisions to change.
    if (total_mem <= available_memory_bytes_) {
        result.total_memory_bytes = total_mem;
        result.total_recompute_ns = 0;
        result.memory_savings_percent = 0.0;
        return result;
    }

    // ── Step 2: Sort layer indices by saving_ratio descending ───────────
    // Best recompute candidates first (most memory freed per compute ns).

    std::vector<size_t> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), size_t{0});
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&](size_t a, size_t b) {
            return result.decisions[a].saving_ratio > result.decisions[b].saving_ratio;
        });

    // ── Step 3: Greedily convert best candidates to RECOMPUTE ───────────

    for (size_t idx : sorted_indices) {
        if (total_mem <= available_memory_bytes_) {
            break;
        }
        // Skip layers with zero activation (nothing to free).
        if (layer_activation_bytes[idx] == 0) {
            continue;
        }
        result.decisions[idx].decision = CheckpointDecision::RECOMPUTE;
        total_mem -= layer_activation_bytes[idx];
    }

    // ── Step 4: OFFLOAD optimisation ────────────────────────────────────
    // For each RECOMPUTE layer, check whether offloading to CPU memory
    // would be cheaper than recomputing.  PCIe bandwidth (~32 GB/s) is
    // roughly 100× slower than HBM (~3.3 TB/s), so the per-byte offload
    // cost is estimated as 100× the HBM per-byte cost.
    // A round-trip (write to CPU, read back during backward) costs 2×
    // the offload transfer time.

    const double offload_cost_per_byte_ns = memory_cost_per_byte_ns_ * 100.0;

    for (size_t i = 0; i < n; ++i) {
        if (result.decisions[i].decision != CheckpointDecision::RECOMPUTE) {
            continue;
        }
        // Estimate round-trip offload transfer cost.
        double offload_transfer_ns =
            2.0 * static_cast<double>(layer_activation_bytes[i]) * offload_cost_per_byte_ns;
        double recompute_cost_ns = static_cast<double>(layer_recompute_ns[i]);

        // If offloading is cheaper than recomputing, switch to OFFLOAD.
        if (offload_transfer_ns < recompute_cost_ns) {
            result.decisions[i].decision = CheckpointDecision::OFFLOAD;
        }
    }

    // ── Step 5: OFFLOAD safety net ──────────────────────────────────────
    // After recompute + offload optimisation, if the remaining SAVE layers
    // still exceed the GPU memory budget (e.g. a single layer is larger
    // than the entire budget), OFFLOAD the largest SAVE layers until we
    // fit.  We prefer OFFLOAD over RECOMPUTE here because these layers
    // were NOT selected as good recompute candidates (they have low
    // saving_ratio, meaning recomputation is expensive relative to the
    // memory saved).

    // Recompute total_mem for only SAVE layers.
    total_mem = 0;
    for (size_t i = 0; i < n; ++i) {
        if (result.decisions[i].decision == CheckpointDecision::SAVE) {
            total_mem += layer_activation_bytes[i];
        }
    }

    if (total_mem > available_memory_bytes_) {
        // Collect remaining SAVE layer indices, sorted by activation size
        // descending (offload the largest first to free the most memory).
        std::vector<size_t> save_indices;
        save_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (result.decisions[i].decision == CheckpointDecision::SAVE) {
                save_indices.push_back(i);
            }
        }
        std::sort(save_indices.begin(), save_indices.end(),
            [&](size_t a, size_t b) {
                return layer_activation_bytes[a] > layer_activation_bytes[b];
            });

        for (size_t idx : save_indices) {
            if (total_mem <= available_memory_bytes_) {
                break;
            }
            result.decisions[idx].decision = CheckpointDecision::OFFLOAD;
            total_mem -= layer_activation_bytes[idx];
        }
    }

    // ── Step 6: Compute final aggregate statistics ──────────────────────

    result.total_memory_bytes = 0;
    result.total_recompute_ns = 0;
    for (size_t i = 0; i < n; ++i) {
        switch (result.decisions[i].decision) {
            case CheckpointDecision::SAVE:
                result.total_memory_bytes += layer_activation_bytes[i];
                break;
            case CheckpointDecision::RECOMPUTE:
                result.total_recompute_ns += layer_recompute_ns[i];
                break;
            case CheckpointDecision::OFFLOAD:
                // Activation moved to CPU – no GPU memory cost and no
                // recompute cost during backward (only PCIe transfer,
                // which is not tracked in this struct).
                break;
        }
    }

    // Memory savings as a percentage of the all-SAVE baseline.
    if (baseline_memory > 0) {
        result.memory_savings_percent =
            100.0 * static_cast<double>(baseline_memory - result.total_memory_bytes)
                 / static_cast<double>(baseline_memory);
    } else {
        result.memory_savings_percent = 0.0;
    }

    return result;
}

} // namespace symplex::fault_tolerance
