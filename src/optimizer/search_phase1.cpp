// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/search_phase1.h"
#include "symplex/schedule/tiling.h"
#include <algorithm>
#include <cmath>

namespace symplex::optimizer {

namespace {

/// Populate the cost-model fields of an ExtendedTileConfig for a
/// matmul-shaped 3-D tile.
void annotate_matmul_tile(
    ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    // Guard: must be 3-D
    if (cfg.inner_tiles.size() < 3) return;

    int64_t tm = cfg.inner_tiles[0];
    int64_t tn = cfg.inner_tiles[1];
    int64_t tk = cfg.inner_tiles[2];

    cfg.compute_ops  = compute_matmul_ops(tm, tn, tk);
    cfg.bytes_moved  = compute_matmul_bytes(tm, tn, tk, target.bytes_per_element);
    cfg.operational_intensity = cfg.calc_operational_intensity();
    cfg.compute_bound = target.is_compute_bound(cfg.compute_ops, cfg.bytes_moved);
    cfg.occupancy     = estimate_occupancy(cfg, target);
    cfg.estimated_latency_ns = estimate_latency_ns(cfg, target);

    // Initial score: prefer higher operational intensity, then higher occupancy.
    // This score will be refined in phase 2.
    cfg.score = cfg.operational_intensity + static_cast<double>(cfg.occupancy) * 0.01;
}

/// Populate the cost-model fields for a 2-D tile (e.g. outer-product
/// or contraction with one dimension folded in).
void annotate_2d_tile(
    ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    if (cfg.inner_tiles.size() < 2) return;

    int64_t tm = cfg.inner_tiles[0];
    int64_t tn = cfg.inner_tiles[1];

    // For 2-D tiles we model a generic elementwise/contraction with
    // 1 FMA per element and 2 input + 1 output tensor per element.
    int64_t tile_vol   = tm * tn;
    cfg.compute_ops    = 2 * tile_vol;   // 1 FMA per element = 2 ops
    cfg.bytes_moved    = 3 * tile_vol * target.bytes_per_element;
    cfg.operational_intensity = cfg.calc_operational_intensity();
    cfg.compute_bound  = target.is_compute_bound(cfg.compute_ops, cfg.bytes_moved);
    cfg.occupancy      = estimate_occupancy(cfg, target);
    cfg.estimated_latency_ns = estimate_latency_ns(cfg, target);
    cfg.score = cfg.operational_intensity + static_cast<double>(cfg.occupancy) * 0.01;
}

} // anonymous namespace

// ── Phase 1 implementation ───────────────────────────────────────────────

std::vector<ExtendedTileConfig> phase1_roofline_pruning(
    size_t ndim,
    const hardware::HardwareTarget& target,
    int64_t max_tensor_dim
) {
    // Step 1: enumerate all hardware-aligned tile configurations
    //         using the schedule namespace utility.
    auto base_configs = schedule::generate_hardware_aligned_tiles(
        ndim, target, max_tensor_dim);

    // Step 2: wrap each TileConfig into an ExtendedTileConfig and
    //         compute cost annotations.
    std::vector<ExtendedTileConfig> annotated;
    annotated.reserve(base_configs.size());

    for (auto& bc : base_configs) {
        ExtendedTileConfig ec(std::move(bc));

        if (ndim >= 3) {
            annotate_matmul_tile(ec, target);
        } else {
            annotate_2d_tile(ec, target);
        }

        annotated.push_back(std::move(ec));
    }

    if (annotated.empty()) return {};

    // Step 3: determine whether any tile is compute-bound.
    bool any_compute_bound = false;
    for (const auto& cfg : annotated) {
        if (cfg.compute_bound) {
            any_compute_bound = true;
            break;
        }
    }

    // Step 4: prune.
    //   - If at least one tile is compute-bound, drop all memory-bound tiles.
    //   - If NO tile can be compute-bound, keep all (we'll pick the best
    //     memory-bound one later).
    std::vector<ExtendedTileConfig> pruned;
    pruned.reserve(annotated.size());

    if (any_compute_bound) {
        for (auto& cfg : annotated) {
            if (cfg.compute_bound) {
                pruned.push_back(std::move(cfg));
            }
        }
    } else {
        // No compute-bound configuration is achievable – keep the top
        // candidates by operational intensity.
        std::sort(annotated.begin(), annotated.end(),
            [](const ExtendedTileConfig& a, const ExtendedTileConfig& b) {
                return a.operational_intensity > b.operational_intensity;
            });
        // Keep at most 200 to avoid quadratic blowup in later phases
        size_t keep = std::min(annotated.size(), size_t(200));
        for (size_t i = 0; i < keep; ++i) {
            pruned.push_back(std::move(annotated[i]));
        }
    }

    return pruned;
}

} // namespace symplex::optimizer
