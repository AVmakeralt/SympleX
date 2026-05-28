// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/search_phase2.h"
#include <algorithm>
#include <cmath>

namespace symplex::optimizer {

namespace {

/// Compute SRAM utilization sub-score [0, 1].
/// We want the tile to use a large fraction of SRAM (amortises latency)
/// but not exceed it.  The base TileConfig already guarantees that the
/// footprint fits, so we simply measure the ratio.
double sram_utilization_score(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    return cfg.sram_utilization(target.max_sram_bytes);
}

/// Compute Tensor Core utilization sub-score [0, 1].
/// Measures how many native MMA operations the inner tile contains
/// and whether any dimension has a "tail" (non-aligned remainder).
/// A perfect score means every dimension is a multiple of the native
/// MMA dimension AND the number of MMA ops is at least 1 per SM
/// tensor core.
double tensor_core_utilization_score(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    const auto& tc = target.gpu.tensor_core;

    // Alignment fraction (0..1): fraction of dimensions perfectly aligned
    double alignment = cfg.tensor_core_alignment(tc.m, tc.n, tc.k);

    // Throughput factor: how many MMA ops fit in the tile, relative to
    // a "full SM" (num_tensor_cores MMA ops per SM per cycle).
    if (cfg.inner_tiles.size() < 3) {
        // For 2-D tiles Tensor Core utilisation is inherently poor.
        return alignment * 0.5;
    }

    int64_t tm = cfg.inner_tiles[0];
    int64_t tn = cfg.inner_tiles[1];
    int64_t tk = cfg.inner_tiles[2];

    // Number of MMA ops per tile = (tm/m) * (tn/n) * (tk/k)
    double mma_ops = 1.0;
    if (tc.m > 0) mma_ops *= static_cast<double>(tm) / tc.m;
    if (tc.n > 0) mma_ops *= static_cast<double>(tn) / tc.n;
    if (tc.k > 0) mma_ops *= static_cast<double>(tk) / tc.k;

    // Normalize: full utilization = at least 1 MMA per TC in the SM
    double mma_per_tc = mma_ops / static_cast<double>(tc.ops_per_mma());
    double throughput = std::min(mma_per_tc, 1.0);

    // Composite: weight alignment more heavily
    return 0.6 * alignment + 0.4 * throughput;
}

/// Compute memory coalescing sub-score [0, 1].
/// GPU memory transactions are 128 bytes wide.  Each thread in a warp
/// should access a contiguous 128-byte segment.  For FP16, a segment
/// holds 128/2 = 64 elements.  The innermost tile dimension (typically N
/// for column-major B) should be a multiple of 64 for perfect coalescing.
double memory_coalescing_score(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    // Number of elements per 128-byte transaction
    int64_t elems_per_txn = target.memory_alignment / target.bytes_per_element;
    if (elems_per_txn <= 0) elems_per_txn = 64;  // safe default

    double score = 0.0;
    int64_t dims = static_cast<int64_t>(cfg.inner_tiles.size());

    // Check each dimension; weight innermost dimension more heavily.
    for (int64_t d = 0; d < dims; ++d) {
        bool aligned = (cfg.inner_tiles[d] % elems_per_txn == 0);
        double dim_weight = (d == dims - 1) ? 0.4 : (0.6 / (dims - 1));
        score += aligned ? dim_weight : 0.0;
    }

    return score;
}

/// Compute the composite score for a candidate.
/// Weights:
///   SRAM utilization      : 0.35
///   Tensor Core utilization: 0.45
///   Memory coalescing      : 0.20
double composite_score(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    double sram   = sram_utilization_score(cfg, target);
    double tcu    = tensor_core_utilization_score(cfg, target);
    double coal   = memory_coalescing_score(cfg, target);
    return 0.35 * sram + 0.45 * tcu + 0.20 * coal;
}

} // anonymous namespace

// ── Phase 2 implementation ───────────────────────────────────────────────

std::vector<ExtendedTileConfig> phase2_symmetry_alignment(
    std::vector<ExtendedTileConfig> candidates,
    const hardware::HardwareTarget& target
) {
    const auto& tc = target.gpu.tensor_core;

    // Step 1: filter – drop any candidate whose inner tile dimensions
    //         are NOT aligned with the Tensor Core native dimensions.
    //         For each dimension, inner_tile must be a multiple of the
    //         corresponding TC native dimension.
    std::vector<ExtendedTileConfig> aligned;
    aligned.reserve(candidates.size());

    for (auto& cfg : candidates) {
        bool pass = true;

        if (cfg.inner_tiles.size() >= 3) {
            // 3-D tile: M, N, K must each align
            if (tc.m > 0 && cfg.inner_tiles[0] % tc.m != 0) pass = false;
            if (tc.n > 0 && cfg.inner_tiles[1] % tc.n != 0) pass = false;
            if (tc.k > 0 && cfg.inner_tiles[2] % tc.k != 0) pass = false;
        } else if (cfg.inner_tiles.size() >= 2) {
            // 2-D tile: at least the first two dimensions should align
            if (tc.m > 0 && cfg.inner_tiles[0] % tc.m != 0) pass = false;
            if (tc.n > 0 && cfg.inner_tiles[1] % tc.n != 0) pass = false;
        }

        if (pass) {
            aligned.push_back(std::move(cfg));
        }
    }

    // If alignment filtering eliminates everything, fall back to the
    // original candidates (they already survived the SRAM/roofline
    // filter in phase 1).
    if (aligned.empty()) {
        aligned = std::move(candidates);
    }

    // Step 2: score each surviving candidate.
    for (auto& cfg : aligned) {
        cfg.score = composite_score(cfg, target);
    }

    // Step 3: sort by composite score descending.
    std::sort(aligned.begin(), aligned.end(),
        [](const ExtendedTileConfig& a, const ExtendedTileConfig& b) {
            return a.score > b.score;
        });

    // Step 4: cap the output to a manageable size for phase 3.
    //         Keep the top 200.
    if (aligned.size() > 200) {
        aligned.resize(200);
    }

    return aligned;
}

} // namespace symplex::optimizer
