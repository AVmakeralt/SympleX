// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/tile_config.h"
#include <cmath>
#include <algorithm>

namespace symplex::optimizer {

// ── Free-standing utility functions for ExtendedTileConfig ────────────────

/// Compute the matmul FMA operations for a 3-D tile (M, N, K).
/// Each FMA = 1 multiply + 1 add = 2 ops, so total = 2*M*N*K.
int64_t compute_matmul_ops(int64_t tm, int64_t tn, int64_t tk) {
    return 2 * tm * tn * tk;
}

/// Compute the total bytes transferred for a matmul tile (M, N, K).
/// Three tensors participate in one tile:
///   A : M × K elements
///   B : K × N elements
///   C : M × N elements
/// Total bytes = (M*K + K*N + M*N) × bytes_per_element.
int64_t compute_matmul_bytes(
    int64_t tm, int64_t tn, int64_t tk,
    int64_t bytes_per_element
) {
    int64_t elements = tm * tk   // tensor A
                     + tk * tn   // tensor B
                     + tm * tn;  // tensor C (partial sum output)
    return elements * bytes_per_element;
}

/// Compute the operational intensity (FLOPS / byte) for a matmul tile.
double compute_operational_intensity(
    int64_t tm, int64_t tn, int64_t tk,
    int64_t bytes_per_element
) {
    int64_t ops   = compute_matmul_ops(tm, tn, tk);
    int64_t bytes = compute_matmul_bytes(tm, tn, tk, bytes_per_element);
    if (bytes <= 0) return 0.0;
    return static_cast<double>(ops) / static_cast<double>(bytes);
}

/// Compute a simple analytical occupancy estimate for a tile configuration.
/// Returns the number of active warps per SM.
int64_t estimate_occupancy(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    // Each inner-tile element roughly corresponds to one thread.
    int64_t total_threads = cfg.inner_volume();
    if (total_threads <= 0) return 0;

    // Shared memory consumed per thread block
    int64_t smem_per_block = static_cast<int64_t>(
        cfg.sram_footprint(3, target.bytes_per_element, true));

    // Rough register estimate: each thread needs registers for its
    // accumulator fragment and input prefetch.  A conservative estimate
    // is ~40 registers for a Tensor-Core matmul kernel.
    int64_t regs_per_thread = 40;

    int64_t occ = target.gpu.sm.compute_occupancy(regs_per_thread, smem_per_block);
    return occ;
}

/// Compute estimated latency (in nanoseconds) using the roofline model:
///   time = max(compute_time, memory_time)
/// where
///   compute_time = compute_ops / peak_flops
///   memory_time  = bytes_moved / peak_bandwidth_bytes_per_ns
double estimate_latency_ns(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    double peak_flops = target.peak_flops_fp16();              // FLOP/s
    double peak_bw    = target.gpu.memory.global_bw_gbps * 1e9; // bytes/s

    double compute_time_s = (peak_flops > 0.0)
        ? static_cast<double>(cfg.compute_ops) / peak_flops
        : 1e30;
    double memory_time_s  = (peak_bw > 0.0)
        ? static_cast<double>(cfg.bytes_moved) / peak_bw
        : 1e30;

    double time_s = std::max(compute_time_s, memory_time_s);
    return time_s * 1e9;  // convert to nanoseconds
}

} // namespace symplex::optimizer
