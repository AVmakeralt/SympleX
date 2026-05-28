// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/costmodel/roofline.h"
#include <algorithm>
#include <cmath>

namespace symplex::costmodel {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RooflineModel::RooflineModel(const hardware::HardwareTarget& target)
    : target_(target)
{}

// ---------------------------------------------------------------------------
// Core roofline analysis
// ---------------------------------------------------------------------------

RooflinePoint RooflineModel::analyze(int64_t compute_ops, int64_t bytes_moved) const {
    RooflinePoint rp{};

    // Guard against degenerate inputs
    if (compute_ops <= 0) {
        rp.operational_intensity = 0.0;
        rp.achievable_gflops     = 0.0;
        rp.is_compute_bound      = false;
        rp.compute_time_ns       = 0.0;
        rp.memory_time_ns        = 0.0;
        rp.total_time_ns         = 0.0;
        rp.efficiency_percent    = 0.0;
        return rp;
    }

    // Operational intensity (FLOPS per byte)
    double oi = (bytes_moved > 0)
                ? static_cast<double>(compute_ops) / static_cast<double>(bytes_moved)
                : 1e30;  // Effectively infinite OI if no bytes moved

    rp.operational_intensity = oi;

    // Peak throughput values
    const double peak_flops  = target_.peak_flops_fp16();           // FLOPS/s
    const double peak_bw_bps = target_.gpu.memory.global_bw_gbps    // bytes/s
                             * 1e9;

    // Compute time: time if compute were the only bottleneck
    // compute_ops / peak_flops gives seconds; multiply by 1e9 for ns.
    double compute_time_s = static_cast<double>(compute_ops) / peak_flops;
    rp.compute_time_ns    = compute_time_s * 1e9;

    // Memory time: time if memory were the only bottleneck
    double memory_time_s = (bytes_moved > 0)
                           ? static_cast<double>(bytes_moved) / peak_bw_bps
                           : 0.0;
    rp.memory_time_ns    = memory_time_s * 1e9;

    // The roofline model: total time is dominated by the slower resource.
    // total_time = max(compute_time, memory_time)
    double total_time_s  = std::max(compute_time_s, memory_time_s);
    rp.total_time_ns     = total_time_s * 1e9;

    // Achievable GFLOPS = compute_ops / total_time
    rp.achievable_gflops = (total_time_s > 0.0)
                           ? (static_cast<double>(compute_ops) / total_time_s) * 1e-9
                           : 0.0;

    // Classification
    rp.is_compute_bound = (oi >= knee_intensity());

    // Efficiency: fraction of peak compute actually achieved
    double peak_gflops_val = peak_gflops();
    rp.efficiency_percent = (peak_gflops_val > 0.0)
                            ? (rp.achievable_gflops / peak_gflops_val) * 100.0
                            : 0.0;

    return rp;
}

// ---------------------------------------------------------------------------
// Matmul tile convenience
// ---------------------------------------------------------------------------

RooflinePoint RooflineModel::analyze_matmul_tile(
    int64_t tm, int64_t tn, int64_t tk
) const {
    // Per inner tile:
    //   FMA ops  = tm * tn * tk, each FMA = 2 FLOPS
    //   Bytes for A tile = tm * tk * bytes_per_element
    //   Bytes for B tile = tk * tn * bytes_per_element
    //   Bytes for C tile = tm * tn * bytes_per_element  (write-back)
    int64_t compute_ops = 2 * tm * tn * tk;

    int64_t bpe = target_.bytes_per_element;
    int64_t bytes_a = tm * tk * bpe;
    int64_t bytes_b = tk * tn * bpe;
    int64_t bytes_c = tm * tn * bpe;
    int64_t bytes_moved = bytes_a + bytes_b + bytes_c;

    return analyze(compute_ops, bytes_moved);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double RooflineModel::knee_intensity() const {
    // Knee OI = Peak_FLOPS / Peak_BW  (FLOPS/byte)
    double peak_flops  = target_.peak_flops_fp16();
    double peak_bw_bps = target_.gpu.memory.global_bw_gbps * 1e9;
    return (peak_bw_bps > 0.0) ? (peak_flops / peak_bw_bps) : 1e30;
}

double RooflineModel::peak_gflops() const {
    // peak_flops_fp16() returns FLOPS/s; convert to GFLOPS
    return target_.peak_flops_fp16() * 1e-9;
}

double RooflineModel::peak_bandwidth_gbps() const {
    return static_cast<double>(target_.gpu.memory.global_bw_gbps);
}

} // namespace symplex::costmodel
