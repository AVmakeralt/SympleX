// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include <cstdint>

namespace symplex::costmodel {

/// RooflinePoint: the result of a roofline analysis for a single kernel
/// or tile configuration.
struct RooflinePoint {
    double operational_intensity;  // FLOPS/byte
    double achievable_gflops;      // What GFLOPS this kernel can achieve
    bool   is_compute_bound;       // True if OI >= knee point
    double compute_time_ns;        // Time spent on compute alone
    double memory_time_ns;         // Time spent on memory transfers alone
    double total_time_ns;          // Dominant time (max of compute & memory)
    double efficiency_percent;     // % of peak FLOPS achieved
};

/// RooflineModel: models a kernel's performance as the minimum of
/// compute throughput and memory throughput, following the classical
/// roofline model (Williams, Waterman, Patterson, 2009).
///
/// The achievable performance is:
///   Achievable = min(Peak_FLOPS, Peak_BW * Operational_Intensity)
/// where Operational_Intensity = FLOPS / Bytes_moved.
///
/// The kernel is compute-bound when its OI exceeds the "knee" of the
/// roofline (Peak_FLOPS / Peak_BW), and memory-bound otherwise.
class RooflineModel {
public:
    explicit RooflineModel(const hardware::HardwareTarget& target);

    /// Analyze an arbitrary kernel given its total compute operations
    /// and total bytes moved between HBM and SRAM.
    RooflinePoint analyze(int64_t compute_ops, int64_t bytes_moved) const;

    /// Analyze a specific tile configuration for a matrix multiplication
    /// C[M,N] += A[M,K] * B[K,N] tiled with inner dimensions (tm, tn, tk).
    ///
    /// Compute ops per tile = 2 * tm * tn * tk  (multiply-add = 2 ops)
    /// Bytes moved per tile = tm*tk + tk*tn + tm*tn  (A + B + C, per element)
    RooflinePoint analyze_matmul_tile(int64_t tm, int64_t tn, int64_t tk) const;

    /// Return the operational intensity at the roofline knee point.
    /// Kernels with OI >= this value are compute-bound.
    double knee_intensity() const;

    /// Peak FP16 FLOPS in GFLOPS.
    double peak_gflops() const;

    /// Peak HBM bandwidth in GB/s.
    double peak_bandwidth_gbps() const;

private:
    hardware::HardwareTarget target_;
};

} // namespace symplex::costmodel
