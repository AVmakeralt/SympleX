// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include "symplex/costmodel/analytical.h"
#include <cstdint>

namespace symplex::costmodel {

/// ProfileResult: the result of profiling a kernel on real hardware
/// (or an analytical fallback with simulated variance).
struct ProfileResult {
    double mean_latency_ns;          // Mean measured latency
    double min_latency_ns;           // Minimum measured latency
    double max_latency_ns;           // Maximum measured latency
    double std_dev_ns;               // Standard deviation across iterations
    int64_t active_warps;            // Active warps observed during profiling
    int64_t sm_efficiency_percent;   // SM utilization from profiler
    bool   valid;                    // Whether the result is trustworthy
    bool   is_measured;              // true if measured on real hardware (CUDA),
                                     // false if derived from analytical fallback
    bool   high_confidence;          // true if is_measured or fallback noise is
                                     // within acceptable bounds (±1%)
};

/// EmpiricalCostModel: profiles kernels on real hardware when CUDA is
/// available, or falls back to the analytical model with added noise
/// to simulate empirical variance.
///
/// When CUDA is available (SYMPLEX_ENABLE_CUDA defined and a GPU is
/// present), this model:
///   1. Compiles a specialized PTX kernel for the given tile sizes
///   2. Launches it with warmup + profile iterations
///   3. Measures latency using cudaEvent_t timers
///
/// When CUDA is not available, it falls back to AnalyticalCostModel
/// and adds Gaussian-like noise to simulate measurement variance.
class EmpiricalCostModel {
public:
    explicit EmpiricalCostModel(const hardware::HardwareTarget& target);

    /// Profile a matrix multiplication kernel with the given tile config.
    /// Falls back to analytical estimation if CUDA is unavailable.
    ///
    /// @param M, N, K   Matrix dimensions
    /// @param tile       Tile configuration
    /// @param warmup_iters  Number of warmup iterations (default 10)
    /// @param profile_iters Number of profiled iterations (default 100)
    ProfileResult profile_matmul(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile,
        int64_t warmup_iters = 10,
        int64_t profile_iters = 100
    );

    /// Check whether a CUDA-capable GPU is available for profiling.
    bool is_cuda_available() const;

private:
    hardware::HardwareTarget target_;
    AnalyticalCostModel analytical_fallback_;
};

} // namespace symplex::costmodel
