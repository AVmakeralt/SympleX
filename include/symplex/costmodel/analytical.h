// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include <cstdint>

namespace symplex::costmodel {

/// AnalyticalEstimate: a first-principles latency estimate for a tiled
/// kernel.  Every timing component is broken down so the optimizer can
/// understand the bottleneck.
struct AnalyticalEstimate {
    double latency_ns;           // Total estimated kernel latency
    double compute_ns;           // Time in Tensor Core / CUDA cores
    double sram_load_ns;         // Time to load operands into SRAM
    double sram_store_ns;        // Time to store results from SRAM
    double hbm_load_ns;          // Time to DMA from HBM to SRAM
    double hbm_store_ns;         // Time to DMA from SRAM to HBM
    double overlap_factor;       // Fraction of memory time hidden by pipelining
    int64_t sram_bytes_used;     // SRAM footprint for the tile (bytes)
    int64_t register_bytes_used; // Register file footprint (bytes)
    int64_t occupancy_warps;     // Active warps per SM
};

/// AnalyticalCostModel: estimates kernel latency from first principles,
/// modelling each stage of the GPU pipeline (HBM → SRAM → Registers →
/// Compute → SRAM → HBM) and accounting for software pipelining
/// overlap and occupancy effects.
class AnalyticalCostModel {
public:
    explicit AnalyticalCostModel(const hardware::HardwareTarget& target);

    /// Estimate latency for a tiled matrix multiplication
    ///   C[M,N] += A[M,K] * B[K,N]
    /// with the given TileConfig (inner_tiles = {tm, tn, tk}).
    AnalyticalEstimate estimate_matmul(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile
    ) const;

    /// Estimate latency for a tiled 2D convolution
    ///   output[batch, oc, oh, ow] += input[batch, ic, ih, iw] * kernel[oc, ic, kh, kw]
    AnalyticalEstimate estimate_conv2d(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        const schedule::TileConfig& tile
    ) const;

private:
    hardware::HardwareTarget target_;
};

} // namespace symplex::costmodel
