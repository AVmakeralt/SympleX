// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/schedule_tree.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

namespace symplex::schedule {

/// TileConfig: describes a specific tiling of the iteration space.
struct TileConfig {
    std::vector<int64_t> outer_tiles;   // Size of each outer tile dimension
    std::vector<int64_t> inner_tiles;   // Size of each inner tile dimension

    TileConfig() = default;

    TileConfig(std::vector<int64_t> outer, std::vector<int64_t> inner)
        : outer_tiles(std::move(outer)), inner_tiles(std::move(inner)) {}

    /// Total number of elements in the outer tile.
    [[nodiscard]] int64_t outer_volume() const {
        int64_t vol = 1;
        for (auto t : outer_tiles) vol *= t;
        return vol;
    }

    /// Total number of elements in the inner tile.
    [[nodiscard]] int64_t inner_volume() const {
        int64_t vol = 1;
        for (auto t : inner_tiles) vol *= t;
        return vol;
    }

    /// Number of tiled dimensions.
    [[nodiscard]] size_t ndim() const { return outer_tiles.size(); }

    /// SRAM footprint in bytes for this tile (assuming FP16 and double-buffering).
    [[nodiscard]] size_t sram_footprint(
        size_t n_tensors = 3,           // Typical: input A, input B, output C
        size_t bytes_per_element = 2,    // FP16
        bool double_buffer = true
    ) const {
        size_t bytes = n_tensors * static_cast<size_t>(inner_volume())
                       * bytes_per_element;
        if (double_buffer) bytes *= 2;
        return bytes;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tile{outer=[";
        for (size_t i = 0; i < outer_tiles.size(); ++i) {
            if (i > 0) oss << "x";
            oss << outer_tiles[i];
        }
        oss << "], inner=[";
        for (size_t i = 0; i < inner_tiles.size(); ++i) {
            if (i > 0) oss << "x";
            oss << inner_tiles[i];
        }
        oss << "]}";
        return oss.str();
    }
};

/// TilingStrategy: different ways to tile the iteration space.
enum class TilingStrategy {
    RECTANGULAR,       // Simple rectangular tiling
    DIAMOND,           // Diamond tiling for stencil computations
    HIERARCHICAL,      // Multi-level tiling (GPU grid > SM > warp > tensor core)
    OVERLAPPED,        // Overlapped tiling for pipeline parallelism
};

/// TilingResult: outcome of applying a tiling strategy.
struct TilingResult {
    ScheduleTreePtr tree;              // The modified schedule tree
    TileConfig config;                 // The applied tile configuration
    std::vector<int64_t> grid_dims;    // Resulting GPU grid dimensions
    std::vector<int64_t> block_dims;   // Resulting GPU block dimensions
};

/// apply_tiling: apply a rectangular tiling to a schedule tree's band node.
/// Splits each loop in the band into an outer tile loop and an inner element loop.
inline TilingResult apply_rectangular_tiling(
    ScheduleTreePtr band_node,
    const TileConfig& config,
    const hardware::HardwareTarget& target
) {
    TilingResult result;
    result.config = config;

    // Tile the band node
    auto inner_band = band_node->tile_band(config.inner_tiles);

    // Compute GPU grid/block dimensions
    size_t n = config.outer_tiles.size();
    result.grid_dims.resize(n);
    result.block_dims.resize(n);

    for (size_t d = 0; d < n; ++d) {
        result.block_dims[d] = config.inner_tiles[d];
        // Grid size = ceil(total_size / tile_size)
        result.grid_dims[d] = (config.outer_tiles[d] + config.inner_tiles[d] - 1)
                              / config.inner_tiles[d];
    }

    result.tree = band_node;
    return result;
}

/// apply_hierarchical_tiling: multi-level tiling that maps directly
/// to the GPU memory hierarchy:
///   Level 1: Grid-level tiling  -> maps to Streaming Multiprocessors
///   Level 2: Block-level tiling -> maps to Shared Memory / SRAM
///   Level 3: Warp-level tiling  -> maps to Register File / Tensor Cores
struct HierarchicalTileConfig {
    TileConfig grid_level;    // Outermost: distributed across SMs
    TileConfig sm_level;      // Fits in SRAM
    TileConfig warp_level;    // Fits in registers, maps to Tensor Cores
};

inline TilingResult apply_hierarchical_tiling(
    ScheduleTreePtr band_node,
    const HierarchicalTileConfig& hconfig,
    const hardware::HardwareTarget& target
) {
    TilingResult result;
    result.config = hconfig.warp_level;  // Innermost tile config

    // Apply three levels of tiling: grid -> SM -> warp
    auto sm_band = band_node->tile_band(hconfig.sm_level.inner_tiles);
    auto warp_band = sm_band->tile_band(hconfig.warp_level.inner_tiles);

    result.grid_dims = hconfig.grid_level.outer_tiles;
    result.block_dims = hconfig.sm_level.inner_tiles;
    result.tree = band_node;
    return result;
}

/// generate_hardware_aligned_tiles: enumerate all tile configurations
/// that align with the hardware's Tensor Core dimensions and SRAM capacity.
/// This implements Phase 1+2 of the superoptimizer search.
inline std::vector<TileConfig> generate_hardware_aligned_tiles(
    size_t ndim,
    const hardware::HardwareTarget& target,
    int64_t max_tensor_dim = 1024
) {
    std::vector<TileConfig> configs;

    const auto& tc = target.gpu.tensor_core;

    // Only step through multiples of the Tensor Core native dimension
    for (int64_t tm = tc.m; tm <= max_tensor_dim; tm += tc.m) {
        for (int64_t tn = tc.n; tn <= max_tensor_dim; tn += tc.n) {
            for (int64_t tk = tc.k; tk <= max_tensor_dim; tk += tc.k) {
                if (ndim == 3) {
                    // Matrix multiply: (M, N, K) tiling
                    TileConfig cfg(
                        {tm, tn, tk},  // outer
                        {tm, tn, tk}   // inner (will be further split by hierarchical tiling)
                    );

                    // Analytic pruning: check SRAM capacity
                    size_t footprint = cfg.sram_footprint(
                        3, target.bytes_per_element, true
                    );
                    if (footprint >
                        static_cast<size_t>(std::max<int64_t>(0, target.max_sram_bytes))) continue;

                    configs.push_back(std::move(cfg));
                } else if (ndim == 2) {
                    // 2D tiling
                    TileConfig cfg({tm, tn}, {tm, tn});
                    size_t footprint = cfg.sram_footprint(
                        3, target.bytes_per_element, true
                    );
                    if (footprint >
                        static_cast<size_t>(std::max<int64_t>(0, target.max_sram_bytes))) continue;
                    configs.push_back(std::move(cfg));
                }
            }
        }
    }

    return configs;
}

} // namespace symplex::schedule
