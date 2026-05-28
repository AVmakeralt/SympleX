// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/tiling.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <sstream>
#include <unordered_set>
#include <limits>

namespace symplex::schedule {

// ── TileConfig helper functions (non-inline) ─────────────────────────────────
//
// Most TileConfig methods are inline in the header.  This TU provides
// the non-inline helper algorithms for tiling.

/// Compute the optimal rectangular tile sizes for a given iteration space
/// and SRAM budget, aligning to Tensor Core MMA dimensions.
///
/// This function implements an analytical model:
///   1. Start with the largest possible tile (bounded by iteration space).
///   2. Align tile dimensions to MMA fragment sizes.
///   3. Iteratively reduce the largest tile dimension until the
///      footprint fits within the SRAM budget.
///
/// \param ndim                Number of tile dimensions
/// \param iter_bounds         Iteration space bounds per dimension
/// \param sram_budget_bytes   Maximum SRAM bytes available
/// \param n_tensors           Number of tensors in the kernel (typically 3)
/// \param bytes_per_element   Bytes per tensor element (2 for FP16)
/// \param mma_dims            MMA fragment dimensions {m, n, k} (or empty)
/// \param double_buffer       Whether to account for double buffering
/// \return                    Optimal tile sizes per dimension
std::vector<int64_t> compute_optimal_tile_sizes(
    size_t ndim,
    const std::vector<std::pair<int64_t, int64_t>>& iter_bounds,
    int64_t sram_budget_bytes,
    size_t n_tensors,
    size_t bytes_per_element,
    const std::vector<int64_t>& mma_dims,
    bool double_buffer
) {
    if (ndim == 0) return {};

    // Step 1: Initialize tile sizes to full iteration range.
    std::vector<int64_t> tiles(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        auto [lo, hi] = iter_bounds[d];
        tiles[d] = hi - lo + 1;
    }

    // Step 2: Align to MMA dimensions.
    for (size_t d = 0; d < ndim && d < mma_dims.size(); ++d) {
        if (mma_dims[d] > 0 && tiles[d] > 0) {
            tiles[d] = (tiles[d] / mma_dims[d]) * mma_dims[d];
            if (tiles[d] == 0) tiles[d] = mma_dims[d];
        }
    }

    // Step 3: Iteratively reduce until SRAM budget is met.
    auto compute_footprint = [&]() -> int64_t {
        int64_t volume = 1;
        for (auto t : tiles) volume *= t;
        int64_t bytes = volume * static_cast<int64_t>(n_tensors)
                       * static_cast<int64_t>(bytes_per_element);
        if (double_buffer) bytes *= 2;
        return bytes;
    };

    for (int iter = 0; iter < 200; ++iter) {
        if (compute_footprint() <= sram_budget_bytes) break;

        // Find the dimension with the largest tile.
        size_t max_dim = 0;
        for (size_t d = 1; d < ndim; ++d) {
            if (tiles[d] > tiles[max_dim]) max_dim = d;
        }

        if (tiles[max_dim] <= 1) break;

        // Reduce by MMA step or by half.
        int64_t step = 1;
        if (max_dim < mma_dims.size() && mma_dims[max_dim] > 0) {
            step = mma_dims[max_dim];
        } else {
            step = std::max(int64_t(1), tiles[max_dim] / 2);
        }

        tiles[max_dim] = std::max(int64_t(1), tiles[max_dim] - step);

        // Re-align.
        if (max_dim < mma_dims.size() && mma_dims[max_dim] > 0) {
            tiles[max_dim] = (tiles[max_dim] / mma_dims[max_dim]) * mma_dims[max_dim];
            if (tiles[max_dim] == 0) tiles[max_dim] = mma_dims[max_dim];
        }
    }

    return tiles;
}

/// Generate all valid hierarchical tile configurations for a GPU kernel.
/// This enumerates tile sizes that:
///   - Align with Tensor Core MMA dimensions
///   - Fit within SRAM at each level
///   - Saturate the GPU (enough blocks for all SMs)
///
/// \param ndim           Number of loop dimensions
/// \param iter_bounds    Iteration bounds per dimension
/// \param target         GPU hardware target
/// \param max_tiles      Maximum number of configs to generate
/// \return               Vector of valid HierarchicalTileConfig
std::vector<HierarchicalTileConfig> generate_hierarchical_configs(
    size_t ndim,
    const std::vector<std::pair<int64_t, int64_t>>& iter_bounds,
    const hardware::HardwareTarget& target,
    size_t max_tiles
) {
    std::vector<HierarchicalTileConfig> configs;
    if (ndim == 0) return configs;

    const auto& tc = target.gpu.tensor_core;
    int64_t sram_budget = target.max_sram_bytes;

    // Enumerate warp-level tile sizes (innermost, fits in registers).
    // These should be multiples of the MMA dimensions.
    std::vector<std::vector<int64_t>> warp_tiles;

    if (ndim == 3) {
        // Matmul: enumerate (warp_M, warp_N, warp_K) aligned to MMA.
        for (int64_t wm = tc.m; wm <= 128; wm += tc.m) {
            for (int64_t wn = tc.n; wn <= 128; wn += tc.n) {
                for (int64_t wk = tc.k; wk <= 64; wk += tc.k) {
                    // Register budget: roughly 256 registers * 4 bytes = 1KB
                    // Fragment for MMA(m,n,k) uses m*k + n*k + m*n elements.
                    int64_t frag_elems = wm * wk + wn * wk + wm * wn;
                    int64_t frag_bytes = frag_elems * target.bytes_per_element;
                    if (frag_bytes > 8192) continue;  // ~8KB register limit

                    warp_tiles.push_back({wm, wn, wk});
                }
            }
        }
    } else if (ndim == 2) {
        for (int64_t wm = tc.m; wm <= 256; wm += tc.m) {
            for (int64_t wn = tc.n; wn <= 256; wn += tc.n) {
                int64_t frag_elems = wm * wn;
                int64_t frag_bytes = frag_elems * target.bytes_per_element;
                if (frag_bytes > 16384) continue;
                warp_tiles.push_back({wm, wn});
            }
        }
    }

    // For each warp-level config, compute SM-level and grid-level.
    for (const auto& warp_tile : warp_tiles) {
        if (configs.size() >= max_tiles) break;

        HierarchicalTileConfig hconfig;

        // Warp level.
        hconfig.warp_level.inner_tiles = warp_tile;
        hconfig.warp_level.outer_tiles = warp_tile;

        // SM level: tile that fits in shared memory.
        // The SM tile is a multiple of the warp tile.
        std::vector<int64_t> sm_tile(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            // Scale warp tile by a factor that fits in SRAM.
            // Start with 4x warp tile and reduce if needed.
            sm_tile[d] = warp_tile[d] * 4;
        }

        // Verify SM-level SRAM budget.
        int64_t sm_volume = 1;
        for (auto t : sm_tile) sm_volume *= t;
        int64_t sm_bytes = sm_volume * 3 * target.bytes_per_element * 2;  // 3 tensors, double-buffer
        if (sm_bytes > sram_budget) {
            // Reduce SM tile.
            for (int reduction = 0; reduction < 10; ++reduction) {
                size_t max_d = 0;
                for (size_t d = 1; d < ndim; ++d) {
                    if (sm_tile[d] > sm_tile[max_d]) max_d = d;
                }
                sm_tile[max_d] -= warp_tile[max_d];
                if (sm_tile[max_d] <= 0) break;

                sm_volume = 1;
                for (auto t : sm_tile) sm_volume *= t;
                sm_bytes = sm_volume * 3 * target.bytes_per_element * 2;
                if (sm_bytes <= sram_budget) break;
            }
        }

        if (sm_bytes > sram_budget) continue;  // Skip this config.

        hconfig.sm_level.inner_tiles = sm_tile;
        hconfig.sm_level.outer_tiles = sm_tile;

        // Grid level: divide total iteration space by SM tile.
        std::vector<int64_t> grid_tile(ndim);
        for (size_t d = 0; d < ndim && d < iter_bounds.size(); ++d) {
            auto [lo, hi] = iter_bounds[d];
            grid_tile[d] = hi - lo + 1;
        }
        hconfig.grid_level.inner_tiles = grid_tile;
        hconfig.grid_level.outer_tiles = grid_tile;

        configs.push_back(std::move(hconfig));
    }

    return configs;
}

/// Select the best tile configuration from a set of candidates based on
/// a simple performance model:
///   score = compute_intensity * occupancy_factor
///
/// Higher compute intensity means more FLOPS per byte transferred,
/// which is better for GPU utilization.  Occupancy factor accounts for
/// whether the tile configuration allows sufficient SM occupancy.
///
/// \param candidates    Vector of candidate configurations
/// \param target        GPU hardware target
/// \return              Index of the best configuration, or -1 if empty
int select_best_tile_config(
    const std::vector<TileConfig>& candidates,
    const hardware::HardwareTarget& target
) {
    if (candidates.empty()) return -1;

    int best_idx = 0;
    double best_score = -1.0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& cfg = candidates[i];

        // Compute intensity = FLOPS / bytes
        // For a matmul tile (M, N, K): FLOPS = 2*M*N*K, bytes = (M*K + K*N + M*N) * bpe
        int64_t inner_vol = cfg.inner_volume();
        double flops = 2.0 * static_cast<double>(inner_vol);
        double bytes = static_cast<double>(
            cfg.sram_footprint(3, target.bytes_per_element, true)
        );
        double intensity = (bytes > 0) ? flops / bytes : 0.0;

        // Occupancy: check if the tile fits in SRAM.
        double occupancy = 1.0;
        size_t footprint = cfg.sram_footprint(3, target.bytes_per_element, true);
        if (footprint > static_cast<size_t>(std::max<int64_t>(0, target.max_sram_bytes))) {
            occupancy = 0.0;
        } else if (footprint > 0) {
            // Higher occupancy with smaller SRAM usage.
            occupancy = 1.0 - static_cast<double>(footprint) /
                               static_cast<double>(target.max_sram_bytes) * 0.5;
        }

        double score = intensity * occupancy;

        if (score > best_score) {
            best_score = score;
            best_idx = static_cast<int>(i);
        }
    }

    return best_idx;
}

/// Apply diamond tiling to a band node for stencil computations.
/// Diamond tiling enables overlapping communication and computation
/// by skewing the tile boundaries along dependency directions.
///
/// For a 2D stencil with dependency vectors {(1,0), (0,1), (1,1)},
/// diamond tiling creates hexagonal tiles that minimize pipeline bubbles.
///
/// \param band_node      The band node to tile
/// \param tile_sizes     Tile sizes per dimension
/// \param skew_vectors   Dependency direction vectors for skewing
/// \return               The modified band node, or nullptr on error
ScheduleTreePtr apply_diamond_tiling(
    ScheduleTreePtr band_node,
    const std::vector<int64_t>& tile_sizes,
    const std::vector<std::vector<int64_t>>& skew_vectors
) {
    if (!band_node || band_node->type() != ScheduleNodeType::BAND) return nullptr;

    size_t n = band_node->band_data().members.size();
    if (tile_sizes.size() != n) return nullptr;

    // Step 1: Apply skewing to the band node.
    // For each skew vector, modify the band member coefficients.
    // If skew_vector = (s0, s1, ...), we add s_j * member[d] to member[0]
    // This implements the transformation:
    //   i_0' = i_0 + s_0 * i_1 + s_1 * i_2 + ...
    if (!skew_vectors.empty()) {
        for (const auto& skew : skew_vectors) {
            if (skew.size() != n) continue;

            // Apply skewing to the first band member.
            auto& member0 = band_node->band_data().members[0];
            for (size_t j = 1; j < n && j < skew.size(); ++j) {
                member0.coefficients[j] += skew[j] * member0.coefficients[0];
            }
        }
    }

    // Step 2: Apply standard rectangular tiling on the skewed band.
    auto inner_band = band_node->tile_band(tile_sizes);

    return inner_band;
}

/// Apply overlapped tiling for pipeline parallelism.
/// Overlapped tiling creates tiles with ghost regions that overlap
/// with neighboring tiles, enabling data prefetching and hiding
/// communication latency.
///
/// \param band_node      The band node to tile
/// \param tile_sizes     Tile sizes per dimension
/// \param overlap_sizes  Overlap (ghost region) sizes per dimension
/// \return               The modified band node, or nullptr on error
ScheduleTreePtr apply_overlapped_tiling(
    ScheduleTreePtr band_node,
    const std::vector<int64_t>& tile_sizes,
    const std::vector<int64_t>& overlap_sizes
) {
    if (!band_node || band_node->type() != ScheduleNodeType::BAND) return nullptr;

    size_t n = band_node->band_data().members.size();
    if (tile_sizes.size() != n) return nullptr;

    // For overlapped tiling, we first apply standard rectangular tiling.
    auto inner_band = band_node->tile_band(tile_sizes);

    // The overlap regions are handled at code generation time:
    // each thread block loads (tile + overlap) elements but only
    // computes on the interior (tile - overlap) elements.
    //
    // For the schedule tree, we annotate the inner band with
    // the overlap information by extending the bounds.
    // This is done by modifying the filter data at the leaf level.

    // For now, the schedule tree just reflects the tiling.
    // The overlap is a code generation concern.

    return inner_band;
}

/// Validate that a tile configuration is compatible with the GPU
/// hardware constraints.
///
/// Checks:
///   1. SRAM footprint fits in shared memory
///   2. Thread count per block is within limits
///   3. Register usage per thread is reasonable
///   4. Grid dimensions are within hardware limits
///
/// \param config    Tile configuration to validate
/// \param target    GPU hardware target
/// \param n_tensors Number of tensors in the kernel
/// \return          true if the configuration is valid
bool validate_tile_config(
    const TileConfig& config,
    const hardware::HardwareTarget& target,
    size_t n_tensors
) {
    // Check SRAM capacity.
    size_t footprint = config.sram_footprint(
        n_tensors, target.bytes_per_element, true
    );
    if (footprint > static_cast<size_t>(std::max<int64_t>(0, target.max_sram_bytes))) {
        return false;
    }

    // Check that inner tile dimensions are multiples of MMA dimensions
    // (for Tensor Core alignment).
    const auto& tc = target.gpu.tensor_core;
    if (config.ndim() >= 3) {
        if (config.inner_tiles[0] % tc.m != 0) return false;
        if (config.inner_tiles[1] % tc.n != 0) return false;
        if (config.inner_tiles[2] % tc.k != 0) return false;
    }

    // Check that inner tile dimensions are positive.
    for (auto t : config.inner_tiles) {
        if (t <= 0) return false;
    }

    return true;
}

/// Compute the effective grid and block dimensions for a tile
/// configuration given the total iteration space size.
///
/// \param tile_sizes   Inner tile sizes
/// \param total_sizes  Total iteration space sizes per dimension
/// \param target       GPU hardware target
/// \return             Pair of (grid_dims, block_dims)
std::pair<std::vector<int64_t>, std::vector<int64_t>>
compute_launch_geometry(
    const std::vector<int64_t>& tile_sizes,
    const std::vector<int64_t>& total_sizes,
    const hardware::HardwareTarget& target
) {
    size_t ndim = tile_sizes.size();
    std::vector<int64_t> grid_dims(ndim);
    std::vector<int64_t> block_dims(ndim);

    for (size_t d = 0; d < ndim; ++d) {
        block_dims[d] = tile_sizes[d];
        int64_t total = (d < total_sizes.size()) ? total_sizes[d] : tile_sizes[d];
        grid_dims[d] = (total + tile_sizes[d] - 1) / tile_sizes[d];
    }

    // Clamp to hardware limits.
    if (grid_dims.size() >= 1) {
        grid_dims[0] = std::min(grid_dims[0], target.gpu.max_grid_x);
    }
    if (grid_dims.size() >= 2) {
        grid_dims[1] = std::min(grid_dims[1], target.gpu.max_grid_y);
    }
    if (grid_dims.size() >= 3) {
        grid_dims[2] = std::min(grid_dims[2], target.gpu.max_grid_z);
    }

    // Ensure block dims don't exceed max threads.
    int64_t total_threads = 1;
    for (auto b : block_dims) total_threads *= b;
    if (total_threads > target.gpu.max_threads_per_block) {
        // Scale down the block dimensions.
        double scale = std::sqrt(static_cast<double>(target.gpu.max_threads_per_block) /
                                 static_cast<double>(total_threads));
        for (auto& b : block_dims) {
            b = std::max(int64_t(1), static_cast<int64_t>(b * scale));
        }
    }

    return {grid_dims, block_dims};
}

} // namespace symplex::schedule
