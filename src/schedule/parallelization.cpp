// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/parallelization.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace symplex::schedule {

// ---------------------------------------------------------------------------
// GPUMapping::to_string
// ---------------------------------------------------------------------------
std::string GPUMapping::to_string() const {
    std::ostringstream oss;
    oss << "GPUMapping{grid=[";
    for (size_t i = 0; i < grid_dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << grid_dims[i];
    }
    oss << "], block=[";
    for (size_t i = 0; i < block_dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << block_dims[i];
    }
    oss << "], warps=" << warps_per_block();
    oss << ", smem=" << smem_per_block << "B";
    if (!warp_roles.empty()) {
        oss << ", roles=[";
        for (size_t i = 0; i < warp_roles.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << warp_role_to_string(warp_roles[i]);
        }
        oss << "]";
    }
    oss << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// MarkParallelDims
//
// For each band node in the tree, check each member dimension against
// all dependency vectors.  If no dependency is carried at that
// dimension, mark it as parallel.
// ---------------------------------------------------------------------------
int MarkParallelDims(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace
) {
    int marked = 0;
    auto bands = tree->band_nodes();

    auto all_deps = iterspace.all_dependencies();

    for (auto& band : bands) {
        auto& data = band->band_data();
        size_t n_members = data.members.size();

        for (size_t m = 0; m < n_members; ++m) {
            // Check if any dependency is carried at this dimension
            bool carries_dep = false;

            for (const auto& dep : all_deps) {
                for (const auto& dv : dep.vectors()) {
                    if (m < dv.components.size() && dv.components[m] != 0) {
                        carries_dep = true;
                        break;
                    }
                }
                if (carries_dep) break;
            }

            if (!carries_dep) {
                // This dimension carries no dependencies → can be parallel
                if (!data.members[m].parallel) {
                    band->mark_parallel(m);
                    ++marked;
                }
            } else {
                // This dimension carries dependencies → must be sequential
                if (!data.members[m].coincidence) {
                    band->mark_sequential(m);
                }
            }
        }
    }

    return marked;
}

// ---------------------------------------------------------------------------
// MapToGPU
//
// Strategy:
//   1. Find all band nodes and their parallel/sequential markings.
//   2. Outermost parallel dims → blockIdx
//   3. Innermost parallel dims → threadIdx (up to max_threads_per_block)
//   4. Sequential dims → remain as inner loops
//   5. Estimate shared memory based on inner tile footprints
// ---------------------------------------------------------------------------
GPUMapping MapToGPU(
    const ScheduleTreePtr& tree,
    const hardware::HardwareTarget& target
) {
    GPUMapping mapping;

    // Compute grid and block dimensions from the tree structure
    auto [grid_dims, block_dims] = ComputeGridBlockDims(tree, target);
    mapping.grid_dims = std::move(grid_dims);
    mapping.block_dims = std::move(block_dims);

    // Estimate shared memory per block based on the innermost band's
    // tile sizes.  Walk the tree to find the innermost band.
    auto bands = tree->band_nodes();
    if (!bands.empty()) {
        // The innermost band (last in DFS order) typically corresponds
        // to the innermost tiled loop whose data lives in SRAM
        auto& inner_band = bands.back();
        auto& data = inner_band->band_data();

        // Estimate SRAM: each member contributes its coefficient
        // magnitude * bytes_per_element, with double-buffering
        int64_t smem = 0;
        for (const auto& m : data.members) {
            int64_t extent = 0;
            for (auto c : m.coefficients) {
                extent = std::max(extent, std::abs(c));
            }
            extent = std::max(extent, int64_t(1));
            smem += extent * target.bytes_per_element;
        }
        smem *= target.pipeline_stages;  // Account for double/multi-buffering
        mapping.smem_per_block = smem;
    }

    return mapping;
}

// ---------------------------------------------------------------------------
// ComputeGridBlockDims
//
// Extract the iteration counts from band nodes and assign them to
// grid/block dimensions considering hardware constraints.
// ---------------------------------------------------------------------------
std::pair<std::vector<int64_t>, std::vector<int64_t>> ComputeGridBlockDims(
    const ScheduleTreePtr& tree,
    const hardware::HardwareTarget& target
) {
    auto bands = tree->band_nodes();

    if (bands.empty()) {
        // No bands → trivial kernel with 1 block, 1 thread
        return {{1}, {1}};
    }

    // Collect parallel and sequential dimension extents from all bands.
    // For a hierarchical tiling (grid → SM → warp), the outermost band
    // maps to grid dimensions and the next band maps to block dimensions.
    std::vector<int64_t> parallel_extents;
    std::vector<int64_t> sequential_extents;

    for (auto& band : bands) {
        auto& data = band->band_data();
        for (const auto& m : data.members) {
            int64_t extent = 0;
            for (auto c : m.coefficients) {
                extent = std::max(extent, std::abs(c));
            }
            extent = std::max(extent, int64_t(1));

            if (m.parallel) {
                parallel_extents.push_back(extent);
            } else {
                sequential_extents.push_back(extent);
            }
        }
    }

    std::vector<int64_t> grid_dims;
    std::vector<int64_t> block_dims;

    if (parallel_extents.empty()) {
        // No parallel dimensions – everything is sequential.
        // Use a single block with minimal threads.
        grid_dims = {1};
        block_dims = {1};
    } else {
        // Partition parallel dimensions into grid and block.
        // Strategy: outer parallel dims → grid, inner → block.
        // Constrain block dims to fit in max_threads_per_block.

        int64_t max_threads = target.gpu.max_threads_per_block;
        int64_t max_block_x = target.gpu.max_block_x;
        int64_t max_block_y = target.gpu.max_block_y;
        int64_t max_block_z = target.gpu.max_block_z;

        // Greedily assign inner parallel dims to block until we
        // exceed max_threads_per_block, then assign remaining to grid.
        int64_t block_product = 1;

        // Walk from the innermost parallel dim backward (inner dims
        // are more natural for threadIdx since they provide
        // coalesced memory access and warp-level parallelism).
        for (int i = static_cast<int>(parallel_extents.size()) - 1;
             i >= 0; --i) {
            int64_t ext = parallel_extents[i];
            if (block_product * ext <= max_threads &&
                static_cast<int64_t>(block_dims.size()) < 3) {
                block_dims.insert(block_dims.begin(), ext);
                block_product *= ext;
            } else {
                grid_dims.insert(grid_dims.begin(), ext);
            }
        }

        // Enforce CUDA dimension limits on block dims
        if (block_dims.size() > 0) {
            block_dims[0] = std::min(block_dims[0], max_block_x);
        }
        if (block_dims.size() > 1) {
            block_dims[1] = std::min(block_dims[1], max_block_y);
        }
        if (block_dims.size() > 2) {
            block_dims[2] = std::min(block_dims[2], max_block_z);
        }

        // Ensure grid dimensions don't exceed hardware limits
        if (grid_dims.empty()) {
            grid_dims = {1};
        }
        // Cap grid dimensions
        int64_t max_grid[3] = {
            target.gpu.max_grid_x,
            target.gpu.max_grid_y,
            target.gpu.max_grid_z
        };
        for (size_t d = 0; d < grid_dims.size() && d < 3; ++d) {
            grid_dims[d] = std::min(grid_dims[d], max_grid[d]);
        }
    }

    // Pad to at least 1 dimension each
    if (grid_dims.empty()) grid_dims = {1};
    if (block_dims.empty()) block_dims = {1};

    return {grid_dims, block_dims};
}

// ---------------------------------------------------------------------------
// WarpSpecialization
//
// Assign warp roles for warp-specialized kernels.  The standard pattern
// on Hopper (H100) is:
//   - 1 "producer" warp issues TMA bulk copies
//   - (n-1) "consumer" warps execute MMA operations
//   - Pipeline synchronization via barrier objects
//
// On non-TMA hardware, all warps are MIXED (they both load and compute).
// ---------------------------------------------------------------------------
GPUMapping WarpSpecialization(
    const GPUMapping& mapping,
    bool has_tma,
    int64_t pipeline_stages
) {
    GPUMapping result = mapping;

    int64_t n_warps = result.warps_per_block();
    if (n_warps <= 0) n_warps = 1;

    result.warp_roles.resize(static_cast<size_t>(n_warps));

    if (has_tma && n_warps >= 2 && pipeline_stages >= 2) {
        // Warp-specialized mode:
        //   Warp 0: PRODUCER (issues TMA copies)
        //   Warps 1..n-1: CONSUMER (execute MMA)
        result.warp_roles[0] = WarpRole::PRODUCER;
        for (int64_t w = 1; w < n_warps; ++w) {
            result.warp_roles[static_cast<size_t>(w)] = WarpRole::CONSUMER;
        }
    } else if (has_tma && n_warps == 1) {
        // Single warp must both produce and consume
        result.warp_roles[0] = WarpRole::MIXED;
    } else {
        // No TMA – all warps do both loading and computing
        for (int64_t w = 0; w < n_warps; ++w) {
            result.warp_roles[static_cast<size_t>(w)] = WarpRole::MIXED;
        }
    }

    // Adjust shared memory: add space for pipeline buffers
    // Each pipeline stage needs a buffer in shared memory.
    // The barrier synchronization space is typically 256 bytes per stage.
    int64_t barrier_smem = pipeline_stages * 256;
    result.smem_per_block += barrier_smem;

    return result;
}

} // namespace symplex::schedule
