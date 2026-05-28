// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/schedule_tree.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <string>
#include <optional>
#include <cstdint>

namespace symplex::schedule {

/// WarpRole: the role assigned to a warp within a thread block.
/// Used for warp-specialized kernels where different warps handle
/// different phases (e.g., TMA load vs. compute).
enum class WarpRole {
    PRODUCER,   ///< Loads data from global memory (TMA / cp.async)
    CONSUMER,   ///< Computes on data in shared memory / registers
    MIXED,      ///< Both loads and computes (general purpose)
};

inline std::string warp_role_to_string(WarpRole r) {
    switch (r) {
        case WarpRole::PRODUCER: return "PRODUCER";
        case WarpRole::CONSUMER: return "CONSUMER";
        case WarpRole::MIXED:    return "MIXED";
    }
    return "UNKNOWN";
}

/// GPUMapping: the result of mapping a schedule tree onto GPU hardware.
/// Contains the launch configuration (grid/block dimensions), warp role
/// assignments, and shared memory budget.
struct GPUMapping {
    std::vector<int64_t> grid_dims;        ///< Grid dimensions (blockIdx.x/y/z)
    std::vector<int64_t> block_dims;       ///< Block dimensions (threadIdx.x/y/z)
    std::vector<WarpRole> warp_roles;      ///< Role per warp in the block
    int64_t               smem_per_block;  ///< Shared memory per block (bytes)

    GPUMapping() : smem_per_block(0) {}

    /// Total number of threads per block.
    [[nodiscard]] int64_t threads_per_block() const {
        int64_t total = 1;
        for (auto d : block_dims) total *= d;
        return total;
    }

    /// Total number of warps per block.
    [[nodiscard]] int64_t warps_per_block() const {
        // Assume warp_size = 32
        return (threads_per_block() + 31) / 32;
    }

    /// Total number of blocks in the grid.
    [[nodiscard]] int64_t total_blocks() const {
        int64_t total = 1;
        for (auto d : grid_dims) total *= d;
        return total;
    }

    /// Number of producer warps.
    [[nodiscard]] int64_t num_producer_warps() const {
        int64_t count = 0;
        for (auto r : warp_roles) {
            if (r == WarpRole::PRODUCER) ++count;
        }
        return count;
    }

    /// Number of consumer warps.
    [[nodiscard]] int64_t num_consumer_warps() const {
        int64_t count = 0;
        for (auto r : warp_roles) {
            if (r == WarpRole::CONSUMER) ++count;
        }
        return count;
    }

    std::string to_string() const;
};

/// MarkParallelDims: scan all band nodes in the schedule tree and mark
/// each dimension that carries no dependencies as parallel.
///
/// A band dimension is parallel if and only if no dependency vector has
/// a non-zero component in that dimension (i.e., the dimension is
/// "coincidence-free").  This is determined by checking each dependency
/// polyhedron in the iteration space.
///
/// \param tree         The schedule tree to annotate
/// \param iterspace    The iteration space with dependency information
/// \return             Number of dimensions marked as parallel
int MarkParallelDims(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace
);

/// MapToGPU: convert parallel band dimensions in the schedule tree to
/// GPU blockIdx / threadIdx bindings.
///
/// This maps the outermost parallel dimensions to blockIdx and the
/// innermost parallel dimensions to threadIdx.  Sequential dimensions
/// remain as loop iterations within each thread.
///
/// The mapping strategy depends on the number of parallel dimensions:
///   - 1 parallel dim → blockIdx.x
///   - 2 parallel dims → blockIdx.x, blockIdx.y
///   - 3+ parallel dims → blockIdx.x/y/z, threadIdx.x/y/z
///
/// \param tree         The schedule tree with parallel annotations
/// \param target       The GPU hardware target
/// \return             The GPU mapping (grid/block dims, warp roles, etc.)
GPUMapping MapToGPU(
    const ScheduleTreePtr& tree,
    const hardware::HardwareTarget& target
);

/// ComputeGridBlockDims: calculate the actual GPU grid and block
/// dimensions from the schedule tree's band structure.
///
/// This extracts the iteration counts from band nodes and maps them
/// to grid/block dimensions considering hardware constraints such as
/// max threads per block, max grid dimensions, etc.
///
/// \param tree         The schedule tree
/// \param target       The GPU hardware target
/// \return             Pair of (grid_dims, block_dims)
std::pair<std::vector<int64_t>, std::vector<int64_t>> ComputeGridBlockDims(
    const ScheduleTreePtr& tree,
    const hardware::HardwareTarget& target
);

/// WarpSpecialization: partition warps in a thread block into
/// producer/consumer roles for asynchronous TMA-based kernels.
///
/// In warp-specialized kernels (e.g., Hopper's TMA pipeline):
///   - Producer warps issue TMA copies from global to shared memory
///   - Consumer warps execute the compute kernel (MMA instructions)
///   - The partition is typically 1 producer warp + (n-1) consumer warps
///
/// \param mapping      The GPU mapping to specialize
/// \param has_tma      Whether the target supports TMA
/// \param pipeline_stages  Number of software pipeline stages
/// \return             Updated GPUMapping with warp roles assigned
GPUMapping WarpSpecialization(
    const GPUMapping& mapping,
    bool has_tma,
    int64_t pipeline_stages
);

} // namespace symplex::schedule
