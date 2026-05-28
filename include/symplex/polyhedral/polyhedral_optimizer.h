// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/polyhedral/iteration_space.h"
#include "symplex/polyhedral/affine_map.h"
#include "symplex/schedule/schedule_tree.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <cstdint>
#include <memory>

namespace symplex::polyhedral {

/// PolyhedralOptimizer: takes an IterationSpace and produces an optimal schedule.
/// This is the "real" polyhedral model implementation with:
/// 1. Dependence analysis (already in IterationSpace)
/// 2. Legality checking (transform must preserve all deps)
/// 3. Affine scheduling (Feautrier's algorithm + Pluto-like)
/// 4. Tiling for data locality
/// 5. Parallelism extraction
/// 6. Fusion decisions
class PolyhedralOptimizer {
public:
    struct Config {
        int64_t sram_budget_bytes = 192 * 1024;  // 192KB shared memory (A100)
        int64_t max_tile_dim = 256;
        bool enable_parallelism = true;
        bool enable_fusion = true;
        bool enable_tiling = true;
        bool target_tensor_cores = true;  // Align tile sizes to MMA shapes
        int64_t mma_m = 16, mma_n = 8, mma_k = 16;  // MMA fragment dimensions
    };

    struct ScheduleResult {
        std::vector<AffineMap> schedule_maps;      // One per statement
        schedule::ScheduleTreePtr schedule_tree;    // The schedule tree
        std::vector<int64_t> tile_sizes;            // Optimal tile sizes
        std::vector<bool> parallel_dims;            // Which dims can be parallel
        int64_t estimated_sram_bytes = 0;           // SRAM footprint
        double estimated_latency_ns = 0.0;          // Analytical latency
        bool valid = false;
        // ── Micro-kernel config from Rust engine ───────────────────────────
        double estimated_gflops = 0.0;              // Roofline GFLOPS estimate
        size_t micro_kernel_tile_m = 0;             // Micro-kernel M tile
        size_t micro_kernel_tile_n = 0;             // Micro-kernel N tile
        size_t micro_kernel_tile_k = 0;             // Micro-kernel K tile
        size_t accumulator_registers = 0;           // Accumulator register count
        size_t prefetch_distance = 0;               // Prefetch distance
        uint8_t simd_level = 0;                     // SIMD level (0=None, 4=AVX512)
    };

    explicit PolyhedralOptimizer();
    explicit PolyhedralOptimizer(Config config);

    /// Main entry: compute optimal schedule for an iteration space.
    ScheduleResult optimize(const IterationSpace& ispace);

private:
    Config config_;

    /// Feautrier's 1-d scheduling algorithm.
    /// For each statement, finds the affine schedule that maximizes
    /// parallelism while preserving all dependencies.
    /// Returns one AffineMap per statement.
    std::vector<AffineMap> feautrier_schedule(const IterationSpace& ispace);

    /// Pluto-like multi-dimensional scheduling.
    /// Finds a permutable band of affine loops that maximizes
    /// tiling legality and parallelism.
    std::vector<AffineMap> pluto_schedule(const IterationSpace& ispace);

    /// Compute optimal tile sizes given SRAM budget.
    /// Uses analytical model: tile_footprint = sum of all tile tensors.
    std::vector<int64_t> compute_tile_sizes(
        const IterationSpace& ispace,
        const std::vector<AffineMap>& schedules,
        const std::vector<size_t>& band_dims
    );

    /// Check if a transformation is legal (preserves all dependencies).
    bool is_legal(const AffineMap& T, const IterationSpace& ispace);

    /// Build a schedule tree from schedule maps.
    schedule::ScheduleTreePtr build_schedule_tree(
        const IterationSpace& ispace,
        const std::vector<AffineMap>& schedules
    );

    /// Estimate kernel latency using roofline model.
    double estimate_latency(
        const IterationSpace& ispace,
        const std::vector<int64_t>& tile_sizes
    );

    /// Gather all dependency vectors from the iteration space.
    std::vector<DependencyVector> gather_all_dep_vectors(const IterationSpace& ispace) const;

    /// Greedy Feautrier LP solver for a single statement.
    /// Solves: find theta such that theta . d >= 1 for all deps,
    /// maximizing parallelism (preferring zero coefficients).
    AffineMap solve_feautrier_row(
        size_t ndim,
        const std::vector<DependencyVector>& deps
    );

    /// Pluto cost: count the number of satisfied dependencies for
    /// a candidate scheduling row.
    int64_t pluto_cost(
        const std::vector<int64_t>& row,
        const std::vector<DependencyVector>& deps
    ) const;

    /// Check if a scheduling row preserves all dependencies
    /// (i.e., row . d >= 0 for all dep vectors d, with at least
    /// one strictly positive for each dep).
    bool row_is_legal(
        const std::vector<int64_t>& row,
        const std::vector<DependencyVector>& deps
    ) const;

    /// Align a tile size to MMA fragment dimensions.
    int64_t align_to_mma(int64_t size, int64_t mma_dim) const;

    /// Whether to use the Rust polyhedral engine (primary) vs C++ fallback.
    bool use_rust_engine_;

    /// Optimize using the Rust polyhedral engine via FFI.
    ScheduleResult optimize_via_rust(const IterationSpace& ispace);
};

} // namespace symplex::polyhedral
