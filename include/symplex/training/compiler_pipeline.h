// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/optimizer/superoptimizer.h"
#include "symplex/codegen/code_generator.h"
#include "symplex/schedule/schedule_map.h"
#include "symplex/fusion/fusion_engine.h"
#include <string>
#include <vector>
#include <memory>

namespace symplex::training {

struct PipelineResult {
    std::string ptx_source;
    std::string kernel_name;
    schedule::TileConfig optimal_tile;
    double estimated_latency_ns;
    double speedup_vs_naive;
    std::vector<int64_t> grid_dims;
    std::vector<int64_t> block_dims;
    bool valid;
    std::string error;

    // Fusion engine results — populated during compilation
    fusion::FusionDecision fusion_decision;
    int64_t fusion_hbm_savings_bytes = 0;  // Alias for total_hbm_reduction_bytes
    double fusion_estimated_speedup = 0.0;  // Alias for total_estimated_speedup
};

/// Compiler pipeline with integrated fusion engine.
///
/// Architecture:
///
///   E-Graph Semantic Optimizer  →  discovers equivalent programs
///           ↓
///   Fusion Engine               →  decides WHAT can fuse (semantic)
///           ↓
///   Polyhedral Engine           →  decides WHETHER legal + HOW to schedule
///           ↓
///   MCMC Hardware Search        →  maps to GPU/hardware
///           ↓
///   Kernel Generation           →  emits code
///           ↓
///   Empirical Feedback          →  validates and adapts
///
/// The key architectural principle: fusion decides WHAT (meaning),
/// polyhedral decides WHETHER and HOW (correctness + scheduling).
/// These are fundamentally different search problems.
/// Conflating them creates "archaeological expeditions through affine maps."
class CompilerPipeline {
public:
    explicit CompilerPipeline(hardware::HardwareTarget target);

    // Run the full pipeline: iteration space -> optimized PTX kernel
    // Now includes fusion discovery as Step 0
    PipelineResult compile(const polyhedral::IterationSpace& ispace);

    // Compile with a specific tile configuration (skip optimization)
    PipelineResult compile_with_tile(
        const polyhedral::IterationSpace& ispace,
        const schedule::TileConfig& tile
    );

    // Compile for matmul specifically
    PipelineResult compile_matmul(int64_t M, int64_t N, int64_t K);

    // Compile for conv2d
    PipelineResult compile_conv2d(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride = 1, int64_t pad = 0
    );

    /// Compile with explicit fusion decisions provided externally.
    /// This allows the caller to control fusion boundaries while
    /// still using the polyhedral engine for legality and scheduling.
    PipelineResult compile_with_fusion(
        const polyhedral::IterationSpace& ispace,
        const fusion::FusionDecision& fusion_decision
    );

    /// Discover fusion opportunities for an iteration space.
    /// Returns fusion boundaries without performing compilation.
    fusion::FusionDecision discover_fusion(
        const std::vector<fusion::FusionOp>& ops
    );

    const hardware::HardwareTarget& target() const;

    /// Access the fusion engine directly
    fusion::FusionEngine& fusion_engine() { return *fusion_engine_; }
    const fusion::FusionEngine& fusion_engine() const { return *fusion_engine_; }

private:
    hardware::HardwareTarget target_;
    optimizer::Superoptimizer superopt_;
    codegen::CodeGenerator codegen_;
    std::unique_ptr<fusion::FusionEngine> fusion_engine_;
};

} // namespace symplex::training
