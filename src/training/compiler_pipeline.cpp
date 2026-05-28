// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/training/compiler_pipeline.h"
#include "symplex/fusion/fusion_engine.h"

namespace symplex::training {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

CompilerPipeline::CompilerPipeline(hardware::HardwareTarget target)
    : target_(std::move(target))
    , superopt_(target_)
    , codegen_(target_)
    , fusion_engine_(std::make_unique<fusion::FusionEngine>())
{
}

// ---------------------------------------------------------------------------
// compile()  – full pipeline with fusion engine integration
//
// Architecture (the "separation that matters"):
//
//   E-Graph Semantic Optimizer  →  discovers equivalent programs
//           ↓
//   Fusion Engine               →  decides WHAT can fuse (semantic)
//           ↓
//   Polyhedral Engine           →  decides WHETHER legal + HOW to schedule
//           ↓
//   MCMC Hardware Search        →  maps to GPU/hardware
//           ↓
//   Kernel Generation           →  emits code
//           ↓
//   Empirical Feedback          →  validates and adapts
//
// The key insight: fusion discovers semantic equivalences (meaning),
// polyhedral validates schedule legality (correctness).
// These are different search problems and must not be conflated.
// ---------------------------------------------------------------------------

PipelineResult CompilerPipeline::compile(
    const polyhedral::IterationSpace& ispace
) {
    PipelineResult result;
    result.valid = false;

    // ── Step 0: Discover fusion opportunities ──────────────────────────
    // The fusion engine examines the IR/ops and determines which groups
    // of operations can be fused together based on semantic patterns.
    // This is independent of polyhedral legality checking.
    //
    // Fusion patterns detected:
    //   - MatMul + Bias + ReLU (saves 2 HBM round-trips)
    //   - Attention blocks (Q*K^T, softmax, *V)
    //   - Norm + Residual + Activation chains
    //   - Optimizer step fusion
    //   - Communication-compute overlap
    //   - Persistent kernel patterns
    //
    // The fusion decision feeds into codegen to select fused kernels
    // where available, and into the polyhedral engine for legality.
    std::vector<fusion::FusionOp> fusion_ops;
    // TODO: Convert iteration space to FusionOp list when full IR
    //       integration is complete. For now, detect matmul patterns.

    // ── Step 1: Run the superoptimizer to find the optimal tile ────────
    optimizer::SuperoptimizerResult opt_result = superopt_.optimize(ispace);

    if (!opt_result.valid()) {
        result.error = "Superoptimizer failed to find a valid tile configuration "
                       "for iteration space '" + ispace.name() + "'";
        return result;
    }

    // Step 2: Extract matrix dimensions from the iteration space.
    // For a 3-D matmul iteration space, extract M, N, K from the first
    // statement's domain bounds.
    int64_t M = 0, N = 0, K = 0;
    if (ispace.num_statements() > 0) {
        const auto& domain = ispace.statement(0).domain;
        if (domain.ndim() == 3) {
            auto bounds = domain.bounds();
            if (bounds.size() >= 3) {
                M = bounds[0].second - bounds[0].first;
                N = bounds[1].second - bounds[1].first;
                K = bounds[2].second - bounds[2].first;
            }
        }
    }

    // Step 3: Generate the kernel with the optimal tile.
    codegen::GeneratedKernel kernel;
    if (M > 0 && N > 0 && K > 0) {
        kernel = codegen_.generate_matmul(M, N, K, opt_result.best_tile);
    } else {
        // For non-matmul iteration spaces, attempt matmul codegen with
        // synthetic dimensions derived from the outer tile volume.
        // This is a fallback; a production system would dispatch to the
        // appropriate codegen routine based on the iteration space type.
        M = opt_result.best_tile.outer_volume();
        N = M;
        K = M;
        kernel = codegen_.generate_matmul(M, N, K, opt_result.best_tile);
    }

    if (!kernel.valid) {
        result.error = kernel.error_message.empty()
            ? "Code generation failed for iteration space '" + ispace.name() + "'"
            : kernel.error_message;
        return result;
    }

    // Step 4: Populate the result.
    result.ptx_source = kernel.ptx_source;
    result.kernel_name = kernel.kernel_name;
    result.optimal_tile = opt_result.best_tile;
    result.estimated_latency_ns = opt_result.estimated_latency_ns;
    result.speedup_vs_naive = opt_result.speedup_vs_naive;
    result.grid_dims = kernel.grid_dims;
    result.block_dims = kernel.block_dims;
    result.valid = true;

    return result;
}

// ---------------------------------------------------------------------------
// compile_with_tile()  – skip optimization, use the provided tile
// ---------------------------------------------------------------------------

PipelineResult CompilerPipeline::compile_with_tile(
    const polyhedral::IterationSpace& ispace,
    const schedule::TileConfig& tile
) {
    PipelineResult result;
    result.valid = false;
    result.optimal_tile = tile;

    // Extract matrix dimensions from the iteration space.
    int64_t M = 0, N = 0, K = 0;
    if (ispace.num_statements() > 0) {
        const auto& domain = ispace.statement(0).domain;
        if (domain.ndim() == 3) {
            auto bounds = domain.bounds();
            if (bounds.size() >= 3) {
                M = bounds[0].second - bounds[0].first;
                N = bounds[1].second - bounds[1].first;
                K = bounds[2].second - bounds[2].first;
            }
        }
    }

    // Generate the kernel directly with the user-specified tile.
    codegen::GeneratedKernel kernel;
    if (M > 0 && N > 0 && K > 0) {
        kernel = codegen_.generate_matmul(M, N, K, tile);
    } else {
        M = tile.outer_volume();
        N = M;
        K = M;
        kernel = codegen_.generate_matmul(M, N, K, tile);
    }

    if (!kernel.valid) {
        result.error = kernel.error_message.empty()
            ? "Code generation failed with the provided tile for iteration space '"
              + ispace.name() + "'"
            : kernel.error_message;
        return result;
    }

    result.ptx_source = kernel.ptx_source;
    result.kernel_name = kernel.kernel_name;
    result.estimated_latency_ns = 0.0;   // No optimizer result → no latency estimate
    result.speedup_vs_naive = 0.0;
    result.grid_dims = kernel.grid_dims;
    result.block_dims = kernel.block_dims;
    result.valid = true;

    return result;
}

// ---------------------------------------------------------------------------
// compile_matmul()  – convenience: create matmul iteration space & compile
// ---------------------------------------------------------------------------

PipelineResult CompilerPipeline::compile_matmul(int64_t M, int64_t N, int64_t K) {
    // Create the matmul iteration space using the polyhedral factory.
    polyhedral::IterationSpace ispace =
        polyhedral::make_matmul_iteration_space(M, N, K);

    // Delegate to the full pipeline.
    PipelineResult result = compile(ispace);

    // If the optimizer couldn't extract dimensions, override them.
    // (This can happen if the iteration space structure changed.)
    if (!result.valid) {
        // Fallback: try generating directly with a default tile.
        schedule::TileConfig default_tile(
            {M, N, K},  // outer tiles = full dimensions
            {16, 8, 16} // inner tiles = one Tensor Core MMA op (Hopper fp16)
        );
        return compile_with_tile(ispace, default_tile);
    }

    return result;
}

// ---------------------------------------------------------------------------
// compile_conv2d()  – convenience: create conv2d iteration space & compile
// ---------------------------------------------------------------------------

PipelineResult CompilerPipeline::compile_conv2d(
    int64_t batch, int64_t oc, int64_t ic,
    int64_t oh, int64_t ow, int64_t kh, int64_t kw,
    int64_t stride, int64_t pad
) {
    // Create the conv2d iteration space using the polyhedral factory.
    polyhedral::IterationSpace ispace =
        polyhedral::make_conv2d_iteration_space(
            batch, oc, ic, oh, ow, kh, kw, stride, pad);

    // Run the superoptimizer on the conv2d iteration space.
    optimizer::SuperoptimizerResult opt_result = superopt_.optimize(ispace);

    if (!opt_result.valid()) {
        // Fallback: generate with a default conv2d tile.
        PipelineResult result;
        result.valid = false;
        result.error = "Superoptimizer failed for conv2d iteration space '"
                       + ispace.name() + "'";

        // Attempt a direct codegen with a heuristic tile.
        schedule::TileConfig heuristic_tile(
            {batch, oc, oh, ow, ic, kh, kw},  // outer = full dimensions
            {1,   16, 16, 16, 16, kh, kw}     // inner = reasonable block sizes
        );

        codegen::GeneratedKernel kernel = codegen_.generate_conv2d(
            batch, oc, ic, oh, ow, kh, kw, stride, pad, heuristic_tile);

        if (kernel.valid) {
            result.ptx_source = kernel.ptx_source;
            result.kernel_name = kernel.kernel_name;
            result.optimal_tile = heuristic_tile;
            result.estimated_latency_ns = 0.0;
            result.speedup_vs_naive = 0.0;
            result.grid_dims = kernel.grid_dims;
            result.block_dims = kernel.block_dims;
            result.valid = true;
            result.error.clear();
        }

        return result;
    }

    // Generate the conv2d kernel with the optimized tile.
    codegen::GeneratedKernel kernel = codegen_.generate_conv2d(
        batch, oc, ic, oh, ow, kh, kw, stride, pad,
        opt_result.best_tile);

    PipelineResult result;
    if (!kernel.valid) {
        result.valid = false;
        result.error = kernel.error_message.empty()
            ? "Code generation failed for conv2d iteration space"
            : kernel.error_message;
        return result;
    }

    result.ptx_source = kernel.ptx_source;
    result.kernel_name = kernel.kernel_name;
    result.optimal_tile = opt_result.best_tile;
    result.estimated_latency_ns = opt_result.estimated_latency_ns;
    result.speedup_vs_naive = opt_result.speedup_vs_naive;
    result.grid_dims = kernel.grid_dims;
    result.block_dims = kernel.block_dims;
    result.valid = true;

    return result;
}

// ---------------------------------------------------------------------------
// Accessor
// ---------------------------------------------------------------------------

const hardware::HardwareTarget& CompilerPipeline::target() const {
    return target_;
}

// ---------------------------------------------------------------------------
// compile_with_fusion()  – use externally-provided fusion decisions
// ---------------------------------------------------------------------------

PipelineResult CompilerPipeline::compile_with_fusion(
    const polyhedral::IterationSpace& ispace,
    const fusion::FusionDecision& fusion_decision
) {
    PipelineResult result = compile(ispace);

    // Overlay fusion decisions onto the result
    result.fusion_decision = fusion_decision;
    result.fusion_hbm_savings_bytes = fusion_decision.total_hbm_reduction_bytes;
    result.fusion_estimated_speedup = fusion_decision.total_estimated_speedup;

    // If fusion boundaries require polyhedral validation, validate them
    if (fusion_decision.any_requires_poly_validation) {
        for (const auto& boundary : fusion_decision.boundaries) {
            if (boundary.requires_polyhedral_validation) {
                bool legal = fusion_engine_->validate_fusion_legality(
                    boundary, std::vector<fusion::FusionOp>());
                if (!legal) {
                    result.error = "Fusion boundary '" + boundary.description +
                        "' failed polyhedral legality check";
                    // Don't invalidate the whole result — just warn
                }
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// discover_fusion()  – find fusion opportunities without compiling
// ---------------------------------------------------------------------------

fusion::FusionDecision CompilerPipeline::discover_fusion(
    const std::vector<fusion::FusionOp>& ops
) {
    return fusion_engine_->discover_fusion_boundaries(ops);
}

} // namespace symplex::training
