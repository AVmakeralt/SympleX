// SympleX – Polyhedral Tensor Superoptimizer
// MLIR Lowering Bridge: Converts Optimized SympleX IR to MLIR
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// This bridge emits clean MLIR assembly text that can be:
//   1. Parsed by mlir-opt for optimization
//   2. Lowered by mlir-translate to LLVM IR
//   3. JIT-compiled by LLVM ORC
//
// Pipeline:
//   SympleX IR → MLIR (affine + linalg + tensor dialects)
//   Then external tools can: MLIR → LLVM IR → ORC JIT
//
// Since we can't link against MLIR (massive dependency), we generate
// MLIR-compatible text. This is a well-established pattern used by
// many compiler projects (e.g., Torch-MLIR's initial approach).

#pragma once

#include "symplex/ir/symplex_ir.h"
#include "symplex/polyhedral/affine_map.h"
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <optional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <functional>

namespace symplex::lowering {

// ─────────────────────────────────────────────────────────────────────────
// MLIR Dialects
// ─────────────────────────────────────────────────────────────────────────

/// MLIR Dialects we target.
enum class MLIRDialect {
    AFFINE,    // For loop nests and schedule maps
    LINALG,    // For tensor algebra (matmul, conv, elementwise)
    TENSOR,    // For tensor type definitions
    VECTOR,    // For vectorized/SIMD operations
    GPU,       // For GPU kernel launches
    ARITH,     // For arithmetic operations
    FUNC,      // For function definitions
    SCF,       // For structured control flow (loops, conditionals)
};

inline std::string dialect_to_string(MLIRDialect d) {
    switch (d) {
        case MLIRDialect::AFFINE:  return "affine";
        case MLIRDialect::LINALG:  return "linalg";
        case MLIRDialect::TENSOR:  return "tensor";
        case MLIRDialect::VECTOR:  return "vector";
        case MLIRDialect::GPU:     return "gpu";
        case MLIRDialect::ARITH:   return "arith";
        case MLIRDialect::FUNC:    return "func";
        case MLIRDialect::SCF:     return "scf";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────────────
// MLIRLoweringConfig
// ─────────────────────────────────────────────────────────────────────────

/// MLIRLoweringConfig: configuration for the lowering pass.
struct MLIRLoweringConfig {
    bool emit_affine_loops = true;           // Use affine.for instead of scf.for
    bool emit_linalg_ops = true;             // Use linalg.matmul etc.
    bool emit_gpu_kernel = true;             // Wrap in gpu.launch
    bool target_cuda = true;                 // Emit CUDA-specific patterns
    int64_t vector_length = 128;             // Vector width for vector dialect
    bool emit_tensor_core_intrinsics = true;  // nvgpu.mma_sync etc.
    bool emit_shared_memory = true;          // Use shared memory for tiling
    int64_t shared_memory_bytes = 49152;     // Default shared memory size (48KB)
    bool emit_warp_cooperative = true;       // Warp-cooperative MMA
    std::string kernel_name_prefix = "symplex_";  // Prefix for kernel names
};

// ─────────────────────────────────────────────────────────────────────────
// MLIRLoweringResult
// ─────────────────────────────────────────────────────────────────────────

/// MLIRLoweringResult: the output of the lowering pass.
struct MLIRLoweringResult {
    std::string mlir_text;             // The generated MLIR assembly text
    std::string kernel_name;           // Name of the generated kernel function
    std::vector<std::string> inputs;   // Input tensor names
    std::vector<ir::IRShape> input_shapes; // Input tensor shapes
    std::string output_name;           // Output tensor name
    ir::IRShape output_shape;          // Output tensor shape
    bool valid = false;
    std::string error_message;         // Human-readable error if !valid

    /// Statistics
    int64_t num_loops = 0;
    int64_t num_linalg_ops = 0;
    int64_t num_gpu_ops = 0;
    int64_t num_tensor_core_ops = 0;
};

// ─────────────────────────────────────────────────────────────────────────
// MLIRLowering
// ─────────────────────────────────────────────────────────────────────────

/// MLIRLowering: converts optimized SympleX IR to MLIR text.
///
/// The lowering pipeline is:
///   1. Analyze the IR to determine op kinds and shapes
///   2. Emit module header with required dialects
///   3. Emit function signature with tensor-typed arguments
///   4. For each op:
///      - Elementwise ops → linalg.generic or arith ops
///      - MatMul → linalg.matmul (with optional tiling)
///      - Reductions → linalg.generic with reduction iterator
///      - Fused ops → combined linalg ops
///      - Neural ops → decomposed or fused patterns
///   5. Optionally wrap in gpu.launch for GPU execution
///   6. Optionally emit tensor core intrinsics (nvgpu.mma_sync)
class MLIRLowering {
public:
    explicit MLIRLowering(MLIRLoweringConfig config = {});

    /// Lower a SympleX IR module to MLIR.
    MLIRLoweringResult lower(const ir::SympleXIR& ir_module);

    /// Lower with a specific schedule (polyhedral schedule maps applied).
    MLIRLoweringResult lower_with_schedule(
        const ir::SympleXIR& ir_module,
        const std::vector<polyhedral::AffineMap>& schedules,
        const std::vector<int64_t>& tile_sizes
    );

private:
    MLIRLoweringConfig config_;

    // ── Emit helpers ────────────────────────────────────────────────

    /// Emit module header with required dialect declarations.
    std::string emit_module_header(const std::string& name);

    /// Emit a tensor type declaration, e.g., "tensor<1024x1024xf16>"
    std::string emit_tensor_type(const ir::IRShape& shape, ir::IRDType dtype);

    /// Emit function header with tensor arguments.
    std::string emit_function_header(
        const std::string& name,
        const std::vector<std::string>& input_names,
        const std::vector<ir::IRShape>& input_shapes,
        const std::vector<ir::IRDType>& input_dtypes,
        const ir::IRShape& output_shape,
        ir::IRDType output_dtype,
        bool is_gpu_kernel = false
    );

    /// Emit a linalg operation.
    std::string emit_linalg_op(
        ir::IROp::Kind kind,
        const std::string& result_name,
        const std::vector<std::string>& operand_names,
        const ir::IRShape& result_shape,
        ir::IRDType dtype
    );

    /// Emit an elementwise linalg.generic op.
    std::string emit_elementwise_op(
        ir::IROp::Kind kind,
        const std::string& result_name,
        const std::vector<std::string>& operand_names,
        const ir::IRShape& result_shape,
        ir::IRDType dtype
    );

    /// Emit a reduction op.
    std::string emit_reduction_op(
        ir::IROp::Kind kind,
        const std::string& result_name,
        const std::string& operand_name,
        const ir::IRShape& input_shape,
        const ir::IRShape& result_shape,
        ir::IRDType dtype,
        int64_t axis
    );

    /// Emit an affine loop nest.
    std::string emit_affine_loop(
        const std::string& var, int64_t lo, int64_t hi,
        const std::string& body, int indent = 4
    );

    /// Emit tiled affine loops.
    std::string emit_tiled_loop(
        const std::vector<int64_t>& tile_sizes,
        const std::vector<std::pair<int64_t, int64_t>>& bounds,
        const std::string& body,
        int indent = 4
    );

    /// Emit gpu.launch wrapper.
    std::string emit_gpu_launch(
        const std::string& kernel_body,
        const std::vector<int64_t>& grid_dims,
        const std::vector<int64_t>& block_dims,
        int indent = 4
    );

    /// Emit tensor core MMA operation (nvgpu.mma_sync).
    std::string emit_tensor_core_mma(
        const std::string& A, const std::string& B,
        const std::string& C, int64_t m, int64_t n, int64_t k,
        int indent = 4
    );

    /// Emit a return statement.
    std::string emit_return(const std::string& result, int indent = 4);

    /// Emit an arith constant.
    std::string emit_constant(
        const std::string& name, double value,
        ir::IRDType dtype, int indent = 4
    );

    /// Emit shared memory allocation.
    std::string emit_shared_memory_alloc(
        const std::string& name,
        const ir::IRShape& shape,
        ir::IRDType dtype,
        int indent = 4
    );

    // ── Op classification ───────────────────────────────────────────

    /// Is this op an elementwise operation?
    static bool is_elementwise_op(ir::IROp::Kind kind);

    /// Is this op a reduction?
    static bool is_reduction_op(ir::IROp::Kind kind);

    /// Is this op a matmul variant?
    static bool is_matmul_op(ir::IROp::Kind kind);

    /// Is this op a normalization op?
    static bool is_norm_op(ir::IROp::Kind kind);

    /// Get the arith op string for elementwise ops.
    static std::string arith_op_string(ir::IROp::Kind kind);

    /// Get the linalg op name for elementwise ops.
    static std::string linalg_op_name(ir::IROp::Kind kind);

    // ── ID management ───────────────────────────────────────────────

    /// Generate an SSA name for an IR op.
    std::string ssa_name(int64_t op_id) const;

    /// Generate a unique name for temporaries.
    std::string unique_name(const std::string& prefix);

    int64_t unique_counter_ = 0;

    // ── Indent helper ───────────────────────────────────────────────

    static std::string indent_str(int spaces) {
        return std::string(static_cast<size_t>(spaces), ' ');
    }
};

} // namespace symplex::lowering
