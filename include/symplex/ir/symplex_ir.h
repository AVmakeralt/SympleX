// SympleX – Polyhedral Tensor Superoptimizer
// SympleX IR: Custom Search Space IR for Superoptimization
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// SympleX IR is the "optimization sandbox" IR that owns:
//   - E-graph equivalence classes and algebraic rewrite space
//   - Tensor expressions before scheduling
//   - Fused operator candidates
//   - Polyhedral annotations (iteration spaces attached to ops)
//   - Cost model evaluations
//
// This is NOT MLIR. MLIR is the lowering target, not the working IR.
// SympleX IR is the search space for superoptimization.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cassert>
#include <cmath>
#include <queue>
#include <functional>
#include <memory>

namespace symplex::ir {

// ─────────────────────────────────────────────────────────────────────────
// Core Types
// ─────────────────────────────────────────────────────────────────────────

/// IRDType: data type enumeration matching the e-graph DType.
enum class IRDType : uint8_t {
    FP64, FP32, FP16, BF16, INT8, INT4, UNKNOWN
};

/// Convert IRDType to human-readable string.
inline std::string irdtype_to_string(IRDType dt) {
    switch (dt) {
        case IRDType::FP64:    return "fp64";
        case IRDType::FP32:    return "fp32";
        case IRDType::FP16:    return "fp16";
        case IRDType::BF16:    return "bf16";
        case IRDType::INT8:    return "int8";
        case IRDType::INT4:    return "int4";
        case IRDType::UNKNOWN: return "?";
    }
    return "?";
}

/// Convert IRDType to MLIR-style type string.
inline std::string irdtype_to_mlir(IRDType dt) {
    switch (dt) {
        case IRDType::FP64:    return "f64";
        case IRDType::FP32:    return "f32";
        case IRDType::FP16:    return "f16";
        case IRDType::BF16:    return "bf16";
        case IRDType::INT8:    return "i8";
        case IRDType::INT4:    return "i4";
        case IRDType::UNKNOWN: return "?";
    }
    return "?";
}

/// Bytes per element for each IRDType.
inline int64_t irdtype_bytes(IRDType dt) {
    switch (dt) {
        case IRDType::FP64:    return 8;
        case IRDType::FP32:    return 4;
        case IRDType::FP16:    return 2;
        case IRDType::BF16:    return 2;
        case IRDType::INT8:    return 1;
        case IRDType::INT4:    return 1;  // Packed
        case IRDType::UNKNOWN: return 2;  // Conservative default
    }
    return 2;
}

// ─────────────────────────────────────────────────────────────────────────
// IRShape
// ─────────────────────────────────────────────────────────────────────────

/// IRShape: tensor shape with optional symbolic dimensions.
/// -1 means symbolic (unknown at compile time).
struct IRShape {
    std::vector<int64_t> dims;

    IRShape() = default;
    IRShape(std::initializer_list<int64_t> d) : dims(d) {}
    explicit IRShape(std::vector<int64_t> d) : dims(std::move(d)) {}

    [[nodiscard]] size_t ndim() const { return dims.size(); }
    [[nodiscard]] int64_t operator[](size_t i) const {
        return i < dims.size() ? dims[i] : -1;
    }

    /// Total number of elements. Returns -1 if any dimension is symbolic.
    [[nodiscard]] int64_t num_elements() const {
        int64_t total = 1;
        for (auto d : dims) {
            if (d < 0) return -1;
            total *= d;
        }
        return total;
    }

    /// Is this shape entirely unknown?
    [[nodiscard]] bool is_unknown() const { return dims.empty(); }

    /// Are all dimensions concrete (non-symbolic)?
    [[nodiscard]] bool is_concrete() const {
        for (auto d : dims) {
            if (d < 0) return false;
        }
        return !dims.empty();
    }

    /// Total bytes for this shape given a dtype.
    [[nodiscard]] int64_t bytes(IRDType dt) const {
        int64_t n = num_elements();
        return (n < 0) ? -1 : n * irdtype_bytes(dt);
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << "x";
            if (dims[i] < 0) oss << "?";
            else oss << dims[i];
        }
        oss << "]";
        return oss.str();
    }

    /// MLIR-style shape string, e.g. "1024x1024" (without brackets).
    [[nodiscard]] std::string to_mlir_dims() const {
        std::ostringstream oss;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << "x";
            if (dims[i] < 0) oss << "?";
            else oss << dims[i];
        }
        return oss.str();
    }

    bool operator==(const IRShape& o) const { return dims == o.dims; }
    bool operator!=(const IRShape& o) const { return dims != o.dims; }
};

/// Compute broadcast shape from two input shapes (NumPy-style).
inline IRShape broadcast_shapes(const IRShape& a, const IRShape& b) {
    size_t max_ndim = std::max(a.ndim(), b.ndim());
    std::vector<int64_t> result(max_ndim, -1);
    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t da = (i < a.ndim()) ? a[a.ndim() - 1 - i] : 1;
        int64_t db = (i < b.ndim()) ? b[b.ndim() - 1 - i] : 1;
        if (da == db) {
            result[max_ndim - 1 - i] = da;
        } else if (da == 1) {
            result[max_ndim - 1 - i] = db;
        } else if (db == 1) {
            result[max_ndim - 1 - i] = da;
        } else if (da < 0) {
            result[max_ndim - 1 - i] = (db > 0) ? db : da;
        } else if (db < 0) {
            result[max_ndim - 1 - i] = (da > 0) ? da : db;
        } else {
            result[max_ndim - 1 - i] = -1;
        }
    }
    return IRShape(result);
}

/// Compute matmul output shape: [..., M, K] x [..., K, N] -> [..., M, N].
inline IRShape matmul_shapes(const IRShape& a, const IRShape& b) {
    if (a.ndim() < 1 || b.ndim() < 1) return IRShape({-1});
    if (a.ndim() == 1 && b.ndim() == 1) return IRShape({1});
    if (a.ndim() == 1 && b.ndim() == 2) return IRShape({b[1] < 0 ? -1 : b[1]});
    if (a.ndim() == 2 && b.ndim() == 1) return IRShape({a[0] < 0 ? -1 : a[0]});
    if (a.ndim() >= 2 && b.ndim() >= 2) {
        std::vector<int64_t> a_batch(a.dims.begin(), a.dims.end() - 2);
        std::vector<int64_t> b_batch(b.dims.begin(), b.dims.end() - 2);
        auto batch = broadcast_shapes(IRShape(a_batch), IRShape(b_batch));
        std::vector<int64_t> result = batch.dims;
        result.push_back(a[a.ndim() - 2] < 0 ? -1 : a[a.ndim() - 2]);
        result.push_back(b[b.ndim() - 1] < 0 ? -1 : b[b.ndim() - 1]);
        return IRShape(result);
    }
    return IRShape({-1});
}

/// Compute reduction output shape (remove axis dimension).
inline IRShape reduce_shape(const IRShape& input, int64_t axis) {
    if (input.is_unknown()) return IRShape();
    if (axis < 0 || static_cast<size_t>(axis) >= input.ndim()) return input;
    std::vector<int64_t> result;
    result.reserve(input.ndim() - 1);
    for (size_t i = 0; i < input.ndim(); ++i) {
        if (static_cast<int64_t>(i) != axis) {
            result.push_back(input[i]);
        }
    }
    return IRShape(result);
}

// ─────────────────────────────────────────────────────────────────────────
// IRTensorLayout
// ─────────────────────────────────────────────────────────────────────────

/// IRTensorLayout: memory layout of a tensor.
enum class IRTensorLayout : uint8_t {
    ROW_MAJOR,    // C-contiguous
    COL_MAJOR,    // Fortran-contiguous
    TILED_128,    // Swizzled for 128-byte transaction alignment
    TILED_32,     // MMA fragment layout (m16n8k16 etc.)
    UNKNOWN
};

inline std::string irlayout_to_string(IRTensorLayout layout) {
    switch (layout) {
        case IRTensorLayout::ROW_MAJOR: return "row_major";
        case IRTensorLayout::COL_MAJOR: return "col_major";
        case IRTensorLayout::TILED_128: return "tiled_128";
        case IRTensorLayout::TILED_32:  return "tiled_32";
        case IRTensorLayout::UNKNOWN:   return "unknown";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────────────
// IRAffineAnnotation: Polyhedral Information
// ─────────────────────────────────────────────────────────────────────────

/// IRAffineAnnotation: polyhedral information attached to an IR operation.
/// This carries the iteration space, access relations, and schedule metadata
/// that the polyhedral optimizer uses to reason about loop transformations.
struct IRAffineAnnotation {
    /// The iteration space this operation lives in (as a serializable description).
    /// Each pair is (lo, hi) for one loop dimension.
    std::vector<std::pair<int64_t, int64_t>> loop_bounds;

    /// Access relations: maps iteration vector to tensor coordinates.
    /// Indexed as [access_number][output_dim][input_dim].
    std::vector<std::vector<std::vector<int64_t>>> access_matrices;

    /// Access offsets: constant term for each access relation.
    /// Indexed as [access_number][output_dim].
    std::vector<std::vector<int64_t>> access_offsets;

    /// Is each access a write? (false = read)
    std::vector<bool> access_is_write;

    /// Is each loop dimension parallelizable?
    std::vector<bool> parallel_dims;

    /// Schedule map (if computed by the polyhedral optimizer).
    /// This is the transformation matrix S such that the new iteration
    /// vector = S * old_iteration_vector.
    /// Indexed as [output_dim][input_dim].
    std::optional<std::vector<std::vector<std::vector<int64_t>>>> schedule_matrix;

    /// Number of loop dimensions.
    [[nodiscard]] size_t num_dims() const { return loop_bounds.size(); }

    /// Is a given dimension parallelizable?
    [[nodiscard]] bool is_parallel(size_t dim) const {
        return dim < parallel_dims.size() && parallel_dims[dim];
    }

    /// Number of access relations.
    [[nodiscard]] size_t num_accesses() const { return access_matrices.size(); }

    /// Get loop extent for a dimension (hi - lo).
    [[nodiscard]] int64_t loop_extent(size_t dim) const {
        if (dim >= loop_bounds.size()) return 0;
        return loop_bounds[dim].second - loop_bounds[dim].first;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Affine{loops=[";
        for (size_t i = 0; i < loop_bounds.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "(" << loop_bounds[i].first << "," << loop_bounds[i].second << ")";
            if (i < parallel_dims.size() && parallel_dims[i]) oss << "*";
        }
        oss << "], accesses=" << access_matrices.size();
        if (schedule_matrix.has_value()) {
            oss << ", scheduled";
        }
        oss << "}";
        return oss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────
// IROp: Operation in SympleX IR
// ─────────────────────────────────────────────────────────────────────────

/// IROp: an operation in SympleX IR.
struct IROp {
    enum Kind : uint16_t {
        // Leaf nodes
        SYMBOL,
        CONSTANT,

        // Arithmetic
        ADD, MUL, SUB, DIV, NEG,

        // Linear algebra
        MATMUL, TRANSPOSE, RESHAPE, BROADCAST,

        // Reductions
        REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN,

        // Neural network
        RELU, GELU, SIGMOID, SOFTMAX, LAYERNORM, RMSNORM, DROPOUT,
        EXP, LOG, SQRT, RECIPROCAL,

        // Fused operations (superoptimizer discovery targets)
        FUSED_MATMUL_RELU,
        FUSED_MATMUL_ADD,
        FUSED_MATMUL_ADD_RELU,
        FUSED_GEMM,
        FUSED_SOFTMAX,
        FUSED_LAYERNORM,
        FUSED_RMSNORM,
        FUSED_ADD_LN,
        FUSED_MHA,

        // Meta / tiling
        TILE, UNTILE, IDENTITY,

        // Control flow
        SELECT,  // conditional: select(cond, a, b)
    };

    Kind                kind;
    std::vector<int64_t> operands;   // SSA operand IDs
    IRShape             shape;
    IRDType             dtype = IRDType::FP32;
    IRTensorLayout      layout = IRTensorLayout::UNKNOWN;

    // For constants
    double              float_value = 0.0;
    int64_t             int_value = 0;

    // For named symbols
    std::string         name;

    // For axis-parameterized ops (reductions, softmax, etc.)
    int64_t             axis = -1;

    // Polyhedral annotation (attached to compute-heavy ops)
    std::optional<IRAffineAnnotation> poly_annotation;

    // ── Helpers ─────────────────────────────────────────────────────

    /// How many operands does this op kind expect?
    [[nodiscard]] static int arity(Kind k) {
        switch (k) {
            case SYMBOL:
            case CONSTANT:
                return 0;
            case NEG:
            case TRANSPOSE:
            case RESHAPE:
            case BROADCAST:
            case RELU:
            case GELU:
            case SIGMOID:
            case SOFTMAX:
            case LAYERNORM:
            case RMSNORM:
            case DROPOUT:
            case EXP:
            case LOG:
            case SQRT:
            case RECIPROCAL:
            case REDUCE_SUM:
            case REDUCE_MAX:
            case REDUCE_MEAN:
            case TILE:
            case UNTILE:
            case IDENTITY:
            case FUSED_SOFTMAX:
            case FUSED_LAYERNORM:
            case FUSED_RMSNORM:
                return 1;
            case ADD:
            case MUL:
            case SUB:
            case DIV:
            case MATMUL:
            case FUSED_MATMUL_RELU:
            case FUSED_MATMUL_ADD:
                return 2;
            case FUSED_MATMUL_ADD_RELU:
            case FUSED_GEMM:
            case FUSED_ADD_LN:
            case SELECT:
                return 3;
            case FUSED_MHA:
                return 4;  // Q, K, V, bias
        }
        return 0;
    }

    /// Is this a leaf op (no operands)?
    [[nodiscard]] bool is_leaf() const { return arity(kind) == 0; }

    /// Is this a unary op?
    [[nodiscard]] bool is_unary() const { return arity(kind) == 1; }

    /// Is this a binary op?
    [[nodiscard]] bool is_binary() const { return arity(kind) == 2; }

    /// Is this a fused op?
    [[nodiscard]] bool is_fused() const {
        return kind >= FUSED_MATMUL_RELU && kind <= FUSED_MHA;
    }

    /// Is this a compute-heavy op (candidate for polyhedral annotation)?
    [[nodiscard]] bool is_compute_heavy() const {
        switch (kind) {
            case MATMUL:
            case FUSED_MATMUL_RELU:
            case FUSED_MATMUL_ADD:
            case FUSED_MATMUL_ADD_RELU:
            case FUSED_GEMM:
            case FUSED_MHA:
            case SOFTMAX:
            case FUSED_SOFTMAX:
            case LAYERNORM:
            case FUSED_LAYERNORM:
            case RMSNORM:
            case FUSED_RMSNORM:
            case FUSED_ADD_LN:
            case REDUCE_SUM:
            case REDUCE_MAX:
            case REDUCE_MEAN:
                return true;
            default:
                return false;
        }
    }

    /// Convert Kind to human-readable string.
    [[nodiscard]] static std::string kind_to_string(Kind k) {
        switch (k) {
            case SYMBOL:                  return "Symbol";
            case CONSTANT:                return "Const";
            case ADD:                     return "Add";
            case MUL:                     return "Mul";
            case SUB:                     return "Sub";
            case DIV:                     return "Div";
            case NEG:                     return "Neg";
            case MATMUL:                  return "MatMul";
            case TRANSPOSE:               return "Transpose";
            case RESHAPE:                 return "Reshape";
            case BROADCAST:               return "Broadcast";
            case REDUCE_SUM:              return "ReduceSum";
            case REDUCE_MAX:              return "ReduceMax";
            case REDUCE_MEAN:             return "ReduceMean";
            case RELU:                    return "ReLU";
            case GELU:                    return "GELU";
            case SIGMOID:                 return "Sigmoid";
            case SOFTMAX:                 return "Softmax";
            case LAYERNORM:               return "LayerNorm";
            case RMSNORM:                 return "RMSNorm";
            case DROPOUT:                 return "Dropout";
            case EXP:                     return "Exp";
            case LOG:                     return "Log";
            case SQRT:                    return "Sqrt";
            case RECIPROCAL:              return "Reciprocal";
            case FUSED_MATMUL_RELU:       return "FusedMatMulReLU";
            case FUSED_MATMUL_ADD:        return "FusedMatMulAdd";
            case FUSED_MATMUL_ADD_RELU:   return "FusedMatMulAddReLU";
            case FUSED_GEMM:              return "FusedGEMM";
            case FUSED_SOFTMAX:           return "FusedSoftmax";
            case FUSED_LAYERNORM:         return "FusedLayerNorm";
            case FUSED_RMSNORM:           return "FusedRMSNorm";
            case FUSED_ADD_LN:            return "FusedAddLN";
            case FUSED_MHA:               return "FusedMHA";
            case TILE:                    return "Tile";
            case UNTILE:                  return "Untile";
            case IDENTITY:                return "Identity";
            case SELECT:                  return "Select";
        }
        return "Unknown";
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "%" << id_repr() << " = " << kind_to_string(kind);

        if (!name.empty()) {
            oss << "(" << name << ")";
        } else if (kind == CONSTANT) {
            if (float_value != 0.0 && int_value == 0) {
                oss << "(" << float_value << ")";
            } else {
                oss << "(" << int_value << ")";
            }
        }

        if (axis >= 0) {
            oss << "(axis=" << axis << ")";
        }

        if (!operands.empty()) {
            oss << "(";
            for (size_t i = 0; i < operands.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "%" << operands[i];
            }
            oss << ")";
        }

        oss << " : " << shape.to_string() << ":" << irdtype_to_string(dtype);

        if (poly_annotation.has_value()) {
            oss << " " << poly_annotation->to_string();
        }

        return oss.str();
    }

private:
    // ID representation for string output (will be set by SympleXIR)
    friend class SympleXIR;
    int64_t id_ = -1;

    [[nodiscard]] int64_t id_repr() const { return id_; }
};

// ─────────────────────────────────────────────────────────────────────────
// SympleXIR: The Complete IR Module
// ─────────────────────────────────────────────────────────────────────────

/// SympleXIR: the complete IR module containing a tensor program
/// in the superoptimization search space.
///
/// This IR owns:
///   - A list of operations in SSA form (ops are numbered by insertion order)
///   - Shape and type information for each operation
///   - Polyhedral annotations for compute-heavy operations
///   - Interning for symbols and constants to avoid duplication
///
/// The IR is built top-down (add symbols first, then operations),
/// and the last operation is the root (output) of the program.
class SympleXIR {
public:
    /// Create an empty IR module.
    SympleXIR() = default;

    // ── Building ─────────────────────────────────────────────────────

    /// Add a symbol (input tensor parameter).
    /// Returns the SSA ID of the symbol.
    int64_t add_symbol(const std::string& name, const IRShape& shape,
                       IRDType dtype = IRDType::FP32);

    /// Add a floating-point constant.
    /// Interning: duplicate float constants return the same ID.
    int64_t add_constant(double value, IRDType dtype = IRDType::FP32);

    /// Add an integer constant.
    /// Interning: duplicate int constants return the same ID.
    int64_t add_int_constant(int64_t value, IRDType dtype = IRDType::INT8);

    /// Add a generic operation with explicit result shape.
    int64_t add_op(IROp::Kind kind, const std::vector<int64_t>& operands,
                   const IRShape& result_shape, IRDType dtype = IRDType::FP32);

    /// Add a unary op (shape inherited from operand).
    int64_t add_unary(IROp::Kind kind, int64_t operand);

    /// Add a binary op (shape = broadcast of operand shapes).
    int64_t add_binary(IROp::Kind kind, int64_t lhs, int64_t rhs);

    /// Add a ternary op (e.g., FUSED_GEMM, SELECT).
    int64_t add_ternary(IROp::Kind kind, int64_t a, int64_t b, int64_t c);

    /// Add a matmul op (shape = matmul_shapes).
    int64_t add_matmul(int64_t lhs, int64_t rhs);

    /// Add a reduction op along an axis.
    int64_t add_reduction(IROp::Kind kind, int64_t operand, int64_t axis);

    /// Add an axis-parameterized unary op (softmax, etc.).
    int64_t add_unary_with_axis(IROp::Kind kind, int64_t operand, int64_t axis);

    /// Attach polyhedral annotation to an operation.
    void attach_poly_annotation(int64_t op_id, IRAffineAnnotation annotation);

    // ── Query ────────────────────────────────────────────────────────

    /// Get an operation by ID.
    const IROp& op(int64_t id) const;

    /// Get a mutable reference to an operation by ID.
    IROp& op_mut(int64_t id);

    /// Number of operations in the IR.
    int64_t num_ops() const { return static_cast<int64_t>(ops_.size()); }

    /// The root ID: the last added op is the program output.
    int64_t root_id() const { return next_id_ - 1; }

    /// Get all operations.
    const std::vector<IROp>& ops() const { return ops_; }

    /// Get all input symbol names and their IDs.
    const std::unordered_map<std::string, int64_t>& symbols() const { return symbol_map_; }

    /// Check if an op ID is valid.
    bool is_valid_id(int64_t id) const {
        return id >= 0 && id < static_cast<int64_t>(ops_.size());
    }

    /// Get the shape of an op by ID.
    const IRShape& shape_of(int64_t id) const { return ops_[id].shape; }

    // ── Shape Inference ──────────────────────────────────────────────

    /// Infer all shapes bottom-up.
    void infer_shapes();

    // ── Validation ───────────────────────────────────────────────────

    /// Check the IR is well-formed (SSA, shapes consistent, no cycles).
    bool validate() const;

    // ── Serialization ────────────────────────────────────────────────

    /// Human-readable string representation.
    std::string to_string() const;

    /// Emit as JSON (for debugging, Python interop, etc.).
    std::string to_json() const;

    // ── Conversion ───────────────────────────────────────────────────

    /// Convert from the tracer's serialized TraceGraph data.
    /// params: (name, shape, dtype) — one entry per PARAM node.
    /// ops: (shape, kind_int, operands, name, dtype, axis, float_value)
    ///   kind_int is an IROp::Kind value cast to int64_t.
    ///   operands use sequential IDs (params 0..P-1, ops P..P+O-1).
    ///   axis is meaningful for reductions, softmax, etc. (-1 = unused).
    ///   float_value is meaningful for CONSTANT nodes.
    static SympleXIR from_trace_graph(
        const std::vector<std::tuple<std::string, std::vector<int64_t>, IRDType>>& params,
        const std::vector<std::tuple<IRShape, int64_t, std::vector<int64_t>,
                                    std::string, IRDType, int64_t, double>>& ops
    );

    /// Convert to an e-graph for superoptimization.
    /// Returns (egraph_ptr, root_class_id) pair.
    /// The void* points to a heap-allocated
    /// symplex::optimizer::egraph::EGraph — caller owns the memory.
    std::pair<void*, int64_t> to_egraph() const;

    /// Apply the optimized extraction result back to the IR.
    /// root_class_id identifies the e-class containing the cheapest
    /// equivalent expression for the root of the IR.
    /// cost is the estimated cost of the extracted expression.
    void apply_extraction_result(int64_t root_class_id, double cost);

    // ── Analysis ─────────────────────────────────────────────────────

    /// Collect all ops of a given kind.
    std::vector<int64_t> ops_of_kind(IROp::Kind kind) const;

    /// Collect all ops that have polyhedral annotations.
    std::vector<int64_t> ops_with_poly_annotations() const;

    /// Compute the topological order of ops (should be the insertion order for well-formed IR).
    std::vector<int64_t> topological_order() const;

    /// Count fused ops in the IR.
    size_t count_fused_ops() const;

    /// Estimate total FLOPs for the IR program.
    int64_t estimate_flops() const;

private:
    std::vector<IROp> ops_;
    int64_t next_id_ = 0;

    /// Interning: prevent duplicate constant/symbol definitions
    std::unordered_map<std::string, int64_t> symbol_map_;
    std::unordered_map<double, int64_t> float_const_map_;
    std::unordered_map<int64_t, int64_t> int_const_map_;

    // Extraction metadata (set by apply_extraction_result)
    int64_t extraction_root_class_ = -1;
    double  extraction_cost_ = 0.0;
};

} // namespace symplex::ir
