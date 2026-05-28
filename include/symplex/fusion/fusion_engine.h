// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// Fusion Engine – Semantic fusion boundary discovery and pattern classification.
//
// Architectural principle:
//   FUSION DECIDES WHAT,  POLYHEDRAL DECIDES WHETHER AND HOW.
//
// The FusionEngine discovers *candidate* fusion boundaries by matching
// high-level semantic patterns (e.g. MatMul+Bias+ReLU, Attention blocks,
// Norm+Residual+Activation).  Its decisions are expressed as FusionBoundary
// objects that record which ops to fuse, the identified pattern, and
// estimated benefits (HBM traffic reduction, compute speedup, confidence).
//
// The polyhedral engine then validates legality (preserving dependencies,
// respecting memory hierarchy constraints) and decides the *how* (tiling,
// scheduling, register allocation).  Finally, MCMC search maps the fused
// sub-graph onto hardware.
//
// Pipeline:  FusionEngine → PolyhedralOptimizer → MCMC

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace symplex::fusion {

// ──────────────────────────────────────────────────────────────────────────
// FusionPattern – enumerated semantic fusion patterns
// ──────────────────────────────────────────────────────────────────────────

/// Well-known fusion patterns that appear repeatedly in ML workloads.
/// Each pattern carries a known benefit profile and legality template
/// that guides the polyhedral validation stage.
enum class FusionPattern {
    MATMUL_BIAS_RELU,              ///< GEMM + bias add + activation
    ATTENTION_BLOCK,               ///< Q·K^T → softmax → ·V
    NORM_RESIDUAL_ACTIVATION,      ///< residual add + layer-norm + activation
    OPTIMIZER_STEP,                ///< SGD/Adam momentum + weight update
    BACKWARD_PASS_CHAIN,           ///< Transpose-based grad chains
    COMM_COMPUTE_OVERLAP,          ///< Overlap all-reduce with compute
    PERSISTENT_KERNEL,             ///< Looped pattern → persistent kernel
    ELEMENTWISE_CHAIN,             ///< Consecutive pointwise ops
    REDUCTION_FUSION,              ///< Reduce → broadcast → elementwise
    CUSTOM                         ///< User-defined or unrecognized pattern
};

/// Convert a FusionPattern to its string representation.
std::string fusion_pattern_to_string(FusionPattern p);

// ──────────────────────────────────────────────────────────────────────────
// FusionOp – lightweight operator representation for fusion analysis
// ──────────────────────────────────────────────────────────────────────────

/// A minimal operator descriptor sufficient for fusion pattern matching,
/// memory estimation, and legality checks.  This is *not* a full IR node;
/// it is intentionally lightweight so that the fusion engine can run
/// independently of the rest of the compiler.
struct FusionOp {
    // ── Operator type enumeration ──────────────────────────────────────
    enum class OpType {
        MATMUL,
        ADD,
        MUL,
        SUB,
        DIV,
        RELU,
        GELU,
        SIGMOID,
        SOFTMAX,
        LAYERNORM,
        RMSNORM,
        TRANSPOSE,
        REDUCE_SUM,
        REDUCE_MEAN,
        EXP,
        SQRT,
        RESHAPE,
        BROADCAST,
        RECIPROCAL,
        NEG,
        DROPOUT,
        CONSTANT,
        SYMBOL,         ///< Symbolic / placeholder op
        TANH,
        SILU,
        MUL_SCALAR,
        CLIP_NORM,
        ALL_REDUCE,
        SEND,
        RECV
    };

    // ── Data type enumeration ──────────────────────────────────────────
    enum class DType {
        FP32,
        FP16,
        BF16,
        FP8,
        INT8,
        INT4
    };

    OpType                              type;
    std::vector<int64_t>                output_shape;   ///< Shape of the output tensor
    std::vector<std::vector<int64_t>>   input_shapes;   ///< Shapes of all input tensors
    DType                               dtype;          ///< Element data type
    bool                                is_inplace;     ///< Whether op may alias its input
    int64_t                             memory_bytes;   ///< Total memory footprint (output + inputs)
};

// ──────────────────────────────────────────────────────────────────────────
// FusionBoundary – a candidate fusion region
// ──────────────────────────────────────────────────────────────────────────

/// Represents a set of consecutive operators that the fusion engine
/// recommends fusing into a single kernel.  Each boundary carries
/// estimated benefits and a flag indicating whether polyhedral
/// validation is required before the fusion can be committed.
struct FusionBoundary {
    std::vector<size_t>  op_indices;                   ///< Indices of ops to fuse (into the input op list)
    FusionPattern        pattern;                      ///< Recognized semantic pattern
    int64_t              memory_savings_bytes;          ///< HBM traffic eliminated (read+write)
    double               compute_speedup;              ///< Estimated compute speedup factor
    double               confidence;                   ///< Confidence score [0.0, 1.0]
    std::string          description;                  ///< Human-readable fusion description
    bool                 requires_polyhedral_validation;///< Whether poly engine must validate
};

// ──────────────────────────────────────────────────────────────────────────
// FusionDecision – the complete output of fusion analysis
// ──────────────────────────────────────────────────────────────────────────

/// The full set of fusion recommendations for a sub-graph of operators.
/// The polyhedral engine consumes this and validates/transforms each
/// FusionBoundary independently.
struct FusionDecision {
    std::vector<FusionBoundary> boundaries;             ///< All candidate fusion regions
    double                      total_estimated_speedup;///< Aggregate speedup estimate
    int64_t                     total_hbm_reduction_bytes;///< Aggregate HBM traffic reduction
    bool                        any_requires_poly_validation;///< Any boundary needs poly check
    std::string                 summary;               ///< Brief human-readable summary
};

// ──────────────────────────────────────────────────────────────────────────
// FusionEngine – main class
// ──────────────────────────────────────────────────────────────────────────

/// The FusionEngine discovers fusion boundaries in a flat list of ops.
///
/// Usage:
///   FusionEngine engine;
///   FusionDecision decision = engine.discover_fusion_boundaries(ops);
///   for (const auto& boundary : decision.boundaries) {
///       if (boundary.requires_polyhedral_validation) {
///           // defer to polyhedral engine
///       }
///   }
///
/// Thread safety: FusionEngine is NOT thread-safe.  Use one instance
/// per compilation thread.
class FusionEngine {
public:
    FusionEngine();

    // ── Primary API ────────────────────────────────────────────────────

    /// Discover all candidate fusion boundaries in the given op sequence.
    ///
    /// This is the main entry point.  It runs every pattern detector in
    /// a sliding-window fashion over the op list, collects non-overlapping
    /// boundaries (preferring higher-confidence matches), and aggregates
    /// their estimated benefits.
    ///
    /// \param ops  Flat list of operators in execution order
    /// \return     A FusionDecision with all recommended boundaries
    FusionDecision discover_fusion_boundaries(const std::vector<FusionOp>& ops);

    // ── Pattern classification ─────────────────────────────────────────

    /// Attempt to classify a sub-sequence of ops as a known fusion pattern.
    ///
    /// \param ops      Full op list
    /// \param indices  Subset of op indices to classify
    /// \return         The recognized FusionPattern, or std::nullopt
    std::optional<FusionPattern> classify_pattern(
        const std::vector<FusionOp>& ops,
        const std::vector<size_t>& indices
    );

    // ── Benefit estimation ─────────────────────────────────────────────

    /// Estimate HBM traffic savings for a fusion pattern.
    ///
    /// For each eliminated intermediate tensor, the savings are 2× its
    /// size (one write to HBM eliminated + one read from HBM eliminated).
    ///
    /// \param pattern  The fusion pattern
    /// \param ops      Full op list
    /// \param indices  Subset of op indices in the fusion
    /// \return         Estimated bytes of HBM traffic eliminated
    int64_t estimate_memory_savings(
        FusionPattern pattern,
        const std::vector<FusionOp>& ops,
        const std::vector<size_t>& indices
    );

    /// Estimate compute speedup factor for a fusion pattern.
    ///
    /// The speedup comes from eliminating kernel launch overhead and
    /// improving register/cache reuse within the fused kernel.
    ///
    /// \param pattern  The fusion pattern
    /// \return         Estimated speedup multiplier (1.0 = no benefit)
    double estimate_compute_savings(FusionPattern pattern);

    // ── Legality validation (quick checks) ─────────────────────────────

    /// Perform quick legality checks on a proposed fusion boundary.
    ///
    /// Checks include:
    ///   - Dtype compatibility across fused ops
    ///   - Shape consistency (producer output == consumer input)
    ///   - No aliasing conflicts between inplace ops
    ///   - Communication barriers (cannot fuse across all-reduce boundaries
    ///     without comm-compute overlap)
    ///
    /// Full legality (dependency preservation, memory hierarchy fit) is
    /// deferred to the polyhedral engine.
    ///
    /// \param boundary  The proposed fusion boundary
    /// \param ops       Full op list
    /// \return          true if quick checks pass
    bool validate_fusion_legality(
        const FusionBoundary& boundary,
        const std::vector<FusionOp>& ops
    );

    // ── Alternative generation ─────────────────────────────────────────

    /// Generate multiple fusion alternatives for the given op sequence.
    ///
    /// For each group of ops, produces several FusionBoundary proposals
    /// with different granularities (e.g. fuse just MatMul+ReLU vs.
    /// MatMul+Add+ReLU).  This gives the downstream cost model more
    /// choices.
    ///
    /// \param ops  Full op list
    /// \return     Vector of alternative FusionBoundary proposals
    std::vector<FusionBoundary> propose_fusion_alternatives(
        const std::vector<FusionOp>& ops
    );

private:
    // ── Sliding-window pattern detectors ───────────────────────────────

    /// Detect [MATMUL, ADD, RELU] or [MATMUL, RELU] or [MATMUL, ADD].
    std::optional<FusionBoundary> detect_matmul_bias_relu(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect [MATMUL, SOFTMAX, MATMUL] (Q·K^T, softmax, ·V).
    std::optional<FusionBoundary> detect_attention_block(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect [ADD, LAYERNORM/RMSNORM, RELU/GELU/SILU].
    std::optional<FusionBoundary> detect_norm_residual_activation(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect [MUL, ADD, MUL, ADD] (SGD/Adam optimizer step).
    std::optional<FusionBoundary> detect_optimizer_step(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect chains with transposes (backward-pass gradient computation).
    std::optional<FusionBoundary> detect_backward_pass(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect [ALL_REDUCE/SEND/RECV, ...compute...] for comm-compute overlap.
    std::optional<FusionBoundary> detect_comm_compute_overlap(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect loops of same op pattern → persistent kernel opportunity.
    std::optional<FusionBoundary> detect_persistent_kernel(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect consecutive elementwise ops.
    std::optional<FusionBoundary> detect_elementwise_chain(
        const std::vector<FusionOp>& ops, size_t start
    );

    /// Detect [REDUCE, BROADCAST, elementwise].
    std::optional<FusionBoundary> detect_reduction_fusion(
        const std::vector<FusionOp>& ops, size_t start
    );

    // ── Confidence scoring ─────────────────────────────────────────────

    /// Compute a confidence score for a fusion boundary.
    ///
    /// Base confidence per pattern type, adjusted for:
    ///   - Shape consistency between producer/consumer
    ///   - Dtype consistency
    ///   - Tensor size (larger tensors → higher savings → more confident)
    double compute_confidence(
        FusionPattern pattern,
        const std::vector<FusionOp>& ops,
        const std::vector<size_t>& indices
    );

    /// Base confidence for each pattern type (table lookup).
    static double base_confidence(FusionPattern pattern);
};

// ──────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────

/// Compute the byte size of a tensor given its shape and element dtype.
int64_t tensor_size_bytes(const std::vector<int64_t>& shape, FusionOp::DType dtype);

/// True if the op type is elementwise (applies a scalar function independently
/// to each element: RELU, GELU, SIGMOID, TANH, SILU, EXP, SQRT, RECIPROCAL,
/// NEG, ADD, MUL, SUB, DIV, MUL_SCALAR, DROPOUT).
bool is_elementwise(FusionOp::OpType t);

/// True if the op type is a reduction (REDUCE_SUM, REDUCE_MEAN, SOFTMAX).
bool is_reduction(FusionOp::OpType t);

/// True if the op type is a collective communication op
/// (ALL_REDUCE, SEND, RECV).
bool is_communication(FusionOp::OpType t);

/// True if the op type is a normalization (LAYERNORM, RMSNORM, CLIP_NORM).
bool is_normalization(FusionOp::OpType t);

} // namespace symplex::fusion
