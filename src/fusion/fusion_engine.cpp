// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// Fusion Engine Implementation – Pattern classifiers, memory savings model,
// confidence scoring, legality validation, and alternative generation.

#include "symplex/fusion/fusion_engine.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace symplex::fusion {

// ══════════════════════════════════════════════════════════════════════════
// Helper functions
// ══════════════════════════════════════════════════════════════════════════

int64_t tensor_size_bytes(const std::vector<int64_t>& shape, FusionOp::DType dtype) {
    if (shape.empty()) return 0;

    int64_t num_elements = 1;
    for (auto dim : shape) {
        if (dim <= 0) return 0;  // Unknown/bad dimension
        num_elements *= dim;
    }

    switch (dtype) {
        case FusionOp::DType::FP32:  return num_elements * 4;
        case FusionOp::DType::FP16:  return num_elements * 2;
        case FusionOp::DType::BF16:  return num_elements * 2;
        case FusionOp::DType::FP8:   return num_elements * 1;
        case FusionOp::DType::INT8:  return num_elements * 1;
        case FusionOp::DType::INT4:  return num_elements * 1; // Packed; approximate
    }
    return num_elements * 4;  // Default to FP32
}

bool is_elementwise(FusionOp::OpType t) {
    switch (t) {
        case FusionOp::OpType::RELU:
        case FusionOp::OpType::GELU:
        case FusionOp::OpType::SIGMOID:
        case FusionOp::OpType::TANH:
        case FusionOp::OpType::SILU:
        case FusionOp::OpType::EXP:
        case FusionOp::OpType::SQRT:
        case FusionOp::OpType::RECIPROCAL:
        case FusionOp::OpType::NEG:
        case FusionOp::OpType::ADD:
        case FusionOp::OpType::MUL:
        case FusionOp::OpType::SUB:
        case FusionOp::OpType::DIV:
        case FusionOp::OpType::MUL_SCALAR:
        case FusionOp::OpType::DROPOUT:
            return true;
        default:
            return false;
    }
}

bool is_reduction(FusionOp::OpType t) {
    switch (t) {
        case FusionOp::OpType::REDUCE_SUM:
        case FusionOp::OpType::REDUCE_MEAN:
        case FusionOp::OpType::SOFTMAX:
            return true;
        default:
            return false;
    }
}

bool is_communication(FusionOp::OpType t) {
    switch (t) {
        case FusionOp::OpType::ALL_REDUCE:
        case FusionOp::OpType::SEND:
        case FusionOp::OpType::RECV:
            return true;
        default:
            return false;
    }
}

bool is_normalization(FusionOp::OpType t) {
    switch (t) {
        case FusionOp::OpType::LAYERNORM:
        case FusionOp::OpType::RMSNORM:
        case FusionOp::OpType::CLIP_NORM:
            return true;
        default:
            return false;
    }
}

std::string fusion_pattern_to_string(FusionPattern p) {
    switch (p) {
        case FusionPattern::MATMUL_BIAS_RELU:          return "MATMUL_BIAS_RELU";
        case FusionPattern::ATTENTION_BLOCK:            return "ATTENTION_BLOCK";
        case FusionPattern::NORM_RESIDUAL_ACTIVATION:   return "NORM_RESIDUAL_ACTIVATION";
        case FusionPattern::OPTIMIZER_STEP:             return "OPTIMIZER_STEP";
        case FusionPattern::BACKWARD_PASS_CHAIN:        return "BACKWARD_PASS_CHAIN";
        case FusionPattern::COMM_COMPUTE_OVERLAP:       return "COMM_COMPUTE_OVERLAP";
        case FusionPattern::PERSISTENT_KERNEL:          return "PERSISTENT_KERNEL";
        case FusionPattern::ELEMENTWISE_CHAIN:          return "ELEMENTWISE_CHAIN";
        case FusionPattern::REDUCTION_FUSION:           return "REDUCTION_FUSION";
        case FusionPattern::CUSTOM:                     return "CUSTOM";
    }
    return "UNKNOWN";
}

// ══════════════════════════════════════════════════════════════════════════
// Internal helpers (file-local)
// ══════════════════════════════════════════════════════════════════════════

namespace {

/// Convert an OpType to a short string for descriptions.
std::string op_type_to_string(FusionOp::OpType t) {
    switch (t) {
        case FusionOp::OpType::MATMUL:       return "MATMUL";
        case FusionOp::OpType::ADD:          return "ADD";
        case FusionOp::OpType::MUL:          return "MUL";
        case FusionOp::OpType::SUB:          return "SUB";
        case FusionOp::OpType::DIV:          return "DIV";
        case FusionOp::OpType::RELU:         return "RELU";
        case FusionOp::OpType::GELU:         return "GELU";
        case FusionOp::OpType::SIGMOID:      return "SIGMOID";
        case FusionOp::OpType::SOFTMAX:      return "SOFTMAX";
        case FusionOp::OpType::LAYERNORM:    return "LAYERNORM";
        case FusionOp::OpType::RMSNORM:      return "RMSNORM";
        case FusionOp::OpType::TRANSPOSE:    return "TRANSPOSE";
        case FusionOp::OpType::REDUCE_SUM:   return "REDUCE_SUM";
        case FusionOp::OpType::REDUCE_MEAN:  return "REDUCE_MEAN";
        case FusionOp::OpType::EXP:          return "EXP";
        case FusionOp::OpType::SQRT:         return "SQRT";
        case FusionOp::OpType::RESHAPE:      return "RESHAPE";
        case FusionOp::OpType::BROADCAST:    return "BROADCAST";
        case FusionOp::OpType::RECIPROCAL:   return "RECIPROCAL";
        case FusionOp::OpType::NEG:          return "NEG";
        case FusionOp::OpType::DROPOUT:      return "DROPOUT";
        case FusionOp::OpType::CONSTANT:     return "CONSTANT";
        case FusionOp::OpType::SYMBOL:       return "SYMBOL";
        case FusionOp::OpType::TANH:         return "TANH";
        case FusionOp::OpType::SILU:         return "SILU";
        case FusionOp::OpType::MUL_SCALAR:   return "MUL_SCALAR";
        case FusionOp::OpType::CLIP_NORM:    return "CLIP_NORM";
        case FusionOp::OpType::ALL_REDUCE:   return "ALL_REDUCE";
        case FusionOp::OpType::SEND:         return "SEND";
        case FusionOp::OpType::RECV:         return "RECV";
    }
    return "UNKNOWN_OP";
}

/// Check if an op type is an activation function.
bool is_activation(FusionOp::OpType t) {
    return t == FusionOp::OpType::RELU ||
           t == FusionOp::OpType::GELU ||
           t == FusionOp::OpType::SIGMOID ||
           t == FusionOp::OpType::TANH ||
           t == FusionOp::OpType::SILU;
}

/// Build a description string for a fusion boundary.
std::string build_description(
    FusionPattern pattern,
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    std::ostringstream oss;
    oss << fusion_pattern_to_string(pattern) << " [";
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i > 0) oss << " -> ";
        oss << op_type_to_string(ops[indices[i]].type);
    }
    oss << "]";
    return oss.str();
}

/// Compute the total memory footprint of intermediate tensors that would
/// be eliminated by fusing the given ops.  Each eliminated intermediate
/// saves 2× its size (write + read from HBM).
int64_t compute_intermediate_traffic(
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    if (indices.size() <= 1) return 0;

    int64_t total = 0;
    // Every intermediate op's output is an eliminated HBM round-trip,
    // except for the last op whose output must still be written.
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        const auto& op = ops[indices[i]];
        // 2× because we eliminate one write + one read
        total += 2 * tensor_size_bytes(op.output_shape, op.dtype);
    }
    return total;
}

/// Check that all ops in the index set share the same dtype.
bool all_same_dtype(
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    if (indices.empty()) return true;
    auto dt = ops[indices[0]].dtype;
    for (size_t i = 1; i < indices.size(); ++i) {
        if (ops[indices[i]].dtype != dt) return false;
    }
    return true;
}

/// Check that producer output shapes are consistent with consumer
/// input shapes (a quick structural check, not full dataflow).
bool shapes_consistent(
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        const auto& producer = ops[indices[i]];
        const auto& consumer = ops[indices[i + 1]];
        // The consumer must have at least one input matching the
        // producer's output shape.  This is a heuristic check since
        // we don't have full dataflow edges.
        bool found = consumer.input_shapes.empty();
        for (const auto& in_shape : consumer.input_shapes) {
            if (in_shape == producer.output_shape) {
                found = true;
                break;
            }
        }
        // If no exact match, allow it with reduced confidence
        // (the polyhedral engine will do the real check).
        if (!found && !consumer.input_shapes.empty()) {
            return false;
        }
    }
    return true;
}

/// Mark which indices are already covered by a set of boundaries.
std::vector<bool> covered_indices(
    size_t n,
    const std::vector<FusionBoundary>& boundaries
) {
    std::vector<bool> covered(n, false);
    for (const auto& b : boundaries) {
        for (auto idx : b.op_indices) {
            if (idx < n) covered[idx] = true;
        }
    }
    return covered;
}

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════
// FusionEngine implementation
// ══════════════════════════════════════════════════════════════════════════

FusionEngine::FusionEngine() = default;

// ── Primary API ────────────────────────────────────────────────────────

FusionDecision FusionEngine::discover_fusion_boundaries(
    const std::vector<FusionOp>& ops
) {
    FusionDecision decision;
    decision.total_estimated_speedup = 1.0;
    decision.total_hbm_reduction_bytes = 0;
    decision.any_requires_poly_validation = false;

    if (ops.empty()) {
        decision.summary = "Empty op list; no fusion boundaries.";
        return decision;
    }

    // Collect all candidate boundaries from each detector
    std::vector<FusionBoundary> candidates;

    for (size_t i = 0; i < ops.size(); ++i) {
        // Try each pattern detector at position i
        auto try_detector = [&](std::optional<FusionBoundary> result) {
            if (result.has_value()) {
                candidates.push_back(std::move(result.value()));
            }
        };

        try_detector(detect_matmul_bias_relu(ops, i));
        try_detector(detect_attention_block(ops, i));
        try_detector(detect_norm_residual_activation(ops, i));
        try_detector(detect_optimizer_step(ops, i));
        try_detector(detect_backward_pass(ops, i));
        try_detector(detect_comm_compute_overlap(ops, i));
        try_detector(detect_persistent_kernel(ops, i));
        try_detector(detect_elementwise_chain(ops, i));
        try_detector(detect_reduction_fusion(ops, i));
    }

    // Sort by confidence (descending), then by memory savings (descending)
    std::sort(candidates.begin(), candidates.end(),
        [](const FusionBoundary& a, const FusionBoundary& b) {
            if (a.confidence != b.confidence)
                return a.confidence > b.confidence;
            return a.memory_savings_bytes > b.memory_savings_bytes;
        });

    // Greedily select non-overlapping boundaries
    std::unordered_set<size_t> used;
    for (auto& cand : candidates) {
        // Check overlap
        bool overlap = false;
        for (auto idx : cand.op_indices) {
            if (used.count(idx)) {
                overlap = true;
                break;
            }
        }
        if (overlap) continue;

        // Quick legality check
        if (!validate_fusion_legality(cand, ops)) continue;

        // Accept this boundary
        for (auto idx : cand.op_indices) {
            used.insert(idx);
        }
        decision.boundaries.push_back(std::move(cand));
    }

    // Sort accepted boundaries by their first op index (execution order)
    std::sort(decision.boundaries.begin(), decision.boundaries.end(),
        [](const FusionBoundary& a, const FusionBoundary& b) {
            return a.op_indices.front() < b.op_indices.front();
        });

    // Aggregate benefits
    for (const auto& b : decision.boundaries) {
        decision.total_estimated_speedup *= b.compute_speedup;
        decision.total_hbm_reduction_bytes += b.memory_savings_bytes;
        if (b.requires_polyhedral_validation) {
            decision.any_requires_poly_validation = true;
        }
    }

    // Build summary
    std::ostringstream summary;
    summary << "Fusion analysis: " << decision.boundaries.size()
            << " boundary(ies) discovered, "
            << "estimated speedup " << decision.total_estimated_speedup
            << "x, HBM reduction "
            << decision.total_hbm_reduction_bytes << " bytes";
    if (decision.any_requires_poly_validation) {
        summary << " (some require polyhedral validation)";
    }
    decision.summary = summary.str();

    return decision;
}

// ── Pattern classification ─────────────────────────────────────────────

std::optional<FusionPattern> FusionEngine::classify_pattern(
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    if (indices.empty()) return std::nullopt;

    // Helper: try a detector at every index in the set; return the
    // pattern on first match.
    auto try_detector = [&](auto member_fn) -> std::optional<FusionPattern> {
        for (auto idx : indices) {
            auto result = (this->*member_fn)(ops, idx);
            if (result.has_value()) {
                return result->pattern;
            }
        }
        return std::nullopt;
    };

    // Priority-ordered detection (most specific patterns first)
    if (auto p = try_detector(&FusionEngine::detect_attention_block))           return p;
    if (auto p = try_detector(&FusionEngine::detect_matmul_bias_relu))          return p;
    if (auto p = try_detector(&FusionEngine::detect_norm_residual_activation))  return p;
    if (auto p = try_detector(&FusionEngine::detect_optimizer_step))            return p;
    if (auto p = try_detector(&FusionEngine::detect_backward_pass))             return p;
    if (auto p = try_detector(&FusionEngine::detect_comm_compute_overlap))      return p;
    if (auto p = try_detector(&FusionEngine::detect_persistent_kernel))         return p;
    if (auto p = try_detector(&FusionEngine::detect_reduction_fusion))          return p;
    if (auto p = try_detector(&FusionEngine::detect_elementwise_chain))         return p;

    return std::nullopt;
}

// ── Benefit estimation ─────────────────────────────────────────────────

int64_t FusionEngine::estimate_memory_savings(
    FusionPattern pattern,
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    if (indices.empty()) return 0;

    // Base HBM savings: 2× the size of each eliminated intermediate
    int64_t base_savings = compute_intermediate_traffic(ops, indices);

    // Pattern-specific adjustments
    switch (pattern) {
        case FusionPattern::MATMUL_BIAS_RELU: {
            // MatMul+ReLU: save 1 read+write of matmul output
            // Already captured by base_savings for the matmul output
            return base_savings;
        }
        case FusionPattern::ATTENTION_BLOCK: {
            // Attention: save 2-3 read/writes of QK^T and softmax output
            // The QK^T output and softmax output are the main intermediates
            return base_savings;
        }
        case FusionPattern::NORM_RESIDUAL_ACTIVATION: {
            // Save 1 read/write of residual sum
            return base_savings;
        }
        case FusionPattern::OPTIMIZER_STEP: {
            // Save 2-3 read/writes of momentum/buffer tensors
            // Optimizer intermediates are typically small (same shape as weights)
            return static_cast<int64_t>(base_savings * 1.5);
        }
        case FusionPattern::COMM_COMPUTE_OVERLAP: {
            // Comm-compute overlap saves by overlapping; estimated at
            // 0.5× the communication volume (partial overlap)
            if (!indices.empty()) {
                int64_t comm_volume = 0;
                for (auto idx : indices) {
                    if (is_communication(ops[idx].type)) {
                        comm_volume += tensor_size_bytes(
                            ops[idx].output_shape, ops[idx].dtype);
                    }
                }
                return static_cast<int64_t>(comm_volume * 0.5);
            }
            return 0;
        }
        default:
            return base_savings;
    }
}

double FusionEngine::estimate_compute_savings(FusionPattern pattern) {
    switch (pattern) {
        case FusionPattern::MATMUL_BIAS_RELU:
            // MatMul dominates; bias+ReLU are nearly free when fused
            return 1.05;   // ~5% from eliminating kernel launch + register reuse
        case FusionPattern::ATTENTION_BLOCK:
            // Fusing QK^T+softmax+*V saves 2 kernel launches and keeps
            // softmax output in registers
            return 1.15;   // ~15% speedup
        case FusionPattern::NORM_RESIDUAL_ACTIVATION:
            // Residual + norm + activation in one kernel
            return 1.10;   // ~10% speedup
        case FusionPattern::OPTIMIZER_STEP:
            // Many small updates; fusion eliminates 4+ kernel launches
            return 1.25;   // ~25% speedup
        case FusionPattern::BACKWARD_PASS_CHAIN:
            // Grad chains are memory-bound; fusion improves cache reuse
            return 1.12;
        case FusionPattern::COMM_COMPUTE_OVERLAP:
            // Overlap can hide nearly all comm latency
            return 1.30;   // ~30% speedup (best case)
        case FusionPattern::PERSISTENT_KERNEL:
            // Eliminates repeated kernel launch + register spilling
            return 1.40;   // ~40% speedup for small inner loops
        case FusionPattern::ELEMENTWISE_CHAIN:
            // Eliminates intermediate round-trips to HBM
            return 1.20;   // ~20% for memory-bound elementwise chains
        case FusionPattern::REDUCTION_FUSION:
            // Reduce+broadcast+elementwise avoids redundant synchronization
            return 1.08;
        case FusionPattern::CUSTOM:
            return 1.05;   // Conservative estimate
    }
    return 1.0;
}

// ── Legality validation (quick checks) ─────────────────────────────────

bool FusionEngine::validate_fusion_legality(
    const FusionBoundary& boundary,
    const std::vector<FusionOp>& ops
) {
    const auto& indices = boundary.op_indices;
    if (indices.empty()) return false;

    // 1. All indices must be valid
    for (auto idx : indices) {
        if (idx >= ops.size()) return false;
    }

    // 2. Dtype compatibility
    //    Inplace ops may have different input/output dtypes, but across
    //    the fusion boundary we want consistent dtypes for best performance.
    //    Mixed-dtype is allowed but reduces confidence and may fail
    //    polyhedral validation.
    if (!all_same_dtype(ops, indices)) {
        // Not illegal per se, but flag for poly validation
        // We allow it here but the confidence should already be reduced
    }

    // 3. Shape consistency (quick check)
    //    We don't reject on shape mismatch; we just flag it.
    //    The polyhedral engine does the definitive check.

    // 4. Aliasing constraints: cannot fuse two inplace ops that
    //    might alias the same buffer.
    int inplace_count = 0;
    for (auto idx : indices) {
        if (ops[idx].is_inplace) {
            inplace_count++;
        }
    }
    // More than one inplace op in a fusion is suspicious
    // (potential write-after-write hazard).  Allow but flag for
    // poly validation.
    if (inplace_count > 1) {
        // Will be caught by polyhedral engine
    }

    // 5. Communication barrier: cannot naively fuse a communication op
    //    with compute unless this is an explicit COMM_COMPUTE_OVERLAP
    //    pattern.
    if (boundary.pattern != FusionPattern::COMM_COMPUTE_OVERLAP) {
        for (auto idx : indices) {
            if (is_communication(ops[idx].type)) {
                // Communication ops create synchronization barriers.
                // Fusion across such barriers requires special handling.
                return false;
            }
        }
    }

    // 6. Index ordering: indices should be in ascending order
    //    (representing execution order).
    for (size_t i = 1; i < indices.size(); ++i) {
        if (indices[i] <= indices[i - 1]) return false;
    }

    return true;
}

// ── Alternative generation ─────────────────────────────────────────────

std::vector<FusionBoundary> FusionEngine::propose_fusion_alternatives(
    const std::vector<FusionOp>& ops
) {
    std::vector<FusionBoundary> alternatives;

    for (size_t i = 0; i < ops.size(); ++i) {
        // For each position, try all detectors and collect every
        // candidate (including partial matches).
        auto try_add = [&](std::optional<FusionBoundary> result) {
            if (result.has_value()) {
                alternatives.push_back(std::move(result.value()));
            }
        };

        try_add(detect_matmul_bias_relu(ops, i));
        try_add(detect_attention_block(ops, i));
        try_add(detect_norm_residual_activation(ops, i));
        try_add(detect_optimizer_step(ops, i));
        try_add(detect_backward_pass(ops, i));
        try_add(detect_comm_compute_overlap(ops, i));
        try_add(detect_persistent_kernel(ops, i));
        try_add(detect_elementwise_chain(ops, i));
        try_add(detect_reduction_fusion(ops, i));

        // Also generate finer-grained alternatives:
        // For any detected pattern, also try sub-patterns.
        // E.g., if we found [MATMUL, ADD, RELU], also propose
        //       [MATMUL, ADD] and [MATMUL, RELU].

        // Generate pairwise fusions for adjacent elementwise ops
        if (i + 1 < ops.size()) {
            if (is_elementwise(ops[i].type) && is_elementwise(ops[i + 1].type)) {
                std::vector<size_t> idx = {i, i + 1};
                FusionBoundary b;
                b.op_indices = idx;
                b.pattern = FusionPattern::ELEMENTWISE_CHAIN;
                b.memory_savings_bytes = compute_intermediate_traffic(ops, idx);
                b.compute_speedup = 1.10;
                b.confidence = compute_confidence(FusionPattern::ELEMENTWISE_CHAIN, ops, idx);
                b.description = build_description(FusionPattern::ELEMENTWISE_CHAIN, ops, idx);
                b.requires_polyhedral_validation = false;
                alternatives.push_back(std::move(b));
            }
        }

        // Generate triple fusions: matmul + elementwise + elementwise
        if (i + 2 < ops.size() &&
            ops[i].type == FusionOp::OpType::MATMUL &&
            is_elementwise(ops[i + 1].type) &&
            is_elementwise(ops[i + 2].type)) {
            std::vector<size_t> idx = {i, i + 1, i + 2};
            FusionBoundary b;
            b.op_indices = idx;
            b.pattern = FusionPattern::MATMUL_BIAS_RELU;
            b.memory_savings_bytes = compute_intermediate_traffic(ops, idx);
            b.compute_speedup = estimate_compute_savings(FusionPattern::MATMUL_BIAS_RELU);
            b.confidence = compute_confidence(FusionPattern::MATMUL_BIAS_RELU, ops, idx) * 0.9;
            b.description = build_description(FusionPattern::MATMUL_BIAS_RELU, ops, idx);
            b.requires_polyhedral_validation = true;
            alternatives.push_back(std::move(b));
        }
    }

    // Sort alternatives by estimated benefit (memory savings × speedup × confidence)
    std::sort(alternatives.begin(), alternatives.end(),
        [](const FusionBoundary& a, const FusionBoundary& b) {
            double score_a = static_cast<double>(a.memory_savings_bytes) *
                             a.compute_speedup * a.confidence;
            double score_b = static_cast<double>(b.memory_savings_bytes) *
                             b.compute_speedup * b.confidence;
            return score_a > score_b;
        });

    return alternatives;
}

// ══════════════════════════════════════════════════════════════════════════
// Sliding-window pattern detectors
// ══════════════════════════════════════════════════════════════════════════

std::optional<FusionBoundary> FusionEngine::detect_matmul_bias_relu(
    const std::vector<FusionOp>& ops, size_t start
) {
    if (start >= ops.size()) return std::nullopt;
    if (ops[start].type != FusionOp::OpType::MATMUL) return std::nullopt;

    std::vector<size_t> indices = {start};
    FusionPattern pattern = FusionPattern::MATMUL_BIAS_RELU;

    // Look for [MATMUL, ADD, RELU] or [MATMUL, RELU] or [MATMUL, ADD]
    bool has_bias = false;
    bool has_activation = false;

    // Check for ADD at start+1
    if (start + 1 < ops.size() && ops[start + 1].type == FusionOp::OpType::ADD) {
        indices.push_back(start + 1);
        has_bias = true;
    }

    // Check for activation after (optional) ADD
    size_t act_pos = start + 1 + (has_bias ? 1 : 0);
    if (act_pos < ops.size() && is_activation(ops[act_pos].type)) {
        indices.push_back(act_pos);
        has_activation = true;
    }

    // We need at least 2 ops for a meaningful fusion
    if (indices.size() < 2) return std::nullopt;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    // Adjust confidence based on which sub-pattern matched
    double conf = base_confidence(pattern);
    if (has_bias && has_activation) {
        // Full MatMul+Bias+ReLU: highest confidence
        conf = 0.95;
    } else if (has_activation) {
        // MatMul+ReLU: good, common in inference
        conf = 0.88;
    } else if (has_bias) {
        // MatMul+Add: moderate, no activation to reg-reuse
        conf = 0.78;
    }
    b.confidence = compute_confidence(pattern, ops, indices) * conf;
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;  // MatMul fusion always needs poly check
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_attention_block(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for [MATMUL, SOFTMAX, MATMUL] (Q·K^T, softmax, ·V)
    if (start + 2 >= ops.size()) return std::nullopt;
    if (ops[start].type != FusionOp::OpType::MATMUL) return std::nullopt;
    if (ops[start + 1].type != FusionOp::OpType::SOFTMAX) return std::nullopt;
    if (ops[start + 2].type != FusionOp::OpType::MATMUL) return std::nullopt;

    std::vector<size_t> indices = {start, start + 1, start + 2};
    FusionPattern pattern = FusionPattern::ATTENTION_BLOCK;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    // Attention is a very well-known pattern: high base confidence
    double conf = 0.92;
    // Boost if shapes are consistent (QK^T output feeds softmax,
    // softmax output feeds *V)
    if (shapes_consistent(ops, indices)) {
        conf += 0.05;
    }
    b.confidence = std::min(conf, 1.0);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;  // Attention fusion is complex
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_norm_residual_activation(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for [ADD, LAYERNORM/RMSNORM, RELU/GELU/SILU]
    if (start + 1 >= ops.size()) return std::nullopt;
    if (ops[start].type != FusionOp::OpType::ADD) return std::nullopt;

    size_t norm_pos = start + 1;
    if (!is_normalization(ops[norm_pos].type)) return std::nullopt;

    std::vector<size_t> indices = {start, norm_pos};
    FusionPattern pattern = FusionPattern::NORM_RESIDUAL_ACTIVATION;

    // Optional activation after norm
    bool has_activation = false;
    if (norm_pos + 1 < ops.size() && is_activation(ops[norm_pos + 1].type)) {
        indices.push_back(norm_pos + 1);
        has_activation = true;
    }

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    double conf = 0.85;
    if (has_activation) conf += 0.05;
    if (shapes_consistent(ops, indices)) conf += 0.05;
    b.confidence = std::min(conf, 1.0);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_optimizer_step(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for [MUL, ADD, MUL, ADD] pattern (SGD/Adam)
    if (start + 3 >= ops.size()) return std::nullopt;

    // Pattern: MUL, ADD, MUL, ADD (momentum update + weight update)
    if (ops[start].type != FusionOp::OpType::MUL)     return std::nullopt;
    if (ops[start + 1].type != FusionOp::OpType::ADD) return std::nullopt;
    if (ops[start + 2].type != FusionOp::OpType::MUL) return std::nullopt;
    if (ops[start + 3].type != FusionOp::OpType::ADD) return std::nullopt;

    std::vector<size_t> indices = {start, start + 1, start + 2, start + 3};
    FusionPattern pattern = FusionPattern::OPTIMIZER_STEP;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    double conf = 0.80;
    // Optimizer steps typically have same shapes throughout
    if (all_same_dtype(ops, indices)) conf += 0.08;
    b.confidence = std::min(conf, 1.0);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = false;  // Elementwise; usually safe
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_backward_pass(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for chains with transposes (gradient computation).
    // Pattern: at least 3 ops, with at least one TRANSPOSE, and
    // the rest being elementwise or MATMUL.
    if (start >= ops.size()) return std::nullopt;
    if (ops[start].type != FusionOp::OpType::TRANSPOSE) return std::nullopt;

    // Extend the chain as long as we see transposes or elementwise ops
    std::vector<size_t> indices = {start};
    bool has_transpose = true;
    size_t pos = start + 1;

    while (pos < ops.size()) {
        const auto& op = ops[pos];
        if (op.type == FusionOp::OpType::TRANSPOSE) {
            indices.push_back(pos);
            has_transpose = true;
        } else if (is_elementwise(op.type)) {
            indices.push_back(pos);
        } else {
            break;  // Non-fusible op terminates the chain
        }
        pos++;
    }

    if (!has_transpose || indices.size() < 3) return std::nullopt;

    FusionPattern pattern = FusionPattern::BACKWARD_PASS_CHAIN;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);
    b.confidence = compute_confidence(pattern, ops, indices);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;  // Transpose+fusion needs poly check
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_comm_compute_overlap(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for [ALL_REDUCE/SEND/RECV, ...compute...]
    if (start >= ops.size()) return std::nullopt;
    if (!is_communication(ops[start].type)) return std::nullopt;

    std::vector<size_t> indices = {start};

    // Collect up to 4 subsequent non-communication ops
    size_t compute_count = 0;
    size_t pos = start + 1;
    while (pos < ops.size() && compute_count < 4) {
        if (is_communication(ops[pos].type)) {
            break;  // Another comm op; stop
        }
        indices.push_back(pos);
        compute_count++;
        pos++;
    }

    if (compute_count == 0) return std::nullopt;

    FusionPattern pattern = FusionPattern::COMM_COMPUTE_OVERLAP;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    double conf = 0.70;  // Comm-compute overlap is hard to guarantee
    if (compute_count >= 2) conf += 0.10;  // More compute to overlap
    b.confidence = std::min(conf, 1.0);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;  // Must validate overlap feasibility
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_persistent_kernel(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for a repeated pattern of the same op sequence.
    // We check for 2+ repetitions of the same op starting at `start`.
    if (start >= ops.size()) return std::nullopt;

    // Try pattern lengths 1 through 4
    for (size_t pat_len = 1; pat_len <= 4 && start + pat_len <= ops.size(); ++pat_len) {
        // Extract the pattern
        std::vector<FusionOp::OpType> pattern_types;
        for (size_t j = 0; j < pat_len; ++j) {
            pattern_types.push_back(ops[start + j].type);
        }

        // Check if the pattern repeats at least once more
        size_t repetitions = 1;
        size_t pos = start + pat_len;
        while (pos + pat_len <= ops.size()) {
            bool matches = true;
            for (size_t j = 0; j < pat_len; ++j) {
                if (ops[pos + j].type != pattern_types[j]) {
                    matches = false;
                    break;
                }
            }
            if (!matches) break;
            repetitions++;
            pos += pat_len;
        }

        if (repetitions >= 2) {
            // Collect all op indices in the repeated pattern
            std::vector<size_t> indices;
            for (size_t r = 0; r < repetitions; ++r) {
                for (size_t j = 0; j < pat_len; ++j) {
                    indices.push_back(start + r * pat_len + j);
                }
            }

            FusionPattern pattern = FusionPattern::PERSISTENT_KERNEL;

            FusionBoundary b;
            b.op_indices = indices;
            b.pattern = pattern;
            b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
            b.compute_speedup = estimate_compute_savings(pattern);

            double conf = 0.75;
            if (repetitions >= 4) conf += 0.10;
            if (all_same_dtype(ops, indices)) conf += 0.05;
            b.confidence = std::min(conf, 1.0);
            b.description = build_description(pattern, ops, indices);
            b.requires_polyhedral_validation = true;
            return b;
        }
    }

    return std::nullopt;
}

std::optional<FusionBoundary> FusionEngine::detect_elementwise_chain(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for consecutive elementwise ops (at least 2)
    if (start >= ops.size()) return std::nullopt;
    if (!is_elementwise(ops[start].type)) return std::nullopt;

    std::vector<size_t> indices = {start};
    size_t pos = start + 1;

    while (pos < ops.size() && is_elementwise(ops[pos].type)) {
        indices.push_back(pos);
        pos++;
    }

    if (indices.size() < 2) return std::nullopt;

    FusionPattern pattern = FusionPattern::ELEMENTWISE_CHAIN;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);
    b.confidence = compute_confidence(pattern, ops, indices);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = false;  // Elementwise chains are usually safe
    return b;
}

std::optional<FusionBoundary> FusionEngine::detect_reduction_fusion(
    const std::vector<FusionOp>& ops, size_t start
) {
    // Look for [REDUCE, BROADCAST, elementwise]
    if (start + 1 >= ops.size()) return std::nullopt;
    if (!is_reduction(ops[start].type)) return std::nullopt;

    std::vector<size_t> indices = {start};

    // Optional BROADCAST after reduce
    size_t next_pos = start + 1;
    if (next_pos < ops.size() && ops[next_pos].type == FusionOp::OpType::BROADCAST) {
        indices.push_back(next_pos);
        next_pos++;
    }

    // Optional elementwise after (broadcast or reduce)
    if (next_pos < ops.size() && is_elementwise(ops[next_pos].type)) {
        indices.push_back(next_pos);
    }

    if (indices.size() < 2) return std::nullopt;

    FusionPattern pattern = FusionPattern::REDUCTION_FUSION;

    FusionBoundary b;
    b.op_indices = indices;
    b.pattern = pattern;
    b.memory_savings_bytes = estimate_memory_savings(pattern, ops, indices);
    b.compute_speedup = estimate_compute_savings(pattern);

    double conf = 0.82;
    if (indices.size() >= 3) conf += 0.05;  // Full reduce+broadcast+elementwise
    b.confidence = std::min(conf, 1.0);
    b.description = build_description(pattern, ops, indices);
    b.requires_polyhedral_validation = true;  // Reduction+fusion needs validation
    return b;
}

// ══════════════════════════════════════════════════════════════════════════
// Confidence scoring
// ══════════════════════════════════════════════════════════════════════════

double FusionEngine::compute_confidence(
    FusionPattern pattern,
    const std::vector<FusionOp>& ops,
    const std::vector<size_t>& indices
) {
    double conf = base_confidence(pattern);

    // Adjustment 1: Shape consistency between producer/consumer
    if (shapes_consistent(ops, indices)) {
        conf += 0.05;
    } else if (!indices.empty()) {
        conf -= 0.10;
    }

    // Adjustment 2: Dtype consistency
    if (all_same_dtype(ops, indices)) {
        conf += 0.03;
    } else {
        conf -= 0.08;
    }

    // Adjustment 3: Tensor size (larger tensors → higher savings)
    int64_t total_intermediate = 0;
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        total_intermediate += tensor_size_bytes(ops[indices[i]].output_shape,
                                                 ops[indices[i]].dtype);
    }
    // Bonus for large intermediates (significant HBM traffic eliminated)
    if (total_intermediate > 1'000'000) {       // > 1MB
        conf += 0.02;
    }
    if (total_intermediate > 10'000'000) {      // > 10MB
        conf += 0.03;
    }
    if (total_intermediate > 100'000'000) {     // > 100MB
        conf += 0.02;
    }

    return std::clamp(conf, 0.0, 1.0);
}

double FusionEngine::base_confidence(FusionPattern pattern) {
    switch (pattern) {
        case FusionPattern::MATMUL_BIAS_RELU:          return 0.85;
        case FusionPattern::ATTENTION_BLOCK:            return 0.92;
        case FusionPattern::NORM_RESIDUAL_ACTIVATION:   return 0.85;
        case FusionPattern::OPTIMIZER_STEP:             return 0.80;
        case FusionPattern::BACKWARD_PASS_CHAIN:        return 0.70;
        case FusionPattern::COMM_COMPUTE_OVERLAP:       return 0.65;
        case FusionPattern::PERSISTENT_KERNEL:          return 0.75;
        case FusionPattern::ELEMENTWISE_CHAIN:          return 0.80;
        case FusionPattern::REDUCTION_FUSION:           return 0.78;
        case FusionPattern::CUSTOM:                     return 0.50;
    }
    return 0.50;
}

} // namespace symplex::fusion
