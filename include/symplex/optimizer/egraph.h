// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <variant>
#include <memory>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <deque>
#include <queue>
#include <random>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

namespace symplex::optimizer::egraph {

// ─────────────────────────────────────────────────────────────────────────
// Tensor Expression Language
// ─────────────────────────────────────────────────────────────────────────

/// OpId: identifies the type of operation in a tensor expression.
enum class OpId : uint8_t {
    // Leaf / input nodes
    SYMBOL,         // Named tensor variable (e.g. "A", "W_q")
    CONSTANT,      // Numeric constant (0, 1, etc.)

    // Arithmetic
    ADD,            // Element-wise addition
    MUL,           // Element-wise multiplication (Hadamard)
    SUB,           // Element-wise subtraction
    DIV,           // Element-wise division
    NEG,           // Negation

    // Linear algebra
    MATMUL,        // Matrix multiplication C = A @ B
    TRANSPOSE,     // Matrix transpose
    RESHAPE,       // Reshape tensor dimensions
    BROADCAST,     // Broadcast a smaller tensor

    // Reductions
    REDUCE_SUM,    // Sum reduction along an axis
    REDUCE_MAX,    // Max reduction along an axis
    REDUCE_MEAN,   // Mean reduction along an axis

    // Neural network primitives
    RELU,          // ReLU activation
    GELU,          // GELU activation
    SIGMOID,       // Sigmoid activation
    SOFTMAX,       // Softmax along an axis
    LAYERNORM,     // Layer normalization
    RMSNORM,       // RMS normalization (LLaMA-style)
    DROPOUT,       // Dropout (identity during inference)
    EXP,           // Exponential (for softmax decomposition)
    LOG,           // Natural logarithm (for log-softmax)
    SQRT,          // Square root (for variance normalization)
    RECIPROCAL,    // 1/x (for division-free normalization)

    // Fused operations (these are the targets superoptimizer discovers)
    FUSED_MATMUL_RELU,       // MatMul followed by ReLU
    FUSED_MATMUL_ADD,        // MatMul followed by bias add
    FUSED_MATMUL_ADD_RELU,   // MatMul + bias add + ReLU
    FUSED_GEMM,              // alpha*A@B + beta*C
    FUSED_SOFTMAX,           // Numerically stable fused softmax
    FUSED_LAYERNORM,         // Fused layernorm kernel
    FUSED_RMSNORM,           // Fused rmsnorm kernel
    FUSED_ADD_LN,            // Fused residual-add + layernorm
    FUSED_MHA,               // Fused multi-head attention

    // Tiling / sharding meta-ops
    TILE,           // Tile (split) a computation along a dimension
    UNTILE,         // Merge tiled computations back

    // Data movement
    IDENTITY,       // No-op pass-through
};

/// Hash specialization for OpId (needed for std::unordered_set<OpId> etc.)
struct OpIdHash {
    size_t operator()(OpId op) const noexcept {
        return std::hash<uint8_t>{}(static_cast<uint8_t>(op));
    }
};

inline std::string op_to_string(OpId op) {
    switch (op) {
        case OpId::SYMBOL:              return "Symbol";
        case OpId::CONSTANT:            return "Const";
        case OpId::ADD:                 return "Add";
        case OpId::MUL:                 return "Mul";
        case OpId::SUB:                 return "Sub";
        case OpId::DIV:                 return "Div";
        case OpId::NEG:                 return "Neg";
        case OpId::MATMUL:              return "MatMul";
        case OpId::TRANSPOSE:           return "Transpose";
        case OpId::RESHAPE:             return "Reshape";
        case OpId::BROADCAST:           return "Broadcast";
        case OpId::REDUCE_SUM:          return "ReduceSum";
        case OpId::REDUCE_MAX:          return "ReduceMax";
        case OpId::REDUCE_MEAN:         return "ReduceMean";
        case OpId::RELU:                return "ReLU";
        case OpId::GELU:                return "GELU";
        case OpId::SIGMOID:             return "Sigmoid";
        case OpId::SOFTMAX:             return "Softmax";
        case OpId::LAYERNORM:           return "LayerNorm";
        case OpId::RMSNORM:             return "RMSNorm";
        case OpId::DROPOUT:             return "Dropout";
        case OpId::EXP:                 return "Exp";
        case OpId::LOG:                 return "Log";
        case OpId::SQRT:                return "Sqrt";
        case OpId::RECIPROCAL:          return "Reciprocal";
        case OpId::FUSED_MATMUL_RELU:   return "FusedMatMulReLU";
        case OpId::FUSED_MATMUL_ADD:    return "FusedMatMulAdd";
        case OpId::FUSED_MATMUL_ADD_RELU: return "FusedMatMulAddReLU";
        case OpId::FUSED_GEMM:          return "FusedGEMM";
        case OpId::FUSED_SOFTMAX:       return "FusedSoftmax";
        case OpId::FUSED_LAYERNORM:     return "FusedLayerNorm";
        case OpId::FUSED_RMSNORM:       return "FusedRMSNorm";
        case OpId::FUSED_ADD_LN:        return "FusedAddLN";
        case OpId::FUSED_MHA:           return "FusedMHA";
        case OpId::TILE:                return "Tile";
        case OpId::UNTILE:              return "Untile";
        case OpId::IDENTITY:            return "Identity";
    }
    return "Unknown";
}

/// How many children does each op expect?
inline int op_arity(OpId op) {
    switch (op) {
        case OpId::SYMBOL:
        case OpId::CONSTANT:
            return 0;
        case OpId::NEG:
        case OpId::TRANSPOSE:
        case OpId::RESHAPE:
        case OpId::BROADCAST:
        case OpId::RELU:
        case OpId::GELU:
        case OpId::SIGMOID:
        case OpId::SOFTMAX:
        case OpId::LAYERNORM:
        case OpId::RMSNORM:
        case OpId::DROPOUT:
        case OpId::TILE:
        case OpId::UNTILE:
        case OpId::IDENTITY:
        case OpId::REDUCE_SUM:
        case OpId::REDUCE_MAX:
        case OpId::REDUCE_MEAN:
        case OpId::EXP:
        case OpId::LOG:
        case OpId::SQRT:
        case OpId::RECIPROCAL:
        case OpId::FUSED_SOFTMAX:
        case OpId::FUSED_LAYERNORM:
        case OpId::FUSED_RMSNORM:
            return 1;
        case OpId::ADD:
        case OpId::MUL:
        case OpId::SUB:
        case OpId::DIV:
        case OpId::MATMUL:
        case OpId::FUSED_MATMUL_RELU:
        case OpId::FUSED_MATMUL_ADD:
            return 2;
        case OpId::FUSED_MATMUL_ADD_RELU:
        case OpId::FUSED_GEMM:
        case OpId::FUSED_ADD_LN:
            return 3;
        case OpId::FUSED_MHA:
            return 4;  // Q, K, V, bias
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────
// Tensor Shape / Type Analysis Data
// ─────────────────────────────────────────────────────────────────────────

/// Data type enumeration for tensor elements.
enum class DType : uint8_t {
    FP64, FP32, FP16, BF16, INT8, INT4, UNKNOWN
};

inline std::string dtype_to_string(DType dt) {
    switch (dt) {
        case DType::FP64:    return "fp64";
        case DType::FP32:    return "fp32";
        case DType::FP16:    return "fp16";
        case DType::BF16:    return "bf16";
        case DType::INT8:    return "int8";
        case DType::INT4:    return "int4";
        case DType::UNKNOWN: return "?";
    }
    return "?";
}

/// Memory layout of a tensor.
enum class Layout : uint8_t {
    ROW_MAJOR,       // C-contiguous
    COL_MAJOR,       // Fortran-contiguous
    TILED_128,       // Swizzled for 128-byte transaction alignment
    TILED_32,        // MMA fragment layout (m16n8k16 etc.)
    INTERLEAVED,     // Bank-conflict-free interleaved
    UNKNOWN
};

/// Tensor shape: vector of dimension sizes. -1 means symbolic (unknown).
struct TensorShape {
    std::vector<int64_t> dims;

    TensorShape() = default;
    TensorShape(std::initializer_list<int64_t> d) : dims(d) {}
    explicit TensorShape(std::vector<int64_t> d) : dims(std::move(d)) {}

    size_t ndim() const { return dims.size(); }
    int64_t operator[](size_t i) const { return i < dims.size() ? dims[i] : -1; }

    /// Total number of elements. Returns -1 if any dimension is symbolic.
    int64_t num_elements() const {
        int64_t total = 1;
        for (auto d : dims) {
            if (d < 0) return -1;
            total *= d;
        }
        return total;
    }

    bool is_unknown() const { return dims.empty(); }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << ",";
            if (dims[i] < 0) oss << "?";
            else oss << dims[i];
        }
        oss << "]";
        return oss.str();
    }

    bool operator==(const TensorShape& o) const { return dims == o.dims; }
    bool operator!=(const TensorShape& o) const { return dims != o.dims; }
};

/// Memory residency hint for an e-class.
enum class Residency : uint8_t {
    HBM,       // Global memory (high latency, high capacity)
    SRAM,      // Shared memory (low latency, low capacity)
    REGISTER,  // Thread-private registers
    UNKNOWN
};

/// Sharding metadata for distributed tensor analysis.
struct ShardingInfo {
    /// Which device mesh axis each tensor dimension is sharded over.
    /// -1 means replicated on that dimension.
    std::vector<int> shard_axes;
    int mesh_ndim = 0;  // Number of mesh dimensions

    bool is_replicated() const {
        for (auto a : shard_axes) if (a >= 0) return false;
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "shard[";
        for (size_t i = 0; i < shard_axes.size(); ++i) {
            if (i > 0) oss << ",";
            oss << (shard_axes[i] >= 0 ? std::to_string(shard_axes[i]) : "R");
        }
        oss << "]";
        return oss.str();
    }
};

/// Per-class analysis data attached to each e-class.
/// This enables type-safe, shape-aware, hardware-aware extraction.
struct ClassAnalysis {
    TensorShape  shape;              // Inferred tensor shape
    DType        dtype = DType::UNKNOWN;  // Element data type
    Layout       layout = Layout::UNKNOWN; // Memory layout
    Residency    residency = Residency::UNKNOWN; // Where this tensor lives
    ShardingInfo sharding;           // Distributed sharding info
    int64_t      estimated_flops = 0;   // Estimated FLOPs to compute this
    bool         tc_compatible = false;  // Can map to Tensor Core MMA?
    bool         is_reduction = false;   // Does this expression contain reductions?
    bool         aliases_input = false;  // Is this an alias (view) of an input?
    bool         floating_point_safe = true; // Can floating-point rewrites apply?

    /// Bytes per element for this dtype.
    int64_t bytes_per_element() const {
        switch (dtype) {
            case DType::FP64: return 8;
            case DType::FP32: return 4;
            case DType::FP16: return 2;
            case DType::BF16: return 2;
            case DType::INT8: return 1;
            case DType::INT4: return 1;  // Packed
            case DType::UNKNOWN: return 2;  // Conservative default
        }
        return 2;
    }

    /// Estimated memory traffic in bytes.
    int64_t estimated_bytes() const {
        int64_t n = shape.num_elements();
        if (n < 0) return 0;
        return n * bytes_per_element();
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << shape.to_string() << ":" << dtype_to_string(dtype);
        if (tc_compatible) oss << "+TC";
        if (is_reduction) oss << "+red";
        if (!floating_point_safe) oss << "+fp-unsafe";
        return oss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────
// E-Graph Core Data Structures
// ─────────────────────────────────────────────────────────────────────────

/// ClassId: identifies an equivalence class in the e-graph.
/// All e-nodes in the same class represent equivalent expressions.
using ClassId = int64_t;

/// NodeId: identifies a specific e-node within the e-graph.
using NodeId = int64_t;

/// An invalid/null identifier.
constexpr ClassId NULL_CLASS = -1;
constexpr NodeId NULL_NODE  = -1;

/// ENode: an expression node in the e-graph.
/// This is the atomic unit — an operator applied to a list of child classes.
struct ENode {
    OpId                 op;            // The operation
    std::vector<ClassId> children;      // Children are class IDs, not node IDs
    int64_t              value = 0;     // For CONSTANT nodes (integer)
    double               float_value = 0.0; // For CONSTANT nodes (floating-point)
    std::string          name;          // For SYMBOL nodes
    int64_t              axis  = -1;    // For REDUCE_*, TRANSPOSE, TILE, SOFTMAX, etc.
    int64_t              dim0  = -1;    // Shape info for RESHAPE/TILE
    int64_t              dim1  = -1;
    int64_t              dim2  = -1;
    DType                dtype = DType::UNKNOWN; // Typed constant dtype

    /// Equality comparison for congruence checking.
    /// CRITICAL: Must include ALL fields that semantically differentiate nodes.
    bool operator==(const ENode& other) const {
        if (op != other.op) return false;
        if (children != other.children) return false;
        if (name != other.name) return false;
        if (axis != other.axis) return false;
        if (dim0 != other.dim0) return false;
        if (dim1 != other.dim1) return false;
        if (dim2 != other.dim2) return false;
        if (dtype != other.dtype) return false;
        // For CONSTANT nodes, compare the relevant value field
        if (op == OpId::CONSTANT) {
            // If either has a meaningful float_value (non-zero or explicitly set),
            // compare as floats with epsilon tolerance; otherwise compare ints
            bool use_float = (float_value != 0.0 || other.float_value != 0.0);
            if (use_float) {
                return std::abs(float_value - other.float_value) < 1e-12;
            }
            return value == other.value;
        }
        return value == other.value;
    }

    bool operator!=(const ENode& other) const { return !(*this == other); }

    /// Hash for use in hash maps.
    /// CRITICAL: Must include ALL fields checked by operator==.
    /// Previous implementation omitted axis/dim0/dim1/dim2, causing
    /// congruence collisions where e.g. ReduceSum(x,axis=0) and
    /// ReduceSum(x,axis=1) would incorrectly hash to the same bucket
    /// and potentially be merged as equivalent — which is semantically
    /// wrong and silently corrupts the optimization result.
    struct Hash {
        size_t operator()(const ENode& n) const {
            size_t h = static_cast<size_t>(n.op);
            h ^= std::hash<int64_t>{}(n.value) + 0x9e3779b9 + (h << 6) + (h >> 2);
            // Hash float_value by bit-casting to uint64_t for deterministic hashing
            uint64_t fv_bits = 0;
            std::memcpy(&fv_bits, &n.float_value, sizeof(double));
            h ^= std::hash<uint64_t>{}(fv_bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(n.name) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(n.axis)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(n.dim0)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(n.dim1)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(n.dim2)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(n.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (auto c : n.children) {
                h ^= std::hash<int64_t>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    std::string to_string() const {
        std::ostringstream oss;
        oss << op_to_string(op);
        if (!name.empty()) oss << "(" << name << ")";
        else if (op == OpId::CONSTANT) {
            if (float_value != 0.0) {
                oss << "(" << float_value << ")";
            } else {
                oss << "(" << value << ")";
            }
        }
        else if (axis >= 0) oss << "(axis=" << axis << ")";
        if (dim0 >= 0) oss << "<" << dim0 << "," << dim1 << "," << dim2 << ">";
        if (!children.empty()) {
            oss << "[";
            for (size_t i = 0; i < children.size(); ++i) {
                if (i > 0) oss << ",";
                oss << "c" << children[i];
            }
            oss << "]";
        }
        return oss.str();
    }
};

/// EClass: a set of equivalent e-nodes with attached analysis data.
struct EClass {
    ClassId id;
    std::vector<NodeId> nodes;          // All e-nodes in this class
    std::vector<ClassId> parents;       // Classes that have nodes whose children include this class
    ClassAnalysis        analysis;       // Per-class semantic metadata
};

// ─────────────────────────────────────────────────────────────────────────
// Rewrite Scheduling Infrastructure
// ─────────────────────────────────────────────────────────────────────────

/// Priority level for rewrite rules. Higher priority rules fire first
/// within a saturation iteration. This prevents combinatorial explosion
/// by ensuring important structural rewrites (fusion, factorization) run
/// before exploratory ones (associativity, commutativity).
enum class RulePriority : int {
    CRITICAL = 100,   // Identity elimination, dead code removal
    HIGH     = 75,    // Fusion, factorization, strength reduction
    MEDIUM   = 50,    // Associativity, commutativity, distribution
    LOW      = 25,    // Tiling decomposition, normalization expansion
    EXPLORE  = 10,    // Equivalence discovery with high fanout
};

/// Saturation statistics: tracks what happened during the last saturation run.
struct SaturationStats {
    int iterations_run = 0;
    int total_merges = 0;
    int total_nodes_added = 0;
    int rules_applied = 0;
    int rules_skipped = 0;
    double elapsed_ms = 0.0;
    double final_cost = 0.0;
    bool stopped_early = false;
    std::string stop_reason;
};

/// Configuration for controlling saturation dynamics.
struct SaturationConfig {
    int     max_iters        = 30;      // Hard iteration cap
    int64_t max_nodes        = 100000;  // E-node budget
    int64_t max_classes      = 50000;   // E-class budget
    double  time_budget_ms   = 5000.0;  // Wall-clock time budget (ms)
    bool    cost_guided_filter = true;   // Skip rewrites that increase cost > threshold
    double  cost_increase_tolerance = 2.0; // Max cost increase ratio for new nodes
    int     rule_fanout_limit = 50;     // Max merges per rule per iteration
    bool    probabilistic_apply = false; // Apply rules probabilistically
    double  apply_probability = 0.8;    // Probability of applying each rule
    uint64_t rng_seed = 42;            // Seed for probabilistic application

    // ── Beam Pruning / Node Cap Per E-Class ───────────────────────────
    int64_t max_nodes_per_class = 100;     // Beam width: max nodes per e-class
    bool    beam_pruning = true;            // Enable beam pruning
    double  cost_improvement_threshold = 0.001; // Early stop if cost improvement < this

    // ── Phase-Based Saturation ────────────────────────────────────────
    // Per-priority iteration budget: [CRITICAL, HIGH, MEDIUM, LOW, EXPLORE]
    int phase_iters[5] = {10, 8, 5, 3, 2};
};

/// RewriteRule: a function that, given the current e-graph, produces
/// new equivalences to add. Each rule returns a list of (lhs_class, rhs_class)
/// pairs meaning "these two classes are equivalent."
///
/// Rules can also add entirely new nodes to the graph.
struct RewriteRule {
    std::string     name;
    std::string     description;
    RulePriority    priority = RulePriority::MEDIUM;

    /// Apply the rule to the e-graph. Returns pairs of class IDs
    /// that should be merged (found equivalent).
    ///
    /// The rule may also add new e-nodes to the graph before returning.
    std::function<std::vector<std::pair<ClassId, ClassId>>(class EGraph&)> apply;

    /// Optional: should this rule be applied given current e-graph stats?
    /// Can be used for cost-guided filtering to prevent saturation explosion.
    std::function<bool(const class EGraph&)> should_apply = nullptr;
};

/// ExtractionResult: the cheapest expression extracted from an e-class.
struct ExtractionResult {
    ClassId root_class = NULL_CLASS;
    double  cost = 0.0;
    std::string expr_string;     // Human-readable expression
    std::vector<ENode> nodes;    // The extracted expression tree
    ClassAnalysis analysis;       // Analysis data of the root class
};

// ─────────────────────────────────────────────────────────────────────────
// Merge Deduplication Hash
// ─────────────────────────────────────────────────────────────────────────

/// Custom hash for std::pair<ClassId, ClassId> used in merge deduplication.
struct ClassPairHash {
    size_t operator()(const std::pair<ClassId, ClassId>& p) const {
        size_t h1 = std::hash<ClassId>{}(p.first);
        size_t h2 = std::hash<ClassId>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

// ─────────────────────────────────────────────────────────────────────────
// E-Graph Implementation
// ─────────────────────────────────────────────────────────────────────────

/// EGraph: the core equality saturation data structure.
///
/// Represents a set of expressions and their equivalences compactly.
/// Supports:
///   - Adding new expressions (build mode)
///   - Merging equivalence classes (union)
///   - Congruence closure (maintaining invariants after merge)
///   - Applying rewrite rules (saturation) with priority scheduling
///   - Extracting the cheapest expression (cost-guided, iterative DP)
///   - Operator index for O(k) targeted rule lookups
///   - Dirty class tracking for incremental rule processing
///   - Typed constants (integer + floating-point with DType)
///   - Beam pruning to cap nodes per e-class
///   - Phase-based saturation with per-priority budgets
///   - Early stopping with cost stagnation detection
///   - Saturation statistics tracking
class EGraph {
public:
    EGraph() = default;

    // ── Building ─────────────────────────────────────────────────────

    /// Add a new e-node to the graph. Returns the class it belongs to.
    /// If an identical node already exists, returns the existing class.
    ClassId add_node(ENode node) {
        // Canonicalize children to their find roots
        for (auto& c : node.children) {
            c = find(c);
        }

        // Check for existing node (congruence)
        auto it = node_hash_.find(node);
        if (it != node_hash_.end()) {
            return find(node_classes_[it->second]);
        }

        // Create new node
        NodeId nid = static_cast<NodeId>(nodes_.size());
        nodes_.push_back(node);
        node_classes_.push_back(NULL_CLASS);

        // Create a new e-class for this node
        ClassId cid = make_class();
        node_classes_[nid] = cid;
        classes_[cid].nodes.push_back(nid);

        // Register in hash for future congruence lookups
        node_hash_[node] = nid;

        // Update operator index (Improvement 1)
        op_index_[node.op].push_back(nid);

        // Mark the new node's class as dirty (Improvement 2)
        dirty_classes_.insert(cid);

        // Update parent pointers
        for (auto child_cid : node.children) {
            ClassId child_root = find(child_cid);
            if (child_root >= 0 && child_root < static_cast<ClassId>(classes_.size())) {
                classes_[child_root].parents.push_back(cid);
            }
        }

        // ── Beam Pruning in add_node (Improvement 5) ────────────────────
        // After creating a new e-class, check if the class has more nodes
        // than config_.max_nodes_per_class. If so, keep only the cheapest K
        // nodes (sorted by cost) and remove the rest.
        if (config_.beam_pruning) {
            auto& cls_ref = classes_[cid];
            if (static_cast<int64_t>(cls_ref.nodes.size()) > config_.max_nodes_per_class) {
                // Use a simple heuristic cost: leaf ops cost 1, others cost
                // 1 + number of children. A more precise cost is applied
                // during beam_prune() with the user-provided cost function.
                std::vector<std::pair<double, NodeId>> cost_node_pairs;
                cost_node_pairs.reserve(cls_ref.nodes.size());
                for (NodeId n : cls_ref.nodes) {
                    if (n < 0 || n >= static_cast<NodeId>(nodes_.size())) continue;
                    double c = 1.0 + static_cast<double>(nodes_[n].children.size());
                    cost_node_pairs.emplace_back(c, n);
                }
                std::sort(cost_node_pairs.begin(), cost_node_pairs.end());

                std::unordered_set<NodeId> keep_set;
                for (int64_t i = 0; i < config_.max_nodes_per_class &&
                     i < static_cast<int64_t>(cost_node_pairs.size()); ++i) {
                    keep_set.insert(cost_node_pairs[i].second);
                }

                std::vector<NodeId> pruned_nodes;
                pruned_nodes.reserve(config_.max_nodes_per_class);
                for (NodeId n : cls_ref.nodes) {
                    if (keep_set.count(n)) {
                        pruned_nodes.push_back(n);
                    } else {
                        // Remove from op_index_
                        OpId op = nodes_[n].op;
                        auto it = op_index_.find(op);
                        if (it != op_index_.end()) {
                            auto& vec = it->second;
                            vec.erase(std::remove(vec.begin(), vec.end(), n), vec.end());
                        }
                        // Remove from node_hash_
                        ENode canonical = nodes_[n];
                        for (auto& c : canonical.children) c = find(c);
                        auto nh_it = node_hash_.find(canonical);
                        if (nh_it != node_hash_.end() && nh_it->second == n) {
                            node_hash_.erase(nh_it);
                        }
                    }
                }
                cls_ref.nodes = std::move(pruned_nodes);
            }
        }

        return cid;
    }

    /// Add a symbol (leaf variable) to the graph.
    ClassId add_symbol(const std::string& name) {
        ENode node;
        node.op = OpId::SYMBOL;
        node.name = name;
        return add_node(node);
    }

    /// Add a symbol with shape analysis.
    ClassId add_symbol(const std::string& name, const TensorShape& shape,
                       DType dtype = DType::UNKNOWN, Layout layout = Layout::UNKNOWN) {
        ClassId cid = add_symbol(name);
        auto& cls = classes_[find(cid)];
        cls.analysis.shape = shape;
        cls.analysis.dtype = dtype;
        cls.analysis.layout = layout;
        return cid;
    }

    /// Add a typed integer constant to the graph. (Improvement 3)
    /// Backward compatible: dtype defaults to FP32.
    ClassId add_constant(int64_t value, DType dtype = DType::FP32) {
        ENode node;
        node.op = OpId::CONSTANT;
        node.value = value;
        node.dtype = dtype;
        ClassId cid = add_node(node);
        // Set analysis: scalar constant
        auto& cls = classes_[find(cid)];
        cls.analysis.shape = TensorShape({1});
        cls.analysis.dtype = dtype;
        return cid;
    }

    /// Add a floating-point constant to the graph. (Improvement 3)
    /// This fixes the "vibes math" problem where epsilon was represented
    /// as integer 1. Use add_float_constant(1e-5, DType::FP32) for epsilon.
    ClassId add_float_constant(double value, DType dtype = DType::FP32) {
        ENode node;
        node.op = OpId::CONSTANT;
        node.float_value = value;
        node.value = static_cast<int64_t>(value); // Also store truncated int for compat
        node.dtype = dtype;
        ClassId cid = add_node(node);
        // Set analysis: scalar constant
        auto& cls = classes_[find(cid)];
        cls.analysis.shape = TensorShape({1});
        cls.analysis.dtype = dtype;
        return cid;
    }

    /// Add a unary operation.
    ClassId add_unary(OpId op, ClassId child) {
        assert(op_arity(op) == 1);
        ENode node;
        node.op = op;
        node.children = {child};
        return add_node(node);
    }

    /// Add a binary operation.
    ClassId add_binary(OpId op, ClassId left, ClassId right) {
        assert(op_arity(op) == 2);
        ENode node;
        node.op = op;
        node.children = {left, right};
        return add_node(node);
    }

    /// Add a ternary operation.
    ClassId add_ternary(OpId op, ClassId a, ClassId b, ClassId c) {
        // Fallback: if op_arity doesn't match, treat as generic node
        ENode node;
        node.op = op;
        node.children = {a, b, c};
        return add_node(node);
    }

    /// Add a 4-ary operation.
    ClassId add_quaternary(OpId op, ClassId a, ClassId b, ClassId c, ClassId d) {
        assert(op_arity(op) == 4);
        ENode node;
        node.op = op;
        node.children = {a, b, c, d};
        return add_node(node);
    }

    /// Add a unary op with an axis parameter.
    ClassId add_unary_with_axis(OpId op, ClassId child, int64_t axis) {
        assert(op_arity(op) == 1);
        ENode node;
        node.op = op;
        node.children = {child};
        node.axis = axis;
        return add_node(node);
    }

    /// Add a reshape with shape parameters.
    ClassId add_reshape(ClassId child, int64_t d0, int64_t d1, int64_t d2) {
        ENode node;
        node.op = OpId::RESHAPE;
        node.children = {child};
        node.dim0 = d0;
        node.dim1 = d1;
        node.dim2 = d2;
        return add_node(node);
    }

    /// Add a tile with dimension metadata.
    ClassId add_tile(ClassId child, int64_t axis, int64_t d0 = -1,
                     int64_t d1 = -1, int64_t d2 = -1) {
        ENode node;
        node.op = OpId::TILE;
        node.children = {child};
        node.axis = axis;
        node.dim0 = d0;
        node.dim1 = d1;
        node.dim2 = d2;
        return add_node(node);
    }

    // ── Operator Index (Improvement 1) ────────────────────────────────

    /// Get all node IDs for a given operator type.
    /// Returns a static empty vector if the op has no nodes.
    /// This replaces O(N) global scans in rules with O(k) targeted lookups.
    std::vector<NodeId> nodes_of_op(OpId op) const {
        auto it = op_index_.find(op);
        if (it != op_index_.end()) return it->second;  // Returns a copy
        return {};
    }

    // ── Dirty Class Tracking (Improvement 2) ──────────────────────────

    /// Get the set of classes that have been modified since the last
    /// rebuild/clear_dirty. Rules can use this to only process classes
    /// that changed since the last iteration.
    const std::unordered_set<ClassId>& dirty_classes() const {
        return dirty_classes_;
    }

    /// Clear the dirty class set. Typically called after rules have
    /// processed all dirty classes in an iteration.
    void clear_dirty() { dirty_classes_.clear(); }

    // ── Union-Find ───────────────────────────────────────────────────

    /// Find the canonical representative of a class.
    ClassId find(ClassId cid) {
        if (cid < 0 || cid >= static_cast<ClassId>(parent_.size())) return cid;
        if (parent_[cid] != cid) {
            parent_[cid] = find(parent_[cid]);  // Path compression
        }
        return parent_[cid];
    }

    /// Const find: returns the canonical representative without path compression.
    /// Used by const contexts (should_apply lambdas, cost functions, validation).
    ClassId find(ClassId cid) const {
        if (cid < 0 || cid >= static_cast<ClassId>(parent_.size())) return cid;
        ClassId current = cid;
        while (parent_[current] != current) {
            current = parent_[current];
        }
        return current;
    }

    /// Merge two equivalence classes. Returns the new canonical class.
    /// Uses the worklist-based congruence repair instead of naive rebuild.
    ClassId merge(ClassId a, ClassId b) {
        a = find(a);
        b = find(b);
        if (a == b) return a;

        // Union by rank
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;

        // Merge nodes and parents lists
        auto& class_a = classes_[a];
        auto& class_b = classes_[b];
        class_a.nodes.insert(class_a.nodes.end(),
                             class_b.nodes.begin(), class_b.nodes.end());
        class_a.parents.insert(class_a.parents.end(),
                               class_b.parents.begin(), class_b.parents.end());

        // Merge analysis: keep the more informative analysis
        merge_analysis(class_a.analysis, class_b.analysis);

        // Mark class b as dead
        class_b.nodes.clear();
        class_b.parents.clear();

        // Mark the merged class as dirty (Improvement 2)
        dirty_classes_.insert(a);

        // Invalidate old node hash entries that referenced class b's nodes
        // (The worklist rebuild will fix the congruence table.)

        // Schedule the merged class's parents for congruence repair
        pending_worklist_.push_back(a);
        // Also schedule all parents of both a and b
        for (auto p : class_a.parents) {
            pending_worklist_.push_back(find(p));
        }

        return a;
    }

    /// Check if two classes are equivalent.
    bool are_equivalent(ClassId a, ClassId b) {
        return find(a) == find(b);
    }

    // ── Congruence Closure / Rebuild ─────────────────────────────────

    /// Worklist-based incremental congruence repair.
    /// Instead of scanning ALL nodes (O(n) per iteration, up to 100 iterations),
    /// we only repair the subset of the e-graph that might have been affected
    /// by recent merges. This is the approach used by egg/egglog.
    ///
    /// Algorithm:
    ///   1. Process the worklist of class IDs whose parents might need repair
    ///   2. For each class on the worklist, re-canonicalize its nodes
    ///   3. If we discover new congruences, merge and add to the worklist
    ///   4. Repeat until the worklist is empty
    ///
    /// Complexity: O(k * d) where k = worklist size, d = max parent depth
    ///             instead of O(n * 100) where n = total nodes
    void rebuild() {
        int iterations = 0;
        const int max_rebuild_iters = 50;

        while (!pending_worklist_.empty() && iterations < max_rebuild_iters) {
            iterations++;

            // Deduplicate the worklist
            std::unordered_set<ClassId> work_set;
            std::vector<ClassId> current_work;
            for (auto cid : pending_worklist_) {
                ClassId root = find(cid);
                if (work_set.insert(root).second) {
                    current_work.push_back(root);
                }
            }
            pending_worklist_.clear();

            // For each class in the work set, check its nodes for new congruences
            std::unordered_map<ENode, NodeId, ENode::Hash> new_congruences;

            for (ClassId cid : current_work) {
                if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
                auto& cls = classes_[cid];
                if (cls.nodes.empty()) continue;  // Dead class

                for (NodeId nid : cls.nodes) {
                    if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                    auto& node = nodes_[nid];

                    // Re-canonicalize the node
                    ENode canonical = node;
                    for (auto& c : canonical.children) {
                        c = find(c);
                    }

                    auto it = new_congruences.find(canonical);
                    if (it != new_congruences.end()) {
                        // Found a congruent node! Merge their classes.
                        ClassId c1 = find(node_classes_[nid]);
                        ClassId c2 = find(node_classes_[it->second]);
                        if (c1 != c2) {
                            // Perform raw merge without triggering another rebuild
                            if (rank_[c1] < rank_[c2]) std::swap(c1, c2);
                            parent_[c2] = c1;
                            if (rank_[c1] == rank_[c2]) rank_[c1]++;
                            auto& ca = classes_[c1];
                            auto& cb = classes_[c2];
                            ca.nodes.insert(ca.nodes.end(),
                                            cb.nodes.begin(), cb.nodes.end());
                            ca.parents.insert(ca.parents.end(),
                                              cb.parents.begin(), cb.parents.end());
                            merge_analysis(ca.analysis, cb.analysis);
                            cb.nodes.clear();
                            cb.parents.clear();

                            // Mark merged class as dirty (Improvement 2)
                            dirty_classes_.insert(c1);

                            // Schedule parents of the newly-merged class
                            for (auto p : ca.parents) {
                                pending_worklist_.push_back(find(p));
                            }
                        }
                    } else {
                        new_congruences[canonical] = nid;
                    }
                }
            }

            // Update the main node hash with new congruence info
            // Rebuild the hash from scratch for the affected nodes
            for (ClassId cid : current_work) {
                if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
                auto& cls = classes_[cid];
                for (NodeId nid : cls.nodes) {
                    if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                    ENode canonical = nodes_[nid];
                    for (auto& c : canonical.children) {
                        c = find(c);
                    }
                    node_hash_[canonical] = nid;
                }
            }
        }

        // Clear dirty classes after rebuild (Improvement 2)
        dirty_classes_.clear();
    }

    // ── Analysis Propagation ─────────────────────────────────────────

    /// Update analysis data for a class based on a new node added to it.
    /// This propagates shape, dtype, layout, and FLOP information bottom-up.
    void update_analysis(ClassId cid) {
        cid = find(cid);
        if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) return;
        auto& cls = classes_[cid];

        // For each node in the class, compute its analysis
        // and merge into the class analysis (taking the most informative)
        for (NodeId nid : cls.nodes) {
            if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
            ClassAnalysis node_analysis = compute_node_analysis(nid);
            merge_analysis(cls.analysis, node_analysis);
        }
    }

    /// Compute analysis for a single node based on its op and children.
    ClassAnalysis compute_node_analysis(NodeId nid) const {
        if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) return {};
        const auto& node = nodes_[nid];
        ClassAnalysis result;

        // Get children analyses
        std::vector<ClassAnalysis> child_analyses;
        for (auto c : node.children) {
            ClassId root = find_internal(c);
            if (root >= 0 && root < static_cast<ClassId>(classes_.size())) {
                child_analyses.push_back(classes_[root].analysis);
            } else {
                child_analyses.push_back({});
            }
        }

        switch (node.op) {
            case OpId::SYMBOL:
                // Analysis set by add_symbol() or externally
                break;

            case OpId::CONSTANT:
                result.shape = TensorShape({1});
                result.dtype = (node.dtype != DType::UNKNOWN) ? node.dtype : DType::FP32;
                break;

            case OpId::ADD:
            case OpId::SUB:
            case OpId::MUL:
            case OpId::DIV:
                // Elementwise: shape = broadcast of children shapes
                if (child_analyses.size() >= 2) {
                    result.shape = broadcast_shapes(
                        child_analyses[0].shape, child_analyses[1].shape);
                    result.dtype = promote_dtypes(
                        child_analyses[0].dtype, child_analyses[1].dtype);
                    // Floating-point safety: MUL/DIV with non-trivial tensors
                    // is NOT safe to reorder in FP
                    if (node.op == OpId::MUL || node.op == OpId::DIV) {
                        result.floating_point_safe = false;
                    }
                }
                result.estimated_flops = result.shape.num_elements() > 0
                    ? result.shape.num_elements() : 0;
                break;

            case OpId::NEG:
            case OpId::RELU:
            case OpId::GELU:
            case OpId::SIGMOID:
            case OpId::EXP:
            case OpId::LOG:
            case OpId::SQRT:
            case OpId::RECIPROCAL:
            case OpId::DROPOUT:
                if (!child_analyses.empty()) {
                    result.shape = child_analyses[0].shape;
                    result.dtype = child_analyses[0].dtype;
                }
                result.estimated_flops = result.shape.num_elements() > 0
                    ? result.shape.num_elements() : 0;
                if (node.op == OpId::EXP || node.op == OpId::LOG ||
                    node.op == OpId::SQRT || node.op == OpId::RECIPROCAL) {
                    result.floating_point_safe = false;
                }
                break;

            case OpId::MATMUL:
                if (child_analyses.size() >= 2) {
                    auto& sa = child_analyses[0].shape;
                    auto& sb = child_analyses[1].shape;
                    // C[M,N] = A[M,K] @ B[K,N]
                    int64_t m = (sa.ndim() >= 2) ? sa[sa.ndim()-2] : -1;
                    int64_t k = (sa.ndim() >= 2) ? sa[sa.ndim()-1] : -1;
                    int64_t n = (sb.ndim() >= 2) ? sb[sb.ndim()-1] : -1;
                    if (m > 0 && n > 0) {
                        result.shape = TensorShape({m, n});
                    }
                    result.dtype = promote_dtypes(
                        child_analyses[0].dtype, child_analyses[1].dtype);
                    result.tc_compatible = (m > 0 && n > 0 && k > 0);
                    if (m > 0 && n > 0 && k > 0) {
                        result.estimated_flops = 2 * m * n * k;
                    }
                    result.floating_point_safe = false;  // MatMul reordering is FP-unsafe
                }
                break;

            case OpId::TRANSPOSE:
                if (!child_analyses.empty()) {
                    result.shape = transpose_shape(child_analyses[0].shape);
                    result.dtype = child_analyses[0].dtype;
                }
                break;

            case OpId::RESHAPE:
                if (!child_analyses.empty()) {
                    result.dtype = child_analyses[0].dtype;
                    if (node.dim0 > 0 && node.dim1 > 0 && node.dim2 > 0) {
                        result.shape = TensorShape({node.dim0, node.dim1, node.dim2});
                    }
                }
                break;

            case OpId::REDUCE_SUM:
            case OpId::REDUCE_MAX:
            case OpId::REDUCE_MEAN:
                if (!child_analyses.empty()) {
                    result.shape = reduce_shape(
                        child_analyses[0].shape, node.axis);
                    result.dtype = child_analyses[0].dtype;
                    result.is_reduction = true;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::SOFTMAX:
            case OpId::FUSED_SOFTMAX:
                if (!child_analyses.empty()) {
                    result.shape = child_analyses[0].shape;
                    result.dtype = child_analyses[0].dtype;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::LAYERNORM:
            case OpId::RMSNORM:
            case OpId::FUSED_LAYERNORM:
            case OpId::FUSED_RMSNORM:
                if (!child_analyses.empty()) {
                    result.shape = child_analyses[0].shape;
                    result.dtype = child_analyses[0].dtype;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::FUSED_MATMUL_RELU:
            case OpId::FUSED_MATMUL_ADD:
            case OpId::FUSED_GEMM:
                if (child_analyses.size() >= 2) {
                    auto& sa = child_analyses[0].shape;
                    auto& sb = child_analyses[1].shape;
                    int64_t m = (sa.ndim() >= 2) ? sa[sa.ndim()-2] : -1;
                    int64_t n = (sb.ndim() >= 2) ? sb[sb.ndim()-1] : -1;
                    if (m > 0 && n > 0) {
                        result.shape = TensorShape({m, n});
                    }
                    result.tc_compatible = true;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::FUSED_MATMUL_ADD_RELU:
                if (child_analyses.size() >= 3) {
                    auto& sa = child_analyses[0].shape;
                    auto& sb = child_analyses[1].shape;
                    int64_t m = (sa.ndim() >= 2) ? sa[sa.ndim()-2] : -1;
                    int64_t n = (sb.ndim() >= 2) ? sb[sb.ndim()-1] : -1;
                    if (m > 0 && n > 0) {
                        result.shape = TensorShape({m, n});
                    }
                    result.tc_compatible = true;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::FUSED_ADD_LN:
                if (child_analyses.size() >= 2) {
                    result.shape = child_analyses[0].shape;
                    result.dtype = child_analyses[0].dtype;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::FUSED_MHA:
                if (child_analyses.size() >= 3) {
                    auto& sq = child_analyses[0].shape;
                    int64_t m = (sq.ndim() >= 2) ? sq[sq.ndim()-2] : -1;
                    int64_t n = (sq.ndim() >= 2) ? sq[sq.ndim()-1] : -1;
                    if (m > 0 && n > 0) {
                        result.shape = TensorShape({m, n});
                    }
                    result.tc_compatible = true;
                    result.floating_point_safe = false;
                }
                break;

            case OpId::TILE:
            case OpId::UNTILE:
                if (!child_analyses.empty()) {
                    result.shape = child_analyses[0].shape;
                    result.dtype = child_analyses[0].dtype;
                }
                break;

            case OpId::BROADCAST:
                if (!child_analyses.empty()) {
                    result.dtype = child_analyses[0].dtype;
                    // Broadcast may expand the shape
                    result.shape = child_analyses[0].shape;
                }
                break;

            case OpId::IDENTITY:
                if (!child_analyses.empty()) {
                    result = child_analyses[0];
                }
                break;
        }

        return result;
    }

    // ── Beam Pruning (Improvement 5) ─────────────────────────────────

    /// Beam prune: for each live e-class, compute cost of each node,
    /// keep only the max_nodes_per_class cheapest nodes, and remove
    /// expensive nodes from the class. Also updates op_index_.
    ///
    /// \param op_cost  A function that returns the cost of each operation.
    void beam_prune(std::function<double(OpId, const ENode&)> op_cost) {
        if (!config_.beam_pruning) return;

        int64_t max_per_class = config_.max_nodes_per_class;

        for (ClassId cid = 0; cid < static_cast<ClassId>(classes_.size()); ++cid) {
            auto& cls = classes_[cid];
            if (cls.nodes.empty()) continue;  // Dead class
            if (static_cast<int64_t>(cls.nodes.size()) <= max_per_class) continue;

            // Compute cost for each node in the class
            std::vector<std::pair<double, NodeId>> cost_node;
            cost_node.reserve(cls.nodes.size());
            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                double c = op_cost(nodes_[nid].op, nodes_[nid]);
                cost_node.emplace_back(c, nid);
            }

            // Sort by cost (cheapest first)
            std::sort(cost_node.begin(), cost_node.end());

            // Keep only the cheapest max_per_class nodes
            std::unordered_set<NodeId> keep_set;
            keep_set.reserve(max_per_class);
            for (int64_t i = 0; i < max_per_class && i < static_cast<int64_t>(cost_node.size()); ++i) {
                keep_set.insert(cost_node[i].second);
            }

            // Remove expensive nodes from the class and from op_index_
            std::vector<NodeId> new_nodes;
            new_nodes.reserve(max_per_class);
            for (NodeId nid : cls.nodes) {
                if (keep_set.count(nid)) {
                    new_nodes.push_back(nid);
                } else {
                    // Remove from op_index_
                    OpId op = nodes_[nid].op;
                    auto it = op_index_.find(op);
                    if (it != op_index_.end()) {
                        auto& vec = it->second;
                        vec.erase(std::remove(vec.begin(), vec.end(), nid), vec.end());
                    }
                    // Remove from node_hash_
                    ENode canonical = nodes_[nid];
                    for (auto& c : canonical.children) {
                        c = find(c);
                    }
                    auto nh_it = node_hash_.find(canonical);
                    if (nh_it != node_hash_.end() && nh_it->second == nid) {
                        node_hash_.erase(nh_it);
                    }
                }
            }
            cls.nodes = std::move(new_nodes);

            // Mark as dirty since we modified the class
            dirty_classes_.insert(cid);
        }
    }

    // ── Saturation ───────────────────────────────────────────────────

    /// Run equality saturation using PHASE-BASED scheduling (Improvement 6).
    ///
    /// Phases:
    ///   Phase 1: CRITICAL rules only (identity elimination, dead code)
    ///   Phase 2: HIGH rules (fusion, strength reduction)
    ///   Phase 3: MEDIUM rules (associativity, commutativity)
    ///   Phase 4: LOW rules (tiling decomposition)
    ///   Phase 5: EXPLORE rules (distribution, etc.)
    ///
    /// Each phase has its own iteration count (from config.phase_iters[]).
    /// Lower priority phases get fewer iterations.
    ///
    /// Also includes:
    ///   - Merge deduplication (Improvement 4)
    ///   - Early stopping with cost stagnation (Improvement 7)
    ///   - Beam pruning between phases (Improvement 5)
    ///   - Saturation statistics tracking (Improvement 9)
    ///
    /// \param rules       The rewrite rules to apply.
    /// \param config      Saturation configuration (budgets, limits).
    /// \param root_class  The root class for cost tracking (for early stopping).
    /// \param op_cost     Cost function for beam pruning and early stopping.
    /// \return            Number of iterations actually run.
    int saturate(
        const std::vector<RewriteRule>& rules,
        SaturationConfig config = SaturationConfig{},
        ClassId root_class = NULL_CLASS,
        std::function<double(OpId, const ENode&)> op_cost = nullptr
    ) {
        config_ = config;

        // Reset statistics
        stats_ = SaturationStats{};

        // Initialize RNG for probabilistic application
        std::mt19937 rng(config.rng_seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Early stopping state (Improvement 7)
        best_cost_ = 1e18;
        stagnation_count_ = 0;

        auto start_time = std::chrono::steady_clock::now();
        int64_t nodes_before = static_cast<int64_t>(nodes_.size());

        // Sort rules by priority (descending) for each phase
        std::vector<size_t> rule_order(rules.size());
        std::iota(rule_order.begin(), rule_order.end(), 0);
        std::sort(rule_order.begin(), rule_order.end(),
            [&](size_t a, size_t b) {
                return static_cast<int>(rules[a].priority) >
                       static_cast<int>(rules[b].priority);
            });

        // Define phase boundaries: which priority levels go in which phase
        // Phase 0: CRITICAL, Phase 1: HIGH, Phase 2: MEDIUM,
        // Phase 3: LOW, Phase 4: EXPLORE
        struct PhaseDef {
            int min_priority;  // Inclusive minimum priority value
            int max_priority;  // Inclusive maximum priority value
            int iters;         // Iteration budget for this phase
        };

        PhaseDef phases[5] = {
            { static_cast<int>(RulePriority::CRITICAL), static_cast<int>(RulePriority::CRITICAL), config.phase_iters[0] },
            { static_cast<int>(RulePriority::HIGH),     static_cast<int>(RulePriority::HIGH),     config.phase_iters[1] },
            { static_cast<int>(RulePriority::MEDIUM),   static_cast<int>(RulePriority::MEDIUM),   config.phase_iters[2] },
            { static_cast<int>(RulePriority::LOW),      static_cast<int>(RulePriority::LOW),      config.phase_iters[3] },
            { static_cast<int>(RulePriority::EXPLORE),  static_cast<int>(RulePriority::EXPLORE),  config.phase_iters[4] },
        };

        for (int phase = 0; phase < 5; ++phase) {
            const auto& pd = phases[phase];

            for (int i = 0; i < pd.iters; ++i) {
                // Check global budgets
                if (static_cast<int64_t>(nodes_.size()) > config.max_nodes) {
                    stats_.stopped_early = true;
                    stats_.stop_reason = "node budget exceeded";
                    goto done;
                }
                if (static_cast<int64_t>(classes_.size()) > config.max_classes) {
                    stats_.stopped_early = true;
                    stats_.stop_reason = "class budget exceeded";
                    goto done;
                }

                // Check time budget
                if (config.time_budget_ms > 0) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(
                        now - start_time).count();
                    if (elapsed_ms > config.time_budget_ms) {
                        stats_.stopped_early = true;
                        stats_.stop_reason = "time budget exceeded";
                        goto done;
                    }
                }

                bool any_change = false;
                int total_merges_this_iter = 0;

                // Merge deduplication set (Improvement 4)
                std::unordered_set<std::pair<ClassId, ClassId>, ClassPairHash> seen_merges;

                // Apply rules in priority order that match this phase
                for (auto rule_idx : rule_order) {
                    const auto& rule = rules[rule_idx];
                    int rule_prio = static_cast<int>(rule.priority);

                    // Filter: only apply rules in the current phase's priority range
                    if (rule_prio < pd.min_priority || rule_prio > pd.max_priority) continue;

                    // Check if rule should be applied
                    if (rule.should_apply && !rule.should_apply(*this)) {
                        stats_.rules_skipped++;
                        continue;
                    }

                    // Probabilistic application
                    if (config.probabilistic_apply &&
                        uniform(rng) > config.apply_probability) {
                        stats_.rules_skipped++;
                        continue;
                    }

                    // Cost-guided filtering: if the graph is already large,
                    // skip exploratory rules
                    if (config.cost_guided_filter &&
                        static_cast<int64_t>(nodes_.size()) > config.max_nodes / 2) {
                        if (static_cast<int>(rule.priority) <=
                            static_cast<int>(RulePriority::EXPLORE)) {
                            stats_.rules_skipped++;
                            continue;
                        }
                    }

                    // Apply the rule
                    auto merges = rule.apply(*this);
                    stats_.rules_applied++;

                    // Enforce fanout limit per rule
                    if (config.rule_fanout_limit > 0 &&
                        static_cast<int>(merges.size()) > config.rule_fanout_limit) {
                        merges.resize(config.rule_fanout_limit);
                    }

                    int merges_applied = 0;
                    for (auto& [a, b] : merges) {
                        ClassId ra = find(a);
                        ClassId rb = find(b);
                        if (ra == rb) continue;

                        // Merge deduplication (Improvement 4)
                        auto key = (ra < rb) ? std::make_pair(ra, rb)
                                              : std::make_pair(rb, ra);
                        if (seen_merges.count(key)) continue;
                        seen_merges.insert(key);

                        // Cost-guided filtering: check if the merge would
                        // dramatically increase graph size
                        if (config.cost_guided_filter && merges_applied > config.rule_fanout_limit) {
                            break;
                        }
                        merge(a, b);
                        any_change = true;
                        merges_applied++;
                        stats_.total_merges++;
                    }
                    total_merges_this_iter += merges_applied;
                }

                // Run incremental congruence repair after all rules fire
                rebuild();

                // Update analysis for modified classes
                propagate_analysis();

                // Beam pruning (Improvement 5)
                if (config.beam_pruning && op_cost) {
                    beam_prune(op_cost);
                }

                // Early stopping with cost stagnation (Improvement 7)
                if (root_class != NULL_CLASS && op_cost) {
                    double current_cost = evaluate_class_cost(root_class, op_cost);
                    if (best_cost_ - current_cost > config.cost_improvement_threshold) {
                        best_cost_ = current_cost;
                        stagnation_count_ = 0;
                    } else {
                        stagnation_count_++;
                    }
                    if (stagnation_count_ > 3) {
                        stats_.stopped_early = true;
                        stats_.stop_reason = "cost stagnation (>" + std::to_string(3) + " iters)";
                        stats_.iterations_run++;
                        goto done;
                    }
                }

                stats_.iterations_run++;
                if (!any_change && total_merges_this_iter == 0) break;  // Fixpoint
            }
        }

    done:
        // Finalize statistics
        auto end_time = std::chrono::steady_clock::now();
        stats_.elapsed_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        stats_.total_nodes_added = static_cast<int>(nodes_.size() - nodes_before);

        // Final cost evaluation
        if (root_class != NULL_CLASS && op_cost) {
            stats_.final_cost = evaluate_class_cost(root_class, op_cost);
        }

        return stats_.iterations_run;
    }

    /// Legacy saturate API for backward compatibility.
    int saturate(
        const std::vector<RewriteRule>& rules,
        int max_iters,
        int64_t max_nodes = 100000,
        int64_t max_classes = 50000
    ) {
        SaturationConfig config;
        config.max_iters = max_iters;
        config.max_nodes = max_nodes;
        config.max_classes = max_classes;
        return saturate(rules, config);
    }

    /// Get statistics from the last saturation run (Improvement 9).
    SaturationStats last_saturation_stats() const {
        return stats_;
    }

    // ── Parallel Saturation ──────────────────────────────────────────

    /// Run equality saturation in parallel using std::thread.
    ///
    /// Partitions rules by op type and processes them in parallel.
    /// Each thread works on a disjoint set of e-classes (partitioned
    /// by class_id % num_threads). Results are merged with a
    /// mutex-protected merge queue.
    ///
    /// \param rules       The rewrite rules to apply.
    /// \param config      Saturation configuration.
    /// \param root_class  The root class for cost tracking.
    /// \param op_cost     Cost function for beam pruning and early stopping.
    /// \param num_threads Number of threads to use (default: hardware concurrency).
    /// \return            Number of iterations actually run.
    int parallel_saturate(
        const std::vector<RewriteRule>& rules,
        SaturationConfig config = SaturationConfig{},
        ClassId root_class = NULL_CLASS,
        std::function<double(OpId, const ENode&)> op_cost = nullptr,
        int num_threads = 0
    ) {
        config_ = config;
        stats_ = SaturationStats{};

        if (num_threads <= 0) {
            num_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (num_threads <= 0) num_threads = 4;
        }
        if (num_threads == 1) {
            return saturate(rules, config, root_class, op_cost);
        }

        auto start_time = std::chrono::steady_clock::now();
        int64_t nodes_before = static_cast<int64_t>(nodes_.size());

        // Partition rules by their primary op type
        // Rules that match specific ops go to threads that "own" those ops
        std::vector<std::vector<size_t>> thread_rules(num_threads);
        for (size_t i = 0; i < rules.size(); ++i) {
            // Simple hash-based partitioning
            size_t t = std::hash<std::string>{}(rules[i].name) % static_cast<size_t>(num_threads);
            thread_rules[t].push_back(i);
        }

        // Merge queue protected by mutex
        std::mutex merge_mutex;
        std::vector<std::pair<ClassId, ClassId>> merge_queue;

        // Phase-based iteration
        struct PhaseDef {
            int min_priority;
            int max_priority;
            int iters;
        };

        PhaseDef phases[5] = {
            { static_cast<int>(RulePriority::CRITICAL), static_cast<int>(RulePriority::CRITICAL), config.phase_iters[0] },
            { static_cast<int>(RulePriority::HIGH),     static_cast<int>(RulePriority::HIGH),     config.phase_iters[1] },
            { static_cast<int>(RulePriority::MEDIUM),   static_cast<int>(RulePriority::MEDIUM),   config.phase_iters[2] },
            { static_cast<int>(RulePriority::LOW),      static_cast<int>(RulePriority::LOW),      config.phase_iters[3] },
            { static_cast<int>(RulePriority::EXPLORE),  static_cast<int>(RulePriority::EXPLORE),  config.phase_iters[4] },
        };

        for (int phase = 0; phase < 5; ++phase) {
            const auto& pd = phases[phase];

            for (int iter = 0; iter < pd.iters; ++iter) {
                // Check budgets
                if (static_cast<int64_t>(nodes_.size()) > config.max_nodes ||
                    static_cast<int64_t>(classes_.size()) > config.max_classes) {
                    stats_.stopped_early = true;
                    stats_.stop_reason = "budget exceeded";
                    goto parallel_done;
                }

                merge_queue.clear();
                std::atomic<int> total_applied{0};
                std::atomic<int> total_skipped{0};

                // Launch threads
                std::vector<std::thread> threads;
                for (int t = 0; t < num_threads; ++t) {
                    threads.emplace_back([&, t]() {
                        for (auto rule_idx : thread_rules[t]) {
                            const auto& rule = rules[rule_idx];
                            int rule_prio = static_cast<int>(rule.priority);

                            if (rule_prio < pd.min_priority || rule_prio > pd.max_priority) continue;

                            if (rule.should_apply && !rule.should_apply(*this)) {
                                total_skipped++;
                                continue;
                            }

                            // Apply the rule — this is safe because each thread
                            // adds different nodes (different rule sets)
                            auto merges = rule.apply(*this);

                            // Queue merges for sequential processing
                            {
                                std::lock_guard<std::mutex> lock(merge_mutex);
                                if (config.rule_fanout_limit > 0 &&
                                    static_cast<int>(merges.size()) > config.rule_fanout_limit) {
                                    merges.resize(config.rule_fanout_limit);
                                }
                                for (auto& [a, b] : merges) {
                                    merge_queue.push_back({a, b});
                                }
                            }
                            total_applied++;
                        }
                    });
                }

                for (auto& th : threads) th.join();

                // Process merges sequentially (union-find is not thread-safe)
                std::unordered_set<std::pair<ClassId, ClassId>, ClassPairHash> seen_merges;
                for (auto& [a, b] : merge_queue) {
                    ClassId ra = find(a);
                    ClassId rb = find(b);
                    if (ra == rb) continue;
                    auto key = (ra < rb) ? std::make_pair(ra, rb)
                                          : std::make_pair(rb, ra);
                    if (seen_merges.count(key)) continue;
                    seen_merges.insert(key);
                    merge(a, b);
                    stats_.total_merges++;
                }

                rebuild();
                propagate_analysis();

                if (config.beam_pruning && op_cost) {
                    beam_prune(op_cost);
                }

                stats_.rules_applied += total_applied.load();
                stats_.rules_skipped += total_skipped.load();
                stats_.iterations_run++;

                // Early stopping
                if (root_class != NULL_CLASS && op_cost) {
                    double current_cost = evaluate_class_cost(root_class, op_cost);
                    if (best_cost_ - current_cost > config.cost_improvement_threshold) {
                        best_cost_ = current_cost;
                        stagnation_count_ = 0;
                    } else {
                        stagnation_count_++;
                    }
                    if (stagnation_count_ > 3) {
                        stats_.stopped_early = true;
                        stats_.stop_reason = "cost stagnation";
                        goto parallel_done;
                    }
                }
            }
        }

    parallel_done:
        auto end_time = std::chrono::steady_clock::now();
        stats_.elapsed_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        stats_.total_nodes_added = static_cast<int>(nodes_.size() - nodes_before);

        if (root_class != NULL_CLASS && op_cost) {
            stats_.final_cost = evaluate_class_cost(root_class, op_cost);
        }

        return stats_.iterations_run;
    }

    // ── Cost-Guided Extraction ───────────────────────────────────────

    /// Extract the cheapest expression from the given class.
    /// Uses iterative bottom-up dynamic programming with topological
    /// scheduling instead of recursive traversal to avoid stack blowup
    /// on large saturated graphs.
    ///
    /// Algorithm:
    ///   1. Topologically sort all reachable e-classes from root
    ///   2. Process in reverse topological order (leaves first)
    ///   3. For each class, find the cheapest node
    ///   4. Track the best node per class for reconstruction
    ///
    /// \param root       The class to extract from.
    /// \param op_cost    A function that returns the cost of each operation.
    /// \return           The extraction result.
    ExtractionResult extract(
        ClassId root,
        std::function<double(OpId, const ENode&)> op_cost
    ) const {
        root = find_internal(root);

        // Step 1: Discover all reachable classes via iterative BFS
        // (no recursion, no stack overflow)
        std::vector<ClassId> topo_order;
        std::unordered_set<ClassId> visited;
        std::deque<ClassId> work_queue;
        work_queue.push_back(root);
        visited.insert(root);

        while (!work_queue.empty()) {
            ClassId cid = work_queue.front();
            work_queue.pop_front();
            topo_order.push_back(cid);

            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            const auto& cls = classes_[cid];

            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                for (auto child_cid : nodes_[nid].children) {
                    ClassId child_root = find_internal(child_cid);
                    if (child_root >= 0 && !visited.count(child_root)) {
                        visited.insert(child_root);
                        work_queue.push_back(child_root);
                    }
                }
            }
        }

        // Step 2: Reverse the BFS order to get leaves-first ordering
        // BFS from root gives root-first, children-later. Reverse = leaves first.
        std::reverse(topo_order.begin(), topo_order.end());

        // Step 3: Iterative DP — compute best cost and best node per class
        std::unordered_map<ClassId, double> class_cost;
        std::unordered_map<ClassId, NodeId> class_best_node;

        for (ClassId cid : topo_order) {
            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            const auto& cls = classes_[cid];
            if (cls.nodes.empty()) continue;

            double best_cost = 1e18;
            NodeId best_node = NULL_NODE;

            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                const auto& node = nodes_[nid];
                double c = op_cost(node.op, node);
                for (auto child_cid : node.children) {
                    ClassId child_root = find_internal(child_cid);
                    auto it = class_cost.find(child_root);
                    if (it != class_cost.end()) {
                        c += it->second;
                    } else {
                        c += 1.0;  // Default cost for unknown/unvisited
                    }
                }
                if (c < best_cost) {
                    best_cost = c;
                    best_node = nid;
                }
            }

            class_cost[cid] = best_cost;
            class_best_node[cid] = best_node;
        }

        // Step 4: Reconstruct the expression tree (iteratively)
        ExtractionResult result;
        result.root_class = root;

        auto cost_it = class_cost.find(root);
        result.cost = (cost_it != class_cost.end()) ? cost_it->second : 0.0;

        // Build expression string iteratively using bottom-up memoization
        // (no recursion, no stack overflow risk on deep expressions)
        std::unordered_map<ClassId, std::string> class_string;
        // Process in topo_order (leaves already computed first)
        for (ClassId cid : topo_order) {
            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            auto it = class_best_node.find(cid);
            if (it == class_best_node.end() || it->second == NULL_NODE) {
                class_string[cid] = "?";
                continue;
            }
            const auto& node = nodes_[it->second];
            if (node.children.empty()) {
                if (node.op == OpId::SYMBOL) {
                    class_string[cid] = node.name;
                } else if (node.op == OpId::CONSTANT) {
                    if (node.float_value != 0.0) {
                        class_string[cid] = std::to_string(node.float_value);
                    } else {
                        class_string[cid] = std::to_string(node.value);
                    }
                } else {
                    class_string[cid] = op_to_string(node.op);
                }
                continue;
            }
            std::string s = op_to_string(node.op);
            if (node.axis >= 0) s += "<axis=" + std::to_string(node.axis) + ">";
            s += "(";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i > 0) s += ", ";
                ClassId child_root = find_internal(node.children[i]);
                auto str_it = class_string.find(child_root);
                s += (str_it != class_string.end()) ? str_it->second : "?";
            }
            s += ")";
            class_string[cid] = std::move(s);
        }

        auto root_str_it = class_string.find(root);
        result.expr_string = (root_str_it != class_string.end()) ? root_str_it->second : "?";

        // Collect all nodes in the extracted tree (iteratively)
        std::deque<ClassId> collect_queue;
        std::unordered_set<ClassId> collected;
        collect_queue.push_back(root);

        while (!collect_queue.empty()) {
            ClassId cid = find_internal(collect_queue.front());
            collect_queue.pop_front();
            if (collected.count(cid)) continue;
            collected.insert(cid);

            auto it = class_best_node.find(cid);
            if (it == class_best_node.end() || it->second == NULL_NODE) continue;
            result.nodes.push_back(nodes_[it->second]);

            for (auto child_cid : nodes_[it->second].children) {
                collect_queue.push_back(child_cid);
            }
        }

        // Copy analysis from the root class
        if (root >= 0 && root < static_cast<ClassId>(classes_.size())) {
            result.analysis = classes_[root].analysis;
        }

        return result;
    }

    /// Extract the cheapest expression using shape-aware cost model.
    /// This overload automatically uses the shape-aware cost function
    /// built from the e-graph's own analysis data.
    ///
    /// \param root  The class to extract from.
    /// \return      The extraction result.
    ExtractionResult extract(ClassId root) {
        // Build a cost function using our shape_aware_cost member method.
        // We pre-compute a snapshot of class analyses for the lambda.
        auto cost_fn = [this](OpId op, const ENode& node) -> double {
            std::vector<double> child_costs;
            for (auto child_cid : node.children) {
                ClassId child_root = find_internal(child_cid);
                // Use a default cost if not yet in the cost table
                child_costs.push_back(1.0);
            }
            return shape_aware_cost(op, node, child_costs);
        };
        return extract(root, cost_fn);
    }

    // ── Cost Table (Bottom-Up DP) ────────────────────────────────────

    /// A cost table entry: maps ClassId to its cheapest cost and best node.
    struct CostEntry {
        double  cost = 1e18;
        NodeId  best_node = NULL_NODE;
    };

    /// Compute the cost table using iterative bottom-up DP.
    /// This fills a memo table mapping ClassId -> (best_cost, best_node).
    ///
    /// Algorithm:
    ///   1. Discover all reachable classes via BFS from root
    ///   2. Reverse BFS order gives leaves-first (bottom-up) traversal
    ///   3. For each class, try each e-node, compute:
    ///        node_cost = op_cost(op, node) + sum(child_class_costs)
    ///   4. Pick the cheapest node for each class
    ///
    /// This is NOT naive recursion — it uses topological ordering
    /// to avoid stack overflow and ensure each class is processed
    /// only after all its children.
    ///
    /// \param root     The root class to compute costs for.
    /// \param op_cost  Cost function for operations.
    /// \return         Map from ClassId to CostEntry.
    std::unordered_map<ClassId, CostEntry> compute_cost_table(
        ClassId root,
        std::function<double(OpId, const ENode&)> op_cost
    ) const {
        root = find_internal(root);

        // Step 1: BFS to discover all reachable classes
        std::vector<ClassId> topo_order;
        std::unordered_set<ClassId> visited;
        std::deque<ClassId> work_queue;
        work_queue.push_back(root);
        visited.insert(root);

        while (!work_queue.empty()) {
            ClassId cid = work_queue.front();
            work_queue.pop_front();
            topo_order.push_back(cid);

            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            const auto& cls = classes_[cid];

            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                for (auto child_cid : nodes_[nid].children) {
                    ClassId child_root = find_internal(child_cid);
                    if (child_root >= 0 && !visited.count(child_root)) {
                        visited.insert(child_root);
                        work_queue.push_back(child_root);
                    }
                }
            }
        }

        // Step 2: Reverse for leaves-first ordering
        std::reverse(topo_order.begin(), topo_order.end());

        // Step 3: Iterative bottom-up DP
        std::unordered_map<ClassId, CostEntry> cost_table;

        for (ClassId cid : topo_order) {
            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            const auto& cls = classes_[cid];
            if (cls.nodes.empty()) continue;

            double best_cost = 1e18;
            NodeId best_node = NULL_NODE;

            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                const auto& node = nodes_[nid];
                double c = op_cost(node.op, node);

                // Add child class costs
                for (auto child_cid : node.children) {
                    ClassId child_root = find_internal(child_cid);
                    auto it = cost_table.find(child_root);
                    if (it != cost_table.end()) {
                        c += it->second.cost;
                    } else {
                        c += 1.0;  // Default cost for unvisited
                    }
                }

                if (c < best_cost) {
                    best_cost = c;
                    best_node = nid;
                }
            }

            cost_table[cid] = CostEntry{best_cost, best_node};
        }

        return cost_table;
    }

    // ── Shape-Aware Dynamic Cost Model ───────────────────────────────

    /// Compute a shape-aware cost for an operation given its node and
    /// the costs of its children's classes.
    ///
    /// Costs depend on tensor shape (number of elements), not just op type:
    ///   - MatMul: M*N*K cost
    ///   - ReduceSum: num_elements
    ///   - Elementwise: num_elements
    ///   - Fused ops: discount factor (cheaper than separate ops)
    ///
    /// \param op            The operation type.
    /// \param node          The e-node being costed.
    /// \param child_costs   Costs of the child classes (already computed).
    /// \return              The estimated cost.
    double shape_aware_cost(
        OpId op,
        const ENode& node,
        const std::vector<double>& child_costs
    ) const {
        // Determine element count from node's analysis/shape info
        int64_t num_elements = -1;
        int64_t bpe = 2;  // default bytes per element

        // Try to get shape from children's class analysis
        if (!node.children.empty()) {
            for (auto child_cid : node.children) {
                ClassId root = find_internal(child_cid);
                if (root >= 0 && root < static_cast<ClassId>(classes_.size())) {
                    const auto& analysis = classes_[root].analysis;
                    if (!analysis.shape.is_unknown()) {
                        int64_t ne = analysis.shape.num_elements();
                        if (ne > 0) {
                            num_elements = ne;
                            bpe = analysis.bytes_per_element();
                            break;
                        }
                    }
                }
            }
        }

        // Default if no shape info available
        const double default_MN = 1024.0 * 1024.0;
        double elements = (num_elements > 0) ? static_cast<double>(num_elements) : default_MN;

        // Sum of child costs
        double child_sum = 0.0;
        for (double c : child_costs) child_sum += c;

        // Fused op discount: fused ops are cheaper than separate ops
        // because they avoid intermediate HBM traffic
        constexpr double FUSED_DISCOUNT = 0.7;

        switch (op) {
            case OpId::SYMBOL:
                return elements * bpe;

            case OpId::CONSTANT:
                return 1.0;

            case OpId::ADD:
            case OpId::SUB:
            case OpId::MUL:
            case OpId::DIV:
                // Elementwise: cost proportional to num_elements
                // Read 2 inputs + write 1 output = 3x traffic
                return 3.0 * elements * bpe + child_sum;

            case OpId::NEG:
            case OpId::RELU:
            case OpId::GELU:
            case OpId::SIGMOID:
            case OpId::EXP:
            case OpId::LOG:
            case OpId::SQRT:
            case OpId::RECIPROCAL:
            case OpId::DROPOUT:
            case OpId::TRANSPOSE:
                // Unary: read + write = 2x traffic
                return 2.0 * elements * bpe + child_sum;

            case OpId::MATMUL: {
                // C[M,N] = A[M,K] @ B[K,N]
                // FLOPs = 2*M*K*N
                int64_t M = -1, K = -1, N = -1;
                if (node.children.size() >= 2) {
                    ClassId a_root = find_internal(node.children[0]);
                    ClassId b_root = find_internal(node.children[1]);
                    if (a_root >= 0 && a_root < static_cast<ClassId>(classes_.size()) &&
                        b_root >= 0 && b_root < static_cast<ClassId>(classes_.size())) {
                        auto& sa = classes_[a_root].analysis.shape;
                        auto& sb = classes_[b_root].analysis.shape;
                        M = (sa.ndim() >= 2) ? sa[sa.ndim()-2] : -1;
                        K = (sa.ndim() >= 2) ? sa[sa.ndim()-1] : -1;
                        N = (sb.ndim() >= 2) ? sb[sb.ndim()-1] : -1;
                    }
                }
                if (M > 0 && N > 0 && K > 0) {
                    return 2.0 * M * K * N * bpe + child_sum;
                }
                return 2.0 * default_MN * 512.0 * bpe + child_sum;
            }

            case OpId::REDUCE_SUM:
            case OpId::REDUCE_MAX:
            case OpId::REDUCE_MEAN:
                // Reduction: read all, write reduced
                return 1.5 * elements * bpe + child_sum;

            case OpId::SOFTMAX:
                // Unfused: multiple passes with intermediate materialization
                return 6.0 * elements * bpe + child_sum;

            case OpId::FUSED_MATMUL_RELU:
            case OpId::FUSED_MATMUL_ADD:
                // Fused: discount factor applied
                return (2.0 * default_MN * 512.0 * bpe + child_sum) * FUSED_DISCOUNT;

            case OpId::FUSED_MATMUL_ADD_RELU:
                return (2.0 * default_MN * 512.0 * bpe + child_sum) * 0.55;

            case OpId::FUSED_GEMM:
                return (2.0 * default_MN * 512.0 * bpe + child_sum) * 0.65;

            case OpId::FUSED_SOFTMAX:
                return 4.0 * elements * bpe * FUSED_DISCOUNT + child_sum;

            case OpId::FUSED_LAYERNORM:
            case OpId::FUSED_RMSNORM:
                return 3.0 * elements * bpe * FUSED_DISCOUNT + child_sum;

            case OpId::FUSED_ADD_LN:
                return 4.0 * elements * bpe * FUSED_DISCOUNT + child_sum;

            case OpId::FUSED_MHA:
                return 8.0 * elements * bpe * FUSED_DISCOUNT + child_sum;

            case OpId::LAYERNORM:
            case OpId::RMSNORM:
                return 8.0 * elements * bpe + child_sum;

            case OpId::TILE:
            case OpId::UNTILE:
                return 0.1 * elements * bpe + child_sum;

            case OpId::RESHAPE:
            case OpId::BROADCAST:
            case OpId::IDENTITY:
                return 0.01 * elements * bpe + child_sum;
        }

        return elements * bpe + child_sum;
    }

    // ── Query / Iteration ────────────────────────────────────────────

    /// Iterate over all e-nodes in a class.
    std::vector<NodeId> class_nodes(ClassId cid) const {
        ClassId root = find_internal(cid);
        if (root < 0 || root >= static_cast<ClassId>(classes_.size())) return {};
        return classes_[root].nodes;  // Returns a copy for safety
    }

    /// Iterate over all e-classes.
    const std::vector<EClass>& all_classes() const { return classes_; }

    /// Iterate over all e-nodes.
    const std::vector<ENode>& all_nodes() const { return nodes_; }

    /// Get a specific node.
    ENode node(NodeId nid) const { return nodes_[nid]; }

    /// Get the class of a node.
    ClassId node_class(NodeId nid) const { return find_internal(node_classes_[nid]); }

    /// Get the analysis data for a class.
    ClassAnalysis& class_analysis(ClassId cid) {
        return classes_[find(cid)].analysis;
    }
    const ClassAnalysis& class_analysis(ClassId cid) const {
        return classes_[find_internal(cid)].analysis;
    }

    /// Number of e-classes (including dead ones).
    size_t num_classes() const { return classes_.size(); }

    /// Number of live e-classes.
    size_t num_live_classes() const {
        size_t count = 0;
        for (ClassId i = 0; i < static_cast<ClassId>(classes_.size()); ++i) {
            if (!classes_[i].nodes.empty()) count++;
        }
        return count;
    }

    /// Number of e-nodes.
    size_t num_nodes() const { return nodes_.size(); }

    // ── Public Shape/DType Utilities ─────────────────────────────────
    // These are needed externally by cost functions and validation code.

    /// Broadcast two shapes (NumPy-style broadcasting rules).
    static TensorShape broadcast_shapes(const TensorShape& a, const TensorShape& b) {
        if (a.is_unknown()) return b;
        if (b.is_unknown()) return a;

        size_t max_ndim = std::max(a.ndim(), b.ndim());
        std::vector<int64_t> result(max_ndim);

        for (size_t i = 0; i < max_ndim; ++i) {
            int64_t da = (i + a.ndim() >= max_ndim) ? a[a.ndim() - max_ndim + i] : 1;
            int64_t db = (i + b.ndim() >= max_ndim) ? b[b.ndim() - max_ndim + i] : 1;

            if (da == db) {
                result[i] = da;
            } else if (da == 1) {
                result[i] = db;
            } else if (db == 1) {
                result[i] = da;
            } else {
                result[i] = -1;  // Incompatible shapes
            }
        }

        return TensorShape(std::move(result));
    }

    /// Promote two dtypes to the wider one.
    static DType promote_dtypes(DType a, DType b) {
        if (a == DType::UNKNOWN) return b;
        if (b == DType::UNKNOWN) return a;
        // Promotion order: FP64 > FP32 > BF16 > FP16 > INT8 > INT4
        static const DType order[] = {
            DType::FP64, DType::FP32, DType::BF16, DType::FP16, DType::INT8, DType::INT4
        };
        int ia = -1, ib = -1;
        for (int i = 0; i < 6; ++i) {
            if (order[i] == a) ia = i;
            if (order[i] == b) ib = i;
        }
        if (ia < 0 || ib < 0) return DType::UNKNOWN;
        return order[std::min(ia, ib)];  // Return wider type
    }

    /// Transpose a shape (reverse dimensions).
    static TensorShape transpose_shape(const TensorShape& s) {
        if (s.is_unknown() || s.ndim() < 2) return s;
        std::vector<int64_t> rev(s.dims.rbegin(), s.dims.rend());
        return TensorShape(std::move(rev));
    }

    /// Reduce a shape along an axis.
    static TensorShape reduce_shape(const TensorShape& s, int64_t axis) {
        if (s.is_unknown()) return s;
        if (axis < 0 || axis >= static_cast<int64_t>(s.ndim())) return s;
        auto result = s.dims;
        result[axis] = 1;
        return TensorShape(std::move(result));
    }

    // ── Debug / Visualization ────────────────────────────────────────

    std::string to_string() const {
        std::ostringstream oss;
        oss << "EGraph{classes=" << num_live_classes()
            << ", nodes=" << num_nodes() << "}\n";
        for (ClassId i = 0; i < static_cast<ClassId>(classes_.size()); ++i) {
            if (classes_[i].nodes.empty()) continue;
            oss << "  Class " << i << " [" << classes_[i].analysis.to_string() << "]: {";
            for (size_t j = 0; j < classes_[i].nodes.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << "n" << classes_[i].nodes[j]
                    << "=" << nodes_[classes_[i].nodes[j]].to_string();
            }
            oss << "}\n";
        }
        return oss.str();
    }

private:
    std::vector<ENode>          nodes_;         // All e-nodes
    std::vector<ClassId>        node_classes_;  // NodeId -> ClassId
    std::vector<EClass>         classes_;       // All e-classes
    std::vector<ClassId>        parent_;        // Union-find parent
    std::vector<int>            rank_;          // Union-find rank

    // Hash map for congruence: canonical ENode -> NodeId
    std::unordered_map<ENode, NodeId, ENode::Hash> node_hash_;

    // Worklist for incremental congruence repair
    std::vector<ClassId> pending_worklist_;

    // Operator index: OpId -> list of NodeIds (Improvement 1)
    std::unordered_map<OpId, std::vector<NodeId>, OpIdHash> op_index_;

    // Dirty class tracking (Improvement 2)
    std::unordered_set<ClassId> dirty_classes_;

    // Saturation config stored for beam_prune access
    SaturationConfig config_;

    // Early stopping state (Improvement 7)
    double best_cost_ = 1e18;
    int stagnation_count_ = 0;

    // Saturation statistics (Improvement 9)
    SaturationStats stats_;

    /// Create a new e-class and return its ID.
    ClassId make_class() {
        ClassId id = static_cast<ClassId>(classes_.size());
        classes_.push_back({id, {}, {}});
        parent_.push_back(id);
        rank_.push_back(0);
        return id;
    }

    /// Internal find that doesn't do bounds checking.
    ClassId find_internal(ClassId cid) const {
        if (cid < 0 || cid >= static_cast<ClassId>(parent_.size())) return cid;
        ClassId current = cid;
        while (parent_[current] != current) {
            current = parent_[current];
        }
        return current;
    }

    /// Evaluate the total cost of an e-class (for early stopping).
    double evaluate_class_cost(
        ClassId root,
        const std::function<double(OpId, const ENode&)>& op_cost
    ) const {
        root = find_internal(root);
        // Use the same DP extraction algorithm but only return the cost
        std::vector<ClassId> topo_order;
        std::unordered_set<ClassId> visited;
        std::deque<ClassId> work_queue;
        work_queue.push_back(root);
        visited.insert(root);

        while (!work_queue.empty()) {
            ClassId cid = work_queue.front();
            work_queue.pop_front();
            topo_order.push_back(cid);
            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            for (NodeId nid : classes_[cid].nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                for (auto child_cid : nodes_[nid].children) {
                    ClassId child_root = find_internal(child_cid);
                    if (child_root >= 0 && !visited.count(child_root)) {
                        visited.insert(child_root);
                        work_queue.push_back(child_root);
                    }
                }
            }
        }
        std::reverse(topo_order.begin(), topo_order.end());

        std::unordered_map<ClassId, double> class_cost;
        for (ClassId cid : topo_order) {
            if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
            const auto& cls = classes_[cid];
            if (cls.nodes.empty()) continue;
            double best = 1e18;
            for (NodeId nid : cls.nodes) {
                if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                const auto& node = nodes_[nid];
                double c = op_cost(node.op, node);
                for (auto child_cid : node.children) {
                    ClassId child_root = find_internal(child_cid);
                    auto it = class_cost.find(child_root);
                    if (it != class_cost.end()) c += it->second;
                    else c += 1.0;
                }
                if (c < best) best = c;
            }
            class_cost[cid] = best;
        }
        auto it = class_cost.find(root);
        return (it != class_cost.end()) ? it->second : 1e18;
    }

    /// Merge two class analyses, keeping the most informative information.
    void merge_analysis(ClassAnalysis& dst, const ClassAnalysis& src) {
        // Shape: prefer the more informative (fewer unknowns)
        if (dst.shape.is_unknown() && !src.shape.is_unknown()) {
            dst.shape = src.shape;
        } else if (!dst.shape.is_unknown() && !src.shape.is_unknown()) {
            // If both have shapes, they should be compatible; keep the larger
            if (src.shape.ndim() > dst.shape.ndim()) {
                dst.shape = src.shape;
            }
        }

        // Dtype: prefer known over unknown
        if (dst.dtype == DType::UNKNOWN && src.dtype != DType::UNKNOWN) {
            dst.dtype = src.dtype;
        }

        // Layout: prefer known over unknown
        if (dst.layout == Layout::UNKNOWN && src.layout != Layout::UNKNOWN) {
            dst.layout = src.layout;
        }

        // FLOPs: take the maximum (either path could be used)
        dst.estimated_flops = std::max(dst.estimated_flops, src.estimated_flops);

        // TC compatibility: true if either path is compatible
        dst.tc_compatible = dst.tc_compatible || src.tc_compatible;

        // Reduction: true if either path has reductions
        dst.is_reduction = dst.is_reduction || src.is_reduction;

        // FP safety: both must be safe for the result to be safe
        dst.floating_point_safe = dst.floating_point_safe && src.floating_point_safe;

        // Aliasing: true if either path aliases
        dst.aliases_input = dst.aliases_input || src.aliases_input;

        // Residency: prefer the lower-latency location
        if (dst.residency == Residency::UNKNOWN && src.residency != Residency::UNKNOWN) {
            dst.residency = src.residency;
        } else if (dst.residency == Residency::HBM && src.residency != Residency::HBM) {
            dst.residency = src.residency;
        }

        // Sharding: prefer non-replicated
        if (dst.sharding.is_replicated() && !src.sharding.is_replicated()) {
            dst.sharding = src.sharding;
        }
    }

    /// Propagate analysis data through the e-graph bottom-up.
    /// Uses dirty_classes_ when available for incremental propagation,
    /// otherwise falls back to full recomputation.
    void propagate_analysis() {
        if (!dirty_classes_.empty()) {
            // Incremental: only recompute dirty classes
            for (ClassId cid : dirty_classes_) {
                if (cid < 0 || cid >= static_cast<ClassId>(classes_.size())) continue;
                if (classes_[cid].nodes.empty()) continue;
                for (NodeId nid : classes_[cid].nodes) {
                    if (nid < 0 || nid >= static_cast<NodeId>(nodes_.size())) continue;
                    ClassAnalysis node_analysis = compute_node_analysis(nid);
                    merge_analysis(classes_[cid].analysis, node_analysis);
                }
            }
        } else {
            // Full recompute (backward compat path)
            for (ClassId cid = 0; cid < static_cast<ClassId>(classes_.size()); ++cid) {
                if (classes_[cid].nodes.empty()) continue;
                for (NodeId nid : classes_[cid].nodes) {
                    ClassAnalysis node_analysis = compute_node_analysis(nid);
                    merge_analysis(classes_[cid].analysis, node_analysis);
                }
            }
        }
    }

};


// ─────────────────────────────────────────────────────────────────────────
// Standard Tensor Algebra Rewrite Rules
// ─────────────────────────────────────────────────────────────────────────

/// Create all built-in rewrite rules for tensor expression optimization.
std::vector<RewriteRule> standard_tensor_rewrite_rules();

/// Create rewrite rules specifically for transformer/attention optimization.
std::vector<RewriteRule> transformer_rewrite_rules();

/// Create fusion discovery rules (find opportunities for operator fusion).
std::vector<RewriteRule> fusion_rewrite_rules();

/// Create tiling discovery rules (explore tiling decomposition alternatives).
std::vector<RewriteRule> tiling_rewrite_rules();

/// Create normalization-specific rules (LayerNorm, RMSNorm, etc.).
std::vector<RewriteRule> normalization_rewrite_rules();

/// Create all rewrite rules (aggregated and prioritized).
std::vector<RewriteRule> all_rewrite_rules();

// ─────────────────────────────────────────────────────────────────────────
// Distributivity Gating
// ─────────────────────────────────────────────────────────────────────────

/// Should the distributivity rule be applied given the current e-graph state?
///
/// Distributivity (A * (B + C) == A*B + A*C) is the primary source of
/// e-graph saturation explosion. This gate checks three conditions and
/// returns true only if at least one holds:
///
///   1. Reuse detection: Does at least one of the distributed terms
///      (A*B or A*C) already exist in the graph? If so, distributing
///      enables factorization or common subexpression elimination.
///
///   2. Cost-guided: Would the distributed form improve cost?
///      If distributing a scalar multiply enables fusion with adjacent
///      matmuls, it's worth exploring.
///
///   3. Depth check: Is the expression depth below a threshold?
///      Deep expressions are more likely to cause explosion.
///
/// Returns false if none of these hold, preventing the distributivity
/// rule from firing and exploding the e-graph.
///
/// \param g                The e-graph to check.
/// \param max_depth        Maximum expression depth to allow (default: 10).
/// \param max_graph_size   Maximum graph size to consider (default: 5000).
/// \return                 True if distributivity should be applied.
bool should_apply_distributivity(
    const EGraph& g,
    int max_depth = 10,
    int64_t max_graph_size = 5000
);

// ─────────────────────────────────────────────────────────────────────────
// Cost Functions for Extraction
// ─────────────────────────────────────────────────────────────────────────

/// Create a cost function based on memory traffic (bytes transferred).
/// Uses analysis data when available, falls back to heuristics.
std::function<double(OpId, const ENode&)>
memory_traffic_cost_fn(int64_t bytes_per_element = 2);

/// Create a cost function based on compute cost (FLOPs).
std::function<double(OpId, const ENode&)>
compute_cost_fn(int64_t m = 1024, int64_t n = 1024, int64_t k = 1024);

/// Create a combined cost function (weighted blend of memory and compute).
std::function<double(OpId, const ENode&)>
combined_cost_fn(
    double memory_weight = 0.6,
    double compute_weight = 0.4,
    int64_t bytes_per_element = 2,
    int64_t m = 1024, int64_t n = 1024, int64_t k = 1024
);

/// Create a hardware-aware cost function that considers Tensor Cores,
/// SRAM pressure, and memory hierarchy.
std::function<double(OpId, const ENode&)>
hardware_aware_cost_fn(
    int64_t sram_budget_bytes = 228000,
    double tc_speedup = 4.0,
    int64_t bytes_per_element = 2,
    int64_t m = 1024, int64_t n = 1024, int64_t k = 1024
);

/// Create a shape-aware cost function (Improvement 8).
/// This cost function looks up class analysis data from the EGraph to get
/// actual tensor shapes, FLOPs, TC compatibility etc. instead of using
/// hardcoded M=1024 assumptions.
///
/// \param g                   The e-graph to query for analysis data.
/// \param sram_budget_bytes   SRAM budget for tiling decisions.
/// \param tc_speedup          Tensor Core speedup factor.
std::function<double(OpId, const ENode&)>
shape_aware_cost_fn(
    const EGraph& g,
    int64_t sram_budget_bytes = 228000,
    double tc_speedup = 4.0
);

// ─────────────────────────────────────────────────────────────────────────
// Polyhedral Validation
// ─────────────────────────────────────────────────────────────────────────

/// Dependency kinds for polyhedral validation.
enum class DependencyKind : uint8_t {
    RAW,    // Read-After-Write (true dependency)
    WAR,    // Write-After-Read (anti-dependency)
    WAW,    // Write-After-Write (output dependency)
};

/// A dependency between two e-classes in the original program.
struct Dependency {
    ClassId         source;
    ClassId         target;
    DependencyKind  kind;
    std::vector<int64_t> distance_vector;  // Polyhedral distance vector
};

/// Validate that an extracted expression preserves data dependencies.
/// This is the "polyhedral guardrail" — it rejects any extracted program
/// that would violate the original computation's semantics.
///
/// Checks:
///   - No read-before-write dependencies are violated
///   - All dependency distance vectors remain lexicographically positive
///   - Reduction operations maintain correct ordering
///   - Floating-point reordering is not applied to FP-unsafe expressions
///   - No cyclic dependencies are introduced by rewrites
bool validate_extracted_program(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps
);

/// Backward-compatible overload.
bool validate_extracted_program(
    const ExtractionResult& extracted,
    const std::vector<std::pair<ClassId, ClassId>>& dep_pairs
);

// ─────────────────────────────────────────────────────────────────────────
// Enhanced Polyhedral Validation (Improvement 10)
// ─────────────────────────────────────────────────────────────────────────

/// Validation result with detailed diagnostic information.
struct ValidationResult {
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void add_error(const std::string& msg) {
        is_valid = false;
        errors.push_back(msg);
    }

    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ValidationResult{valid=" << (is_valid ? "true" : "false");
        if (!errors.empty()) {
            oss << ", errors=[";
            for (size_t i = 0; i < errors.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "\"" << errors[i] << "\"";
            }
            oss << "]";
        }
        if (!warnings.empty()) {
            oss << ", warnings=[";
            for (size_t i = 0; i < warnings.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "\"" << warnings[i] << "\"";
            }
            oss << "]";
        }
        oss << "}";
        return oss.str();
    }
};

/// Validate that the extracted nodes form a DAG (no cycles).
/// Cyclic equivalences in the e-graph can sometimes produce
/// circular extraction results, which are unsound.
///
/// \param extracted  The extraction result to validate.
/// \return           ValidationResult with any cycle errors.
ValidationResult validate_no_cycle(const ExtractionResult& extracted);

/// Validate that floating-point-unsafe reorders aren't applied
/// to FP-sensitive operations. For example, reordering additions
/// around a MUL or reordering reductions across non-associative
/// FP operations is semantically incorrect.
///
/// \param extracted   The extraction result to validate.
/// \param class_analyses  Map from ClassId to ClassAnalysis for checking
///                        FP safety of intermediate results.
/// \return            ValidationResult with any FP safety errors.
ValidationResult validate_fp_safety(
    const ExtractionResult& extracted,
    const std::unordered_map<ClassId, ClassAnalysis>& class_analyses
);

/// Overload that uses the e-graph directly for analysis lookup.
ValidationResult validate_fp_safety(
    const ExtractionResult& extracted,
    const EGraph& g
);

/// Validate that reduction operations maintain correct ordering.
/// Reductions over non-associative operations (e.g. FP sum) must
/// not be reordered, split, or merged in ways that change the
/// accumulation order unless the ClassAnalysis explicitly marks
/// the expression as floating_point_safe.
///
/// \param extracted  The extraction result to validate.
/// \param original_deps  Original dependencies to check against.
/// \return           ValidationResult with any reduction ordering errors.
ValidationResult validate_reduction_ordering(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps
);

/// Comprehensive validation: runs all validation checks and returns
/// the combined result.
ValidationResult validate_extracted_program_full(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps,
    const EGraph& g
);

} // namespace symplex::optimizer::egraph
