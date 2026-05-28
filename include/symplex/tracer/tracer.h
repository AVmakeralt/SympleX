// SympleX – Polyhedral Tensor Superoptimizer
// Python Proxy Tensor Tracer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// The tracer replaces real tensors with Proxy Tensors that record ops.
// Python execution becomes a graph-writing script pretending to be math:
//   a + b → emits Add(a, b)
//   a * b → emits Mul(a, b)
//   sin(a) → emits Sin(a)
//
// This is the SSA DAG that feeds into the SympleX optimizer.

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
#include <numeric>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

namespace symplex::tracer {

// ─────────────────────────────────────────────────────────────────────────
// Operation Types
// ─────────────────────────────────────────────────────────────────────────

/// TraceOp: operation types for the trace graph.
enum class TraceOp : uint8_t {
    ADD, MUL, SUB, DIV, NEG,
    MATMUL, TRANSPOSE, RESHAPE, BROADCAST,
    REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN,
    RELU, GELU, SIGMOID, SOFTMAX, LAYERNORM, RMSNORM,
    EXP, LOG, SQRT, RECIPROCAL,
    DROPOUT,  // identity during inference
    SELECT,   // conditional: select(cond, a, b)
    CONST,    // constant
    PARAM,    // named parameter
};

/// Convert TraceOp to human-readable string.
inline std::string trace_op_to_string(TraceOp op) {
    switch (op) {
        case TraceOp::ADD:          return "Add";
        case TraceOp::MUL:          return "Mul";
        case TraceOp::SUB:          return "Sub";
        case TraceOp::DIV:          return "Div";
        case TraceOp::NEG:          return "Neg";
        case TraceOp::MATMUL:       return "MatMul";
        case TraceOp::TRANSPOSE:    return "Transpose";
        case TraceOp::RESHAPE:      return "Reshape";
        case TraceOp::BROADCAST:    return "Broadcast";
        case TraceOp::REDUCE_SUM:   return "ReduceSum";
        case TraceOp::REDUCE_MAX:   return "ReduceMax";
        case TraceOp::REDUCE_MEAN:  return "ReduceMean";
        case TraceOp::RELU:         return "ReLU";
        case TraceOp::GELU:         return "GELU";
        case TraceOp::SIGMOID:      return "Sigmoid";
        case TraceOp::SOFTMAX:      return "Softmax";
        case TraceOp::LAYERNORM:    return "LayerNorm";
        case TraceOp::RMSNORM:      return "RMSNorm";
        case TraceOp::EXP:          return "Exp";
        case TraceOp::LOG:          return "Log";
        case TraceOp::SQRT:         return "Sqrt";
        case TraceOp::RECIPROCAL:   return "Reciprocal";
        case TraceOp::DROPOUT:      return "Dropout";
        case TraceOp::SELECT:       return "Select";
        case TraceOp::CONST:        return "Const";
        case TraceOp::PARAM:        return "Param";
    }
    return "Unknown";
}

/// How many inputs does each op expect?
inline int trace_op_arity(TraceOp op) {
    switch (op) {
        case TraceOp::CONST:
        case TraceOp::PARAM:
            return 0;
        case TraceOp::NEG:
        case TraceOp::TRANSPOSE:
        case TraceOp::RELU:
        case TraceOp::GELU:
        case TraceOp::SIGMOID:
        case TraceOp::SOFTMAX:
        case TraceOp::LAYERNORM:
        case TraceOp::RMSNORM:
        case TraceOp::DROPOUT:
        case TraceOp::EXP:
        case TraceOp::LOG:
        case TraceOp::SQRT:
        case TraceOp::RECIPROCAL:
        case TraceOp::REDUCE_SUM:
        case TraceOp::REDUCE_MAX:
        case TraceOp::REDUCE_MEAN:
        case TraceOp::RESHAPE:
        case TraceOp::BROADCAST:
            return 1;
        case TraceOp::ADD:
        case TraceOp::MUL:
        case TraceOp::SUB:
        case TraceOp::DIV:
        case TraceOp::MATMUL:
            return 2;
        case TraceOp::SELECT:
            return 3;
    }
    return 0;
}

/// Is this op elementwise (for shape inference)?
inline bool trace_op_is_elementwise(TraceOp op) {
    switch (op) {
        case TraceOp::ADD:
        case TraceOp::MUL:
        case TraceOp::SUB:
        case TraceOp::DIV:
        case TraceOp::NEG:
        case TraceOp::RELU:
        case TraceOp::GELU:
        case TraceOp::SIGMOID:
        case TraceOp::EXP:
        case TraceOp::LOG:
        case TraceOp::SQRT:
        case TraceOp::RECIPROCAL:
            return true;
        default:
            return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Trace Node
// ─────────────────────────────────────────────────────────────────────────

/// TraceNode: a single operation in the trace graph.
struct TraceNode {
    int64_t                 id;          // Unique node ID
    TraceOp                 op;          // Operation type
    std::vector<int64_t>    inputs;      // Input node IDs
    double                  float_value; // For CONST nodes
    int64_t                 int_value;   // For CONST nodes (integer)
    std::string             name;        // For PARAM nodes
    int64_t                 axis = -1;   // For reductions, softmax, etc.
    std::vector<int64_t>    shape;       // Tensor shape (may be symbolic, -1 = unknown)
    int                     dtype = 0;   // 0=fp32, 1=fp16, 2=bf16, 3=int8

    TraceNode()
        : id(-1), op(TraceOp::CONST), float_value(0.0), int_value(0) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "%" << id << " = " << trace_op_to_string(op);

        if (op == TraceOp::PARAM) {
            oss << "(" << name << ")";
        } else if (op == TraceOp::CONST) {
            if (float_value != 0.0 && int_value == 0) {
                oss << "(" << float_value << ")";
            } else {
                oss << "(" << int_value << ")";
            }
        }

        if (!inputs.empty()) {
            oss << "(";
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "%" << inputs[i];
            }
            oss << ")";
        }

        if (axis >= 0) {
            oss << " axis=" << axis;
        }

        // Shape annotation
        if (!shape.empty()) {
            oss << " [";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) oss << ",";
                if (shape[i] < 0) oss << "?";
                else oss << shape[i];
            }
            oss << "]";
        }

        // Dtype annotation
        switch (dtype) {
            case 0: oss << " :fp32"; break;
            case 1: oss << " :fp16"; break;
            case 2: oss << " :bf16"; break;
            case 3: oss << " :int8"; break;
        }

        return oss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────
// TensorShape for the tracer
// ─────────────────────────────────────────────────────────────────────────

/// TraceShape: lightweight shape representation for the tracer.
struct TraceShape {
    std::vector<int64_t> dims;

    TraceShape() = default;
    TraceShape(std::initializer_list<int64_t> d) : dims(d) {}
    explicit TraceShape(std::vector<int64_t> d) : dims(std::move(d)) {}

    size_t ndim() const { return dims.size(); }
    int64_t operator[](size_t i) const { return i < dims.size() ? dims[i] : -1; }

    bool is_unknown() const { return dims.empty(); }

    int64_t num_elements() const {
        int64_t total = 1;
        for (auto d : dims) {
            if (d < 0) return -1;
            total *= d;
        }
        return total;
    }

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

    bool operator==(const TraceShape& o) const { return dims == o.dims; }
    bool operator!=(const TraceShape& o) const { return dims != o.dims; }
};

// ─────────────────────────────────────────────────────────────────────────
// Serialized Trace Data (bridge format for tracer → IR conversion)
// ─────────────────────────────────────────────────────────────────────────

/// SerializedTraceData: intermediate format for the tracer→IR bridge.
/// Contains all trace graph data in a serializable format that
/// SympleXIR::from_trace_graph() can consume after type conversion.
///
/// Layout convention: all PARAM nodes are serialized first (as params),
/// then all other nodes (as ops).  Operand IDs are remapped so that
/// params occupy IDs 0..P-1 and ops occupy IDs P..P+O-1.
struct SerializedTraceData {
    /// Params: (name, shape, dtype_as_int)
    /// dtype_as_int uses tracer convention: 0=fp32, 1=fp16, 2=bf16, 3=int8
    std::vector<std::tuple<std::string, std::vector<int64_t>, int>> params;

    /// Ops: (shape, trace_op_as_int, operands, float_value, name, axis, dtype_as_int)
    /// trace_op_as_int is the TraceOp enum value (cast to int).
    /// operands use remapped sequential IDs (params first, then ops).
    /// float_value is meaningful for CONST nodes.
    /// axis is meaningful for reductions, softmax, etc.
    /// dtype_as_int uses tracer convention: 0=fp32, 1=fp16, 2=bf16, 3=int8
    std::vector<std::tuple<std::vector<int64_t>, int, std::vector<int64_t>,
                           double, std::string, int64_t, int>> ops;

    /// Total number of original trace nodes (for validation).
    int64_t num_nodes = 0;
};

// ─────────────────────────────────────────────────────────────────────────
// Node Interning Key
// ─────────────────────────────────────────────────────────────────────────

/// NodeKey: used for interning — identical ops reuse node IDs.
struct NodeKey {
    TraceOp                 op;
    std::vector<int64_t>    inputs;
    double                  float_value;
    int64_t                 int_value;
    std::string             name;
    int64_t                 axis;
    std::vector<int64_t>    shape;    // Included for reshape/broadcast
    int                     dtype;

    bool operator==(const NodeKey& other) const {
        return op == other.op
            && inputs == other.inputs
            && float_value == other.float_value
            && int_value == other.int_value
            && name == other.name
            && axis == other.axis
            && shape == other.shape
            && dtype == other.dtype;
    }

    struct Hash {
        size_t operator()(const NodeKey& k) const {
            size_t h = static_cast<size_t>(k.op);
            h ^= std::hash<int64_t>{}(k.int_value) + 0x9e3779b9 + (h << 6) + (h >> 2);
            // Hash float_value by bit-casting to uint64_t
            uint64_t fv_bits = 0;
            std::memcpy(&fv_bits, &k.float_value, sizeof(double));
            h ^= std::hash<uint64_t>{}(fv_bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(k.name) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(k.axis) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (auto i : k.inputs) {
                h ^= std::hash<int64_t>{}(i) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            for (auto s : k.shape) {
                h ^= std::hash<int64_t>{}(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
};

// ─────────────────────────────────────────────────────────────────────────
// Trace Graph
// ─────────────────────────────────────────────────────────────────────────

/// TraceGraph: the computation graph built by the tracer.
///
/// The tracer ONLY traces, never executes. Pure graph construction
/// as side-effect. Python execution becomes a graph-writing script
/// pretending to be math.
class TraceGraph {
public:
    TraceGraph() = default;

    // ── Leaf Nodes ──────────────────────────────────────────────────

    /// Add a parameter (input tensor) to the graph.
    int64_t param(const std::string& name, const std::vector<int64_t>& shape, int dtype = 0);

    /// Add a floating-point constant to the graph.
    int64_t constant(double value, int dtype = 0);

    /// Add an integer constant to the graph.
    int64_t constant_int(int64_t value, int dtype = 3);

    // ── Unary Ops ───────────────────────────────────────────────────

    int64_t unary_op(TraceOp op, int64_t input, int64_t axis = -1);
    int64_t relu(int64_t x);
    int64_t gelu(int64_t x);
    int64_t sigmoid(int64_t x);
    int64_t exp(int64_t x);
    int64_t log(int64_t x);
    int64_t sqrt(int64_t x);
    int64_t negate(int64_t x);
    int64_t softmax(int64_t x, int64_t axis);
    int64_t layernorm(int64_t x);
    int64_t rmsnorm(int64_t x);
    int64_t dropout(int64_t x);
    int64_t reduce_sum(int64_t x, int64_t axis);
    int64_t reduce_max(int64_t x, int64_t axis);
    int64_t reduce_mean(int64_t x, int64_t axis);
    int64_t transpose(int64_t x);
    int64_t reshape(int64_t x, const std::vector<int64_t>& shape);
    int64_t broadcast(int64_t x, const std::vector<int64_t>& shape);
    int64_t reciprocal(int64_t x);

    // ── Binary Ops ──────────────────────────────────────────────────

    int64_t binary_op(TraceOp op, int64_t left, int64_t right);
    int64_t add(int64_t a, int64_t b);
    int64_t mul(int64_t a, int64_t b);
    int64_t sub(int64_t a, int64_t b);
    int64_t div(int64_t a, int64_t b);
    int64_t matmul(int64_t a, int64_t b);

    // ── Ternary Ops ─────────────────────────────────────────────────

    int64_t select(int64_t cond, int64_t a, int64_t b);

    // ── Node Interning ──────────────────────────────────────────────

    /// Intern a node: identical ops → reuse node ID.
    /// This is CRITICAL — no duplicate nodes for identical ops.
    int64_t intern_node(const TraceNode& node);

    // ── Access ──────────────────────────────────────────────────────

    const TraceNode& node(int64_t id) const;
    const std::vector<TraceNode>& nodes() const { return nodes_; }
    int64_t num_nodes() const { return static_cast<int64_t>(nodes_.size()); }

    /// Get the shape of a node.
    const std::vector<int64_t>& shape_of(int64_t id) const;

    // ── Shape Inference ─────────────────────────────────────────────

    /// Propagate shapes through the graph bottom-up.
    /// Elementwise ops: broadcast shape.
    /// MatMul: MxK * KxN → MxN.
    /// Reductions: remove axis dimension.
    void infer_shapes();

    // ── Validation ──────────────────────────────────────────────────

    /// Check the graph is well-formed (SSA, no cycles, shapes consistent).
    bool validate() const;

    // ── Conversion ──────────────────────────────────────────────────

    /// Convert to SympleX IR / e-graph (bridge to optimizer).
    /// Returns (egraph, root_class_id) pair.
    /// Forward declaration — actual implementation requires egraph.h
    std::pair<void*, int64_t> to_egraph() const;

    // ── Printing ────────────────────────────────────────────────────

    std::string to_string() const;

private:
    std::vector<TraceNode> nodes_;
    int64_t next_id_ = 0;

    /// Interning: hash (op, sorted_inputs, attributes) → existing node ID
    std::unordered_map<NodeKey, int64_t, NodeKey::Hash> intern_map_;

    /// Create a new node (always fresh ID, no interning).
    int64_t create_node(TraceOp op, std::vector<int64_t> inputs,
                        double float_val = 0.0, int64_t int_val = 0,
                        std::string name = "", int64_t axis = -1,
                        std::vector<int64_t> shape = {}, int dtype = 0);

    /// Compute broadcast shape from two input shapes.
    static std::vector<int64_t> broadcast_shapes(
        const std::vector<int64_t>& a, const std::vector<int64_t>& b);

    /// Compute matmul output shape from two input shapes.
    static std::vector<int64_t> matmul_shapes(
        const std::vector<int64_t>& a, const std::vector<int64_t>& b);

    /// Compute reduction output shape.
    static std::vector<int64_t> reduce_shape(
        const std::vector<int64_t>& input_shape, int64_t axis);
};

// ─────────────────────────────────────────────────────────────────────────
// Thread-Local Trace Context
// ─────────────────────────────────────────────────────────────────────────

/// Thread-local current graph for implicit tracing.
/// YES, thread_local. Otherwise you'll cry in parallel execution.
extern thread_local TraceGraph* CURRENT_GRAPH;

/// RAII trace context: sets CURRENT_GRAPH for the duration.
class TraceContext {
public:
    TraceContext() : saved_(CURRENT_GRAPH) {
        graph_ = std::make_unique<TraceGraph>();
        CURRENT_GRAPH = graph_.get();
    }
    ~TraceContext() { CURRENT_GRAPH = saved_; }

    TraceGraph& graph() { return *graph_; }
    const TraceGraph& graph() const { return *graph_; }

    // Non-copyable, non-movable
    TraceContext(const TraceContext&) = delete;
    TraceContext& operator=(const TraceContext&) = delete;

private:
    TraceGraph* saved_;
    std::unique_ptr<TraceGraph> graph_;
};

} // namespace symplex::tracer
