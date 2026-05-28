// SympleX – Polyhedral Tensor Superoptimizer
// Python Proxy Tensor Tracer — Implementation
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/tracer/tracer.h"

namespace symplex::tracer {

// ─────────────────────────────────────────────────────────────────────────
// Thread-local current graph
// ─────────────────────────────────────────────────────────────────────────

thread_local TraceGraph* CURRENT_GRAPH = nullptr;

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Internal Helpers
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::create_node(TraceOp op, std::vector<int64_t> inputs,
                                 double float_val, int64_t int_val,
                                 std::string name, int64_t axis,
                                 std::vector<int64_t> shape, int dtype) {
    int64_t id = next_id_++;
    TraceNode node;
    node.id = id;
    node.op = op;
    node.inputs = std::move(inputs);
    node.float_value = float_val;
    node.int_value = int_val;
    node.name = std::move(name);
    node.axis = axis;
    node.shape = std::move(shape);
    node.dtype = dtype;
    nodes_.push_back(std::move(node));
    return id;
}

std::vector<int64_t> TraceGraph::broadcast_shapes(
        const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    // NumPy-style broadcasting: align from the right, each dimension
    // must be equal, or one of them is 1 (or -1 = symbolic/unknown).
    size_t max_ndim = std::max(a.size(), b.size());
    std::vector<int64_t> result(max_ndim, -1);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (da == db) {
            result[max_ndim - 1 - i] = da;
        } else if (da == 1) {
            result[max_ndim - 1 - i] = db;
        } else if (db == 1) {
            result[max_ndim - 1 - i] = da;
        } else if (da < 0) {
            // Symbolic: prefer the known dimension if one is known
            result[max_ndim - 1 - i] = (db > 0) ? db : da;
        } else if (db < 0) {
            result[max_ndim - 1 - i] = (da > 0) ? da : db;
        } else {
            // Incompatible shapes — mark as symbolic
            result[max_ndim - 1 - i] = -1;
        }
    }
    return result;
}

std::vector<int64_t> TraceGraph::matmul_shapes(
        const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    // MatMul: [..., M, K] x [..., K, N] → [..., M, N]
    // Also handles 1-D x 1-D (dot product), 1-D x 2-D, 2-D x 1-D
    if (a.size() < 1 || b.size() < 1) return {-1};

    // 1-D x 1-D: dot product → scalar
    if (a.size() == 1 && b.size() == 1) {
        return {1};
    }

    // 1-D x 2-D: [K] x [K, N] → [N]
    if (a.size() == 1 && b.size() == 2) {
        return {b[1] < 0 ? -1 : b[1]};
    }

    // 2-D x 1-D: [M, K] x [K] → [M]
    if (a.size() == 2 && b.size() == 1) {
        return {a[0] < 0 ? -1 : a[0]};
    }

    // Batched matmul: broadcast batch dims, then MxN
    if (a.size() >= 2 && b.size() >= 2) {
        std::vector<int64_t> a_batch(a.begin(), a.end() - 2);
        std::vector<int64_t> b_batch(b.begin(), b.end() - 2);
        auto batch = broadcast_shapes(a_batch, b_batch);

        int64_t M = (a[a.size() - 2] < 0) ? -1 : a[a.size() - 2];
        int64_t N = (b[b.size() - 1] < 0) ? -1 : b[b.size() - 1];

        batch.push_back(M);
        batch.push_back(N);
        return batch;
    }

    return {-1};
}

std::vector<int64_t> TraceGraph::reduce_shape(
        const std::vector<int64_t>& input_shape, int64_t axis) {
    if (input_shape.empty()) return {};
    if (axis < 0 || static_cast<size_t>(axis) >= input_shape.size()) {
        // Invalid axis: return shape unchanged (or empty)
        return input_shape;
    }
    std::vector<int64_t> result;
    result.reserve(input_shape.size() - 1);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (static_cast<int64_t>(i) != axis) {
            result.push_back(input_shape[i]);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Node Interning
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::intern_node(const TraceNode& node) {
    NodeKey key;
    key.op = node.op;
    key.inputs = node.inputs;
    key.float_value = node.float_value;
    key.int_value = node.int_value;
    key.name = node.name;
    key.axis = node.axis;
    key.shape = node.shape;
    key.dtype = node.dtype;

    auto it = intern_map_.find(key);
    if (it != intern_map_.end()) {
        return it->second;
    }

    int64_t id = create_node(node.op, node.inputs, node.float_value,
                             node.int_value, node.name, node.axis,
                             node.shape, node.dtype);
    intern_map_[key] = id;
    return id;
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Leaf Nodes
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::param(const std::string& name,
                           const std::vector<int64_t>& shape, int dtype) {
    // Params are NOT interned — each param call creates a unique input
    // even if the name and shape match. The name is just a label.
    // (Two params named "x" might be different inputs.)
    // However, within a single trace session, the same param name
    // should map to the same node. We use the intern map for this.
    NodeKey key;
    key.op = TraceOp::PARAM;
    key.inputs = {};
    key.float_value = 0.0;
    key.int_value = 0;
    key.name = name;
    key.axis = -1;
    key.shape = shape;
    key.dtype = dtype;

    auto it = intern_map_.find(key);
    if (it != intern_map_.end()) {
        return it->second;
    }

    int64_t id = create_node(TraceOp::PARAM, {}, 0.0, 0, name, -1, shape, dtype);
    intern_map_[key] = id;
    return id;
}

int64_t TraceGraph::constant(double value, int dtype) {
    // Constants ARE interned — same value → same node.
    TraceNode node;
    node.op = TraceOp::CONST;
    node.float_value = value;
    node.int_value = static_cast<int64_t>(value);
    node.shape = {1};  // Scalar constant
    node.dtype = dtype;
    return intern_node(node);
}

int64_t TraceGraph::constant_int(int64_t value, int dtype) {
    TraceNode node;
    node.op = TraceOp::CONST;
    node.float_value = static_cast<double>(value);
    node.int_value = value;
    node.shape = {1};  // Scalar constant
    node.dtype = dtype;
    return intern_node(node);
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Unary Ops
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::unary_op(TraceOp op, int64_t input, int64_t axis) {
    assert(input >= 0 && input < num_nodes());

    // Compute output shape
    std::vector<int64_t> out_shape = nodes_[input].shape;

    TraceNode node;
    node.op = op;
    node.inputs = {input};
    node.axis = axis;
    node.shape = std::move(out_shape);
    node.dtype = nodes_[input].dtype;
    return intern_node(node);
}

int64_t TraceGraph::relu(int64_t x)       { return unary_op(TraceOp::RELU, x); }
int64_t TraceGraph::gelu(int64_t x)       { return unary_op(TraceOp::GELU, x); }
int64_t TraceGraph::sigmoid(int64_t x)    { return unary_op(TraceOp::SIGMOID, x); }
int64_t TraceGraph::exp(int64_t x)        { return unary_op(TraceOp::EXP, x); }
int64_t TraceGraph::log(int64_t x)        { return unary_op(TraceOp::LOG, x); }
int64_t TraceGraph::sqrt(int64_t x)       { return unary_op(TraceOp::SQRT, x); }
int64_t TraceGraph::negate(int64_t x)     { return unary_op(TraceOp::NEG, x); }
int64_t TraceGraph::reciprocal(int64_t x) { return unary_op(TraceOp::RECIPROCAL, x); }
int64_t TraceGraph::layernorm(int64_t x)  { return unary_op(TraceOp::LAYERNORM, x); }
int64_t TraceGraph::rmsnorm(int64_t x)    { return unary_op(TraceOp::RMSNORM, x); }
int64_t TraceGraph::dropout(int64_t x)    { return unary_op(TraceOp::DROPOUT, x); }

int64_t TraceGraph::softmax(int64_t x, int64_t axis) {
    return unary_op(TraceOp::SOFTMAX, x, axis);
}

int64_t TraceGraph::reduce_sum(int64_t x, int64_t axis) {
    assert(x >= 0 && x < num_nodes());
    auto out_shape = reduce_shape(nodes_[x].shape, axis);
    TraceNode node;
    node.op = TraceOp::REDUCE_SUM;
    node.inputs = {x};
    node.axis = axis;
    node.shape = std::move(out_shape);
    node.dtype = nodes_[x].dtype;
    return intern_node(node);
}

int64_t TraceGraph::reduce_max(int64_t x, int64_t axis) {
    assert(x >= 0 && x < num_nodes());
    auto out_shape = reduce_shape(nodes_[x].shape, axis);
    TraceNode node;
    node.op = TraceOp::REDUCE_MAX;
    node.inputs = {x};
    node.axis = axis;
    node.shape = std::move(out_shape);
    node.dtype = nodes_[x].dtype;
    return intern_node(node);
}

int64_t TraceGraph::reduce_mean(int64_t x, int64_t axis) {
    assert(x >= 0 && x < num_nodes());
    auto out_shape = reduce_shape(nodes_[x].shape, axis);
    TraceNode node;
    node.op = TraceOp::REDUCE_MEAN;
    node.inputs = {x};
    node.axis = axis;
    node.shape = std::move(out_shape);
    node.dtype = nodes_[x].dtype;
    return intern_node(node);
}

int64_t TraceGraph::transpose(int64_t x) {
    assert(x >= 0 && x < num_nodes());
    auto in_shape = nodes_[x].shape;
    std::vector<int64_t> out_shape;
    if (in_shape.size() >= 2) {
        // Swap last two dimensions
        out_shape = in_shape;
        std::swap(out_shape[out_shape.size() - 2], out_shape[out_shape.size() - 1]);
    } else {
        out_shape = in_shape;
    }
    TraceNode node;
    node.op = TraceOp::TRANSPOSE;
    node.inputs = {x};
    node.shape = std::move(out_shape);
    node.dtype = nodes_[x].dtype;
    return intern_node(node);
}

int64_t TraceGraph::reshape(int64_t x, const std::vector<int64_t>& shape) {
    assert(x >= 0 && x < num_nodes());
    TraceNode node;
    node.op = TraceOp::RESHAPE;
    node.inputs = {x};
    node.shape = shape;
    node.dtype = nodes_[x].dtype;
    // Note: shape is stored in the node for reshape, so interning
    // different reshape targets produces different nodes.
    return intern_node(node);
}

int64_t TraceGraph::broadcast(int64_t x, const std::vector<int64_t>& shape) {
    assert(x >= 0 && x < num_nodes());
    TraceNode node;
    node.op = TraceOp::BROADCAST;
    node.inputs = {x};
    node.shape = shape;
    node.dtype = nodes_[x].dtype;
    return intern_node(node);
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Binary Ops
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::binary_op(TraceOp op, int64_t left, int64_t right) {
    assert(left >= 0 && left < num_nodes());
    assert(right >= 0 && right < num_nodes());

    std::vector<int64_t> out_shape;
    if (trace_op_is_elementwise(op)) {
        out_shape = broadcast_shapes(nodes_[left].shape, nodes_[right].shape);
    } else if (op == TraceOp::MATMUL) {
        out_shape = matmul_shapes(nodes_[left].shape, nodes_[right].shape);
    } else {
        // Default: use left shape
        out_shape = nodes_[left].shape;
    }

    TraceNode node;
    node.op = op;
    node.inputs = {left, right};
    node.shape = std::move(out_shape);
    node.dtype = nodes_[left].dtype;
    return intern_node(node);
}

int64_t TraceGraph::add(int64_t a, int64_t b)    { return binary_op(TraceOp::ADD, a, b); }
int64_t TraceGraph::mul(int64_t a, int64_t b)    { return binary_op(TraceOp::MUL, a, b); }
int64_t TraceGraph::sub(int64_t a, int64_t b)    { return binary_op(TraceOp::SUB, a, b); }
int64_t TraceGraph::div(int64_t a, int64_t b)    { return binary_op(TraceOp::DIV, a, b); }
int64_t TraceGraph::matmul(int64_t a, int64_t b) { return binary_op(TraceOp::MATMUL, a, b); }

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Ternary Ops
// ─────────────────────────────────────────────────────────────────────────

int64_t TraceGraph::select(int64_t cond, int64_t a, int64_t b) {
    assert(cond >= 0 && cond < num_nodes());
    assert(a >= 0 && a < num_nodes());
    assert(b >= 0 && b < num_nodes());

    // Shape follows the broadcast of a and b (cond is typically same shape)
    auto out_shape = broadcast_shapes(nodes_[a].shape, nodes_[b].shape);

    TraceNode node;
    node.op = TraceOp::SELECT;
    node.inputs = {cond, a, b};
    node.shape = std::move(out_shape);
    node.dtype = nodes_[a].dtype;
    return intern_node(node);
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Access
// ─────────────────────────────────────────────────────────────────────────

const TraceNode& TraceGraph::node(int64_t id) const {
    assert(id >= 0 && id < static_cast<int64_t>(nodes_.size()));
    return nodes_[id];
}

const std::vector<int64_t>& TraceGraph::shape_of(int64_t id) const {
    assert(id >= 0 && id < static_cast<int64_t>(nodes_.size()));
    return nodes_[id].shape;
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Shape Inference
// ─────────────────────────────────────────────────────────────────────────

void TraceGraph::infer_shapes() {
    // Bottom-up propagation: iterate over nodes in ID order (which is
    // topological order for a well-formed SSA graph).
    for (auto& n : nodes_) {
        switch (n.op) {
            case TraceOp::PARAM:
            case TraceOp::CONST:
                // Shape already set at creation time
                break;

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
            case TraceOp::RECIPROCAL: {
                if (n.inputs.size() == 1) {
                    n.shape = nodes_[n.inputs[0]].shape;
                } else if (n.inputs.size() == 2) {
                    n.shape = broadcast_shapes(nodes_[n.inputs[0]].shape,
                                               nodes_[n.inputs[1]].shape);
                }
                break;
            }

            case TraceOp::MATMUL: {
                if (n.inputs.size() == 2) {
                    n.shape = matmul_shapes(nodes_[n.inputs[0]].shape,
                                            nodes_[n.inputs[1]].shape);
                }
                break;
            }

            case TraceOp::TRANSPOSE: {
                if (!n.inputs.empty()) {
                    auto s = nodes_[n.inputs[0]].shape;
                    if (s.size() >= 2) {
                        std::swap(s[s.size() - 2], s[s.size() - 1]);
                    }
                    n.shape = std::move(s);
                }
                break;
            }

            case TraceOp::RESHAPE:
                // Shape is explicitly set; keep as-is
                break;

            case TraceOp::BROADCAST:
                // Shape is explicitly set; keep as-is
                break;

            case TraceOp::REDUCE_SUM:
            case TraceOp::REDUCE_MAX:
            case TraceOp::REDUCE_MEAN: {
                if (!n.inputs.empty()) {
                    n.shape = reduce_shape(nodes_[n.inputs[0]].shape, n.axis);
                }
                break;
            }

            case TraceOp::SOFTMAX:
            case TraceOp::LAYERNORM:
            case TraceOp::RMSNORM:
            case TraceOp::DROPOUT: {
                if (!n.inputs.empty()) {
                    n.shape = nodes_[n.inputs[0]].shape;
                }
                break;
            }

            case TraceOp::SELECT: {
                if (n.inputs.size() >= 3) {
                    n.shape = broadcast_shapes(nodes_[n.inputs[1]].shape,
                                               nodes_[n.inputs[2]].shape);
                }
                break;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Validation
// ─────────────────────────────────────────────────────────────────────────

bool TraceGraph::validate() const {
    // 1. All input references must be valid node IDs
    for (const auto& n : nodes_) {
        for (auto input_id : n.inputs) {
            if (input_id < 0 || input_id >= n.id) {
                // SSA violation: input must be defined before use
                return false;
            }
        }
    }

    // 2. Check for cycles via topological sort (Kahn's algorithm)
    std::vector<int> in_degree(nodes_.size(), 0);
    std::vector<std::vector<int64_t>> adj(nodes_.size());
    for (const auto& n : nodes_) {
        for (auto input_id : n.inputs) {
            adj[input_id].push_back(n.id);
            in_degree[n.id]++;
        }
    }

    std::queue<int64_t> queue;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) {
            queue.push(static_cast<int64_t>(i));
        }
    }

    size_t visited = 0;
    while (!queue.empty()) {
        int64_t id = queue.front();
        queue.pop();
        visited++;
        for (auto next : adj[id]) {
            in_degree[next]--;
            if (in_degree[next] == 0) {
                queue.push(next);
            }
        }
    }

    if (visited != nodes_.size()) {
        // Cycle detected
        return false;
    }

    // 3. Shape consistency: for binary elementwise ops, shapes must be
    //    broadcast-compatible (we already compute broadcast shapes, so
    //    just check the result isn't empty/error).
    for (const auto& n : nodes_) {
        if (trace_op_is_elementwise(n.op) && n.inputs.size() == 2) {
            // Check that the node's shape matches the broadcast of its inputs
            auto expected = broadcast_shapes(nodes_[n.inputs[0]].shape,
                                             nodes_[n.inputs[1]].shape);
            if (!n.shape.empty() && !expected.empty()) {
                // Allow symbolic (-1) dimensions to differ
                if (n.shape.size() != expected.size()) {
                    return false;
                }
                for (size_t i = 0; i < n.shape.size(); ++i) {
                    if (n.shape[i] > 0 && expected[i] > 0 && n.shape[i] != expected[i]) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Conversion (tracer → serialized bridge format)
// ─────────────────────────────────────────────────────────────────────────

std::pair<void*, int64_t> TraceGraph::to_egraph() const {
    // Serialize the trace graph into a SerializedTraceData heap object.
    // The caller (bridge code) can reinterpret the void* as
    // SerializedTraceData* and convert the types for SympleXIR consumption.
    //
    // Layout convention: all PARAM nodes are emitted first (into params),
    // then all other nodes (into ops).  Operand IDs are remapped so that
    // params occupy IDs 0..P-1 and ops occupy IDs P..P+O-1, matching
    // the order in which SympleXIR::from_trace_graph adds them.

    auto* data = new SerializedTraceData();

    // Build a mapping from original trace node ID → sequential bridge ID.
    std::unordered_map<int64_t, int64_t> id_remap;
    int64_t next_bridge_id = 0;

    // First pass: collect PARAM nodes.
    for (const auto& n : nodes_) {
        if (n.op == TraceOp::PARAM) {
            id_remap[n.id] = next_bridge_id++;
            data->params.emplace_back(n.name, n.shape, n.dtype);
        }
    }

    // Second pass: collect non-PARAM nodes.
    for (const auto& n : nodes_) {
        if (n.op != TraceOp::PARAM) {
            id_remap[n.id] = next_bridge_id++;

            // Remap operand IDs from original trace IDs to bridge IDs.
            std::vector<int64_t> remapped_inputs;
            remapped_inputs.reserve(n.inputs.size());
            for (auto input_id : n.inputs) {
                auto it = id_remap.find(input_id);
                remapped_inputs.push_back(it != id_remap.end() ? it->second : input_id);
            }

            data->ops.emplace_back(n.shape, static_cast<int>(n.op),
                                   remapped_inputs, n.float_value,
                                   n.name, n.axis, n.dtype);
        }
    }

    data->num_nodes = static_cast<int64_t>(nodes_.size());
    return {static_cast<void*>(data), data->num_nodes};
}

// ─────────────────────────────────────────────────────────────────────────
// TraceGraph — Printing
// ─────────────────────────────────────────────────────────────────────────

std::string TraceGraph::to_string() const {
    std::ostringstream oss;
    oss << "TraceGraph(" << nodes_.size() << " nodes):\n";
    for (const auto& n : nodes_) {
        oss << "  " << n.to_string() << "\n";
    }
    return oss.str();
}

} // namespace symplex::tracer
