// SympleX – Polyhedral Tensor Superoptimizer
// Python Proxy Tensor Tracer — pybind11 Bindings
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// This module exposes:
//   - Tensor class (wraps int64_t node_id + reference to TraceGraph)
//     with Python operator overloads (__add__, __mul__, __sub__, __truediv__, __neg__)
//     and activation/reduction methods (.relu(), .gelu(), .sigmoid(), .softmax(), etc.)
//   - Graph class (wraps TraceGraph) with .param(), .constant(), .trace() methods
//   - trace() function / context manager for implicit tracing

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>

#include "symplex/tracer/tracer.h"

#include <memory>
#include <sstream>

namespace py = pybind11;
using namespace symplex::tracer;

// ─────────────────────────────────────────────────────────────────────────
// Python-facing Tensor class
// ─────────────────────────────────────────────────────────────────────────

/// PyTensor: a proxy tensor that records operations into the trace graph.
/// Every arithmetic operation emits a new node in the graph instead of
/// computing a value. Python execution becomes a graph-writing script
/// pretending to be math.
class PyTensor {
public:
    /// Construct a tensor backed by node_id in the given graph.
    PyTensor(int64_t node_id, std::shared_ptr<TraceGraph> graph)
        : node_id_(node_id), graph_(std::move(graph)) {}

    int64_t node_id() const { return node_id_; }
    std::shared_ptr<TraceGraph> graph() const { return graph_; }

    // ── Shape property ──────────────────────────────────────────────

    std::vector<int64_t> shape() const {
        if (!graph_) return {};
        return graph_->shape_of(node_id_);
    }

    // ── Dtype property ──────────────────────────────────────────────

    int dtype() const {
        if (!graph_) return 0;
        return graph_->node(node_id_).dtype;
    }

    // ── Python Operator Overloads ───────────────────────────────────

    PyTensor operator+(const PyTensor& other) const {
        auto g = shared_graph(other);
        return PyTensor(g->add(node_id_, other.node_id_), g);
    }

    PyTensor operator*(const PyTensor& other) const {
        auto g = shared_graph(other);
        return PyTensor(g->mul(node_id_, other.node_id_), g);
    }

    PyTensor operator-(const PyTensor& other) const {
        auto g = shared_graph(other);
        return PyTensor(g->sub(node_id_, other.node_id_), g);
    }

    PyTensor operator/(const PyTensor& other) const {
        auto g = shared_graph(other);
        return PyTensor(g->div(node_id_, other.node_id_), g);
    }

    PyTensor operator-() const {
        return PyTensor(graph_->negate(node_id_), graph_);
    }

    // ── Right-side scalar operators (Tensor op scalar) ─────────────
    // e.g. tensor + 1.0, tensor * 2.0

    PyTensor add_scalar(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->add(node_id_, c), graph_);
    }

    PyTensor mul_scalar(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->mul(node_id_, c), graph_);
    }

    PyTensor sub_scalar(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->sub(node_id_, c), graph_);
    }

    PyTensor div_scalar(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->div(node_id_, c), graph_);
    }

    // ── Reverse operators (scalar op Tensor) ─────────────────────────
    // Python will call __radd__ etc. when the left operand doesn't know
    // how to handle the operation (e.g. float + Tensor).

    PyTensor __radd__(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->add(c, node_id_), graph_);
    }

    PyTensor __rmul__(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->mul(c, node_id_), graph_);
    }

    PyTensor __rsub__(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->sub(c, node_id_), graph_);
    }

    PyTensor __rtruediv__(double scalar) const {
        auto c = graph_->constant(scalar);
        return PyTensor(graph_->div(c, node_id_), graph_);
    }

    // ── Activation Functions ────────────────────────────────────────

    PyTensor relu() const {
        return PyTensor(graph_->relu(node_id_), graph_);
    }

    PyTensor gelu() const {
        return PyTensor(graph_->gelu(node_id_), graph_);
    }

    PyTensor sigmoid() const {
        return PyTensor(graph_->sigmoid(node_id_), graph_);
    }

    PyTensor exp() const {
        return PyTensor(graph_->exp(node_id_), graph_);
    }

    PyTensor log() const {
        return PyTensor(graph_->log(node_id_), graph_);
    }

    PyTensor sqrt() const {
        return PyTensor(graph_->sqrt(node_id_), graph_);
    }

    PyTensor reciprocal() const {
        return PyTensor(graph_->reciprocal(node_id_), graph_);
    }

    PyTensor softmax(int64_t axis = -1) const {
        return PyTensor(graph_->softmax(node_id_, axis), graph_);
    }

    PyTensor layernorm() const {
        return PyTensor(graph_->layernorm(node_id_), graph_);
    }

    PyTensor rmsnorm() const {
        return PyTensor(graph_->rmsnorm(node_id_), graph_);
    }

    PyTensor dropout() const {
        return PyTensor(graph_->dropout(node_id_), graph_);
    }

    // ── Reductions ──────────────────────────────────────────────────

    PyTensor reduce_sum(int64_t axis) const {
        return PyTensor(graph_->reduce_sum(node_id_, axis), graph_);
    }

    PyTensor reduce_max(int64_t axis) const {
        return PyTensor(graph_->reduce_max(node_id_, axis), graph_);
    }

    PyTensor reduce_mean(int64_t axis) const {
        return PyTensor(graph_->reduce_mean(node_id_, axis), graph_);
    }

    // ── Shape Operations ────────────────────────────────────────────

    PyTensor matmul(const PyTensor& other) const {
        auto g = shared_graph(other);
        return PyTensor(g->matmul(node_id_, other.node_id_), g);
    }

    PyTensor transpose() const {
        return PyTensor(graph_->transpose(node_id_), graph_);
    }

    PyTensor reshape(const std::vector<int64_t>& new_shape) const {
        return PyTensor(graph_->reshape(node_id_, new_shape), graph_);
    }

    PyTensor broadcast(const std::vector<int64_t>& new_shape) const {
        return PyTensor(graph_->broadcast(node_id_, new_shape), graph_);
    }

    // ── String Representation ───────────────────────────────────────

    std::string repr() const {
        if (!graph_) return "Tensor(<null>)";
        auto& n = graph_->node(node_id_);
        std::ostringstream oss;
        oss << "Tensor(" << n.to_string() << ")";
        return oss.str();
    }

private:
    int64_t node_id_;
    std::shared_ptr<TraceGraph> graph_;

    /// Verify both tensors share the same graph and return it.
    std::shared_ptr<TraceGraph> shared_graph(const PyTensor& other) const {
        if (graph_.get() != other.graph_.get()) {
            throw std::runtime_error(
                "Cannot combine tensors from different trace graphs. "
                "All tensors in an expression must belong to the same Graph.");
        }
        return graph_;
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Python-facing Graph class
// ─────────────────────────────────────────────────────────────────────────

/// PyGraph: Python wrapper for TraceGraph.
class PyGraph {
public:
    PyGraph() : graph_(std::make_shared<TraceGraph>()) {}

    // ── Leaf Nodes ──────────────────────────────────────────────────

    PyTensor param(const std::string& name, const std::vector<int64_t>& shape,
                   int dtype = 0) {
        return PyTensor(graph_->param(name, shape, dtype), graph_);
    }

    PyTensor constant(double value, int dtype = 0) {
        return PyTensor(graph_->constant(value, dtype), graph_);
    }

    PyTensor constant_int(int64_t value, int dtype = 3) {
        return PyTensor(graph_->constant_int(value, dtype), graph_);
    }

    // ── Trace a function ────────────────────────────────────────────

    /// Trace a Python function: call it with param tensors.
    /// Usage:
    ///   result = graph.trace(my_model, "a", [1024, 1024], "b", [1024, 1024])
    ///
    /// Or pass tensors directly:
    ///   a = graph.param("a", [1024, 1024])
    ///   b = graph.param("b", [1024, 1024])
    ///   result = graph.trace(my_model, a, b)
    py::object trace(py::function fn, py::args args, py::kwargs kwargs) {
        // Convert any string+shape pairs to param tensors.
        // We use a while-loop so that string+shape pairs are consumed
        // in a single iteration without a manual i++ that would combine
        // with the for-loop increment to skip the element after the
        // shape.
        py::list processed_args;
        size_t i = 0;
        while (i < args.size()) {
            py::object arg = args[i];
            if (py::isinstance<py::str>(arg)) {
                // Next arg should be the shape
                if (i + 1 < args.size()) {
                    std::string name = arg.cast<std::string>();
                    py::object shape_obj = args[i + 1];
                    std::vector<int64_t> shape = shape_obj.cast<std::vector<int64_t>>();
                    processed_args.append(py::cast(graph_->param(name, shape)));
                    i += 2;  // Skip both name and shape; continue loop
                    continue;
                }
            }
            // Default: pass the arg through as-is
            processed_args.append(arg);
            ++i;
        }

        // Call the function with the processed args
        py::object result = fn(*processed_args, **kwargs);
        return result;
    }

    // ── Graph Operations ────────────────────────────────────────────

    void infer_shapes() {
        graph_->infer_shapes();
    }

    bool validate() const {
        return graph_->validate();
    }

    int64_t num_nodes() const {
        return graph_->num_nodes();
    }

    std::string to_string() const {
        return graph_->to_string();
    }

    // ── Access underlying graph ─────────────────────────────────────

    std::shared_ptr<TraceGraph> graph_ptr() const { return graph_; }

private:
    std::shared_ptr<TraceGraph> graph_;
};

// ─────────────────────────────────────────────────────────────────────────
// trace() function — context manager for implicit tracing
// ─────────────────────────────────────────────────────────────────────────

/// TraceContext wrapper for Python `with trace():` usage.
class PyTraceContext {
public:
    PyTraceContext() = default;

    void enter() {
        // Construct the TraceContext here, when __enter__ is called,
        // NOT in the constructor.  Python's `with` statement creates
        // the object first, then calls __enter__ — if we set
        // CURRENT_GRAPH in the constructor, it is set too early and
        // may be visible to code between construction and __enter__.
        ctx_.emplace();
    }

    void exit(py::object, py::object, py::object) {
        // Destroy the TraceContext explicitly when __exit__ is called.
        // This restores CURRENT_GRAPH to its previous value.
        // If we relied on the destructor running at Python GC time,
        // CURRENT_GRAPH would remain set for an indeterminate period.
        ctx_.reset();
    }

    TraceGraph& graph() { return ctx_->graph(); }

private:
    // Deferred construction: the TraceContext is not created until
    // enter() is called, and is destroyed as soon as exit() runs.
    std::optional<TraceContext> ctx_;
};

// ─────────────────────────────────────────────────────────────────────────
// Module-level convenience functions
// ─────────────────────────────────────────────────────────────────────────

/// Module-level relu: symplex.relu(x) → x.relu()
static PyTensor module_relu(const PyTensor& x) { return x.relu(); }
static PyTensor module_gelu(const PyTensor& x) { return x.gelu(); }
static PyTensor module_sigmoid(const PyTensor& x) { return x.sigmoid(); }
static PyTensor module_softmax(const PyTensor& x, int64_t axis = -1) { return x.softmax(axis); }
static PyTensor module_layernorm(const PyTensor& x) { return x.layernorm(); }
static PyTensor module_rmsnorm(const PyTensor& x) { return x.rmsnorm(); }
static PyTensor module_exp(const PyTensor& x) { return x.exp(); }
static PyTensor module_log(const PyTensor& x) { return x.log(); }
static PyTensor module_sqrt(const PyTensor& x) { return x.sqrt(); }
static PyTensor module_dropout(const PyTensor& x) { return x.dropout(); }
static PyTensor module_matmul(const PyTensor& a, const PyTensor& b) { return a.matmul(b); }

// ─────────────────────────────────────────────────────────────────────────
// pybind11 Module Definition
// ─────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_symplex, m) {
    m.doc() = R"doc(
SympleX Proxy Tensor Tracer

The tracer replaces real tensors with Proxy Tensors that record ops.
Python execution becomes a graph-writing script pretending to be math:

    a + b → emits Add(a, b)
    a * b → emits Mul(a, b)
    sin(a) → emits Sin(a)

Usage:
    import symplex

    graph = symplex.Graph()
    a = graph.param("a", [1024, 1024])
    b = graph.param("b", [1024, 1024])
    c = a + b * a
    result = symplex.relu(c)
    print(graph)  # Shows the trace graph
)doc";

    // ── TraceOp enum ────────────────────────────────────────────────

    py::enum_<TraceOp>(m, "TraceOp")
        .value("ADD", TraceOp::ADD)
        .value("MUL", TraceOp::MUL)
        .value("SUB", TraceOp::SUB)
        .value("DIV", TraceOp::DIV)
        .value("NEG", TraceOp::NEG)
        .value("MATMUL", TraceOp::MATMUL)
        .value("TRANSPOSE", TraceOp::TRANSPOSE)
        .value("RESHAPE", TraceOp::RESHAPE)
        .value("BROADCAST", TraceOp::BROADCAST)
        .value("REDUCE_SUM", TraceOp::REDUCE_SUM)
        .value("REDUCE_MAX", TraceOp::REDUCE_MAX)
        .value("REDUCE_MEAN", TraceOp::REDUCE_MEAN)
        .value("RELU", TraceOp::RELU)
        .value("GELU", TraceOp::GELU)
        .value("SIGMOID", TraceOp::SIGMOID)
        .value("SOFTMAX", TraceOp::SOFTMAX)
        .value("LAYERNORM", TraceOp::LAYERNORM)
        .value("RMSNORM", TraceOp::RMSNORM)
        .value("EXP", TraceOp::EXP)
        .value("LOG", TraceOp::LOG)
        .value("SQRT", TraceOp::SQRT)
        .value("RECIPROCAL", TraceOp::RECIPROCAL)
        .value("DROPOUT", TraceOp::DROPOUT)
        .value("SELECT", TraceOp::SELECT)
        .value("CONST", TraceOp::CONST)
        .value("PARAM", TraceOp::PARAM)
        .def("__repr__", [](TraceOp op) { return trace_op_to_string(op); });

    // ── Tensor class ────────────────────────────────────────────────

    py::class_<PyTensor>(m, "Tensor")
        .def_property_readonly("shape", &PyTensor::shape,
            "Get the tensor shape as a list of ints (-1 = symbolic)")
        .def_property_readonly("dtype", &PyTensor::dtype,
            "Get the tensor dtype (0=fp32, 1=fp16, 2=bf16, 3=int8)")

        // Python operator overloads
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def(py::self - py::self)
        .def(py::self / py::self)
        .def(-py::self)

        // Reverse operators for scalar op Tensor
        .def("__radd__", &PyTensor::__radd__)
        .def("__rmul__", &PyTensor::__rmul__)
        .def("__rsub__", &PyTensor::__rsub__)
        .def("__rtruediv__", &PyTensor::__rtruediv__)

        // Tensor op scalar (right-side scalar)
        .def("__add__", &PyTensor::add_scalar, py::is_operator())
        .def("__mul__", &PyTensor::mul_scalar, py::is_operator())
        .def("__sub__", &PyTensor::sub_scalar, py::is_operator())
        .def("__truediv__", &PyTensor::div_scalar, py::is_operator())

        // MatMul uses @ operator in Python
        .def("__matmul__", &PyTensor::matmul)
        .def("__rmatmul__", [](const PyTensor& self, const PyTensor& other) {
            return other.matmul(self);
        })

        // Activation functions
        .def("relu", &PyTensor::relu,
            "Apply ReLU activation: max(0, x)")
        .def("gelu", &PyTensor::gelu,
            "Apply GELU activation")
        .def("sigmoid", &PyTensor::sigmoid,
            "Apply sigmoid activation")
        .def("exp", &PyTensor::exp,
            "Apply exponential function")
        .def("log", &PyTensor::log,
            "Apply natural logarithm")
        .def("sqrt", &PyTensor::sqrt,
            "Apply square root")
        .def("reciprocal", &PyTensor::reciprocal,
            "Apply reciprocal (1/x)")
        .def("softmax", &PyTensor::softmax, py::arg("axis") = -1,
            "Apply softmax along the given axis")
        .def("layernorm", &PyTensor::layernorm,
            "Apply layer normalization")
        .def("rmsnorm", &PyTensor::rmsnorm,
            "Apply RMS normalization (LLaMA-style)")
        .def("dropout", &PyTensor::dropout,
            "Apply dropout (identity during inference)")

        // Reductions
        .def("reduce_sum", &PyTensor::reduce_sum, py::arg("axis"),
            "Sum reduction along the given axis")
        .def("reduce_max", &PyTensor::reduce_max, py::arg("axis"),
            "Max reduction along the given axis")
        .def("reduce_mean", &PyTensor::reduce_mean, py::arg("axis"),
            "Mean reduction along the given axis")

        // Shape operations
        .def("matmul", &PyTensor::matmul,
            "Matrix multiplication with another tensor")
        .def("transpose", &PyTensor::transpose,
            "Transpose the last two dimensions")
        .def("reshape", &PyTensor::reshape,
            "Reshape to the given shape")
        .def("broadcast", &PyTensor::broadcast,
            "Broadcast to the given shape")

        // String representation
        .def("__repr__", &PyTensor::repr)
        .def("__str__", &PyTensor::repr);

    // ── Graph class ─────────────────────────────────────────────────

    py::class_<PyGraph>(m, "Graph")
        .def(py::init<>())

        // Leaf nodes
        .def("param", &PyGraph::param,
            py::arg("name"), py::arg("shape"), py::arg("dtype") = 0,
            R"doc(Add a parameter (input tensor) to the graph.

Args:
    name: Parameter name (e.g. "x", "weight")
    shape: Tensor shape as list of ints (-1 = symbolic/unknown)
    dtype: Data type (0=fp32, 1=fp16, 2=bf16, 3=int8)

Returns:
    A proxy Tensor that records operations into this graph.
)doc")
        .def("constant", &PyGraph::constant,
            py::arg("value"), py::arg("dtype") = 0,
            "Add a floating-point constant to the graph")
        .def("constant_int", &PyGraph::constant_int,
            py::arg("value"), py::arg("dtype") = 3,
            "Add an integer constant to the graph")

        // Trace a function
        .def("trace", &PyGraph::trace,
            py::arg("fn"),
            R"doc(Trace a Python function, recording all tensor operations into the graph.

The function receives proxy Tensors as arguments. All arithmetic
operations on those Tensors are recorded as nodes in the trace graph
instead of being executed.

Usage:
    def my_model(a, b):
        c = a + b * a
        return symplex.relu(c)

    graph = symplex.Graph()
    a = graph.param("a", [1024, 1024])
    b = graph.param("b", [1024, 1024])
    result = graph.trace(my_model, a, b)
)doc")

        // Graph operations
        .def("infer_shapes", &PyGraph::infer_shapes,
            "Propagate shapes through the graph bottom-up")
        .def("validate", &PyGraph::validate,
            "Validate the graph is well-formed (SSA, no cycles, shape consistency)")
        .def("num_nodes", &PyGraph::num_nodes,
            "Return the number of nodes in the graph")

        // String representation
        .def("__repr__", &PyGraph::to_string)
        .def("__str__", &PyGraph::to_string)
        .def("__len__", &PyGraph::num_nodes);

    // ── TraceContext class (context manager for implicit tracing) ────

    py::class_<PyTraceContext>(m, "TraceContext")
        .def(py::init<>(),
            R"doc(Create a trace context for implicit tracing.

Usage:
    with symplex.TraceContext() as ctx:
        # Operations on proxy tensors inside this block are
        # recorded into ctx's trace graph automatically.
        ...
)doc")
        .def("__enter__", &PyTraceContext::enter,
            "Enter the trace context: sets CURRENT_GRAPH for implicit tracing")
        .def("__exit__", &PyTraceContext::exit,
            "Exit the trace context: restores the previous CURRENT_GRAPH");

    // ── Module-level convenience functions ──────────────────────────

    m.def("relu", &module_relu, py::arg("x"),
        "Apply ReLU activation to a proxy tensor");
    m.def("gelu", &module_gelu, py::arg("x"),
        "Apply GELU activation to a proxy tensor");
    m.def("sigmoid", &module_sigmoid, py::arg("x"),
        "Apply sigmoid activation to a proxy tensor");
    m.def("softmax", &module_softmax, py::arg("x"), py::arg("axis") = -1,
        "Apply softmax to a proxy tensor along the given axis");
    m.def("layernorm", &module_layernorm, py::arg("x"),
        "Apply layer normalization to a proxy tensor");
    m.def("rmsnorm", &module_rmsnorm, py::arg("x"),
        "Apply RMS normalization to a proxy tensor");
    m.def("exp", &module_exp, py::arg("x"),
        "Apply exponential function to a proxy tensor");
    m.def("log", &module_log, py::arg("x"),
        "Apply natural logarithm to a proxy tensor");
    m.def("sqrt", &module_sqrt, py::arg("x"),
        "Apply square root to a proxy tensor");
    m.def("dropout", &module_dropout, py::arg("x"),
        "Apply dropout to a proxy tensor (identity during inference)");
    m.def("matmul", &module_matmul, py::arg("a"), py::arg("b"),
        "Matrix multiply two proxy tensors");
}
