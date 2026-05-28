// SympleX – Polyhedral Tensor Superoptimizer
// SympleX IR Implementation
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/ir/symplex_ir.h"
#include "symplex/optimizer/egraph.h"

namespace symplex::ir {

// ─────────────────────────────────────────────────────────────────────────
// Building
// ─────────────────────────────────────────────────────────────────────────

int64_t SympleXIR::add_symbol(const std::string& name, const IRShape& shape,
                               IRDType dtype) {
    // Interning: same name → same ID
    auto it = symbol_map_.find(name);
    if (it != symbol_map_.end()) {
        return it->second;
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = IROp::Kind::SYMBOL;
    op.shape = shape;
    op.dtype = dtype;
    op.name = name;
    op.id_ = id;
    ops_.push_back(std::move(op));

    symbol_map_[name] = id;
    return id;
}

int64_t SympleXIR::add_constant(double value, IRDType dtype) {
    // Interning: same float value → same ID
    auto it = float_const_map_.find(value);
    if (it != float_const_map_.end()) {
        return it->second;
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = IROp::Kind::CONSTANT;
    op.shape = IRShape({1});  // Scalar constant
    op.dtype = dtype;
    op.float_value = value;
    op.int_value = static_cast<int64_t>(value);
    op.id_ = id;
    ops_.push_back(std::move(op));

    float_const_map_[value] = id;
    return id;
}

int64_t SympleXIR::add_int_constant(int64_t value, IRDType dtype) {
    // Interning: same int value → same ID
    auto it = int_const_map_.find(value);
    if (it != int_const_map_.end()) {
        return it->second;
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = IROp::Kind::CONSTANT;
    op.shape = IRShape({1});  // Scalar constant
    op.dtype = dtype;
    op.float_value = static_cast<double>(value);
    op.int_value = value;
    op.id_ = id;
    ops_.push_back(std::move(op));

    int_const_map_[value] = id;
    return id;
}

int64_t SympleXIR::add_op(IROp::Kind kind, const std::vector<int64_t>& operands,
                           const IRShape& result_shape, IRDType dtype) {
    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = operands;
    op.shape = result_shape;
    op.dtype = dtype;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

int64_t SympleXIR::add_unary(IROp::Kind kind, int64_t operand) {
    assert(is_valid_id(operand));
    const auto& input = ops_[operand];

    IRShape result_shape = input.shape;
    // For reshape/broadcast, the shape should be set explicitly via add_op
    // For transpose, swap last two dims
    if (kind == IROp::Kind::TRANSPOSE && input.shape.ndim() >= 2) {
        result_shape = input.shape;
        std::swap(result_shape.dims[result_shape.ndim() - 2],
                  result_shape.dims[result_shape.ndim() - 1]);
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = {operand};
    op.shape = result_shape;
    op.dtype = input.dtype;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

int64_t SympleXIR::add_binary(IROp::Kind kind, int64_t lhs, int64_t rhs) {
    assert(is_valid_id(lhs));
    assert(is_valid_id(rhs));
    const auto& left = ops_[lhs];
    const auto& right = ops_[rhs];

    IRShape result_shape;
    if (kind == IROp::Kind::MATMUL) {
        result_shape = matmul_shapes(left.shape, right.shape);
    } else {
        // Elementwise: broadcast
        result_shape = broadcast_shapes(left.shape, right.shape);
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = {lhs, rhs};
    op.shape = result_shape;
    op.dtype = left.dtype;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

int64_t SympleXIR::add_ternary(IROp::Kind kind, int64_t a, int64_t b, int64_t c) {
    assert(is_valid_id(a));
    assert(is_valid_id(b));
    assert(is_valid_id(c));

    IRShape result_shape;
    if (kind == IROp::Kind::SELECT) {
        // Shape = broadcast(b.shape, c.shape)
        result_shape = broadcast_shapes(ops_[b].shape, ops_[c].shape);
    } else if (kind == IROp::Kind::FUSED_GEMM || kind == IROp::Kind::FUSED_MATMUL_ADD_RELU) {
        // GEMM: alpha*A*B + beta*C → result shape = matmul_shapes(A, B)
        result_shape = matmul_shapes(ops_[a].shape, ops_[b].shape);
    } else if (kind == IROp::Kind::FUSED_ADD_LN) {
        // Add+LN: output shape = input shape
        result_shape = broadcast_shapes(ops_[a].shape, ops_[b].shape);
    } else {
        result_shape = ops_[a].shape;
    }

    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = {a, b, c};
    op.shape = result_shape;
    op.dtype = ops_[a].dtype;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

int64_t SympleXIR::add_matmul(int64_t lhs, int64_t rhs) {
    return add_binary(IROp::Kind::MATMUL, lhs, rhs);
}

int64_t SympleXIR::add_reduction(IROp::Kind kind, int64_t operand, int64_t axis) {
    assert(is_valid_id(operand));
    assert(kind == IROp::Kind::REDUCE_SUM ||
           kind == IROp::Kind::REDUCE_MAX ||
           kind == IROp::Kind::REDUCE_MEAN);

    const auto& input = ops_[operand];
    IRShape result_shape = reduce_shape(input.shape, axis);

    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = {operand};
    op.shape = result_shape;
    op.dtype = input.dtype;
    op.axis = axis;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

int64_t SympleXIR::add_unary_with_axis(IROp::Kind kind, int64_t operand, int64_t axis) {
    assert(is_valid_id(operand));

    const auto& input = ops_[operand];

    int64_t id = next_id_++;
    IROp op;
    op.kind = kind;
    op.operands = {operand};
    op.shape = input.shape;  // Same shape for softmax, etc.
    op.dtype = input.dtype;
    op.axis = axis;
    op.id_ = id;
    ops_.push_back(std::move(op));
    return id;
}

void SympleXIR::attach_poly_annotation(int64_t op_id, IRAffineAnnotation annotation) {
    assert(is_valid_id(op_id));
    ops_[op_id].poly_annotation = std::move(annotation);
}

// ─────────────────────────────────────────────────────────────────────────
// Query
// ─────────────────────────────────────────────────────────────────────────

const IROp& SympleXIR::op(int64_t id) const {
    assert(is_valid_id(id));
    return ops_[id];
}

IROp& SympleXIR::op_mut(int64_t id) {
    assert(is_valid_id(id));
    return ops_[id];
}

// ─────────────────────────────────────────────────────────────────────────
// Shape Inference
// ─────────────────────────────────────────────────────────────────────────

void SympleXIR::infer_shapes() {
    // Bottom-up propagation: iterate over ops in insertion order
    // (which is topological order for well-formed SSA).
    for (auto& ir_op : ops_) {
        switch (ir_op.kind) {
            case IROp::Kind::SYMBOL:
            case IROp::Kind::CONSTANT:
                // Shape already set at creation time
                break;

            case IROp::Kind::ADD:
            case IROp::Kind::MUL:
            case IROp::Kind::SUB:
            case IROp::Kind::DIV: {
                if (ir_op.operands.size() == 2) {
                    ir_op.shape = broadcast_shapes(
                        ops_[ir_op.operands[0]].shape,
                        ops_[ir_op.operands[1]].shape);
                }
                break;
            }

            case IROp::Kind::NEG:
            case IROp::Kind::RELU:
            case IROp::Kind::GELU:
            case IROp::Kind::SIGMOID:
            case IROp::Kind::EXP:
            case IROp::Kind::LOG:
            case IROp::Kind::SQRT:
            case IROp::Kind::RECIPROCAL:
            case IROp::Kind::DROPOUT:
            case IROp::Kind::IDENTITY:
            case IROp::Kind::FUSED_SOFTMAX:
            case IROp::Kind::FUSED_LAYERNORM:
            case IROp::Kind::FUSED_RMSNORM: {
                if (!ir_op.operands.empty()) {
                    ir_op.shape = ops_[ir_op.operands[0]].shape;
                }
                break;
            }

            case IROp::Kind::MATMUL: {
                if (ir_op.operands.size() == 2) {
                    ir_op.shape = matmul_shapes(
                        ops_[ir_op.operands[0]].shape,
                        ops_[ir_op.operands[1]].shape);
                }
                break;
            }

            case IROp::Kind::TRANSPOSE: {
                if (!ir_op.operands.empty()) {
                    auto s = ops_[ir_op.operands[0]].shape;
                    if (s.ndim() >= 2) {
                        std::swap(s.dims[s.ndim() - 2], s.dims[s.ndim() - 1]);
                    }
                    ir_op.shape = std::move(s);
                }
                break;
            }

            case IROp::Kind::RESHAPE:
            case IROp::Kind::BROADCAST:
                // Shape is explicitly set; keep as-is
                break;

            case IROp::Kind::REDUCE_SUM:
            case IROp::Kind::REDUCE_MAX:
            case IROp::Kind::REDUCE_MEAN: {
                if (!ir_op.operands.empty()) {
                    ir_op.shape = reduce_shape(
                        ops_[ir_op.operands[0]].shape, ir_op.axis);
                }
                break;
            }

            case IROp::Kind::SOFTMAX:
            case IROp::Kind::LAYERNORM:
            case IROp::Kind::RMSNORM: {
                if (!ir_op.operands.empty()) {
                    ir_op.shape = ops_[ir_op.operands[0]].shape;
                }
                break;
            }

            case IROp::Kind::FUSED_MATMUL_RELU:
            case IROp::Kind::FUSED_MATMUL_ADD: {
                if (ir_op.operands.size() >= 2) {
                    ir_op.shape = matmul_shapes(
                        ops_[ir_op.operands[0]].shape,
                        ops_[ir_op.operands[1]].shape);
                }
                break;
            }

            case IROp::Kind::FUSED_MATMUL_ADD_RELU:
            case IROp::Kind::FUSED_GEMM: {
                if (ir_op.operands.size() >= 2) {
                    ir_op.shape = matmul_shapes(
                        ops_[ir_op.operands[0]].shape,
                        ops_[ir_op.operands[1]].shape);
                }
                break;
            }

            case IROp::Kind::FUSED_ADD_LN: {
                if (ir_op.operands.size() >= 2) {
                    ir_op.shape = broadcast_shapes(
                        ops_[ir_op.operands[0]].shape,
                        ops_[ir_op.operands[1]].shape);
                }
                break;
            }

            case IROp::Kind::FUSED_MHA: {
                // Fused MHA: Q[M,H,D], K[M,H,D], V[M,H,D]
                // Output shape = Q.shape (same as input)
                if (!ir_op.operands.empty()) {
                    ir_op.shape = ops_[ir_op.operands[0]].shape;
                }
                break;
            }

            case IROp::Kind::TILE:
            case IROp::Kind::UNTILE: {
                if (!ir_op.operands.empty()) {
                    ir_op.shape = ops_[ir_op.operands[0]].shape;
                }
                break;
            }

            case IROp::Kind::SELECT: {
                if (ir_op.operands.size() >= 3) {
                    ir_op.shape = broadcast_shapes(
                        ops_[ir_op.operands[1]].shape,
                        ops_[ir_op.operands[2]].shape);
                }
                break;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────

bool SympleXIR::validate() const {
    // 1. All operand references must be valid SSA IDs (defined before use)
    for (const auto& ir_op : ops_) {
        for (auto operand_id : ir_op.operands) {
            if (operand_id < 0 || operand_id >= ir_op.id_) {
                // SSA violation: operand must be defined before use
                return false;
            }
        }
    }

    // 2. Check for cycles via topological sort (Kahn's algorithm)
    std::vector<int> in_degree(ops_.size(), 0);
    std::vector<std::vector<int64_t>> adj(ops_.size());
    for (const auto& ir_op : ops_) {
        for (auto operand_id : ir_op.operands) {
            adj[operand_id].push_back(ir_op.id_);
            in_degree[ir_op.id_]++;
        }
    }

    std::queue<int64_t> queue;
    for (size_t i = 0; i < ops_.size(); ++i) {
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

    if (visited != ops_.size()) {
        // Cycle detected
        return false;
    }

    // 3. Shape consistency: for binary elementwise ops, shapes must be
    //    broadcast-compatible
    for (const auto& ir_op : ops_) {
        if (ir_op.kind == IROp::Kind::ADD ||
            ir_op.kind == IROp::Kind::MUL ||
            ir_op.kind == IROp::Kind::SUB ||
            ir_op.kind == IROp::Kind::DIV) {
            if (ir_op.operands.size() == 2) {
                auto expected = broadcast_shapes(
                    ops_[ir_op.operands[0]].shape,
                    ops_[ir_op.operands[1]].shape);
                if (!ir_op.shape.is_unknown() && !expected.is_unknown()) {
                    if (ir_op.shape.ndim() != expected.ndim()) {
                        return false;
                    }
                    for (size_t i = 0; i < ir_op.shape.ndim(); ++i) {
                        if (ir_op.shape[i] > 0 && expected[i] > 0 &&
                            ir_op.shape[i] != expected[i]) {
                            return false;
                        }
                    }
                }
            }
        }

        // MatMul: check that inner dimensions match
        if (ir_op.kind == IROp::Kind::MATMUL && ir_op.operands.size() == 2) {
            const auto& a_shape = ops_[ir_op.operands[0]].shape;
            const auto& b_shape = ops_[ir_op.operands[1]].shape;
            if (a_shape.ndim() >= 2 && b_shape.ndim() >= 2) {
                int64_t a_inner = a_shape[a_shape.ndim() - 1];
                int64_t b_inner = b_shape[b_shape.ndim() - 2];
                if (a_inner > 0 && b_inner > 0 && a_inner != b_inner) {
                    return false;  // Inner dimensions must match
                }
            }
        }
    }

    // 4. Polyhedral annotation consistency: loop bounds should be non-empty
    //    and access matrices should have correct dimensions
    for (const auto& ir_op : ops_) {
        if (ir_op.poly_annotation.has_value()) {
            const auto& pa = *ir_op.poly_annotation;
            if (pa.loop_bounds.empty()) {
                return false;  // Must have at least one loop dimension
            }
            if (pa.access_matrices.size() != pa.access_offsets.size()) {
                return false;  // Must match
            }
            if (pa.access_matrices.size() != pa.access_is_write.size()) {
                return false;  // Must match
            }
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────

std::string SympleXIR::to_string() const {
    std::ostringstream oss;
    oss << "SympleXIR(" << ops_.size() << " ops):\n";
    for (const auto& ir_op : ops_) {
        oss << "  " << ir_op.to_string() << "\n";
    }
    return oss.str();
}

std::string SympleXIR::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"num_ops\": " << ops_.size() << ",\n";
    oss << "  \"root_id\": " << (ops_.empty() ? -1 : root_id()) << ",\n";
    oss << "  \"ops\": [\n";
    for (size_t i = 0; i < ops_.size(); ++i) {
        const auto& ir_op = ops_[i];
        oss << "    {\n";
        oss << "      \"id\": " << ir_op.id_ << ",\n";
        oss << "      \"kind\": \"" << IROp::kind_to_string(ir_op.kind) << "\",\n";

        // Operands
        oss << "      \"operands\": [";
        for (size_t j = 0; j < ir_op.operands.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << ir_op.operands[j];
        }
        oss << "],\n";

        // Shape
        oss << "      \"shape\": [";
        for (size_t j = 0; j < ir_op.shape.ndim(); ++j) {
            if (j > 0) oss << ", ";
            if (ir_op.shape[j] < 0) oss << "null";
            else oss << ir_op.shape[j];
        }
        oss << "],\n";

        // Dtype
        oss << "      \"dtype\": \"" << irdtype_to_string(ir_op.dtype) << "\",\n";

        // Layout
        oss << "      \"layout\": \"" << irlayout_to_string(ir_op.layout) << "\",\n";

        // Name (for symbols)
        if (!ir_op.name.empty()) {
            oss << "      \"name\": \"" << ir_op.name << "\",\n";
        }

        // Constant value
        if (ir_op.kind == IROp::Kind::CONSTANT) {
            oss << "      \"float_value\": " << ir_op.float_value << ",\n";
            oss << "      \"int_value\": " << ir_op.int_value << ",\n";
        }

        // Axis
        if (ir_op.axis >= 0) {
            oss << "      \"axis\": " << ir_op.axis << ",\n";
        }

        // Polyhedral annotation
        if (ir_op.poly_annotation.has_value()) {
            const auto& pa = *ir_op.poly_annotation;
            oss << "      \"poly_annotation\": {\n";
            oss << "        \"num_dims\": " << pa.num_dims() << ",\n";
            oss << "        \"loop_bounds\": [";
            for (size_t j = 0; j < pa.loop_bounds.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << "[" << pa.loop_bounds[j].first << ", "
                    << pa.loop_bounds[j].second << "]";
            }
            oss << "],\n";
            oss << "        \"num_accesses\": " << pa.num_accesses() << ",\n";
            oss << "        \"parallel_dims\": [";
            for (size_t j = 0; j < pa.parallel_dims.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << (pa.parallel_dims[j] ? "true" : "false");
            }
            oss << "],\n";
            oss << "        \"has_schedule\": "
                << (pa.schedule_matrix.has_value() ? "true" : "false") << "\n";
            oss << "      },\n";
        }

        oss << "      \"is_fused\": " << (ir_op.is_fused() ? "true" : "false") << "\n";
        oss << "    }";
        if (i + 1 < ops_.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";

    // Symbols
    oss << "  \"symbols\": {\n";
    size_t sym_count = 0;
    for (const auto& [name, id] : symbol_map_) {
        if (sym_count > 0) oss << ",\n";
        oss << "    \"" << name << "\": " << id;
        sym_count++;
    }
    oss << "\n  }\n";

    oss << "}\n";
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────
// Conversion (tracer → IR → egraph pipeline)
// ─────────────────────────────────────────────────────────────────────────

SympleXIR SympleXIR::from_trace_graph(
    const std::vector<std::tuple<std::string, std::vector<int64_t>, IRDType>>& params,
    const std::vector<std::tuple<IRShape, int64_t, std::vector<int64_t>,
                                std::string, IRDType, int64_t, double>>& ops)
{
    // Build a SympleXIR from the serialized trace description.
    // Convention: params are added first (IDs 0..P-1), then ops (IDs P..P+O-1).
    // The operand IDs in each op use this same numbering, so we remap
    // from the serialized ID space to the actual IR IDs returned by
    // the add_* methods.

    SympleXIR ir;
    std::unordered_map<int64_t, int64_t> id_remap;

    // ── Add params (symbols) ──────────────────────────────────────────
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& [name, shape, dtype] = params[i];
        int64_t new_id = ir.add_symbol(name, IRShape(shape), dtype);
        id_remap[static_cast<int64_t>(i)] = new_id;
    }

    // ── Add ops ───────────────────────────────────────────────────────
    int64_t base_id = static_cast<int64_t>(params.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& [shape, kind_int, operands, name, dtype, axis, float_value] = ops[i];
        auto kind = static_cast<IROp::Kind>(kind_int);

        // Remap operand IDs from serialized space → actual IR IDs
        std::vector<int64_t> remapped;
        remapped.reserve(operands.size());
        for (auto op_id : operands) {
            auto it = id_remap.find(op_id);
            remapped.push_back(it != id_remap.end() ? it->second : op_id);
        }

        int64_t new_id = -1;

        switch (kind) {
            // ── Leaf nodes ──
            case IROp::Kind::SYMBOL:
                new_id = ir.add_symbol(name, shape, dtype);
                break;

            case IROp::Kind::CONSTANT:
                if (float_value != 0.0) {
                    new_id = ir.add_constant(float_value, dtype);
                } else {
                    // Attempt to recover int value from float_value (which was
                    // set from the original int_value in the tracer).
                    new_id = ir.add_int_constant(
                        static_cast<int64_t>(float_value), dtype);
                }
                break;

            // ── Reductions (need axis) ──
            case IROp::Kind::REDUCE_SUM:
            case IROp::Kind::REDUCE_MAX:
            case IROp::Kind::REDUCE_MEAN:
                if (!remapped.empty()) {
                    new_id = ir.add_reduction(kind, remapped[0], axis);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Axis-parameterized unary ops ──
            case IROp::Kind::SOFTMAX:
                if (!remapped.empty()) {
                    new_id = ir.add_unary_with_axis(kind, remapped[0], axis);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Shape-explicit unary ops ──
            case IROp::Kind::RESHAPE:
            case IROp::Kind::BROADCAST:
                new_id = ir.add_op(kind, remapped, shape, dtype);
                break;

            // ── Binary ops ──
            case IROp::Kind::ADD:
            case IROp::Kind::MUL:
            case IROp::Kind::SUB:
            case IROp::Kind::DIV:
            case IROp::Kind::MATMUL:
                if (remapped.size() >= 2) {
                    new_id = ir.add_binary(kind, remapped[0], remapped[1]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Ternary ops ──
            case IROp::Kind::SELECT:
                if (remapped.size() >= 3) {
                    new_id = ir.add_ternary(kind, remapped[0], remapped[1], remapped[2]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Fused binary ops ──
            case IROp::Kind::FUSED_MATMUL_RELU:
            case IROp::Kind::FUSED_MATMUL_ADD:
                if (remapped.size() >= 2) {
                    new_id = ir.add_binary(kind, remapped[0], remapped[1]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Fused ternary ops ──
            case IROp::Kind::FUSED_MATMUL_ADD_RELU:
            case IROp::Kind::FUSED_GEMM:
            case IROp::Kind::FUSED_ADD_LN:
                if (remapped.size() >= 3) {
                    new_id = ir.add_ternary(kind, remapped[0], remapped[1], remapped[2]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Fused 4-ary ops ──
            case IROp::Kind::FUSED_MHA:
                new_id = ir.add_op(kind, remapped, shape, dtype);
                break;

            // ── Fused unary ops ──
            case IROp::Kind::FUSED_SOFTMAX:
                if (!remapped.empty() && axis >= 0) {
                    new_id = ir.add_unary_with_axis(kind, remapped[0], axis);
                } else if (!remapped.empty()) {
                    new_id = ir.add_unary(kind, remapped[0]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            case IROp::Kind::FUSED_LAYERNORM:
            case IROp::Kind::FUSED_RMSNORM:
                if (!remapped.empty()) {
                    new_id = ir.add_unary(kind, remapped[0]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;

            // ── Plain unary ops ──
            case IROp::Kind::NEG:
            case IROp::Kind::TRANSPOSE:
            case IROp::Kind::RELU:
            case IROp::Kind::GELU:
            case IROp::Kind::SIGMOID:
            case IROp::Kind::EXP:
            case IROp::Kind::LOG:
            case IROp::Kind::SQRT:
            case IROp::Kind::RECIPROCAL:
            case IROp::Kind::DROPOUT:
            case IROp::Kind::LAYERNORM:
            case IROp::Kind::RMSNORM:
            case IROp::Kind::TILE:
            case IROp::Kind::UNTILE:
            case IROp::Kind::IDENTITY:
                if (!remapped.empty()) {
                    new_id = ir.add_unary(kind, remapped[0]);
                } else {
                    new_id = ir.add_op(kind, remapped, shape, dtype);
                }
                break;
        }

        id_remap[base_id + static_cast<int64_t>(i)] = new_id;
    }

    return ir;
}

// ─────────────────────────────────────────────────────────────────────────
// IR → E-Graph Bridge
// ─────────────────────────────────────────────────────────────────────────

namespace {

/// Map IRDType → egraph DType.
optimizer::egraph::DType irdtype_to_egraph(IRDType dt) {
    switch (dt) {
        case IRDType::FP64:    return optimizer::egraph::DType::FP64;
        case IRDType::FP32:    return optimizer::egraph::DType::FP32;
        case IRDType::FP16:    return optimizer::egraph::DType::FP16;
        case IRDType::BF16:    return optimizer::egraph::DType::BF16;
        case IRDType::INT8:    return optimizer::egraph::DType::INT8;
        case IRDType::INT4:    return optimizer::egraph::DType::INT4;
        case IRDType::UNKNOWN: return optimizer::egraph::DType::UNKNOWN;
    }
    return optimizer::egraph::DType::UNKNOWN;
}

/// Map IROp::Kind → egraph OpId.
optimizer::egraph::OpId ir_kind_to_egraph(IROp::Kind kind) {
    switch (kind) {
        case IROp::Kind::SYMBOL:                  return optimizer::egraph::OpId::SYMBOL;
        case IROp::Kind::CONSTANT:                return optimizer::egraph::OpId::CONSTANT;
        case IROp::Kind::ADD:                     return optimizer::egraph::OpId::ADD;
        case IROp::Kind::MUL:                     return optimizer::egraph::OpId::MUL;
        case IROp::Kind::SUB:                     return optimizer::egraph::OpId::SUB;
        case IROp::Kind::DIV:                     return optimizer::egraph::OpId::DIV;
        case IROp::Kind::NEG:                     return optimizer::egraph::OpId::NEG;
        case IROp::Kind::MATMUL:                  return optimizer::egraph::OpId::MATMUL;
        case IROp::Kind::TRANSPOSE:               return optimizer::egraph::OpId::TRANSPOSE;
        case IROp::Kind::RESHAPE:                 return optimizer::egraph::OpId::RESHAPE;
        case IROp::Kind::BROADCAST:               return optimizer::egraph::OpId::BROADCAST;
        case IROp::Kind::REDUCE_SUM:              return optimizer::egraph::OpId::REDUCE_SUM;
        case IROp::Kind::REDUCE_MAX:              return optimizer::egraph::OpId::REDUCE_MAX;
        case IROp::Kind::REDUCE_MEAN:             return optimizer::egraph::OpId::REDUCE_MEAN;
        case IROp::Kind::RELU:                    return optimizer::egraph::OpId::RELU;
        case IROp::Kind::GELU:                    return optimizer::egraph::OpId::GELU;
        case IROp::Kind::SIGMOID:                 return optimizer::egraph::OpId::SIGMOID;
        case IROp::Kind::SOFTMAX:                 return optimizer::egraph::OpId::SOFTMAX;
        case IROp::Kind::LAYERNORM:               return optimizer::egraph::OpId::LAYERNORM;
        case IROp::Kind::RMSNORM:                 return optimizer::egraph::OpId::RMSNORM;
        case IROp::Kind::DROPOUT:                 return optimizer::egraph::OpId::DROPOUT;
        case IROp::Kind::EXP:                     return optimizer::egraph::OpId::EXP;
        case IROp::Kind::LOG:                     return optimizer::egraph::OpId::LOG;
        case IROp::Kind::SQRT:                    return optimizer::egraph::OpId::SQRT;
        case IROp::Kind::RECIPROCAL:              return optimizer::egraph::OpId::RECIPROCAL;
        case IROp::Kind::FUSED_MATMUL_RELU:       return optimizer::egraph::OpId::FUSED_MATMUL_RELU;
        case IROp::Kind::FUSED_MATMUL_ADD:        return optimizer::egraph::OpId::FUSED_MATMUL_ADD;
        case IROp::Kind::FUSED_MATMUL_ADD_RELU:   return optimizer::egraph::OpId::FUSED_MATMUL_ADD_RELU;
        case IROp::Kind::FUSED_GEMM:              return optimizer::egraph::OpId::FUSED_GEMM;
        case IROp::Kind::FUSED_SOFTMAX:           return optimizer::egraph::OpId::FUSED_SOFTMAX;
        case IROp::Kind::FUSED_LAYERNORM:         return optimizer::egraph::OpId::FUSED_LAYERNORM;
        case IROp::Kind::FUSED_RMSNORM:           return optimizer::egraph::OpId::FUSED_RMSNORM;
        case IROp::Kind::FUSED_ADD_LN:            return optimizer::egraph::OpId::FUSED_ADD_LN;
        case IROp::Kind::FUSED_MHA:               return optimizer::egraph::OpId::FUSED_MHA;
        case IROp::Kind::TILE:                    return optimizer::egraph::OpId::TILE;
        case IROp::Kind::UNTILE:                  return optimizer::egraph::OpId::UNTILE;
        case IROp::Kind::IDENTITY:                return optimizer::egraph::OpId::IDENTITY;
        case IROp::Kind::SELECT:                  return optimizer::egraph::OpId::IDENTITY; // no SELECT in egraph; fall back
    }
    return optimizer::egraph::OpId::IDENTITY;
}

} // anonymous namespace

std::pair<void*, int64_t> SympleXIR::to_egraph() const {
    if (ops_.empty()) {
        return {nullptr, -1};
    }

    auto* eg = new optimizer::egraph::EGraph();

    // Map from IR op ID → egraph class ID
    std::unordered_map<int64_t, optimizer::egraph::ClassId> ir_to_eg;

    for (const auto& ir_op : ops_) {
        optimizer::egraph::ClassId cid;

        if (ir_op.kind == IROp::Kind::SYMBOL) {
            // Add as a symbol with shape/type analysis
            auto eshape = optimizer::egraph::TensorShape(ir_op.shape.dims);
            cid = eg->add_symbol(ir_op.name, eshape,
                                 irdtype_to_egraph(ir_op.dtype));

        } else if (ir_op.kind == IROp::Kind::CONSTANT) {
            // Add as typed constant
            if (ir_op.float_value != 0.0) {
                cid = eg->add_float_constant(ir_op.float_value,
                                             irdtype_to_egraph(ir_op.dtype));
            } else {
                cid = eg->add_constant(ir_op.int_value,
                                       irdtype_to_egraph(ir_op.dtype));
            }

        } else {
            // Build an ENode for the operation
            optimizer::egraph::ENode enode;
            enode.op   = ir_kind_to_egraph(ir_op.kind);
            enode.name = ir_op.name;
            enode.axis = ir_op.axis;

            // Map operand IR IDs → egraph class IDs
            for (auto operand_id : ir_op.operands) {
                auto it = ir_to_eg.find(operand_id);
                if (it != ir_to_eg.end()) {
                    enode.children.push_back(it->second);
                }
                // If not found, skip (shouldn't happen for well-formed IR)
            }

            cid = eg->add_node(enode);
        }

        ir_to_eg[ir_op.id_] = cid;
    }

    // The root class ID corresponds to the last op
    int64_t root_class = -1;
    if (!ops_.empty()) {
        auto it = ir_to_eg.find(root_id());
        if (it != ir_to_eg.end()) {
            root_class = it->second;
        }
    }

    return {static_cast<void*>(eg), root_class};
}

void SympleXIR::apply_extraction_result(int64_t root_class_id, double cost) {
    // Record the extraction metadata.  A full reconstruction of the
    // IR from the e-graph's extracted expression tree requires access
    // to the EGraph object (which is held by the caller).  The bridge
    // module (ir/egraph_bridge.cpp) is responsible for walking the
    // ExtractionResult and calling add_* methods to rebuild the IR.
    //
    // Here we store the provenance so that downstream passes can
    // query whether the IR has been through e-graph optimization.

    extraction_root_class_ = root_class_id;
    extraction_cost_ = cost;
}

// ─────────────────────────────────────────────────────────────────────────
// Analysis
// ─────────────────────────────────────────────────────────────────────────

std::vector<int64_t> SympleXIR::ops_of_kind(IROp::Kind kind) const {
    std::vector<int64_t> result;
    for (const auto& ir_op : ops_) {
        if (ir_op.kind == kind) {
            result.push_back(ir_op.id_);
        }
    }
    return result;
}

std::vector<int64_t> SympleXIR::ops_with_poly_annotations() const {
    std::vector<int64_t> result;
    for (const auto& ir_op : ops_) {
        if (ir_op.poly_annotation.has_value()) {
            result.push_back(ir_op.id_);
        }
    }
    return result;
}

std::vector<int64_t> SympleXIR::topological_order() const {
    // For well-formed SSA, the insertion order IS the topological order.
    // We verify this via Kahn's algorithm (same as validate).
    std::vector<int64_t> order;
    order.reserve(ops_.size());

    std::vector<int> in_degree(ops_.size(), 0);
    std::vector<std::vector<int64_t>> adj(ops_.size());
    for (const auto& ir_op : ops_) {
        for (auto operand_id : ir_op.operands) {
            adj[operand_id].push_back(ir_op.id_);
            in_degree[ir_op.id_]++;
        }
    }

    std::queue<int64_t> queue;
    for (size_t i = 0; i < ops_.size(); ++i) {
        if (in_degree[i] == 0) {
            queue.push(static_cast<int64_t>(i));
        }
    }

    while (!queue.empty()) {
        int64_t id = queue.front();
        queue.pop();
        order.push_back(id);
        for (auto next : adj[id]) {
            in_degree[next]--;
            if (in_degree[next] == 0) {
                queue.push(next);
            }
        }
    }

    return order;
}

size_t SympleXIR::count_fused_ops() const {
    size_t count = 0;
    for (const auto& ir_op : ops_) {
        if (ir_op.is_fused()) {
            count++;
        }
    }
    return count;
}

int64_t SympleXIR::estimate_flops() const {
    int64_t total_flops = 0;
    for (const auto& ir_op : ops_) {
        int64_t nelem = ir_op.shape.num_elements();
        if (nelem < 0) nelem = 1;  // Conservative for symbolic shapes

        switch (ir_op.kind) {
            case IROp::Kind::MATMUL: {
                // FLOPs for matmul: 2 * M * N * K
                if (ir_op.operands.size() >= 2) {
                    const auto& a_shape = ops_[ir_op.operands[0]].shape;
                    if (a_shape.ndim() >= 2) {
                        int64_t K = a_shape[a_shape.ndim() - 1];
                        if (K < 0) K = 1;
                        total_flops += 2 * nelem * K;
                    }
                }
                break;
            }

            case IROp::Kind::FUSED_MATMUL_RELU:
            case IROp::Kind::FUSED_MATMUL_ADD:
            case IROp::Kind::FUSED_MATMUL_ADD_RELU:
            case IROp::Kind::FUSED_GEMM: {
                // Matmul + elementwise: 2*M*N*K + M*N
                if (ir_op.operands.size() >= 2) {
                    const auto& a_shape = ops_[ir_op.operands[0]].shape;
                    if (a_shape.ndim() >= 2) {
                        int64_t K = a_shape[a_shape.ndim() - 1];
                        if (K < 0) K = 1;
                        total_flops += 2 * nelem * K + nelem;
                    }
                }
                break;
            }

            case IROp::Kind::ADD:
            case IROp::Kind::MUL:
            case IROp::Kind::SUB:
            case IROp::Kind::DIV:
            case IROp::Kind::NEG:
                total_flops += nelem;
                break;

            case IROp::Kind::RELU:
            case IROp::Kind::GELU:
            case IROp::Kind::SIGMOID:
            case IROp::Kind::EXP:
            case IROp::Kind::LOG:
            case IROp::Kind::SQRT:
            case IROp::Kind::RECIPROCAL:
                total_flops += nelem;
                break;

            case IROp::Kind::SOFTMAX:
            case IROp::Kind::FUSED_SOFTMAX:
                // Softmax: exp + sum + div = ~3 * nelem
                total_flops += 3 * nelem;
                break;

            case IROp::Kind::LAYERNORM:
            case IROp::Kind::FUSED_LAYERNORM:
            case IROp::Kind::RMSNORM:
            case IROp::Kind::FUSED_RMSNORM:
            case IROp::Kind::FUSED_ADD_LN:
                // Norm: mean/var/gamma/beta = ~5 * nelem
                total_flops += 5 * nelem;
                break;

            case IROp::Kind::REDUCE_SUM:
            case IROp::Kind::REDUCE_MAX:
            case IROp::Kind::REDUCE_MEAN:
                total_flops += nelem;
                break;

            case IROp::Kind::FUSED_MHA:
                // MHA: Q*K^T + softmax + *V = ~4*M*N*K
                if (ir_op.operands.size() >= 2) {
                    const auto& a_shape = ops_[ir_op.operands[0]].shape;
                    if (a_shape.ndim() >= 2) {
                        int64_t K = a_shape[a_shape.ndim() - 1];
                        if (K < 0) K = 1;
                        total_flops += 4 * nelem * K;
                    }
                }
                break;

            default:
                // Leaf ops, transpose, reshape, etc. — negligible FLOPs
                break;
        }
    }
    return total_flops;
}

} // namespace symplex::ir
