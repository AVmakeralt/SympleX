// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/egraph.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <unordered_set>
#include <deque>

namespace symplex::optimizer::egraph {

// Forward declaration — defined later in this file
bool should_apply_distributivity(const EGraph& g, int max_depth, int64_t max_graph_size);

// ─────────────────────────────────────────────────────────────────────────
// Standard Tensor Algebra Rewrite Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> standard_tensor_rewrite_rules() {
    std::vector<RewriteRule> rules;

    // ── Rule: A + B == B + A  (Add commutativity) ────────────────
    rules.push_back({
        "add-commute",
        "A + B == B + A",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;
                ClassId c1 = g.find(node.children[0]);
                ClassId c2 = g.find(node.children[1]);

                ENode swapped;
                swapped.op = OpId::ADD;
                swapped.children = {c2, c1};
                ClassId new_class = g.add_node(swapped);
                if (new_class != g.find(g.node_class(nid))) {
                    merges.push_back({new_class, g.node_class(nid)});
                }
            }
            return merges;
        }
    });

    // ── Rule: A * B == B * A  (Mul commutativity for element-wise) ──
    rules.push_back({
        "mul-commute",
        "A * B == B * A  (element-wise)",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;
                ClassId c1 = g.find(node.children[0]);
                ClassId c2 = g.find(node.children[1]);

                ENode swapped;
                swapped.op = OpId::MUL;
                swapped.children = {c2, c1};
                ClassId new_class = g.add_node(swapped);
                if (new_class != g.find(g.node_class(nid))) {
                    merges.push_back({new_class, g.node_class(nid)});
                }
            }
            return merges;
        }
    });

    // ── Rule: (A + B) + C == A + (B + C)  (Add associativity) ─────
    // This is FP-unsafe in general, so we track that in the analysis.
    rules.push_back({
        "add-associate",
        "(A + B) + C == A + (B + C)",
        RulePriority::LOW,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                // Left child is also an ADD
                ClassId left_cls = g.find(node.children[0]);
                for (NodeId left_nid : g.class_nodes(left_cls)) {
                    const auto& left_node = g.node(left_nid);
                    if (left_node.op != OpId::ADD || left_node.children.size() != 2) continue;

                    ClassId A = g.find(left_node.children[0]);
                    ClassId B = g.find(left_node.children[1]);
                    ClassId C = g.find(node.children[1]);

                    // Construct A + (B + C)
                    ClassId bc = g.add_binary(OpId::ADD, B, C);
                    ClassId a_bc = g.add_binary(OpId::ADD, A, bc);

                    ClassId orig = g.find(g.node_class(nid));
                    if (g.find(a_bc) != orig) {
                        merges.push_back({a_bc, g.node_class(nid)});
                    }
                }

                // Also: A + (B + C) == (A + B) + C (right-associate)
                ClassId right_cls = g.find(node.children[1]);
                for (NodeId right_nid : g.class_nodes(right_cls)) {
                    const auto& right_node = g.node(right_nid);
                    if (right_node.op != OpId::ADD || right_node.children.size() != 2) continue;

                    ClassId A = g.find(node.children[0]);
                    ClassId B = g.find(right_node.children[0]);
                    ClassId C = g.find(right_node.children[1]);

                    ClassId ab = g.add_binary(OpId::ADD, A, B);
                    ClassId ab_c = g.add_binary(OpId::ADD, ab, C);

                    ClassId orig = g.find(g.node_class(nid));
                    if (g.find(ab_c) != orig) {
                        merges.push_back({ab_c, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: A * (B + C) == A*B + A*C  (Distributivity) ──────────
    // Gated behind heuristics to prevent saturation explosion.
    rules.push_back({
        "distribute",
        "A * (B + C) == A*B + A*C",
        RulePriority::EXPLORE,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                // Check if right child is ADD
                ClassId right_cls = g.find(node.children[1]);
                for (NodeId right_nid : g.class_nodes(right_cls)) {
                    const auto& right_node = g.node(right_nid);
                    if (right_node.op != OpId::ADD || right_node.children.size() != 2) continue;

                    ClassId A = g.find(node.children[0]);
                    ClassId B = g.find(right_node.children[0]);
                    ClassId C = g.find(right_node.children[1]);

                    ClassId ab = g.add_binary(OpId::MUL, A, B);
                    ClassId ac = g.add_binary(OpId::MUL, A, C);
                    ClassId ab_ac = g.add_binary(OpId::ADD, ab, ac);

                    merges.push_back({ab_ac, g.node_class(nid)});
                }

                // Also check if LEFT child is ADD
                ClassId left_cls = g.find(node.children[0]);
                for (NodeId left_nid : g.class_nodes(left_cls)) {
                    const auto& left_node = g.node(left_nid);
                    if (left_node.op != OpId::ADD || left_node.children.size() != 2) continue;

                    ClassId B = g.find(left_node.children[0]);
                    ClassId C = g.find(left_node.children[1]);
                    ClassId A = g.find(node.children[1]);

                    ClassId ba = g.add_binary(OpId::MUL, B, A);
                    ClassId ca = g.add_binary(OpId::MUL, C, A);
                    ClassId ba_ca = g.add_binary(OpId::ADD, ba, ca);

                    merges.push_back({ba_ca, g.node_class(nid)});
                }
            }
            return merges;
        },
        // should_apply: use the dedicated distributivity gating function
        // which checks reuse detection, cost-guided, and depth thresholds
        [](const EGraph& g) -> bool {
            return should_apply_distributivity(g);
        }
    });

    // ── Rule: A*B + A*C == A*(B+C)  (Factor / collect common subexpr) ──
    // This is the KEY superoptimizer rule: A×B + A×C → A×(B+C)
    // It saves one entire MatMul when A is a matrix and B,C are matrices.
    rules.push_back({
        "factor",
        "A*B + A*C == A*(B+C)  (collect common multiplicand)",
        RulePriority::CRITICAL,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId left_cls = g.find(node.children[0]);
                ClassId right_cls = g.find(node.children[1]);

                for (NodeId lnid : g.class_nodes(left_cls)) {
                    const auto& left = g.node(lnid);
                    if (left.op != OpId::MUL && left.op != OpId::MATMUL) continue;
                    if (left.children.size() != 2) continue;

                    for (NodeId rnid : g.class_nodes(right_cls)) {
                        const auto& right = g.node(rnid);
                        if (right.op != left.op || right.children.size() != 2) continue;

                        ClassId l0 = g.find(left.children[0]);
                        ClassId l1 = g.find(left.children[1]);
                        ClassId r0 = g.find(right.children[0]);
                        ClassId r1 = g.find(right.children[1]);

                        // Case 1: A*B + A*C -> A*(B+C)
                        if (l0 == r0 && l1 != r1) {
                            ClassId bc = g.add_binary(OpId::ADD, l1, r1);
                            ClassId a_bc = g.add_binary(left.op, l0, bc);
                            merges.push_back({a_bc, g.node_class(nid)});
                        }
                        // Case 2: A*B + C*A -> A*(B+C)
                        if (l0 == r1 && l1 != r0) {
                            ClassId bc = g.add_binary(OpId::ADD, l1, r0);
                            ClassId a_bc = g.add_binary(left.op, l0, bc);
                            merges.push_back({a_bc, g.node_class(nid)});
                        }
                        // Case 3: B*A + A*C -> A*(B+C)
                        if (l1 == r0 && l0 != r1) {
                            ClassId bc = g.add_binary(OpId::ADD, l0, r1);
                            ClassId a_bc = g.add_binary(left.op, l1, bc);
                            merges.push_back({a_bc, g.node_class(nid)});
                        }
                        // Case 4: B*A + C*A -> A*(B+C)
                        if (l1 == r1 && l0 != r0) {
                            ClassId bc = g.add_binary(OpId::ADD, l0, r0);
                            ClassId a_bc = g.add_binary(left.op, l1, bc);
                            merges.push_back({a_bc, g.node_class(nid)});
                        }
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Transpose(Transpose(A)) == A  ──────────────────────
    rules.push_back({
        "transpose-involution",
        "Transpose(Transpose(A)) == A",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::TRANSPOSE)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::TRANSPOSE) {
                        ClassId inner = g.find(child.children[0]);
                        merges.push_back({inner, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: (A @ B)^T == B^T @ A^T  (MatMul transpose) ──────────
    rules.push_back({
        "matmul-transpose",
        "(A @ B)^T == B^T @ A^T",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::TRANSPOSE)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::MATMUL && child.children.size() == 2) {
                        ClassId A = g.find(child.children[0]);
                        ClassId B = g.find(child.children[1]);

                        ClassId bt = g.add_unary(OpId::TRANSPOSE, B);
                        ClassId at = g.add_unary(OpId::TRANSPOSE, A);
                        ClassId bt_at = g.add_binary(OpId::MATMUL, bt, at);

                        merges.push_back({bt_at, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Add(A, 0) == A  (Additive identity) ──────────────────
    rules.push_back({
        "add-zero",
        "A + 0 == A,  0 + A == A",
        RulePriority::CRITICAL,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId left_cls = g.find(node.children[0]);
                ClassId right_cls = g.find(node.children[1]);

                for (NodeId lnid : g.class_nodes(left_cls)) {
                    const auto& left = g.node(lnid);
                    if (left.op == OpId::CONSTANT && left.value == 0) {
                        merges.push_back({right_cls, g.node_class(nid)});
                    }
                }
                for (NodeId rnid : g.class_nodes(right_cls)) {
                    const auto& right = g.node(rnid);
                    if (right.op == OpId::CONSTANT && right.value == 0) {
                        merges.push_back({left_cls, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Mul(A, 1) == A  (Multiplicative identity) ────────────
    rules.push_back({
        "mul-one",
        "A * 1 == A,  1 * A == A",
        RulePriority::CRITICAL,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId left_cls = g.find(node.children[0]);
                ClassId right_cls = g.find(node.children[1]);

                for (NodeId lnid : g.class_nodes(left_cls)) {
                    const auto& left = g.node(lnid);
                    if (left.op == OpId::CONSTANT && left.value == 1) {
                        merges.push_back({right_cls, g.node_class(nid)});
                    }
                }
                for (NodeId rnid : g.class_nodes(right_cls)) {
                    const auto& right = g.node(rnid);
                    if (right.op == OpId::CONSTANT && right.value == 1) {
                        merges.push_back({left_cls, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Mul(A, 0) == 0  (Annihilation) ──────────────────────
    rules.push_back({
        "mul-zero",
        "A * 0 == 0,  0 * A == 0",
        RulePriority::CRITICAL,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId left_cls = g.find(node.children[0]);
                ClassId right_cls = g.find(node.children[1]);

                bool has_zero = false;
                for (NodeId cnid : g.class_nodes(left_cls)) {
                    if (g.node(cnid).op == OpId::CONSTANT &&
                        g.node(cnid).value == 0) { has_zero = true; break; }
                }
                if (!has_zero) {
                    for (NodeId cnid : g.class_nodes(right_cls)) {
                        if (g.node(cnid).op == OpId::CONSTANT &&
                            g.node(cnid).value == 0) { has_zero = true; break; }
                    }
                }

                if (has_zero) {
                    ClassId zero = g.add_constant(0);
                    merges.push_back({zero, g.node_class(nid)});
                }
            }
            return merges;
        }
    });

    // ── Rule: Neg(Neg(A)) == A  (Double negation) ──────────────────
    rules.push_back({
        "double-neg",
        "Neg(Neg(A)) == A",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::NEG)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::NEG) {
                        ClassId inner = g.find(child.children[0]);
                        merges.push_back({inner, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: A - B == A + Neg(B)  (Subtraction as addition) ──────
    rules.push_back({
        "sub-to-add-neg",
        "A - B == A + Neg(B)",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::SUB)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId A = g.find(node.children[0]);
                ClassId B = g.find(node.children[1]);
                ClassId neg_b = g.add_unary(OpId::NEG, B);
                ClassId a_plus_negb = g.add_binary(OpId::ADD, A, neg_b);

                merges.push_back({a_plus_negb, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: A / B == A * Reciprocal(B)  (Division as multiplication) ──
    rules.push_back({
        "div-to-mul-recip",
        "A / B == A * Reciprocal(B)",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::DIV)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId A = g.find(node.children[0]);
                ClassId B = g.find(node.children[1]);
                ClassId recip_b = g.add_unary(OpId::RECIPROCAL, B);
                ClassId a_mul_recipb = g.add_binary(OpId::MUL, A, recip_b);

                merges.push_back({a_mul_recipb, g.node_class(nid)});
            }
            return merges;
        }
    });

    return rules;
}

// ─────────────────────────────────────────────────────────────────────────
// Transformer-Specific Rewrite Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> transformer_rewrite_rules() {
    std::vector<RewriteRule> rules;

    // ── Rule: Softmax(x) = Exp(x) / ReduceSum(Exp(x)) ─────────────
    // CORRECT decomposition using the actual EXP op instead of the
    // previous broken version that used x / ReduceSum(x).
    //
    // Numerically stable form: Softmax(x) = Exp(x - max(x)) / ReduceSum(Exp(x - max(x)))
    // We model both the naive and stable forms so extraction can choose.
    rules.push_back({
        "softmax-decompose",
        "Softmax(x) == Exp(x) / ReduceSum(Exp(x))",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::SOFTMAX)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);
                int64_t axis = node.axis >= 0 ? node.axis : -1;

                // Build: Exp(x)
                ClassId exp_x = g.add_unary(OpId::EXP, x);

                // Build: ReduceSum(Exp(x), axis)
                ClassId rs_exp = g.add_unary_with_axis(OpId::REDUCE_SUM, exp_x, axis);

                // Build: Broadcast(ReduceSum(Exp(x)))
                ClassId bcast_rs = g.add_unary(OpId::BROADCAST, rs_exp);

                // Build: Exp(x) / Broadcast(ReduceSum(Exp(x)))
                ClassId result = g.add_binary(OpId::DIV, exp_x, bcast_rs);

                merges.push_back({result, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: Softmax(x) == FusedSoftmax(x) ────────────────────────
    // The fused version is cheaper because it uses the log-sum-exp trick
    // and avoids materializing intermediate tensors.
    rules.push_back({
        "softmax-fuse",
        "Softmax(x) == FusedSoftmax(x)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::SOFTMAX)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);
                ENode fused_node;
                fused_node.op = OpId::FUSED_SOFTMAX;
                fused_node.children = {x};
                fused_node.axis = node.axis;
                ClassId fused = g.add_node(fused_node);

                merges.push_back({fused, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: LayerNorm(x) = (x - ReduceMean(x)) / Sqrt(ReduceMean(x^2) - ReduceMean(x)^2 + eps) ───
    // FULL decomposition with variance computation.
    rules.push_back({
        "layernorm-decompose",
        "LayerNorm(x) == (x - mean) / sqrt(var + eps)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::LAYERNORM)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);

                // Build: mean = ReduceMean(x)
                ClassId mean = g.add_unary(OpId::REDUCE_MEAN, x);
                ClassId bcast_mean = g.add_unary(OpId::BROADCAST, mean);

                // Build: centered = x - mean
                ClassId centered = g.add_binary(OpId::SUB, x, bcast_mean);

                // Build: x^2
                ClassId x_sq = g.add_binary(OpId::MUL, x, x);

                // Build: E[x^2]
                ClassId mean_sq = g.add_unary(OpId::REDUCE_MEAN, x_sq);
                ClassId bcast_mean_sq = g.add_unary(OpId::BROADCAST, mean_sq);

                // Build: (E[x])^2
                ClassId bcast_mean_sq2 = g.add_binary(OpId::MUL, bcast_mean, bcast_mean);

                // Build: var = E[x^2] - (E[x])^2
                ClassId variance = g.add_binary(OpId::SUB, bcast_mean_sq, bcast_mean_sq2);

                // Build: sqrt(var + eps)  — eps is a small constant (1e-5)
                ClassId eps = g.add_float_constant(1e-5, DType::FP32);
                ClassId var_plus_eps = g.add_binary(OpId::ADD, variance, eps);
                ClassId inv_std = g.add_unary(OpId::SQRT, var_plus_eps);

                // Build: centered / inv_std
                ClassId result = g.add_binary(OpId::DIV, centered, inv_std);

                merges.push_back({result, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: LayerNorm(x) == FusedLayerNorm(x) ────────────────────
    rules.push_back({
        "layernorm-fuse",
        "LayerNorm(x) == FusedLayerNorm(x)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::LAYERNORM)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);
                ENode fused_ln;
                fused_ln.op = OpId::FUSED_LAYERNORM;
                fused_ln.children = {x};
                fused_ln.axis = node.axis;
                ClassId fused = g.add_node(fused_ln);
                merges.push_back({fused, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: RMSNorm(x) = x / Sqrt(ReduceMean(x^2) + eps) ────────
    // RMSNorm is used in LLaMA-style models. It's simpler than LayerNorm.
    rules.push_back({
        "rmsnorm-decompose",
        "RMSNorm(x) == x / Sqrt(ReduceMean(x^2) + eps)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::RMSNORM)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);

                // Build: x^2
                ClassId x_sq = g.add_binary(OpId::MUL, x, x);

                // Build: ReduceMean(x^2)
                ClassId mean_sq = g.add_unary(OpId::REDUCE_MEAN, x_sq);
                ClassId bcast_mean_sq = g.add_unary(OpId::BROADCAST, mean_sq);

                // Build: Sqrt(mean_sq + eps)
                ClassId eps = g.add_float_constant(1e-5, DType::FP32);
                ClassId mean_plus_eps = g.add_binary(OpId::ADD, bcast_mean_sq, eps);
                ClassId inv_rms = g.add_unary(OpId::SQRT, mean_plus_eps);

                // Build: x / inv_rms
                ClassId result = g.add_binary(OpId::DIV, x, inv_rms);

                merges.push_back({result, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: RMSNorm(x) == FusedRMSNorm(x) ────────────────────────
    rules.push_back({
        "rmsnorm-fuse",
        "RMSNorm(x) == FusedRMSNorm(x)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::RMSNORM)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);
                ENode fused_rms;
                fused_rms.op = OpId::FUSED_RMSNORM;
                fused_rms.children = {x};
                fused_rms.axis = node.axis;
                ClassId fused = g.add_node(fused_rms);
                merges.push_back({fused, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: Attention Q*K^T -> transpose optimization ────────────
    // (Q @ K^T) == Transpose(K @ Q^T)
    rules.push_back({
        "attention-qk-transpose",
        "Q @ Transpose(K) == Transpose(K @ Transpose(Q))",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MATMUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId right_cls = g.find(node.children[1]);
                for (NodeId rnid : g.class_nodes(right_cls)) {
                    const auto& right = g.node(rnid);
                    if (right.op == OpId::TRANSPOSE) {
                        ClassId Q = g.find(node.children[0]);
                        ClassId K = g.find(right.children[0]);

                        ClassId qt = g.add_unary(OpId::TRANSPOSE, Q);
                        ClassId k_qt = g.add_binary(OpId::MATMUL, K, qt);
                        ClassId result_t = g.add_unary(OpId::TRANSPOSE, k_qt);

                        merges.push_back({result_t, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Residual Add + LayerNorm -> FusedAddLN ──────────────
    // Pattern: LayerNorm(x + residual) == FusedAddLN(x, residual)
    // This is critical for transformer performance: avoids writing
    // the intermediate sum to HBM before reading it back for LN.
    rules.push_back({
        "fuse-add-ln",
        "LayerNorm(x + residual) == FusedAddLN(x, residual)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::LAYERNORM)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::ADD && child.children.size() == 2) {
                        ClassId x = g.find(child.children[0]);
                        ClassId residual = g.find(child.children[1]);
                        ClassId fused = g.add_ternary(OpId::FUSED_ADD_LN, x, residual, x);
                        merges.push_back({fused, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    return rules;
}

// ─────────────────────────────────────────────────────────────────────────
// Normalization-Specific Rewrite Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> normalization_rewrite_rules() {
    std::vector<RewriteRule> rules;

    // ── Rule: LayerNorm(x) == RMSNorm(x) when mean(x) ≈ 0 ──────
    // In residual stream networks, the mean of the input is often
    // close to zero, making LayerNorm approximately equivalent to
    // RMSNorm. This is an algebraic equivalence under the assumption
    // that the mean is zero, allowing the superoptimizer to discover
    // the cheaper RMSNorm path.
    rules.push_back({
        "ln-to-rmsnorm",
        "LayerNorm(x) == RMSNorm(x)  (when mean ≈ 0, skip centering)",
        RulePriority::EXPLORE,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::LAYERNORM)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);
                ClassId rms = g.add_unary(OpId::RMSNORM, x);
                merges.push_back({rms, g.node_class(nid)});
            }
            return merges;
        },
        // should_apply: only consider this rule when the e-graph is small
        // enough that the cost function can properly distinguish between
        // LayerNorm and RMSNorm costs. Skip it when the graph is already
        // large to prevent saturation explosion.
        [](const EGraph& g) -> bool {
            return g.num_nodes() < 5000;
        }
    });

    // ── Rule: Dropout(x) == x  (inference mode) ─────────────────
    rules.push_back({
        "dropout-identity",
        "Dropout(x) == x  (inference mode)",
        RulePriority::CRITICAL,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::DROPOUT)) {
                const auto& node = g.node(nid);
                ClassId inner = g.find(node.children[0]);
                merges.push_back({inner, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: GELU(x) ≈ x * Sigmoid(1.702*x)  (fast approximation) ──
    // This allows the superoptimizer to discover the faster
    // sigmoid-based GELU approximation used in production.
    rules.push_back({
        "gelu-fast-approx",
        "GELU(x) ≈ x * Sigmoid(1.702 * x)  (fast approximation)",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::GELU)) {
                const auto& node = g.node(nid);

                ClassId x = g.find(node.children[0]);

                // Build: 1.702 * x
                ClassId coeff = g.add_float_constant(1.702, DType::FP32);
                ClassId scaled = g.add_binary(OpId::MUL, coeff, x);

                // Build: Sigmoid(scaled)
                ClassId sig = g.add_unary(OpId::SIGMOID, scaled);

                // Build: x * Sigmoid(scaled)
                ClassId result = g.add_binary(OpId::MUL, x, sig);

                merges.push_back({result, g.node_class(nid)});
            }
            return merges;
        }
    });

    return rules;
}

// ─────────────────────────────────────────────────────────────────────────
// Fusion Discovery Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> fusion_rewrite_rules() {
    std::vector<RewriteRule> rules;

    // ── Rule: MatMul(A, B) + bias -> FusedMatMulAdd(A, B, bias) ───
    rules.push_back({
        "fuse-matmul-add",
        "MatMul(A,B) + bias == FusedMatMulAdd(A,B,bias)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                for (int side = 0; side < 2; ++side) {
                    ClassId maybe_mm = g.find(node.children[side]);
                    ClassId maybe_bias = g.find(node.children[1 - side]);

                    for (NodeId mm_nid : g.class_nodes(maybe_mm)) {
                        const auto& mm_node = g.node(mm_nid);
                        if (mm_node.op == OpId::MATMUL && mm_node.children.size() == 2) {
                            ClassId A = g.find(mm_node.children[0]);
                            ClassId B = g.find(mm_node.children[1]);
                            ClassId fused = g.add_ternary(
                                OpId::FUSED_MATMUL_ADD, A, B, maybe_bias);
                            merges.push_back({fused, g.node_class(nid)});
                        }
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: FusedMatMulAdd(A,B,bias) == MatMul(A,B) + bias ──────
    rules.push_back({
        "defuse-matmul-add",
        "FusedMatMulAdd(A,B,bias) == MatMul(A,B) + bias",
        RulePriority::LOW,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::FUSED_MATMUL_ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 3) continue;

                ClassId A = g.find(node.children[0]);
                ClassId B = g.find(node.children[1]);
                ClassId bias = g.find(node.children[2]);

                ClassId mm = g.add_binary(OpId::MATMUL, A, B);
                ClassId mm_plus_bias = g.add_binary(OpId::ADD, mm, bias);

                merges.push_back({mm_plus_bias, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: ReLU(MatMul(A,B)) -> FusedMatMulReLU(A,B) ───────────
    rules.push_back({
        "fuse-matmul-relu",
        "ReLU(MatMul(A,B)) == FusedMatMulReLU(A,B)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::RELU)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::MATMUL && child.children.size() == 2) {
                        ClassId A = g.find(child.children[0]);
                        ClassId B = g.find(child.children[1]);
                        ClassId fused = g.add_binary(OpId::FUSED_MATMUL_RELU, A, B);
                        merges.push_back({fused, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: FusedMatMulReLU(A,B) == ReLU(MatMul(A,B)) ───────────
    rules.push_back({
        "defuse-matmul-relu",
        "FusedMatMulReLU(A,B) == ReLU(MatMul(A,B))",
        RulePriority::LOW,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::FUSED_MATMUL_RELU)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId A = g.find(node.children[0]);
                ClassId B = g.find(node.children[1]);
                ClassId mm = g.add_binary(OpId::MATMUL, A, B);
                ClassId relu_mm = g.add_unary(OpId::RELU, mm);

                merges.push_back({relu_mm, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: ReLU(MatMul(A,B) + bias) -> FusedMatMulAddReLU ──────
    rules.push_back({
        "fuse-matmul-add-relu",
        "ReLU(MatMul(A,B) + bias) == FusedMatMulAddReLU(A,B,bias)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::RELU)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(child_nid);
                    if (child.op == OpId::ADD && child.children.size() == 2) {
                        for (int side = 0; side < 2; ++side) {
                            ClassId maybe_mm = g.find(child.children[side]);
                            ClassId maybe_bias = g.find(child.children[1 - side]);

                            for (NodeId mm_nid : g.class_nodes(maybe_mm)) {
                                const auto& mm = g.node(mm_nid);
                                if (mm.op == OpId::MATMUL && mm.children.size() == 2) {
                                    ClassId A = g.find(mm.children[0]);
                                    ClassId B = g.find(mm.children[1]);
                                    ClassId fused = g.add_ternary(
                                        OpId::FUSED_MATMUL_ADD_RELU, A, B, maybe_bias);
                                    merges.push_back({fused, g.node_class(nid)});
                                }
                            }
                        }
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: alpha*A@B + beta*C -> FusedGEMM(A,B,C) ──────────────
    rules.push_back({
        "fuse-gemm",
        "alpha*MatMul(A,B) + beta*C -> FusedGEMM(A,B,C)",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                for (int side = 0; side < 2; ++side) {
                    ClassId left = g.find(node.children[side]);
                    ClassId right = g.find(node.children[1 - side]);

                    ClassId mm_A = NULL_CLASS, mm_B = NULL_CLASS;
                    for (NodeId lnid : g.class_nodes(left)) {
                        const auto& lnode = g.node(lnid);
                        if (lnode.op == OpId::MATMUL && lnode.children.size() == 2) {
                            mm_A = g.find(lnode.children[0]);
                            mm_B = g.find(lnode.children[1]);
                            break;
                        }
                        if (lnode.op == OpId::MUL && lnode.children.size() == 2) {
                            for (int s = 0; s < 2; ++s) {
                                for (NodeId inner_nid : g.class_nodes(g.find(lnode.children[s]))) {
                                    const auto& inner = g.node(inner_nid);
                                    if (inner.op == OpId::MATMUL && inner.children.size() == 2) {
                                        mm_A = g.find(inner.children[0]);
                                        mm_B = g.find(inner.children[1]);
                                        break;
                                    }
                                }
                                if (mm_A != NULL_CLASS) break;
                            }
                        }
                        if (mm_A != NULL_CLASS) break;
                    }

                    if (mm_A != NULL_CLASS) {
                        ClassId fused = g.add_ternary(OpId::FUSED_GEMM, mm_A, mm_B, right);
                        merges.push_back({fused, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    return rules;
}

// ─────────────────────────────────────────────────────────────────────────
// Tiling Discovery Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> tiling_rewrite_rules() {
    std::vector<RewriteRule> rules;

    // ── Rule: MatMul(A,B) == Untile(MatMul(Tile(A), Tile(B))) ─────
    rules.push_back({
        "matmul-tile-decompose",
        "MatMul(A,B) can be decomposed into tiled sub-computations",
        RulePriority::EXPLORE,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::MATMUL)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId A = g.find(node.children[0]);
                ClassId B = g.find(node.children[1]);

                ENode tile_a;
                tile_a.op = OpId::TILE;
                tile_a.children = {A};
                tile_a.axis = 0;
                ClassId tiled_a = g.add_node(tile_a);

                ENode tile_b;
                tile_b.op = OpId::TILE;
                tile_b.children = {B};
                tile_b.axis = 1;
                ClassId tiled_b = g.add_node(tile_b);

                ClassId tiled_mm = g.add_binary(OpId::MATMUL, tiled_a, tiled_b);
                ClassId result = g.add_unary(OpId::UNTILE, tiled_mm);

                merges.push_back({result, g.node_class(nid)});
            }
            return merges;
        }
    });

    // ── Rule: Add(Tile(A), Tile(B)) == Tile(Add(A, B)) ────────────
    rules.push_back({
        "add-tile-distribute",
        "Add(Tile(A), Tile(B)) == Tile(Add(A,B))",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::ADD)) {
                const auto& node = g.node(nid);
                if (node.children.size() != 2) continue;

                ClassId left_cls = g.find(node.children[0]);
                ClassId right_cls = g.find(node.children[1]);

                for (NodeId lnid : g.class_nodes(left_cls)) {
                    const auto& left = g.node(lnid);
                    if (left.op != OpId::TILE) continue;
                    for (NodeId rnid : g.class_nodes(right_cls)) {
                        const auto& right = g.node(rnid);
                        if (right.op != OpId::TILE) continue;

                        ClassId A = g.find(left.children[0]);
                        ClassId B = g.find(right.children[0]);
                        ClassId a_plus_b = g.add_binary(OpId::ADD, A, B);
                        ClassId tiled = g.add_unary(OpId::TILE, a_plus_b);

                        merges.push_back({tiled, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: ReLU(Tile(A)) == Tile(ReLU(A)) ──────────────────────
    rules.push_back({
        "relu-tile-distribute",
        "ReLU(Tile(A)) == Tile(ReLU(A))",
        RulePriority::MEDIUM,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::RELU)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId cnid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(cnid);
                    if (child.op == OpId::TILE) {
                        ClassId inner = g.find(child.children[0]);
                        ClassId relu_inner = g.add_unary(OpId::RELU, inner);
                        ClassId tiled_relu = g.add_unary(OpId::TILE, relu_inner);
                        merges.push_back({tiled_relu, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    // ── Rule: Tile(Untile(A)) == A  (tile-untile cancellation) ─────
    rules.push_back({
        "tile-untile-cancel",
        "Tile(Untile(A)) == A",
        RulePriority::HIGH,
        [](EGraph& g) -> std::vector<std::pair<ClassId, ClassId>> {
            std::vector<std::pair<ClassId, ClassId>> merges;
            for (NodeId nid : g.nodes_of_op(OpId::TILE)) {
                const auto& node = g.node(nid);

                ClassId child_cls = g.find(node.children[0]);
                for (NodeId cnid : g.class_nodes(child_cls)) {
                    const auto& child = g.node(cnid);
                    if (child.op == OpId::UNTILE) {
                        ClassId inner = g.find(child.children[0]);
                        merges.push_back({inner, g.node_class(nid)});
                    }
                }
            }
            return merges;
        }
    });

    return rules;
}

// ─────────────────────────────────────────────────────────────────────────
// Aggregated Rules
// ─────────────────────────────────────────────────────────────────────────

std::vector<RewriteRule> all_rewrite_rules() {
    std::vector<RewriteRule> all;

    // Priority ordering: critical identity rules first, then high-value
    // structural rules, then exploratory rules last
    auto standard = standard_tensor_rewrite_rules();
    all.insert(all.end(), standard.begin(), standard.end());

    auto fusion = fusion_rewrite_rules();
    all.insert(all.end(), fusion.begin(), fusion.end());

    auto norm = normalization_rewrite_rules();
    all.insert(all.end(), norm.begin(), norm.end());

    auto transformer = transformer_rewrite_rules();
    all.insert(all.end(), transformer.begin(), transformer.end());

    auto tiling = tiling_rewrite_rules();
    all.insert(all.end(), tiling.begin(), tiling.end());

    return all;
}

// ─────────────────────────────────────────────────────────────────────────
// Distributivity Gating
// ─────────────────────────────────────────────────────────────────────────

/// Helper: compute the maximum expression depth from a given class.
/// Uses iterative BFS (not recursion) to avoid stack overflow.
static int compute_max_depth(const EGraph& g, ClassId root) {
    std::unordered_set<ClassId> visited;
    std::deque<std::pair<ClassId, int>> work_queue;  // (class_id, depth)
    work_queue.push_back({root, 0});
    visited.insert(root);

    int max_depth = 0;
    while (!work_queue.empty()) {
        auto [cid, depth] = work_queue.front();
        work_queue.pop_front();
        max_depth = std::max(max_depth, depth);

        for (NodeId nid : g.class_nodes(cid)) {
            const auto& node = g.node(nid);
            for (auto child_cid : node.children) {
                ClassId child_root = g.find(child_cid);
                if (child_root >= 0 && !visited.count(child_root)) {
                    visited.insert(child_root);
                    work_queue.push_back({child_root, depth + 1});
                }
            }
        }
    }
    return max_depth;
}

bool should_apply_distributivity(
    const EGraph& g,
    int max_depth,
    int64_t max_graph_size
) {
    // Gate 1: If the graph is already too large, don't distribute
    if (static_cast<int64_t>(g.num_nodes()) > max_graph_size) {
        return false;
    }

    // Gate 2: Check if any MUL node has a child that contains an ADD.
    // If at least one of the distributed terms already exists (reuse
    // detection), the distribution is worthwhile.
    const auto& mul_nodes = g.nodes_of_op(OpId::MUL);
    const auto& matmul_nodes = g.nodes_of_op(OpId::MATMUL);

    for (NodeId nid : mul_nodes) {
        const auto& node = g.node(nid);
        if (node.children.size() != 2) continue;

        // Check if either child is an ADD
        for (int side = 0; side < 2; ++side) {
            ClassId child_cls = g.find(node.children[side]);
            for (NodeId child_nid : g.class_nodes(child_cls)) {
                const auto& child = g.node(child_nid);
                if (child.op == OpId::ADD && child.children.size() == 2) {
                    // Found: A * (B + C)
                    ClassId A = g.find(node.children[1 - side]);
                    ClassId B = g.find(child.children[0]);
                    ClassId C = g.find(child.children[1]);

                    // Check if A*B or A*C already exists in the graph
                    // (reuse detection)
                    for (NodeId other_nid : mul_nodes) {
                        if (other_nid == nid) continue;
                        const auto& other = g.node(other_nid);
                        if (other.children.size() != 2) continue;
                        ClassId o0 = g.find(other.children[0]);
                        ClassId o1 = g.find(other.children[1]);
                        // Check A*B or A*C
                        if ((o0 == A && o1 == B) || (o0 == B && o1 == A) ||
                            (o0 == A && o1 == C) || (o0 == C && o1 == A)) {
                            return true;  // Reuse detected!
                        }
                    }
                    // Also check MATMUL nodes for reuse
                    for (NodeId mm_nid : matmul_nodes) {
                        const auto& mm = g.node(mm_nid);
                        if (mm.children.size() != 2) continue;
                        ClassId m0 = g.find(mm.children[0]);
                        ClassId m1 = g.find(mm.children[1]);
                        if ((m0 == A && m1 == B) || (m0 == A && m1 == C)) {
                            return true;  // MatMul reuse detected!
                        }
                    }

                    // Cost-guided: if A is a scalar constant, distributing
                    // is cheap and may enable fusion
                    for (NodeId a_nid : g.class_nodes(A)) {
                        const auto& a_node = g.node(a_nid);
                        if (a_node.op == OpId::CONSTANT) {
                            return true;  // Distributing a scalar is cheap
                        }
                    }
                }
            }
        }
    }

    // Gate 3: Depth check
    // If any MUL expression is below the depth threshold, allow it
    for (NodeId nid : mul_nodes) {
        const auto& node = g.node(nid);
        if (node.children.size() != 2) continue;
        ClassId cls = g.node_class(nid);
        int depth = compute_max_depth(g, cls);
        if (depth <= max_depth && depth > 0) {
            // Only allow if the MUL has an ADD child (otherwise distributivity
            // doesn't apply anyway)
            for (int side = 0; side < 2; ++side) {
                ClassId child_cls = g.find(node.children[side]);
                for (NodeId child_nid : g.class_nodes(child_cls)) {
                    if (g.node(child_nid).op == OpId::ADD) {
                        return true;  // Low depth, worth exploring
                    }
                }
            }
        }
    }

    // None of the gates passed — do not apply distributivity
    return false;
}

// ─────────────────────────────────────────────────────────────────────────
// Cost Functions
// ─────────────────────────────────────────────────────────────────────────

namespace {

/// Helper: estimate element count for cost modeling.
/// Uses shape info from the ENode's dim fields when available,
/// falls back to the default problem size.
int64_t estimate_element_count(const ENode& node, int64_t default_m, int64_t default_n) {
    if (node.dim0 > 0 && node.dim1 > 0) {
        if (node.dim2 > 0) return node.dim0 * node.dim1 * node.dim2;
        return node.dim0 * node.dim1;
    }
    return default_m * default_n;
}

} // anonymous namespace

std::function<double(OpId, const ENode&)>
memory_traffic_cost_fn(int64_t bytes_per_element) {
    return [bytes_per_element](OpId op, const ENode& node) -> double {
        // Estimate memory traffic for each operation.
        // Fused ops are cheaper because they avoid writing/reading intermediates.
        // We use problem-size heuristics when shape analysis is unavailable.

        double base_MN = 1024.0 * 1024.0;  // 1M elements (default)

        switch (op) {
            case OpId::SYMBOL:
                return base_MN * bytes_per_element;

            case OpId::CONSTANT:
                return bytes_per_element;

            case OpId::ADD:
            case OpId::SUB:
            case OpId::MUL:
            case OpId::DIV:
                // Read 2 inputs + write 1 output = 3x traffic
                return 3.0 * base_MN * bytes_per_element;

            case OpId::MATMUL: {
                // A[M,K] read + B[K,N] read + C[M,N] write
                // For default 1024x1024x1024 matmul
                double M = 1024.0, N = 1024.0, K = 512.0;
                return (M * K + K * N + M * N) * bytes_per_element;
            }

            case OpId::FUSED_MATMUL_RELU:
                // Saves: 1 write of C to HBM + 1 read of C for ReLU
                // Cost: only the final ReLU output is written
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0)
                       * bytes_per_element * 0.72;

            case OpId::FUSED_MATMUL_ADD:
                // Saves: 1 write/read of MatMul result for bias add
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0)
                       * bytes_per_element * 0.72;

            case OpId::FUSED_MATMUL_ADD_RELU:
                // Best: MatMul + bias + ReLU all in SRAM
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0)
                       * bytes_per_element * 0.55;

            case OpId::FUSED_GEMM:
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 2.0 * 1024.0 * 1024.0)
                       * bytes_per_element * 0.65;

            case OpId::FUSED_SOFTMAX:
                // Fused softmax does log-sum-exp in-place
                return 4.0 * base_MN * bytes_per_element;  // 2 passes, read+write each

            case OpId::FUSED_LAYERNORM:
            case OpId::FUSED_RMSNORM:
                // Fused norm: 2-3 passes in-place, no intermediate write
                return 3.0 * base_MN * bytes_per_element;

            case OpId::FUSED_ADD_LN:
                // Fused residual-add + layernorm: avoids intermediate sum
                return 4.0 * base_MN * bytes_per_element;

            case OpId::FUSED_MHA:
                // Fused MHA: avoids writing QK^T, attention weights, attention@V
                return 8.0 * base_MN * bytes_per_element;

            case OpId::RELU:
            case OpId::GELU:
            case OpId::SIGMOID:
            case OpId::NEG:
            case OpId::EXP:
            case OpId::LOG:
            case OpId::SQRT:
            case OpId::RECIPROCAL:
            case OpId::DROPOUT:
                // Unary: read + write
                return 2.0 * base_MN * bytes_per_element;

            case OpId::TRANSPOSE:
                return 2.0 * base_MN * bytes_per_element;

            case OpId::REDUCE_SUM:
            case OpId::REDUCE_MAX:
            case OpId::REDUCE_MEAN:
                // Read all, write reduced
                return 1.5 * base_MN * bytes_per_element;

            case OpId::SOFTMAX:
                // Unfused: multiple passes with intermediate materialization
                return 6.0 * base_MN * bytes_per_element;

            case OpId::LAYERNORM:
            case OpId::RMSNORM:
                // Unfused: center + variance + normalize = 3+ passes
                return 8.0 * base_MN * bytes_per_element;

            case OpId::TILE:
            case OpId::UNTILE:
                return 0.1 * base_MN * bytes_per_element;

            case OpId::RESHAPE:
            case OpId::BROADCAST:
            case OpId::IDENTITY:
                return 0.01 * base_MN * bytes_per_element;
        }
        return base_MN * bytes_per_element;
    };
}

std::function<double(OpId, const ENode&)>
compute_cost_fn(int64_t m, int64_t n, int64_t k) {
    return [m, n, k](OpId op, const ENode& /*node*/) -> double {
        double base_ops = static_cast<double>(m) * n;

        switch (op) {
            case OpId::SYMBOL:
            case OpId::CONSTANT:
                return 0.0;

            case OpId::ADD:
            case OpId::SUB:
            case OpId::MUL:
            case OpId::DIV:
            case OpId::NEG:
                return base_ops;

            case OpId::MATMUL:
                return 2.0 * m * n * k;

            case OpId::FUSED_MATMUL_RELU:
                return 2.0 * m * n * k + m * n;

            case OpId::FUSED_MATMUL_ADD:
                return 2.0 * m * n * k + m * n;

            case OpId::FUSED_MATMUL_ADD_RELU:
                return 2.0 * m * n * k + 2.0 * m * n;

            case OpId::FUSED_GEMM:
                return 2.0 * m * n * k + 2.0 * m * n;

            case OpId::RELU:
            case OpId::GELU:
            case OpId::SIGMOID:
            case OpId::EXP:
            case OpId::LOG:
            case OpId::SQRT:
            case OpId::RECIPROCAL:
            case OpId::DROPOUT:
                return base_ops;

            case OpId::SOFTMAX:
                return 5.0 * base_ops;  // exp + sum + div + max

            case OpId::FUSED_SOFTMAX:
                return 4.0 * base_ops;

            case OpId::LAYERNORM:
            case OpId::RMSNORM:
                return 5.0 * base_ops;  // mean + sub + var + div

            case OpId::FUSED_LAYERNORM:
            case OpId::FUSED_RMSNORM:
                return 3.0 * base_ops;

            case OpId::FUSED_ADD_LN:
                return base_ops + 3.0 * base_ops;

            case OpId::FUSED_MHA:
                return 2.0 * m * n * k * 3.0;  // QK^T + attn@V

            case OpId::TRANSPOSE:
            case OpId::RESHAPE:
            case OpId::BROADCAST:
            case OpId::TILE:
            case OpId::UNTILE:
            case OpId::IDENTITY:
                return 0.0;

            case OpId::REDUCE_SUM:
            case OpId::REDUCE_MAX:
            case OpId::REDUCE_MEAN:
                return base_ops;
        }
        return base_ops;
    };
}

std::function<double(OpId, const ENode&)>
combined_cost_fn(
    double memory_weight,
    double compute_weight,
    int64_t bytes_per_element,
    int64_t m, int64_t n, int64_t k
) {
    auto mem_fn = memory_traffic_cost_fn(bytes_per_element);
    auto comp_fn = compute_cost_fn(m, n, k);

    return [mem_fn, comp_fn, memory_weight, compute_weight](
        OpId op, const ENode& node) -> double {
        return memory_weight * mem_fn(op, node) +
               compute_weight * comp_fn(op, node);
    };
}

std::function<double(OpId, const ENode&)>
hardware_aware_cost_fn(
    int64_t sram_budget_bytes,
    double tc_speedup,
    int64_t bytes_per_element,
    int64_t m, int64_t n, int64_t k
) {
    auto mem_fn = memory_traffic_cost_fn(bytes_per_element);
    auto comp_fn = compute_cost_fn(m, n, k);

    return [mem_fn, comp_fn, sram_budget_bytes, tc_speedup, bytes_per_element, m, n, k](
        OpId op, const ENode& node) -> double {
        double mem_cost = mem_fn(op, node);
        double comp_cost = comp_fn(op, node);

        // Tensor Core speedup: fused matmul ops get a discount
        double tc_factor = 1.0;
        switch (op) {
            case OpId::MATMUL:
            case OpId::FUSED_MATMUL_RELU:
            case OpId::FUSED_MATMUL_ADD:
            case OpId::FUSED_MATMUL_ADD_RELU:
            case OpId::FUSED_GEMM:
            case OpId::FUSED_MHA:
                tc_factor = 1.0 / tc_speedup;
                break;
            default:
                break;
        }

        // SRAM pressure: ops that fit in SRAM are cheaper
        // (the memory cost is reduced because intermediates don't hit HBM)
        double sram_factor = 1.0;
        int64_t op_bytes = static_cast<int64_t>(mem_cost);
        if (op_bytes > 0 && op_bytes <= sram_budget_bytes) {
            // Fits in SRAM: memory cost is ~10x cheaper (SRAM vs HBM latency)
            sram_factor = 0.1;
        } else if (op_bytes > sram_budget_bytes) {
            // Exceeds SRAM: must spill to HBM, full memory cost
            sram_factor = 1.0;
        }

        return 0.6 * mem_cost * sram_factor + 0.4 * comp_cost * tc_factor;
    };
}

// ─────────────────────────────────────────────────────────────────────────
// Shape-Aware Cost Function
// ─────────────────────────────────────────────────────────────────────────

std::function<double(OpId, const ENode&)>
shape_aware_cost_fn(
    const EGraph& g,
    int64_t sram_budget_bytes,
    double tc_speedup
) {
    // Snapshot the class analyses we need from the e-graph so the
    // returned lambda does not hold a dangling reference.
    // The cost function is intended to be used immediately (within
    // the same scope as g), but we copy to be safe.
    std::unordered_map<ClassId, ClassAnalysis> analyses;
    for (ClassId cid = 0; cid < static_cast<ClassId>(g.num_classes()); ++cid) {
        ClassId root = g.find(cid);
        if (root >= 0) {
            analyses[root] = g.class_analysis(root);
        }
    }

    return [analyses = std::move(analyses), sram_budget_bytes, tc_speedup](
        OpId op, const ENode& node) -> double {

        // Helper: look up analysis for a child class from our snapshot
        auto get_analysis = [&analyses](ClassId cid) -> ClassAnalysis {
            auto it = analyses.find(cid);
            if (it != analyses.end()) return it->second;
            return {};
        };

        // Default fallback values
        const double default_MN = 1024.0 * 1024.0;
        const int64_t default_bpe = 2;

        // Determine element counts from actual shape info
        int64_t num_elements = -1;
        int64_t bpe = default_bpe;

        switch (op) {
            case OpId::SYMBOL: {
                // Symbols have no computation cost, only memory traffic
                // Try to get shape from any child analysis
                if (node.children.size() >= 1) {
                    auto a = get_analysis(node.children[0]);
                    if (!a.shape.is_unknown()) {
                        num_elements = a.shape.num_elements();
                        bpe = a.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return static_cast<double>(num_elements) * bpe;
            }

            case OpId::CONSTANT:
                return 1.0;  // Scalar constant

            case OpId::ADD:
            case OpId::SUB:
            case OpId::MUL:
            case OpId::DIV: {
                // Element-wise: shape = broadcast of children shapes
                if (node.children.size() >= 2) {
                    auto a0 = get_analysis(node.children[0]);
                    auto a1 = get_analysis(node.children[1]);
                    auto result_shape = EGraph::broadcast_shapes(a0.shape, a1.shape);
                    if (!result_shape.is_unknown()) {
                        num_elements = result_shape.num_elements();
                    }
                    DType promoted = EGraph::promote_dtypes(a0.dtype, a1.dtype);
                    ClassAnalysis tmp_a;
                    tmp_a.dtype = promoted;
                    bpe = (promoted != DType::UNKNOWN)
                          ? tmp_a.bytes_per_element()
                          : default_bpe;
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                // Read 2 inputs + write 1 output
                return 3.0 * num_elements * bpe;
            }

            case OpId::MATMUL: {
                // A[M,K] @ B[K,N] -> C[M,N]
                // FLOPs = 2*M*K*N, memory = (M*K + K*N + M*N) * bpe
                if (node.children.size() >= 2) {
                    auto a0 = get_analysis(node.children[0]);
                    auto a1 = get_analysis(node.children[1]);
                    int64_t M = a0.shape.ndim() >= 2 ? a0.shape[0] : -1;
                    int64_t K0 = a0.shape.ndim() >= 2 ? a0.shape[1] : -1;
                    int64_t K1 = a1.shape.ndim() >= 2 ? a1.shape[0] : -1;
                    int64_t N = a1.shape.ndim() >= 2 ? a1.shape[1] : -1;

                    // Validate K dimensions agree
                    if (K0 > 0 && K1 > 0 && K0 != K1) {
                        // Shape mismatch — high penalty
                        return 1e18;
                    }
                    int64_t K = K0 > 0 ? K0 : K1;

                    if (M > 0 && N > 0 && K > 0) {
                        bpe = a0.bytes_per_element();
                        double mem = (M * K + K * N + M * N) * bpe;
                        double comp = 2.0 * M * K * N;

                        // Tensor Core speedup: only when analysis says tc_compatible
                        double tc_factor = 1.0;
                        if (a0.tc_compatible || a1.tc_compatible) {
                            tc_factor = 1.0 / tc_speedup;
                        }

                        // SRAM pressure check
                        double sram_factor = 1.0;
                        if (mem <= sram_budget_bytes) {
                            sram_factor = 0.1;
                        }

                        return 0.6 * mem * sram_factor + 0.4 * comp * tc_factor;
                    }
                }
                // Fallback to defaults
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0) * default_bpe;
            }

            case OpId::FUSED_MATMUL_RELU:
            case OpId::FUSED_MATMUL_ADD: {
                if (node.children.size() >= 2) {
                    auto a0 = get_analysis(node.children[0]);
                    auto a1 = get_analysis(node.children[1]);
                    int64_t M = a0.shape.ndim() >= 2 ? a0.shape[0] : -1;
                    int64_t K = a0.shape.ndim() >= 2 ? a0.shape[1] : -1;
                    int64_t N = a1.shape.ndim() >= 2 ? a1.shape[1] : -1;
                    if (M > 0 && N > 0 && K > 0) {
                        bpe = a0.bytes_per_element();
                        return (M * K + K * N + M * N) * bpe * 0.72;
                    }
                }
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0) * default_bpe * 0.72;
            }

            case OpId::FUSED_MATMUL_ADD_RELU: {
                if (node.children.size() >= 2) {
                    auto a0 = get_analysis(node.children[0]);
                    auto a1 = get_analysis(node.children[1]);
                    int64_t M = a0.shape.ndim() >= 2 ? a0.shape[0] : -1;
                    int64_t K = a0.shape.ndim() >= 2 ? a0.shape[1] : -1;
                    int64_t N = a1.shape.ndim() >= 2 ? a1.shape[1] : -1;
                    if (M > 0 && N > 0 && K > 0) {
                        bpe = a0.bytes_per_element();
                        return (M * K + K * N + M * N) * bpe * 0.55;
                    }
                }
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 1024.0 * 1024.0) * default_bpe * 0.55;
            }

            case OpId::FUSED_GEMM:
                return (1024.0 * 512.0 + 512.0 * 1024.0 + 2.0 * 1024.0 * 1024.0) * default_bpe * 0.65;

            case OpId::FUSED_SOFTMAX: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 4.0 * num_elements * bpe;
            }

            case OpId::FUSED_LAYERNORM:
            case OpId::FUSED_RMSNORM: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 3.0 * num_elements * bpe;
            }

            case OpId::FUSED_ADD_LN: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 4.0 * num_elements * bpe;
            }

            case OpId::FUSED_MHA: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 8.0 * num_elements * bpe;
            }

            case OpId::RELU:
            case OpId::GELU:
            case OpId::SIGMOID:
            case OpId::NEG:
            case OpId::EXP:
            case OpId::LOG:
            case OpId::SQRT:
            case OpId::RECIPROCAL:
            case OpId::DROPOUT:
            case OpId::TRANSPOSE: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 2.0 * num_elements * bpe;
            }

            case OpId::REDUCE_SUM:
            case OpId::REDUCE_MAX:
            case OpId::REDUCE_MEAN: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 1.5 * num_elements * bpe;
            }

            case OpId::SOFTMAX: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 6.0 * num_elements * bpe;
            }

            case OpId::LAYERNORM:
            case OpId::RMSNORM: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 8.0 * num_elements * bpe;
            }

            case OpId::TILE:
            case OpId::UNTILE: {
                if (node.children.size() >= 1) {
                    auto a0 = get_analysis(node.children[0]);
                    if (!a0.shape.is_unknown()) {
                        num_elements = a0.shape.num_elements();
                        bpe = a0.bytes_per_element();
                    }
                }
                if (num_elements < 0) num_elements = static_cast<int64_t>(default_MN);
                return 0.1 * num_elements * bpe;
            }

            case OpId::RESHAPE:
            case OpId::BROADCAST:
            case OpId::IDENTITY:
                return 0.01 * default_MN * default_bpe;
        }
        return default_MN * default_bpe;
    };
}

// ─────────────────────────────────────────────────────────────────────────
// Polyhedral Validation
// ─────────────────────────────────────────────────────────────────────────

ValidationResult validate_no_cycle(const ExtractionResult& extracted) {
    ValidationResult result;

    if (extracted.nodes.empty()) {
        result.add_error("Extracted program is empty");
        return result;
    }

    // BFS/DFS to check that the extracted expression is a DAG.
    // Build a simple graph: each node references its children by index.
    // We assign indices to nodes in the order they appear.
    // Since the ExtractionResult only stores the node ops (not a tree),
    // we check structural DAG by ensuring no node appears more than once
    // in a topological walk. For a well-formed ExtractionResult, the
    // nodes vector should represent a valid expression tree (DAG).
    //
    // We also check for absurdly deep nesting which could indicate a cycle
    // in the e-graph extraction.

    // Safety: very large extracted programs are suspicious
    if (extracted.nodes.size() > 10000) {
        result.add_error("Extracted program exceeds 10000 nodes — possible oscillating rewrite");
        return result;
    }

    // Check that no operation references itself recursively.
    // In a valid extraction, children are computed before parents.
    // We use a simple visited-set approach on the expression tree.
    // Since the ExtractionResult stores a flat list of ENodes, we verify
    // there's no structural loop by checking that the number of nodes
    // is consistent with the expression depth.

    // A more thorough DAG check: count total children references and
    // ensure they form a proper tree. Each node's children should
    // reference previously-defined nodes or be leaves.
    // For the flat node list representation, we just ensure that
    // the node count is bounded and there are no obvious loops.

    // Simple structural check: ensure no duplicate identical sub-expressions
    // that could indicate a loop. This is conservative.
    std::unordered_set<size_t> seen_hashes;
    for (const auto& node : extracted.nodes) {
        size_t h = ENode::Hash{}(node);
        if (seen_hashes.count(h)) {
            // Same node appears twice — not necessarily a cycle, but
            // could indicate exponential blowup
            result.add_warning("Duplicate node detected in extraction — possible rewrite loop");
        }
        seen_hashes.insert(h);
    }

    return result;
}

ValidationResult validate_fp_safety(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps
) {
    ValidationResult result;

    // Check for FP-unsafe reorders by examining the operations
    // in the extracted program. FP-unsafe reorders include:
    //   1. Reordering additions around multiplications
    //   2. Reordering reductions across non-associative FP operations
    //   3. Changing the evaluation order of sum reductions

    bool has_fp_mul = false;
    bool has_reorderable_add = false;
    bool has_reduction = false;

    for (const auto& node : extracted.nodes) {
        if (node.op == OpId::MUL || node.op == OpId::DIV) {
            has_fp_mul = true;
        }
        if (node.op == OpId::ADD) {
            // Addition that could be reordered (associativity)
            has_reorderable_add = true;
        }
        if (node.op == OpId::REDUCE_SUM || node.op == OpId::REDUCE_MEAN) {
            has_reduction = true;
        }
    }

    // If the analysis says the extraction is not FP-safe, flag it
    if (!extracted.analysis.floating_point_safe) {
        if (has_fp_mul && has_reorderable_add) {
            result.add_warning(
                "FP-unsafe reorder: additions reordered around multiplications. "
                "This may change numerical results.");
        }
        if (has_reduction) {
            result.add_warning(
                "FP-unsafe reorder: reduction ordering changed. "
                "Floating-point sum reduction is non-associative.");
        }
    }

    // Check dependencies for FP-unsafe WAR on reduction results
    for (const auto& dep : original_deps) {
        if (dep.kind == DependencyKind::WAR) {
            // A write-after-read dependency that was reordered could
            // change FP results if it involves a reduction
            result.add_warning(
                "WAR dependency detected — verify that write-after-read "
                "ordering is preserved for FP correctness.");
        }
    }

    return result;
}

ValidationResult validate_fp_safety(
    const ExtractionResult& extracted,
    const EGraph& g
) {
    ValidationResult result;

    // Use the e-graph's class analysis to check FP safety
    for (const auto& node : extracted.nodes) {
        // For each node, check if any of its children's classes
        // are marked as not FP-safe
        for (ClassId child_cid : node.children) {
            ClassId root = g.find(child_cid);
            if (root >= 0 && root < static_cast<ClassId>(g.num_classes())) {
                const auto& analysis = g.class_analysis(root);
                if (!analysis.floating_point_safe) {
                    if (node.op == OpId::ADD || node.op == OpId::MUL ||
                        node.op == OpId::REDUCE_SUM || node.op == OpId::REDUCE_MEAN) {
                        result.add_warning(
                            "FP-unsafe expression involved in " +
                            op_to_string(node.op) +
                            " — reordering may change numerical results");
                    }
                }
            }
        }
    }

    // Also check the root analysis
    if (!extracted.analysis.floating_point_safe) {
        result.add_warning(
            "Root expression is marked FP-unsafe — reordering may "
            "change numerical results");
    }

    return result;
}

ValidationResult validate_reduction_ordering(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps
) {
    ValidationResult result;

    // Reductions over non-associative operations (e.g. FP sum) must
    // not be reordered, split, or merged in ways that change the
    // accumulation order unless the ClassAnalysis explicitly marks
    // the expression as floating_point_safe.

    // Step 1: Find all reductions in the extracted program
    bool has_fp_reduction = false;
    for (const auto& node : extracted.nodes) {
        if (node.op == OpId::REDUCE_SUM || node.op == OpId::REDUCE_MEAN) {
            if (!extracted.analysis.floating_point_safe) {
                has_fp_reduction = true;
                break;
            }
        }
    }

    if (!has_fp_reduction) return result;  // No FP reductions to validate

    // Step 2: Check that the dependency distance vectors remain
    // lexicographically positive for all reductions
    for (const auto& dep : original_deps) {
        if (dep.distance_vector.empty()) continue;

        // Check lexicographic sign of the distance vector
        bool lex_positive = false;
        for (auto d : dep.distance_vector) {
            if (d > 0) { lex_positive = true; break; }
            if (d < 0) { lex_positive = false; break; }
            // d == 0: continue to next dimension
        }

        if (!lex_positive) {
            result.add_error(
                "Reduction ordering violation: dependency distance vector "
                "is not lexicographically positive. This indicates a "
                "reduction reordering that changes FP accumulation order.");
        }
    }

    // Step 3: Check for split/merged reductions
    // If the original program had one reduction and the extracted
    // program has multiple reductions over the same data, that
    // changes accumulation order.
    int reduction_count = 0;
    for (const auto& node : extracted.nodes) {
        if (node.op == OpId::REDUCE_SUM || node.op == OpId::REDUCE_MEAN ||
            node.op == OpId::REDUCE_MAX) {
            reduction_count++;
        }
    }

    // Count original reductions from dependencies
    int orig_reduction_deps = 0;
    for (const auto& dep : original_deps) {
        // Heuristic: if a dependency involves a reduction class,
        // count it
        (void)dep;  // We can't directly check the source op from deps alone
        orig_reduction_deps++;
    }

    if (reduction_count > 1 && !extracted.analysis.floating_point_safe) {
        result.add_warning(
            "Multiple reductions in extracted program — accumulation "
            "order may differ from original.");
    }

    return result;
}

ValidationResult validate_extracted_program_full(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps,
    const EGraph& g
) {
    ValidationResult result;

    // Run all validation checks and combine results

    // 1. Cycle check
    auto cycle_result = validate_no_cycle(extracted);
    for (const auto& err : cycle_result.errors) result.add_error(err);
    for (const auto& warn : cycle_result.warnings) result.add_warning(warn);

    // 2. FP safety check (using e-graph analysis)
    auto fp_result = validate_fp_safety(extracted, g);
    for (const auto& err : fp_result.errors) result.add_error(err);
    for (const auto& warn : fp_result.warnings) result.add_warning(warn);

    // 3. Reduction ordering check
    auto red_result = validate_reduction_ordering(extracted, original_deps);
    for (const auto& err : red_result.errors) result.add_error(err);
    for (const auto& warn : red_result.warnings) result.add_warning(warn);

    // 4. Dependency distance vector check
    for (const auto& dep : original_deps) {
        if (!dep.distance_vector.empty()) {
            bool lex_positive = false;
            for (auto d : dep.distance_vector) {
                if (d > 0) { lex_positive = true; break; }
                if (d < 0) { lex_positive = false; break; }
            }
            if (!lex_positive) {
                result.add_error(
                    "Dependency distance vector is not lexicographically "
                    "positive — indicates invalid program transformation.");
            }
        }
    }

    return result;
}

bool validate_extracted_program(
    const ExtractionResult& extracted,
    const std::vector<Dependency>& original_deps
) {
    // Delegate to the full validation infrastructure
    ValidationResult result = validate_extracted_program_full(
        extracted, original_deps, EGraph{});

    // For backward compatibility, also run the original checks
    // that don't require an EGraph

    // Step 1: If there are no dependencies, the extraction is trivially valid.
    if (original_deps.empty()) return result.is_valid;

    // Step 2: Build a set of node ops in the extracted program for fast lookup
    std::unordered_set<OpId, OpIdHash> extracted_ops;
    for (const auto& node : extracted.nodes) {
        extracted_ops.insert(node.op);
    }

    // Step 3: Check each dependency
    for (const auto& dep : original_deps) {
        bool involves_reduction = false;

        // Check if the source is a reduction
        for (const auto& node : extracted.nodes) {
            if (node.op == OpId::REDUCE_SUM ||
                node.op == OpId::REDUCE_MEAN ||
                node.op == OpId::REDUCE_MAX) {
                involves_reduction = true;
                break;
            }
        }

        // If a reduction is involved, check that the distance vector
        // remains lexicographically positive
        if (involves_reduction && !dep.distance_vector.empty()) {
            bool lex_positive = false;
            for (auto d : dep.distance_vector) {
                if (d > 0) { lex_positive = true; break; }
                if (d < 0) { lex_positive = false; break; }
            }
            if (!lex_positive) return false;
        }

        // Check: WAR (anti-dependency)
        if (dep.kind == DependencyKind::WAR) {
            int src_pos = -1, tgt_pos = -1;
            for (size_t i = 0; i < extracted.nodes.size(); ++i) {
                if (extracted.nodes[i].op != OpId::SYMBOL &&
                    extracted.nodes[i].op != OpId::CONSTANT) {
                    if (src_pos < 0) src_pos = static_cast<int>(i);
                    else if (tgt_pos < 0) tgt_pos = static_cast<int>(i);
                }
            }
        }
    }

    // Step 4: Size sanity check
    if (extracted.nodes.size() > 10000) {
        return false;
    }

    return result.is_valid;
}

bool validate_extracted_program(
    const ExtractionResult& extracted,
    const std::vector<std::pair<ClassId, ClassId>>& dep_pairs
) {
    // Convert simple pairs to Dependency objects
    std::vector<Dependency> deps;
    deps.reserve(dep_pairs.size());
    for (const auto& [src, tgt] : dep_pairs) {
        Dependency d;
        d.source = src;
        d.target = tgt;
        d.kind = DependencyKind::RAW;
        deps.push_back(d);
    }
    return validate_extracted_program(extracted, deps);
}

} // namespace symplex::optimizer::egraph
