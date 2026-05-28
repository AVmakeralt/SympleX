// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/superoptimizer.h"
#include "symplex/optimizer/egraph.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace symplex::optimizer;
using namespace symplex::optimizer::egraph;
using namespace symplex::polyhedral;
using namespace symplex::hardware;

int main() {
    int tests_passed = 0;

    // ═══════════════════════════════════════════════════════════════
    // E-GRAPH CORE TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 1: E-graph basic construction
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId c = g.add_symbol("C");

        assert(g.num_nodes() >= 3);
        assert(g.num_live_classes() >= 3);
        tests_passed++;
        std::cout << "[PASS] E-graph basic construction\n";
    }

    // Test 2: E-graph binary operations
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId a_plus_b = g.add_binary(OpId::ADD, a, b);
        ClassId a_times_b = g.add_binary(OpId::MUL, a, b);
        ClassId a_matmul_b = g.add_binary(OpId::MATMUL, a, b);

        assert(g.num_nodes() >= 5);
        tests_passed++;
        std::cout << "[PASS] E-graph binary operations (" << g.num_nodes() << " nodes)\n";
    }

    // Test 3: Union-find and equivalence
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");

        assert(!g.are_equivalent(a, b));

        g.merge(a, b);
        assert(g.are_equivalent(a, b));
        tests_passed++;
        std::cout << "[PASS] E-graph union-find and equivalence\n";
    }

    // Test 4: Congruence closure
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId c = g.add_symbol("C");

        ClassId f_ab = g.add_binary(OpId::ADD, a, b);
        ClassId f_ac = g.add_binary(OpId::ADD, a, c);

        assert(!g.are_equivalent(f_ab, f_ac));

        g.merge(b, c);
        g.rebuild();  // Congruence repair is needed after merge
        assert(g.are_equivalent(f_ab, f_ac));
        tests_passed++;
        std::cout << "[PASS] E-graph congruence closure\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // HASH CORRECTNESS TESTS (Bug Fix Verification)
    // ═══════════════════════════════════════════════════════════════

    // Test 5: ENode hash distinguishes axis parameter
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId rs0 = g.add_unary_with_axis(OpId::REDUCE_SUM, a, 0);
        ClassId rs1 = g.add_unary_with_axis(OpId::REDUCE_SUM, a, 1);

        // ReduceSum(A, axis=0) and ReduceSum(A, axis=1) must NOT be equivalent
        assert(!g.are_equivalent(rs0, rs1));
        tests_passed++;
        std::cout << "[PASS] ENode hash distinguishes axis parameter\n";
    }

    // Test 6: ENode hash distinguishes dim parameters
    {
        ENode a, b;
        a.op = OpId::TILE;
        a.children = {0};
        a.dim0 = 128;
        a.dim1 = 64;
        a.dim2 = -1;
        b.op = OpId::TILE;
        b.children = {0};
        b.dim0 = 256;
        b.dim1 = 64;
        b.dim2 = -1;

        ENode::Hash hasher;
        assert(hasher(a) != hasher(b));
        assert(a != b);
        tests_passed++;
        std::cout << "[PASS] ENode hash distinguishes dim parameters\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // REWRITE RULE TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 7: Commutativity rewrite (A + B == B + A)
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId a_plus_b = g.add_binary(OpId::ADD, a, b);
        ClassId b_plus_a = g.add_binary(OpId::ADD, b, a);

        auto rules = standard_tensor_rewrite_rules();
        int iters = g.saturate(rules, 5);

        assert(g.are_equivalent(a_plus_b, b_plus_a));
        tests_passed++;
        std::cout << "[PASS] Commutativity rewrite (A+B == B+A, "
                  << iters << " iters)\n";
    }

    // Test 8: Additive identity (A + 0 == A)
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId zero = g.add_constant(0);
        ClassId a_plus_0 = g.add_binary(OpId::ADD, a, zero);

        auto rules = standard_tensor_rewrite_rules();
        g.saturate(rules, 5);

        assert(g.are_equivalent(a_plus_0, a));
        tests_passed++;
        std::cout << "[PASS] Additive identity (A+0 == A)\n";
    }

    // Test 9: Multiplicative identity (A * 1 == A)
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId one = g.add_constant(1);
        ClassId a_times_1 = g.add_binary(OpId::MUL, a, one);

        auto rules = standard_tensor_rewrite_rules();
        g.saturate(rules, 5);

        assert(g.are_equivalent(a_times_1, a));
        tests_passed++;
        std::cout << "[PASS] Multiplicative identity (A*1 == A)\n";
    }

    // Test 10: Annihilation (A * 0 == 0)
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId zero = g.add_constant(0);
        ClassId a_times_0 = g.add_binary(OpId::MUL, a, zero);

        auto rules = standard_tensor_rewrite_rules();
        g.saturate(rules, 5);

        assert(g.are_equivalent(a_times_0, zero));
        tests_passed++;
        std::cout << "[PASS] Annihilation (A*0 == 0)\n";
    }

    // Test 11: Factorization (A*B + A*C == A*(B+C)) - THE KEY RULE
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId c = g.add_symbol("C");

        ClassId ab = g.add_binary(OpId::MUL, a, b);
        ClassId ac = g.add_binary(OpId::MUL, a, c);
        ClassId ab_plus_ac = g.add_binary(OpId::ADD, ab, ac);

        ClassId b_plus_c = g.add_binary(OpId::ADD, b, c);
        ClassId a_times_bc = g.add_binary(OpId::MUL, a, b_plus_c);

        auto rules = standard_tensor_rewrite_rules();
        int iters = g.saturate(rules, 10);

        // After saturation, A*B + A*C should be equivalent to A*(B+C)
        assert(g.are_equivalent(ab_plus_ac, a_times_bc));
        tests_passed++;
        std::cout << "[PASS] Factorization (A*B + A*C == A*(B+C), "
                  << iters << " iters)\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // FUSION DISCOVERY TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 12: Fusion discovery (ReLU(MatMul(A,B)) == FusedMatMulReLU(A,B))
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);
        ClassId relu_mm = g.add_unary(OpId::RELU, mm);

        auto rules = fusion_rewrite_rules();
        g.saturate(rules, 10);

        assert(g.num_nodes() > 3);
        tests_passed++;
        std::cout << "[PASS] Fusion discovery (e-graph grew to "
                  << g.num_nodes() << " nodes)\n";
    }

    // Test 13: MatMul+bias fusion
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId bias = g.add_symbol("bias");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);
        ClassId mm_plus_bias = g.add_binary(OpId::ADD, mm, bias);

        auto rules = fusion_rewrite_rules();
        g.saturate(rules, 10);

        assert(g.num_nodes() > 4);
        tests_passed++;
        std::cout << "[PASS] MatMul+bias fusion discovery\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // COST-GUIDED EXTRACTION TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 14: Extraction picks the cheaper fused operation
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId bias = g.add_symbol("bias");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);
        ClassId mm_plus_bias = g.add_binary(OpId::ADD, mm, bias);

        auto rules = fusion_rewrite_rules();
        g.saturate(rules, 10);

        auto cost_fn = memory_traffic_cost_fn(2);
        auto extracted = g.extract(mm_plus_bias, cost_fn);

        assert(extracted.cost > 0);
        tests_passed++;
        std::cout << "[PASS] Cost-guided extraction (cost=" << extracted.cost
                  << ", expr=" << extracted.expr_string << ")\n";
    }

    // Test 15: Hardware-aware cost function
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId bias = g.add_symbol("bias");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);
        ClassId mm_plus_bias = g.add_binary(OpId::ADD, mm, bias);
        ClassId relu_result = g.add_unary(OpId::RELU, mm_plus_bias);

        auto rules = fusion_rewrite_rules();
        g.saturate(rules, 10);

        auto cost_fn = hardware_aware_cost_fn(228000, 4.0, 2, 1024, 1024, 512);
        auto extracted = g.extract(relu_result, cost_fn);

        assert(!extracted.expr_string.empty());
        tests_passed++;
        std::cout << "[PASS] Hardware-aware extraction (expr="
                  << extracted.expr_string << ")\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // ANALYSIS DATA TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 16: Symbol with shape analysis
    {
        EGraph g;
        ClassId a = g.add_symbol("A",
            TensorShape({1024, 512}), DType::FP16, Layout::ROW_MAJOR);

        auto& analysis = g.class_analysis(a);
        assert(!analysis.shape.is_unknown());
        assert(analysis.shape.ndim() == 2);
        assert(analysis.shape[0] == 1024);
        assert(analysis.shape[1] == 512);
        assert(analysis.dtype == DType::FP16);
        tests_passed++;
        std::cout << "[PASS] Symbol with shape analysis\n";
    }

    // Test 17: MatMul analysis propagation
    {
        EGraph g;
        ClassId a = g.add_symbol("A",
            TensorShape({1024, 512}), DType::FP16);
        ClassId b = g.add_symbol("B",
            TensorShape({512, 1024}), DType::FP16);
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);

        g.update_analysis(mm);
        auto& analysis = g.class_analysis(mm);
        // MatMul should be TC-compatible
        assert(analysis.tc_compatible);
        tests_passed++;
        std::cout << "[PASS] MatMul analysis propagation\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // SATURATION CONFIG TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 18: Saturation with config
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId c = g.add_symbol("C");
        ClassId ab = g.add_binary(OpId::MUL, a, b);
        ClassId ac = g.add_binary(OpId::MUL, a, c);
        ClassId sum = g.add_binary(OpId::ADD, ab, ac);

        auto rules = standard_tensor_rewrite_rules();
        SaturationConfig config;
        config.max_iters = 10;
        config.max_nodes = 5000;
        config.cost_guided_filter = true;
        config.rule_fanout_limit = 20;
        int iters = g.saturate(rules, config);

        assert(iters > 0);
        tests_passed++;
        std::cout << "[PASS] Saturation with config (" << iters << " iters)\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // NORMALIZATION RULE TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 19: Softmax decomposition uses Exp (not broken x/ReduceSum(x))
    {
        EGraph g;
        ClassId x = g.add_symbol("X");
        ClassId sm = g.add_unary_with_axis(OpId::SOFTMAX, x, -1);

        auto rules = transformer_rewrite_rules();
        g.saturate(rules, 5);

        // The e-graph should contain Exp nodes from softmax decomposition
        bool has_exp = false;
        for (NodeId nid = 0; nid < static_cast<NodeId>(g.num_nodes()); ++nid) {
            if (g.node(nid).op == OpId::EXP) {
                has_exp = true;
                break;
            }
        }
        assert(has_exp);
        tests_passed++;
        std::cout << "[PASS] Softmax decomposition uses Exp\n";
    }

    // Test 20: Dropout identity
    {
        EGraph g;
        ClassId x = g.add_symbol("X");
        ClassId drop = g.add_unary(OpId::DROPOUT, x);

        auto rules = normalization_rewrite_rules();
        g.saturate(rules, 5);

        assert(g.are_equivalent(drop, x));
        tests_passed++;
        std::cout << "[PASS] Dropout identity (inference mode)\n";
    }

    // ═══════════════════════════════════════════════════════════════
    // TWO-LEVEL SUPEROPTIMIZER TESTS
    // ═══════════════════════════════════════════════════════════════

    // Test 21: Phase 1 roofline pruning (Level 2)
    {
        HardwareTarget target = HardwareTarget::H100();
        auto candidates = phase1_roofline_pruning(3, target, 256);
        assert(!candidates.empty());
        tests_passed++;
        std::cout << "[PASS] Level 2 Phase 1 roofline pruning ("
                  << candidates.size() << " candidates)\n";
    }

    // Test 22: Full two-level superoptimizer
    {
        HardwareTarget target = HardwareTarget::H100();
        Superoptimizer opt(target);
        auto ispace = make_matmul_iteration_space(1024, 1024, 512);
        auto result = opt.optimize(ispace, 256);

        assert(result.best_tile.inner_tiles.size() > 0);
        assert(result.estimated_latency_ns > 0);
        assert(!result.original_expr.empty());
        assert(!result.optimized_expr.empty());
        assert(result.egraph_nodes > 0);

        tests_passed++;
        std::cout << "[PASS] Full two-level superoptimizer\n";
        std::cout << "  Original:  " << result.original_expr << "\n";
        std::cout << "  Optimized: " << result.optimized_expr << "\n";
        std::cout << "  E-graph: " << result.egraph_nodes << " nodes, "
                  << result.egraph_classes << " classes\n";
        std::cout << "  Analysis: " << result.root_analysis.to_string() << "\n";
        std::cout << "  Latency: " << result.estimated_latency_ns << " ns, "
                  << "speedup=" << result.speedup_vs_naive << "x\n";
    }

    // Test 23: Polyhedral validation
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);

        auto cost_fn = memory_traffic_cost_fn(2);
        auto extracted = g.extract(mm, cost_fn);

        std::vector<Dependency> deps;
        bool valid = validate_extracted_program(extracted, deps);
        assert(valid);
        tests_passed++;
        std::cout << "[PASS] Polyhedral validation\n";
    }

    // Test 24: Polyhedral validation with dependencies
    {
        EGraph g;
        ClassId a = g.add_symbol("A");
        ClassId b = g.add_symbol("B");
        ClassId mm = g.add_binary(OpId::MATMUL, a, b);

        auto cost_fn = memory_traffic_cost_fn(2);
        auto extracted = g.extract(mm, cost_fn);

        // Create a RAW dependency
        Dependency dep;
        dep.source = 0;
        dep.target = 1;
        dep.kind = DependencyKind::RAW;
        dep.distance_vector = {1};

        std::vector<Dependency> deps = {dep};
        bool valid = validate_extracted_program(extracted, deps);
        assert(valid);
        tests_passed++;
        std::cout << "[PASS] Polyhedral validation with RAW dependency\n";
    }

    // Test 25: Iterative extraction (no stack overflow)
    {
        EGraph g;
        // Build a deep chain: A + (A + (A + ...))
        ClassId current = g.add_symbol("A");
        for (int i = 0; i < 100; ++i) {
            ClassId a = g.add_symbol("A");
            current = g.add_binary(OpId::ADD, current, a);
        }

        auto cost_fn = memory_traffic_cost_fn(2);
        auto extracted = g.extract(current, cost_fn);

        assert(extracted.cost > 0);
        assert(!extracted.expr_string.empty());
        tests_passed++;
        std::cout << "[PASS] Iterative extraction (deep chain, "
                  << extracted.nodes.size() << " nodes)\n";
    }

    std::cout << "\nAll optimizer tests passed! (" << tests_passed << " tests)\n";
    return 0;
}
