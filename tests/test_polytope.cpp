// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/integer_polytope.h"
#include "symplex/polyhedral/affine_map.h"
#include "symplex/polyhedral/dependency.h"
#include "symplex/polyhedral/iteration_space.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace symplex::polyhedral;

int main() {
    // Test 1: IntegerPolytope basic operations
    {
        IntegerPolytope poly(2);
        poly.add_range_bound(0, 0, 3);
        poly.add_range_bound(1, 0, 4);

        assert(poly.contains({0, 0}));
        assert(poly.contains({3, 4}));
        assert(!poly.contains({4, 0}));
        assert(!poly.contains({0, 5}));

        auto count = poly.count_points();
        assert(count == 20);  // 4 * 5

        std::cout << "[PASS] IntegerPolytope basic operations\n";
    }

    // Test 2: make_rectangular_polytope
    {
        auto poly = make_rectangular_polytope({{0, 4}, {0, 5}});
        assert(poly.ndim() == 2);
        assert(poly.count_points() == 20);
        std::cout << "[PASS] make_rectangular_polytope\n";
    }

    // Test 3: AffineMap identity
    {
        auto id = AffineMap::identity(3);
        auto result = id.apply({1, 2, 3});
        assert(result == std::vector<int64_t>({1, 2, 3}));
        std::cout << "[PASS] AffineMap identity\n";
    }

    // Test 4: AffineMap composition
    {
        AffineMap f(2, 2);
        f.matrix_at(0, 0) = 2; f.matrix_at(1, 1) = 3;
        auto g = AffineMap::identity(2);
        auto fg = f.compose(g);
        auto result = fg.apply({1, 1});
        assert(result[0] == 2);
        assert(result[1] == 3);
        std::cout << "[PASS] AffineMap composition\n";
    }

    // Test 5: Projection (Fourier-Motzkin)
    {
        auto poly = make_rectangular_polytope({{0, 5}, {0, 3}});
        auto projected = poly.project_out(1);
        assert(projected.ndim() == 1);
        assert(projected.contains({0}));
        assert(projected.contains({4}));
        std::cout << "[PASS] Fourier-Motzkin projection\n";
    }

    // Test 6: Matmul iteration space
    {
        auto ispace = make_matmul_iteration_space(128, 128, 64);
        assert(ispace.num_statements() == 1);
        assert(ispace.statements()[0].domain.ndim() == 3);
        std::cout << "[PASS] Matmul iteration space\n";
    }

    std::cout << "All polytope tests passed!\n";
    return 0;
}
