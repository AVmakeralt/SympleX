// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/integer_polytope.h"
#include "symplex/polyhedral/affine_map.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <limits>

namespace symplex::polyhedral {

// ── Compile-time verification ────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<Inequality>);
static_assert(std::is_default_constructible_v<Equality>);
// IntegerPolytope is not default-constructible (requires ndim).
static_assert(std::is_constructible_v<IntegerPolytope, size_t>);

// ── Non-inline helper functions for IntegerPolytope ──────────────────────────

/// Remove redundant constraints from a polytope.
/// A constraint is redundant if it is implied by the other constraints.
/// This uses a simple sampling-based approach: for each constraint,
/// check if removing it would allow any previously-excluded point.
IntegerPolytope remove_redundant_constraints(const IntegerPolytope& poly) {
    const auto& ineqs = poly.inequalities();
    const auto& eqs = poly.equalities();
    size_t ndim = poly.ndim();

    if (ineqs.size() <= 1) return poly;

    std::vector<Inequality> essential_ineqs;

    for (size_t i = 0; i < ineqs.size(); ++i) {
        // Check if ineqs[i] is implied by the other constraints.
        // Sample points from the other constraints and check if they
        // all satisfy ineqs[i].
        bool redundant = true;

        // Build a temporary polytope without constraint i.
        IntegerPolytope reduced(ndim);
        for (size_t j = 0; j < ineqs.size(); ++j) {
            if (j != i) reduced.add_inequality(ineqs[j]);
        }
        for (const auto& eq : eqs) {
            reduced.add_equality(eq);
        }

        // Try to find a point in the reduced polytope that violates
        // constraint i. If we can, constraint i is not redundant.
        auto points = reduced.enumerate_points();
        for (const auto& pt : points) {
            if (!ineqs[i].satisfied(pt)) {
                redundant = false;
                break;
            }
        }

        if (!redundant) {
            essential_ineqs.push_back(ineqs[i]);
        }
    }

    IntegerPolytope result(ndim);
    for (auto& ineq : essential_ineqs) result.add_inequality(std::move(ineq));
    for (const auto& eq : eqs) result.add_equality(eq);

    return result;
}

/// Compute the volume of the polytope (number of integer points).
/// For rectangular polytopes, this is the product of range widths.
/// For general polytopes, this uses point enumeration (only for small polytopes).
int64_t compute_volume(const IntegerPolytope& poly) {
    return poly.count_points();
}

/// Compute the convex hull of two polytopes.
/// The convex hull is the smallest polytope containing both inputs.
IntegerPolytope convex_hull(
    const IntegerPolytope& a,
    const IntegerPolytope& b
) {
    assert(a.ndim() == b.ndim());
    size_t ndim = a.ndim();

    // For simplicity, compute the bounding box (over-approximation).
    auto bounds_a = a.bounds();
    auto bounds_b = b.bounds();

    IntegerPolytope result(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        int64_t lo = std::min(bounds_a[d].first, bounds_b[d].first);
        int64_t hi = std::max(bounds_a[d].second, bounds_b[d].second);
        result.add_range_bound(d, lo, hi);
    }

    return result;
}

/// Check if two polytopes are equal (contain the same set of points).
/// Only reliable for small, bounded polytopes.
bool polytopes_equal(
    const IntegerPolytope& a,
    const IntegerPolytope& b
) {
    if (a.ndim() != b.ndim()) return false;

    // Both must be empty, or contain the same points.
    auto pts_a = a.enumerate_points();
    auto pts_b = b.enumerate_points();

    if (pts_a.size() != pts_b.size()) return false;

    std::sort(pts_a.begin(), pts_a.end());
    std::sort(pts_b.begin(), pts_b.end());

    return pts_a == pts_b;
}

/// Apply Fourier-Motzkin elimination to project out multiple dimensions.
/// Eliminates dimensions in the order given.
IntegerPolytope project_out_multi(
    const IntegerPolytope& poly,
    const std::vector<size_t>& dims
) {
    // Sort dimensions in descending order to avoid index shifting.
    auto sorted_dims = dims;
    std::sort(sorted_dims.rbegin(), sorted_dims.rend());

    IntegerPolytope result = poly;
    for (size_t d : sorted_dims) {
        if (d < result.ndim()) {
            result = result.project_out(d);
        }
    }
    return result;
}

/// Compute the image of a polytope under an AffineMap.
IntegerPolytope apply_affine_map(
    const IntegerPolytope& poly,
    const AffineMap& map
) {
    return poly.image(map.matrix(), map.offset());
}

/// Simplify a polytope by removing obviously redundant constraints
/// (e.g., 0 >= 0 which is always true, or constraints with all-zero
/// coefficients that are trivially satisfied or unsatisfied).
IntegerPolytope simplify_polytope(const IntegerPolytope& poly) {
    size_t ndim = poly.ndim();
    IntegerPolytope result(ndim);

    for (const auto& ineq : poly.inequalities()) {
        // Check if all coefficients are zero.
        bool all_zero = std::all_of(
            ineq.coefficients.begin(), ineq.coefficients.end(),
            [](int64_t c) { return c == 0; }
        );

        if (all_zero) {
            // 0*i + b >= 0  →  b >= 0
            if (ineq.constant >= 0) {
                // Trivially satisfied: skip.
                continue;
            } else {
                // Trivially unsatisfied: the polytope is empty.
                // Return an empty polytope with a contradictory constraint.
                result.add_inequality(ineq);
                return result;
            }
        }

        result.add_inequality(ineq);
    }

    for (const auto& eq : poly.equalities()) {
        bool all_zero = std::all_of(
            eq.coefficients.begin(), eq.coefficients.end(),
            [](int64_t c) { return c == 0; }
        );

        if (all_zero) {
            if (eq.constant == 0) {
                continue;  // 0 == 0, trivially satisfied.
            } else {
                // Contradiction: polytope is empty.
                // Convert to an impossible inequality.
                Inequality impossible(ndim, -1);
                result.add_inequality(std::move(impossible));
                return result;
            }
        }

        result.add_equality(eq);
    }

    return result;
}

} // namespace symplex::polyhedral
