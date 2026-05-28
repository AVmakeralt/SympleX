// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/iteration_space.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <limits>

namespace symplex::polyhedral {

// ── Compile-time verification ────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<IterationSpace>);
// Statement is not default-constructible (requires name + domain).
static_assert(std::is_constructible_v<Statement, std::string, IntegerPolytope>);

// ── Non-inline helper functions for IterationSpace ───────────────────────────

/// Compute the total number of iterations in the iteration space.
/// This is the product of the dimension ranges for each statement's domain.
int64_t total_iteration_count(const IterationSpace& ispace) {
    int64_t total = 0;
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        auto count = ispace.statement(s).domain.count_points();
        if (count < 0) {
            // Cannot determine exact count; estimate from bounds.
            auto bounds = ispace.statement(s).domain.bounds();
            int64_t stmt_total = 1;
            for (const auto& [lo, hi] : bounds) {
                stmt_total *= (hi - lo + 1);
            }
            total += stmt_total;
        } else {
            total += count;
        }
    }
    return total;
}

/// Compute the arithmetic intensity of the iteration space
/// (FLOPS per byte of data moved).
double arithmetic_intensity(
    const IterationSpace& ispace,
    size_t bytes_per_element
) {
    if (ispace.num_statements() == 0) return 0.0;

    // Estimate FLOPS: 2 per iteration (multiply-add).
    int64_t total_iters = total_iteration_count(ispace);
    double flops = static_cast<double>(total_iters) * 2.0;

    // Estimate bytes moved: count unique tensor elements accessed.
    int64_t total_elements = 0;
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        for (size_t ai = 0; ai < ispace.statement(s).accesses.size(); ++ai) {
            const auto& access = ispace.statement(s).accesses[ai];
            auto bounds = ispace.statement(s).domain.bounds();
            int64_t elements = 1;
            for (const auto& [lo, hi] : bounds) {
                elements *= (hi - lo + 1);
            }
            // Scale by access map output dimensions to reflect actual data touched.
            elements *= static_cast<int64_t>(std::max<size_t>(1, access.access_map.n_out()));
            total_elements += elements;
        }
    }

    double bytes = static_cast<double>(total_elements) * bytes_per_element;
    if (bytes <= 0.0) return 0.0;

    return flops / bytes;
}

/// Check if the iteration space is a simple rectangular loop nest
/// (all statements have rectangular domains with no cross-dimensional
/// constraints).
bool is_rectangular(const IterationSpace& ispace) {
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        const auto& ineqs = ispace.statement(s).domain.inequalities();
        for (const auto& ineq : ineqs) {
            // A rectangular constraint has at most one non-zero coefficient.
            size_t nnz = 0;
            for (auto c : ineq.coefficients) {
                if (c != 0) ++nnz;
            }
            if (nnz > 1) return false;
        }
        // Equalities would make it non-rectangular (unless trivial).
        if (!ispace.statement(s).domain.equalities().empty()) return false;
    }
    return true;
}

/// Compute the number of data reuse opportunities in the iteration space.
/// This counts the number of times a tensor element is accessed by
/// multiple iterations.
int64_t compute_reuse_factor(const IterationSpace& ispace) {
    if (ispace.num_statements() == 0) return 0;

    int64_t reuse = 0;
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        const auto& stmt = ispace.statement(s);
        for (const auto& acc : stmt.accesses) {
            // Estimate: total iterations / unique elements accessed.
            auto bounds = stmt.domain.bounds();
            int64_t total_iters = 1;
            for (const auto& [lo, hi] : bounds) {
                total_iters *= (hi - lo + 1);
            }

            // Unique elements: determined by the access map's output dimensionality.
            int64_t unique_elements = 1;
            for (size_t d = 0; d < acc.access_map.n_out(); ++d) {
                // Estimate based on the range of each output dimension.
                int64_t range = 0;
                for (size_t j = 0; j < acc.access_map.n_in(); ++j) {
                    range += std::abs(acc.access_map.matrix()[d][j]) *
                             (bounds[j].second - bounds[j].first + 1);
                }
                unique_elements *= std::max(int64_t(1), range);
            }

            if (unique_elements > 0) {
                reuse += total_iters / unique_elements;
            }
        }
    }

    return reuse;
}

/// Merge two iteration spaces (union of their statements).
IterationSpace merge_iteration_spaces(
    const IterationSpace& a,
    const IterationSpace& b
) {
    IterationSpace result(a.name() + "+" + b.name());

    for (size_t s = 0; s < a.num_statements(); ++s) {
        result.add_statement(a.statement(s));
    }
    for (size_t s = 0; s < b.num_statements(); ++s) {
        result.add_statement(b.statement(s));
    }

    return result;
}

/// Validate that all statements in the iteration space have consistent
/// dimensions (same number of iteration variables).
bool validate_consistency(const IterationSpace& ispace) {
    if (ispace.num_statements() <= 1) return true;

    size_t ndim = ispace.statement(0).domain.ndim();
    for (size_t s = 1; s < ispace.num_statements(); ++s) {
        if (ispace.statement(s).domain.ndim() != ndim) return false;
    }
    return true;
}

} // namespace symplex::polyhedral
