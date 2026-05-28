// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>

namespace symplex::polyhedral {

/// Represents a single linear inequality constraint:  a^T * i + b >= 0
/// where i is a vector of integer loop iterators.
struct Inequality {
    std::vector<int64_t> coefficients;   // A row: one coefficient per iterator dimension
    int64_t               constant;       // b (the constant term)

    Inequality() : constant(0) {}
    Inequality(size_t ndim, int64_t c) : coefficients(ndim, 0), constant(c) {}

    /// Evaluate the constraint at a concrete point.
    [[nodiscard]] int64_t evaluate(const std::vector<int64_t>& point) const {
        int64_t sum = constant;
        for (size_t d = 0; d < coefficients.size() && d < point.size(); ++d) {
            sum += coefficients[d] * point[d];
        }
        return sum;
    }

    /// Is the constraint satisfied at `point`?
    [[nodiscard]] bool satisfied(const std::vector<int64_t>& point) const {
        return evaluate(point) >= 0;
    }

    std::string to_string() const {
        std::ostringstream oss;
        bool first = true;
        for (size_t d = 0; d < coefficients.size(); ++d) {
            if (coefficients[d] == 0) continue;
            if (!first) oss << (coefficients[d] > 0 ? " + " : " - ");
            else if (coefficients[d] < 0) oss << "-";
            int64_t abs_c = std::abs(coefficients[d]);
            if (abs_c != 1) oss << abs_c << "*";
            oss << "i" << d;
            first = false;
        }
        if (constant != 0 || first) {
            if (!first && constant > 0) oss << " + ";
            else if (!first && constant < 0) oss << " - ";
            else if (first && constant < 0) oss << "-";
            oss << std::abs(constant);
        }
        oss << " >= 0";
        return oss.str();
    }
};

/// Represents a single linear equality constraint:  a^T * i + b == 0
struct Equality {
    std::vector<int64_t> coefficients;
    int64_t               constant;

    Equality() : constant(0) {}
    Equality(size_t ndim, int64_t c) : coefficients(ndim, 0), constant(c) {}

    std::string to_string() const {
        std::ostringstream oss;
        bool first = true;
        for (size_t d = 0; d < coefficients.size(); ++d) {
            if (coefficients[d] == 0) continue;
            if (!first) oss << (coefficients[d] > 0 ? " + " : " - ");
            else if (coefficients[d] < 0) oss << "-";
            int64_t abs_c = std::abs(coefficients[d]);
            if (abs_c != 1) oss << abs_c << "*";
            oss << "i" << d;
            first = false;
        }
        if (constant != 0 || first) {
            if (!first && constant > 0) oss << " + ";
            else if (!first && constant < 0) oss << " - ";
            else if (first && constant < 0) oss << "-";
            oss << std::abs(constant);
        }
        oss << " == 0";
        return oss.str();
    }

    [[nodiscard]] int64_t evaluate(const std::vector<int64_t>& point) const {
        int64_t sum = constant;
        for (size_t d = 0; d < coefficients.size() && d < point.size(); ++d) {
            sum += coefficients[d] * point[d];
        }
        return sum;
    }

    [[nodiscard]] bool satisfied(const std::vector<int64_t>& point) const {
        return evaluate(point) == 0;
    }
};

/// IntegerPolytope:  { i in Z^n | A*i + b >= 0  and  equalities }
///
/// This is the fundamental geometric object.  An AI training loop nest
/// is represented as a polytope: each loop iterator becomes one dimension,
/// and loop bounds become the inequality constraints.
class IntegerPolytope {
public:
    /// Construct an empty polytope with `ndim` iterator dimensions.
    explicit IntegerPolytope(size_t ndim)
        : ndim_(ndim) {}

    /// Construct from explicit constraint matrices.
    IntegerPolytope(
        size_t ndim,
        std::vector<Inequality> ineqs,
        std::vector<Equality>   eqs = {}
    )
        : ndim_(ndim)
        , inequalities_(std::move(ineqs))
        , equalities_(std::move(eqs))
    {}

    // ── Accessors ───────────────────────────────────────────────────

    [[nodiscard]] size_t ndim() const { return ndim_; }

    [[nodiscard]] const std::vector<Inequality>& inequalities() const { return inequalities_; }
    [[nodiscard]] const std::vector<Equality>&   equalities()   const { return equalities_;   }

    // ── Constraint mutation ─────────────────────────────────────────

    void add_inequality(Inequality ineq) {
        // Pad or trim to match ndim_
        if (ineq.coefficients.size() < ndim_) {
            ineq.coefficients.resize(ndim_, 0);
        }
        inequalities_.push_back(std::move(ineq));
    }

    void add_equality(Equality eq) {
        if (eq.coefficients.size() < ndim_) {
            eq.coefficients.resize(ndim_, 0);
        }
        equalities_.push_back(std::move(eq));
    }

    /// Add a simple lower bound:  i[dim] >= lo
    void add_lower_bound(size_t dim, int64_t lo) {
        assert(dim < ndim_);
        Inequality ineq(ndim_, 0);
        ineq.coefficients[dim] = 1;
        ineq.constant = -lo;
        add_inequality(ineq);
    }

    /// Add a simple upper bound:  i[dim] <= hi  ->  -i[dim] + hi >= 0
    void add_upper_bound(size_t dim, int64_t hi) {
        assert(dim < ndim_);
        Inequality ineq(ndim_, 0);
        ineq.coefficients[dim] = -1;
        ineq.constant = hi;
        add_inequality(ineq);
    }

    /// Add a range bound:  lo <= i[dim] <= hi
    void add_range_bound(size_t dim, int64_t lo, int64_t hi) {
        add_lower_bound(dim, lo);
        add_upper_bound(dim, hi);
    }

    // ── Point membership ────────────────────────────────────────────

    /// Check whether a concrete integer point belongs to this polytope.
    [[nodiscard]] bool contains(const std::vector<int64_t>& point) const {
        if (point.size() != ndim_) return false;
        for (const auto& ineq : inequalities_) {
            if (!ineq.satisfied(point)) return false;
        }
        for (const auto& eq : equalities_) {
            if (!eq.satisfied(point)) return false;
        }
        return true;
    }

    // ── Set operations ──────────────────────────────────────────────

    /// Intersection with another polytope (must have same ndim).
    [[nodiscard]] IntegerPolytope intersect(const IntegerPolytope& other) const {
        assert(ndim_ == other.ndim_);
        IntegerPolytope result(ndim_);
        result.inequalities_ = inequalities_;
        result.equalities_   = equalities_;
        for (const auto& ineq : other.inequalities_) result.inequalities_.push_back(ineq);
        for (const auto& eq   : other.equalities_)   result.equalities_.push_back(eq);
        return result;
    }

    /// Cartesian product (creates a higher-dimensional polytope).
    [[nodiscard]] IntegerPolytope cartesian_product(const IntegerPolytope& other) const {
        size_t new_ndim = ndim_ + other.ndim_;
        IntegerPolytope result(new_ndim);

        // Lift our constraints into the combined space
        for (auto ineq : inequalities_) {
            ineq.coefficients.resize(new_ndim, 0);
            result.inequalities_.push_back(std::move(ineq));
        }
        for (auto eq : equalities_) {
            eq.coefficients.resize(new_ndim, 0);
            result.equalities_.push_back(std::move(eq));
        }

        // Lift other's constraints into the combined space (offset by ndim_)
        for (auto ineq : other.inequalities_) {
            std::vector<int64_t> coeffs(new_ndim, 0);
            for (size_t d = 0; d < ineq.coefficients.size(); ++d) {
                coeffs[ndim_ + d] = ineq.coefficients[d];
            }
            ineq.coefficients = std::move(coeffs);
            result.inequalities_.push_back(std::move(ineq));
        }
        for (auto eq : other.equalities_) {
            std::vector<int64_t> coeffs(new_ndim, 0);
            for (size_t d = 0; d < eq.coefficients.size(); ++d) {
                coeffs[ndim_ + d] = eq.coefficients[d];
            }
            eq.coefficients = std::move(coeffs);
            result.equalities_.push_back(std::move(eq));
        }

        return result;
    }

    // ── Image / preimage under affine map ───────────────────────────

    /// Apply an affine transformation described by matrix M (ndim_out x ndim_in)
    /// and offset vector c (ndim_out). Returns the image polytope.
    /// This implements:  new_point = M * old_point + c
    [[nodiscard]] IntegerPolytope image(
        const std::vector<std::vector<int64_t>>& M,
        const std::vector<int64_t>& c
    ) const {
        size_t new_ndim = M.size();
        IntegerPolytope result(new_ndim);

        // For each original constraint  a^T * i + b >= 0
        // substitute  i = M^{-1} * (j - c)  when M is square and invertible
        // For simplicity we handle the square invertible case.
        if (new_ndim == ndim_) {
            auto inv = invert_matrix(M);
            if (inv.has_value()) {
                for (const auto& ineq : inequalities_) {
                    Inequality new_ineq(new_ndim, 0);
                    // a^T * M^{-1} * j  -  a^T * M^{-1} * c  +  b >= 0
                    for (size_t j = 0; j < new_ndim; ++j) {
                        int64_t coeff = 0;
                        for (size_t k = 0; k < ndim_; ++k) {
                            coeff += ineq.coefficients[k] * (*inv)[k][j];
                        }
                        new_ineq.coefficients[j] = coeff;
                    }
                    int64_t offset = 0;
                    for (size_t k = 0; k < ndim_; ++k) {
                        int64_t inner = 0;
                        for (size_t j = 0; j < new_ndim; ++j) {
                            inner += (*inv)[k][j] * c[j];
                        }
                        offset += ineq.coefficients[k] * inner;
                    }
                    new_ineq.constant = ineq.constant - offset;
                    result.inequalities_.push_back(std::move(new_ineq));
                }
                for (const auto& eq : equalities_) {
                    Equality new_eq(new_ndim, 0);
                    for (size_t j = 0; j < new_ndim; ++j) {
                        int64_t coeff = 0;
                        for (size_t k = 0; k < ndim_; ++k) {
                            coeff += eq.coefficients[k] * (*inv)[k][j];
                        }
                        new_eq.coefficients[j] = coeff;
                    }
                    int64_t offset = 0;
                    for (size_t k = 0; k < ndim_; ++k) {
                        int64_t inner = 0;
                        for (size_t j = 0; j < new_ndim; ++j) {
                            inner += (*inv)[k][j] * c[j];
                        }
                        offset += eq.coefficients[k] * inner;
                    }
                    new_eq.constant = eq.constant - offset;
                    result.equalities_.push_back(std::move(new_eq));
                }
                return result;
            }
        }

        // Fallback: return a trivially constrained polytope (over-approximation)
        for (size_t d = 0; d < new_ndim; ++d) {
            result.add_range_bound(d, 0, 1'000'000);  // loose bound
        }
        return result;
    }

    // ── Enumeration (for small polytopes only) ──────────────────────

    /// Enumerate all integer points inside the polytope.
    /// WARNING: only feasible for small, bounded polytopes.
    [[nodiscard]] std::vector<std::vector<int64_t>> enumerate_points() const {
        std::vector<std::vector<int64_t>> points;
        std::vector<int64_t> current(ndim_, 0);

        // First, determine conservative bounds for each dimension
        std::vector<std::pair<int64_t, int64_t>> bounds(ndim_, {0, 0});
        for (size_t d = 0; d < ndim_; ++d) {
            int64_t lo = INT64_MIN / 2;
            int64_t hi = INT64_MAX / 2;
            for (const auto& ineq : inequalities_) {
                if (d < ineq.coefficients.size()) {
                    int64_t c = ineq.coefficients[d];
                    if (c > 0) {
                        // c*i[d] + ... + b >= 0  =>  i[d] >= (-b - ...)/c
                        // approximate: i[d] >= -b/c (conservative)
                        int64_t approx = -ineq.constant / c - 1;
                        lo = std::max(lo, approx);
                    } else if (c < 0) {
                        int64_t approx = ineq.constant / (-c) + 1;
                        hi = std::min(hi, approx);
                    }
                }
            }
            // Safety: cap enumeration range
            lo = std::max(lo, int64_t(-1024));
            hi = std::min(hi, int64_t(1024));
            bounds[d] = {lo, hi};
        }

        enumerate_recursive(points, current, bounds, 0);
        return points;
    }

    /// Count the number of integer points using Ehrhart-theoretic approach.
    /// For rectangular polytopes this is just the product of range widths.
    [[nodiscard]] int64_t count_points() const {
        // For simple rectangular polytopes (only range bounds), compute exactly
        std::vector<std::pair<int64_t, int64_t>> bounds(ndim_);
        for (size_t d = 0; d < ndim_; ++d) {
            int64_t lo = 0, hi = 0;
            bool found_lo = false, found_hi = false;
            for (const auto& ineq : inequalities_) {
                if (d < ineq.coefficients.size()) {
                    if (ineq.coefficients[d] == 1 && !found_lo) {
                        lo = -ineq.constant;
                        found_lo = true;
                    } else if (ineq.coefficients[d] == -1 && !found_hi) {
                        hi = ineq.constant;
                        found_hi = true;
                    }
                }
            }
            if (!found_lo || !found_hi) return -1; // Cannot determine exact count
            bounds[d] = {lo, hi};
        }

        int64_t count = 1;
        for (const auto& [lo, hi] : bounds) {
            if (hi < lo) return 0;
            count *= (hi - lo + 1);
        }
        return count;
    }

    /// Best-effort per-dimension bounds extraction.
    /// Returns [lo, hi] for each dimension. When explicit affine unit
    /// bounds are absent, conservative fallback values are used.
    [[nodiscard]] std::vector<std::pair<int64_t, int64_t>> bounds() const {
        std::vector<std::pair<int64_t, int64_t>> result(ndim_, {0, 0});
        for (size_t d = 0; d < ndim_; ++d) {
            int64_t lo = INT64_MIN / 2;
            int64_t hi = INT64_MAX / 2;
            for (const auto& ineq : inequalities_) {
                if (d >= ineq.coefficients.size()) continue;
                if (ineq.coefficients[d] == 1) {
                    lo = std::max(lo, -ineq.constant);
                } else if (ineq.coefficients[d] == -1) {
                    hi = std::min(hi, ineq.constant);
                }
            }
            if (lo <= INT64_MIN / 4) lo = 0;
            if (hi >= INT64_MAX / 4) hi = lo;
            result[d] = {lo, hi};
        }
        return result;
    }

    // ── Dimension-wise projection ───────────────────────────────────

    /// Project out dimension `dim` via Fourier-Motzkin elimination.
    [[nodiscard]] IntegerPolytope project_out(size_t dim) const {
        assert(dim < ndim_);
        size_t new_ndim = ndim_ - 1;
        IntegerPolytope result(new_ndim);

        // Split inequalities based on coefficient sign at `dim`
        std::vector<Inequality> positive;  // coeff[dim] > 0
        std::vector<Inequality> negative;  // coeff[dim] < 0
        std::vector<Inequality> zero;      // coeff[dim] == 0

        for (auto ineq : inequalities_) {
            int64_t c = (dim < ineq.coefficients.size()) ? ineq.coefficients[dim] : 0;
            ineq.coefficients.erase(ineq.coefficients.begin() + static_cast<ptrdiff_t>(dim));
            if (c > 0) {
                // Normalize: divide by |c|
                int64_t abs_c = c;
                for (auto& v : ineq.coefficients) v /= abs_c;
                ineq.constant /= abs_c;
                positive.push_back(std::move(ineq));
            } else if (c < 0) {
                int64_t abs_c = -c;
                for (auto& v : ineq.coefficients) v /= abs_c;
                ineq.constant /= abs_c;
                negative.push_back(std::move(ineq));
            } else {
                zero.push_back(std::move(ineq));
            }
        }

        // Inequalities not involving dim pass through
        for (auto& ineq : zero) result.add_inequality(std::move(ineq));

        // Combine each positive with each negative (Fourier-Motzkin)
        for (const auto& p : positive) {
            for (const auto& n : negative) {
                Inequality combined(new_ndim, 0);
                for (size_t d = 0; d < new_ndim; ++d) {
                    combined.coefficients[d] = p.coefficients[d] + n.coefficients[d];
                }
                combined.constant = p.constant + n.constant;
                result.add_inequality(std::move(combined));
            }
        }

        // Handle equalities: solve for dim and substitute
        for (auto eq : equalities_) {
            int64_t c = (dim < eq.coefficients.size()) ? eq.coefficients[dim] : 0;
            if (c != 0) {
                eq.coefficients.erase(eq.coefficients.begin() + static_cast<ptrdiff_t>(dim));
                // This equality becomes an inequality pair after substitution
                for (auto& v : eq.coefficients) v /= c;
                eq.constant /= c;

                // Substitute into all inequalities that had dim != 0
                // For simplicity, add as both >=0 and <=0
                Inequality as_ineq_pos;
                as_ineq_pos.coefficients = eq.coefficients;
                as_ineq_pos.constant = eq.constant;
                result.add_inequality(as_ineq_pos);

                Inequality as_ineq_neg;
                as_ineq_neg.coefficients.resize(eq.coefficients.size());
                for (size_t d = 0; d < eq.coefficients.size(); ++d) {
                    as_ineq_neg.coefficients[d] = -eq.coefficients[d];
                }
                as_ineq_neg.constant = -eq.constant;
                result.add_inequality(as_ineq_neg);
            } else {
                eq.coefficients.erase(eq.coefficients.begin() + static_cast<ptrdiff_t>(dim));
                result.add_equality(std::move(eq));
            }
        }

        return result;
    }

    /// Lexicographic minimum: find the lexicographically smallest point.
    [[nodiscard]] std::optional<std::vector<int64_t>> lexmin() const {
        auto points = enumerate_points();
        if (points.empty()) return std::nullopt;
        std::sort(points.begin(), points.end());
        return points.front();
    }

    /// Lexicographic maximum: find the lexicographically largest point.
    [[nodiscard]] std::optional<std::vector<int64_t>> lexmax() const {
        auto points = enumerate_points();
        if (points.empty()) return std::nullopt;
        std::sort(points.begin(), points.end());
        return points.back();
    }

    /// Is the polytope empty? (contains no integer points)
    [[nodiscard]] bool is_empty() const {
        return enumerate_points().empty();
    }

    /// Check if this polytope is a subset of `other`.
    [[nodiscard]] bool is_subset_of(const IntegerPolytope& other) const {
        // Every point in this must also be in other
        auto points = enumerate_points();
        for (const auto& pt : points) {
            if (!other.contains(pt)) return false;
        }
        return true;
    }

    // ── String representation ───────────────────────────────────────

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Polytope{ndim=" << ndim_ << ", ineqs=[";
        for (size_t i = 0; i < inequalities_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << inequalities_[i].to_string();
        }
        oss << "], eqs=[";
        for (size_t i = 0; i < equalities_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << equalities_[i].to_string();
        }
        oss << "]}";
        return oss.str();
    }

private:
    size_t ndim_;
    std::vector<Inequality> inequalities_;
    std::vector<Equality>   equalities_;

    /// Recursive enumeration helper.
    void enumerate_recursive(
        std::vector<std::vector<int64_t>>& points,
        std::vector<int64_t>& current,
        const std::vector<std::pair<int64_t, int64_t>>& bounds,
        size_t dim
    ) const {
        if (dim == ndim_) {
            if (contains(current)) {
                points.push_back(current);
            }
            return;
        }
        for (int64_t v = bounds[dim].first; v <= bounds[dim].second; ++v) {
            current[dim] = v;
            // Early pruning: check all constraints that only involve dims 0..dim
            bool feasible = true;
            for (const auto& ineq : inequalities_) {
                bool only_involves_current_dims = true;
                for (size_t d = dim + 1; d < ndim_; ++d) {
                    if (d < ineq.coefficients.size() && ineq.coefficients[d] != 0) {
                        only_involves_current_dims = false;
                        break;
                    }
                }
                if (only_involves_current_dims && !ineq.satisfied(current)) {
                    feasible = false;
                    break;
                }
            }
            if (feasible) {
                enumerate_recursive(points, current, bounds, dim + 1);
            }
        }
    }

    /// Attempt to invert a square integer matrix using Gaussian elimination
    /// with rational arithmetic (returned as integer matrix scaled by determinant).
    static std::optional<std::vector<std::vector<int64_t>>>
    invert_matrix(const std::vector<std::vector<int64_t>>& M) {
        size_t n = M.size();
        if (n == 0) return std::nullopt;
        for (const auto& row : M) {
            if (row.size() != n) return std::nullopt;
        }

        // Augmented matrix [M | I] using double for simplicity
        std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug[i][j] = static_cast<double>(M[i][j]);
            }
            aug[i][n + i] = 1.0;
        }

        // Gaussian elimination with partial pivoting
        for (size_t col = 0; col < n; ++col) {
            // Find pivot
            size_t pivot = col;
            for (size_t row = col + 1; row < n; ++row) {
                if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) {
                    pivot = row;
                }
            }
            if (std::abs(aug[pivot][col]) < 1e-12) return std::nullopt;

            std::swap(aug[col], aug[pivot]);

            double pivot_val = aug[col][col];
            for (size_t j = 0; j < 2 * n; ++j) {
                aug[col][j] /= pivot_val;
            }
            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Extract inverse and round to integer
        std::vector<std::vector<int64_t>> inv(n, std::vector<int64_t>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inv[i][j] = static_cast<int64_t>(std::round(aug[i][n + j]));
            }
        }
        return inv;
    }
};

/// Factory: create a rectangular polytope representing a loop nest.
/// e.g. for a 4D loop with bounds [0,B), [0,H), [0,W), [0,C):
inline IntegerPolytope make_rectangular_polytope(
    const std::vector<std::pair<int64_t, int64_t>>& ranges
) {
    size_t ndim = ranges.size();
    IntegerPolytope poly(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        poly.add_range_bound(d, ranges[d].first, ranges[d].second - 1);
    }
    return poly;
}

} // namespace symplex::polyhedral
