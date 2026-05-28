// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/polyhedral/integer_polytope.h"
#include "symplex/polyhedral/affine_map.h"
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

namespace symplex::polyhedral {

/// DependencyType: the kind of data dependence between two operations.
enum class DependencyType {
    RAW,   // Read After Write  (true dependence)
    WAR,   // Write After Read  (anti-dependence)
    WAW,   // Write After Write (output dependence)
};

inline std::string dependency_type_to_string(DependencyType dt) {
    switch (dt) {
        case DependencyType::RAW: return "RAW";
        case DependencyType::WAR: return "WAR";
        case DependencyType::WAW: return "WAW";
    }
    return "UNKNOWN";
}

/// DependencyVector: a single dependence vector  d = i_sink - i_source
/// in the iteration space.
struct DependencyVector {
    std::vector<int64_t> components;  // direction per dimension
    DependencyType       type;

    /// Is this a lexicographically positive dependency? (i.e., d >= 0 componentwise)
    [[nodiscard]] bool is_lex_positive() const {
        return std::all_of(components.begin(), components.end(),
                           [](int64_t c) { return c >= 0; });
    }

    /// Is this a distance vector (all components are non-negative constants)?
    [[nodiscard]] bool is_distance_vector() const {
        return is_lex_positive();
    }

    /// Is this a loop-carried dependency in dimension `d`?
    [[nodiscard]] bool is_carried_at(size_t d) const {
        return d < components.size() && components[d] > 0;
    }

    /// Is dimension `d` free of this dependency (i.e., can be parallelized)?
    [[nodiscard]] bool is_parallel_at(size_t d) const {
        return d < components.size() && components[d] == 0;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << dependency_type_to_string(type) << "(";
        for (size_t d = 0; d < components.size(); ++d) {
            if (d > 0) oss << ", ";
            oss << components[d];
        }
        oss << ")";
        return oss.str();
    }
};

/// AccessRelation: describes how a statement accesses a tensor.
///   access(is) = M * is + c
/// where is is the iteration vector and the result selects tensor elements.
struct AccessRelation {
    AffineMap access_map;   // maps iteration vector to tensor coordinates
    enum Mode { READ, WRITE } mode;
    std::string array_name; // name of the accessed array (for self-dep detection)

    AccessRelation(AffineMap map, Mode m, std::string name = "")
        : access_map(std::move(map)), mode(m), array_name(std::move(name)) {}
};

/// DependencyPolyhedron: the set of all dependence vectors between two
/// statements in the iteration space.
///
/// Mathematically:  D = { d | exists i1, i2 in I such that
///   access1(i1) = access2(i2)  AND  i2 - i1 = d  AND  constraints }
///
/// This is computed by intersecting the access relation domains and
/// expressing the difference as a new polytope in the "d" space.
class DependencyPolyhedron {
public:
    DependencyPolyhedron(
        const IntegerPolytope& domain1,
        const IntegerPolytope& domain2,
        const AccessRelation&  access1,
        const AccessRelation&  access2,
        DependencyType         dtype
    )
        : dtype_(dtype)
        , ndim_(domain1.ndim())
    {
        compute_dependency_vectors(domain1, domain2, access1, access2);
    }

    /// Direct construction from pre-computed vectors.
    DependencyPolyhedron(
        std::vector<DependencyVector> vecs,
        DependencyType dtype,
        size_t ndim
    )
        : dtype_(dtype)
        , ndim_(ndim)
        , vectors_(std::move(vecs))
    {}

    // ── Accessors ───────────────────────────────────────────────────

    [[nodiscard]] DependencyType type() const { return dtype_; }
    [[nodiscard]] size_t ndim() const { return ndim_; }
    [[nodiscard]] const std::vector<DependencyVector>& vectors() const { return vectors_; }

    // ── Dependency queries ──────────────────────────────────────────

    /// Does any dependency exist at all?
    [[nodiscard]] bool has_dependencies() const { return !vectors_.empty(); }

    /// Is the given dimension free of all dependencies? (can be parallelized)
    [[nodiscard]] bool is_parallelizable(size_t dim) const {
        for (const auto& dv : vectors_) {
            if (dv.is_carried_at(dim)) return false;
        }
        return true;
    }

    /// Get all dimensions that can be parallelized (no carried dependencies).
    [[nodiscard]] std::vector<size_t> parallelizable_dims() const {
        std::vector<size_t> result;
        for (size_t d = 0; d < ndim_; ++d) {
            if (is_parallelizable(d)) result.push_back(d);
        }
        return result;
    }

    /// Get all dimensions that carry at least one dependency (must be sequential).
    [[nodiscard]] std::vector<size_t> sequential_dims() const {
        std::vector<size_t> result;
        for (size_t d = 0; d < ndim_; ++d) {
            if (!is_parallelizable(d)) result.push_back(d);
        }
        return result;
    }

    /// Check if a given affine transformation preserves all dependencies.
    /// A transformation T preserves dependencies iff for every dependency vector d,
    /// T(d) is lexicographically positive.
    [[nodiscard]] bool is_valid_transformation(const AffineMap& T) const {
        for (const auto& dv : vectors_) {
            auto transformed = T.apply(dv.components);
            // Check lexicographic positivity
            bool lex_pos = false;
            for (size_t d = 0; d < transformed.size(); ++d) {
                if (transformed[d] > 0) { lex_pos = true; break; }
                if (transformed[d] < 0) { lex_pos = false; break; }
                // transformed[d] == 0: continue checking
            }
            if (!lex_pos) return false;
        }
        return true;
    }

    /// Filter: keep only dependencies of a specific type.
    [[nodiscard]] DependencyPolyhedron filter_by_type(DependencyType t) const {
        std::vector<DependencyVector> filtered;
        for (const auto& dv : vectors_) {
            if (dv.type == t) filtered.push_back(dv);
        }
        return DependencyPolyhedron(std::move(filtered), t, ndim_);
    }

    /// Merge two dependency polyhedra.
    [[nodiscard]] DependencyPolyhedron merge(const DependencyPolyhedron& other) const {
        assert(ndim_ == other.ndim_);
        std::vector<DependencyVector> merged = vectors_;
        merged.insert(merged.end(), other.vectors_.begin(), other.vectors_.end());
        return DependencyPolyhedron(std::move(merged), dtype_, ndim_);
    }

    // ── String representation ───────────────────────────────────────

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "DepPoly{type=" << dependency_type_to_string(dtype_)
            << ", ndim=" << ndim_ << ", vectors=[";
        for (size_t i = 0; i < vectors_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << vectors_[i].to_string();
        }
        oss << "]}";
        return oss.str();
    }

private:
    DependencyType dtype_;
    size_t ndim_;
    std::vector<DependencyVector> vectors_;

    /// Compute dependency vectors by checking all pairs of iteration points
    /// where the two access relations touch the same memory location.
    void compute_dependency_vectors(
        const IntegerPolytope& domain1,
        const IntegerPolytope& domain2,
        const AccessRelation&  access1,
        const AccessRelation&  access2
    ) {
        // Enumerate points in both domains (feasible for small analysis spaces)
        auto points1 = domain1.enumerate_points();
        auto points2 = domain2.enumerate_points();

        for (const auto& i1 : points1) {
            auto addr1 = access1.access_map.apply(i1);
            for (const auto& i2 : points2) {
                auto addr2 = access2.access_map.apply(i2);

                // Same memory location?
                if (addr1 != addr2) continue;

                // Compute dependency vector: d = i2 - i1
                DependencyVector dv;
                dv.type = dtype_;
                dv.components.resize(ndim_);
                for (size_t d = 0; d < ndim_; ++d) {
                    dv.components[d] = i2[d] - i1[d];
                }

                // Only keep lexicographically positive vectors
                // (a dependency from i1 to i2 means i2 happens after i1)
                bool lex_pos = false;
                for (size_t d = 0; d < ndim_; ++d) {
                    if (dv.components[d] > 0) { lex_pos = true; break; }
                    if (dv.components[d] < 0) break;
                }
                if (lex_pos) {
                    // Check for duplicate
                    bool duplicate = false;
                    for (const auto& existing : vectors_) {
                        if (existing.components == dv.components && existing.type == dv.type) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        vectors_.push_back(std::move(dv));
                    }
                }
            }
        }
    }
};

} // namespace symplex::polyhedral
