// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/dependency.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace symplex::polyhedral {

// ── Compile-time verification ────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<DependencyVector>);
// AccessRelation is not default-constructible (requires AffineMap + Mode).
static_assert(std::is_constructible_v<AccessRelation, AffineMap, AccessRelation::Mode>);

// ── Non-inline helper functions for dependency analysis ──────────────────────

/// Compute the set of all unique dependency directions across all
/// dependency vectors.  A direction is a simplified representation
/// where only the sign of each component matters: (+, 0, -).
///
/// For example, (2, 0, -1) has direction (+, 0, -).
std::vector<std::vector<int8_t>> compute_dependency_directions(
    const std::vector<DependencyVector>& vectors
) {
    std::vector<std::vector<int8_t>> directions;

    for (const auto& dv : vectors) {
        std::vector<int8_t> dir(dv.components.size());
        for (size_t d = 0; d < dv.components.size(); ++d) {
            if (dv.components[d] > 0) dir[d] = 1;
            else if (dv.components[d] < 0) dir[d] = -1;
            else dir[d] = 0;
        }

        // Check for uniqueness.
        bool found = false;
        for (const auto& existing : directions) {
            if (existing == dir) { found = true; break; }
        }
        if (!found) {
            directions.push_back(std::move(dir));
        }
    }

    return directions;
}

/// Classify a dependency vector as one of: loop-independent or loop-carried.
/// A dependency is loop-independent if all components are zero (which
/// shouldn't happen for valid dependencies) or if it can be satisfied
/// within a single iteration of the outermost loop.
///
/// A dependency is loop-carried at dimension d if d is the outermost
/// dimension with a non-zero component.
int classify_dependency_level(const DependencyVector& dv) {
    for (size_t d = 0; d < dv.components.size(); ++d) {
        if (dv.components[d] != 0) {
            return static_cast<int>(d);
        }
    }
    return -1;  // Zero vector: no dependency.
}

/// Compute the distance of a dependency vector, which is the
/// lexicographic minimum distance.  For distance vectors (all
/// non-negative), this is just the first non-zero component.
int64_t dependency_distance(const DependencyVector& dv) {
    for (size_t d = 0; d < dv.components.size(); ++d) {
        if (dv.components[d] != 0) {
            return dv.components[d];
        }
    }
    return 0;
}

/// Check if two access relations can potentially conflict
/// (i.e., access the same memory location).
bool accesses_may_conflict(
    const AccessRelation& a,
    const AccessRelation& b
) {
    // If they access different numbers of dimensions, they can't conflict.
    if (a.access_map.n_out() != b.access_map.n_out()) return false;

    // If at least one is a read, there's no conflict unless the other writes.
    if (a.mode == AccessRelation::READ && b.mode == AccessRelation::READ) {
        return false;  // Read-read: no dependency.
    }

    // At least one write: potential conflict.
    return true;
}

/// Merge dependency vectors that have the same direction and type.
/// This reduces the number of vectors while preserving all
/// constraint information.
std::vector<DependencyVector> merge_dependency_vectors(
    const std::vector<DependencyVector>& vectors
) {
    // Group by (direction, type) and keep the one with the smallest distance.
    std::vector<DependencyVector> merged;

    for (const auto& dv : vectors) {
        bool found = false;
        for (auto& existing : merged) {
            if (existing.type == dv.type &&
                existing.components.size() == dv.components.size()) {
                // Check if directions are the same.
                bool same_dir = true;
                for (size_t d = 0; d < dv.components.size(); ++d) {
                    int8_t s1 = (dv.components[d] > 0) ? 1 : (dv.components[d] < 0) ? -1 : 0;
                    int8_t s2 = (existing.components[d] > 0) ? 1 : (existing.components[d] < 0) ? -1 : 0;
                    if (s1 != s2) { same_dir = false; break; }
                }

                if (same_dir) {
                    // Keep the vector with the smallest distance (tightest constraint).
                    bool replace = true;
                    for (size_t d = 0; d < dv.components.size(); ++d) {
                        if (std::abs(dv.components[d]) > std::abs(existing.components[d])) {
                            replace = false;
                            break;
                        }
                    }
                    if (replace) {
                        existing = dv;
                    }
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            merged.push_back(dv);
        }
    }

    return merged;
}

/// Compute the dependency polyhedron for a self-dependency
/// (a statement that has a dependency with itself).
DependencyPolyhedron compute_self_dependency(
    const IntegerPolytope& domain,
    const AccessRelation& write_access,
    const AccessRelation& read_access,
    DependencyType dtype
) {
    // Self-dependency: i1 and i2 are in the same domain.
    return DependencyPolyhedron(domain, domain, write_access, read_access, dtype);
}

} // namespace symplex::polyhedral
