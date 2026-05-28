// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/union_map.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace symplex::polyhedral {

// ── Compile-time verification ────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<UnionMap>);
// BasicMap is not default-constructible (requires IntegerPolytope + AffineMap).
static_assert(std::is_constructible_v<UnionMap::BasicMap, IntegerPolytope, AffineMap>);

// ── Non-inline helper functions for UnionMap ─────────────────────────────────

/// Compute the range (image) of a UnionMap.
/// The range is the union of images of all basic maps.
IntegerPolytope compute_range(const UnionMap& umap) {
    if (umap.empty()) return IntegerPolytope(0);

    // Collect all output points by applying the union map to all
    // points in each basic map's domain.
    size_t ndim_out = umap.maps()[0].map.n_out();
    if (ndim_out == 0) return IntegerPolytope(0);

    // Compute bounds for the output space.
    std::vector<std::pair<int64_t, int64_t>> bounds(
        ndim_out, {INT64_MAX, INT64_MIN}
    );

    for (const auto& bm : umap.maps()) {
        auto input_pts = bm.domain.enumerate_points();
        for (const auto& pt : input_pts) {
            auto out_pt = bm.map.apply(pt);
            for (size_t d = 0; d < ndim_out && d < out_pt.size(); ++d) {
                bounds[d].first = std::min(bounds[d].first, out_pt[d]);
                bounds[d].second = std::max(bounds[d].second, out_pt[d]);
            }
        }
    }

    return make_rectangular_polytope(bounds);
}

/// Intersect a UnionMap with a domain polytope.
/// Returns a new UnionMap whose basic maps have their domains
/// intersected with the given polytope.
UnionMap intersect_domain(
    const UnionMap& umap,
    const IntegerPolytope& domain
) {
    UnionMap result;
    for (const auto& bm : umap.maps()) {
        auto intersected = bm.domain.intersect(domain);
        result.add_basic_map(intersected, bm.map);
    }
    return result;
}

/// Compute the number of distinct affine functions in a UnionMap.
size_t count_distinct_maps(const UnionMap& umap) {
    if (umap.empty()) return 0;

    std::vector<std::pair<
        std::vector<std::vector<int64_t>>,
        std::vector<int64_t>
    >> seen;

    for (const auto& bm : umap.maps()) {
        bool found = false;
        const auto& m = bm.map.matrix();
        const auto& c = bm.map.offset();

        for (const auto& [sm, sc] : seen) {
            if (sm == m && sc == c) {
                found = true;
                break;
            }
        }

        if (!found) {
            seen.emplace_back(m, c);
        }
    }

    return seen.size();
}

/// Simplify a UnionMap by merging basic maps that have adjacent domains
/// and identical affine functions.
UnionMap simplify_union_map(const UnionMap& umap) {
    if (umap.size() <= 1) return umap;

    // Group basic maps by their affine function.
    struct MapKey {
        std::vector<std::vector<int64_t>> matrix;
        std::vector<int64_t> offset;

        bool operator==(const MapKey& other) const {
            return matrix == other.matrix && offset == other.offset;
        }
    };

    // Simple grouping by identity.
    std::vector<MapKey> keys;
    std::vector<std::vector<size_t>> groups;

    for (size_t i = 0; i < umap.maps().size(); ++i) {
        const auto& bm = umap.maps()[i];
        MapKey key{bm.map.matrix(), bm.map.offset()};

        bool found = false;
        for (size_t g = 0; g < keys.size(); ++g) {
            if (keys[g] == key) {
                groups[g].push_back(i);
                found = true;
                break;
            }
        }

        if (!found) {
            keys.push_back(std::move(key));
            groups.push_back({i});
        }
    }

    // For each group, create one basic map with a merged domain.
    UnionMap result;
    for (size_t g = 0; g < groups.size(); ++g) {
        if (groups[g].size() == 1) {
            result.add_basic_map(umap.maps()[groups[g][0]]);
        } else {
            // Merge domains: take the convex hull.
            IntegerPolytope merged_domain = umap.maps()[groups[g][0]].domain;
            for (size_t i = 1; i < groups[g].size(); ++i) {
                // Approximate union by taking broader bounds.
                auto bounds_a = merged_domain.bounds();
                auto bounds_b = umap.maps()[groups[g][i]].domain.bounds();

                IntegerPolytope hull(merged_domain.ndim());
                for (size_t d = 0; d < bounds_a.size() && d < bounds_b.size(); ++d) {
                    int64_t lo = std::min(bounds_a[d].first, bounds_b[d].first);
                    int64_t hi = std::max(bounds_a[d].second, bounds_b[d].second);
                    hull.add_range_bound(d, lo, hi);
                }
                merged_domain = hull;
            }
            result.add_basic_map(merged_domain, umap.maps()[groups[g][0]].map);
        }
    }

    return result;
}

/// Create a UnionMap from an AffineMap with a full domain.
UnionMap union_map_from_affine(
    const AffineMap& map,
    const IntegerPolytope& domain
) {
    UnionMap result;
    result.add_basic_map(domain, map);
    return result;
}

} // namespace symplex::polyhedral
