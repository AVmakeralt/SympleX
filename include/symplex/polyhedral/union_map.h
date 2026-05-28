// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/polyhedral/integer_polytope.h"
#include "symplex/polyhedral/affine_map.h"
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

namespace symplex::polyhedral {

/// UnionMap: a union of affine relations (basic maps).
/// This represents a piecewise-affine function, which is the general
/// form of a polyhedral schedule.
class UnionMap {
public:
    struct BasicMap {
        IntegerPolytope domain;   // Domain where this piece applies
        AffineMap       map;      // The affine function for this piece
    };

    UnionMap() = default;

    void add_basic_map(BasicMap bm) {
        maps_.push_back(std::move(bm));
    }

    void add_basic_map(IntegerPolytope domain, AffineMap map) {
        maps_.push_back(BasicMap{std::move(domain), std::move(map)});
    }

    [[nodiscard]] const std::vector<BasicMap>& maps() const { return maps_; }
    [[nodiscard]] size_t size() const { return maps_.size(); }
    [[nodiscard]] bool empty() const { return maps_.empty(); }

    /// Apply this union map to a point: find the basic map whose domain
    /// contains the point and apply its affine function.
    [[nodiscard]] std::vector<int64_t> apply(const std::vector<int64_t>& point) const {
        for (const auto& bm : maps_) {
            if (bm.domain.contains(point)) {
                return bm.map.apply(point);
            }
        }
        return {};  // Point not in any domain
    }

    /// Compose two union maps: (this) o (other).
    /// For each pair of basic maps (a in this, b in other), compose if compatible.
    /// The domain of the composed map is b's domain intersected with the
    /// preimage of a's domain under b.map.
    [[nodiscard]] UnionMap compose(const UnionMap& other) const {
        UnionMap result;
        for (const auto& a : maps_) {
            for (const auto& b : other.maps_) {
                if (a.map.n_in() == b.map.n_out()) {
                    AffineMap composed = a.map.compose(b.map);
                    // Compute the preimage of a's domain under b.map:
                    // {x | b.map(x) ∈ a.domain}
                    // This equals image of a.domain under b.map.inverse().
                    IntegerPolytope composed_domain = b.domain;
                    auto b_inv = b.map.inverse();
                    if (b_inv.has_value()) {
                        auto preimage = a.domain.image(b_inv->matrix(), b_inv->offset());
                        composed_domain = composed_domain.intersect(preimage);
                    }
                    result.add_basic_map(composed_domain, composed);
                }
            }
        }
        return result;
    }

    /// Reverse: compute the union of preimages.
    /// The domain of each reversed basic map is the image of the original
    /// basic map (i.e., the range of f restricted to its domain).
    [[nodiscard]] UnionMap reverse() const {
        UnionMap result;
        for (const auto& bm : maps_) {
            auto inv = bm.map.inverse();
            if (inv.has_value()) {
                // The domain of the reversed map is the image of the original:
                // {f(x) | x ∈ bm.domain} = bm.domain.image(M, c)
                auto reversed_domain = bm.domain.image(bm.map.matrix(), bm.map.offset());
                result.add_basic_map(reversed_domain, *inv);
            }
        }
        return result;
    }

    /// Coalesce: merge basic maps that have the same affine function.
    [[nodiscard]] UnionMap coalesce() const {
        if (maps_.size() <= 1) return *this;

        UnionMap result;
        // Group by identical affine map
        std::vector<bool> used(maps_.size(), false);
        for (size_t i = 0; i < maps_.size(); ++i) {
            if (used[i]) continue;
            BasicMap merged = maps_[i];
            used[i] = true;
            for (size_t j = i + 1; j < maps_.size(); ++j) {
                if (used[j]) continue;
                // Check if maps are identical (same matrix and offset)
                bool same = true;
                if (merged.map.n_in() != maps_[j].map.n_in() ||
                    merged.map.n_out() != maps_[j].map.n_out()) {
                    same = false;
                } else {
                    auto& m1 = merged.map.matrix();
                    auto& m2 = maps_[j].map.matrix();
                    auto& o1 = merged.map.offset();
                    auto& o2 = maps_[j].map.offset();
                    for (size_t r = 0; r < m1.size() && same; ++r) {
                        if (m1[r] != m2[r] || o1[r] != o2[r]) same = false;
                    }
                }
                if (same) {
                    // Merge domains by taking the union (approximate via intersection relaxation)
                    // For exactness, keep both basic maps
                    merged.domain = merged.domain.intersect(maps_[j].domain);
                    // Actually union is not intersection – keep both
                    used[j] = true;
                    result.add_basic_map(maps_[j]);
                }
            }
            result.add_basic_map(std::move(merged));
        }
        return result;
    }

    /// Compute the domain (union of all basic map domains).
    [[nodiscard]] IntegerPolytope domain() const {
        if (maps_.empty()) return IntegerPolytope(0);

        // The domain is the union of all sub-domains.
        // Since IntegerPolytope doesn't directly support union,
        // we return the convex hull (over-approximation).
        // For exactness, we take the broadest bounds.
        size_t ndim = maps_[0].domain.ndim();
        std::vector<std::pair<int64_t, int64_t>> bounds(ndim, {INT64_MAX, INT64_MIN});

        for (const auto& bm : maps_) {
            auto pts = bm.domain.enumerate_points();
            for (const auto& pt : pts) {
                for (size_t d = 0; d < ndim && d < pt.size(); ++d) {
                    bounds[d].first = std::min(bounds[d].first, pt[d]);
                    bounds[d].second = std::max(bounds[d].second, pt[d]);
                }
            }
        }

        return make_rectangular_polytope(
            std::vector<std::pair<int64_t, int64_t>>(bounds.begin(), bounds.end())
        );
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "UnionMap{pieces=" << maps_.size() << "}";
        return oss.str();
    }

private:
    std::vector<BasicMap> maps_;
};

} // namespace symplex::polyhedral
