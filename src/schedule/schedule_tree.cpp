// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/schedule_tree.h"
#include "symplex/schedule/schedule_builder.h"
#include "symplex/polyhedral/iteration_space.h"

#include <algorithm>
#include <numeric>
#include <cassert>
#include <sstream>
#include <queue>

namespace symplex::schedule {

// ── Free functions for schedule tree construction and manipulation ────────────
// These complement the inline methods in the header with more complex
// algorithms that require pulling in the IterationSpace dependency.

/// Build a schedule tree from an iteration space.
/// Creates:
///   DOMAIN(name)
///     └─ BAND(identity schedule, n_loops=ndim)
///          └─ [SEQUENCE if multiple statements]
///               └─ FILTER(stmt_name) × num_statements
///                    └─ LEAF
ScheduleTreePtr build_from_iteration_space(
    const polyhedral::IterationSpace& ispace
) {
    auto root = ScheduleTree::create(ScheduleNodeType::DOMAIN);
    root->set_domain_name(ispace.name());

    if (ispace.num_statements() == 0) {
        root->add_child(ScheduleNodeType::LEAF);
        return root;
    }

    size_t ndim = ispace.statement(0).domain.ndim();

    // Create a band node representing the identity schedule.
    auto band = root->add_child(ScheduleNodeType::BAND);
    band->band_data().members.resize(ndim);
    band->band_data().permutable = true;

    for (size_t d = 0; d < ndim; ++d) {
        auto& member = band->band_data().members[d];
        member.coefficients.resize(ndim, 0);
        member.coefficients[d] = 1;  // Identity: schedule(i) = i_d
        member.constant = 0;
        member.parallel = false;      // Will be determined by dependency analysis
        member.coincidence = false;
    }

    // If only one statement, add a single leaf under the band.
    if (ispace.num_statements() == 1) {
        band->add_child(ScheduleNodeType::LEAF);
    } else {
        // Multiple statements: SEQUENCE of FILTER + LEAF.
        auto seq = band->add_child(ScheduleNodeType::SEQUENCE);
        for (size_t s = 0; s < ispace.num_statements(); ++s) {
            auto filter = seq->add_child(ScheduleNodeType::FILTER);
            filter->filter_data().statement_name = ispace.statement(s).name;

            auto bounds = ispace.statement(s).domain.bounds();
            filter->filter_data().lower_bounds.resize(bounds.size());
            filter->filter_data().upper_bounds.resize(bounds.size());
            for (size_t d = 0; d < bounds.size(); ++d) {
                filter->filter_data().lower_bounds[d] = bounds[d].first;
                filter->filter_data().upper_bounds[d] = bounds[d].second;
            }

            filter->add_child(ScheduleNodeType::LEAF);
        }
    }

    return root;
}

/// Apply tiling to a band node in the schedule tree.
/// Splits each band member into an outer tile loop and an inner
/// element loop. Returns the inner band node.
///
/// Before:
///   BAND[m0, m1, m2, ...]
///     └─ children
///
/// After:
///   BAND[m0_tiled, m1_tiled, m2_tiled, ...]   (outer tile loops)
///     └─ BAND[m0, m1, m2, ...]                 (inner element loops)
///          └─ children
ScheduleTreePtr apply_tiling(
    const ScheduleTreePtr& band_node,
    const std::vector<int64_t>& tile_sizes
) {
    if (!band_node || band_node->type() != ScheduleNodeType::BAND) {
        return nullptr;
    }

    size_t n = band_node->band_data().members.size();
    if (tile_sizes.size() != n) return nullptr;

    // Use the built-in tile_band method.
    auto inner_band = band_node->tile_band(tile_sizes);

    // Update the outer band coefficients to reflect tile strides.
    // The outer loop should step by tile_size[i] in dimension i.
    for (size_t i = 0; i < n; ++i) {
        auto& outer_member = band_node->band_data().members[i];

        // The outer loop iterates over tile indices.
        // The tile_band method already sets up the outer/inner split.
        // Mark outer loops as potentially parallel if they iterate
        // over independent tiles.
        if (inner_band->band_data().members[i].parallel) {
            outer_member.parallel = true;
        }
    }

    // Inner band: the element loop iterates within a tile.
    // Mark inner dimensions based on the original parallelism.
    for (size_t i = 0; i < n; ++i) {
        // The innermost dimension of a tile is often sequential
        // (carrying dependencies), while outer tile dims can be parallel.
        if (inner_band->band_data().members[i].parallel) {
            // Keep parallel marking from original
        }
    }

    return inner_band;
}

/// Mark parallelism in band nodes based on dependency analysis.
/// For each band member, check if any dependency vector has a non-zero
/// component in that dimension. If not, the dimension is parallel.
///
/// Returns the total number of dimensions marked as parallel.
int mark_parallelism(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& ispace
) {
    if (!tree) return 0;

    int count = 0;

    // Find all band nodes.
    auto bands = tree->find_nodes(ScheduleNodeType::BAND);

    for (auto& band : bands) {
        size_t n_members = band->band_data().members.size();
        for (size_t m = 0; m < n_members; ++m) {
            // A band member is parallel if dimension m doesn't carry
            // any dependency.
            bool carries_dep = false;

            // Check all dependency types.
            for (const auto& dep : ispace.raw_deps()) {
                if (!dep.is_parallelizable(m)) {
                    carries_dep = true;
                    break;
                }
            }
            if (!carries_dep) {
                for (const auto& dep : ispace.war_deps()) {
                    if (!dep.is_parallelizable(m)) {
                        carries_dep = true;
                        break;
                    }
                }
            }
            if (!carries_dep) {
                for (const auto& dep : ispace.waw_deps()) {
                    if (!dep.is_parallelizable(m)) {
                        carries_dep = true;
                        break;
                    }
                }
            }

            if (carries_dep) {
                band->mark_sequential(m);
            } else {
                band->mark_parallel(m);
                ++count;
            }
        }
    }

    return count;
}

/// Compute concrete loop bounds from schedule maps.
/// Given a schedule tree and iteration space, extracts the lower and
/// upper bounds for each loop in the tree.
///
/// Returns a vector of (lower, upper) pairs, one per loop dimension
/// in the schedule tree.
std::vector<std::pair<int64_t, int64_t>> compute_loop_bounds(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& ispace
) {
    std::vector<std::pair<int64_t, int64_t>> bounds;

    if (!tree || ispace.num_statements() == 0) return bounds;

    // Walk the tree to find band nodes and compute bounds.
    // For each band member, the loop bounds come from the
    // iteration domain projected through the schedule.
    auto bands = tree->find_nodes(ScheduleNodeType::BAND);

    for (const auto& band : bands) {
        const auto& members = band->band_data().members;
        size_t ndim = ispace.statement(0).domain.ndim();
        auto domain_bounds = ispace.statement(0).domain.bounds();

        for (const auto& member : members) {
            // The schedule for this member is:
            //   phi(i) = sum_j coeff[j] * i_j + constant
            // The loop bounds are the min and max of phi over the domain.

            int64_t lo = member.constant;
            int64_t hi = member.constant;

            for (size_t j = 0; j < member.coefficients.size() && j < ndim; ++j) {
                int64_t coeff = member.coefficients[j];
                if (coeff == 0) continue;

                auto [d_lo, d_hi] = domain_bounds[j];
                if (coeff > 0) {
                    lo += coeff * d_lo;
                    hi += coeff * d_hi;
                } else {
                    lo += coeff * d_hi;
                    hi += coeff * d_lo;
                }
            }

            bounds.emplace_back(lo, hi);
        }
    }

    return bounds;
}

/// Compute the depth of the schedule tree (maximum path length from
/// root to any leaf).
int tree_depth(const ScheduleTree& tree) {
    if (tree.children().empty()) return 0;
    int max_depth = 0;
    for (const auto& child : tree.children()) {
        max_depth = std::max(max_depth, tree_depth(*child));
    }
    return max_depth + 1;
}

/// Count the total number of nodes in the schedule tree.
size_t tree_size(const ScheduleTree& tree) {
    size_t count = 1;
    for (const auto& child : tree.children()) {
        count += tree_size(*child);
    }
    return count;
}

/// Find the first band node in the tree (BFS order).
ScheduleTreePtr find_first_band(const ScheduleTreePtr& tree) {
    if (!tree) return nullptr;

    std::queue<ScheduleTreePtr> queue;
    queue.push(tree);

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();

        if (node->type() == ScheduleNodeType::BAND) {
            return node;
        }

        for (const auto& child : node->children()) {
            queue.push(child);
        }
    }

    return nullptr;
}

/// Collect all leaf statement names from the schedule tree.
std::vector<std::string> collect_statement_names(const ScheduleTreePtr& tree) {
    std::vector<std::string> names;

    if (!tree) return names;

    tree->dfs([&](const ScheduleTree& node) {
        if (node.type() == ScheduleNodeType::FILTER) {
            names.push_back(node.filter_data().statement_name);
        }
    });

    return names;
}

} // namespace symplex::schedule
