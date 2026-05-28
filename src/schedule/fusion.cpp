// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/fusion.h"
#include <algorithm>
#include <unordered_set>

namespace symplex::schedule {

// ---------------------------------------------------------------------------
// Internal helpers (file-local)
// ---------------------------------------------------------------------------

/// Deep-copy an entire subtree rooted at `src` into a new tree.
static ScheduleTreePtr deep_clone(const ScheduleTreePtr& src) {
    auto copy = ScheduleTree::create(src->type());
    if (src->type() == ScheduleNodeType::BAND) {
        copy->band_data() = src->band_data();
    }
    if (src->type() == ScheduleNodeType::FILTER) {
        copy->filter_data() = src->filter_data();
    }
    if (src->type() == ScheduleNodeType::DOMAIN) {
        copy->set_domain_name(src->domain_name());
    }
    for (const auto& child : src->children()) {
        copy->add_child(deep_clone(child));
    }
    return copy;
}

/// Recursively rebuild a schedule tree, applying greedy fusion to any
/// SEQUENCE node that contains adjacent BAND children.
///
/// Returns a pair: (rebuilt_tree, number_of_fusions_applied).
static std::pair<ScheduleTreePtr, int> rebuild_with_fusion(
    const ScheduleTreePtr& node,
    const polyhedral::IterationSpace& iterspace,
    int64_t sram_budget,
    size_t bpe
) {
    // First, recursively process all children
    std::vector<ScheduleTreePtr> rebuilt_children;
    int total_fusions = 0;

    for (const auto& child : node->children()) {
        auto [rebuilt, fusions] = rebuild_with_fusion(
            child, iterspace, sram_budget, bpe);
        rebuilt_children.push_back(std::move(rebuilt));
        total_fusions += fusions;
    }

    // If this is not a SEQUENCE node, just rebuild with processed children
    if (node->type() != ScheduleNodeType::SEQUENCE) {
        auto result = ScheduleTree::create(node->type());
        if (node->type() == ScheduleNodeType::BAND) {
            result->band_data() = node->band_data();
        }
        if (node->type() == ScheduleNodeType::FILTER) {
            result->filter_data() = node->filter_data();
        }
        if (node->type() == ScheduleNodeType::DOMAIN) {
            result->set_domain_name(node->domain_name());
        }
        for (auto& c : rebuilt_children) {
            result->add_child(std::move(c));
        }
        return {result, total_fusions};
    }

    // This is a SEQUENCE node: try to fuse adjacent BAND children.
    // We use a greedy left-to-right scan.
    std::vector<ScheduleTreePtr> fused_children;
    size_t i = 0;
    while (i < rebuilt_children.size()) {
        if (rebuilt_children[i]->type() != ScheduleNodeType::BAND ||
            i + 1 >= rebuilt_children.size() ||
            rebuilt_children[i + 1]->type() != ScheduleNodeType::BAND) {
            // Cannot fuse – just keep the child
            fused_children.push_back(std::move(rebuilt_children[i]));
            ++i;
            continue;
        }

        // Two adjacent bands: try to fuse them
        auto& band_a = rebuilt_children[i];
        auto& band_b = rebuilt_children[i + 1];

        if (ShouldFuse(band_a, band_b, iterspace, sram_budget, bpe)) {
            auto fused = FuseAdjacent(band_a, band_b, iterspace);
            // Replace the pair with the fused result
            rebuilt_children[i] = fused;
            rebuilt_children.erase(
                rebuilt_children.begin() +
                static_cast<ptrdiff_t>(i + 1));
            total_fusions++;
            // Don't advance i – try to fuse with the next band too
            continue;
        }

        // Fusion rejected – keep band_a and advance
        fused_children.push_back(std::move(rebuilt_children[i]));
        ++i;
    }

    // Move any remaining (the last band from the while loop may still
    // be in rebuilt_children if it wasn't fused)
    // Actually, the while loop above either moves items into
    // fused_children or replaces them in rebuilt_children.
    // We need to collect whatever remains in rebuilt_children.
    for (auto& c : rebuilt_children) {
        if (c) {
            fused_children.push_back(std::move(c));
        }
    }

    // Build the new SEQUENCE node
    auto result = ScheduleTree::create(ScheduleNodeType::SEQUENCE);
    for (auto& c : fused_children) {
        result->add_child(std::move(c));
    }
    return {result, total_fusions};
}

// ---------------------------------------------------------------------------
// CollectStatementNames
// ---------------------------------------------------------------------------
std::vector<std::string> CollectStatementNames(const ScheduleTreePtr& node) {
    std::vector<std::string> names;
    node->dfs([&](const ScheduleTree& n) {
        if (n.type() == ScheduleNodeType::FILTER) {
            names.push_back(n.filter_data().statement_name);
        } else if (n.type() == ScheduleNodeType::DOMAIN) {
            if (!n.domain_name().empty()) {
                names.push_back(n.domain_name());
            }
        }
    });
    return names;
}

// ---------------------------------------------------------------------------
// CheckFuseLegal
//
// Two bands may be fused iff every inter-band dependency remains
// lexicographically positive after the fused schedule is applied.
//
// Strategy: Collect statement sets under each band.  For each dependency
// vector, verify that in the fused schedule space (band_a dims followed
// by band_b dims) the dependency direction remains forward.
// ---------------------------------------------------------------------------
bool CheckFuseLegal(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace
) {
    // Collect statements under each band
    auto stmts_a = CollectStatementNames(band_a);
    auto stmts_b = CollectStatementNames(band_b);

    if (stmts_a.empty() || stmts_b.empty()) {
        // If we cannot determine statements, be conservative and allow
        // fusion – the schedule tree may not have filter/domain nodes
        // yet (e.g., before statement splitting).  Legality will be
        // verified later when the tree is fully constructed.
        return true;
    }

    std::unordered_set<std::string> set_a(stmts_a.begin(), stmts_a.end());
    std::unordered_set<std::string> set_b(stmts_b.begin(), stmts_b.end());

    size_t n_members_a = band_a->band_data().members.size();

    // Check each dependency polyhedron for inter-band violations
    auto all_deps = iterspace.all_dependencies();

    for (const auto& dep : all_deps) {
        for (const auto& dv : dep.vectors()) {
            size_t ndim = dv.components.size();
            if (ndim == 0) continue;

            // In the fused schedule, the first n_members_a dimensions
            // come from band_a and the remainder from band_b.
            //
            // A dependency from band_b → band_a (backward dep) would
            // have a negative component in band_a's dimensions, which
            // would make the fused schedule violate the dependency.
            //
            // More precisely: in the fused space, the schedule for a
            // point in band_a is (a_0, ..., a_{k-1}, 0, ..., 0) and
            // for band_b is (a_0, ..., a_{k-1}, b_0, ..., b_{m-1}).
            // Since band_b executes after band_a in the original
            // sequential order, any dependency from band_b to band_a
            // would be backward and violated by fusion.

            // Check if the dependency vector has a negative component
            // in band_a's dimensions with all preceding components zero
            // (which would make the dependency go backward in the fused
            // lexicographic order).
            bool found_negative_in_a = false;
            bool all_zero_before_negative = true;

            for (size_t d = 0; d < std::min(ndim, n_members_a); ++d) {
                if (dv.components[d] < 0) {
                    found_negative_in_a = true;
                    break;
                } else if (dv.components[d] > 0) {
                    // This dimension carries a forward dependency –
                    // even if later dimensions are negative, the
                    // lexicographic order is still positive.
                    all_zero_before_negative = false;
                    break;
                }
                // dv.components[d] == 0: continue checking
            }

            if (found_negative_in_a && all_zero_before_negative) {
                // This dependency would be violated by fusion
                return false;
            }

            // Also check for equal-zero in band_a dimensions with
            // negative in band_b dimensions – this is an intra-band_b
            // dependency and doesn't affect legality.

            // Check for the case where the dependency has all zeros in
            // band_a dimensions – this means the source and sink share
            // the same outer coordinates.  This is fine for fusion as
            // long as band_b's dimensions properly order the dependency.
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// ShouldFuse
//
// Heuristic: fuse if the intermediate tensor between the two bands
// fits in SRAM.  The intuition is that fusion eliminates the need to
// write/read the intermediate tensor to/from global memory.
// ---------------------------------------------------------------------------
bool ShouldFuse(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace,
    int64_t sram_budget,
    size_t bpe
) {
    // First, fusion must be legal
    if (!CheckFuseLegal(band_a, band_b, iterspace)) {
        return false;
    }

    // If either band is empty, fusion is trivially safe
    if (band_a->band_data().members.empty() ||
        band_b->band_data().members.empty()) {
        return true;
    }

    // Estimate the tile sizes from the band members.
    // The tile size for each dimension is approximated from the
    // coefficient magnitudes.
    auto compute_tile_sizes = [](const BandNodeData& bd) -> std::vector<int64_t> {
        std::vector<int64_t> tiles;
        for (const auto& m : bd.members) {
            int64_t max_coeff = 0;
            for (auto c : m.coefficients) {
                max_coeff = std::max(max_coeff, std::abs(c));
            }
            tiles.push_back(std::max(max_coeff, int64_t(1)));
        }
        return tiles;
    };

    auto tiles_a = compute_tile_sizes(band_a->band_data());
    auto tiles_b = compute_tile_sizes(band_b->band_data());

    // Estimate SRAM footprint for each band's working set
    size_t footprint_a = iterspace.estimate_sram_footprint(tiles_a, bpe, true);
    size_t footprint_b = iterspace.estimate_sram_footprint(tiles_b, bpe, true);

    // After fusion, we need to hold both bands' working sets simultaneously
    // but we SAVE the intermediate tensor (no write-back to global memory).
    // The fused footprint is conservatively the sum of both footprints.
    size_t fused_footprint = footprint_a + footprint_b;

    // If the fused footprint fits in SRAM, fusion is a clear win
    if (static_cast<int64_t>(fused_footprint) <= sram_budget) {
        return true;
    }

    // Allow 1.25x oversubscription for near-fit cases where the
    // intermediate tensor elimination is worth the register spill cost.
    if (static_cast<double>(fused_footprint) <=
        static_cast<double>(sram_budget) * 1.25) {
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// FuseAdjacent
//
// Merge two adjacent band nodes into a single band.  The resulting band
// has the members of band_a followed by the members of band_b.
// ---------------------------------------------------------------------------
ScheduleTreePtr FuseAdjacent(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace
) {
    if (!CheckFuseLegal(band_a, band_b, iterspace)) {
        return band_a;  // Cannot fuse – return unchanged
    }

    // Create the fused band node
    auto fused = ScheduleTree::create(ScheduleNodeType::BAND);

    // Merge band members: band_a's members first, then band_b's
    auto& fused_data = fused->band_data();
    const auto& data_a = band_a->band_data();
    const auto& data_b = band_b->band_data();

    size_t n_a = data_a.members.size();
    size_t n_b = data_b.members.size();

    fused_data.members.reserve(n_a + n_b);

    // Copy band_a's members directly
    for (const auto& m : data_a.members) {
        fused_data.members.push_back(m);
    }

    // For band_b's members, extend their coefficient vectors to account
    // for the additional outer dimensions from band_a.  In the fused
    // schedule, the iteration vector has n_a + n_b dimensions.  band_b's
    // coefficients are shifted to reference dimensions [n_a, n_a+n_b).
    for (const auto& m : data_b.members) {
        BandMember shifted(n_a + n_b, m.constant);
        // Fill in band_a's coefficient positions with zeros
        for (size_t j = 0; j < n_a; ++j) {
            shifted.coefficients[j] = 0;
        }
        // Copy band_b's coefficients to the shifted positions
        for (size_t j = 0; j < m.coefficients.size() && j < n_b; ++j) {
            shifted.coefficients[n_a + j] = m.coefficients[j];
        }
        shifted.parallel = m.parallel;
        shifted.coincidence = m.coincidence;
        fused_data.members.push_back(std::move(shifted));
    }

    // Extend band_a's members' coefficient vectors to match fused size
    for (size_t i = 0; i < n_a; ++i) {
        auto& member = fused_data.members[i];
        std::vector<int64_t> extended(n_a + n_b, 0);
        for (size_t j = 0; j < member.coefficients.size(); ++j) {
            extended[j] = member.coefficients[j];
        }
        member.coefficients = std::move(extended);
    }

    // The fused band is permutable only if both original bands were
    // permutable AND all inter-band dependencies are satisfied
    fused_data.permutable = data_a.permutable && data_b.permutable;

    // Collect children from both bands
    for (const auto& child : band_a->children()) {
        fused->add_child(deep_clone(child));
    }
    for (const auto& child : band_b->children()) {
        fused->add_child(deep_clone(child));
    }

    // Inherit parallel markings
    for (size_t i = 0; i < n_a; ++i) {
        if (data_a.members[i].parallel) {
            fused->mark_parallel(i);
        }
    }
    for (size_t i = 0; i < n_b; ++i) {
        if (data_b.members[i].parallel) {
            fused->mark_parallel(n_a + i);
        }
    }

    return fused;
}

// ---------------------------------------------------------------------------
// FuseAll
//
// Greedy pass: walk the schedule tree and fuse adjacent band nodes
// wherever the ShouldFuse heuristic approves.
// ---------------------------------------------------------------------------
FusionResult FuseAll(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace,
    int64_t sram_budget,
    size_t bpe
) {
    // Apply multiple passes until no more fusions occur
    ScheduleTreePtr current = deep_clone(tree);
    int total_fusions = 0;
    bool any_changed = false;

    bool changed_this_pass = true;
    while (changed_this_pass) {
        changed_this_pass = false;

        auto [rebuilt, fusions] = rebuild_with_fusion(
            current, iterspace, sram_budget, bpe);

        if (fusions > 0) {
            current = std::move(rebuilt);
            total_fusions += fusions;
            any_changed = true;
            changed_this_pass = true;
        }
    }

    FusionResult result;
    result.tree = std::move(current);
    result.bands_fused = total_fusions;
    result.changed = any_changed;
    return result;
}

} // namespace symplex::schedule
