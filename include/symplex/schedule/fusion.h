// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/schedule_tree.h"
#include "symplex/polyhedral/iteration_space.h"
#include <vector>
#include <string>
#include <optional>

namespace symplex::schedule {

/// FusionResult: outcome of a fusion decision.
struct FusionResult {
    ScheduleTreePtr tree;           // The schedule tree after fusion
    int             bands_fused;    // Number of band pairs fused
    bool            changed;        // Did any fusion occur?

    FusionResult()
        : bands_fused(0), changed(false) {}

    FusionResult(ScheduleTreePtr t, int n, bool c)
        : tree(std::move(t)), bands_fused(n), changed(c) {}
};

/// CheckFuseLegal: verify that fusing two adjacent band nodes does not
/// violate any data dependencies in the iteration space.
///
/// Two bands can be legally fused if and only if every dependency vector
/// that crosses from the first band's statements to the second band's
/// statements remains lexicographically positive after fusion (i.e., the
/// fused schedule preserves the execution order required by dependencies).
///
/// \param band_a       First (earlier) band node
/// \param band_b       Second (later) band node
/// \param iterspace    The iteration space with pre-computed dependencies
/// \return             true if fusing is legal, false otherwise
bool CheckFuseLegal(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace
);

/// ShouldFuse: heuristic decision on whether to fuse two adjacent bands.
///
/// The heuristic considers:
///   1. Whether the intermediate tensor (produced by band_a, consumed by
///      band_b) fits in SRAM after fusion.  If it does, fusion eliminates
///      global memory traffic for that tensor – a clear win.
///   2. Whether fusion would increase parallelism (more parallel loop
///      dimensions in the fused band).
///   3. Whether the fused loop body would exceed register pressure limits.
///
/// \param band_a       First (earlier) band node
/// \param band_b       Second (later) band node
/// \param iterspace    The iteration space with dependency information
/// \param sram_budget  Maximum SRAM bytes available (e.g., shared memory)
/// \param bpe          Bytes per element (default 2 for FP16)
/// \return             true if the heuristic recommends fusion
bool ShouldFuse(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace,
    int64_t sram_budget,
    size_t bpe = 2
);

/// FuseAdjacent: attempt to merge two adjacent band nodes into a single
/// band node if no dependencies prevent it.
///
/// The fused band has members from both bands concatenated.  The outer
/// dimensions of band_a become the outer dimensions of the fused band;
/// the dimensions of band_b are appended as inner dimensions.  The
/// permutable flag of the fused band is set to true only if both original
/// bands were permutable and all fused dimensions are independent.
///
/// \param band_a       First (earlier) band node (will be modified in-place)
/// \param band_b       Second (later) band node
/// \param iterspace    The iteration space with dependency information
/// \return             The fused band node if fusion succeeded, or band_a
///                     unchanged if fusion was illegal
ScheduleTreePtr FuseAdjacent(
    const ScheduleTreePtr& band_a,
    const ScheduleTreePtr& band_b,
    const polyhedral::IterationSpace& iterspace
);

/// FuseAll: greedily fuse all possible adjacent bands in the schedule
/// tree.  Walks the tree looking for SEQUENCE nodes whose children are
/// BAND nodes and attempts to fuse each adjacent pair.
///
/// \param tree         The schedule tree to transform
/// \param iterspace    The iteration space with dependency information
/// \param sram_budget  SRAM budget for the ShouldFuse heuristic
/// \param bpe          Bytes per element (default 2 for FP16)
/// \return             The result of the greedy fusion pass
FusionResult FuseAll(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace,
    int64_t sram_budget,
    size_t bpe = 2
);

/// Collect the statement names reachable from a subtree.
/// Used internally to determine which dependencies cross band boundaries.
std::vector<std::string> CollectStatementNames(const ScheduleTreePtr& node);

} // namespace symplex::schedule
