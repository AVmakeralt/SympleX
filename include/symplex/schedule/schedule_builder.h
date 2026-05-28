// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/schedule_tree.h"
#include "symplex/polyhedral/iteration_space.h"
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

namespace symplex::schedule {

/// Build a schedule tree from an iteration space.
/// Creates: DOMAIN → BAND(identity) → [SEQUENCE → FILTER+LEAF | LEAF]
ScheduleTreePtr build_from_iteration_space(
    const polyhedral::IterationSpace& ispace
);

/// Apply tiling to a band node. Returns the inner band node.
ScheduleTreePtr apply_tiling(
    const ScheduleTreePtr& band_node,
    const std::vector<int64_t>& tile_sizes
);

/// Mark parallelism in band nodes based on dependency analysis.
/// Returns the total number of dimensions marked as parallel.
int mark_parallelism(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& ispace
);

/// Compute concrete loop bounds from schedule maps.
/// Returns a vector of (lower, upper) pairs.
std::vector<std::pair<int64_t, int64_t>> compute_loop_bounds(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& ispace
);

} // namespace symplex::schedule
