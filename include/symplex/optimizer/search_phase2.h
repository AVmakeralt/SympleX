// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/optimizer/tile_config.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>

namespace symplex::optimizer {

/// Phase 2: Compute-Symmetry Alignment
///
/// Takes the pruned candidates from Phase 1 and further filters them by
/// ensuring tile dimensions align with Tensor Core fragment sizes.
/// Each surviving candidate is scored on three sub-metrics:
///
///   1. SRAM Utilization %     – how much of the available shared memory
///                               the tile uses (higher is better, up to
///                               the hardware limit).
///   2. Tensor Core Utilization % – how well the inner tile maps to
///                               native MMA fragment dimensions.
///   3. Memory Coalescing       – whether the tile dimensions are
///                               multiples of the memory transaction
///                               width (128 bytes / bytes_per_element).
///
/// Candidates are sorted by composite score in descending order.
///
/// \param candidates  Pruned configs from phase 1 (moved in).
/// \param target      Hardware target specification.
/// \return            Scored and sorted configs that pass alignment.
std::vector<ExtendedTileConfig> phase2_symmetry_alignment(
    std::vector<ExtendedTileConfig> candidates,
    const hardware::HardwareTarget& target
);

} // namespace symplex::optimizer
