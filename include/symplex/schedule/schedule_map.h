// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/schedule_tree.h"
#include "symplex/schedule/tiling.h"
#include "symplex/schedule/fusion.h"
#include "symplex/schedule/parallelization.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <string>
#include <optional>
#include <memory>

namespace symplex::schedule {

/// ScheduleMap: the core mapping Φ that assigns a time-stamp and
/// hardware location to every point in the iteration space.
///
/// The ScheduleMap wraps a ScheduleTree with the additional metadata
/// needed to generate executable code:
///   - The iteration space it was derived from
///   - The GPU mapping (grid/block dims, warp roles, smem)
///   - The tile configuration used
///
/// The schedule map is the main artifact of the polyhedral scheduling
/// pipeline.  It is constructed by `build_schedule_map` and can be
/// further refined by `map_to_hardware`.
class ScheduleMap {
public:
    ScheduleMap() = default;

    /// Construct from components.
    ScheduleMap(
        ScheduleTreePtr tree,
        polyhedral::IterationSpace iterspace,
        GPUMapping gpu_mapping,
        TileConfig tile_config
    )
        : tree_(std::move(tree))
        , iterspace_(std::move(iterspace))
        , gpu_mapping_(std::move(gpu_mapping))
        , tile_config_(std::move(tile_config))
    {}

    // ── Accessors ───────────────────────────────────────────────────

    [[nodiscard]] const ScheduleTreePtr& tree() const { return tree_; }
    [[nodiscard]] ScheduleTreePtr& tree() { return tree_; }

    [[nodiscard]] const polyhedral::IterationSpace& iterspace() const { return iterspace_; }
    [[nodiscard]] polyhedral::IterationSpace& iterspace() { return iterspace_; }

    [[nodiscard]] const GPUMapping& gpu_mapping() const { return gpu_mapping_; }
    [[nodiscard]] GPUMapping& gpu_mapping() { return gpu_mapping_; }

    [[nodiscard]] const TileConfig& tile_config() const { return tile_config_; }
    [[nodiscard]] TileConfig& tile_config() { return tile_config_; }

    // ── Operations ──────────────────────────────────────────────────

    /// Apply the full scheduling pipeline to the schedule tree:
    ///   1. Compute dependencies (if not already computed)
    ///   2. Apply tiling
    ///   3. Apply fusion
    ///   4. Mark parallel dimensions
    ///   5. Map to GPU hardware
    ///
    /// \param target   The hardware target
    /// \return         true if the pipeline succeeded
    bool apply(const hardware::HardwareTarget& target);

    /// Dump a human-readable representation of the schedule map.
    [[nodiscard]] std::string dump() const;

    /// Validate that the schedule map is internally consistent:
    ///   - The tree is non-null
    ///   - All dependencies are satisfied by the schedule
    ///   - GPU mapping respects hardware constraints
    ///   - Tile configuration fits in SRAM
    ///
    /// \param target   The hardware target to validate against
    /// \return         true if the schedule map is valid
    bool validate(const hardware::HardwareTarget& target) const;

private:
    ScheduleTreePtr               tree_;
    polyhedral::IterationSpace    iterspace_;
    GPUMapping                    gpu_mapping_;
    TileConfig                    tile_config_;
};

/// build_schedule_map: the main entry point for constructing a schedule
/// from an iteration space.  This implements the full pipeline:
///
///   1. Create a schedule tree from the iteration space
///   2. Compute all data dependencies (RAW, WAR, WAW)
///   3. Apply hierarchical tiling aligned to Tensor Core
///   4. Apply operator fusion (greedy)
///   5. Mark parallel dimensions
///   6. Map to GPU hardware
///
/// \param iterspace   The iteration space to schedule
/// \param target      The hardware target
/// \return            The constructed schedule map
ScheduleMap build_schedule_map(
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
);

/// map_to_hardware: take an existing schedule tree and hardware target
/// and produce the GPU mapping.  This is a lower-level entry point
/// used when the schedule tree has been manually constructed or
/// modified.
///
/// \param tree        The schedule tree
/// \param iterspace   The iteration space (for dependency info)
/// \param target      The hardware target
/// \return            The GPU mapping
GPUMapping map_to_hardware(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
);

/// build_initial_tree: construct the initial schedule tree from an
/// iteration space.  Creates a DOMAIN node with nested BAND and
/// LEAF nodes for each statement.
///
/// \param iterspace   The iteration space
/// \return            The initial schedule tree
ScheduleTreePtr build_initial_tree(
    const polyhedral::IterationSpace& iterspace
);

/// select_tile_config: choose a tile configuration that aligns with
/// the hardware's Tensor Core dimensions and fits within SRAM.
///
/// \param iterspace   The iteration space
/// \param target      The hardware target
/// \return            The selected tile configuration, or nullopt
///                     if no valid configuration exists
std::optional<TileConfig> select_tile_config(
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
);

} // namespace symplex::schedule
