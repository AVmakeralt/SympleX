// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "symplex/distributed/mesh.h"

namespace symplex::fault_tolerance {

/// CommunicatorGroup: a set of devices that share an NCCL communicator
/// (e.g. all devices in one row or column of the mesh).
struct CommunicatorGroup {
    int64_t group_id;
    std::vector<int64_t> device_ids;
    bool active;
};

/// CommunicatorRepair: in-place repair of NCCL/collective communicator
/// groups after device failures.  Removes dead devices from groups,
/// rebuilds group membership, and can emit rerouted collective
/// operation code that avoids dead devices.
class CommunicatorRepair {
public:
    explicit CommunicatorRepair(distributed::ClusterMesh& mesh);

    // ── Repair ────────────────────────────────────────────────────────

    /// Rebuild communicator groups after the given devices have failed.
    /// Dead devices are removed from their groups; groups that become
    /// empty are deactivated.
    bool repair_after_failure(const std::vector<int64_t>& failed_devices);

    // ── Queries ───────────────────────────────────────────────────────

    /// Return the active communicator group containing `device_id`,
    /// or nullptr if the device is not in any active group.
    const CommunicatorGroup* group_for_device(int64_t device_id) const;

    /// Return pointers to all active communicator groups.
    std::vector<const CommunicatorGroup*> active_groups() const;

    // ── Initialization ────────────────────────────────────────────────

    /// Build communicator groups based on the current mesh topology.
    /// For each mesh dimension, a group is created for each "row" or
    /// "column" of devices along that dimension.
    void initialize_groups();

    // ── Rerouting ─────────────────────────────────────────────────────

    /// Emit a code string representing a rerouted collective operation
    /// that avoids dead devices between `src_device` and `dst_device`.
    /// `original_op` is the name of the original NCCL operation
    /// (e.g. "ncclAllReduce").
    std::string emit_rerouted_collective(
        const std::string& original_op,
        int64_t src_device,
        int64_t dst_device
    ) const;

private:
    distributed::ClusterMesh& mesh_;
    std::vector<CommunicatorGroup> groups_;

    /// Build communicator groups along a single mesh dimension.
    void build_groups_along_dim(size_t mesh_dim);

    /// Find an alternative path (list of intermediate device IDs)
    /// from `src` to `dst` that avoids all devices in `exclude`.
    /// Uses BFS over the mesh adjacency graph.
    std::vector<int64_t> find_alternative_path(
        int64_t src, int64_t dst,
        const std::vector<int64_t>& exclude
    ) const;
};

} // namespace symplex::fault_tolerance
