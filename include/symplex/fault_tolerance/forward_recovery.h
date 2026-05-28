// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "symplex/distributed/mesh.h"

namespace symplex::fault_tolerance {

/// RecoveryPlan: the result of a forward-recovery analysis, describing
/// which devices have failed, which survive, how work is redistributed,
/// and whether the mesh topology must be rebuilt.
struct RecoveryPlan {
    std::vector<int64_t> failed_devices;
    std::vector<int64_t> surviving_devices;
    std::vector<int64_t> redistributed_work;  // New work units per surviving device
    bool needs_remesh;                         // Does the mesh topology need rebuilding?
    int64_t estimated_recovery_ns;             // Time to complete recovery
    bool success;
    std::string error_message;
};

/// ForwardRecovery: resilient forward recovery mechanism that handles
/// node failures without stopping training.  When devices die, their
/// work is redistributed across surviving devices and the mesh topology
/// is repaired in-place.
class ForwardRecovery {
public:
    explicit ForwardRecovery(distributed::ClusterMesh& mesh);

    // ── Main entry point ──────────────────────────────────────────────

    /// Called when one or more device failures are detected.
    /// Marks dead devices, redistributes work, and optionally
    /// reconstructs the mesh.
    RecoveryPlan recover(const std::vector<int64_t>& failed_devices);

    // ── Work redistribution ───────────────────────────────────────────

    /// Redistribute `total_work_units` proportionally across the
    /// surviving devices, taking over the workload of the failed ones.
    RecoveryPlan redistribute_workload(
        const std::vector<int64_t>& failed,
        const std::vector<int64_t>& survivors,
        int64_t total_work_units
    );

    // ── Mesh reconstruction ───────────────────────────────────────────

    /// Reconstruct the network mesh to bypass dead nodes.
    /// Marks failed devices as dead in the mesh so that subsequent
    /// operations skip them.  Returns true if reconstruction succeeded.
    bool reconstruct_mesh(const std::vector<int64_t>& failed_devices);

    // ── Feasibility ───────────────────────────────────────────────────

    /// Check whether recovery is possible given `num_failed` devices.
    bool can_recover(int64_t num_failed) const;

    /// Minimum number of devices required for training.
    int64_t min_devices() const;

    /// Set the minimum number of devices required for training.
    void set_min_devices(int64_t n);

private:
    distributed::ClusterMesh& mesh_;
    int64_t min_devices_ = 1;

    /// Estimate how long recovery will take (nanoseconds).
    int64_t estimate_recovery_time(int64_t num_failed) const;
};

} // namespace symplex::fault_tolerance
