// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/fault_tolerance/forward_recovery.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace symplex::fault_tolerance {

// ── Constructor ─────────────────────────────────────────────────────────

ForwardRecovery::ForwardRecovery(distributed::ClusterMesh& mesh)
    : mesh_(mesh)
{}

// ── Main entry point ────────────────────────────────────────────────────

RecoveryPlan ForwardRecovery::recover(const std::vector<int64_t>& failed_devices) {
    RecoveryPlan plan;
    plan.failed_devices = failed_devices;
    plan.needs_remesh = false;
    plan.success = false;
    plan.estimated_recovery_ns = 0;

    // Validate that recovery is possible
    if (!can_recover(static_cast<int64_t>(failed_devices.size()))) {
        plan.success = false;
        plan.error_message = "cannot recover: too many devices failed (" +
                             std::to_string(failed_devices.size()) +
                             " failed, " + std::to_string(min_devices_) +
                             " minimum required)";
        return plan;
    }

    // Collect surviving device IDs
    const auto& all_devices = mesh_.devices();
    std::unordered_set<int64_t> failed_set(failed_devices.begin(), failed_devices.end());

    for (const auto& dev : all_devices) {
        if (failed_set.find(dev.global_id) == failed_set.end() && dev.alive) {
            plan.surviving_devices.push_back(dev.global_id);
        }
    }

    // Double-check: after filtering, do we still have enough?
    if (static_cast<int64_t>(plan.surviving_devices.size()) < min_devices_) {
        plan.success = false;
        plan.error_message = "cannot recover: only " +
                             std::to_string(plan.surviving_devices.size()) +
                             " surviving devices, need " +
                             std::to_string(min_devices_);
        return plan;
    }

    // Mark failed devices as dead in the mesh
    bool mesh_ok = reconstruct_mesh(failed_devices);
    if (!mesh_ok) {
        plan.success = false;
        plan.error_message = "mesh reconstruction failed";
        return plan;
    }

    // Decide if the mesh topology needs rebuilding.
    // If any dead device leaves a "hole" in the mesh grid that cannot
    // be filled by simple rerouting, we need a remesh.
    // Heuristic: if more than half the devices along any mesh dimension
    // are dead, we need a remesh.
    plan.needs_remesh = false;
    for (size_t dim = 0; dim < mesh_.ndim(); ++dim) {
        // Count alive devices along each "slice" of this dimension
        // For simplicity, check the global ratio
        int64_t dim_size = mesh_.size(dim);
        // If a large fraction of devices have died overall, flag remesh
        double death_ratio = static_cast<double>(failed_devices.size()) /
                             static_cast<double>(mesh_.total_devices());
        if (death_ratio > 0.5) {
            plan.needs_remesh = true;
            break;
        }
        // Also check if any single row/column is completely dead
        // We enumerate all coordinate slices along this dimension
        std::vector<int64_t> other_dims;
        for (size_t d = 0; d < mesh_.ndim(); ++d) {
            if (d != dim) other_dims.push_back(static_cast<int64_t>(d));
        }

        // For each slice, count alive devices
        int64_t num_slices = 1;
        for (auto d : other_dims) {
            num_slices *= mesh_.size(static_cast<size_t>(d));
        }

        for (int64_t s = 0; s < num_slices; ++s) {
            // Decode slice index into coordinates for non-target dims
            std::vector<int64_t> fixed_coords(mesh_.ndim(), 0);
            int64_t tmp = s;
            for (auto d : other_dims) {
                fixed_coords[static_cast<size_t>(d)] = tmp % mesh_.size(static_cast<size_t>(d));
                tmp /= mesh_.size(static_cast<size_t>(d));
            }

            // Count alive devices along this slice
            int64_t alive_in_slice = 0;
            for (int64_t i = 0; i < dim_size; ++i) {
                fixed_coords[dim] = i;
                distributed::MeshCoord coord{std::vector<int64_t>(fixed_coords)};
                try {
                    const auto& dev = mesh_.device_at(coord);
                    if (dev.alive && failed_set.find(dev.global_id) == failed_set.end()) {
                        alive_in_slice++;
                    }
                } catch (...) {
                    // Coordinate out of range – skip
                }
            }

            // If the entire slice is dead, we need a remesh
            if (alive_in_slice == 0) {
                plan.needs_remesh = true;
                break;
            }
        }

        if (plan.needs_remesh) break;
    }

    // Redistribute work across surviving devices.
    // The total work is proportional to the original number of devices.
    int64_t total_work = mesh_.total_devices();
    RecoveryPlan redist = redistribute_workload(
        failed_devices, plan.surviving_devices, total_work);
    plan.redistributed_work = redist.redistributed_work;

    // Estimate recovery time
    plan.estimated_recovery_ns = estimate_recovery_time(
        static_cast<int64_t>(failed_devices.size()));

    plan.success = true;
    return plan;
}

// ── Work redistribution ─────────────────────────────────────────────────

RecoveryPlan ForwardRecovery::redistribute_workload(
    const std::vector<int64_t>& failed,
    const std::vector<int64_t>& survivors,
    int64_t total_work_units
) {
    RecoveryPlan plan;
    plan.failed_devices = failed;
    plan.surviving_devices = survivors;
    plan.needs_remesh = false;
    plan.estimated_recovery_ns = 0;
    plan.success = true;

    if (survivors.empty()) {
        plan.success = false;
        plan.error_message = "no surviving devices to redistribute work to";
        return plan;
    }

    int64_t num_survivors = static_cast<int64_t>(survivors.size());

    // Base work per surviving device (integer division)
    int64_t base_work = total_work_units / num_survivors;
    int64_t remainder = total_work_units % num_survivors;

    // Distribute the remainder one unit at a time across the first
    // `remainder` surviving devices, so that the total is exact.
    plan.redistributed_work.resize(static_cast<size_t>(num_survivors), base_work);
    for (int64_t i = 0; i < remainder; ++i) {
        plan.redistributed_work[static_cast<size_t>(i)]++;
    }

    return plan;
}

// ── Mesh reconstruction ─────────────────────────────────────────────────

bool ForwardRecovery::reconstruct_mesh(const std::vector<int64_t>& failed_devices) {
    // Mark each failed device as dead in the mesh.
    // The mesh itself handles the alive/dead flag; communicator groups
    // and routing tables are rebuilt by CommunicatorRepair.
    for (int64_t id : failed_devices) {
        if (id < 0 || id >= mesh_.total_devices()) {
            continue; // Invalid ID – skip
        }
        mesh_.mark_device_dead(id);
    }

    // Verify that enough devices remain alive after marking
    if (mesh_.num_alive_devices() < min_devices_) {
        return false;
    }

    return true;
}

// ── Feasibility ─────────────────────────────────────────────────────────

bool ForwardRecovery::can_recover(int64_t num_failed) const {
    int64_t alive = mesh_.num_alive_devices();
    // We subtract num_failed because the mesh may not have been
    // updated yet when this method is called (i.e. devices are
    // about to be marked dead).
    int64_t projected_alive = alive - num_failed;
    return projected_alive >= min_devices_;
}

int64_t ForwardRecovery::min_devices() const {
    return min_devices_;
}

void ForwardRecovery::set_min_devices(int64_t n) {
    min_devices_ = n;
}

// ── Recovery time estimation ────────────────────────────────────────────

int64_t ForwardRecovery::estimate_recovery_time(int64_t num_failed) const {
    // Recovery time is dominated by:
    //   1. NCCL communicator reinitialization: ~500ms per device group
    //   2. Work redistribution overhead: ~10ms per surviving device
    //   3. State checkpointing / rollback: ~200ms
    //   4. Mesh rebuild if needed: ~1s
    //
    // These are rough estimates; the actual time depends on the
    // cluster size and interconnect.

    const int64_t comm_reinit_ns = 500'000'000LL;  // 500 ms
    const int64_t work_redist_ns = 10'000'000LL;    // 10 ms
    const int64_t checkpoint_ns  = 200'000'000LL;   // 200 ms
    const int64_t mesh_rebuild_ns = 1'000'000'000LL; // 1 s

    int64_t num_survivors = mesh_.num_alive_devices() - num_failed;
    if (num_survivors < 0) num_survivors = 0;

    int64_t total_ns = 0;

    // Communicator reinit: one per mesh dimension
    total_ns += static_cast<int64_t>(mesh_.ndim()) * comm_reinit_ns;

    // Work redistribution: proportional to surviving device count
    total_ns += num_survivors * work_redist_ns;

    // Checkpoint cost
    total_ns += checkpoint_ns;

    // Mesh rebuild cost (if more than half the devices in any
    // dimension have failed, we likely need a full remesh)
    double failure_ratio = static_cast<double>(num_failed) /
                           static_cast<double>(mesh_.total_devices());
    if (failure_ratio > 0.5) {
        total_ns += mesh_rebuild_ns;
    }

    return total_ns;
}

} // namespace symplex::fault_tolerance
