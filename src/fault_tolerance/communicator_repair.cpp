// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/fault_tolerance/communicator_repair.h"

#include <algorithm>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace symplex::fault_tolerance {

// ── Constructor ─────────────────────────────────────────────────────────

CommunicatorRepair::CommunicatorRepair(distributed::ClusterMesh& mesh)
    : mesh_(mesh)
{}

// ── Initialization ──────────────────────────────────────────────────────

void CommunicatorRepair::initialize_groups() {
    groups_.clear();

    // Build one set of groups per mesh dimension.
    // For a 3D mesh [dp, pp, tp], we get:
    //   - groups along dim 0 (dp):   each group has all devices sharing the same (pp, tp)
    //   - groups along dim 1 (pp):   each group has all devices sharing the same (dp, tp)
    //   - groups along dim 2 (tp):   each group has all devices sharing the same (dp, pp)
    for (size_t dim = 0; dim < mesh_.ndim(); ++dim) {
        build_groups_along_dim(dim);
    }
}

void CommunicatorRepair::build_groups_along_dim(size_t mesh_dim) {
    if (mesh_dim >= mesh_.ndim()) {
        return;
    }

    // Collect all coordinate slices along this dimension.
    // A "slice" is defined by fixing all other dimensions and varying
    // the target dimension.  Each slice becomes a CommunicatorGroup.

    // Build the set of "other" dimension indices
    std::vector<size_t> other_dims;
    for (size_t d = 0; d < mesh_.ndim(); ++d) {
        if (d != mesh_dim) {
            other_dims.push_back(d);
        }
    }

    // Compute the number of slices: the product of sizes of all
    // non-target dimensions.
    int64_t num_slices = 1;
    for (size_t d : other_dims) {
        num_slices *= mesh_.size(d);
    }

    int64_t dim_size = mesh_.size(mesh_dim);

    for (int64_t s = 0; s < num_slices; ++s) {
        CommunicatorGroup group;
        group.group_id = static_cast<int64_t>(groups_.size());
        group.active = true;

        // Decode the slice index `s` into fixed coordinates for the
        // non-target dimensions.
        std::vector<int64_t> fixed_coords(mesh_.ndim(), 0);
        int64_t tmp = s;
        for (auto it = other_dims.rbegin(); it != other_dims.rend(); ++it) {
            fixed_coords[*it] = tmp % mesh_.size(*it);
            tmp /= mesh_.size(*it);
        }

        // Enumerate all devices along the target dimension with the
        // fixed coordinates.
        for (int64_t i = 0; i < dim_size; ++i) {
            fixed_coords[mesh_dim] = i;
            distributed::MeshCoord coord{std::vector<int64_t>(fixed_coords)};
            try {
                const auto& dev = mesh_.device_at(coord);
                if (dev.alive) {
                    group.device_ids.push_back(dev.global_id);
                }
            } catch (...) {
                // Invalid coordinate – skip
            }
        }

        // A group with fewer than 2 alive devices cannot perform
        // collective operations on this dimension, but we still keep
        // it (marked active) for routing purposes.
        groups_.push_back(std::move(group));
    }
}

// ── Repair ──────────────────────────────────────────────────────────────

bool CommunicatorRepair::repair_after_failure(
    const std::vector<int64_t>& failed_devices
) {
    std::unordered_set<int64_t> failed_set(failed_devices.begin(),
                                            failed_devices.end());

    // Mark failed devices as dead in the mesh first
    for (int64_t id : failed_devices) {
        mesh_.mark_device_dead(id);
    }

    // If groups haven't been initialized yet, build them now
    if (groups_.empty()) {
        initialize_groups();
        return true;
    }

    // Remove dead devices from every group and deactivate empty groups
    for (auto& group : groups_) {
        // Erase-remove idiom: remove all failed device IDs
        auto new_end = std::remove_if(
            group.device_ids.begin(),
            group.device_ids.end(),
            [&failed_set](int64_t id) {
                return failed_set.count(id) > 0;
            }
        );
        group.device_ids.erase(new_end, group.device_ids.end());

        // Deactivate groups that have no surviving devices
        if (group.device_ids.empty()) {
            group.active = false;
        }
    }

    // After removing dead devices, some groups may have too few members
    // for meaningful collectives.  We keep them active for routing but
    // note that collectives along a group with 0 or 1 members are
    // effectively no-ops.

    return true;
}

// ── Queries ─────────────────────────────────────────────────────────────

const CommunicatorGroup* CommunicatorRepair::group_for_device(
    int64_t device_id
) const {
    // Return the first active group that contains the device.
    for (const auto& group : groups_) {
        if (!group.active) continue;
        auto it = std::find(group.device_ids.begin(),
                            group.device_ids.end(),
                            device_id);
        if (it != group.device_ids.end()) {
            return &group;
        }
    }
    return nullptr;
}

std::vector<const CommunicatorGroup*> CommunicatorRepair::active_groups() const {
    std::vector<const CommunicatorGroup*> result;
    for (const auto& group : groups_) {
        if (group.active) {
            result.push_back(&group);
        }
    }
    return result;
}

// ── Rerouting ───────────────────────────────────────────────────────────

std::vector<int64_t> CommunicatorRepair::find_alternative_path(
    int64_t src, int64_t dst,
    const std::vector<int64_t>& exclude
) const {
    // BFS over the mesh adjacency graph.
    // Two devices are adjacent if they share a communicator group
    // (i.e. they are in the same row or column of the mesh).

    std::unordered_set<int64_t> excluded(exclude.begin(), exclude.end());
    excluded.insert(src); // Don't revisit the source

    // Build an adjacency map from the communicator groups
    std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
    for (const auto& group : groups_) {
        if (!group.active) continue;
        for (int64_t a : group.device_ids) {
            for (int64_t b : group.device_ids) {
                if (a != b) {
                    adjacency[a].push_back(b);
                }
            }
        }
    }

    // BFS
    std::queue<int64_t> q;
    std::unordered_map<int64_t, int64_t> parent;
    q.push(src);
    parent[src] = -1;

    while (!q.empty()) {
        int64_t current = q.front();
        q.pop();

        if (current == dst) {
            // Reconstruct path
            std::vector<int64_t> path;
            int64_t node = dst;
            while (node != -1 && node != src) {
                path.push_back(node);
                auto it = parent.find(node);
                if (it == parent.end()) break;
                node = it->second;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        auto it = adjacency.find(current);
        if (it == adjacency.end()) continue;

        for (int64_t neighbor : it->second) {
            if (parent.find(neighbor) != parent.end()) continue;
            if (excluded.count(neighbor) > 0) continue;

            // Also skip dead devices
            const auto& devices = mesh_.devices();
            if (neighbor >= 0 && neighbor < static_cast<int64_t>(devices.size())) {
                if (!devices[static_cast<size_t>(neighbor)].alive) continue;
            }

            parent[neighbor] = current;
            q.push(neighbor);
        }
    }

    // No path found
    return {};
}

std::string CommunicatorRepair::emit_rerouted_collective(
    const std::string& original_op,
    int64_t src_device,
    int64_t dst_device
) const {
    std::ostringstream oss;

    // Check if src and dst are in the same communicator group and
    // both alive.
    const auto& devices = mesh_.devices();

    auto is_alive = [&](int64_t id) -> bool {
        if (id < 0 || id >= static_cast<int64_t>(devices.size())) return false;
        return devices[static_cast<size_t>(id)].alive;
    };

    // If both devices are alive and share a group, the original
    // collective can proceed without rerouting.
    bool same_group = false;
    for (const auto& group : groups_) {
        if (!group.active) continue;
        bool has_src = std::find(group.device_ids.begin(),
                                  group.device_ids.end(),
                                  src_device) != group.device_ids.end();
        bool has_dst = std::find(group.device_ids.begin(),
                                  group.device_ids.end(),
                                  dst_device) != group.device_ids.end();
        if (has_src && has_dst) {
            same_group = true;
            break;
        }
    }

    if (same_group && is_alive(src_device) && is_alive(dst_device)) {
        // No rerouting needed
        oss << "// Direct path exists between device " << src_device
            << " and " << dst_device << "\n";
        oss << original_op << "(sendbuf, recvbuf, count, datatype, op, "
            << "comm[" << src_device << "], stream[" << src_device << "]);\n";
        return oss.str();
    }

    // Find an alternative path
    std::vector<int64_t> dead_ids;
    for (const auto& dev : devices) {
        if (!dev.alive) {
            dead_ids.push_back(dev.global_id);
        }
    }

    std::vector<int64_t> path = find_alternative_path(
        src_device, dst_device, dead_ids);

    if (path.empty()) {
        oss << "// ERROR: no alternative path from device " << src_device
            << " to " << dst_device << "\n";
        oss << "// Original operation: " << original_op << "\n";
        oss << "// Falling back to broadcast-based reroute\n";
        oss << "ncclBroadcast(sendbuf, recvbuf, count, datatype, "
            << "/*root=*/" << src_device << ", "
            << "comm[" << src_device << "], stream[" << src_device << "]);\n";
        return oss.str();
    }

    // Emit a sequence of point-to-point send/recv hops along the path
    oss << "// Rerouted " << original_op << " from device "
        << src_device << " to " << dst_device << "\n";
    oss << "// Path: " << src_device;
    for (int64_t hop : path) {
        oss << " -> " << hop;
    }
    oss << "\n";

    // Build the relay: each intermediate device receives from the
    // previous hop and sends to the next.
    int64_t prev = src_device;
    for (size_t i = 0; i < path.size(); ++i) {
        int64_t current = path[i];

        // The last hop is the destination
        if (i == path.size() - 1) {
            oss << "ncclSend(sendbuf, count, datatype, /*dst=*/"
                << current << ", comm[" << prev << "], stream["
                << prev << "]);\n";
            oss << "ncclRecv(recvbuf, count, datatype, /*src=*/"
                << prev << ", comm[" << current << "], stream["
                << current << "]);\n";
        } else {
            // Intermediate relay: receive then forward
            int64_t next = path[i + 1];

            oss << "// Relay hop: device " << current << "\n";
            oss << "ncclRecv(tmp_buf_" << current << ", count, datatype, /*src=*/"
                << prev << ", comm[" << current << "], stream["
                << current << "]);\n";
            oss << "ncclSend(tmp_buf_" << current << ", count, datatype, /*dst=*/"
                << next << ", comm[" << current << "], stream["
                << current << "]);\n";
        }

        prev = current;
    }

    return oss.str();
}

} // namespace symplex::fault_tolerance
