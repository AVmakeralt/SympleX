// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <numeric>

#include "symplex/hardware/hardware_target.h"

namespace symplex::distributed {

/// MeshDimension: a named dimension in the cluster device mesh.
struct MeshDimension {
    std::string name;     // "x", "y", "z", "data_parallel", "tensor_parallel", etc.
    int64_t size;         // Number of devices in this dimension

    MeshDimension() : size(0) {}
    MeshDimension(std::string n, int64_t s) : name(std::move(n)), size(s) {}

    std::string to_string() const {
        return name + "=" + std::to_string(size);
    }
};

/// MeshCoord: a coordinate within the multi-dimensional device mesh.
struct MeshCoord {
    std::vector<int64_t> coords;  // One coordinate per mesh dimension

    MeshCoord() = default;
    explicit MeshCoord(std::vector<int64_t> c) : coords(std::move(c)) {}

    int64_t& operator[](size_t i) { return coords[i]; }
    const int64_t& operator[](size_t i) const { return coords[i]; }
    size_t size() const { return coords.size(); }
    bool empty() const { return coords.empty(); }

    bool operator==(const MeshCoord& other) const {
        return coords == other.coords;
    }
    bool operator!=(const MeshCoord& other) const {
        return coords != other.coords;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < coords.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << coords[i];
        }
        oss << ")";
        return oss.str();
    }
};

/// DeviceDescriptor: full description of a single device in the cluster.
struct DeviceDescriptor {
    int64_t global_id = -1;
    int64_t local_id = -1;          // Within its node
    int64_t node_id = -1;
    MeshCoord mesh_coord;
    std::string hostname;
    int64_t gpu_id = -1;
    hardware::HardwareTarget target;
    bool alive = true;

    std::string to_string() const {
        return "Device{gid=" + std::to_string(global_id) +
               ", lid=" + std::to_string(local_id) +
               ", node=" + std::to_string(node_id) +
               ", gpu=" + std::to_string(gpu_id) +
               ", coord=" + mesh_coord.to_string() +
               ", alive=" + (alive ? "yes" : "no") + "}";
    }
};

/// ClusterMesh: represents the multi-dimensional mesh topology of a
/// distributed training cluster. Devices are arranged in a logical
/// multi-dimensional grid where each dimension corresponds to a
/// parallelism strategy (e.g., tensor parallel, pipeline parallel,
/// data parallel).
class ClusterMesh {
public:
    ClusterMesh() = default;
    ClusterMesh(std::vector<MeshDimension> dims, std::vector<DeviceDescriptor> devices);

    // ── Accessors ─────────────────────────────────────────────────────

    /// Number of dimensions in the mesh.
    size_t ndim() const;

    /// Size of a specific mesh dimension.
    int64_t size(size_t dim) const;

    /// Total number of device slots in the mesh.
    int64_t total_devices() const;

    /// All mesh dimensions.
    const std::vector<MeshDimension>& dimensions() const;

    /// All device descriptors.
    const std::vector<DeviceDescriptor>& devices() const;

    // ── Device lookup ─────────────────────────────────────────────────

    /// Get device at a specific mesh coordinate (const).
    const DeviceDescriptor& device_at(const MeshCoord& coord) const;

    /// Get device at a specific mesh coordinate (mutable).
    DeviceDescriptor& device_at(const MeshCoord& coord);

    /// Get all devices along a mesh dimension, fixing other coordinates.
    std::vector<DeviceDescriptor> devices_along(size_t dim, const MeshCoord& fixed) const;

    // ── Mesh operations ───────────────────────────────────────────────

    /// Extract a submesh spanning [start, end) along a given dimension.
    ClusterMesh submesh(size_t dim, int64_t start, int64_t end) const;

    /// Convert a linear device index to a mesh coordinate.
    MeshCoord linear_to_coord(int64_t linear_id) const;

    /// Convert a mesh coordinate to a linear device index.
    int64_t coord_to_linear(const MeshCoord& coord) const;

    // ── Health ────────────────────────────────────────────────────────

    /// Mark a device as dead (failed).
    void mark_device_dead(int64_t global_id);

    /// Mark a device as alive (recovered).
    void mark_device_alive(int64_t global_id);

    /// Count the number of alive devices.
    int64_t num_alive_devices() const;

    /// Get the global IDs of all dead devices.
    std::vector<int64_t> dead_device_ids() const;

    /// String representation of the entire mesh.
    std::string to_string() const;

    /// Factory: create a standard 2D mesh for tensor+pipeline parallelism
    /// with an outer data-parallel dimension.
    /// Mesh dimensions: [dp_size, pp_size, tp_size]
    static ClusterMesh create_2d_mesh(
        int64_t tp_size, int64_t pp_size, int64_t dp_size,
        const hardware::HardwareTarget& target
    );

private:
    std::vector<MeshDimension> dims_;
    std::vector<DeviceDescriptor> devices_;
};

} // namespace symplex::distributed
