// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/distributed/mesh.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace symplex::distributed {

// ── Constructor ────────────────────────────────────────────────────────

ClusterMesh::ClusterMesh(std::vector<MeshDimension> dims, std::vector<DeviceDescriptor> devices)
    : dims_(std::move(dims)), devices_(std::move(devices))
{
    // Validate: total mesh size must equal number of devices
    int64_t expected = 1;
    for (const auto& d : dims_) {
        expected *= d.size;
    }
    if (!devices_.empty() && static_cast<int64_t>(devices_.size()) != expected) {
        throw std::invalid_argument(
            "ClusterMesh: device count (" + std::to_string(devices_.size()) +
            ") does not match mesh volume (" + std::to_string(expected) + ")");
    }
}

// ── Accessors ──────────────────────────────────────────────────────────

size_t ClusterMesh::ndim() const {
    return dims_.size();
}

int64_t ClusterMesh::size(size_t dim) const {
    if (dim >= dims_.size()) {
        throw std::out_of_range(
            "ClusterMesh::size: dimension " + std::to_string(dim) +
            " out of range (ndim=" + std::to_string(dims_.size()) + ")");
    }
    return dims_[dim].size;
}

int64_t ClusterMesh::total_devices() const {
    if (dims_.empty()) return 0;
    int64_t total = 1;
    for (const auto& d : dims_) {
        total *= d.size;
    }
    return total;
}

const std::vector<MeshDimension>& ClusterMesh::dimensions() const {
    return dims_;
}

const std::vector<DeviceDescriptor>& ClusterMesh::devices() const {
    return devices_;
}

// ── Device lookup ──────────────────────────────────────────────────────

const DeviceDescriptor& ClusterMesh::device_at(const MeshCoord& coord) const {
    int64_t linear = coord_to_linear(coord);
    if (linear < 0 || linear >= static_cast<int64_t>(devices_.size())) {
        throw std::out_of_range(
            "ClusterMesh::device_at: coordinate " + coord.to_string() +
            " maps to out-of-range linear index " + std::to_string(linear));
    }
    return devices_[static_cast<size_t>(linear)];
}

DeviceDescriptor& ClusterMesh::device_at(const MeshCoord& coord) {
    int64_t linear = coord_to_linear(coord);
    if (linear < 0 || linear >= static_cast<int64_t>(devices_.size())) {
        throw std::out_of_range(
            "ClusterMesh::device_at: coordinate " + coord.to_string() +
            " maps to out-of-range linear index " + std::to_string(linear));
    }
    return devices_[static_cast<size_t>(linear)];
}

std::vector<DeviceDescriptor> ClusterMesh::devices_along(
    size_t dim, const MeshCoord& fixed
) const {
    if (dim >= dims_.size()) {
        throw std::out_of_range(
            "ClusterMesh::devices_along: dimension " + std::to_string(dim) +
            " out of range");
    }
    if (fixed.size() != dims_.size()) {
        throw std::invalid_argument(
            "ClusterMesh::devices_along: coordinate dimension mismatch");
    }

    std::vector<DeviceDescriptor> result;
    result.reserve(static_cast<size_t>(dims_[dim].size));

    MeshCoord coord = fixed;
    for (int64_t i = 0; i < dims_[dim].size; ++i) {
        coord[dim] = i;
        result.push_back(device_at(coord));
    }
    return result;
}

// ── Mesh operations ────────────────────────────────────────────────────

ClusterMesh ClusterMesh::submesh(size_t dim, int64_t start, int64_t end) const {
    if (dim >= dims_.size()) {
        throw std::out_of_range(
            "ClusterMesh::submesh: dimension " + std::to_string(dim) +
            " out of range");
    }
    if (start < 0 || end > dims_[dim].size || start >= end) {
        throw std::invalid_argument(
            "ClusterMesh::submesh: invalid range [" +
            std::to_string(start) + ", " + std::to_string(end) +
            ") for dimension " + std::to_string(dim) +
            " of size " + std::to_string(dims_[dim].size));
    }

    // Build new dimensions with the sliced dimension shrunk
    std::vector<MeshDimension> new_dims = dims_;
    new_dims[dim].size = end - start;

    // Collect devices that fall within the submesh range
    std::vector<DeviceDescriptor> new_devices;
    int64_t new_total = 1;
    for (const auto& d : new_dims) {
        new_total *= d.size;
    }
    new_devices.reserve(static_cast<size_t>(new_total));

    // Iterate over all coordinates in the new submesh
    // We translate coordinates by offsetting the sliced dimension
    MeshCoord coord(std::vector<int64_t>(new_dims.size(), 0));

    // Use a recursive lambda or iterative enumeration
    // Iterative approach: enumerate all valid coordinate combos
    std::vector<int64_t> current(new_dims.size(), 0);

    bool done = false;
    while (!done) {
        // Translate to original mesh coordinate by offsetting dim
        MeshCoord orig_coord;
        orig_coord.coords = current;
        orig_coord[dim] += start;

        // Find the device at the original coordinate
        const DeviceDescriptor& orig_dev = device_at(orig_coord);

        // Create a new device descriptor for the submesh
        DeviceDescriptor new_dev = orig_dev;
        new_dev.mesh_coord = MeshCoord(current);
        // Remap global_id: use the linear index in the submesh
        new_dev.global_id = static_cast<int64_t>(new_devices.size());
        new_devices.push_back(std::move(new_dev));

        // Increment the coordinate (little-endian)
        int64_t d = static_cast<int64_t>(new_dims.size()) - 1;
        while (d >= 0) {
            current[static_cast<size_t>(d)]++;
            if (current[static_cast<size_t>(d)] < new_dims[static_cast<size_t>(d)].size) {
                break;
            }
            current[static_cast<size_t>(d)] = 0;
            d--;
        }
        if (d < 0) {
            done = true;
        }
    }

    return ClusterMesh(std::move(new_dims), std::move(new_devices));
}

MeshCoord ClusterMesh::linear_to_coord(int64_t linear_id) const {
    if (dims_.empty()) {
        return MeshCoord{};
    }
    if (linear_id < 0 || linear_id >= total_devices()) {
        throw std::out_of_range(
            "ClusterMesh::linear_to_coord: linear_id " +
            std::to_string(linear_id) + " out of range");
    }

    // Row-major decomposition: last dimension varies fastest
    std::vector<int64_t> coords(dims_.size());
    int64_t remaining = linear_id;
    for (size_t i = dims_.size(); i-- > 0; ) {
        coords[i] = remaining % dims_[i].size;
        remaining /= dims_[i].size;
    }
    return MeshCoord(std::move(coords));
}

int64_t ClusterMesh::coord_to_linear(const MeshCoord& coord) const {
    if (coord.size() != dims_.size()) {
        throw std::invalid_argument(
            "ClusterMesh::coord_to_linear: coordinate dimension mismatch");
    }

    // Row-major: first dimension is the slowest-varying
    int64_t linear = 0;
    int64_t stride = 1;
    for (size_t i = dims_.size(); i-- > 0; ) {
        if (coord[i] < 0 || coord[i] >= dims_[i].size) {
            throw std::out_of_range(
                "ClusterMesh::coord_to_linear: coordinate " +
                std::to_string(coord[i]) + " out of range for dimension " +
                std::to_string(i) + " (size=" + std::to_string(dims_[i].size) + ")");
        }
        linear += coord[i] * stride;
        stride *= dims_[i].size;
    }
    return linear;
}

// ── Health ─────────────────────────────────────────────────────────────

void ClusterMesh::mark_device_dead(int64_t global_id) {
    if (global_id < 0 || global_id >= static_cast<int64_t>(devices_.size())) {
        return;
    }
    devices_[static_cast<size_t>(global_id)].alive = false;
}

void ClusterMesh::mark_device_alive(int64_t global_id) {
    if (global_id < 0 || global_id >= static_cast<int64_t>(devices_.size())) {
        return;
    }
    devices_[static_cast<size_t>(global_id)].alive = true;
}

int64_t ClusterMesh::num_alive_devices() const {
    int64_t count = 0;
    for (const auto& dev : devices_) {
        if (dev.alive) ++count;
    }
    return count;
}

std::vector<int64_t> ClusterMesh::dead_device_ids() const {
    std::vector<int64_t> dead;
    for (const auto& dev : devices_) {
        if (!dev.alive) {
            dead.push_back(dev.global_id);
        }
    }
    return dead;
}

// ── String representation ──────────────────────────────────────────────

std::string ClusterMesh::to_string() const {
    std::ostringstream oss;
    oss << "ClusterMesh{dims=[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims_[i].to_string();
    }
    oss << "], total=" << total_devices()
        << ", alive=" << num_alive_devices()
        << ", devices=[";
    for (size_t i = 0; i < devices_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << devices_[i].to_string();
    }
    oss << "]}";
    return oss.str();
}

// ── Factory ────────────────────────────────────────────────────────────

ClusterMesh ClusterMesh::create_2d_mesh(
    int64_t tp_size, int64_t pp_size, int64_t dp_size,
    const hardware::HardwareTarget& target
) {
    std::vector<MeshDimension> dims;
    dims.emplace_back("data_parallel", dp_size);
    dims.emplace_back("pipeline_parallel", pp_size);
    dims.emplace_back("tensor_parallel", tp_size);

    int64_t total = dp_size * pp_size * tp_size;
    std::vector<DeviceDescriptor> devices;
    devices.reserve(static_cast<size_t>(total));

    int64_t global_id = 0;
    for (int64_t dp = 0; dp < dp_size; ++dp) {
        for (int64_t pp = 0; pp < pp_size; ++pp) {
            for (int64_t tp = 0; tp < tp_size; ++tp) {
                DeviceDescriptor dev;
                dev.global_id = global_id;
                dev.local_id = tp;  // Within node, each GPU gets a local id
                dev.node_id = dp * pp_size + pp;  // Simplified node assignment
                dev.mesh_coord = MeshCoord({dp, pp, tp});
                dev.hostname = "node" + std::to_string(dev.node_id);
                dev.gpu_id = tp;
                dev.target = target;
                dev.alive = true;
                devices.push_back(std::move(dev));
                ++global_id;
            }
        }
    }

    return ClusterMesh(std::move(dims), std::move(devices));
}

} // namespace symplex::distributed
