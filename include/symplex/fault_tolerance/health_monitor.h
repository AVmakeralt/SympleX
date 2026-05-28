// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "symplex/distributed/mesh.h"

namespace symplex::fault_tolerance {

/// DeviceHealth: the health status of a single cluster device.
enum class DeviceHealth {
    HEALTHY,
    DEGRADED,       // Slow but functional
    SUSPECT,        // Possibly failing
    DEAD,           // Confirmed dead
    RECOVERING,     // Coming back online
};

/// HealthEvent: a recorded health state transition for a device.
struct HealthEvent {
    int64_t device_id;
    DeviceHealth status;
    int64_t timestamp_ns;
    std::string reason;

    std::string to_string() const;
};

/// HealthMonitor: tracks heartbeat signals, detects timeouts, and
/// maintains the current health status of every device in a ClusterMesh.
class HealthMonitor {
public:
    explicit HealthMonitor(const distributed::ClusterMesh& mesh);

    // ── Reporting ─────────────────────────────────────────────────────

    /// Record a health event directly.
    void report_event(HealthEvent event);

    /// Record a heartbeat from a device (e.g. a periodic liveness ping).
    void report_heartbeat(int64_t device_id, int64_t timestamp_ns);

    /// Report that a device has missed a heartbeat window (timeout).
    void report_timeout(int64_t device_id);

    /// Report an error on a device (e.g. CUDA error, ECC error).
    void report_error(int64_t device_id, const std::string& error_msg);

    // ── Querying ──────────────────────────────────────────────────────

    /// Get the current health status of a device.
    DeviceHealth device_health(int64_t device_id) const;

    /// True if the device is not DEAD.
    bool is_alive(int64_t device_id) const;

    /// True if the device is in the SUSPECT state.
    bool is_suspect(int64_t device_id) const;

    /// Return all device IDs currently in the given health state.
    std::vector<int64_t> devices_with_health(DeviceHealth status) const;

    /// Return all events that occurred at or after `since_timestamp_ns`.
    std::vector<HealthEvent> recent_events(int64_t since_timestamp_ns) const;

    // ── Periodic check ────────────────────────────────────────────────

    /// Scan all devices for heartbeat timeouts and return any new events.
    /// Intended to be called in a periodic loop.
    std::vector<HealthEvent> check_health();

    // ── Configuration ─────────────────────────────────────────────────

    /// Set the heartbeat timeout window in nanoseconds.
    void set_heartbeat_timeout_ns(int64_t timeout_ns);

    /// Set the number of consecutive missed heartbeats before a device is
    /// declared DEAD.
    void set_max_missed_heartbeats(int64_t max);

private:
    distributed::ClusterMesh mesh_;
    std::vector<DeviceHealth> device_health_;
    std::vector<int64_t> last_heartbeat_ns_;
    std::vector<int64_t> missed_heartbeats_;
    std::vector<HealthEvent> event_log_;
    int64_t heartbeat_timeout_ns_ = 30'000'000'000LL;  // 30 seconds
    int64_t max_missed_heartbeats_ = 3;
};

} // namespace symplex::fault_tolerance
