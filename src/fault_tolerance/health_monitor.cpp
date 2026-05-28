// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/fault_tolerance/health_monitor.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace symplex::fault_tolerance {

// ── Helpers ─────────────────────────────────────────────────────────────

static std::string device_health_to_string(DeviceHealth h) {
    switch (h) {
        case DeviceHealth::HEALTHY:    return "HEALTHY";
        case DeviceHealth::DEGRADED:   return "DEGRADED";
        case DeviceHealth::SUSPECT:    return "SUSPECT";
        case DeviceHealth::DEAD:       return "DEAD";
        case DeviceHealth::RECOVERING: return "RECOVERING";
    }
    return "UNKNOWN";
}

static int64_t now_ns() {
    auto tp = std::chrono::steady_clock::now();
    auto dur = tp.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
}

// ── HealthEvent ─────────────────────────────────────────────────────────

std::string HealthEvent::to_string() const {
    std::ostringstream oss;
    oss << "HealthEvent{device=" << device_id
        << ", status=" << device_health_to_string(status)
        << ", ts=" << timestamp_ns
        << ", reason='" << reason << "'}";
    return oss.str();
}

// ── Constructor ─────────────────────────────────────────────────────────

HealthMonitor::HealthMonitor(const distributed::ClusterMesh& mesh)
    : mesh_(mesh)
{
    const int64_t n = mesh_.total_devices();
    device_health_.resize(static_cast<size_t>(n), DeviceHealth::HEALTHY);
    last_heartbeat_ns_.resize(static_cast<size_t>(n), 0);
    missed_heartbeats_.resize(static_cast<size_t>(n), 0);
}

// ── Reporting ───────────────────────────────────────────────────────────

void HealthMonitor::report_event(HealthEvent event) {
    const int64_t id = event.device_id;
    if (id < 0 || id >= static_cast<int64_t>(device_health_.size())) {
        return; // Invalid device – silently ignore
    }

    // Update the device health vector
    device_health_[static_cast<size_t>(id)] = event.status;

    // If transitioning away from SUSPECT, reset missed-heartbeat counter
    if (event.status != DeviceHealth::SUSPECT) {
        missed_heartbeats_[static_cast<size_t>(id)] = 0;
    }

    // If the device is confirmed dead, also mark it in the mesh
    if (event.status == DeviceHealth::DEAD) {
        mesh_.mark_device_dead(id);
    }

    // If the device is recovering, mark it alive in the mesh
    if (event.status == DeviceHealth::RECOVERING ||
        event.status == DeviceHealth::HEALTHY) {
        mesh_.mark_device_alive(id);
    }

    // Append to the event log
    event_log_.push_back(std::move(event));
}

void HealthMonitor::report_heartbeat(int64_t device_id, int64_t timestamp_ns) {
    if (device_id < 0 || device_id >= static_cast<int64_t>(device_health_.size())) {
        return;
    }

    const size_t idx = static_cast<size_t>(device_id);
    last_heartbeat_ns_[idx] = timestamp_ns;
    missed_heartbeats_[idx] = 0;

    DeviceHealth prev = device_health_[idx];

    // A heartbeat from a SUSPECT or RECOVERING device means it is back
    if (prev == DeviceHealth::SUSPECT || prev == DeviceHealth::RECOVERING) {
        HealthEvent evt;
        evt.device_id = device_id;
        evt.status = DeviceHealth::HEALTHY;
        evt.timestamp_ns = timestamp_ns;
        evt.reason = "heartbeat received after " + device_health_to_string(prev);
        report_event(std::move(evt));
    }

    // A heartbeat from a DEAD device signals recovery
    if (prev == DeviceHealth::DEAD) {
        HealthEvent evt;
        evt.device_id = device_id;
        evt.status = DeviceHealth::RECOVERING;
        evt.timestamp_ns = timestamp_ns;
        evt.reason = "heartbeat from previously dead device";
        report_event(std::move(evt));
    }
}

void HealthMonitor::report_timeout(int64_t device_id) {
    if (device_id < 0 || device_id >= static_cast<int64_t>(device_health_.size())) {
        return;
    }

    const size_t idx = static_cast<size_t>(device_id);
    DeviceHealth prev = device_health_[idx];

    // Ignore timeouts from already-dead devices
    if (prev == DeviceHealth::DEAD) {
        return;
    }

    missed_heartbeats_[idx]++;

    if (prev == DeviceHealth::HEALTHY || prev == DeviceHealth::DEGRADED) {
        // First timeout → SUSPECT
        HealthEvent evt;
        evt.device_id = device_id;
        evt.status = DeviceHealth::SUSPECT;
        evt.timestamp_ns = now_ns();
        evt.reason = "heartbeat timeout (missed=" +
                     std::to_string(missed_heartbeats_[idx]) + ")";
        report_event(std::move(evt));
    } else if (prev == DeviceHealth::SUSPECT) {
        // Already suspect – check if we should declare dead
        if (missed_heartbeats_[idx] >= max_missed_heartbeats_) {
            HealthEvent evt;
            evt.device_id = device_id;
            evt.status = DeviceHealth::DEAD;
            evt.timestamp_ns = now_ns();
            evt.reason = "exceeded max missed heartbeats (" +
                         std::to_string(max_missed_heartbeats_) + ")";
            report_event(std::move(evt));
        }
    } else if (prev == DeviceHealth::RECOVERING) {
        // Recovery failed – back to suspect
        HealthEvent evt;
        evt.device_id = device_id;
        evt.status = DeviceHealth::SUSPECT;
        evt.timestamp_ns = now_ns();
        evt.reason = "timeout during recovery";
        report_event(std::move(evt));
    }
}

void HealthMonitor::report_error(int64_t device_id, const std::string& error_msg) {
    if (device_id < 0 || device_id >= static_cast<int64_t>(device_health_.size())) {
        return;
    }

    const size_t idx = static_cast<size_t>(device_id);
    DeviceHealth prev = device_health_[idx];

    // If already dead, nothing more to do
    if (prev == DeviceHealth::DEAD) {
        return;
    }

    // Determine the new status based on the current status and error severity.
    // For now, a non-fatal error degrades the device; repeated errors may
    // escalate to SUSPECT.
    DeviceHealth new_status;
    if (prev == DeviceHealth::HEALTHY) {
        new_status = DeviceHealth::DEGRADED;
    } else if (prev == DeviceHealth::DEGRADED) {
        new_status = DeviceHealth::SUSPECT;
    } else {
        // Already SUSPECT or RECOVERING – stay SUSPECT
        new_status = DeviceHealth::SUSPECT;
    }

    HealthEvent evt;
    evt.device_id = device_id;
    evt.status = new_status;
    evt.timestamp_ns = now_ns();
    evt.reason = "error: " + error_msg;
    report_event(std::move(evt));
}

// ── Querying ────────────────────────────────────────────────────────────

DeviceHealth HealthMonitor::device_health(int64_t device_id) const {
    if (device_id < 0 || device_id >= static_cast<int64_t>(device_health_.size())) {
        return DeviceHealth::DEAD;  // Unknown device treated as dead
    }
    return device_health_[static_cast<size_t>(device_id)];
}

bool HealthMonitor::is_alive(int64_t device_id) const {
    return device_health(device_id) != DeviceHealth::DEAD;
}

bool HealthMonitor::is_suspect(int64_t device_id) const {
    return device_health(device_id) == DeviceHealth::SUSPECT;
}

std::vector<int64_t> HealthMonitor::devices_with_health(DeviceHealth status) const {
    std::vector<int64_t> result;
    for (size_t i = 0; i < device_health_.size(); ++i) {
        if (device_health_[i] == status) {
            result.push_back(static_cast<int64_t>(i));
        }
    }
    return result;
}

std::vector<HealthEvent> HealthMonitor::recent_events(int64_t since_timestamp_ns) const {
    std::vector<HealthEvent> result;
    for (const auto& evt : event_log_) {
        if (evt.timestamp_ns >= since_timestamp_ns) {
            result.push_back(evt);
        }
    }
    return result;
}

// ── Periodic health check ───────────────────────────────────────────────

std::vector<HealthEvent> HealthMonitor::check_health() {
    std::vector<HealthEvent> new_events;
    const int64_t current_ns = now_ns();

    for (size_t i = 0; i < device_health_.size(); ++i) {
        DeviceHealth h = device_health_[i];
        // Skip already-dead devices
        if (h == DeviceHealth::DEAD) {
            continue;
        }

        int64_t last_hb = last_heartbeat_ns_[i];
        // If we have never received a heartbeat, treat the device as
        // healthy until the timeout expires.  A last_hb of 0 means we
        // have not yet received any heartbeat; we seed it with the
        // current time on the first check so that the device is not
        // immediately flagged as timed out.
        if (last_hb == 0) {
            last_heartbeat_ns_[i] = current_ns;
            continue;
        }

        int64_t elapsed = current_ns - last_hb;
        if (elapsed > heartbeat_timeout_ns_) {
            // Heartbeat timeout detected
            int64_t missed = elapsed / heartbeat_timeout_ns_;
            if (missed < 1) missed = 1;

            // Accumulate missed heartbeats
            int64_t device_id = static_cast<int64_t>(i);
            missed_heartbeats_[i] += static_cast<int64_t>(missed);

            if (h == DeviceHealth::HEALTHY || h == DeviceHealth::DEGRADED) {
                HealthEvent evt;
                evt.device_id = device_id;
                evt.status = DeviceHealth::SUSPECT;
                evt.timestamp_ns = current_ns;
                evt.reason = "heartbeat timeout during periodic check (missed=" +
                             std::to_string(missed_heartbeats_[i]) + ")";
                report_event(std::move(evt));
                new_events.push_back(event_log_.back());
            } else if (h == DeviceHealth::SUSPECT) {
                if (missed_heartbeats_[i] >= max_missed_heartbeats_) {
                    HealthEvent evt;
                    evt.device_id = device_id;
                    evt.status = DeviceHealth::DEAD;
                    evt.timestamp_ns = current_ns;
                    evt.reason = "exceeded max missed heartbeats during periodic check (" +
                                 std::to_string(max_missed_heartbeats_) + ")";
                    report_event(std::move(evt));
                    new_events.push_back(event_log_.back());
                }
            } else if (h == DeviceHealth::RECOVERING) {
                // Recovery timed out
                HealthEvent evt;
                evt.device_id = device_id;
                evt.status = DeviceHealth::SUSPECT;
                evt.timestamp_ns = current_ns;
                evt.reason = "recovery timeout during periodic check";
                report_event(std::move(evt));
                new_events.push_back(event_log_.back());
            }
        } else if (h == DeviceHealth::SUSPECT) {
            // Suspect but heartbeat arrived within the window – the
            // periodic check did not call report_timeout, but the
            // heartbeat may have arrived between checks.  If the
            // missed counter was already reset by report_heartbeat,
            // this path is a no-op.
        }
    }

    return new_events;
}

// ── Configuration ───────────────────────────────────────────────────────

void HealthMonitor::set_heartbeat_timeout_ns(int64_t timeout_ns) {
    heartbeat_timeout_ns_ = timeout_ns;
}

void HealthMonitor::set_max_missed_heartbeats(int64_t max) {
    max_missed_heartbeats_ = max;
}

} // namespace symplex::fault_tolerance
