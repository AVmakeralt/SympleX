// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/training/memory_watchdog.h"
#include <chrono>
#include <algorithm>

namespace symplex::training {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MemoryWatchdog::MemoryWatchdog(const hardware::HardwareTarget& target)
    : target_(target)
    , allocated_bytes_(0)
{
}

// ---------------------------------------------------------------------------
// Take a memory snapshot
// ---------------------------------------------------------------------------

MemorySnapshot MemoryWatchdog::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemorySnapshot snap;
    // Total GPU memory capacity in bytes
    snap.total_bytes = target_.gpu.memory.global_capacity_gb * 1024LL * 1024LL * 1024LL;
    snap.used_bytes = allocated_bytes_;
    snap.free_bytes = snap.total_bytes - snap.used_bytes;
    if (snap.free_bytes < 0) snap.free_bytes = 0;

    if (snap.total_bytes > 0) {
        snap.utilization_percent =
            static_cast<double>(snap.used_bytes) / static_cast<double>(snap.total_bytes);
    } else {
        snap.utilization_percent = 0.0;
    }

    // Use steady_clock as a monotonic timestamp proxy
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    snap.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    return snap;
}

// ---------------------------------------------------------------------------
// Register a pressure callback
// ---------------------------------------------------------------------------

void MemoryWatchdog::on_pressure(PressureCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    pressure_callbacks_.push_back(std::move(cb));
}

// ---------------------------------------------------------------------------
// Check if memory pressure is critical or at warning level
// ---------------------------------------------------------------------------

bool MemoryWatchdog::is_critical() const {
    MemorySnapshot snap = snapshot();
    return snap.utilization_percent >= critical_threshold_;
}

bool MemoryWatchdog::is_warning() const {
    MemorySnapshot snap = snapshot();
    return snap.utilization_percent >= warning_threshold_;
}

// ---------------------------------------------------------------------------
// Recommended reduction factor for micro-batch size
// ---------------------------------------------------------------------------

double MemoryWatchdog::recommended_reduction_factor() const {
    MemorySnapshot snap = snapshot();
    if (snap.utilization_percent >= critical_threshold_) {
        // Critical: halve the batch size
        return 0.5;
    }
    if (snap.utilization_percent >= warning_threshold_) {
        // Warning: reduce to 75%
        return 0.75;
    }
    // Normal: no reduction needed
    return 1.0;
}

// ---------------------------------------------------------------------------
// Threshold accessors and mutators
// ---------------------------------------------------------------------------

double MemoryWatchdog::warning_threshold() const {
    return warning_threshold_;
}

double MemoryWatchdog::critical_threshold() const {
    return critical_threshold_;
}

void MemoryWatchdog::set_warning_threshold(double t) {
    warning_threshold_ = std::clamp(t, 0.0, 1.0);
}

void MemoryWatchdog::set_critical_threshold(double t) {
    critical_threshold_ = std::clamp(t, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Track allocations / deallocations
// ---------------------------------------------------------------------------

void MemoryWatchdog::allocate(int64_t bytes) {
    if (bytes <= 0) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        allocated_bytes_ += bytes;
    }
    check_pressure_and_notify();
}

void MemoryWatchdog::deallocate(int64_t bytes) {
    if (bytes <= 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_bytes_ -= bytes;
    if (allocated_bytes_ < 0) allocated_bytes_ = 0;
    // No need to notify on deallocation — pressure can only decrease
}

void MemoryWatchdog::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_bytes_ = 0;
}

// ---------------------------------------------------------------------------
// Internal: check memory pressure and fire callbacks if needed
// ---------------------------------------------------------------------------

void MemoryWatchdog::check_pressure_and_notify() {
    // Take the snapshot outside the lock (snapshot() takes its own lock).
    // However, we must be careful: snapshot() acquires mutex_, and so does
    // this method's callers.  Since check_pressure_and_notify is called
    // AFTER the mutex_ is released by allocate(), this is safe.
    MemorySnapshot snap = snapshot();

    if (snap.utilization_percent < warning_threshold_) {
        return;  // No pressure — nothing to do
    }

    // Fire all registered callbacks with the current snapshot.
    // We deliberately do NOT hold mutex_ here to avoid deadlock
    // if a callback calls back into the watchdog.
    std::vector<PressureCallback> callbacks_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_copy = pressure_callbacks_;
    }

    for (const auto& cb : callbacks_copy) {
        cb(snap);
    }
}

} // namespace symplex::training
