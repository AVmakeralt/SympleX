// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once
#include "symplex/training/dynamic_batch.h"
#include "symplex/hardware/hardware_target.h"
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <mutex>

namespace symplex::training {

struct MemorySnapshot {
    int64_t total_bytes;
    int64_t used_bytes;
    int64_t free_bytes;
    double utilization_percent;
    int64_t timestamp_ns;
};

class MemoryWatchdog {
public:
    explicit MemoryWatchdog(const hardware::HardwareTarget& target);

    // Take a memory snapshot
    MemorySnapshot snapshot() const;

    // Register a callback for when memory pressure exceeds threshold
    using PressureCallback = std::function<void(const MemorySnapshot&)>;
    void on_pressure(PressureCallback cb);

    // Check if memory pressure is critical
    bool is_critical() const;
    bool is_warning() const;

    // Get recommended micro-batch size reduction factor
    double recommended_reduction_factor() const;

    // Thresholds
    double warning_threshold() const;   // e.g., 0.85
    double critical_threshold() const;  // e.g., 0.95

    void set_warning_threshold(double t);
    void set_critical_threshold(double t);

    // Track memory allocations/deallocations
    void allocate(int64_t bytes);
    void deallocate(int64_t bytes);
    void reset();

private:
    hardware::HardwareTarget target_;
    int64_t allocated_bytes_;
    mutable std::mutex mutex_;
    std::vector<PressureCallback> pressure_callbacks_;
    double warning_threshold_ = 0.85;
    double critical_threshold_ = 0.95;

    void check_pressure_and_notify();
};

} // namespace symplex::training
