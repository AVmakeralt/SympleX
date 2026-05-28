// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/fault_tolerance/health_monitor.h"
#include "symplex/fault_tolerance/forward_recovery.h"
#include "symplex/fault_tolerance/communicator_repair.h"
#include "symplex/fault_tolerance/checkpoint.h"
#include "symplex/distributed/mesh.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>
#include <cassert>

using namespace symplex::fault_tolerance;
using namespace symplex::distributed;
using namespace symplex::hardware;

int main() {
    HardwareTarget target = HardwareTarget::H100();

    // Test 1: Health monitor
    {
        auto mesh = ClusterMesh::create_2d_mesh(2, 2, 2, target);
        HealthMonitor monitor(mesh);
        monitor.report_heartbeat(0, 1000);
        monitor.report_heartbeat(1, 1000);
        assert(monitor.is_alive(0));
        assert(monitor.is_alive(1));
        std::cout << "[PASS] Health monitor\n";
    }

    // Test 2: Forward recovery
    {
        auto mesh = ClusterMesh::create_2d_mesh(2, 2, 2, target);
        ForwardRecovery recovery(mesh);
        recovery.set_min_devices(4);
        assert(recovery.can_recover(3));
        auto plan = recovery.recover({0, 1});
        assert(plan.success);
        std::cout << "[PASS] Forward recovery\n";
    }

    // Test 3: Checkpoint planner
    {
        CheckpointPlanner planner(8LL * 1024 * 1024 * 1024);  // 8 GB
        std::vector<int64_t> activation_bytes(24);
        std::vector<int64_t> recompute_ns(24);
        for (int i = 0; i < 24; i++) {
            activation_bytes[i] = 512LL * 1024 * 1024;  // 512 MB per layer
            recompute_ns[i] = 1'000'000;  // 1ms
        }
        auto plan = planner.plan(activation_bytes, recompute_ns);
        assert(!plan.decisions.empty());
        assert(plan.memory_savings_percent >= 0);
        std::cout << "[PASS] Checkpoint planner (savings=" << plan.memory_savings_percent << "%)\n";
    }

    // Test 4: Communicator repair
    {
        auto mesh = ClusterMesh::create_2d_mesh(2, 2, 2, target);
        CommunicatorRepair repair(mesh);
        repair.initialize_groups();
        mesh.mark_device_dead(0);
        bool ok = repair.repair_after_failure({0});
        assert(ok);
        std::cout << "[PASS] Communicator repair\n";
    }

    std::cout << "All fault tolerance tests passed!\n";
    return 0;
}
