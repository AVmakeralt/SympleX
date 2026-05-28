// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/distributed/mesh.h"
#include "symplex/distributed/sharding.h"
#include "symplex/distributed/nccl_bridge.h"
#include "symplex/distributed/pipeline_overlap.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>
#include <cassert>

using namespace symplex::distributed;
using namespace symplex::hardware;

int main() {
    HardwareTarget target = HardwareTarget::H100();

    // Test 1: Cluster mesh creation
    {
        auto mesh = ClusterMesh::create_2d_mesh(4, 2, 8, target);
        assert(mesh.total_devices() == 64);
        assert(mesh.ndim() == 3);
        std::cout << "[PASS] Cluster mesh creation (" << mesh.total_devices() << " devices)\n";
    }

    // Test 2: Sharding analysis
    {
        auto mesh = ClusterMesh::create_2d_mesh(4, 2, 8, target);
        ShardingAnalyzer analyzer(mesh);
        auto plan = analyzer.analyze_matmul(4096, 4096, 4096);
        assert(!plan.tensor_shards.empty());
        assert(plan.total_communication_bytes > 0);
        std::cout << "[PASS] Sharding analysis (comm=" << plan.total_communication_bytes / 1024 / 1024 << " MB)\n";
    }

    // Test 3: Pipeline overlap
    {
        auto mesh = ClusterMesh::create_2d_mesh(4, 2, 8, target);
        NCCLBridge nccl(mesh);
        PipelineOverlapper pipeline(mesh, nccl);
        auto schedule = pipeline.compute_schedule(4, 16, 100000, 200000, 50000);
        assert(schedule.efficiency > 0);
        std::cout << "[PASS] Pipeline overlap (efficiency=" << schedule.efficiency * 100 << "%)\n";
    }

    // Test 4: Device health
    {
        auto mesh = ClusterMesh::create_2d_mesh(2, 2, 2, target);
        assert(mesh.num_alive_devices() == 8);
        mesh.mark_device_dead(3);
        assert(mesh.num_alive_devices() == 7);
        assert(mesh.dead_device_ids().size() == 1);
        std::cout << "[PASS] Device health tracking\n";
    }

    std::cout << "All distributed tests passed!\n";
    return 0;
}
