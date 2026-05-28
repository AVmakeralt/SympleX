// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/schedule_tree.h"
#include "symplex/schedule/tiling.h"
#include "symplex/schedule/fusion.h"
#include "symplex/schedule/parallelization.h"
#include "symplex/schedule/schedule_map.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>
#include <cassert>

using namespace symplex::schedule;
using namespace symplex::hardware;

int main() {
    // Test 1: ScheduleTree construction
    {
        auto root = ScheduleTree::create(ScheduleNodeType::DOMAIN);
        root->set_domain_name("test_domain");
        auto band = root->add_child(ScheduleNodeType::BAND);
        band->band_data().members.resize(3);
        band->band_data().permutable = true;
        auto leaf = band->add_child(ScheduleNodeType::LEAF);

        assert(root->children().size() == 1);
        assert(root->children()[0]->type() == ScheduleNodeType::BAND);
        std::cout << "[PASS] ScheduleTree construction\n";
    }

    // Test 2: Tiling
    {
        auto band = ScheduleTree::create(ScheduleNodeType::BAND);
        band->band_data().members.resize(3);
        band->band_data().permutable = true;
        band->add_child(ScheduleNodeType::LEAF);

        TileConfig tile({128, 128, 64}, {32, 32, 32});
        HardwareTarget target = HardwareTarget::Generic();
        auto result = apply_rectangular_tiling(band, tile, target);

        assert(!result.grid_dims.empty());
        assert(!result.block_dims.empty());
        std::cout << "[PASS] Rectangular tiling\n";
    }

    // Test 3: Hardware-aligned tile generation
    {
        HardwareTarget target = HardwareTarget::H100();
        auto tiles = generate_hardware_aligned_tiles(3, target, 256);
        assert(!tiles.empty());
        // All tiles should have SRAM footprint within budget
        for (const auto& t : tiles) {
            assert(t.sram_footprint() <= static_cast<size_t>(
                target.max_sram_bytes + target.max_sram_bytes / 4));  // Allow 25% overhead
        }
        std::cout << "[PASS] Hardware-aligned tile generation (" << tiles.size() << " configs)\n";
    }

    // Test 4: GPU mapping
    {
        auto band = ScheduleTree::create(ScheduleNodeType::BAND);
        band->band_data().members.resize(2);
        band->band_data().permutable = true;
        band->band_data().members[0].parallel = true;
        band->band_data().members[1].parallel = true;
        band->add_child(ScheduleNodeType::LEAF);

        HardwareTarget target = HardwareTarget::H100();
        auto mapping = MapToGPU(band, target);
        assert(!mapping.grid_dims.empty());
        assert(!mapping.block_dims.empty());
        std::cout << "[PASS] GPU mapping\n";
    }

    std::cout << "All schedule tests passed!\n";
    return 0;
}
