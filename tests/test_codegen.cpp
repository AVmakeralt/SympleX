// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/code_generator.h"
#include "symplex/codegen/wmma.h"
#include "symplex/codegen/swizzle.h"
#include "symplex/codegen/register_allocator.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include <iostream>
#include <cassert>

using namespace symplex::codegen;
using namespace symplex::hardware;
using namespace symplex::schedule;

int main() {
    HardwareTarget target = HardwareTarget::H100();

    // Test 1: WMMA generator
    {
        WMMAGenerator wmma(target);
        assert(wmma.config().m == 16);
        assert(wmma.config().n == 8);
        assert(wmma.config().k == 16);
        int64_t count = wmma.wmma_count(64, 64, 64);
        assert(count > 0);
        std::cout << "[PASS] WMMA generator (count=" << count << ")\n";
    }

    // Test 2: Swizzle generator
    {
        SwizzleGenerator swizzle(target);
        bool conflicts = swizzle.has_bank_conflicts(32, 32, 32);
        int bits = swizzle.find_optimal_swizzle_bits(32, 32, 32);
        std::cout << "[PASS] Swizzle generator (conflicts=" << conflicts
                  << ", optimal_bits=" << bits << ")\n";
    }

    // Test 3: Register allocator
    {
        RegisterAllocator reg(target);
        auto alloc = reg.allocate_for_matmul(64, 64, 32);
        std::cout << "[PASS] Register allocator (total=" << alloc.total_registers
                  << ", fits=" << alloc.fits_in_hw << ")\n";
    }

    // Test 4: Full code generation
    {
        CodeGenerator gen(target);
        TileConfig tile({128, 128, 64}, {64, 64, 32});
        auto kernel = gen.generate_matmul(1024, 1024, 512, tile);
        assert(kernel.valid);
        assert(!kernel.ptx_source.empty());
        std::cout << "[PASS] Code generation (PTX size=" << kernel.ptx_source.size()
                  << " bytes, grid=[" << kernel.grid_dims[0] << "," << kernel.grid_dims[1] << "])\n";
    }

    std::cout << "All codegen tests passed!\n";
    return 0;
}
