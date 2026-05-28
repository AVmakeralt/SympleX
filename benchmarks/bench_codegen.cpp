// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/code_generator.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include <iostream>
#include <chrono>

using namespace symplex::codegen;
using namespace symplex::hardware;
using namespace symplex::schedule;

int main() {
    HardwareTarget target = HardwareTarget::H100();
    CodeGenerator gen(target);

    struct BenchCase {
        std::string name;
        int64_t M, N, K;
        TileConfig tile;
    };

    std::vector<BenchCase> cases = {
        {"small_tc16", 128, 128, 128, TileConfig({128, 128, 128}, {16, 16, 16})},
        {"medium_tc64", 1024, 1024, 512, TileConfig({1024, 1024, 512}, {64, 64, 32})},
        {"large_tc128", 4096, 4096, 1024, TileConfig({4096, 4096, 1024}, {128, 128, 64})},
    };

    for (const auto& bc : cases) {
        auto start = std::chrono::high_resolution_clock::now();
        auto kernel = gen.generate_matmul(bc.M, bc.N, bc.K, bc.tile);
        auto end = std::chrono::high_resolution_clock::now();

        auto us = std::chrono::duration<double, std::micro>(end - start).count();

        std::cout << bc.name << ": " << us << "us, "
                  << "ptx_size=" << kernel.ptx_source.size() << " bytes, "
                  << "valid=" << kernel.valid << "\n";
    }

    return 0;
}
