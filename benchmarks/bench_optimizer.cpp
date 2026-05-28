// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/superoptimizer.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>
#include <chrono>

using namespace symplex::optimizer;
using namespace symplex::polyhedral;
using namespace symplex::hardware;

int main() {
    HardwareTarget target = HardwareTarget::H100();
    Superoptimizer opt(target);

    struct BenchCase {
        std::string name;
        int64_t M, N, K;
    };

    std::vector<BenchCase> cases = {
        {"small_128", 128, 128, 128},
        {"medium_1k", 1024, 1024, 512},
        {"large_4k", 4096, 4096, 1024},
        {"transformer_8k", 8192, 8192, 2048},
    };

    for (const auto& bc : cases) {
        auto ispace = make_matmul_iteration_space(bc.M, bc.N, bc.K);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = opt.optimize(ispace, 512);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << bc.name << ": " << ms << "ms, "
                  << "speedup=" << result.speedup_vs_naive << "x, "
                  << "latency=" << result.estimated_latency_ns << "ns\n";
    }

    return 0;
}
