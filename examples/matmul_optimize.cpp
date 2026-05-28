// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/training/compiler_pipeline.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>

using namespace symplex::training;
using namespace symplex::hardware;

int main() {
    std::cout << "=== SympleX Matmul Optimization Example ===\n\n";

    // Target: NVIDIA H100
    HardwareTarget target = HardwareTarget::H100();
    std::cout << "Hardware: " << target.to_string() << "\n\n";

    // Create the compiler pipeline
    CompilerPipeline pipeline(target);

    // Optimize a 4096x4096x2048 matrix multiplication
    int64_t M = 4096, N = 4096, K = 2048;
    std::cout << "Optimizing matmul: C[" << M << "x" << N << "] += A[" << M << "x" << K
              << "] * B[" << K << "x" << N << "]\n\n";

    auto result = pipeline.compile_matmul(M, N, K);

    if (result.valid) {
        std::cout << "Optimization successful!\n";
        std::cout << "  Kernel name: " << result.kernel_name << "\n";
        std::cout << "  Estimated latency: " << result.estimated_latency_ns << " ns\n";
        std::cout << "  Speedup vs naive: " << result.speedup_vs_naive << "x\n";
        std::cout << "  Grid dims: [" << result.grid_dims[0] << ", " << result.grid_dims[1] << "]\n";
        std::cout << "  Block dims: [" << result.block_dims[0] << ", " << result.block_dims[1] << "]\n";
        std::cout << "  PTX source size: " << result.ptx_source.size() << " bytes\n";

        // Print first 500 chars of PTX
        std::cout << "\n--- PTX Preview (first 500 chars) ---\n";
        std::cout << result.ptx_source.substr(0, 500) << "...\n";
    } else {
        std::cout << "Optimization failed: " << result.error << "\n";
    }

    return 0;
}
