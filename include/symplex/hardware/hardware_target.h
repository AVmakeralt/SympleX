// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

namespace symplex::hardware {

/// TensorCoreSpec: describes the native MMA (Matrix Multiply-Accumulate)
/// dimensions of a GPU's Tensor Core.
struct TensorCoreSpec {
    int64_t m = 16;       // MMA M dimension
    int64_t n = 8;        // MMA N dimension (Hopper uses 16x8x16 for FP16)
    int64_t k = 16;       // MMA K dimension
    std::string dtype = "fp16";  // Data type: fp16, bf16, fp8, int8

    /// Number of FMA operations per Tensor Core instruction.
    [[nodiscard]] int64_t ops_per_mma() const { return m * n * k; }

    /// Clock cycles per MMA instruction (architecture-dependent).
    int64_t cycles_per_mma = 1;

    std::string to_string() const {
        return "TensorCore{" + std::to_string(m) + "x" +
               std::to_string(n) + "x" + std::to_string(k) + " " + dtype + "}";
    }
};

/// SMProfile: profile of a single Streaming Multiprocessor.
struct SMProfile {
    int64_t max_warps = 48;             // Maximum concurrent warps
    int64_t max_threads = 1536;         // Maximum concurrent threads
    int64_t shared_mem_bytes = 228 * 1024;  // Shared memory (SRAM) in bytes
    int64_t register_file_bytes = 256 * 1024;  // Register file size
    int64_t num_tensor_cores = 4;       // Tensor Cores per SM
    int64_t num_fp32_cores = 128;       // FP32 CUDA cores per SM
    int64_t max_registers_per_thread = 255;

    /// Compute occupancy: how many warps can be active given
    /// register and shared memory constraints.
    [[nodiscard]] int64_t compute_occupancy(
        int64_t registers_per_thread,
        int64_t shared_mem_per_block
    ) const {
        // Register-limited warps
        int64_t threads_per_warp = 32;
        int64_t regs_per_warp = registers_per_thread * threads_per_warp;
        int64_t total_reg_capacity = register_file_bytes / 4;  // 4 bytes per register
        int64_t reg_limited_warps = total_reg_capacity / regs_per_warp;

        // Shared-memory-limited blocks (assuming 1 block per SM for simplicity)
        int64_t smem_blocks = (shared_mem_per_block <= shared_mem_bytes) ? 1 : 0;
        int64_t smem_limited_warps = smem_blocks * max_warps;

        int64_t limited = std::min({max_warps, reg_limited_warps, smem_limited_warps});
        return std::max(limited, int64_t(0));
    }

    std::string to_string() const {
        return "SM{warps=" + std::to_string(max_warps) +
               ", threads=" + std::to_string(max_threads) +
               ", sram=" + std::to_string(shared_mem_bytes / 1024) + "KB}";
    }
};

/// MemorySpec: specifications of the GPU memory hierarchy.
struct MemorySpec {
    int64_t global_bw_gbps = 3350;      // Global memory (HBM) bandwidth in GB/s
    int64_t shared_bw_gbps = 19000;     // Shared memory bandwidth in GB/s
    int64_t global_capacity_gb = 80;    // HBM capacity in GB
    int64_t l2_cache_kb = 51200;        // L2 cache size in KB

    /// Latency in cycles (approximate).
    int64_t global_latency_cycles = 300;
    int64_t shared_latency_cycles = 30;
    int64_t register_latency_cycles = 1;

    std::string to_string() const {
        return "Mem{hbm_bw=" + std::to_string(global_bw_gbps) +
               "GB/s, sram_bw=" + std::to_string(shared_bw_gbps) + "GB/s}";
    }
};

/// GPUTopology: the complete physical layout of a GPU.
struct GPUTopology {
    int64_t num_sms = 132;             // Number of Streaming Multiprocessors
    int64_t num_tensor_cores = 132 * 4; // Total = num_sms * sm.num_tensor_cores
    int64_t warp_size = 32;
    int64_t max_grid_x = 2147483647;
    int64_t max_grid_y = 65535;
    int64_t max_grid_z = 65535;
    int64_t max_block_x = 1024;
    int64_t max_block_y = 1024;
    int64_t max_block_z = 64;
    int64_t max_threads_per_block = 1024;

    SMProfile sm;
    TensorCoreSpec tensor_core;
    MemorySpec memory;

    GPUTopology() = default;

    /// Total FLOPS for FP16 Tensor Core operations.
    [[nodiscard]] double peak_tflops_fp16() const {
        // Each Tensor Core does m*n*k FMA ops per clock
        // FMA = 2 ops (multiply + add)
        double clock_ghz = 1.83;  // Default boost clock
        return num_tensor_cores * tensor_core.ops_per_mma() * 2.0 *
               clock_ghz * 1e-3;  // TFLOPS
    }

    /// Total FLOPS for FP32 CUDA core operations.
    [[nodiscard]] double peak_tflops_fp32() const {
        double clock_ghz = 1.83;
        return num_sms * sm.num_fp32_cores * 2.0 * clock_ghz * 1e-3;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "GPU{sms=" << num_sms
            << ", " << tensor_core.to_string()
            << ", " << sm.to_string()
            << ", " << memory.to_string()
            << ", peak_fp16=" << peak_tflops_fp16() << " TFLOPS}";
        return oss.str();
    }
};

/// HardwareTarget: the complete hardware specification used by the
/// superoptimizer for cost modeling and schedule generation.
struct HardwareTarget {
    GPUTopology gpu;

    // Derived fields for convenience
    int64_t max_sram_bytes = gpu.sm.shared_mem_bytes;
    int64_t bytes_per_element = 2;      // FP16

    /// Alignment requirement for memory allocations (128 bytes for HBM bus width).
    int64_t memory_alignment = 128;

    /// Number of double-buffer stages (2 = ping-pong, 3+ = software pipeline).
    int64_t pipeline_stages = 2;

    /// Whether TMA (Tensor Memory Accelerator) is available.
    bool has_tma = true;

    /// Whether async copy (cp.async) is available.
    bool has_async_copy = true;

    /// Compute the operational intensity threshold (FLOPS/byte) above which
    /// a kernel is compute-bound rather than memory-bound.
    [[nodiscard]] double roofline_intensity_threshold() const {
        // Operational intensity at the roofline knee:
        // I_peak = Peak_FLOPS / Peak_BW
        double peak_flops = peak_flops_fp16();
        double peak_bw = gpu.memory.global_bw_gbps * 1e9;  // bytes/s
        return peak_flops / peak_bw;  // FLOPS/byte
    }

    /// Peak FP16 FLOPS.
    [[nodiscard]] double peak_flops_fp16() const {
        return gpu.peak_tflops_fp16() * 1e12;
    }

    /// Check if a given tile configuration's operational intensity
    /// exceeds the roofline threshold (i.e., is compute-bound).
    [[nodiscard]] bool is_compute_bound(
        int64_t compute_ops,
        int64_t bytes_moved
    ) const {
        if (bytes_moved <= 0) return true;
        double intensity = static_cast<double>(compute_ops) /
                          static_cast<double>(bytes_moved);
        return intensity >= roofline_intensity_threshold();
    }

    /// Validate a tile configuration against hardware constraints.
    [[nodiscard]] bool validate_tile_config(
        const std::vector<int64_t>& inner_tiles,
        int64_t registers_per_thread
    ) const {
        // Check thread count
        // Kernel launch geometry is implementation-dependent. We use a
        // conservative heuristic of one warp per CTA unless overridden by
        // downstream codegen policy.
        int64_t total_threads = gpu.warp_size;
        if (total_threads > gpu.max_threads_per_block) return false;

        // Check SRAM capacity
        int64_t sram_needed = 0;
        for (auto t : inner_tiles) sram_needed += t * bytes_per_element;
        sram_needed *= pipeline_stages;  // Double-buffer
        if (sram_needed > max_sram_bytes) return false;

        // Check occupancy
        int64_t smem_per_block = sram_needed;
        int64_t occupancy = gpu.sm.compute_occupancy(
            registers_per_thread, smem_per_block
        );
        if (occupancy <= 0) return false;

        return true;
    }

    std::string to_string() const {
        return gpu.to_string() + " [sram=" +
               std::to_string(max_sram_bytes / 1024) + "KB, bpe=" +
               std::to_string(bytes_per_element) + ", tma=" +
               (has_tma ? "yes" : "no") + "]";
    }

    /// Factory: create a target for NVIDIA H100 (Hopper).
    static HardwareTarget H100() {
        HardwareTarget t;
        t.gpu.num_sms = 132;
        t.gpu.warp_size = 32;
        t.gpu.sm.max_warps = 48;
        t.gpu.sm.max_threads = 1536;
        t.gpu.sm.shared_mem_bytes = 228 * 1024;
        t.gpu.sm.register_file_bytes = 256 * 1024;
        t.gpu.sm.num_tensor_cores = 4;
        t.gpu.sm.num_fp32_cores = 128;
        t.gpu.sm.max_registers_per_thread = 255;
        t.gpu.tensor_core.m = 16;
        t.gpu.tensor_core.n = 8;
        t.gpu.tensor_core.k = 16;
        t.gpu.tensor_core.dtype = "fp16";
        t.gpu.memory.global_bw_gbps = 3350;
        t.gpu.memory.shared_bw_gbps = 19000;
        t.gpu.memory.global_capacity_gb = 80;
        t.gpu.memory.l2_cache_kb = 51200;
        t.gpu.num_tensor_cores = t.gpu.num_sms * t.gpu.sm.num_tensor_cores;
        t.max_sram_bytes = t.gpu.sm.shared_mem_bytes;
        t.has_tma = true;
        t.has_async_copy = true;
        t.pipeline_stages = 2;
        return t;
    }

    /// Factory: create a target for NVIDIA A100 (Ampere).
    static HardwareTarget A100() {
        HardwareTarget t;
        t.gpu.num_sms = 108;
        t.gpu.warp_size = 32;
        t.gpu.sm.max_warps = 48;
        t.gpu.sm.max_threads = 1536;
        t.gpu.sm.shared_mem_bytes = 164 * 1024;
        t.gpu.sm.register_file_bytes = 256 * 1024;
        t.gpu.sm.num_tensor_cores = 4;
        t.gpu.sm.num_fp32_cores = 64;
        t.gpu.sm.max_registers_per_thread = 255;
        t.gpu.tensor_core.m = 16;
        t.gpu.tensor_core.n = 8;
        t.gpu.tensor_core.k = 16;
        t.gpu.tensor_core.dtype = "fp16";
        t.gpu.memory.global_bw_gbps = 2039;
        t.gpu.memory.shared_bw_gbps = 19000;
        t.gpu.memory.global_capacity_gb = 80;
        t.gpu.memory.l2_cache_kb = 40960;
        t.gpu.num_tensor_cores = t.gpu.num_sms * t.gpu.sm.num_tensor_cores;
        t.max_sram_bytes = t.gpu.sm.shared_mem_bytes;
        t.has_tma = false;
        t.has_async_copy = true;
        t.pipeline_stages = 2;
        return t;
    }

    /// Factory: create a target for NVIDIA V100 (Volta).
    static HardwareTarget V100() {
        HardwareTarget t;
        t.gpu.num_sms = 80;
        t.gpu.warp_size = 32;
        t.gpu.sm.max_warps = 48;
        t.gpu.sm.max_threads = 1536;
        t.gpu.sm.shared_mem_bytes = 96 * 1024;
        t.gpu.sm.register_file_bytes = 256 * 1024;
        t.gpu.sm.num_tensor_cores = 8;   // Volta has 8 TC per SM
        t.gpu.sm.num_fp32_cores = 64;
        t.gpu.sm.max_registers_per_thread = 255;
        t.gpu.tensor_core.m = 8;        // Volta: 8x8x4 for FP16 HMMA
        t.gpu.tensor_core.n = 8;
        t.gpu.tensor_core.k = 4;
        t.gpu.tensor_core.dtype = "fp16";
        t.gpu.memory.global_bw_gbps = 900;
        t.gpu.memory.shared_bw_gbps = 14000;
        t.gpu.memory.global_capacity_gb = 32;
        t.gpu.memory.l2_cache_kb = 6144;
        t.gpu.num_tensor_cores = t.gpu.num_sms * t.gpu.sm.num_tensor_cores;
        t.max_sram_bytes = t.gpu.sm.shared_mem_bytes;
        t.has_tma = false;
        t.has_async_copy = false;
        t.pipeline_stages = 2;
        return t;
    }

    /// Factory: create a target for NVIDIA B200 (Blackwell).
    static HardwareTarget B200() {
        HardwareTarget t;
        t.gpu.num_sms = 160;
        t.gpu.warp_size = 32;
        t.gpu.sm.max_warps = 64;
        t.gpu.sm.max_threads = 2048;
        t.gpu.sm.shared_mem_bytes = 304 * 1024;
        t.gpu.sm.register_file_bytes = 512 * 1024;
        t.gpu.sm.num_tensor_cores = 8;
        t.gpu.sm.num_fp32_cores = 128;
        t.gpu.sm.max_registers_per_thread = 255;
        t.gpu.tensor_core.m = 16;
        t.gpu.tensor_core.n = 8;
        t.gpu.tensor_core.k = 32;     // Blackwell doubled K dimension for FP16
        t.gpu.tensor_core.dtype = "fp16";
        t.gpu.memory.global_bw_gbps = 8000;  // HBM3e
        t.gpu.memory.shared_bw_gbps = 38000;
        t.gpu.memory.global_capacity_gb = 192;
        t.gpu.memory.l2_cache_kb = 65536;
        t.gpu.num_tensor_cores = t.gpu.num_sms * t.gpu.sm.num_tensor_cores;
        t.max_sram_bytes = t.gpu.sm.shared_mem_bytes;
        t.has_tma = true;
        t.has_async_copy = true;
        t.pipeline_stages = 3;
        return t;
    }

    /// Factory: create a generic GPU target (conservative defaults).
    static HardwareTarget Generic() {
        HardwareTarget t;
        t.gpu.num_sms = 84;
        t.gpu.sm.max_warps = 48;
        t.gpu.sm.max_threads = 1536;
        t.gpu.sm.shared_mem_bytes = 164 * 1024;
        t.gpu.sm.register_file_bytes = 256 * 1024;
        t.gpu.sm.num_tensor_cores = 4;
        t.gpu.sm.num_fp32_cores = 128;
        t.gpu.tensor_core.m = 16;
        t.gpu.tensor_core.n = 8;
        t.gpu.tensor_core.k = 16;
        t.gpu.tensor_core.dtype = "fp16";
        t.gpu.memory.global_bw_gbps = 2000;
        t.gpu.memory.shared_bw_gbps = 19000;
        t.gpu.memory.global_capacity_gb = 40;
        t.gpu.memory.l2_cache_kb = 4096;
        t.gpu.num_tensor_cores = t.gpu.num_sms * t.gpu.sm.num_tensor_cores;
        t.max_sram_bytes = t.gpu.sm.shared_mem_bytes;
        t.has_tma = false;
        t.has_async_copy = true;
        t.pipeline_stages = 2;
        return t;
    }
};

} // namespace symplex::hardware
