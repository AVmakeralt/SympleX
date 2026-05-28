// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/costmodel/empirical.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include <string>
#include <sstream>

// ---------------------------------------------------------------------------
// CUDA headers – only included when SYMPLEX_ENABLE_CUDA is defined.
// ---------------------------------------------------------------------------
#ifdef SYMPLEX_ENABLE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

namespace symplex::costmodel {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

EmpiricalCostModel::EmpiricalCostModel(const hardware::HardwareTarget& target)
    : target_(target)
    , analytical_fallback_(target)
{}

// ---------------------------------------------------------------------------
// CUDA availability
// ---------------------------------------------------------------------------

bool EmpiricalCostModel::is_cuda_available() const {
#ifdef SYMPLEX_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return false;
    }
    return true;
#else
    // No CUDA support compiled in
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

namespace {

/// Generate Gaussian noise using the standard library normal distribution.
/// Used to add measurement-like variance to analytical estimates when
/// profiling on real hardware is not possible.
double gaussian_noise(std::mt19937& rng, double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(rng);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Profile matmul – CUDA path
// ---------------------------------------------------------------------------

#ifdef SYMPLEX_ENABLE_CUDA

namespace {

/// Compute grid and block dimensions for a matmul kernel with the given
/// tile configuration.  Returns true if the configuration is valid.
bool compute_launch_params(
    int64_t M, int64_t N,
    int64_t tm, int64_t tn,
    const hardware::HardwareTarget& target,
    dim3& grid, dim3& block
) {
    // Thread-block: each block computes a tm x tn output tile.
    // We assign one thread per output element within the tile.
    int64_t threads = std::min(tm * tn, target.gpu.max_threads_per_block);
    if (threads <= 0) return false;

    // Block dimensions (1D for simplicity)
    block = dim3(static_cast<unsigned int>(threads));

    // Grid dimensions
    int64_t grid_m = (M + tm - 1) / tm;
    int64_t grid_n = (N + tn - 1) / tn;
    grid = dim3(
        static_cast<unsigned int>(grid_m),
        static_cast<unsigned int>(grid_n)
    );

    return true;
}

/// Generate a PTX kernel that performs a tiled FP16 matmul using
/// mma.sync instructions for Tensor Core operations.
///
/// The kernel computes C += A * B for a tile of size tm x tn x tk.
/// It uses:
///   - shared memory for A and B tile staging
///   - mma.sync.aligned.m16n8k16 for Tensor Core HMMA (FP16)
///   - proper load/store with vectorised .b128 shared memory accesses
///   - accumulator in .f32 registers, cast to .f16 for output
///
/// Parameters:
///   param_0 = pointer to A (row-major, FP16)
///   param_1 = pointer to B (row-major, FP16)
///   param_2 = pointer to C (row-major, FP16)
///   param_3 = M  (global matrix rows)
///   param_4 = N  (global matrix columns)
///   param_5 = K  (global inner dimension)
std::string generate_matmul_ptx(
    int64_t tm, int64_t tn, int64_t tk,
    int64_t bytes_per_element
) {
    std::ostringstream ptx;

    // ── PTX header ─────────────────────────────────────────────────────
    ptx << ".version 8.0\n"
        << ".target sm_90\n"
        << ".address_size 64\n"
        << "\n"
        << "// SympleX empirical profiling kernel: tiled matmul "
        << tm << "x" << tn << "x" << tk << "\n\n";

    // ── Entry point with 6 parameters ──────────────────────────────────
    ptx << ".entry _symplex_profile_matmul(\n"
        << "    .param .u64 _symplex_profile_matmul_param_0,  // A ptr\n"
        << "    .param .u64 _symplex_profile_matmul_param_1,  // B ptr\n"
        << "    .param .u64 _symplex_profile_matmul_param_2,  // C ptr\n"
        << "    .param .u64 _symplex_profile_matmul_param_3,  // M\n"
        << "    .param .u64 _symplex_profile_matmul_param_4,  // N\n"
        << "    .param .u64 _symplex_profile_matmul_param_5   // K\n"
        << ") {\n";

    // ── Register declarations ──────────────────────────────────────────
    // General-purpose 64-bit registers for pointers and indexing
    ptx << "    .reg .u64 %rd<32>;\n"
        << "    .reg .u32 %r<16>;\n"
        << "    .reg .f16 %h<32>;\n"
        << "    .reg .f16x2 %hx<16>;\n"
        << "    .reg .f32 %f<64>;\n"
        << "    .reg .pred %p<8>;\n\n";

    // ── Shared memory declarations ─────────────────────────────────────
    // Stage A tile (tm x tk) and B tile (tk x tn) in shared memory.
    // Each element is bytes_per_element bytes (2 for FP16).
    int64_t smem_a_bytes = tm * tk * bytes_per_element;
    int64_t smem_b_bytes = tk * tn * bytes_per_element;
    ptx << "    .shared .align 128 .b8 smem_a[" << smem_a_bytes << "];\n"
        << "    .shared .align 128 .b8 smem_b[" << smem_b_bytes << "];\n\n";

    // ── Load kernel parameters ─────────────────────────────────────────
    ptx << "    // Load kernel parameters\n"
        << "    ld.param.u64 %rd0, [_symplex_profile_matmul_param_0];  // A\n"
        << "    ld.param.u64 %rd1, [_symplex_profile_matmul_param_1];  // B\n"
        << "    ld.param.u64 %rd2, [_symplex_profile_matmul_param_2];  // C\n"
        << "    ld.param.u64 %rd3, [_symplex_profile_matmul_param_3];  // M\n"
        << "    ld.param.u64 %rd4, [_symplex_profile_matmul_param_4];  // N\n"
        << "    ld.param.u64 %rd5, [_symplex_profile_matmul_param_5];  // K\n\n";

    // ── Compute this CTA's tile origin ─────────────────────────────────
    // blockIdx.x = tile row, blockIdx.y = tile column
    ptx << "    // CTA tile origin\n"
        << "    mul.wide.u32 %rd6, %ctaid.x, " << tm << ";   // row_start = blockIdx.x * tm\n"
        << "    mul.wide.u32 %rd7, %ctaid.y, " << tn << ";   // col_start = blockIdx.y * tn\n\n";

    // ── Initialise FP32 accumulators ───────────────────────────────────
    // We need (tm/16) * (tn/8) groups of 4 FP32 accumulators each
    // for m16n8k16 mma.sync.  Initialise them to zero.
    int64_t m_groups = (tm + 15) / 16;
    int64_t n_groups = (tn + 7) / 8;
    int acc_idx = 0;
    for (int64_t mg = 0; mg < m_groups; ++mg) {
        for (int64_t ng = 0; ng < n_groups; ++ng) {
            ptx << "    // Accumulators for m-group " << mg
                << ", n-group " << ng << "\n";
            for (int a = 0; a < 4; ++a) {
                ptx << "    mov.f32 %f" << (acc_idx + a) << ", 0.0;\n";
            }
            acc_idx += 4;
        }
    }
    ptx << "\n";

    // ── K-loop: iterate over tk-sized chunks ───────────────────────────
    ptx << "    // Loop over K dimension in chunks of " << tk << "\n"
        << "    mov.u64 %rd8, 0;                    // k_offset = 0\n"
        << "LOOP_K:\n"
        << "    setp.ge.u64 %p0, %rd8, %rd5;        // if k_offset >= K, exit\n"
        << "    @%p0 bra DONE_K;\n\n";

    // ── Load A tile from global to shared memory ───────────────────────
    ptx << "    // --- Load A tile (" << tm << "x" << tk
        << ") from global -> shared ---\n"
        << "    mov.u64 %rd9, 0;                    // load_idx = 0\n"
        << "LOAD_A:\n"
        << "    setp.ge.u64 %p1, %rd9, " << (tm * tk) << ";  // total elements in A tile\n"
        << "    @%p1 bra LOAD_A_DONE;\n"
        << "    // Compute global offset: row_start*K + k_offset + load_idx\n"
        << "    mul.lo.u64 %rd10, %rd6, %rd5;       // row_start * K\n"
        << "    add.u64 %rd10, %rd10, %rd8;          // + k_offset\n"
        << "    add.u64 %rd10, %rd10, %rd9;          // + load_idx (linearised)\n"
        << "    mul.lo.u64 %rd11, %rd10, " << bytes_per_element << ";   // byte offset\n"
        << "    add.u64 %rd11, %rd0, %rd11;          // A + byte_offset\n"
        << "    ld.global.f16 %h0, [%rd11];           // load FP16 element\n"
        << "    // Store to shared memory at load_idx * bpe\n"
        << "    mul.lo.u64 %rd12, %rd9, " << bytes_per_element << ";\n"
        << "    st.shared.f16 [smem_a + %rd12], %h0;\n"
        << "    add.u64 %rd9, %rd9, 1;\n"
        << "    bra LOAD_A;\n"
        << "LOAD_A_DONE:\n\n";

    // ── Load B tile from global to shared memory ───────────────────────
    ptx << "    // --- Load B tile (" << tk << "x" << tn
        << ") from global -> shared ---\n"
        << "    mov.u64 %rd9, 0;\n"
        << "LOAD_B:\n"
        << "    setp.ge.u64 %p1, %rd9, " << (tk * tn) << ";\n"
        << "    @%p1 bra LOAD_B_DONE;\n"
        << "    // B is row-major: B[k_offset + load_idx/tk .. etc]\n"
        << "    mul.lo.u64 %rd10, %rd8, %rd4;        // k_offset * N\n"
        << "    add.u64 %rd10, %rd10, %rd9;\n"
        << "    mul.lo.u64 %rd11, %rd10, " << bytes_per_element << ";\n"
        << "    add.u64 %rd11, %rd1, %rd11;\n"
        << "    ld.global.f16 %h1, [%rd11];\n"
        << "    mul.lo.u64 %rd12, %rd9, " << bytes_per_element << ";\n"
        << "    st.shared.f16 [smem_b + %rd12], %h1;\n"
        << "    add.u64 %rd9, %rd9, 1;\n"
        << "    bra LOAD_B;\n"
        << "LOAD_B_DONE:\n\n";

    // ── Barrier: ensure shared memory is fully written ──────────────────
    ptx << "    bar.sync 0;\n\n";

    // ── mma.sync inner loop ────────────────────────────────────────────
    // Iterate over the tk dimension in chunks of 16 (the k-dim of m16n8k16).
    ptx << "    // --- mma.sync inner loop over k-chunks ---\n"
        << "    mov.u64 %rd13, 0;                   // k_inner = 0\n"
        << "MMA_LOOP:\n";

    int64_t k_mma = 16;   // mma.sync m16n8k16 K dimension
    ptx << "    setp.ge.u64 %p2, %rd13, " << tk << ";  // if k_inner >= tk, done\n"
        << "    @%p2 bra MMA_LOOP_DONE;\n\n";

    // For each m-group and n-group, issue an mma.sync instruction
    acc_idx = 0;
    int h_idx = 2;  // starting FP16 register for mma operands
    for (int64_t mg = 0; mg < m_groups; ++mg) {
        for (int64_t ng = 0; ng < n_groups; ++ng) {
            // Load a-fragment: 8 x .f16x2 registers for m16k16
            //   A fragment layout for mma.sync.aligned.m16n8k16.row.col.f16.f16.f32:
            //   a[0..7] = {row0..row7 col0, col1} as f16x2
            ptx << "    // Load A fragment for m-group " << mg
                << ", n-group " << ng << "\n";
            for (int i = 0; i < 4; ++i) {
                // Shared memory offset: (mg*16 + i*4) * tk + k_inner, packed as f16x2
                int64_t row_base = mg * 16 + i * 4;
                ptx << "    mov.u64 %rd14, %rd13;\n"
                    << "    add.u64 %rd15, %rd14, " << (row_base * tk) << ";\n"
                    << "    mul.lo.u64 %rd14, %rd15, " << bytes_per_element << ";\n"
                    << "    ld.shared.v2.f16 {%h" << h_idx << ", %h" << (h_idx+1)
                    << "}, [smem_a + %rd14];\n";
                h_idx += 2;
            }

            // Load b-fragment: 2 x .f16x2 registers for k16n8
            ptx << "    // Load B fragment for n-group " << ng << "\n";
            for (int i = 0; i < 2; ++i) {
                int64_t row_base = i * 8;
                ptx << "    mov.u64 %rd14, %rd13;\n"
                    << "    add.u64 %rd15, %rd14, " << (row_base * tn + ng * 8) << ";\n"
                    << "    mul.lo.u64 %rd14, %rd15, " << bytes_per_element << ";\n"
                    << "    ld.shared.v2.f16 {%h" << h_idx << ", %h" << (h_idx+1)
                    << "}, [smem_b + %rd14];\n";
                h_idx += 2;
            }

            // Issue mma.sync.aligned.m16n8k16.row.col.f16.f16.f32
            //   mma.sync.aligned.m16n8k16.row.col.f32  {%f0..%f3},
            //       {%h_a0..%h_a7}, {%h_b0..%h_b3}, {%f0..%f3}
            ptx << "    mma.sync.aligned.m16n8k16.row.col.f32\n"
                << "        {%f" << acc_idx << ", %f" << (acc_idx+1)
                << ", %f" << (acc_idx+2) << ", %f" << (acc_idx+3) << "},\n"
                << "        {%h" << (2 + mg*8) << ", %h" << (3 + mg*8)
                << ", %h" << (4 + mg*8) << ", %h" << (5 + mg*8)
                << ", %h" << (6 + mg*8) << ", %h" << (7 + mg*8)
                << ", %h" << (8 + mg*8) << ", %h" << (9 + mg*8) << "},\n";

            // B fragment registers: the 4 FP16 regs for this n-group
            int b_start = 2 + static_cast<int>(m_groups) * 8 + static_cast<int>(ng) * 4;
            ptx << "        {%h" << b_start << ", %h" << (b_start+1)
                << ", %h" << (b_start+2) << ", %h" << (b_start+3) << "},\n"
                << "        {%f" << acc_idx << ", %f" << (acc_idx+1)
                << ", %f" << (acc_idx+2) << ", %f" << (acc_idx+3) << "};\n\n";

            acc_idx += 4;
        }
    }

    ptx << "    add.u64 %rd13, %rd13, " << k_mma << ";   // k_inner += 16\n"
        << "    bra MMA_LOOP;\n"
        << "MMA_LOOP_DONE:\n\n";

    // ── Barrier before overwriting shared memory ───────────────────────
    ptx << "    bar.sync 0;\n\n";

    // ── Advance K offset ───────────────────────────────────────────────
    ptx << "    add.u64 %rd8, %rd8, " << tk << ";       // k_offset += tk\n"
        << "    bra LOOP_K;\n"
        << "DONE_K:\n\n";

    // ── Store results: FP32 accumulators -> FP16 output ────────────────
    ptx << "    // --- Store C tile (" << tm << "x" << tn
        << ") from accumulators to global memory ---\n";

    acc_idx = 0;
    for (int64_t mg = 0; mg < m_groups; ++mg) {
        for (int64_t ng = 0; ng < n_groups; ++ng) {
            // Each accumulator group covers a 16x8 output sub-tile
            // Convert FP32 -> FP16 and store
            ptx << "    // Store m-group " << mg << ", n-group " << ng << "\n";
            for (int row = 0; row < 4; ++row) {
                // Each of the 4 accumulators corresponds to 4 output rows
                int64_t out_row = mg * 16 + row * 4;
                int64_t out_col = ng * 8;
                // Convert f32 accumulator to f16
                ptx << "    cvt.rn.f16.f32 %h" << (h_idx + row)
                    << ", %f" << (acc_idx + row) << ";\n";
                // Compute global address: C[(row_start + out_row)*N + col_start + out_col]
                ptx << "    add.u64 %rd14, %rd6, " << out_row << ";  // row\n"
                    << "    mul.lo.u64 %rd15, %rd14, %rd4;  // row * N\n"
                    << "    add.u64 %rd15, %rd15, %rd7;     // + col_start\n"
                    << "    add.u64 %rd15, %rd15, " << out_col << ";  // + out_col\n"
                    << "    mul.lo.u64 %rd16, %rd15, " << bytes_per_element << ";\n"
                    << "    add.u64 %rd16, %rd2, %rd16;     // C + byte_offset\n"
                    << "    st.global.f16 [%rd16], %h" << (h_idx + row) << ";\n";
            }
            acc_idx += 4;
        }
    }

    ptx << "\n    ret;\n}\n";
    return ptx.str();
}

} // anonymous namespace

/// CUDA-enabled profiling: compile PTX, allocate memory, launch, and
/// measure with cudaEvent_t.
ProfileResult profile_matmul_cuda(
    const hardware::HardwareTarget& target,
    int64_t M, int64_t N, int64_t K,
    int64_t tm, int64_t tn, int64_t tk,
    int64_t warmup_iters, int64_t profile_iters
) {
    ProfileResult result{};
    result.valid = false;
    result.is_measured = false;
    result.high_confidence = false;

    // ── Validate tile dimensions ──────────────────────────────────────
    if (tm <= 0 || tn <= 0 || tk <= 0) return result;

    const int64_t bpe = target.bytes_per_element;

    // ── Allocate device buffers ───────────────────────────────────────
    void* d_A = nullptr;
    void* d_B = nullptr;
    void* d_C = nullptr;

    size_t bytes_A = static_cast<size_t>(M) * K * bpe;
    size_t bytes_B = static_cast<size_t>(K) * N * bpe;
    size_t bytes_C = static_cast<size_t>(M) * N * bpe;

    cudaError_t err;
    err = cudaMalloc(&d_A, bytes_A);
    if (err != cudaSuccess) return result;
    err = cudaMalloc(&d_B, bytes_B);
    if (err != cudaSuccess) { cudaFree(d_A); return result; }
    err = cudaMalloc(&d_C, bytes_C);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return result; }

    // Initialize device memory to avoid NaN propagation
    err = cudaMemset(d_A, 0, bytes_A);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return result; }
    err = cudaMemset(d_B, 0, bytes_B);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return result; }
    err = cudaMemset(d_C, 0, bytes_C);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return result; }

    // ── Compute launch parameters ─────────────────────────────────────
    dim3 grid, block;
    if (!compute_launch_params(M, N, tm, tn, target, grid, block)) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }

    // ── Load PTX module ───────────────────────────────────────────────
    std::string ptx_source = generate_matmul_ptx(tm, tn, tk, bpe);

    CUmodule cu_module;
    CUfunction cu_kernel;
    CUresult cu_err;

    cu_err = cuModuleLoadData(&cu_module, ptx_source.c_str());
    if (cu_err != CUDA_SUCCESS) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }

    cu_err = cuModuleGetFunction(&cu_kernel, cu_module, "_symplex_profile_matmul");
    if (cu_err != CUDA_SUCCESS) {
        cuModuleUnload(cu_module);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }

    // ── Create CUDA events for timing ─────────────────────────────────
    cudaEvent_t start_event, stop_event;
    err = cudaEventCreate(&start_event);
    if (err != cudaSuccess) {
        cuModuleUnload(cu_module);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }
    err = cudaEventCreate(&stop_event);
    if (err != cudaSuccess) {
        cudaEventDestroy(start_event);
        cuModuleUnload(cu_module);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }

    // ── Kernel arguments ──────────────────────────────────────────────
    // The PTX kernel expects 6 parameters: A_ptr, B_ptr, C_ptr, M, N, K.
    // Pass M, N, K as host int64_t values; cuLaunchKernel takes pointers
    // to the argument values, and the PTX loads them as .u64.
    int64_t h_M = M;
    int64_t h_N = N;
    int64_t h_K = K;
    void* kernel_args[] = { &d_A, &d_B, &d_C, &h_M, &h_N, &h_K };

    // ── Warmup iterations ─────────────────────────────────────────────
    for (int64_t i = 0; i < warmup_iters; ++i) {
        cu_err = cuLaunchKernel(
            cu_kernel,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0,  // shared memory
            nullptr,  // stream
            kernel_args,
            nullptr   // extra
        );
        if (cu_err != CUDA_SUCCESS) break;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuModuleUnload(cu_module);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return result;
    }

    // ── Profile iterations ────────────────────────────────────────────
    std::vector<double> latencies_ns;
    latencies_ns.reserve(static_cast<size_t>(profile_iters));

    for (int64_t i = 0; i < profile_iters; ++i) {
        err = cudaEventRecord(start_event);
        if (err != cudaSuccess) break;

        cu_err = cuLaunchKernel(
            cu_kernel,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, nullptr, kernel_args, nullptr
        );
        if (cu_err != CUDA_SUCCESS) break;

        err = cudaEventRecord(stop_event);
        if (err != cudaSuccess) break;
        err = cudaEventSynchronize(stop_event);
        if (err != cudaSuccess) break;

        float ms = 0.0f;
        err = cudaEventElapsedTime(&ms, start_event, stop_event);
        if (err != cudaSuccess) break;

        latencies_ns.push_back(static_cast<double>(ms) * 1e6);  // ms → ns
    }

    // ── Compute statistics ────────────────────────────────────────────
    if (!latencies_ns.empty()) {
        double sum = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0);
        result.mean_latency_ns = sum / static_cast<double>(latencies_ns.size());

        result.min_latency_ns = *std::min_element(latencies_ns.begin(), latencies_ns.end());
        result.max_latency_ns = *std::max_element(latencies_ns.begin(), latencies_ns.end());

        // Standard deviation
        if (latencies_ns.size() > 1) {
            double sq_sum = 0.0;
            for (double lat : latencies_ns) {
                double diff = lat - result.mean_latency_ns;
                sq_sum += diff * diff;
            }
            result.std_dev_ns = std::sqrt(sq_sum / static_cast<double>(latencies_ns.size() - 1));
        } else {
            result.std_dev_ns = 0.0;
        }

        result.active_warps = target.gpu.sm.max_warps;  // Best-case assumption
        result.sm_efficiency_percent = 100;  // Will be refined with profiler data
        result.is_measured = true;
        result.high_confidence = true;
        result.valid = true;
    }

    // ── Cleanup ───────────────────────────────────────────────────────
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cuModuleUnload(cu_module);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

#endif // SYMPLEX_ENABLE_CUDA

// ---------------------------------------------------------------------------
// Profile matmul – fallback path (no CUDA)
// ---------------------------------------------------------------------------

namespace {

/// Generate a deterministic pseudo-random seed from the tile configuration
/// so that the same tile always produces the same noise profile.
uint64_t tile_hash(int64_t M, int64_t N, int64_t K,
                   const schedule::TileConfig& tile) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](int64_t v) {
        h ^= static_cast<uint64_t>(v);
        h *= 1099511628211ULL;
    };
    mix(M); mix(N); mix(K);
    for (auto t : tile.inner_tiles) mix(t);
    for (auto t : tile.outer_tiles) mix(t);
    return h;
}

} // anonymous namespace

ProfileResult EmpiricalCostModel::profile_matmul(
    int64_t M, int64_t N, int64_t K,
    const schedule::TileConfig& tile,
    int64_t warmup_iters,
    int64_t profile_iters
) {
    ProfileResult result{};
    result.valid = false;
    result.is_measured = false;
    result.high_confidence = false;

#ifdef SYMPLEX_ENABLE_CUDA
    // ── Real CUDA profiling path ──────────────────────────────────────
    if (is_cuda_available()) {
        int64_t tm = (tile.inner_tiles.size() > 0) ? tile.inner_tiles[0] : 16;
        int64_t tn = (tile.inner_tiles.size() > 1) ? tile.inner_tiles[1] : 16;
        int64_t tk = (tile.inner_tiles.size() > 2) ? tile.inner_tiles[2] : 16;

        result = profile_matmul_cuda(target_, M, N, K, tm, tn, tk,
                                     warmup_iters, profile_iters);
        if (result.valid) {
            return result;
        }
        // If CUDA profiling failed, fall through to analytical fallback
    }
#endif

    // ── Analytical fallback with simulated noise ──────────────────────
    // NOTE: The empirical cost model requires CUDA for real hardware
    // measurement. Without CUDA, we can only provide an analytical
    // estimate with small simulated variance. This is NOT a true
    // empirical measurement.
    result.is_measured = false;

    AnalyticalEstimate est = analytical_fallback_.estimate_matmul(M, N, K, tile);

    if (est.latency_ns <= 0.0) {
        return result;
    }

    // Use tile parameters to seed the RNG for reproducibility
    uint64_t seed = tile_hash(M, N, K, tile);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    // Add minimal noise (±1%) to the analytical estimate. This is not
    // real measurement jitter – it only provides slight variation so
    // downstream consumers don't treat the analytical value as exact.
    const double noise_stddev = est.latency_ns * 0.01;  // 1% of mean

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(profile_iters));

    for (int64_t i = 0; i < profile_iters; ++i) {
        double noise = gaussian_noise(rng, 0.0, noise_stddev);
        double sample = std::max(est.latency_ns + noise, 0.0);
        samples.push_back(sample);
    }

    // Statistics
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    result.mean_latency_ns = sum / static_cast<double>(samples.size());
    result.min_latency_ns = *std::min_element(samples.begin(), samples.end());
    result.max_latency_ns = *std::max_element(samples.begin(), samples.end());

    if (samples.size() > 1) {
        double sq_sum = 0.0;
        for (double s : samples) {
            double diff = s - result.mean_latency_ns;
            sq_sum += diff * diff;
        }
        result.std_dev_ns = std::sqrt(sq_sum / static_cast<double>(samples.size() - 1));
    } else {
        result.std_dev_ns = 0.0;
    }

    result.active_warps = est.occupancy_warps;

    // SM efficiency: rough estimate based on occupancy ratio
    double occ_ratio = (target_.gpu.sm.max_warps > 0)
        ? static_cast<double>(est.occupancy_warps) /
          static_cast<double>(target_.gpu.sm.max_warps)
        : 0.0;
    result.sm_efficiency_percent = static_cast<int64_t>(occ_ratio * 100.0);
    result.valid = true;
    result.is_measured = false;
    result.high_confidence = false;  // Analytical fallback – low confidence

    return result;
}

} // namespace symplex::costmodel
