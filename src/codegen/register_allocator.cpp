// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/register_allocator.h"
#include "symplex/codegen/wmma.h"
#include <algorithm>
#include <cmath>

namespace symplex::codegen {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RegisterAllocator::RegisterAllocator(const hardware::HardwareTarget& target)
    : target_(target)
{}

// ---------------------------------------------------------------------------
// Per-fragment register computation
// ---------------------------------------------------------------------------

int64_t RegisterAllocator::compute_a_regs(int64_t tile_m, int64_t tile_k) const {
    const auto& tc = target_.gpu.tensor_core;
    if (tile_m % tc.m != 0 || tile_k % tc.k != 0) return 0;

    int64_t rep_m = tile_m / tc.m;
    int64_t rep_k = tile_k / tc.k;

    // Elements per thread per single MMA operation for A fragment
    // A fragment has m*k elements, distributed across 32 threads.
    int64_t a_elems = tc.m * tc.k;
    int64_t warp_size = target_.gpu.warp_size;
    int64_t elems_per_thread = (a_elems + warp_size - 1) / warp_size;

    // Bytes per element for the A dtype
    int64_t bpe = (tc.dtype == "fp16" || tc.dtype == "bf16") ? 2
                : (tc.dtype == "fp8" || tc.dtype == "int8")  ? 1
                : (tc.dtype == "tf32")                        ? 4
                : 2;

    // Packed registers: for fp16/bf16, two elements per 32-bit register.
    // For fp32/tf32, one element per register.
    // For fp8/int8, four elements per 32-bit register.
    int64_t elems_per_reg = 4 / bpe;  // How many elements fit in a 32-bit register
    if (elems_per_reg < 1) elems_per_reg = 1;

    int64_t regs_per_mma = (elems_per_thread + elems_per_reg - 1) / elems_per_reg;

    return regs_per_mma * rep_m * rep_k;
}

int64_t RegisterAllocator::compute_b_regs(int64_t tile_k, int64_t tile_n) const {
    const auto& tc = target_.gpu.tensor_core;
    if (tile_k % tc.k != 0 || tile_n % tc.n != 0) return 0;

    int64_t rep_k = tile_k / tc.k;
    int64_t rep_n = tile_n / tc.n;

    int64_t b_elems = tc.k * tc.n;
    int64_t warp_size = target_.gpu.warp_size;
    int64_t elems_per_thread = (b_elems + warp_size - 1) / warp_size;

    int64_t bpe = (tc.dtype == "fp16" || tc.dtype == "bf16") ? 2
                : (tc.dtype == "fp8" || tc.dtype == "int8")  ? 1
                : (tc.dtype == "tf32")                        ? 4
                : 2;

    int64_t elems_per_reg = 4 / bpe;
    if (elems_per_reg < 1) elems_per_reg = 1;

    int64_t regs_per_mma = (elems_per_thread + elems_per_reg - 1) / elems_per_reg;

    return regs_per_mma * rep_n * rep_k;
}

int64_t RegisterAllocator::compute_c_regs(int64_t tile_m, int64_t tile_n) const {
    const auto& tc = target_.gpu.tensor_core;
    if (tile_m % tc.m != 0 || tile_n % tc.n != 0) return 0;

    int64_t rep_m = tile_m / tc.m;
    int64_t rep_n = tile_n / tc.n;

    // C accumulator is always fp32 (4 bytes per element, 1 element per register)
    int64_t c_elems = tc.m * tc.n;
    int64_t warp_size = target_.gpu.warp_size;
    int64_t elems_per_thread = (c_elems + warp_size - 1) / warp_size;

    // fp32 accumulator: 1 element per 32-bit register
    int64_t regs_per_mma = elems_per_thread;

    return regs_per_mma * rep_m * rep_n;
}

// ---------------------------------------------------------------------------
// Supporting register counts
// ---------------------------------------------------------------------------

int64_t RegisterAllocator::compute_pointer_regs() const {
    // Pointers needed for a matmul kernel:
    //   - A global pointer
    //   - B global pointer
    //   - C global pointer
    //   - A shared memory base
    //   - B shared memory base
    //   - Parameters pointer (for dim constants)
    //   - TMA barrier descriptor (if has_tma)
    //   - TMA tensor descriptor for A (if has_tma)
    //   - TMA tensor descriptor for B (if has_tma)
    int64_t ptrs = 6;  // A_gmem, B_gmem, C_gmem, A_smem, B_smem, params

    if (target_.has_tma) {
        ptrs += 3;  // TMA barrier, A tensor desc, B tensor desc
    }

    return ptrs;
}

int64_t RegisterAllocator::compute_index_regs(int64_t pipeline_stages) const {
    // Loop indices and computed offsets:
    //   - blockIdx.x, blockIdx.y
    //   - threadIdx.x, threadIdx.y, threadIdx.z
    //   - K-loop iterator
    //   - Global row offset (blockIdx.x * tile_m + threadIdx.x)
    //   - Global col offset (blockIdx.y * tile_n + threadIdx.y)
    //   - Shared memory write offset (for async copy)
    //   - Shared memory read offset (for WMMA load)
    //   - Per-pipeline-stage offsets
    int64_t indices = 8;

    // Additional indices per pipeline stage (double/triple buffering offsets)
    indices += (pipeline_stages - 1) * 2;

    return indices;
}

int64_t RegisterAllocator::compute_predicate_regs() const {
    // Predicates:
    //   - Boundary check for M dimension
    //   - Boundary check for N dimension
    //   - Boundary check for K dimension
    //   - Loop continuation predicate
    //   - Epilogue mask
    return 5;
}

// ---------------------------------------------------------------------------
// Main allocation routine
// ---------------------------------------------------------------------------

RegisterAllocation RegisterAllocator::allocate_for_matmul(
    int64_t tile_m, int64_t tile_n, int64_t tile_k,
    int64_t pipeline_stages
) const {
    RegisterAllocation alloc;

    // Single-stage fragment registers
    int64_t a_regs_single = compute_a_regs(tile_m, tile_k);
    int64_t b_regs_single = compute_b_regs(tile_k, tile_n);
    int64_t c_regs        = compute_c_regs(tile_m, tile_n);

    if (a_regs_single == 0 || b_regs_single == 0 || c_regs == 0) {
        // Invalid tile (not aligned to MMA dimensions)
        alloc.fits_in_hw = false;
        return alloc;
    }

    // With software pipelining, we hold pipeline_stages copies of A and B
    // fragments (one for compute, rest for prefetching next tiles).
    // C fragment is persistent and only needs one copy.
    alloc.wmma_a_regs = a_regs_single * pipeline_stages;
    alloc.wmma_b_regs = b_regs_single * pipeline_stages;
    alloc.wmma_c_regs = c_regs;

    alloc.pointer_regs   = compute_pointer_regs();
    alloc.index_regs     = compute_index_regs(pipeline_stages);
    alloc.predicate_regs = compute_predicate_regs();

    alloc.total_registers =
        alloc.wmma_a_regs +
        alloc.wmma_b_regs +
        alloc.wmma_c_regs +
        alloc.pointer_regs +
        alloc.index_regs +
        alloc.predicate_regs;

    alloc.fits_in_hw = alloc.total_registers <= target_.gpu.sm.max_registers_per_thread;
    alloc.spare_regs = target_.gpu.sm.max_registers_per_thread - alloc.total_registers;
    if (!alloc.fits_in_hw) {
        alloc.spare_regs = 0;
    }

    return alloc;
}

// ---------------------------------------------------------------------------
// Feasibility check
// ---------------------------------------------------------------------------

bool RegisterAllocator::is_feasible(const RegisterAllocation& alloc) const {
    if (!alloc.fits_in_hw) return false;

    // Leave at least 8 registers of headroom for the compiler's own use
    // (stack frame, spill temps, etc.)
    if (alloc.spare_regs < 8) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Tile size reduction
// ---------------------------------------------------------------------------

std::vector<int64_t> RegisterAllocator::suggest_reduced_tile(
    int64_t tile_m, int64_t tile_n, int64_t tile_k
) const {
    const auto& tc = target_.gpu.tensor_core;
    int64_t pipeline_stages = target_.pipeline_stages;

    // Strategy: try reducing each dimension by one MMA step at a time.
    // Prioritise reducing K first (reduces A and B fragments simultaneously),
    // then M, then N.
    // Keep reducing until the allocation fits.

    int64_t cur_m = tile_m;
    int64_t cur_n = tile_n;
    int64_t cur_k = tile_k;

    auto try_allocation = [&](int64_t m, int64_t n, int64_t k) -> bool {
        auto alloc = allocate_for_matmul(m, n, k, pipeline_stages);
        return is_feasible(alloc);
    };

    // First try reducing K
    while (cur_k > tc.k && !try_allocation(cur_m, cur_n, cur_k)) {
        cur_k -= tc.k;
    }

    // Then try reducing M
    while (cur_m > tc.m && !try_allocation(cur_m, cur_n, cur_k)) {
        cur_m -= tc.m;
    }

    // Then try reducing N
    while (cur_n > tc.n && !try_allocation(cur_m, cur_n, cur_k)) {
        cur_n -= tc.n;
    }

    // If still not feasible at minimum tile, return the minimum MMA tile
    if (!try_allocation(cur_m, cur_n, cur_k)) {
        return {tc.m, tc.n, tc.k};
    }

    return {cur_m, cur_n, cur_k};
}

} // namespace symplex::codegen
