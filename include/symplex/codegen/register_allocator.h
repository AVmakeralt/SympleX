// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include <cstdint>
#include <vector>

namespace symplex::codegen {

/// RegisterAllocation: breakdown of 32-bit register usage for a kernel.
struct RegisterAllocation {
    int64_t total_registers  = 0;   // Sum of all register categories
    int64_t wmma_a_regs      = 0;   // Registers for A matrix fragments
    int64_t wmma_b_regs      = 0;   // Registers for B matrix fragments
    int64_t wmma_c_regs      = 0;   // Registers for C accumulator fragments
    int64_t pointer_regs     = 0;   // Registers for base pointers (A, B, C, smem, etc.)
    int64_t index_regs       = 0;   // Registers for loop indices and offsets
    int64_t predicate_regs   = 0;   // Registers for predicates / condition flags
    int64_t spare_regs       = 0;   // Remaining registers (headroom)
    bool    fits_in_hw       = false; // Whether total ≤ max_registers_per_thread
};

/// RegisterAllocator: computes register pressure for a given tile
/// configuration and pipeline depth, and suggests reduced tile sizes
/// when the allocation exceeds the hardware limit.
class RegisterAllocator {
public:
    explicit RegisterAllocator(const hardware::HardwareTarget& target);

    /// Compute the full register allocation for a matrix-multiply kernel
    /// with the given tile sizes and pipeline depth.
    /// @param tile_m   M dimension of the CTA tile
    /// @param tile_n   N dimension of the CTA tile
    /// @param tile_k   K dimension of the CTA tile
    /// @param pipeline_stages  Number of software-pipeline stages (≥2)
    RegisterAllocation allocate_for_matmul(
        int64_t tile_m, int64_t tile_n, int64_t tile_k,
        int64_t pipeline_stages = 2
    ) const;

    /// Check whether an allocation is feasible on the target hardware.
    bool is_feasible(const RegisterAllocation& alloc) const;

    /// Suggest a reduced tile size [new_m, new_n, new_k] that fits within
    /// the register budget.  Each dimension is reduced by the smallest
    /// MMA step that brings the allocation under the limit.
    std::vector<int64_t> suggest_reduced_tile(
        int64_t tile_m, int64_t tile_n, int64_t tile_k
    ) const;

private:
    hardware::HardwareTarget target_;

    /// Compute per-thread A-fragment register count for one pipeline stage.
    int64_t compute_a_regs(int64_t tile_m, int64_t tile_k) const;

    /// Compute per-thread B-fragment register count for one pipeline stage.
    int64_t compute_b_regs(int64_t tile_k, int64_t tile_n) const;

    /// Compute per-thread C-fragment register count (persistent across K).
    int64_t compute_c_regs(int64_t tile_m, int64_t tile_n) const;

    /// Compute pointer register count for the matmul kernel.
    int64_t compute_pointer_regs() const;

    /// Compute index / loop-counter register count.
    int64_t compute_index_regs(int64_t pipeline_stages) const;

    /// Compute predicate register count.
    int64_t compute_predicate_regs() const;
};

} // namespace symplex::codegen
