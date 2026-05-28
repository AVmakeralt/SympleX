// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/codegen/ptx_emitter.h"
#include "symplex/codegen/register_allocator.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include <cstdint>
#include <string>
#include <vector>

namespace symplex::codegen {

/// GeneratedKernel: the complete output of the code generator for a
/// single kernel, including the PTX source, metadata, and validity status.
struct GeneratedKernel {
    std::string ptx_source;             // Complete PTX assembly source
    std::string kernel_name;            // Entry-point name
    int64_t     shared_mem_bytes = 0;   // Shared memory consumed per CTA
    int64_t     register_count   = 0;   // Per-thread register count
    std::vector<int64_t> grid_dims;     // Grid dimensions [x, y, z]
    std::vector<int64_t> block_dims;    // Block dimensions [x, y, z]
    bool        valid     = false;      // Whether generation succeeded
    std::string error_message;          // Human-readable error if !valid
};

/// CodeGenerator: top-level code generation interface that validates
/// tile configurations, checks register and shared-memory feasibility,
/// and delegates to PTXEmitter for the actual PTX generation.
class CodeGenerator {
public:
    explicit CodeGenerator(const hardware::HardwareTarget& target);

    /// Generate a complete matmul kernel.
    /// @param M, N, K  Global matrix dimensions
    /// @param tile     Tile configuration
    /// @param depth    Recursion depth for auto-repair (internal use)
    GeneratedKernel generate_matmul(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile,
        int depth = 0
    );

    /// Generate a complete conv2d kernel.
    GeneratedKernel generate_conv2d(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride, int64_t pad,
        const schedule::TileConfig& tile
    );

private:
    hardware::HardwareTarget target_;
    PTXEmitter     ptx_emitter_;
    RegisterAllocator reg_alloc_;

    /// Validate a tile configuration for matmul.
    /// Returns true if the tile is valid; sets error_message otherwise.
    bool validate_matmul_tile(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile,
        std::string& error_message
    ) const;

    /// Validate a tile configuration for conv2d.
    bool validate_conv2d_tile(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride, int64_t pad,
        const schedule::TileConfig& tile,
        std::string& error_message
    ) const;
};

} // namespace symplex::codegen
