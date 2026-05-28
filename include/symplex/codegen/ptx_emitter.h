// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/codegen/wmma.h"
#include "symplex/codegen/swizzle.h"
#include "symplex/codegen/register_allocator.h"
#include "symplex/schedule/tiling.h"
#include <cstdint>
#include <string>

namespace symplex::codegen {

/// PTXEmitter: generates complete PTX assembly kernels for matrix
/// multiplication and convolution operations targeting NVIDIA Tensor Cores.
class PTXEmitter {
public:
    explicit PTXEmitter(const hardware::HardwareTarget& target);

    /// Generate a complete PTX kernel for matrix multiplication:
    ///   C[M,N] = A[M,K] * B[K,N]
    /// @param M, N, K   Global matrix dimensions
    /// @param tile       Tile configuration (outer/inner tile sizes)
    /// @param kernel_name  Name of the generated kernel entry point
    std::string emit_matmul_kernel(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile,
        const std::string& kernel_name = "symplex_matmul"
    ) const;

    /// Generate a complete PTX kernel for 2D convolution:
    ///   output[batch,oc,oh,ow] = input[batch,ic,ih,iw] * kernel[oc,ic,kh,kw]
    std::string emit_conv2d_kernel(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride, int64_t pad,
        const schedule::TileConfig& tile,
        const std::string& kernel_name = "symplex_conv2d"
    ) const;

private:
    hardware::HardwareTarget target_;
    WMMAGenerator      wmma_gen_;
    SwizzleGenerator   swizzle_gen_;
    RegisterAllocator  reg_alloc_;

    // ── PTX section builders ──────────────────────────────────────

    /// Emit the PTX file header (.version, .target, .address_size).
    std::string emit_ptx_header(const std::string& kernel_name) const;

    /// Emit parameter declarations for the matmul kernel.
    std::string emit_ptx_param_decls(int64_t M, int64_t N, int64_t K) const;

    /// Emit shared memory declarations for the matmul tile.
    std::string emit_ptx_shared_mem_decl(
        int64_t tile_m, int64_t tile_n, int64_t tile_k
    ) const;

    /// Emit prologue: compute blockIdx, threadIdx, global offsets.
    std::string emit_ptx_prologue() const;

    /// Emit the main tiling loop over K with WMMA pipeline.
    std::string emit_ptx_tiling_loop(
        int64_t M, int64_t N, int64_t K,
        const schedule::TileConfig& tile
    ) const;

    /// Emit epilogue: write C fragment to global memory and return.
    /// @param c_store_code  PTX instructions for the WMMA store (emitted
    ///                      inside the in-bounds guard, between the
    ///                      boundary check branch and EPILOGUE_SKIP).
    std::string emit_ptx_epilogue(const std::string& c_store_code = "") const;

    // ── Conv2d helpers ────────────────────────────────────────────

    /// Emit parameter declarations for the conv2d kernel.
    std::string emit_conv2d_param_decls(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride, int64_t pad
    ) const;

    /// Emit the im2col-style tiling loop for conv2d.
    std::string emit_conv2d_tiling_loop(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride, int64_t pad,
        const schedule::TileConfig& tile
    ) const;

    // ── Utility ───────────────────────────────────────────────────

    /// Determine the PTX .target architecture string.
    std::string ptx_target_arch() const;

    /// Compute shared memory size in bytes for a given tile.
    int64_t compute_smem_bytes(
        int64_t tile_m, int64_t tile_n, int64_t tile_k
    ) const;

    /// Compute grid and block dimensions.
    void compute_launch_dims(
        int64_t M, int64_t N,
        int64_t tile_m, int64_t tile_n,
        int64_t& grid_x, int64_t& grid_y,
        int64_t& block_x, int64_t& block_y
    ) const;
};

} // namespace symplex::codegen
