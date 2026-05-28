// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include <cstdint>
#include <string>

namespace symplex::codegen {

/// WMMAConfig: describes the WMMA/MMA fragment layout for a single
/// warp-level matrix multiply-accumulate operation.
struct WMMAConfig {
    int64_t m = 16;             // MMA fragment M dimension
    int64_t n = 8;              // MMA fragment N dimension
    int64_t k = 16;             // MMA fragment K dimension
    std::string a_dtype = "fp16";   // Data type of A matrix ("fp16", "bf16", "fp8", "int8")
    std::string b_dtype = "fp16";   // Data type of B matrix
    std::string acc_dtype = "fp32"; // Accumulator data type (always fp32 for WMMA)
    int64_t a_elements_per_frag = 0; // Elements in A fragment per warp
    int64_t b_elements_per_frag = 0; // Elements in B fragment per warp
    int64_t c_elements_per_frag = 0; // Elements in C fragment per warp
    int64_t bytes_per_a_frag = 0;    // Bytes occupied by A fragment
    int64_t bytes_per_b_frag = 0;    // Bytes occupied by B fragment
    int64_t bytes_per_c_frag = 0;    // Bytes occupied by C fragment
};

/// WMMAGenerator: generates PTX WMMA/MMA instruction strings for
/// warp-level matrix operations on NVIDIA Tensor Cores.
class WMMAGenerator {
public:
    explicit WMMAGenerator(const hardware::HardwareTarget& target);

    /// Generate PTX instruction to load A fragment from shared memory.
    /// @param tile_m  M dimension of the tile (must be multiple of wmma_m)
    /// @param tile_k  K dimension of the tile (must be multiple of wmma_k)
    /// @param smem_ptr  PTX register name holding the shared memory base pointer
    /// @param stride    Leading dimension stride (in elements) of A in shared memory
    std::string emit_wmma_load_a(
        int64_t tile_m, int64_t tile_k,
        const std::string& smem_ptr, int64_t stride
    ) const;

    /// Generate PTX instruction to load B fragment from shared memory.
    /// @param tile_k  K dimension of the tile
    /// @param tile_n  N dimension of the tile
    /// @param smem_ptr  PTX register name holding the shared memory base pointer
    /// @param stride    Leading dimension stride (in elements) of B in shared memory
    std::string emit_wmma_load_b(
        int64_t tile_k, int64_t tile_n,
        const std::string& smem_ptr, int64_t stride
    ) const;

    /// Generate PTX instruction for WMMA/MMA compute.
    std::string emit_wmma_mma(
        int64_t tile_m, int64_t tile_n, int64_t tile_k
    ) const;

    /// Generate PTX instruction to store C fragment to global memory.
    /// @param tile_m  M dimension of the output tile
    /// @param tile_n  N dimension of the output tile
    /// @param gmem_ptr  PTX register name holding the global memory base pointer
    /// @param stride    Leading dimension stride (in elements) of C in global memory
    std::string emit_wmma_store_c(
        int64_t tile_m, int64_t tile_n,
        const std::string& gmem_ptr, int64_t stride
    ) const;

    /// Access the current WMMA configuration.
    const WMMAConfig& config() const { return wmma_config_; }

    /// Compute the number of WMMA operations needed for a given tile.
    int64_t wmma_count(int64_t tile_m, int64_t tile_n, int64_t tile_k) const;

    /// Compute register usage for a given tile (in 32-bit registers).
    int64_t register_count(int64_t tile_m, int64_t tile_n, int64_t tile_k) const;

private:
    hardware::HardwareTarget target_;
    WMMAConfig wmma_config_;

    /// Initialize WMMA config based on hardware target.
    void init_config();

    /// Compute bytes per element for a given dtype string.
    static int64_t bytes_for_dtype(const std::string& dtype);

    /// Determine the PTX type string for a given dtype.
    static std::string ptx_type_for_dtype(const std::string& dtype);
};

} // namespace symplex::codegen
