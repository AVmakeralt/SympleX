// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/code_generator.h"
#include <sstream>
#include <algorithm>

namespace symplex::codegen {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

CodeGenerator::CodeGenerator(const hardware::HardwareTarget& target)
    : target_(target)
    , ptx_emitter_(target)
    , reg_alloc_(target)
{}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool CodeGenerator::validate_matmul_tile(
    int64_t M, int64_t N, int64_t K,
    const schedule::TileConfig& tile,
    std::string& error_message
) const {
    const auto& tc = target_.gpu.tensor_core;

    // Check that tile has at least 3 inner dimensions
    if (tile.inner_tiles.size() < 3) {
        error_message = "Matmul tile must have at least 3 inner dimensions "
                        "(M, N, K), got " + std::to_string(tile.inner_tiles.size());
        return false;
    }

    int64_t tile_m = tile.inner_tiles[0];
    int64_t tile_n = tile.inner_tiles[1];
    int64_t tile_k = tile.inner_tiles[2];

    // Dimensions must be positive
    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        error_message = "Tile dimensions must be positive, got ("
                      + std::to_string(tile_m) + ", "
                      + std::to_string(tile_n) + ", "
                      + std::to_string(tile_k) + ")";
        return false;
    }

    // Dimensions must be multiples of the MMA fragment dimensions
    if (tile_m % tc.m != 0) {
        error_message = "tile_m=" + std::to_string(tile_m)
                      + " is not a multiple of MMA m=" + std::to_string(tc.m);
        return false;
    }
    if (tile_n % tc.n != 0) {
        error_message = "tile_n=" + std::to_string(tile_n)
                      + " is not a multiple of MMA n=" + std::to_string(tc.n);
        return false;
    }
    if (tile_k % tc.k != 0) {
        error_message = "tile_k=" + std::to_string(tile_k)
                      + " is not a multiple of MMA k=" + std::to_string(tc.k);
        return false;
    }

    // Check shared memory capacity
    int64_t bpe = target_.bytes_per_element;
    int64_t stages = target_.pipeline_stages;
    int64_t smem_a = tile_m * tile_k * bpe * stages;
    int64_t smem_b = tile_k * tile_n * bpe * stages;
    int64_t total_smem = smem_a + smem_b;

    if (total_smem > target_.max_sram_bytes) {
        error_message = "Shared memory overflow: "
                      + std::to_string(total_smem) + " bytes needed, "
                      + std::to_string(target_.max_sram_bytes) + " bytes available";
        return false;
    }

    // Check register feasibility
    auto alloc = reg_alloc_.allocate_for_matmul(tile_m, tile_n, tile_k, stages);
    if (!reg_alloc_.is_feasible(alloc)) {
        error_message = "Register overflow: "
                      + std::to_string(alloc.total_registers) + " registers needed, "
                      + std::to_string(target_.gpu.sm.max_registers_per_thread)
                      + " max available (need 8 spare for compiler)";
        return false;
    }

    // Check thread count
    int64_t block_threads = target_.gpu.warp_size;
    if (block_threads > target_.gpu.max_threads_per_block) {
        error_message = "Too many threads per block: "
                      + std::to_string(block_threads);
        return false;
    }

    // Check that global dimensions are positive
    if (M <= 0 || N <= 0 || K <= 0) {
        error_message = "Global dimensions must be positive: M="
                      + std::to_string(M) + ", N=" + std::to_string(N)
                      + ", K=" + std::to_string(K);
        return false;
    }

    // Validate against HardwareTarget's own validate_tile_config
    if (!target_.validate_tile_config(tile.inner_tiles, alloc.total_registers)) {
        error_message = "Tile configuration rejected by hardware validation";
        return false;
    }

    return true;
}

bool CodeGenerator::validate_conv2d_tile(
    int64_t batch, int64_t oc, int64_t ic,
    int64_t oh, int64_t ow, int64_t kh, int64_t kw,
    int64_t stride, int64_t pad,
    const schedule::TileConfig& tile,
    std::string& error_message
) const {
    const auto& tc = target_.gpu.tensor_core;

    if (tile.inner_tiles.size() < 3) {
        error_message = "Conv2d tile must have at least 3 inner dimensions, got "
                      + std::to_string(tile.inner_tiles.size());
        return false;
    }

    int64_t tile_m = tile.inner_tiles[0];
    int64_t tile_n = tile.inner_tiles[1];
    int64_t tile_k = tile.inner_tiles[2];

    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        error_message = "Tile dimensions must be positive";
        return false;
    }

    if (tile_m % tc.m != 0 || tile_n % tc.n != 0 || tile_k % tc.k != 0) {
        error_message = "Tile dimensions not aligned to MMA fragment sizes";
        return false;
    }

    // Shared memory check
    int64_t bpe = target_.bytes_per_element;
    int64_t stages = target_.pipeline_stages;
    int64_t total_smem = (tile_m * tile_k + tile_k * tile_n) * bpe * stages;

    if (total_smem > target_.max_sram_bytes) {
        error_message = "Shared memory overflow for conv2d: "
                      + std::to_string(total_smem) + " bytes";
        return false;
    }

    // Register feasibility
    auto alloc = reg_alloc_.allocate_for_matmul(tile_m, tile_n, tile_k, stages);
    if (!reg_alloc_.is_feasible(alloc)) {
        error_message = "Register overflow for conv2d: "
                      + std::to_string(alloc.total_registers) + " registers";
        return false;
    }

    // Check global dimensions
    if (batch <= 0 || oc <= 0 || ic <= 0 || oh <= 0 || ow <= 0 || kh <= 0 || kw <= 0) {
        error_message = "Conv2d dimensions must be positive";
        return false;
    }

    if (stride <= 0) {
        error_message = "Conv2d stride must be positive";
        return false;
    }

    if (pad < 0) {
        error_message = "Conv2d pad must be non-negative";
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Matmul generation
// ---------------------------------------------------------------------------

GeneratedKernel CodeGenerator::generate_matmul(
    int64_t M, int64_t N, int64_t K,
    const schedule::TileConfig& tile,
    int depth
) {
    GeneratedKernel result;
    result.kernel_name = "symplex_matmul";

    // Validate
    std::string error;
    if (!validate_matmul_tile(M, N, K, tile, error)) {
        // Best-effort auto-repair: if a user/requested tile is not feasible,
        // shrink it to a nearby register-feasible Tensor Core aligned tile.
        // Guard against infinite recursion: if we've already repaired too
        // many times, accept the best available tile even if imperfect.
        constexpr int MAX_REPAIR_DEPTH = 5;

        auto suggested = reg_alloc_.suggest_reduced_tile(
            tile.inner_tiles.size() > 0 ? tile.inner_tiles[0] : target_.gpu.tensor_core.m,
            tile.inner_tiles.size() > 1 ? tile.inner_tiles[1] : target_.gpu.tensor_core.n,
            tile.inner_tiles.size() > 2 ? tile.inner_tiles[2] : target_.gpu.tensor_core.k
        );
        if (suggested.size() == 3) {
            schedule::TileConfig repaired = tile;
            repaired.inner_tiles = suggested;
            if (repaired.outer_tiles.size() >= 3) {
                repaired.outer_tiles[0] = std::max(repaired.outer_tiles[0], repaired.inner_tiles[0]);
                repaired.outer_tiles[1] = std::max(repaired.outer_tiles[1], repaired.inner_tiles[1]);
                repaired.outer_tiles[2] = std::max(repaired.outer_tiles[2], repaired.inner_tiles[2]);
            } else {
                repaired.outer_tiles = repaired.inner_tiles;
            }

            std::string repaired_error;
            if (validate_matmul_tile(M, N, K, repaired, repaired_error)) {
                return generate_matmul(M, N, K, repaired, depth + 1);
            }

            // If repaired tile also fails and we haven't exceeded max depth,
            // try one more round of repair.  Otherwise, fall through and
            // use the best-effort repaired tile anyway.
            if (depth < MAX_REPAIR_DEPTH) {
                return generate_matmul(M, N, K, repaired, depth + 1);
            }
            // Max depth reached: use the best-effort repaired tile even
            // though it doesn't perfectly validate.  This avoids infinite
            // recursion and still produces a usable kernel.
        }

        result.valid = false;
        result.error_message = error;
        return result;
    }

    int64_t tile_m = tile.inner_tiles[0];
    int64_t tile_n = tile.inner_tiles[1];
    int64_t tile_k = tile.inner_tiles[2];
    int64_t stages = target_.pipeline_stages;
    int64_t bpe = target_.bytes_per_element;

    // Compute metadata
    result.shared_mem_bytes = (tile_m * tile_k + tile_k * tile_n) * bpe * stages;
    auto alloc = reg_alloc_.allocate_for_matmul(tile_m, tile_n, tile_k, stages);
    result.register_count = alloc.total_registers;

    // Grid and block dimensions
    int64_t grid_x = (N + tile_n - 1) / tile_n;
    int64_t grid_y = (M + tile_m - 1) / tile_m;
    result.grid_dims  = {grid_x, grid_y, 1};
    result.block_dims = {target_.gpu.warp_size, 1, 1};

    // Generate PTX
    try {
        result.ptx_source = ptx_emitter_.emit_matmul_kernel(
            M, N, K, tile, result.kernel_name);
        result.valid = true;
    } catch (const std::exception& e) {
        result.valid = false;
        result.error_message = std::string("PTX emission failed: ") + e.what();
    }

    return result;
}

// ---------------------------------------------------------------------------
// Conv2d generation
// ---------------------------------------------------------------------------

GeneratedKernel CodeGenerator::generate_conv2d(
    int64_t batch, int64_t oc, int64_t ic,
    int64_t oh, int64_t ow, int64_t kh, int64_t kw,
    int64_t stride, int64_t pad,
    const schedule::TileConfig& tile
) {
    GeneratedKernel result;
    result.kernel_name = "symplex_conv2d";

    // Validate
    std::string error;
    if (!validate_conv2d_tile(batch, oc, ic, oh, ow, kh, kw, stride, pad,
                              tile, error)) {
        result.valid = false;
        result.error_message = error;
        return result;
    }

    int64_t tile_m = tile.inner_tiles[0];
    int64_t tile_n = tile.inner_tiles[1];
    int64_t tile_k = tile.inner_tiles[2];
    int64_t stages = target_.pipeline_stages;
    int64_t bpe = target_.bytes_per_element;

    // Compute metadata
    result.shared_mem_bytes = (tile_m * tile_k + tile_k * tile_n) * bpe * stages;
    auto alloc = reg_alloc_.allocate_for_matmul(tile_m, tile_n, tile_k, stages);
    result.register_count = alloc.total_registers;

    // Grid dimensions for conv2d:
    //   Grid X = ceil(oc / tile_n)
    //   Grid Y = ceil(batch * oh * ow / tile_m)
    //   Grid Z = 1
    int64_t eff_M = batch * oh * ow;
    int64_t grid_x = (oc + tile_n - 1) / tile_n;
    int64_t grid_y = (eff_M + tile_m - 1) / tile_m;
    result.grid_dims  = {grid_x, grid_y, 1};
    result.block_dims = {target_.gpu.warp_size, 1, 1};

    // Generate PTX
    try {
        result.ptx_source = ptx_emitter_.emit_conv2d_kernel(
            batch, oc, ic, oh, ow, kh, kw, stride, pad,
            tile, result.kernel_name);
        result.valid = true;
    } catch (const std::exception& e) {
        result.valid = false;
        result.error_message = std::string("PTX emission failed: ") + e.what();
    }

    return result;
}

} // namespace symplex::codegen
