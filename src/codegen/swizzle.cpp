// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/swizzle.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace symplex::codegen {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SwizzleGenerator::SwizzleGenerator(const hardware::HardwareTarget& target)
    : target_(target)
{}

// ---------------------------------------------------------------------------
// Swizzle address computation
// ---------------------------------------------------------------------------

int64_t SwizzleGenerator::swizzle_address(
    int64_t row, int64_t col, int64_t stride
) const {
    // Determine the optimal swizzle bits for this tile layout.
    int swizzle_bits = find_optimal_swizzle_bits(row + 1 > 0 ? row + 1 : 1,
                                                  col + 1 > 0 ? col + 1 : 1,
                                                  stride);

    // Apply XOR swizzle to the column:
    //   col_swizzled = col ^ (row << swizzle_bits)
    // More precisely, the swizzle XORs the upper address bits derived from
    // the column with row-derived bits:
    //   col_swizzled = col ^ ((row & ((1 << swizzle_bits) - 1)) << (swizzle_bits))
    // This spreads consecutive rows across different bank groups.
    int64_t mask = (int64_t(1) << swizzle_bits) - 1;
    int64_t col_swizzled = col ^ ((row & mask) << swizzle_bits);

    // Byte address = (row * stride + col_swizzled) * bytes_per_element
    int64_t bpe = target_.bytes_per_element;
    return (row * stride + col_swizzled) * bpe;
}

// ---------------------------------------------------------------------------
// Bank conflict analysis
// ---------------------------------------------------------------------------

int SwizzleGenerator::count_bank_conflicts(
    int64_t tile_m, int64_t tile_n, int64_t stride, int swizzle_bits
) const {
    int64_t bpe = target_.bytes_per_element;
    int64_t warp_size = target_.gpu.warp_size;

    // Simulate a warp of threads accessing shared memory.
    // In a typical WMMA/MMA pattern, each warp thread reads a strided
    // subset of elements. We model the access pattern as:
    //   thread t accesses element (row, col) where:
    //     row = t / (tile_n)   (rounded down)
    //     col = t % (tile_n)   (for row-major access)
    // More precisely, for a WMMA load, each thread accesses a specific
    // set of elements. We approximate by simulating all 32 threads
    // reading one element each in a row-major sweep across the tile.

    std::unordered_map<int, int> bank_access_count;

    int total_conflicts = 0;
    int64_t threads = std::min(warp_size, tile_m * tile_n);

    for (int64_t t = 0; t < threads; ++t) {
        // Row-major access pattern: thread t reads element at (row, col)
        int64_t row = t / tile_n;
        int64_t col = t % tile_n;

        // Apply swizzle to the column
        int64_t mask = (int64_t(1) << swizzle_bits) - 1;
        int64_t col_swizzled = col ^ ((row & mask) << swizzle_bits);

        // Compute byte address
        int64_t byte_addr = (row * stride + col_swizzled) * bpe;

        // Compute bank
        int bank = bank_of(byte_addr);

        bank_access_count[bank]++;

        // If this bank was already accessed by a previous thread in the
        // same warp, that's a conflict.
        if (bank_access_count[bank] > 1) {
            total_conflicts++;
        }
    }

    return total_conflicts;
}

bool SwizzleGenerator::has_bank_conflicts(
    int64_t tile_m, int64_t tile_n, int64_t stride
) const {
    // Test with no swizzle first
    return count_bank_conflicts(tile_m, tile_n, stride, 0) > 0;
}

// ---------------------------------------------------------------------------
// Optimal swizzle bits
// ---------------------------------------------------------------------------

int SwizzleGenerator::find_optimal_swizzle_bits(
    int64_t tile_m, int64_t tile_n, int64_t stride
) const {
    int best_bits = 0;
    int best_conflicts = count_bank_conflicts(tile_m, tile_n, stride, 0);

    // If already conflict-free, no swizzle needed
    if (best_conflicts == 0) return 0;

    // Test swizzle_bits from 1 to 4
    for (int bits = 1; bits <= 4; ++bits) {
        int conflicts = count_bank_conflicts(tile_m, tile_n, stride, bits);
        if (conflicts < best_conflicts) {
            best_conflicts = conflicts;
            best_bits = bits;
        }
        // If we've eliminated all conflicts, stop early
        if (best_conflicts == 0) break;
    }

    return best_bits;
}

// ---------------------------------------------------------------------------
// PTX emission: swizzle on store (write to shared memory)
// ---------------------------------------------------------------------------

std::string SwizzleGenerator::emit_smem_swizzle(
    int64_t tile_m, int64_t tile_n, int64_t stride,
    const std::string& row_reg,
    const std::string& col_reg,
    const std::string& smem_base_reg,
    const std::string& smem_offset_reg
) const {
    std::ostringstream oss;
    int swizzle_bits = find_optimal_swizzle_bits(tile_m, tile_n, stride);

    oss << "    // --- Shared Memory Swizzle (write) ---\n";
    oss << "    // Tile: " << tile_m << "x" << tile_n
        << ", stride=" << stride
        << ", swizzle_bits=" << swizzle_bits << "\n";

    if (swizzle_bits == 0) {
        oss << "    // No swizzle required (bank-conflict-free layout)\n";
        return oss.str();
    }

    int64_t bpe = target_.bytes_per_element;
    // The swizzle mask in elements (before scaling by bytes_per_element)
    int64_t elem_mask = (int64_t(1) << swizzle_bits) - 1;
    int64_t byte_shift = swizzle_bits;

    oss << "    // XOR swizzle: col_swizzled = col ^ ((row & 0x"
        << std::hex << elem_mask << std::dec
        << ") << " << swizzle_bits << ")\n";

    // Emit PTX inline assembly for the swizzle transform.
    // Each thread computes its swizzled column index before writing.
    // We use a generic template that the PTXEmitter fills in with
    // concrete register names.
    oss << "    // PTX swizzle transform:\n";
    oss << "    //   .reg .u64 %swiz_tmp;\n";
    oss << "    //   and.b64 %swiz_tmp, %row, " << elem_mask << ";\n";
    oss << "    //   shl.b64 %swiz_tmp, %swiz_tmp, " << byte_shift << ";\n";
    oss << "    //   xor.b64 %col_swizzled, %col, %swiz_tmp;\n";

    // Emit concrete inline asm block
    oss << "    {\n";
    oss << "        // Compute swizzled shared memory address\n";
    oss << "        // row_mask = row & " << elem_mask << "\n";
    oss << "        // col_swizzled = col ^ (row_mask << " << swizzle_bits << ")\n";
    oss << "        // byte_offset = (row * " << stride << " + col_swizzled) * "
        << bpe << "\n";
    oss << "        .reg .u64 %swiz_row_mask;\n";
    oss << "        .reg .u64 %swiz_col_swiz;\n";
    oss << "        and.b64 %swiz_row_mask, " << row_reg << ", " << elem_mask << ";\n";
    oss << "        shl.b64 %swiz_row_mask, %swiz_row_mask, " << byte_shift << ";\n";
    oss << "        xor.b64 %swiz_col_swiz, " << col_reg << ", %swiz_row_mask;\n";
    oss << "        mad.lo.u64 " << smem_offset_reg << ", " << row_reg << ", " << (stride * bpe)
        << ", 0;\n";
    oss << "        mad.lo.u64 " << smem_offset_reg << ", %swiz_col_swiz, " << bpe
        << ", " << smem_offset_reg << ";\n";
    oss << "        // Store to shared memory at [" << smem_base_reg << " + " << smem_offset_reg << "]\n";
    oss << "    }\n";

    return oss.str();
}

// ---------------------------------------------------------------------------
// PTX emission: inverse swizzle (read from shared memory)
// ---------------------------------------------------------------------------

std::string SwizzleGenerator::emit_smem_unswizzle(
    int64_t tile_m, int64_t tile_n, int64_t stride,
    const std::string& row_reg,
    const std::string& col_reg,
    const std::string& smem_base_reg,
    const std::string& smem_offset_reg
) const {
    std::ostringstream oss;
    int swizzle_bits = find_optimal_swizzle_bits(tile_m, tile_n, stride);

    oss << "    // --- Shared Memory Unswizzle (read) ---\n";
    oss << "    // Tile: " << tile_m << "x" << tile_n
        << ", stride=" << stride
        << ", swizzle_bits=" << swizzle_bits << "\n";

    if (swizzle_bits == 0) {
        oss << "    // No unswizzle required (bank-conflict-free layout)\n";
        return oss.str();
    }

    // For XOR-based swizzle, the inverse is the same as the forward transform
    // because XOR is its own inverse: (col ^ x) ^ x = col
    int64_t elem_mask = (int64_t(1) << swizzle_bits) - 1;
    int64_t byte_shift = swizzle_bits;
    int64_t bpe = target_.bytes_per_element;

    oss << "    // XOR unswizzle (same as swizzle for XOR): "
        << "col = col_swizzled ^ ((row & 0x"
        << std::hex << elem_mask << std::dec
        << ") << " << swizzle_bits << ")\n";

    oss << "    {\n";
    oss << "        // Compute unswizzled column for reading\n";
    oss << "        .reg .u64 %uswiz_row_mask;\n";
    oss << "        .reg .u64 %uswiz_col_orig;\n";
    oss << "        and.b64 %uswiz_row_mask, " << row_reg << ", " << elem_mask << ";\n";
    oss << "        shl.b64 %uswiz_row_mask, %uswiz_row_mask, " << byte_shift << ";\n";
    oss << "        xor.b64 %uswiz_col_orig, " << col_reg << ", %uswiz_row_mask;\n";
    oss << "        mad.lo.u64 " << smem_offset_reg << ", " << row_reg << ", " << (stride * bpe)
        << ", 0;\n";
    oss << "        mad.lo.u64 " << smem_offset_reg << ", %uswiz_col_orig, " << bpe
        << ", " << smem_offset_reg << ";\n";
    oss << "        // Load from shared memory at [" << smem_base_reg << " + " << smem_offset_reg << "]\n";
    oss << "    }\n";

    return oss.str();
}

} // namespace symplex::codegen
