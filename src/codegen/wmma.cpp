// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/codegen/wmma.h"
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace symplex::codegen {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int64_t WMMAGenerator::bytes_for_dtype(const std::string& dtype) {
    if (dtype == "fp16" || dtype == "bf16") return 2;
    if (dtype == "fp8"  || dtype == "int8" || dtype == "e4m3" || dtype == "e5m2") return 1;
    if (dtype == "fp32" || dtype == "int32") return 4;
    if (dtype == "tf32") return 4;  // TF32 stored in 32-bit registers
    return 2;  // default to fp16
}

std::string WMMAGenerator::ptx_type_for_dtype(const std::string& dtype) {
    if (dtype == "fp16") return ".f16";
    if (dtype == "bf16") return ".bf16";
    if (dtype == "fp8" || dtype == "e4m3") return ".e4m3";
    if (dtype == "e5m2") return ".e5m2";
    if (dtype == "int8") return ".s8";
    if (dtype == "fp32") return ".f32";
    if (dtype == "tf32") return ".tf32";
    if (dtype == "int32") return ".s32";
    return ".f16";
}

// ---------------------------------------------------------------------------
// Construction / initialisation
// ---------------------------------------------------------------------------

WMMAGenerator::WMMAGenerator(const hardware::HardwareTarget& target)
    : target_(target)
{
    init_config();
}

void WMMAGenerator::init_config() {
    const auto& tc = target_.gpu.tensor_core;

    wmma_config_.m = tc.m;
    wmma_config_.n = tc.n;
    wmma_config_.k = tc.k;
    wmma_config_.a_dtype  = tc.dtype;
    wmma_config_.b_dtype  = tc.dtype;
    wmma_config_.acc_dtype = "fp32";

    int64_t warp_size = target_.gpu.warp_size;  // 32

    // Total elements in each MMA fragment (per warp, before distribution)
    wmma_config_.a_elements_per_frag = wmma_config_.m * wmma_config_.k;
    wmma_config_.b_elements_per_frag = wmma_config_.k * wmma_config_.n;
    wmma_config_.c_elements_per_frag = wmma_config_.m * wmma_config_.n;

    // Byte sizes of full fragments
    int64_t a_bpe = bytes_for_dtype(wmma_config_.a_dtype);
    int64_t b_bpe = bytes_for_dtype(wmma_config_.b_dtype);
    int64_t c_bpe = bytes_for_dtype(wmma_config_.acc_dtype);

    wmma_config_.bytes_per_a_frag = wmma_config_.a_elements_per_frag * a_bpe;
    wmma_config_.bytes_per_b_frag = wmma_config_.b_elements_per_frag * b_bpe;
    wmma_config_.bytes_per_c_frag = wmma_config_.c_elements_per_frag * c_bpe;

    // For mma.sync (Ampere+), elements are distributed across the warp.
    // Each thread holds a_elements_per_frag / warp_size elements.
    // For fp16/bf16, two elements are packed per 32-bit register (.f16x2 / .bf16x2).
    // For fp8/int8, four elements are packed per 32-bit register.
    // For fp32 accumulator, one element per 32-bit register.
    (void)warp_size;  // Used in register_count() below
}

// ---------------------------------------------------------------------------
// PTX instruction emission
// ---------------------------------------------------------------------------

std::string WMMAGenerator::emit_wmma_load_a(
    int64_t tile_m, int64_t tile_k,
    const std::string& smem_ptr, int64_t stride
) const {
    std::ostringstream oss;
    int64_t wm = wmma_config_.m;
    int64_t wk = wmma_config_.k;

    if (tile_m % wm != 0 || tile_k % wk != 0) {
        oss << "/* ERROR: tile_m=" << tile_m << " not multiple of wmma_m=" << wm
            << " or tile_k=" << tile_k << " not multiple of wmma_k=" << wk << " */\n";
        return oss.str();
    }

    int64_t rep_m = tile_m / wm;
    int64_t rep_k = tile_k / wk;

    // Choose instruction style based on architecture
    bool use_mma_sync = (target_.gpu.sm.max_registers_per_thread >= 255);  // Ampere+

    if (use_mma_sync) {
        // mma.sync path: load from shared memory using LDSM (load matrix from shared)
        // or standard LDS with swizzle. We emit the LDSM variant when available.
        std::string dtype_ptx = ptx_type_for_dtype(wmma_config_.a_dtype);
        // LDSM loads 8 bytes per thread for .f16 row-major 16x16
        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rk = 0; rk < rep_k; ++rk) {
                int64_t row_off = rm * wm;
                int64_t col_off = rk * wk;
                oss << "    // Load A fragment [" << rm << "][" << rk << "]\n";
                oss << "    ldsm.sync.aligned.m" << wm << "n" << wmma_config_.n
                    << "k" << wk << ".row"
                    << dtype_ptx << "x2"
                    << " {%R_a" << rm << "_" << rk << "_0, %R_a" << rm << "_" << rk << "_1"
                    << ", %R_a" << rm << "_" << rk << "_2, %R_a" << rm << "_" << rk << "_3}"
                    << ", [" << smem_ptr << " + " << (row_off * stride + col_off)
                    << "];\n";
            }
        }
    } else {
        // Legacy WMMA path (Volta/Turing)
        std::string dtype_str = ptx_type_for_dtype(wmma_config_.a_dtype);
        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rk = 0; rk < rep_k; ++rk) {
                int64_t row_off = rm * wm;
                int64_t col_off = rk * wk;
                oss << "    // WMMA load A [" << rm << "][" << rk << "]\n";
                oss << "    wmma.load.a.sync.aligned.row"
                    << ".m" << wm << "n" << wmma_config_.n << "k" << wk
                    << ".global" << dtype_str
                    << " {%R_a" << rm << "_" << rk << "_0, %R_a" << rm << "_" << rk << "_1"
                    << ", %R_a" << rm << "_" << rk << "_2, %R_a" << rm << "_" << rk << "_3}"
                    << ", [" << smem_ptr << " + " << (row_off * stride + col_off)
                    << "], " << stride << ";\n";
            }
        }
    }

    return oss.str();
}

std::string WMMAGenerator::emit_wmma_load_b(
    int64_t tile_k, int64_t tile_n,
    const std::string& smem_ptr, int64_t stride
) const {
    std::ostringstream oss;
    int64_t wk = wmma_config_.k;
    int64_t wn = wmma_config_.n;

    if (tile_k % wk != 0 || tile_n % wn != 0) {
        oss << "/* ERROR: tile_k=" << tile_k << " not multiple of wmma_k=" << wk
            << " or tile_n=" << tile_n << " not multiple of wmma_n=" << wn << " */\n";
        return oss.str();
    }

    int64_t rep_k = tile_k / wk;
    int64_t rep_n = tile_n / wn;

    bool use_mma_sync = (target_.gpu.sm.max_registers_per_thread >= 255);

    if (use_mma_sync) {
        std::string dtype_ptx = ptx_type_for_dtype(wmma_config_.b_dtype);
        for (int64_t rn = 0; rn < rep_n; ++rn) {
            for (int64_t rk = 0; rk < rep_k; ++rk) {
                int64_t row_off = rk * wk;
                int64_t col_off = rn * wn;
                oss << "    // Load B fragment [" << rk << "][" << rn << "]\n";
                oss << "    ldsm.sync.aligned.m" << wmma_config_.m << "n" << wn
                    << "k" << wk << ".col"
                    << dtype_ptx << "x2"
                    << " {%R_b" << rn << "_" << rk << "_0, %R_b" << rn << "_" << rk << "_1}"
                    << ", [" << smem_ptr << " + " << (row_off * stride + col_off)
                    << "];\n";
            }
        }
    } else {
        std::string dtype_str = ptx_type_for_dtype(wmma_config_.b_dtype);
        for (int64_t rn = 0; rn < rep_n; ++rn) {
            for (int64_t rk = 0; rk < rep_k; ++rk) {
                int64_t row_off = rk * wk;
                int64_t col_off = rn * wn;
                oss << "    // WMMA load B [" << rk << "][" << rn << "]\n";
                oss << "    wmma.load.b.sync.aligned.col"
                    << ".m" << wmma_config_.m << "n" << wn << "k" << wk
                    << ".global" << dtype_str
                    << " {%R_b" << rn << "_" << rk << "_0, %R_b" << rn << "_" << rk << "_1}"
                    << ", [" << smem_ptr << " + " << (row_off * stride + col_off)
                    << "], " << stride << ";\n";
            }
        }
    }

    return oss.str();
}

std::string WMMAGenerator::emit_wmma_mma(
    int64_t tile_m, int64_t tile_n, int64_t tile_k
) const {
    std::ostringstream oss;
    int64_t wm = wmma_config_.m;
    int64_t wn = wmma_config_.n;
    int64_t wk = wmma_config_.k;

    if (tile_m % wm != 0 || tile_n % wn != 0 || tile_k % wk != 0) {
        oss << "/* ERROR: tile dimensions not aligned to WMMA dimensions */\n";
        return oss.str();
    }

    int64_t rep_m = tile_m / wm;
    int64_t rep_n = tile_n / wn;
    int64_t rep_k = tile_k / wk;

    bool use_mma_sync = (target_.gpu.sm.max_registers_per_thread >= 255);

    std::string a_ptx = ptx_type_for_dtype(wmma_config_.a_dtype);
    std::string b_ptx = ptx_type_for_dtype(wmma_config_.b_dtype);
    std::string c_ptx = ptx_type_for_dtype(wmma_config_.acc_dtype);

    if (use_mma_sync) {
        // mma.sync.aligned.mMnNkK.row.row (A row-major, B row-major layout)
        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rn = 0; rn < rep_n; ++rn) {
                // Accumulate over K
                for (int64_t rk = 0; rk < rep_k; ++rk) {
                    oss << "    // MMA compute [" << rm << "][" << rn << "][" << rk << "]\n";
                    oss << "    mma.sync.aligned.m" << wm << "n" << wn << "k" << wk
                        << ".row.row" << a_ptx << b_ptx << c_ptx << c_ptx
                        << " {%R_c" << rm << "_" << rn << "_0, %R_c" << rm << "_" << rn << "_1"
                        << ", %R_c" << rm << "_" << rn << "_2, %R_c" << rm << "_" << rn << "_3}"
                        << ", {%R_a" << rm << "_" << rk << "_0, %R_a" << rm << "_" << rk << "_1"
                        << ", %R_a" << rm << "_" << rk << "_2, %R_a" << rm << "_" << rk << "_3}"
                        << ", {%R_b" << rn << "_" << rk << "_0, %R_b" << rn << "_" << rk << "_1}"
                        << ", {%R_c" << rm << "_" << rn << "_0, %R_c" << rm << "_" << rn << "_1"
                        << ", %R_c" << rm << "_" << rn << "_2, %R_c" << rm << "_" << rn << "_3}"
                        << ";\n";
                }
            }
        }
    } else {
        // Legacy WMMA mma instruction (Volta/Turing)
        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rn = 0; rn < rep_n; ++rn) {
                for (int64_t rk = 0; rk < rep_k; ++rk) {
                    oss << "    // WMMA MMA [" << rm << "][" << rn << "][" << rk << "]\n";
                    oss << "    wmma.mma.sync.aligned.row.row"
                        << ".m" << wm << "n" << wn << "k" << wk
                        << a_ptx << b_ptx << c_ptx
                        << " {%R_c" << rm << "_" << rn << "_0, %R_c" << rm << "_" << rn << "_1"
                        << ", %R_c" << rm << "_" << rn << "_2, %R_c" << rm << "_" << rn << "_3}"
                        << ", {%R_a" << rm << "_" << rk << "_0, %R_a" << rm << "_" << rk << "_1"
                        << ", %R_a" << rm << "_" << rk << "_2, %R_a" << rm << "_" << rk << "_3}"
                        << ", {%R_b" << rn << "_" << rk << "_0, %R_b" << rn << "_" << rk << "_1}"
                        << ", {%R_c" << rm << "_" << rn << "_0, %R_c" << rm << "_" << rn << "_1"
                        << ", %R_c" << rm << "_" << rn << "_2, %R_c" << rm << "_" << rn << "_3}"
                        << ";\n";
                }
            }
        }
    }

    return oss.str();
}

std::string WMMAGenerator::emit_wmma_store_c(
    int64_t tile_m, int64_t tile_n,
    const std::string& gmem_ptr, int64_t stride
) const {
    std::ostringstream oss;
    int64_t wm = wmma_config_.m;
    int64_t wn = wmma_config_.n;

    if (tile_m % wm != 0 || tile_n % wn != 0) {
        oss << "/* ERROR: tile dimensions not aligned to WMMA dimensions */\n";
        return oss.str();
    }

    int64_t rep_m = tile_m / wm;
    int64_t rep_n = tile_n / wn;

    std::string c_ptx = ptx_type_for_dtype(wmma_config_.acc_dtype);

    bool use_mma_sync = (target_.gpu.sm.max_registers_per_thread >= 255);

    if (use_mma_sync) {
        // mma.sync: store each C fragment using vectorised global stores.
        //
        // Each thread in a warp holds 4 fp32 accumulator values for a
        // single MMA fragment (m16n8k16).  The per-thread write offset
        // depends on the thread's lane ID within the warp.
        //
        // For m16n8: thread layout is 8 rows x 4 columns.
        //   lane_row = lane_id / 4
        //   lane_col = lane_id % 4
        //   Accumulator i maps to row lane_row + 8*(i/2), col lane_col + (i%2)*4
        //   Byte offset = (lane_row + 8*(i/2)) * stride_bytes + (lane_col + (i%2)*4) * 4
        //
        // We emit register declarations and compute the offsets once at the
        // top of the store block so they can be reused for every C fragment.

        int64_t bpe = bytes_for_dtype(wmma_config_.acc_dtype);  // 4 for fp32

        oss << "    // Per-thread C-matrix offset registers (computed from lane ID)\n";
        oss << "    .reg .u64 %R_c_off_0, %R_c_off_1, %R_c_off_2, %R_c_off_3;\n";
        oss << "    .reg .u64 %c_lane_row, %c_lane_col;\n";
        oss << "    .reg .u64 %c_tmp;\n";
        oss << "    // lane_row = lane_id / 4\n";
        oss << "    shr.u64 %c_lane_row, %rd_lane_id, 2;\n";
        oss << "    // lane_col = lane_id % 4\n";
        oss << "    and.b64  %c_lane_col, %rd_lane_id, 3;\n";
        oss << "\n";

        // offset_0: row lane_row + 0, col lane_col + 0
        //   byte_off = lane_row * stride * bpe + lane_col * bpe
        oss << "    // %R_c_off_0 = lane_row * stride_bytes + lane_col * "
            << bpe << "\n";
        oss << "    mul.lo.u64 %c_tmp, %c_lane_row, " << (stride * bpe) << ";\n";
        oss << "    mad.lo.u64 %R_c_off_0, %c_lane_col, " << bpe
            << ", %c_tmp;\n";

        // offset_1: row lane_row + 0, col lane_col + 4
        //   byte_off = offset_0 + 4 * bpe
        oss << "    // %R_c_off_1 = %R_c_off_0 + " << (4 * bpe) << "\n";
        oss << "    add.u64 %R_c_off_1, %R_c_off_0, " << (4 * bpe) << ";\n";

        // offset_2: row lane_row + 8, col lane_col + 0
        //   byte_off = offset_0 + 8 * stride * bpe
        oss << "    // %R_c_off_2 = %R_c_off_0 + " << (8 * stride * bpe) << "\n";
        oss << "    add.u64 %R_c_off_2, %R_c_off_0, " << (8 * stride * bpe) << ";\n";

        // offset_3: row lane_row + 8, col lane_col + 4
        //   byte_off = offset_2 + 4 * bpe  (= offset_0 + 8*stride*bpe + 4*bpe)
        oss << "    // %R_c_off_3 = %R_c_off_2 + " << (4 * bpe) << "\n";
        oss << "    add.u64 %R_c_off_3, %R_c_off_2, " << (4 * bpe) << ";\n";
        oss << "\n";

        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rn = 0; rn < rep_n; ++rn) {
                int64_t row_off = rm * wm;
                int64_t col_off = rn * wn;
                oss << "    // Store C fragment [" << rm << "][" << rn << "]\n";
                // Each thread holds 4 fp32 values; write them to global memory
                // We emit a series of .f32 stores (or .f16x2 if converting down)
                for (int i = 0; i < 4; ++i) {
                    oss << "    st.global" << c_ptx
                        << " [" << gmem_ptr << " + " << (row_off * stride + col_off)
                        << " + %R_c_off_" << i << "], %R_c" << rm << "_" << rn << "_" << i
                        << ";\n";
                }
            }
        }
    } else {
        // Legacy WMMA store
        for (int64_t rm = 0; rm < rep_m; ++rm) {
            for (int64_t rn = 0; rn < rep_n; ++rn) {
                int64_t row_off = rm * wm;
                int64_t col_off = rn * wn;
                oss << "    // WMMA store C [" << rm << "][" << rn << "]\n";
                oss << "    wmma.store.d.sync.aligned.row"
                    << ".m" << wm << "n" << wn << "k" << wmma_config_.k
                    << ".global" << c_ptx
                    << " [" << gmem_ptr << " + " << (row_off * stride + col_off) << "]"
                    << ", {%R_c" << rm << "_" << rn << "_0, %R_c" << rm << "_" << rn << "_1"
                    << ", %R_c" << rm << "_" << rn << "_2, %R_c" << rm << "_" << rn << "_3}"
                    << ", " << stride << ";\n";
            }
        }
    }

    return oss.str();
}

// ---------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------

int64_t WMMAGenerator::wmma_count(
    int64_t tile_m, int64_t tile_n, int64_t tile_k
) const {
    if (wmma_config_.m == 0 || wmma_config_.n == 0 || wmma_config_.k == 0) return 0;
    if (tile_m % wmma_config_.m != 0 ||
        tile_n % wmma_config_.n != 0 ||
        tile_k % wmma_config_.k != 0) {
        return 0;
    }
    int64_t rep_m = tile_m / wmma_config_.m;
    int64_t rep_n = tile_n / wmma_config_.n;
    int64_t rep_k = tile_k / wmma_config_.k;
    return rep_m * rep_n * rep_k;
}

int64_t WMMAGenerator::register_count(
    int64_t tile_m, int64_t tile_n, int64_t tile_k
) const {
    if (tile_m % wmma_config_.m != 0 ||
        tile_n % wmma_config_.n != 0 ||
        tile_k % wmma_config_.k != 0) {
        return 0;
    }

    int64_t rep_m = tile_m / wmma_config_.m;
    int64_t rep_n = tile_n / wmma_config_.n;
    int64_t rep_k = tile_k / wmma_config_.k;

    // Per-MMA-operation register counts (32-bit registers per thread):
    // A fragment: for m16n8k16 fp16 -> 4 regs (8 fp16 packed as .f16x2)
    //             for m16n8k32 fp16 -> 8 regs (16 fp16 packed as .f16x2)
    // B fragment: for m16n8k16 fp16 -> 2 regs (4 fp16 packed as .f16x2)
    //             for m16n8k32 fp16 -> 4 regs
    // C fragment: 4 regs per MMA (4 fp32 values), but C persists across k iterations

    int64_t a_bpe = bytes_for_dtype(wmma_config_.a_dtype);
    int64_t b_bpe = bytes_for_dtype(wmma_config_.b_dtype);
    int64_t c_bpe = bytes_for_dtype(wmma_config_.acc_dtype);

    // Elements per thread per MMA operation
    int64_t a_elems_per_thread = wmma_config_.a_elements_per_frag / 32;  // 32 threads/warp
    int64_t b_elems_per_thread = wmma_config_.b_elements_per_frag / 32;
    int64_t c_elems_per_thread = wmma_config_.c_elements_per_frag / 32;

    // Registers per thread per MMA operation
    // For packed types (fp16/bf16): 2 elements per 32-bit register
    // For fp32: 1 element per register
    int64_t a_regs_per_mma = (a_elems_per_thread * a_bpe + 3) / 4;  // ceil to 32-bit regs
    int64_t b_regs_per_mma = (b_elems_per_thread * b_bpe + 3) / 4;
    int64_t c_regs_per_mma = (c_elems_per_thread * c_bpe + 3) / 4;

    // A registers: we need one A fragment per (rep_m, rep_k) pair but only
    // rep_k fragments active at a time along K (single K slice).
    // With software pipelining, we may hold two K slices simultaneously.
    // Conservative: all A fragments across K for current stage.
    int64_t total_a_regs = a_regs_per_mma * rep_m * rep_k;
    int64_t total_b_regs = b_regs_per_mma * rep_n * rep_k;

    // C registers: one per (rep_m, rep_n) — persists across all K iterations.
    int64_t total_c_regs = c_regs_per_mma * rep_m * rep_n;

    return total_a_regs + total_b_regs + total_c_regs;
}

} // namespace symplex::codegen
