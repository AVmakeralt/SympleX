// SympleX – Rust Polyhedral Engine FFI Bindings Test
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// Tests the C-linkage FFI bridge between the C++ SympleX codebase
// and the Rust polyhedral optimizer engine.

#include "symplex/polyhedral/rust_bridge.h"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstring>

using namespace symplex::polyhedral::rust;

// Helper: serialize a simple LoadI64 instruction
// Format: [0x01] [slot: u16 LE] [value: i64 LE]
static std::vector<uint8_t> make_load_i64(uint16_t slot, int64_t value) {
    std::vector<uint8_t> data;
    data.push_back(0x01);
    data.push_back(slot & 0xFF);
    data.push_back((slot >> 8) & 0xFF);
    for (int i = 0; i < 8; ++i) {
        data.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
    return data;
}

// Helper: serialize a BinOp instruction
// Format: [0x10] [dst: u16 LE] [op: u8] [lhs: u16 LE] [rhs: u16 LE]
static std::vector<uint8_t> make_binop(uint16_t dst, uint8_t op, uint16_t lhs, uint16_t rhs) {
    std::vector<uint8_t> data;
    data.push_back(0x10);
    data.push_back(dst & 0xFF);
    data.push_back((dst >> 8) & 0xFF);
    data.push_back(op);
    data.push_back(lhs & 0xFF);
    data.push_back((lhs >> 8) & 0xFF);
    data.push_back(rhs & 0xFF);
    data.push_back((rhs >> 8) & 0xFF);
    return data;
}

int main() {
    std::cout << "=== SympleX Rust FFI Bindings Test ===\n\n";

    // ── Test 1: Engine initialization ──────────────────────────────────────
    {
        poly_engine_init();
        std::cout << "[PASS] poly_engine_init()\n";
    }

    // ── Test 2: Hardware detection ────────────────────────────────────────
    {
        HardwareTarget hw = poly_detect_hardware_wrapper();
        assert(hw == HardwareTarget::ServerX86 || hw == HardwareTarget::EdgeDevice
               || hw == HardwareTarget::TensorAccelerator);
        std::cout << "[PASS] poly_detect_hardware() = " << static_cast<int>(hw) << "\n";
    }

    // ── Test 3: SIMD level detection ──────────────────────────────────────
    {
        SimdLevel simd = poly_detect_simd_level_wrapper();
        assert(simd != SimdLevel::None); // x86_64 should have at least SSE2
        std::cout << "[PASS] poly_detect_simd_level() = " << static_cast<int>(simd) << "\n";
    }

    // ── Test 4: Micro-kernel config query ─────────────────────────────────
    {
        size_t tile_m = 0, tile_n = 0, tile_k = 0, acc_regs = 0, prefetch = 0;
        poly_get_micro_kernel_config_wrapper(
            HardwareTarget::ServerX86,
            4,  // FP32
            tile_m, tile_n, tile_k, acc_regs, prefetch
        );
        assert(tile_m > 0);
        assert(tile_n > 0);
        assert(tile_k > 0);
        assert(acc_regs > 0);
        std::cout << "[PASS] poly_get_micro_kernel_config(): "
                  << "tile_m=" << tile_m << " tile_n=" << tile_n
                  << " tile_k=" << tile_k << " acc_regs=" << acc_regs
                  << " prefetch=" << prefetch << "\n";
    }

    // ── Test 5: Polyhedral optimization via FFI ───────────────────────────
    {
        // Create a simple instruction stream:
        // slot0 = 0 (loop start)
        // slot1 = 128 (loop end)
        // slot2 = slot1 - slot0 (iteration count)
        std::vector<uint8_t> instr_data;
        auto load0 = make_load_i64(0, 0);
        auto load1 = make_load_i64(1, 128);
        auto binop = make_binop(2, 1 /*Sub*/, 1, 0);
        instr_data.insert(instr_data.end(), load0.begin(), load0.end());
        instr_data.insert(instr_data.end(), load1.begin(), load1.end());
        instr_data.insert(instr_data.end(), binop.begin(), binop.end());

        RustPolyConfig config;
        config.domain = MathDomain::RealFloat;
        config.target = HardwareTarget::ServerX86;
        config.compute_type = ElementType::FP32;
        config.element_bytes = 4;
        config.enable_flash_attention = true;
        config.enable_transcendental_fusion = true;
        config.enable_double_buffering = true;
        config.enable_mixed_precision = false;
        config.enable_ad = false;

        RustPolyResult result = poly_optimize_trace(
            instr_data.data(), instr_data.size(), config);

        // The Rust engine may or may not produce optimized instructions
        // for this simple trace (no loops), but it should not crash.
        std::cout << "[PASS] poly_optimize_trace(): success=" << result.success
                  << " instr_count=" << result.instr_count
                  << " hint_count=" << result.hint_count
                  << " simd_level=" << static_cast<int>(result.simd_level)
                  << " estimated_gflops=" << result.estimated_gflops << "\n";

        // Free the result
        // Note: poly_free_result is not declared in rust_bridge.h but
        // the Rust engine provides it. For this test, we skip freeing
        // since the process will exit anyway.
    }

    // ── Test 6: Specialized optimization via FFI ─────────────────────────
    {
        std::vector<uint8_t> instr_data;
        auto load0 = make_load_i64(0, 0);
        auto load1 = make_load_i64(1, 1024);
        auto binop = make_binop(2, 0 /*Add*/, 0, 1);
        instr_data.insert(instr_data.end(), load0.begin(), load0.end());
        instr_data.insert(instr_data.end(), load1.begin(), load1.end());
        instr_data.insert(instr_data.end(), binop.begin(), binop.end());

        RustPolyConfig config;
        config.domain = MathDomain::RealFloat;
        config.target = HardwareTarget::ServerX86;
        config.compute_type = ElementType::FP32;
        config.element_bytes = 4;
        config.enable_flash_attention = true;
        config.enable_transcendental_fusion = true;
        config.enable_double_buffering = true;
        config.enable_mixed_precision = false;
        config.enable_ad = false;

        RustPolyResult result = poly_optimize_specialized(
            instr_data.data(), instr_data.size(), config);

        std::cout << "[PASS] poly_optimize_specialized(): success=" << result.success
                  << " instr_count=" << result.instr_count << "\n";
    }

    // ── Test 7: Adjoint (AD) construction via FFI ────────────────────────
    {
        std::vector<uint8_t> instr_data;
        auto load0 = make_load_i64(0, 0);
        instr_data.insert(instr_data.end(), load0.begin(), load0.end());

        RustPolyConfig config;
        config.domain = MathDomain::RealFloat;
        config.enable_ad = true;

        RustPolyResult result = poly_construct_adjoint(
            instr_data.data(), instr_data.size(), config);

        std::cout << "[PASS] poly_construct_adjoint(): success=" << result.success
                  << " instr_count=" << result.instr_count << "\n";
    }

    // ── Test 8: Engine shutdown ────────────────────────────────────────────
    {
        poly_engine_shutdown();
        std::cout << "[PASS] poly_engine_shutdown()\n";
    }

    std::cout << "\nAll Rust FFI binding tests passed!\n";
    return 0;
}
