// SympleX – Polyhedral Rust Engine Bridge
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// This header provides the C interface to the Rust polyhedral engine
// (polyhedral.rs). The C++ PolyhedralOptimizer class delegates to the
// Rust engine for all core optimization passes.
//
// The FFI boundary uses a C-compatible struct (CRustPolyResult) with raw
// pointers that must be freed via poly_free_result(). The C++ wrapper
// class RustPolyResult automatically handles this RAII-style.

#ifndef SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H
#define SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cstring>

namespace symplex::polyhedral::rust {

// ── Rust engine FFI types ──────────────────────────────────────────────────

/// Math domain classification
enum class MathDomain : uint8_t {
    RealFloat = 0,       // Traditional ML: f32, f16, bf16
    ExactFraction = 1,   // Rational Calculus & Number Theory
    SymbolicVariable = 2, // Opaque mathematical symbols
};

/// Hardware target classification
enum class HardwareTarget : uint8_t {
    ServerX86 = 0,
    EdgeDevice = 1,
    TensorAccelerator = 2,
};

/// Element type for mixed-precision
enum class ElementType : uint8_t {
    FP64 = 0,
    FP32 = 1,
    FP16 = 2,
    BF16 = 3,
    INT8 = 4,
    INT4 = 5,
};

/// SIMD level detected at runtime
enum class SimdLevel : uint8_t {
    None = 0,
    SSE2 = 1,
    AVX = 2,
    AVX2FMA = 3,
    AVX512 = 4,
};

/// Transcendental function kind
enum class TranscendentalKind : uint8_t {
    Exp = 0,
    Log = 1,
    Sigmoid = 2,
    Tanh = 3,
    Gelu = 4,
    Relu = 5,
    Silu = 6,
    Softmax = 7,
    Custom = 8,
};

// ── Rust engine configuration ──────────────────────────────────────────────

struct RustPolyConfig {
    MathDomain domain = MathDomain::RealFloat;
    HardwareTarget target = HardwareTarget::ServerX86;
    ElementType compute_type = ElementType::FP32;
    size_t element_bytes = 4;
    bool enable_flash_attention = true;
    bool enable_transcendental_fusion = true;
    bool enable_double_buffering = true;
    bool enable_mixed_precision = false;
    bool enable_ad = false; // automatic differentiation
};

// ── C-linkage FFI result struct (matches Rust's FfiPolyResult) ─────────────
// This is the actual struct returned by the Rust FFI functions.
// It contains raw pointers that must be freed via poly_free_result().

struct CRustPolyResult {
    uint8_t* optimized_instrs;
    size_t optimized_instrs_len;
    uint8_t* hints;
    size_t hints_len;
    size_t instr_count;
    size_t hint_count;
    bool success;
    size_t tile_m;
    size_t tile_n;
    size_t tile_k;
    size_t accumulator_registers;
    size_t prefetch_distance;
    uint8_t simd_level;
    double estimated_gflops;
};

// ── C++ RAII wrapper for the FFI result ────────────────────────────────────
// Automatically frees the Rust-allocated buffers when destroyed.

class RustPolyResult {
public:
    /// Optimized instruction stream (serialized, copied from Rust buffer)
    std::vector<uint8_t> optimized_instrs;
    /// SIMD/AMX hint table (serialized, copied from Rust buffer)
    std::vector<uint8_t> hints;
    /// Number of optimized instructions
    size_t instr_count = 0;
    /// Number of hints emitted
    size_t hint_count = 0;
    /// Whether the optimization was successful
    bool success = false;
    /// Micro-kernel tile M dimension
    size_t tile_m = 0;
    /// Micro-kernel tile N dimension
    size_t tile_n = 0;
    /// Micro-kernel tile K dimension
    size_t tile_k = 0;
    /// Accumulator register count
    size_t accumulator_registers = 0;
    /// Prefetch distance
    size_t prefetch_distance = 0;
    /// SIMD level used
    SimdLevel simd_level = SimdLevel::None;
    /// Estimated roofline GFLOPS
    double estimated_gflops = 0.0;

    RustPolyResult() = default;

    /// Construct from a C-linkage result, copying the data and freeing the Rust buffers.
    explicit RustPolyResult(CRustPolyResult&& raw) {
        success = raw.success;
        instr_count = raw.instr_count;
        hint_count = raw.hint_count;
        tile_m = raw.tile_m;
        tile_n = raw.tile_n;
        tile_k = raw.tile_k;
        accumulator_registers = raw.accumulator_registers;
        prefetch_distance = raw.prefetch_distance;
        simd_level = static_cast<SimdLevel>(raw.simd_level);
        estimated_gflops = raw.estimated_gflops;

        // Copy the instruction bytes from the Rust buffer
        if (raw.optimized_instrs && raw.optimized_instrs_len > 0) {
            optimized_instrs.assign(raw.optimized_instrs,
                                   raw.optimized_instrs + raw.optimized_instrs_len);
        }

        // Copy the hints bytes from the Rust buffer
        if (raw.hints && raw.hints_len > 0) {
            hints.assign(raw.hints, raw.hints + raw.hints_len);
        }

        // Free the Rust-allocated buffers
        poly_free_result(&raw);
    }

    // Non-copyable, movable
    RustPolyResult(const RustPolyResult&) = delete;
    RustPolyResult& operator=(const RustPolyResult&) = delete;
    RustPolyResult(RustPolyResult&&) = default;
    RustPolyResult& operator=(RustPolyResult&&) = default;
};

// ── Rust engine API (extern "C" FFI) ────────────────────────────────────
// These functions are implemented in Rust with #[no_mangle] extern "C".
// The C++ side calls them through this C-linkage wrapper.

extern "C" {

/// Initialize the Rust polyhedral engine.
/// Must be called once before any optimization requests.
void poly_engine_init();

/// Shutdown the Rust polyhedral engine.
/// Frees all Rust-side resources.
void poly_engine_shutdown();

/// Run the standard polyhedral optimization pipeline on a trace.
/// Returns a CRustPolyResult whose buffers must be freed via poly_free_result().
CRustPolyResult poly_optimize_trace(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig* config
);

/// Run the specialized ML/math optimization pipeline.
/// Returns a CRustPolyResult whose buffers must be freed via poly_free_result().
CRustPolyResult poly_optimize_specialized(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig* config
);

/// Construct the adjoint (reverse-mode AD) SCoP.
/// Returns the gradient instruction stream.
/// Returns a CRustPolyResult whose buffers must be freed via poly_free_result().
CRustPolyResult poly_construct_adjoint(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig* config
);

/// Detect the hardware target at runtime.
uint8_t poly_detect_hardware();

/// Detect the SIMD level at runtime.
uint8_t poly_detect_simd_level();

/// Get the MicroKernelConfig for the given hardware target and element size.
/// The output parameters are written via pointers (matching the C ABI).
void poly_get_micro_kernel_config(
    uint8_t target,
    size_t element_bytes,
    size_t* out_tile_m,
    size_t* out_tile_n,
    size_t* out_tile_k,
    size_t* out_acc_regs,
    size_t* out_prefetch_dist
);

/// Free a previously returned CRustPolyResult's allocated buffers.
void poly_free_result(CRustPolyResult* result);

} // extern "C"

// ── C++ convenience wrappers ─────────────────────────────────────────────
// These inline wrappers provide type-safe access to the C-linkage FFI
// functions by converting between C++ enum types and the raw uint8_t
// used in the FFI boundary.

inline void poly_engine_init_wrapper() { poly_engine_init(); }
inline void poly_engine_shutdown_wrapper() { poly_engine_shutdown(); }

inline HardwareTarget poly_detect_hardware_wrapper() {
    return static_cast<HardwareTarget>(poly_detect_hardware());
}

inline SimdLevel poly_detect_simd_level_wrapper() {
    return static_cast<SimdLevel>(poly_detect_simd_level());
}

/// Run the standard optimization pipeline, returning a RAII-managed result.
inline RustPolyResult poly_optimize_trace_wrapper(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
) {
    auto raw = poly_optimize_trace(instr_data, instr_len, &config);
    return RustPolyResult(std::move(raw));
}

/// Run the specialized optimization pipeline, returning a RAII-managed result.
inline RustPolyResult poly_optimize_specialized_wrapper(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
) {
    auto raw = poly_optimize_specialized(instr_data, instr_len, &config);
    return RustPolyResult(std::move(raw));
}

/// Construct the adjoint SCoP, returning a RAII-managed result.
inline RustPolyResult poly_construct_adjoint_wrapper(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
) {
    auto raw = poly_construct_adjoint(instr_data, instr_len, &config);
    return RustPolyResult(std::move(raw));
}

inline void poly_get_micro_kernel_config_wrapper(
    HardwareTarget target,
    size_t element_bytes,
    size_t& out_tile_m,
    size_t& out_tile_n,
    size_t& out_tile_k,
    size_t& out_acc_regs,
    size_t& out_prefetch_dist
) {
    poly_get_micro_kernel_config(
        static_cast<uint8_t>(target),
        element_bytes,
        &out_tile_m,
        &out_tile_n,
        &out_tile_k,
        &out_acc_regs,
        &out_prefetch_dist
    );
}

} // namespace symplex::polyhedral::rust

#endif // SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H
