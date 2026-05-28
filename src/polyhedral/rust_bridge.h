// SympleX – Polyhedral Rust Engine Bridge
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// This header provides the C interface to the Rust polyhedral engine
// (polyhedral.rs). The C++ PolyhedralOptimizer class delegates to the
// Rust engine for all core optimization passes, since the Rust engine
// has the full implementation including:
//   - N-dimensional TensorAccessRelation
//   - UTVPI exact integer solver
//   - Hierarchical 3-tier tiling
//   - AMX/SME micro-kernel emission
//   - Software pipelining
//   - FlashAttention online softmax
//   - Reverse-mode automatic differentiation
//   - Mixed-precision support
//   - Parametric polyhedral boundaries (dynamic shapes)
//   - Exact rational arithmetic (FieldFraction)
//   - And all other ML/Math specialization features
//
// The old C++ implementation is retained as a fallback but the Rust
// engine is the primary optimizer.

#ifndef SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H
#define SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

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
    bool enable_mixed_precision = true;
    bool enable_ad = false; // automatic differentiation
};

// ── Rust engine result ─────────────────────────────────────────────────────

struct RustPolyResult {
    /// Optimized instruction stream (serialized)
    std::vector<uint8_t> optimized_instrs;
    /// SIMD/AMX hint table (serialized)
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
};

// ── Rust engine API (FFI) ──────────────────────────────────────────────────

/// Initialize the Rust polyhedral engine.
/// Must be called once before any optimization requests.
void poly_engine_init();

/// Shutdown the Rust polyhedral engine.
/// Frees all Rust-side resources.
void poly_engine_shutdown();

/// Run the standard polyhedral optimization pipeline on a trace.
/// This is the equivalent of optimize_trace_polyhedral() in Rust.
RustPolyResult poly_optimize_trace(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
);

/// Run the specialized ML/math optimization pipeline.
/// This is the equivalent of optimize_trace_polyhedral_specialized() in Rust.
RustPolyResult poly_optimize_specialized(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
);

/// Construct the adjoint (reverse-mode AD) SCoP.
/// Returns the gradient instruction stream.
RustPolyResult poly_construct_adjoint(
    const uint8_t* instr_data,
    size_t instr_len,
    const RustPolyConfig& config
);

/// Detect the hardware target at runtime.
HardwareTarget poly_detect_hardware();

/// Detect the SIMD level at runtime.
SimdLevel poly_detect_simd_level();

/// Get the MicroKernelConfig for the given hardware target and element size.
void poly_get_micro_kernel_config(
    HardwareTarget target,
    size_t element_bytes,
    size_t& out_tile_m,
    size_t& out_tile_n,
    size_t& out_tile_k,
    size_t& out_acc_regs,
    size_t& out_prefetch_dist
);

} // namespace symplex::polyhedral::rust

#endif // SYMPLEX_POLYHEDRAL_RUST_BRIDGE_H
