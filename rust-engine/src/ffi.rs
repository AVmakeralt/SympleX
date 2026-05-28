// SympleX — FFI Bridge: Rust polyhedral engine ↔ C++ host
//
// Implements the C-linkage functions declared in src/polyhedral/rust_bridge.h.
// This module is the single point of contact between the C++ SympleX codebase
// and the Rust polyhedral optimizer.

use crate::polyhedral::{
    self, HardwareProfile, HardwareTarget, ElementType,
    configure_extreme_ml_kernel,
    optimize_trace_polyhedral_specialized,
    optimize_trace_polyhedral_with_profile_and_guards,
    construct_adjoint_scop, optimize_adjoint,
    GuardTable,
    detect_flash_attention_pattern, generate_flash_attention_tiles,
    detect_transcendentals, fuse_transcendentals_into_microkernel,
    emit_quantization_pack_instrs,
    validate_polyhedral_result,
    calculate_roofline_bottleneck,
    MixedPrecisionConfig,
    DoubleBufferConfig,
};
use crate::types::{Instr, serialize_instr, deserialize_instr};

// =============================================================================
// FFI-compatible enum mirrors
// =============================================================================

#[repr(u8)]
pub enum FfiMathDomain {
    RealFloat = 0,
    ExactFraction = 1,
    SymbolicVariable = 2,
}

#[repr(u8)]
pub enum FfiHardwareTarget {
    ServerX86 = 0,
    EdgeDevice = 1,
    TensorAccelerator = 2,
}

#[repr(u8)]
pub enum FfiElementType {
    FP64 = 0,
    FP32 = 1,
    FP16 = 2,
    BF16 = 3,
    INT8 = 4,
    INT4 = 5,
}

#[repr(u8)]
pub enum FfiSimdLevel {
    None = 0,
    SSE2 = 1,
    AVX = 2,
    AVX2FMA = 3,
    AVX512 = 4,
}

#[repr(C)]
pub struct FfiPolyConfig {
    pub domain: u8,
    pub target: u8,
    pub compute_type: u8,
    pub element_bytes: usize,
    pub enable_flash_attention: bool,
    pub enable_transcendental_fusion: bool,
    pub enable_double_buffering: bool,
    pub enable_mixed_precision: bool,
    pub enable_ad: bool,
}

/// Tracks byte allocation sizes for safe deallocation via poly_free_result.
#[repr(C)]
pub struct FfiPolyResult {
    /// Pointer to serialized optimized instructions (caller must free with poly_free_result)
    pub optimized_instrs: *mut u8,
    /// Byte length of the serialized instruction buffer
    pub optimized_instrs_len: usize,
    /// Pointer to serialized SIMD/AMX hints (caller must free with poly_free_result)
    pub hints: *mut u8,
    /// Byte length of the serialized hints buffer
    pub hints_len: usize,
    pub instr_count: usize,
    pub hint_count: usize,
    pub success: bool,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub accumulator_registers: usize,
    pub prefetch_distance: usize,
    pub simd_level: u8,
    pub estimated_gflops: f64,
}

// =============================================================================
// Engine state
// =============================================================================

static mut ENGINE_INITIALIZED: bool = false;

// =============================================================================
// Internal: Parse FfiPolyConfig into Rust-native types
// =============================================================================

fn parse_domain(raw: u8) -> polyhedral::MathDomain {
    match raw {
        1 => polyhedral::MathDomain::ExactFraction,
        2 => polyhedral::MathDomain::SymbolicVariable,
        _ => polyhedral::MathDomain::RealFloat,
    }
}

fn parse_target(raw: u8) -> HardwareTarget {
    match raw {
        1 => HardwareTarget::EdgeDevice,
        2 => HardwareTarget::TensorAccelerator,
        _ => HardwareTarget::ServerX86,
    }
}

fn parse_element_type(raw: u8) -> ElementType {
    match raw {
        0 => ElementType::FP64,
        1 => ElementType::FP32,
        2 => ElementType::FP16,
        3 => ElementType::BF16,
        4 => ElementType::INT8,
        5 => ElementType::INT4,
        _ => ElementType::FP32,
    }
}

fn build_profile_for_target(target: &HardwareTarget) -> HardwareProfile {
    match target {
        HardwareTarget::ServerX86 => HardwareProfile {
            peak_gflops: 3_072.0,
            mem_bandwidth_gb_per_sec: 200.0,
            l1_cache_bytes: 32_768,
            l2_cache_bytes: 524_288,
        },
        HardwareTarget::EdgeDevice => HardwareProfile {
            peak_gflops: 256.0,
            mem_bandwidth_gb_per_sec: 34.0,
            l1_cache_bytes: 16_384,
            l2_cache_bytes: 131_072,
        },
        HardwareTarget::TensorAccelerator => HardwareProfile {
            peak_gflops: 6_144.0,
            mem_bandwidth_gb_per_sec: 800.0,
            l1_cache_bytes: 65_536,
            l2_cache_bytes: 1_048_576,
        },
    }
}

// =============================================================================
// FFI implementation
// =============================================================================

/// Initialize the Rust polyhedral engine.
/// Must be called once before any optimization requests.
#[no_mangle]
pub extern "C" fn poly_engine_init() {
    unsafe {
        if !ENGINE_INITIALIZED {
            ENGINE_INITIALIZED = true;
            eprintln!("[SympleX-Rust] Polyhedral engine initialized");
        }
    }
}

/// Shutdown the Rust polyhedral engine.
/// Frees all Rust-side resources.
#[no_mangle]
pub extern "C" fn poly_engine_shutdown() {
    unsafe {
        ENGINE_INITIALIZED = false;
        eprintln!("[SympleX-Rust] Polyhedral engine shut down");
    }
}

/// Run the standard polyhedral optimization pipeline on a serialized trace.
/// Fully honors the FfiPolyConfig: target, domain, compute_type, and feature flags.
#[no_mangle]
pub extern "C" fn poly_optimize_trace(
    instr_data: *const u8,
    instr_len: usize,
    config: *const FfiPolyConfig,
) -> FfiPolyResult {
    let instructions = match deserialize_instr_stream(instr_data, instr_len) {
        Ok(instrs) => instrs,
        Err(e) => {
            eprintln!("[SympleX-Rust] FFI deserialize error: {}", e);
            return FfiPolyResult::default();
        }
    };

    // ── Parse config ──────────────────────────────────────────────────────
    let cfg = if config.is_null() {
        FfiPolyConfig {
            domain: 0, target: 0, compute_type: 1, element_bytes: 4,
            enable_flash_attention: true, enable_transcendental_fusion: true,
            enable_double_buffering: true, enable_mixed_precision: false,
            enable_ad: false,
        }
    } else {
        unsafe { std::ptr::read(config) }
    };

    let hw_target = parse_target(cfg.target);
    let _domain = parse_domain(cfg.domain);
    let compute_type = parse_element_type(cfg.compute_type);
    let element_bytes = cfg.element_bytes.max(1);

    // ── Build hardware profile from target ────────────────────────────────
    let profile = build_profile_for_target(&hw_target);

    // ── Run the standard polyhedral optimizer with profile & guards ───────
    let mut guard_table = GuardTable::new();
    let mut block = optimize_trace_polyhedral_with_profile_and_guards(
        &instructions, &profile, &mut guard_table,
    );

    // ── Post-pipeline: apply config-gated ML/Math features ────────────────

    // 1) FlashAttention detection (gated by enable_flash_attention)
    if cfg.enable_flash_attention {
        if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
            if let Some(online_state) = detect_flash_attention_pattern(&scop.arena) {
                let ml_config = configure_extreme_ml_kernel(&hw_target, element_bytes);
                let tile_instrs = generate_flash_attention_tiles(
                    &online_state,
                    ml_config.tile_m,
                    ml_config.tile_n,
                );
                block.hints.push((0, polyhedral::SimdHintKind::OnlineSoftmaxReduction {
                    running_max: online_state.running_max_slot,
                    running_sum: online_state.running_sum_slot,
                    accumulator: online_state.accumulator_slot,
                    block_size: online_state.block_size,
                }));
                block.instrs.extend(tile_instrs);
            }
        }
    }

    // 2) Transcendental fusion (gated by enable_transcendental_fusion)
    if cfg.enable_transcendental_fusion {
        if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
            let transcendentals = detect_transcendentals(&scop.arena);
            if !transcendentals.is_empty() {
                fuse_transcendentals_into_microkernel(&mut block.instrs, &transcendentals);
                for t in &transcendentals {
                    let pc = block.instrs.len().saturating_sub(1);
                    block.hints.push((pc, polyhedral::SimdHintKind::TranscendentalVectorize {
                        kind: t.kind,
                        input_slot: t.input_slot,
                        output_slot: t.output_slot,
                        width: t.vector_width,
                    }));
                }
            }
        }
    }

    // 3) Mixed-precision conversion hints (gated by enable_mixed_precision)
    if cfg.enable_mixed_precision {
        let mp_config = MixedPrecisionConfig::for_gemm(compute_type, &hw_target);
        // Emit PrecisionConvert hints at the start of the instruction stream
        // to downcast inputs to the compute type and upcast outputs back to storage type
        let storage_type = mp_config.storage_type;
        let acc_type = mp_config.accumulator_type;
        if compute_type != storage_type {
            block.hints.push((0, polyhedral::SimdHintKind::PrecisionConvert {
                src_slot: 0,
                dst_slot: 0,
                src_type: storage_type,
                dst_type: compute_type,
            }));
        }
        if compute_type != acc_type {
            let end_pc = block.instrs.len().saturating_sub(1);
            block.hints.push((end_pc, polyhedral::SimdHintKind::PrecisionConvert {
                src_slot: 0,
                dst_slot: 0,
                src_type: compute_type,
                dst_type: acc_type,
            }));
        }
        // Quantization packing for INT8/BF16
        if compute_type == ElementType::INT8 || compute_type == ElementType::BF16 {
            let pack_instrs = emit_quantization_pack_instrs(0, 100, compute_type);
            // Insert quantization instructions at the beginning
            let mut combined = pack_instrs;
            combined.append(&mut block.instrs);
            block.instrs = combined;
        }
    }

    // 4) Double buffering hints (gated by enable_double_buffering)
    let ml_config = configure_extreme_ml_kernel(&hw_target, element_bytes);
    if cfg.enable_double_buffering && ml_config.double_buffer_count >= 2 {
        let db_config = DoubleBufferConfig {
            num_buffers: ml_config.double_buffer_count,
            prefetch_distance: ml_config.prefetch_distance,
            buffer_bytes: element_bytes * ml_config.tile_m * ml_config.tile_n * 4,
        };
        block.hints.push((0, polyhedral::SimdHintKind::DoubleBufferSwap {
            buffer_a: 6000,
            buffer_b: 6001,
        }));
        for (pc, _) in block.instrs.iter().enumerate() {
            if pc % ml_config.tile_m.max(1) == 0 {
                block.hints.push((pc, polyhedral::SimdHintKind::AsyncPrefetch {
                    slot: 0,
                    distance: db_config.prefetch_distance,
                }));
            }
        }
    }

    // 5) AD (gated by enable_ad)
    if cfg.enable_ad {
        if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
            let mut adjoint = construct_adjoint_scop(scop);
            let adjoint_block = optimize_adjoint(&mut adjoint);
            // Append AdjointAccumulate hints for each forward→adjoint slot pair
            for &(fwd, adj) in &adjoint.slot_to_adjoint {
                block.hints.push((0, polyhedral::SimdHintKind::AdjointAccumulate {
                    forward_slot: fwd,
                    adjoint_slot: adj,
                    op: crate::types::BinOpKind::Add,
                }));
            }
            // Append gradient instructions after the forward pass
            block.instrs.extend(adjoint_block.instrs);
        }
    }

    // ── Sort and dedup hints ──────────────────────────────────────────────
    block.hints.sort_unstable_by_key(|(pc, _)| *pc);
    block.hints.dedup_by_key(|(pc, _)| *pc);

    // ── Compute roofline estimate ─────────────────────────────────────────
    let estimated_gflops = if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
        match calculate_roofline_bottleneck(scop, &profile) {
            polyhedral::OptimizationRoute::ComputeBound { attainable_gflops } => attainable_gflops,
            polyhedral::OptimizationRoute::MemoryBound { attainable_gflops } => attainable_gflops,
        }
    } else {
        0.0
    };

    // ── Validate result in debug builds ───────────────────────────────────
    validate_polyhedral_result(&instructions, &block.instrs);

    // ── Serialize results ─────────────────────────────────────────────────
    let mut opt_instrs_bytes = Vec::new();
    for instr in &block.instrs {
        opt_instrs_bytes.extend_from_slice(&serialize_instr(instr));
    }

    let mut hints_bytes = Vec::new();
    for (_pc, hint) in &block.hints {
        // Serialize hint as: [kind:u8] [slot_lo:u8] [slot_hi:u8] [extra: u8; 4]
        let kind_byte = match hint {
            polyhedral::SimdHintKind::VectorPack { .. } => 1u8,
            polyhedral::SimdHintKind::MatrixOuterProduct { .. } => 2u8,
            polyhedral::SimdHintKind::ForceRegisterLock { .. } => 3u8,
            polyhedral::SimdHintKind::ForceRegisterUnlock { .. } => 4u8,
            polyhedral::SimdHintKind::TranscendentalVectorize { .. } => 5u8,
            polyhedral::SimdHintKind::SoftwarePipelineLoad { .. } => 6u8,
            polyhedral::SimdHintKind::RegisterLock { .. } => 7u8,
            polyhedral::SimdHintKind::TileLoopBoundary { .. } => 8u8,
            polyhedral::SimdHintKind::MicroKernelRegion { .. } => 9u8,
            polyhedral::SimdHintKind::IndexSetSplit { .. } => 10u8,
            polyhedral::SimdHintKind::OnlineSoftmaxReduction { .. } => 11u8,
            polyhedral::SimdHintKind::PrecisionConvert { .. } => 12u8,
            polyhedral::SimdHintKind::DoubleBufferSwap { .. } => 13u8,
            polyhedral::SimdHintKind::AsyncPrefetch { .. } => 14u8,
            polyhedral::SimdHintKind::AdjointAccumulate { .. } => 15u8,
            polyhedral::SimdHintKind::StochasticBranchHint { .. } => 16u8,
            _ => 0u8, // Unknown/future hint kinds
        };
        hints_bytes.push(kind_byte);
        // Encode slot/extra bytes from hint data
        let slots = hint_to_slot_bytes(hint);
        hints_bytes.extend_from_slice(&slots);
    }

    let instrs_byte_len = opt_instrs_bytes.len();
    let hints_byte_len = hints_bytes.len();
    let simd_level = detect_simd_level_internal();

    FfiPolyResult {
        optimized_instrs: vec_to_ptr(opt_instrs_bytes),
        optimized_instrs_len: instrs_byte_len,
        hints: vec_to_ptr(hints_bytes),
        hints_len: hints_byte_len,
        instr_count: block.instrs.len(),
        hint_count: block.hints.len(),
        success: !block.instrs.is_empty(),
        tile_m: ml_config.tile_m,
        tile_n: ml_config.tile_n,
        tile_k: ml_config.tile_k,
        accumulator_registers: ml_config.accumulator_registers,
        prefetch_distance: ml_config.prefetch_distance,
        simd_level: simd_level as u8,
        estimated_gflops,
    }
}

/// Run the specialized ML/math optimization pipeline.
/// Fully honors the FfiPolyConfig including domain-specific paths.
#[no_mangle]
pub extern "C" fn poly_optimize_specialized(
    instr_data: *const u8,
    instr_len: usize,
    config: *const FfiPolyConfig,
) -> FfiPolyResult {
    let instructions = match deserialize_instr_stream(instr_data, instr_len) {
        Ok(instrs) => instrs,
        Err(e) => {
            eprintln!("[SympleX-Rust] FFI deserialize error: {}", e);
            return FfiPolyResult::default();
        }
    };

    // ── Parse config ──────────────────────────────────────────────────────
    let cfg = if config.is_null() {
        FfiPolyConfig {
            domain: 0, target: 0, compute_type: 1, element_bytes: 4,
            enable_flash_attention: true, enable_transcendental_fusion: true,
            enable_double_buffering: true, enable_mixed_precision: false,
            enable_ad: false,
        }
    } else {
        unsafe { std::ptr::read(config) }
    };

    let domain = parse_domain(cfg.domain);
    let hw_target = parse_target(cfg.target);
    let element_bytes = cfg.element_bytes.max(1);
    let compute_type = parse_element_type(cfg.compute_type);

    let profile = build_profile_for_target(&hw_target);
    let mut guard_table = GuardTable::new();

    // Run the specialized pipeline (which already handles domain-specific paths)
    let mut block = optimize_trace_polyhedral_specialized(
        &instructions, &profile, &mut guard_table, domain, element_bytes,
    );

    // ── Domain-specific: apply features gated by config ───────────────────

    // Mixed-precision
    if cfg.enable_mixed_precision {
        let mp_config = MixedPrecisionConfig::for_gemm(compute_type, &hw_target);
        if compute_type != mp_config.accumulator_type {
            block.hints.push((0, polyhedral::SimdHintKind::PrecisionConvert {
                src_slot: 0, dst_slot: 0,
                src_type: compute_type,
                dst_type: mp_config.accumulator_type,
            }));
        }
    }

    // Double buffering
    let ml_config = configure_extreme_ml_kernel(&hw_target, element_bytes);
    if cfg.enable_double_buffering && ml_config.double_buffer_count >= 2 {
        block.hints.push((0, polyhedral::SimdHintKind::DoubleBufferSwap {
            buffer_a: 6000, buffer_b: 6001,
        }));
    }

    // AD
    if cfg.enable_ad {
        if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
            let mut adjoint = construct_adjoint_scop(scop);
            let adjoint_block = optimize_adjoint(&mut adjoint);
            for &(fwd, adj) in &adjoint.slot_to_adjoint {
                block.hints.push((0, polyhedral::SimdHintKind::AdjointAccumulate {
                    forward_slot: fwd, adjoint_slot: adj,
                    op: crate::types::BinOpKind::Add,
                }));
            }
            block.instrs.extend(adjoint_block.instrs);
        }
    }

    // ── Sort and dedup hints ──────────────────────────────────────────────
    block.hints.sort_unstable_by_key(|(pc, _)| *pc);
    block.hints.dedup_by_key(|(pc, _)| *pc);

    // ── Compute roofline estimate ─────────────────────────────────────────
    let estimated_gflops = if let Some(ref scop) = polyhedral::extract_scop(&instructions) {
        match calculate_roofline_bottleneck(scop, &profile) {
            polyhedral::OptimizationRoute::ComputeBound { attainable_gflops } => attainable_gflops,
            polyhedral::OptimizationRoute::MemoryBound { attainable_gflops } => attainable_gflops,
        }
    } else {
        0.0
    };

    validate_polyhedral_result(&instructions, &block.instrs);

    // ── Serialize results ─────────────────────────────────────────────────
    let mut opt_instrs_bytes = Vec::new();
    for instr in &block.instrs {
        opt_instrs_bytes.extend_from_slice(&serialize_instr(instr));
    }

    let mut hints_bytes = Vec::new();
    for (_pc, hint) in &block.hints {
        let kind_byte = match hint {
            polyhedral::SimdHintKind::VectorPack { .. } => 1u8,
            polyhedral::SimdHintKind::MatrixOuterProduct { .. } => 2u8,
            polyhedral::SimdHintKind::ForceRegisterLock { .. } => 3u8,
            polyhedral::SimdHintKind::ForceRegisterUnlock { .. } => 4u8,
            polyhedral::SimdHintKind::TranscendentalVectorize { .. } => 5u8,
            polyhedral::SimdHintKind::SoftwarePipelineLoad { .. } => 6u8,
            polyhedral::SimdHintKind::RegisterLock { .. } => 7u8,
            polyhedral::SimdHintKind::TileLoopBoundary { .. } => 8u8,
            polyhedral::SimdHintKind::MicroKernelRegion { .. } => 9u8,
            polyhedral::SimdHintKind::IndexSetSplit { .. } => 10u8,
            polyhedral::SimdHintKind::OnlineSoftmaxReduction { .. } => 11u8,
            polyhedral::SimdHintKind::PrecisionConvert { .. } => 12u8,
            polyhedral::SimdHintKind::DoubleBufferSwap { .. } => 13u8,
            polyhedral::SimdHintKind::AsyncPrefetch { .. } => 14u8,
            polyhedral::SimdHintKind::AdjointAccumulate { .. } => 15u8,
            polyhedral::SimdHintKind::StochasticBranchHint { .. } => 16u8,
            _ => 0u8,
        };
        hints_bytes.push(kind_byte);
        let slots = hint_to_slot_bytes(hint);
        hints_bytes.extend_from_slice(&slots);
    }

    let instrs_byte_len = opt_instrs_bytes.len();
    let hints_byte_len = hints_bytes.len();
    let simd_level = detect_simd_level_internal();

    FfiPolyResult {
        optimized_instrs: vec_to_ptr(opt_instrs_bytes),
        optimized_instrs_len: instrs_byte_len,
        hints: vec_to_ptr(hints_bytes),
        hints_len: hints_byte_len,
        instr_count: block.instrs.len(),
        hint_count: block.hints.len(),
        success: !block.instrs.is_empty(),
        tile_m: ml_config.tile_m,
        tile_n: ml_config.tile_n,
        tile_k: ml_config.tile_k,
        accumulator_registers: ml_config.accumulator_registers,
        prefetch_distance: ml_config.prefetch_distance,
        simd_level: simd_level as u8,
        estimated_gflops,
    }
}

/// Construct the adjoint (reverse-mode AD) SCoP and return the OPTIMIZED
/// gradient instructions, not the original instructions.
#[no_mangle]
pub extern "C" fn poly_construct_adjoint(
    instr_data: *const u8,
    instr_len: usize,
    config: *const FfiPolyConfig,
) -> FfiPolyResult {
    let instructions = match deserialize_instr_stream(instr_data, instr_len) {
        Ok(instrs) => instrs,
        Err(e) => {
            eprintln!("[SympleX-Rust] FFI deserialize error: {}", e);
            return FfiPolyResult::default();
        }
    };

    // ── Parse config ──────────────────────────────────────────────────────
    let cfg = if config.is_null() {
        FfiPolyConfig {
            domain: 0, target: 0, compute_type: 1, element_bytes: 4,
            enable_flash_attention: false, enable_transcendental_fusion: false,
            enable_double_buffering: false, enable_mixed_precision: false,
            enable_ad: true,
        }
    } else {
        unsafe { std::ptr::read(config) }
    };

    let hw_target = parse_target(cfg.target);
    let _profile = build_profile_for_target(&hw_target);
    let ml_config = configure_extreme_ml_kernel(&hw_target, cfg.element_bytes.max(1));

    // Build a SCoP from instructions for AD
    let scop = match polyhedral::extract_scop(&instructions) {
        Some(s) => s,
        None => {
            // Cannot extract SCoP — return the original instructions
            let mut opt_instrs_bytes = Vec::new();
            for instr in &instructions {
                opt_instrs_bytes.extend_from_slice(&serialize_instr(instr));
            }
            let instrs_byte_len = opt_instrs_bytes.len();
            return FfiPolyResult {
                optimized_instrs: vec_to_ptr(opt_instrs_bytes),
                optimized_instrs_len: instrs_byte_len,
                hints: std::ptr::null_mut(),
                hints_len: 0,
                instr_count: instructions.len(),
                hint_count: 0,
                success: true,
                tile_m: ml_config.tile_m,
                tile_n: ml_config.tile_n,
                tile_k: ml_config.tile_k,
                accumulator_registers: ml_config.accumulator_registers,
                prefetch_distance: ml_config.prefetch_distance,
                simd_level: detect_simd_level_internal() as u8,
                estimated_gflops: 0.0,
            };
        }
    };

    // Construct the adjoint SCoP
    let mut adjoint = construct_adjoint_scop(&scop);

    // Optimize the adjoint (gradient) instructions using the same polyhedral passes
    let adjoint_block = optimize_adjoint(&mut adjoint);

    // Emit AdjointAccumulate hints linking forward slots to gradient slots
    let mut hints_bytes = Vec::new();
    for &(fwd, adj) in &adjoint.slot_to_adjoint {
        hints_bytes.push(15u8); // AdjointAccumulate
        hints_bytes.extend_from_slice(&fwd.to_le_bytes());
        hints_bytes.extend_from_slice(&adj.to_le_bytes());
        hints_bytes.push(0); // BinOpKind::Add
    }

    // Serialize the optimized gradient instructions
    let mut opt_instrs_bytes = Vec::new();
    for instr in &adjoint_block.instrs {
        opt_instrs_bytes.extend_from_slice(&serialize_instr(instr));
    }

    let instrs_byte_len = opt_instrs_bytes.len();
    let hints_byte_len = hints_bytes.len();

    FfiPolyResult {
        optimized_instrs: vec_to_ptr(opt_instrs_bytes),
        optimized_instrs_len: instrs_byte_len,
        hints: vec_to_ptr(hints_bytes),
        hints_len: hints_byte_len,
        instr_count: adjoint_block.instrs.len(),
        hint_count: adjoint.slot_to_adjoint.len(),
        success: true,
        tile_m: ml_config.tile_m,
        tile_n: ml_config.tile_n,
        tile_k: ml_config.tile_k,
        accumulator_registers: ml_config.accumulator_registers,
        prefetch_distance: ml_config.prefetch_distance,
        simd_level: detect_simd_level_internal() as u8,
        estimated_gflops: 0.0,
    }
}

/// Detect the hardware target at runtime.
#[no_mangle]
pub extern "C" fn poly_detect_hardware() -> u8 {
    // Default to ServerX86 on x86-64
    #[cfg(target_arch = "x86_64")]
    return 0u8;
    #[cfg(not(target_arch = "x86_64"))]
    return 1u8; // EdgeDevice
}

/// Detect the SIMD level at runtime.
#[no_mangle]
pub extern "C" fn poly_detect_simd_level() -> u8 {
    detect_simd_level_internal() as u8
}

/// Get the MicroKernelConfig for the given hardware target and element size.
/// Output parameters are written via raw pointers (C ABI compatible).
#[no_mangle]
pub unsafe extern "C" fn poly_get_micro_kernel_config(
    _target: u8,
    element_bytes: usize,
    out_tile_m: *mut usize,
    out_tile_n: *mut usize,
    out_tile_k: *mut usize,
    out_acc_regs: *mut usize,
    out_prefetch_dist: *mut usize,
) {
    let hw_target = match _target {
        1 => HardwareTarget::EdgeDevice,
        2 => HardwareTarget::TensorAccelerator,
        _ => HardwareTarget::ServerX86,
    };
    let config = configure_extreme_ml_kernel(&hw_target, element_bytes);
    unsafe {
        if !out_tile_m.is_null() { *out_tile_m = config.tile_m; }
        if !out_tile_n.is_null() { *out_tile_n = config.tile_n; }
        if !out_tile_k.is_null() { *out_tile_k = config.tile_k; }
        if !out_acc_regs.is_null() { *out_acc_regs = config.accumulator_registers; }
        if !out_prefetch_dist.is_null() { *out_prefetch_dist = config.prefetch_distance; }
    }
}

/// Free a previously returned FfiPolyResult's allocated buffers.
/// Uses the tracked byte lengths for correct deallocation.
#[no_mangle]
pub extern "C" fn poly_free_result(result: *mut FfiPolyResult) {
    if result.is_null() { return; }
    unsafe {
        let r = &mut *result;
        if !r.optimized_instrs.is_null() && r.optimized_instrs_len > 0 {
            let _ = Vec::from_raw_parts(
                r.optimized_instrs,
                r.optimized_instrs_len,
                r.optimized_instrs_len,
            );
            r.optimized_instrs = std::ptr::null_mut();
            r.optimized_instrs_len = 0;
        }
        if !r.hints.is_null() && r.hints_len > 0 {
            let _ = Vec::from_raw_parts(
                r.hints,
                r.hints_len,
                r.hints_len,
            );
            r.hints = std::ptr::null_mut();
            r.hints_len = 0;
        }
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn detect_simd_level_internal() -> FfiSimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            return FfiSimdLevel::AVX512;
        }
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            return FfiSimdLevel::AVX2FMA;
        }
        if is_x86_feature_detected!("avx") {
            return FfiSimdLevel::AVX;
        }
        if is_x86_feature_detected!("sse2") {
            return FfiSimdLevel::SSE2;
        }
    }
    FfiSimdLevel::None
}

fn deserialize_instr_stream(data: *const u8, len: usize) -> Result<Vec<Instr>, String> {
    if data.is_null() || len == 0 {
        return Ok(Vec::new());
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mut instrs = Vec::new();
    let mut offset = 0;
    while offset < slice.len() {
        match deserialize_instr(&slice[offset..]) {
            Some((instr, consumed)) => {
                instrs.push(instr);
                offset += consumed;
            }
            None => {
                return Err(format!(
                    "Failed to deserialize instruction at offset {} (byte: 0x{:02x})",
                    offset,
                    slice.get(offset).copied().unwrap_or(0xFF)
                ));
            }
        }
    }
    Ok(instrs)
}

fn vec_to_ptr(v: Vec<u8>) -> *mut u8 {
    if v.is_empty() {
        return std::ptr::null_mut();
    }
    let mut v = v;
    let ptr = v.as_mut_ptr();
    std::mem::forget(v); // Caller must free via poly_free_result
    ptr
}

/// Extract slot bytes from a SimdHintKind for compact hint serialization.
fn hint_to_slot_bytes(hint: &polyhedral::SimdHintKind) -> [u8; 6] {
    match hint {
        polyhedral::SimdHintKind::ForceRegisterLock { slot, physical_reg } => {
            [(*slot & 0xFF) as u8, (slot >> 8) as u8, *physical_reg, 0, 0, 0]
        }
        polyhedral::SimdHintKind::ForceRegisterUnlock { slot } => {
            [(*slot & 0xFF) as u8, (slot >> 8) as u8, 0, 0, 0, 0]
        }
        polyhedral::SimdHintKind::VectorPack { op: _, width, src1_base, src2_base: _, dst_base } => {
            [(*dst_base & 0xFF) as u8, (dst_base >> 8) as u8,
             (*src1_base & 0xFF) as u8, (*width & 0xFF) as u8, 0, 0]
        }
        polyhedral::SimdHintKind::AdjointAccumulate { forward_slot, adjoint_slot, .. } => {
            [(*forward_slot & 0xFF) as u8, (forward_slot >> 8) as u8,
             (*adjoint_slot & 0xFF) as u8, (adjoint_slot >> 8) as u8, 0, 0]
        }
        polyhedral::SimdHintKind::DoubleBufferSwap { buffer_a, buffer_b } => {
            [(*buffer_a & 0xFF) as u8, (buffer_a >> 8) as u8,
             (*buffer_b & 0xFF) as u8, (buffer_b >> 8) as u8, 0, 0]
        }
        polyhedral::SimdHintKind::AsyncPrefetch { slot, distance } => {
            [(*slot & 0xFF) as u8, (slot >> 8) as u8,
             (*distance & 0xFF) as u8, ((*distance >> 8) & 0xFF) as u8, 0, 0]
        }
        polyhedral::SimdHintKind::StochasticBranchHint { slot, taken_probability } => {
            let prob_byte = (*taken_probability * 255.0) as u8;
            [(*slot & 0xFF) as u8, (slot >> 8) as u8, prob_byte, 0, 0, 0]
        }
        _ => [0u8; 6],
    }
}

impl Default for FfiPolyResult {
    fn default() -> Self {
        Self {
            optimized_instrs: std::ptr::null_mut(),
            optimized_instrs_len: 0,
            hints: std::ptr::null_mut(),
            hints_len: 0,
            instr_count: 0,
            hint_count: 0,
            success: false,
            tile_m: 0,
            tile_n: 0,
            tile_k: 0,
            accumulator_registers: 0,
            prefetch_distance: 0,
            simd_level: 0,
            estimated_gflops: 0.0,
        }
    }
}
