// SympleX — PyO3 Python Extension Module
//
// Bridges the Rust polyhedral optimizer into Python via PyO3.
// This is the native `_symplex_core` module imported by the pure-Python
// `symplex` package. It exposes:
//   - optimize_trace()        — run the polyhedral optimizer on an instruction trace
//   - optimize_specialized()  — run the ML/math-specialized pipeline
//   - grad()                  — construct adjoint (reverse-mode AD)
//   - detect_hardware()       — query CPU SIMD level and target
//   - micro_kernel_config()   — query tile sizes for the detected hardware
//   - serialize_instructions() — convert Python trace tuples to binary

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use symplex_engine::types::{Instr, BinOpKind, UnOpKind, serialize_instr, deserialize_instr, Value};
use symplex_engine::phase3_jit::{
    self as p3, NativeCode,
};
use symplex_engine::polyhedral::{
    self, HardwareProfile, HardwareTarget, ElementType, MathDomain,
    GuardTable, MixedPrecisionConfig,
    configure_extreme_ml_kernel,
    optimize_trace_polyhedral_with_profile_and_guards,
    optimize_trace_polyhedral_specialized,
    construct_adjoint_scop, optimize_adjoint,
    detect_flash_attention_pattern, generate_flash_attention_tiles,
    detect_transcendentals, fuse_transcendentals_into_microkernel,
    emit_quantization_pack_instrs,
    validate_polyhedral_result,
    calculate_roofline_bottleneck,
    OptimizationRoute,
    DoubleBufferConfig,
};
use symplex_engine::x86_emitter::{
    self, detect_isa_level, vector_width, ISALevel,
};
use symplex_engine::cuda_backend;
use rayon::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn build_profile(target: &HardwareTarget) -> HardwareProfile {
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

fn parse_target(raw: &str) -> HardwareTarget {
    match raw {
        "edge" => HardwareTarget::EdgeDevice,
        "tensor" => HardwareTarget::TensorAccelerator,
        _ => HardwareTarget::ServerX86,
    }
}

fn parse_element_type(raw: &str) -> ElementType {
    match raw {
        "fp64" => ElementType::FP64,
        "fp16" => ElementType::FP16,
        "bf16" => ElementType::BF16,
        "int8" => ElementType::INT8,
        "int4" => ElementType::INT4,
        _ => ElementType::FP32,
    }
}

fn parse_domain(raw: &str) -> MathDomain {
    match raw {
        "fraction" => MathDomain::ExactFraction,
        "symbolic" => MathDomain::SymbolicVariable,
        _ => MathDomain::RealFloat,
    }
}

fn element_size(et: &ElementType) -> usize {
    match et {
        ElementType::FP64 => 8,
        ElementType::FP32 => 4,
        ElementType::FP16 => 2,
        ElementType::BF16 => 2,
        ElementType::INT8 => 1,
        ElementType::INT4 => 1,
    }
}

fn detect_simd() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            return "avx512".to_string();
        }
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            return "avx2_fma".to_string();
        }
        if is_x86_feature_detected!("avx") {
            return "avx".to_string();
        }
        if is_x86_feature_detected!("sse2") {
            return "sse2".to_string();
        }
    }
    "none".to_string()
}

fn deserialize_stream(data: &[u8]) -> Result<Vec<Instr>, String> {
    let mut instrs = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        match deserialize_instr(&data[offset..]) {
            Some((instr, consumed)) => {
                instrs.push(instr);
                offset += consumed;
            }
            None => {
                return Err(format!(
                    "Failed to deserialize instruction at offset {} (byte: 0x{:02x})",
                    offset,
                    data.get(offset).copied().unwrap_or(0xFF)
                ));
            }
        }
    }
    Ok(instrs)
}

// ── Core optimization pipeline ──────────────────────────────────────────────

struct OptResult {
    opt_instrs: Vec<Instr>,
    hints: Vec<(usize, polyhedral::SimdHintKind)>,
    estimated_gflops: f64,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    accumulator_registers: usize,
    prefetch_distance: usize,
}

fn run_optimize_pipeline(
    instructions: &[Instr],
    target: &HardwareTarget,
    domain: MathDomain,
    element_type: ElementType,
    element_bytes: usize,
    enable_flash_attention: bool,
    enable_transcendental_fusion: bool,
    enable_double_buffering: bool,
    enable_mixed_precision: bool,
    enable_ad: bool,
    specialized: bool,
) -> OptResult {
    let profile = build_profile(target);
    let mut guard_table = GuardTable::new();

    let mut block = if specialized {
        optimize_trace_polyhedral_specialized(
            instructions, &profile, &mut guard_table, domain, element_bytes,
        )
    } else {
        optimize_trace_polyhedral_with_profile_and_guards(
            instructions, &profile, &mut guard_table,
        )
    };

    // 1) FlashAttention
    if enable_flash_attention {
        if let Some(ref scop) = polyhedral::extract_scop(instructions) {
            if let Some(online_state) = detect_flash_attention_pattern(&scop.arena) {
                let ml_config = configure_extreme_ml_kernel(target, element_bytes);
                let tile_instrs = generate_flash_attention_tiles(
                    &online_state, ml_config.tile_m, ml_config.tile_n,
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

    // 2) Transcendental fusion
    if enable_transcendental_fusion {
        if let Some(ref scop) = polyhedral::extract_scop(instructions) {
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

    // 3) Mixed-precision
    if enable_mixed_precision {
        let mp_config = MixedPrecisionConfig::for_gemm(element_type, target);
        let storage_type = mp_config.storage_type;
        let acc_type = mp_config.accumulator_type;
        if element_type != storage_type {
            block.hints.push((0, polyhedral::SimdHintKind::PrecisionConvert {
                src_slot: 0, dst_slot: 0,
                src_type: storage_type, dst_type: element_type,
            }));
        }
        if element_type != acc_type {
            let end_pc = block.instrs.len().saturating_sub(1);
            block.hints.push((end_pc, polyhedral::SimdHintKind::PrecisionConvert {
                src_slot: 0, dst_slot: 0,
                src_type: element_type, dst_type: acc_type,
            }));
        }
        if element_type == ElementType::INT8 || element_type == ElementType::BF16 {
            let pack_instrs = emit_quantization_pack_instrs(0, 100, element_type);
            let mut combined = pack_instrs;
            combined.append(&mut block.instrs);
            block.instrs = combined;
        }
    }

    // 4) Double buffering
    let ml_config = configure_extreme_ml_kernel(target, element_bytes);
    if enable_double_buffering && ml_config.double_buffer_count >= 2 {
        let _db_config = DoubleBufferConfig {
            num_buffers: ml_config.double_buffer_count,
            prefetch_distance: ml_config.prefetch_distance,
            buffer_bytes: element_bytes * ml_config.tile_m * ml_config.tile_n * 4,
        };
        block.hints.push((0, polyhedral::SimdHintKind::DoubleBufferSwap {
            buffer_a: 6000, buffer_b: 6001,
        }));
        for (pc, _) in block.instrs.iter().enumerate() {
            if pc % ml_config.tile_m.max(1) == 0 {
                block.hints.push((pc, polyhedral::SimdHintKind::AsyncPrefetch {
                    slot: 0,
                    distance: ml_config.prefetch_distance,
                }));
            }
        }
    }

    // 5) AD
    if enable_ad {
        if let Some(ref scop) = polyhedral::extract_scop(instructions) {
            let mut adjoint = construct_adjoint_scop(scop);
            let adjoint_block = optimize_adjoint(&mut adjoint);
            for &(fwd, adj) in &adjoint.slot_to_adjoint {
                block.hints.push((0, polyhedral::SimdHintKind::AdjointAccumulate {
                    forward_slot: fwd,
                    adjoint_slot: adj,
                    op: BinOpKind::Add,
                }));
            }
            block.instrs.extend(adjoint_block.instrs);
        }
    }

    // Sort and dedup hints
    block.hints.sort_unstable_by_key(|(pc, _)| *pc);
    block.hints.dedup_by_key(|(pc, _)| *pc);

    // Roofline estimate
    let estimated_gflops = if let Some(ref scop) = polyhedral::extract_scop(instructions) {
        match calculate_roofline_bottleneck(scop, &profile) {
            OptimizationRoute::ComputeBound { attainable_gflops } => attainable_gflops,
            OptimizationRoute::MemoryBound { attainable_gflops } => attainable_gflops,
        }
    } else {
        0.0
    };

    validate_polyhedral_result(instructions, &block.instrs);

    OptResult {
        opt_instrs: block.instrs,
        hints: block.hints,
        estimated_gflops,
        tile_m: ml_config.tile_m,
        tile_n: ml_config.tile_n,
        tile_k: ml_config.tile_k,
        accumulator_registers: ml_config.accumulator_registers,
        prefetch_distance: ml_config.prefetch_distance,
    }
}

fn serialize_hints(hints: &[(usize, polyhedral::SimdHintKind)]) -> Vec<u8> {
    let mut hint_bytes = Vec::new();
    for (_pc, hint) in hints {
        let kind_byte: u8 = match hint {
            polyhedral::SimdHintKind::VectorPack { .. } => 1,
            polyhedral::SimdHintKind::MatrixOuterProduct { .. } => 2,
            polyhedral::SimdHintKind::ForceRegisterLock { .. } => 3,
            polyhedral::SimdHintKind::ForceRegisterUnlock { .. } => 4,
            polyhedral::SimdHintKind::TranscendentalVectorize { .. } => 5,
            polyhedral::SimdHintKind::SoftwarePipelineLoad { .. } => 6,
            polyhedral::SimdHintKind::RegisterLock { .. } => 7,
            polyhedral::SimdHintKind::TileLoopBoundary { .. } => 8,
            polyhedral::SimdHintKind::MicroKernelRegion { .. } => 9,
            polyhedral::SimdHintKind::IndexSetSplit { .. } => 10,
            polyhedral::SimdHintKind::OnlineSoftmaxReduction { .. } => 11,
            polyhedral::SimdHintKind::PrecisionConvert { .. } => 12,
            polyhedral::SimdHintKind::DoubleBufferSwap { .. } => 13,
            polyhedral::SimdHintKind::AsyncPrefetch { .. } => 14,
            polyhedral::SimdHintKind::AdjointAccumulate { .. } => 15,
            polyhedral::SimdHintKind::StochasticBranchHint { .. } => 16,
            _ => 0,
        };
        hint_bytes.push(kind_byte);
        hint_bytes.extend_from_slice(&[0u8; 6]);
    }
    hint_bytes
}

fn serialize_instrs(instrs: &[Instr]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for instr in instrs {
        bytes.extend_from_slice(&serialize_instr(instr));
    }
    bytes
}

fn build_opt_dict<'py>(
    py: Python<'py>,
    res: &OptResult,
    opt_bytes: &[u8],
    hint_bytes: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("optimized_trace", PyBytes::new(py, opt_bytes))?;
    dict.set_item("hints", PyBytes::new(py, hint_bytes))?;
    dict.set_item("instr_count", res.opt_instrs.len())?;
    dict.set_item("hint_count", res.hints.len())?;
    dict.set_item("tile_m", res.tile_m)?;
    dict.set_item("tile_n", res.tile_n)?;
    dict.set_item("tile_k", res.tile_k)?;
    dict.set_item("accumulator_registers", res.accumulator_registers)?;
    dict.set_item("prefetch_distance", res.prefetch_distance)?;
    dict.set_item("simd_level", detect_simd())?;
    dict.set_item("estimated_gflops", res.estimated_gflops)?;
    Ok(dict)
}

// ── Python-exposed functions ────────────────────────────────────────────────

/// Run the polyhedral optimizer on a serialized instruction trace.
#[pyfunction]
#[pyo3(signature = (
    trace_bytes,
    target="server",
    domain="real",
    element_type="fp32",
    enable_flash_attention=true,
    enable_transcendental_fusion=true,
    enable_double_buffering=true,
    enable_mixed_precision=false,
    enable_ad=false,
))]
fn optimize_trace<'py>(
    py: Python<'py>,
    trace_bytes: Bound<'py, PyBytes>,
    target: &str,
    domain: &str,
    element_type: &str,
    enable_flash_attention: bool,
    enable_transcendental_fusion: bool,
    enable_double_buffering: bool,
    enable_mixed_precision: bool,
    enable_ad: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let data = trace_bytes.as_bytes();
    let instructions = deserialize_stream(data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    let hw_target = parse_target(target);
    let math_domain = parse_domain(domain);
    let elem_type = parse_element_type(element_type);
    let ebytes = element_size(&elem_type);

    let res = run_optimize_pipeline(
        &instructions, &hw_target, math_domain, elem_type, ebytes,
        enable_flash_attention, enable_transcendental_fusion,
        enable_double_buffering, enable_mixed_precision, enable_ad,
        false,
    );

    let opt_bytes = serialize_instrs(&res.opt_instrs);
    let hint_bytes = serialize_hints(&res.hints);
    build_opt_dict(py, &res, &opt_bytes, &hint_bytes)
}

/// Run the ML/math-specialized optimization pipeline.
#[pyfunction]
#[pyo3(signature = (
    trace_bytes,
    target="server",
    domain="real",
    element_type="fp32",
    enable_flash_attention=true,
    enable_transcendental_fusion=true,
    enable_double_buffering=true,
    enable_mixed_precision=false,
    enable_ad=false,
))]
fn optimize_specialized<'py>(
    py: Python<'py>,
    trace_bytes: Bound<'py, PyBytes>,
    target: &str,
    domain: &str,
    element_type: &str,
    enable_flash_attention: bool,
    enable_transcendental_fusion: bool,
    enable_double_buffering: bool,
    enable_mixed_precision: bool,
    enable_ad: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let data = trace_bytes.as_bytes();
    let instructions = deserialize_stream(data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    let hw_target = parse_target(target);
    let math_domain = parse_domain(domain);
    let elem_type = parse_element_type(element_type);
    let ebytes = element_size(&elem_type);

    let res = run_optimize_pipeline(
        &instructions, &hw_target, math_domain, elem_type, ebytes,
        enable_flash_attention, enable_transcendental_fusion,
        enable_double_buffering, enable_mixed_precision, enable_ad,
        true,
    );

    let opt_bytes = serialize_instrs(&res.opt_instrs);
    let hint_bytes = serialize_hints(&res.hints);
    build_opt_dict(py, &res, &opt_bytes, &hint_bytes)
}

/// Construct the adjoint (reverse-mode AD) for a trace.
#[pyfunction]
#[pyo3(signature = (trace_bytes, target="server", element_type="fp32"))]
fn grad<'py>(
    py: Python<'py>,
    trace_bytes: Bound<'py, PyBytes>,
    target: &str,
    element_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let data = trace_bytes.as_bytes();
    let instructions = deserialize_stream(data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    let hw_target = parse_target(target);
    let elem_type = parse_element_type(element_type);
    let ebytes = element_size(&elem_type);
    let ml_config = configure_extreme_ml_kernel(&hw_target, ebytes);

    let scop = match polyhedral::extract_scop(&instructions) {
        Some(s) => s,
        None => {
            let opt_bytes = serialize_instrs(&instructions);
            let dict = PyDict::new(py);
            dict.set_item("gradient_trace", PyBytes::new(py, &opt_bytes))?;
            dict.set_item("adjoint_pairs", PyList::empty(py))?;
            dict.set_item("success", false)?;
            return Ok(dict);
        }
    };

    let mut adjoint = construct_adjoint_scop(&scop);
    let adjoint_block = optimize_adjoint(&mut adjoint);

    let pairs: Vec<(u16, u16)> = adjoint.slot_to_adjoint.clone();
    let opt_bytes = serialize_instrs(&adjoint_block.instrs);

    let dict = PyDict::new(py);
    dict.set_item("gradient_trace", PyBytes::new(py, &opt_bytes))?;
    dict.set_item("adjoint_pairs", pairs)?;
    dict.set_item("success", true)?;
    dict.set_item("tile_m", ml_config.tile_m)?;
    dict.set_item("tile_n", ml_config.tile_n)?;

    Ok(dict)
}

/// Detect hardware capabilities at runtime.
#[pyfunction]
fn detect_hardware<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let target = if cfg!(target_arch = "x86_64") { "server" } else { "edge" };
    let hw = parse_target(target);
    let profile = build_profile(&hw);

    let dict = PyDict::new(py);
    dict.set_item("target", target)?;
    dict.set_item("simd_level", detect_simd())?;
    dict.set_item("peak_gflops", profile.peak_gflops)?;
    dict.set_item("mem_bandwidth_gb_per_sec", profile.mem_bandwidth_gb_per_sec)?;
    dict.set_item("l1_cache_bytes", profile.l1_cache_bytes)?;
    dict.set_item("l2_cache_bytes", profile.l2_cache_bytes)?;

    Ok(dict)
}

/// Get the micro-kernel tile configuration.
#[pyfunction]
#[pyo3(signature = (target="server", element_type="fp32"))]
fn micro_kernel_config<'py>(
    py: Python<'py>,
    target: &str,
    element_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let hw_target = parse_target(target);
    let elem_type = parse_element_type(element_type);
    let ebytes = element_size(&elem_type);
    let config = configure_extreme_ml_kernel(&hw_target, ebytes);

    let dict = PyDict::new(py);
    dict.set_item("tile_m", config.tile_m)?;
    dict.set_item("tile_n", config.tile_n)?;
    dict.set_item("tile_k", config.tile_k)?;
    dict.set_item("accumulator_registers", config.accumulator_registers)?;
    dict.set_item("prefetch_distance", config.prefetch_distance)?;
    dict.set_item("double_buffer_count", config.double_buffer_count)?;

    Ok(dict)
}

/// Serialize a list of instruction tuples into the compact binary format.
#[pyfunction]
fn serialize_instructions<'py>(
    py: Python<'py>,
    instrs: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut result = Vec::new();

    for item in &instrs {
        let tuple = item.downcast::<PyTuple>()?;
        let opcode: String = tuple.get_item(0)?.extract()?;

        match opcode.as_str() {
            "load_f64" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: f64 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::LoadF64(slot, val)));
            }
            "load_f32" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: f32 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::LoadF32(slot, val)));
            }
            "load_i64" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: i64 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::LoadI64(slot, val)));
            }
            "load_i32" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: i32 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::LoadI32(slot, val)));
            }
            "load_bool" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: bool = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::LoadBool(slot, val)));
            }
            "binop" => {
                let dst: u16 = tuple.get_item(1)?.extract()?;
                let op_str: String = tuple.get_item(2)?.extract()?;
                let lhs: u16 = tuple.get_item(3)?.extract()?;
                let rhs: u16 = tuple.get_item(4)?.extract()?;
                let op = match op_str.as_str() {
                    "add" => BinOpKind::Add, "sub" => BinOpKind::Sub,
                    "mul" => BinOpKind::Mul, "div" => BinOpKind::Div,
                    "rem" => BinOpKind::Rem, "bitand" => BinOpKind::BitAnd,
                    "bitor" => BinOpKind::BitOr, "bitxor" => BinOpKind::BitXor,
                    "shl" => BinOpKind::Shl, "shr" => BinOpKind::Shr,
                    "eq" => BinOpKind::Eq, "ne" => BinOpKind::Ne,
                    "lt" => BinOpKind::Lt, "le" => BinOpKind::Le,
                    "gt" => BinOpKind::Gt, "ge" => BinOpKind::Ge,
                    "and" => BinOpKind::And, "or" => BinOpKind::Or,
                    "min" => BinOpKind::Min, "max" => BinOpKind::Max,
                    "matmul" => {
                        // Serialize matmul as MatMulInstr (AVX-512 FMA kernel path)
                        result.extend_from_slice(&serialize_instr(&Instr::MatMulInstr(dst, lhs, rhs)));
                        continue;
                    }
                    _ => return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown binop: {}", op_str)
                    )),
                };
                result.extend_from_slice(&serialize_instr(&Instr::BinOp(dst, op, lhs, rhs)));
            }
            "unop" => {
                let dst: u16 = tuple.get_item(1)?.extract()?;
                let op_str: String = tuple.get_item(2)?.extract()?;
                let src: u16 = tuple.get_item(3)?.extract()?;
                let op = match op_str.as_str() {
                    "neg" => UnOpKind::Neg, "not" => UnOpKind::Not,
                    "bitnot" => UnOpKind::BitNot, "abs" => UnOpKind::Abs,
                    _ => return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown unop: {}", op_str)
                    )),
                };
                result.extend_from_slice(&serialize_instr(&Instr::UnOp(dst, op, src)));
            }
            "move" => {
                let dst: u16 = tuple.get_item(1)?.extract()?;
                let src: u16 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::Move(dst, src)));
            }
            "store" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                let val: u16 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::Store(slot, val)));
            }
            "load" => {
                let dst: u16 = tuple.get_item(1)?.extract()?;
                let src: u16 = tuple.get_item(2)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::Load(dst, src)));
            }
            "nop" => {
                result.extend_from_slice(&serialize_instr(&Instr::Nop));
            }
            "return" => {
                let slot: u16 = tuple.get_item(1)?.extract()?;
                result.extend_from_slice(&serialize_instr(&Instr::Return(slot)));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown instruction opcode: {}", opcode)
                ));
            }
        }
    }

    Ok(PyBytes::new(py, &result))
}

// ── JIT Compilation Functions (phase3_jit backend) ──────────────────────────

/// JIT-compile a computation pattern into native x86-64 machine code via phase3_jit.
/// Returns a kernel_id that can be used with jit_execute().
#[pyfunction]
#[pyo3(signature = (
    pattern_type,
    shape_info,
    target="server",
))]
#[allow(unused_variables)]
fn jit_compile<'py>(
    py: Python<'py>,
    pattern_type: &str,
    shape_info: Bound<'py, PyDict>,
    target: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let simd_str = detect_isa_level().to_string();

    let result_dict = PyDict::new(py);
    result_dict.set_item("simd_level", &simd_str)?;
    result_dict.set_item("pattern", pattern_type)?;
    result_dict.set_item("backend", "phase3_jit")?;

    // Build instruction stream based on the pattern type
    let (instructions, param_count) = match pattern_type {
        "elementwise" => {
            // shape_info: { "op": "add"/"sub"/"mul"/"div"/"min"/"max", "n": int }
            let op_str: String = shape_info.get_item("op")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            let op = match op_str.as_str() {
                "add" => BinOpKind::Add,
                "sub" => BinOpKind::Sub,
                "mul" => BinOpKind::Mul,
                "div" => BinOpKind::Div,
                "min" => BinOpKind::Min,
                "max" => BinOpKind::Max,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported elementwise op: {}", op_str)
                )),
            };
            // Build: load two params into slots 0 & 1, apply op → slot 2
            let instrs = vec![
                Instr::LoadF64(0, 0.0),
                Instr::LoadF64(1, 0.0),
                Instr::BinOp(2, op, 0, 1),
            ];
            result_dict.set_item("n_elements", n)?;
            (instrs, 2u16)
        }
        "matmul" => {
            // shape_info: { "M": int, "N": int, "K": int }
            let m: usize = shape_info.get_item("M")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("N")?.unwrap().extract()?;
            let k: usize = shape_info.get_item("K")?.unwrap().extract()?;
            // Build: load three params (A, B, C), multiply A*B → slot 3, add C → slot 4
            let instrs = vec![
                Instr::LoadF64(0, 0.0),  // A element
                Instr::LoadF64(1, 0.0),  // B element
                Instr::LoadF64(2, 0.0),  // C element (accumulator)
                Instr::BinOp(3, BinOpKind::Mul, 0, 1),  // A * B
                Instr::BinOp(4, BinOpKind::Add, 2, 3),  // C + A*B
            ];
            result_dict.set_item("M", m)?;
            result_dict.set_item("N", n)?;
            result_dict.set_item("K", k)?;
            result_dict.set_item("estimated_gflops", 0.0)?;
            (instrs, 3u16)
        }
        "fma" => {
            // shape_info: { "n": int }
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            // Build: load a, b, c; compute a*b + c
            let instrs = vec![
                Instr::LoadF64(0, 0.0),  // a
                Instr::LoadF64(1, 0.0),  // b
                Instr::LoadF64(2, 0.0),  // c
                Instr::BinOp(3, BinOpKind::Mul, 0, 1),  // a * b
                Instr::BinOp(4, BinOpKind::Add, 3, 2),  // a*b + c
            ];
            result_dict.set_item("n_elements", n)?;
            (instrs, 3u16)
        }
        "reduction" => {
            // shape_info: { "op": "add"/"max"/"min", "n": int }
            let op_str: String = shape_info.get_item("op")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            let op = match op_str.as_str() {
                "add" => BinOpKind::Add,
                "max" => BinOpKind::Max,
                "min" => BinOpKind::Min,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported reduction op: {}", op_str)
                )),
            };
            // Build: load two values, apply reduction op
            let instrs = vec![
                Instr::LoadF64(0, 0.0),  // accumulator
                Instr::LoadF64(1, 0.0),  // next value
                Instr::BinOp(2, op, 0, 1),
            ];
            result_dict.set_item("n_elements", n)?;
            (instrs, 2u16)
        }
        _ => {
            result_dict.set_item("success", false)?;
            result_dict.set_item("error", format!("Unknown pattern type: {}", pattern_type))?;
            return Ok(result_dict);
        }
    };

    // Compile via phase3_jit
    let name = "jit_pattern_kernel";
    let mut compiled = p3::compile_ops(name, &instructions)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "phase3_jit compile_ops failed"
        ))?;

    compiled.param_count = param_count;

    // Translate to native code
    let native = p3::translate(&compiled)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "phase3_jit translate failed"
        ))?;

    // Finalize the arena
    p3::finalize_arena();

    let code_size = native.code_size();
    let slot_count = native.slot_count;

    // Store in global kernel table
    let kernel_id = {
        let mut kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let id = kernels.len();
        kernels.push(Some(native));
        id
    };

    result_dict.set_item("kernel_id", kernel_id)?;
    result_dict.set_item("code_size", code_size)?;
    result_dict.set_item("slot_count", slot_count)?;
    result_dict.set_item("param_count", param_count)?;
    result_dict.set_item("success", true)?;
    Ok(result_dict)
}

/// Execute a JIT-compiled kernel with the given arrays via phase3_jit.
/// Returns the result as a NumPy array.
///
/// arrays: list of numpy arrays. For elementwise: [dst, a, b].
/// For matmul: [C, A, B]. For FMA: [dst, a, b, c]. For reduction: [dst, src].
/// All arrays must be contiguous float64.
///
/// Note: This function extracts the first element from each array as f64,
/// executes via phase3_jit, and writes the scalar result back to the first
/// element of the output array. For full array processing, prefer
/// phase3_compile + phase3_execute_int/phase3_execute_f64.
#[pyfunction]
#[pyo3(signature = (kernel_id, arrays))]
#[allow(unused_variables)]
fn jit_execute<'py>(
    py: Python<'py>,
    kernel_id: usize,
    arrays: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Look up the kernel in P3_KERNELS and execute while holding the lock
    // (NativeCode contains executable memory pointers and cannot be cloned)
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;

    // Extract f64 values from the input arrays
    let mut values: Vec<Value> = Vec::new();
    for arr_obj in &arrays {
        let arr = arr_obj.clone();
        // Get the first element as f64
        let item: f64 = arr.call_method1("__getitem__", (0,))?.extract()?;
        values.push(Value::F64(item));
    }

    // Execute the kernel via phase3_jit
    let result = p3::execute(native, &values)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("JIT execution error: {:?}", e)
        ))?;

    // Write result back to the first array's first element
    let result_f64 = match result {
        Value::F64(v) => v,
        Value::I64(v) => v as f64,
        Value::I32(v) => v as f64,
        _ => 0.0,
    };
    arrays[0].call_method1("__setitem__", (0, result_f64))?;

    // Return the first array (which is the output buffer)
    Ok(arrays[0].clone())
}

// ── AVX-512 Multi-Stream JIT Matmul ─────────────────────────────────────────

/// Parallel matmul using rayon — splits M dimension across all CPU cores.
/// Each core computes its rows of C independently (no write conflicts).
#[pyfunction]
#[pyo3(signature = (a_ptr, b_ptr, c_ptr, m, n, k))]
fn parallel_matmul(a_ptr: usize, b_ptr: usize, c_ptr: usize, m: i64, n: i64, k: i64) -> i64 {
    if m == 0 || n == 0 || k == 0 { return 0; }
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr as *const f32, (m * k) as usize);
        let b = std::slice::from_raw_parts(b_ptr as *const f32, (k * n) as usize);
        let c = std::slice::from_raw_parts_mut(c_ptr as *mut f32, (m * n) as usize);
        x86_emitter::parallel_matmul(a, b, c, m as usize, n as usize, k as usize);
    }
    0
}

/// JIT-compiled parallel matmul — uses AVX-512 multi-stream kernel when available.
#[pyfunction]
#[pyo3(signature = (a_ptr, b_ptr, c_ptr, m, n, k))]
fn jit_parallel_matmul(a_ptr: usize, b_ptr: usize, c_ptr: usize, m: i64, n: i64, k: i64) -> i64 {
    if m == 0 || n == 0 || k == 0 { return 0; }
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr as *const f32, (m * k) as usize);
        let b = std::slice::from_raw_parts(b_ptr as *const f32, (k * n) as usize);
        let c = std::slice::from_raw_parts_mut(c_ptr as *mut f32, (m * n) as usize);
        x86_emitter::jit_parallel_matmul(a, b, c, m as usize, n as usize, k as usize);
    }
    0
}

/// Check AVX-512 availability
#[pyfunction]
fn has_avx512() -> bool {
    detect_isa_level() == ISALevel::AVX512
}

/// Check AVX2 availability
#[pyfunction]
fn has_avx2() -> bool {
    let level = detect_isa_level();
    level == ISALevel::AVX2 || level == ISALevel::AVX512
}

/// Get ISA level string
#[pyfunction]
fn detect_isa() -> String {
    detect_isa_level().to_string()
}

/// Get vector width (floats per vector op)
#[pyfunction]
fn vec_width() -> usize {
    vector_width()
}

/// Get number of CPU cores
#[pyfunction]
fn num_cores() -> usize {
    num_cpus::get()
}

/// Get JIT engine info string
#[pyfunction]
fn jit_info() -> String {
    let mut info = String::from("SympleX JIT Engine v3.0.0\nArchitecture: x86-64\n");
    #[cfg(target_arch = "x86_64")]
    {
        info.push_str(&format!("ISA Level: {}\n", detect_isa_level()));
        info.push_str(&format!("Vector Width: {} floats\n", vector_width()));
        info.push_str(&format!("CPU Cores: {}\n", num_cpus::get()));
        info.push_str(if is_x86_feature_detected!("avx2") { "AVX2: available\n" } else { "AVX2: not available\n" });
        if is_x86_feature_detected!("avx512f") { info.push_str("AVX-512F: available\n"); }
        if is_x86_feature_detected!("fma") { info.push_str("FMA3: available\n"); }
    }
    info.push_str("\nOptimization Rules:\n");
    info.push_str("  Y: Multi-stream interleaved AVX-512 (4 ZMM streams)\n");
    info.push_str("  W: Multi-byte NOP stencils (zero μ-ops)\n");
    info.push_str("  K+O: Software-pipelined load-compute interleaving\n");
    info.push_str("  B+U: Context invariant inlining (baked immediates)\n");
    info.push_str("  S: 64-byte cache-line alignment\n");
    info.push_str("\nMulti-threading:\n");
    info.push_str(&format!("  rayon parallel matmul ({} threads)\n", num_cpus::get()));
    info
}

// ── Phase3 JIT — Primary JIT backend (iced-x86, correct encoding) ──────────

use std::sync::Mutex;

/// Global store for phase3 JIT-compiled kernels.
static P3_KERNELS: Mutex<Vec<Option<NativeCode>>> = Mutex::new(Vec::new());

/// Compile an instruction trace via phase3_jit and return a kernel ID.
/// The trace is given as serialized bytes (same format as serialize_instructions).
/// `param_count` specifies how many of the first slots are function parameters.
#[pyfunction]
#[pyo3(signature = (trace_bytes, param_count=0))]
fn phase3_compile<'py>(py: Python<'py>, trace_bytes: Vec<u8>, param_count: u16) -> PyResult<Bound<'py, PyDict>> {
    let data = &trace_bytes;
    let instructions = deserialize_stream(data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    // Compile via phase3_jit
    let name = "p3_kernel";
    let mut compiled = p3::compile_ops(name, &instructions)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "phase3_jit compile_ops failed"
        ))?;

    compiled.param_count = param_count;

    // Translate to native code
    let native = p3::translate(&compiled)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "phase3_jit translate failed"
        ))?;

    // Finalize the arena
    p3::finalize_arena();

    let code_size = native.code_size();
    let slot_count = native.slot_count;

    // Store in global kernel table
    let kernel_id = {
        let mut kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let id = kernels.len();
        kernels.push(Some(native));
        id
    };

    // Build result dict using the GIL we already hold (py parameter)
    let dict = PyDict::new(py);
    dict.set_item("kernel_id", kernel_id)?;
    dict.set_item("code_size", code_size)?;
    dict.set_item("slot_count", slot_count)?;
    dict.set_item("param_count", param_count)?;
    dict.set_item("instr_count", instructions.len())?;
    dict.set_item("backend", "phase3_jit")?;
    dict.set_item("success", true)?;
    Ok(dict)
}

/// Compile an instruction trace via the SSA path of phase3_jit.
/// This uses FlatIrFunction + translate_from_ir which applies
/// parallel-move phi destruction, dominance-based register allocation,
/// and SSA-aware register coalescing — producing higher-quality native code
/// than the standard bytecode path.
///
/// Returns a dict with kernel_id and metadata, same as phase3_compile.
#[pyfunction]
#[pyo3(signature = (trace_bytes, param_count=0))]
fn phase3_compile_ssa<'py>(py: Python<'py>, trace_bytes: Vec<u8>, param_count: u16) -> PyResult<Bound<'py, PyDict>> {
    use symplex_engine::phase3_jit::{
        translate_from_ir, FlatIrFunction, FlatBlock, FlatInstr,
        BlockId, ValueId, IrType, IrOp, EffectFlags, AliasKind, Ownership,
    };

    let instructions = deserialize_stream(&trace_bytes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e)
    })?;

    if instructions.is_empty() {
        let dict = PyDict::new(py);
        dict.set_item("success", false)?;
        dict.set_item("error", "No instructions provided")?;
        return Ok(dict);
    }

    let instr_count = instructions.len();

    // Convert flat instructions to a FlatIrFunction for SSA compilation
    let mut flat_instrs = Vec::new();
    for (i, instr) in instructions.iter().enumerate() {
        let result = match instr {
            Instr::BinOp(d, _, _, _) | Instr::UnOp(d, _, _) => Some(ValueId(i as u32)),
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) => Some(ValueId(i as u32)),
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => Some(ValueId(i as u32)),
            Instr::LoadBool(d, _) => Some(ValueId(i as u32)),
            Instr::Move(d, _) | Instr::Load(d, _) => Some(ValueId(i as u32)),
            _ => None,
        };

        let op = match instr {
            Instr::LoadI32(_, v) => IrOp::ConstInt { value: *v as i64, ty: IrType::Int { width: 32, signed: true } },
            Instr::LoadI64(_, v) => IrOp::ConstInt { value: *v, ty: IrType::Int { width: 64, signed: true } },
            Instr::LoadBool(_, v) => IrOp::ConstBool { value: *v },
            Instr::BinOp(_, op, l, r) => IrOp::BinOp { op: *op, lhs: ValueId(*l as u32), rhs: ValueId(*r as u32) },
            Instr::UnOp(_, op, s) => IrOp::UnOp { op: *op, operand: ValueId(*s as u32) },
            Instr::Move(_, s) => IrOp::Move { src: ValueId(*s as u32) },
            Instr::Return(s) => IrOp::Ret { value: Some(ValueId(*s as u32)) },
            _ => IrOp::Nop,
        };

        flat_instrs.push(FlatInstr {
            result,
            dst: result,
            op,
            effect: EffectFlags::PURE,
            effects: EffectFlags::PURE,
            alias: AliasKind::Unknown,
            ownership: Ownership::Copy,
        });
    }

    let mut func = FlatIrFunction {
        name: "phase3_ssa_kernel".to_string(),
        params: (0..param_count).map(|i| (ValueId(i as u32), IrType::Int { width: 64, signed: true })).collect(),
        ret_ty: IrType::Int { width: 64, signed: true },
        blocks: vec![FlatBlock {
            id: BlockId(0),
            instrs: flat_instrs,
            terminated: false,
            params: Vec::new(),
        }],
        entry: BlockId(0),
        num_values: instr_count as u32,
    };

    let native = translate_from_ir(&mut func)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "phase3_jit SSA compilation failed"
        ))?;

    // Finalize the arena
    p3::finalize_arena();

    let code_size = native.code_size();
    let slot_count = native.slot_count;

    // Store in global kernel table
    let kernel_id = {
        let mut kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let id = kernels.len();
        kernels.push(Some(native));
        id
    };

    let dict = PyDict::new(py);
    dict.set_item("kernel_id", kernel_id)?;
    dict.set_item("code_size", code_size)?;
    dict.set_item("slot_count", slot_count)?;
    dict.set_item("param_count", param_count)?;
    dict.set_item("instr_count", instr_count)?;
    dict.set_item("backend", "phase3_jit_ssa")?;
    dict.set_item("success", true)?;
    Ok(dict)
}

/// Execute a phase3-compiled kernel with integer arguments.
/// Returns the result as an i64.
#[pyfunction]
fn phase3_execute_int(kernel_id: usize, args: Vec<i64>) -> PyResult<i64> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;

    let values: Vec<Value> = args.iter().map(|&v| Value::I64(v)).collect();
    match p3::execute(native, &values) {
        Ok(Value::I64(v)) => Ok(v),
        Ok(Value::I32(v)) => Ok(v as i64),
        Ok(Value::F64(v)) => Ok(v.to_bits() as i64),
        Ok(_) => Ok(0),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("phase3_jit execution error: {:?}", e)
        )),
    }
}

/// Execute a phase3-compiled kernel with f64 arguments.
/// Returns the result as an f64.
#[pyfunction]
fn phase3_execute_f64(kernel_id: usize, args: Vec<f64>) -> PyResult<f64> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;

    let values: Vec<Value> = args.iter().map(|&v| Value::F64(v)).collect();
    match p3::execute(native, &values) {
        Ok(Value::F64(v)) => Ok(v),
        Ok(Value::I64(v)) => Ok(f64::from_bits(v as u64)),
        Ok(Value::I32(v)) => Ok(v as f64),
        Ok(_) => Ok(0.0),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("phase3_jit execution error: {:?}", e)
        )),
    }
}

/// Execute a phase3-compiled kernel over entire arrays (Rust-side element loop).
///
/// This is the critical path that replaces Python's `interpret_trace` loop.
/// Instead of iterating over array elements in Python (which costs ~7.3s for
/// large arrays due to Python interpreter overhead), we loop in Rust:
///
///   for i in 0..n_elements:
///       values[0..param_count] = input_arrays[*][i]  // gather
///       result = execute(kernel, values)              // JIT-compiled
///       output[i] = result                            // scatter
///
/// Parameters:
///   kernel_id: ID from phase3_compile
///   input_ptrs: list of usize pointers to contiguous f64 input arrays
///   output_ptr: usize pointer to contiguous f64 output array
///   n_elements: number of elements to process
///   param_count: how many of the first slots are function parameters
#[pyfunction]
#[pyo3(signature = (kernel_id, input_ptrs, output_ptr, n_elements, param_count))]
fn phase3_execute_arrays(
    kernel_id: usize,
    input_ptrs: Vec<usize>,
    output_ptr: usize,
    n_elements: usize,
    param_count: u16,
) -> PyResult<f64> {
    // Acquire the lock, get the kernel's function pointer and code size, then release
    let (func_ptr, code_size) = {
        let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let native = kernels.get(kernel_id)
            .and_then(|k| k.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("Invalid kernel_id: {}", kernel_id)
            ))?;
        (native.mem_entry(), native.code_size())
    };

    let n_params = param_count as usize;
    if input_ptrs.len() < n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Expected {} input pointers, got {}", n_params, input_ptrs.len())
        ));
    }

    // Build raw pointer array for inputs
    let input_slices: Vec<*const f64> = input_ptrs.iter()
        .map(|&p| p as *const f64)
        .collect();

    let output_slice = output_ptr as *mut f64;

    let start = std::time::Instant::now();

    unsafe {
        // Build a NativeCode-like wrapper just for execution
        // We have the function pointer, so we can call it directly
        let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(func_ptr);

        for i in 0..n_elements {
            // Gather input values into a slot array
            let mut slots: Vec<i64> = vec![0i64; 256]; // generous slot buffer
            for p in 0..n_params {
                let val = *input_slices[p].add(i);
                slots[p] = val.to_bits() as i64;
            }

            // Execute compiled kernel
            let result_bits = f(slots.as_mut_ptr());
            let out_val = f64::from_bits(result_bits as u64);
            *output_slice.add(i) = out_val;
        }
    }

    // Prevent code_size from being unused
    let _ = code_size;

    let elapsed = start.elapsed().as_secs_f64();
    Ok(elapsed)
}

/// Compile and execute a trace over arrays in one step (end-to-end JIT path).
///
/// This function wires the entire JIT pipeline:
///   Python trace → serialize → phase3_jit compile → native x86-64 → execute loop
///
/// Parameters:
///   trace_bytes: serialized instruction trace
///   input_ptrs: list of usize pointers to contiguous f64 input arrays
///   output_ptr: usize pointer to contiguous f64 output array
///   n_elements: number of elements to process
///   param_count: how many of the first slots are function parameters
///
/// Returns: dict with kernel_id, exec_time_seconds, success
#[pyfunction]
#[pyo3(signature = (trace_bytes, input_ptrs, output_ptr, n_elements, param_count))]
fn trace_jit_execute<'py>(
    py: Python<'py>,
    trace_bytes: Vec<u8>,
    input_ptrs: Vec<usize>,
    output_ptr: usize,
    n_elements: usize,
    param_count: u16,
) -> PyResult<Bound<'py, PyDict>> {
    // Step 1: Compile the trace
    let compile_dict = phase3_compile(py, trace_bytes, param_count)?;
    let kernel_id: usize = compile_dict.get_item("kernel_id")?.unwrap().extract()?;
    let success: bool = compile_dict.get_item("success")?.unwrap().extract()?;

    if !success {
        let dict = PyDict::new(py);
        dict.set_item("success", false)?;
        dict.set_item("error", "phase3_compile failed")?;
        return Ok(dict);
    }

    // Step 2: Execute over arrays
    let exec_time = phase3_execute_arrays(kernel_id, input_ptrs, output_ptr, n_elements, param_count)?;

    let dict = PyDict::new(py);
    dict.set_item("success", true)?;
    dict.set_item("kernel_id", kernel_id)?;
    dict.set_item("exec_time_seconds", exec_time)?;
    dict.set_item("n_elements", n_elements)?;
    Ok(dict)
}

/// Benchmark a phase3-compiled kernel with integer arguments.
/// Returns seconds per iteration.
#[pyfunction]
fn phase3_bench_int(kernel_id: usize, args: Vec<i64>, iters: usize) -> PyResult<f64> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;

    let values: Vec<Value> = args.iter().map(|&v| Value::I64(v)).collect();

    // Warmup
    for _ in 0..100 {
        let _ = p3::execute(native, &values);
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = p3::execute(native, &values);
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(elapsed / iters as f64)
}

/// Compile and run an instruction trace in one step via phase3_jit (integer args).
#[pyfunction]
fn phase3_run_int(py: Python<'_>, trace_bytes: Vec<u8>, args: Vec<i64>, param_count: u16) -> PyResult<i64> {
    let dict = phase3_compile(py, trace_bytes, param_count)?;
    let kernel_id: usize = dict.get_item("kernel_id")?.unwrap().extract()?;
    phase3_execute_int(kernel_id, args)
}

/// Verify integrity of a phase3-compiled kernel's machine code.
#[pyfunction]
fn phase3_verify(kernel_id: usize) -> PyResult<bool> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;
    Ok(native.verify_integrity())
}

/// Get the code size of a phase3-compiled kernel.
#[pyfunction]
fn phase3_code_size(kernel_id: usize) -> PyResult<usize> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;
    Ok(native.code_size())
}

/// Dump the first N bytes of a phase3-compiled kernel as hex string.
#[pyfunction]
#[pyo3(signature = (kernel_id, max_bytes=64))]
fn phase3_dump_code(kernel_id: usize, max_bytes: usize) -> PyResult<String> {
    let kernels = P3_KERNELS.lock().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let native = kernels.get(kernel_id)
        .and_then(|k| k.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid kernel_id: {}", kernel_id)
        ))?;

    let func_ptr = native.mem_entry() as *const u8;
    let len = native.code_size().min(max_bytes);
    if len == 0 || func_ptr.is_null() {
        return Ok("empty".to_string());
    }
    let code = unsafe { std::slice::from_raw_parts(func_ptr, len) };
    let hex: Vec<String> = code.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(hex.join(" "))
}

// ── SIMD Elementwise f64 Execution ──────────────────────────────────────────

/// Execute a SIMD-accelerated elementwise operation on f64 arrays.
/// op: "add"=0, "sub"=1, "mul"=2, "div"=3, "min"=4, "max"=5
/// dst_ptr, a_ptr, b_ptr: raw pointers to contiguous f64 arrays
/// n: number of elements
/// Returns elapsed time in seconds.
#[pyfunction]
#[pyo3(signature = (op, dst_ptr, a_ptr, b_ptr, n))]
fn simd_elementwise_f64(op: &str, dst_ptr: usize, a_ptr: usize, b_ptr: usize, n: usize) -> PyResult<f64> {
    let op_code: u8 = match op {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        "min" => 4,
        "max" => 5,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported elementwise op: {}. Supported: add, sub, mul, div, min, max", op)
        )),
    };
    let elapsed = x86_emitter::simd_elementwise_f64(op_code, dst_ptr, a_ptr, b_ptr, n);
    if elapsed < 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SIMD elementwise kernel execution failed"
        ));
    }
    Ok(elapsed)
}

/// Get the ISA level used for SIMD elementwise operations.
#[pyfunction]
fn simd_elementwise_isa() -> String {
    let level = x86_emitter::detect_isa_level();
    match level {
        x86_emitter::ISALevel::AVX512 => format!("AVX512_f64_8wide/f32_16wide"),
        x86_emitter::ISALevel::AVX2 => format!("AVX2_f64_4wide/f32_8wide"),
        x86_emitter::ISALevel::SSE => format!("SSE2_f64_scalar/SSE_f32_scalar"),
    }
}

/// Execute a SIMD-accelerated elementwise operation on f32 arrays.
/// op: "add"=0, "sub"=1, "mul"=2, "div"=3, "min"=4, "max"=5
/// dst_ptr, a_ptr, b_ptr: raw pointers to contiguous f32 arrays
/// n: number of elements
/// Returns elapsed time in seconds.
#[pyfunction]
#[pyo3(signature = (op, dst_ptr, a_ptr, b_ptr, n))]
fn simd_elementwise_f32(op: &str, dst_ptr: usize, a_ptr: usize, b_ptr: usize, n: usize) -> PyResult<f64> {
    let op_code: u8 = match op {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        "min" => 4,
        "max" => 5,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported elementwise op: {}. Supported: add, sub, mul, div, min, max", op)
        )),
    };
    let elapsed = x86_emitter::simd_elementwise_f32(op_code, dst_ptr, a_ptr, b_ptr, n);
    if elapsed < 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SIMD elementwise f32 kernel execution failed"
        ));
    }
    Ok(elapsed)
}

/// Execute a SIMD-accelerated reduction on an f32 array.
/// op: "sum"=0, "max"=1, "min"=2
/// data_ptr: raw pointer to contiguous f32 array
/// n: number of elements
/// Returns the reduced scalar value (as f64).
#[pyfunction]
#[pyo3(signature = (op, data_ptr, n))]
fn simd_reduce_f32(op: &str, data_ptr: usize, n: usize) -> PyResult<f64> {
    let op_code: u8 = match op {
        "sum" => 0,
        "max" => 1,
        "min" => 2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported reduction op: {}. Supported: sum, max, min", op)
        )),
    };
    let result = x86_emitter::simd_reduce_f32(op_code, data_ptr, n);
    if result.is_nan() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SIMD reduction f32 kernel execution failed"
        ));
    }
    Ok(result)
}

/// Execute a SIMD-accelerated reduction on an f64 array.
/// op: "sum"=0, "max"=1, "min"=2
/// data_ptr: raw pointer to contiguous f64 array
/// n: number of elements
/// Returns the reduced scalar value.
#[pyfunction]
#[pyo3(signature = (op, data_ptr, n))]
fn simd_reduce_f64(op: &str, data_ptr: usize, n: usize) -> PyResult<f64> {
    let op_code: u8 = match op {
        "sum" => 0,
        "max" => 1,
        "min" => 2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported reduction op: {}. Supported: sum, max, min", op)
        )),
    };
    let result = x86_emitter::simd_reduce_f64(op_code, data_ptr, n);
    if result.is_nan() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SIMD reduction f64 kernel execution failed"
        ));
    }
    Ok(result)
}

/// Execute a fused chain of elementwise operations in a single pass for f32 arrays.
///
/// This is the key performance optimization: instead of writing intermediate
/// results to memory and reading them back, all ops are computed per-element
/// in SIMD registers. For a chain like x*2.0+1.0 -> sum:
///   Old: 3 passes x 800MB = 2.4GB traffic + 2 temp arrays
///   New: 1 pass x 800MB = 800MB traffic, 0 temp arrays
///
/// ops: list of (op, lhs_src, lhs_idx, rhs_src, rhs_idx) tuples
///   op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
///   lhs_src/rhs_src: 0=input_array, 1=constant, 2=previous_op_result
/// input_ptrs: raw pointers to input f32 arrays
/// constants: f32 constant values
/// n: element count
/// reduce_op: 0=sum, 1=max, 2=min, 255=no reduce (write to dst)
/// dst_ptr: output array pointer (used when reduce_op == 255)
#[pyfunction]
fn simd_fused_elementwise_f32(
    ops: Vec<(u8, u8, u8, u8, u8)>,
    input_ptrs: Vec<usize>,
    constants: Vec<f32>,
    n: usize,
    reduce_op: u8,
    dst_ptr: usize,
) -> f64 {
    x86_emitter::simd_fused_elementwise_f32(ops, input_ptrs, constants, n, reduce_op, dst_ptr)
}

/// Execute a fused chain of elementwise operations in a single pass for f64 arrays.
/// Same as simd_fused_elementwise_f32 but for double-precision data.
#[pyfunction]
fn simd_fused_elementwise_f64(
    ops: Vec<(u8, u8, u8, u8, u8)>,
    input_ptrs: Vec<usize>,
    constants: Vec<f64>,
    n: usize,
    reduce_op: u8,
    dst_ptr: usize,
) -> f64 {
    x86_emitter::simd_fused_elementwise_f64(ops, input_ptrs, constants, n, reduce_op, dst_ptr)
}

// ── CUDA Backend — GPU execution functions ──────────────────────────────────

/// Check if CUDA is available on this system.
#[pyfunction]
fn cuda_available() -> bool {
    cuda_backend::CudaRuntime::is_available()
}

/// Get GPU device information as a dictionary.
#[pyfunction]
fn cuda_device_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match cuda_backend::CudaRuntime::new(0) {
        Ok(rt) => {
            match rt.device_info() {
                Ok(info) => {
                    dict.set_item("available", true)?;
                    dict.set_item("name", &info.name)?;
                    dict.set_item("sm_arch", info.sm_arch())?;
                    dict.set_item("compute_capability_major", info.compute_capability_major)?;
                    dict.set_item("compute_capability_minor", info.compute_capability_minor)?;
                    dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
                    dict.set_item("total_memory_gb", info.total_memory_bytes as f64 / 1e9)?;
                    dict.set_item("num_sms", info.num_sms)?;
                    dict.set_item("num_cuda_cores", info.num_cuda_cores())?;
                    dict.set_item("warp_size", info.warp_size)?;
                    dict.set_item("max_threads_per_block", info.max_threads_per_block)?;
                    dict.set_item("max_shared_memory_per_block", info.max_shared_memory_per_block)?;
                    dict.set_item("clock_mhz", info.clock_mhz)?;
                }
                Err(e) => {
                    dict.set_item("available", false)?;
                    dict.set_item("error", format!("{}", e))?;
                }
            }
        }
        Err(e) => {
            dict.set_item("available", false)?;
            dict.set_item("error", format!("{}", e))?;
        }
    }
    Ok(dict)
}

/// Compile a CUDA kernel for a given pattern type. Returns the PTX source as a string.
///
/// pattern_type: "matmul", "elementwise", "fma", "reduction"
/// shape_info: dict with kernel-specific parameters
/// sm_arch: target architecture (default "sm_80")
#[pyfunction]
#[pyo3(signature = (pattern_type, shape_info, sm_arch="sm_80"))]
fn cuda_compile_ptx<'py>(
    _py: Python<'py>,
    pattern_type: &str,
    shape_info: Bound<'py, PyDict>,
    sm_arch: &str,
) -> PyResult<String> {
    let generator = cuda_backend::PtxGenerator::new(sm_arch);

    let kernel = match pattern_type {
        "matmul" => {
            let m: usize = shape_info.get_item("M")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("N")?.unwrap().extract()?;
            let k: usize = shape_info.get_item("K")?.unwrap().extract()?;
            let tile_m: usize = match shape_info.get_item("tile_m") {
                Ok(Some(v)) => v.extract().unwrap_or(32),
                _ => 32,
            };
            let tile_n: usize = match shape_info.get_item("tile_n") {
                Ok(Some(v)) => v.extract().unwrap_or(32),
                _ => 32,
            };
            let tile_k: usize = match shape_info.get_item("tile_k") {
                Ok(Some(v)) => v.extract().unwrap_or(8),
                _ => 8,
            };
            generator.gen_matmul(m, n, k, tile_m, tile_n, tile_k)
        }
        "elementwise" => {
            let op_str: String = shape_info.get_item("op")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            let op = match op_str.as_str() {
                "add" => BinOpKind::Add, "sub" => BinOpKind::Sub,
                "mul" => BinOpKind::Mul, "div" => BinOpKind::Div,
                "min" => BinOpKind::Min, "max" => BinOpKind::Max,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported elementwise op: {}", op_str)
                )),
            };
            generator.gen_elementwise(op, n)
        }
        "fma" => {
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            generator.gen_fma(n)
        }
        "reduction" => {
            let op_str: String = shape_info.get_item("op")?.unwrap().extract()?;
            let n: usize = shape_info.get_item("n")?.unwrap().extract()?;
            let op = match op_str.as_str() {
                "add" => BinOpKind::Add,
                "max" => BinOpKind::Max,
                "min" => BinOpKind::Min,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported reduction op: {}", op_str)
                )),
            };
            generator.gen_reduction(op, n)
        }
        "stencil" => {
            let rows: usize = shape_info.get_item("rows")?.unwrap().extract()?;
            let cols: usize = shape_info.get_item("cols")?.unwrap().extract()?;
            generator.gen_stencil(rows, cols)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown pattern type: {}. Supported: matmul, elementwise, fma, reduction, stencil", pattern_type)
            ));
        }
    };

    Ok(kernel.ptx_source)
}

/// Execute a matmul on GPU: C = A × B (raw pointer interface).
/// Returns 0 on success, -1 on CUDA error.
#[pyfunction]
#[pyo3(signature = (a_ptr, b_ptr, c_ptr, m, n, k))]
fn cuda_matmul(a_ptr: usize, b_ptr: usize, c_ptr: usize, m: i64, n: i64, k: i64) -> i64 {
    if m == 0 || n == 0 || k == 0 { return 0; }
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr as *const f32, (m * k) as usize);
        let b = std::slice::from_raw_parts(b_ptr as *const f32, (k * n) as usize);
        let c = std::slice::from_raw_parts_mut(c_ptr as *mut f32, (m * n) as usize);
        match cuda_backend::cuda_matmul(a, b, c, m as usize, n as usize, k as usize) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }
}

/// Execute a 5-point stencil on GPU (raw pointer interface).
/// Returns 0 on success, -1 on CUDA error.
///
/// src_ptr: pointer to source float32 array (rows x cols)
/// dst_ptr: pointer to destination float32 array ((rows-2) x (cols-2))
/// rows: number of rows in source array
/// cols: number of columns in source array
#[pyfunction]
#[pyo3(signature = (src_ptr, dst_ptr, rows, cols))]
fn cuda_stencil(src_ptr: usize, dst_ptr: usize, rows: i64, cols: i64) -> i64 {
    if rows < 3 || cols < 3 { return 0; }
    let rows = rows as usize;
    let cols = cols as usize;
    let out_rows = rows - 2;
    let out_cols = cols - 2;
    unsafe {
        let src = std::slice::from_raw_parts(src_ptr as *const f32, rows * cols);
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, out_rows * out_cols);
        match cuda_backend::cuda_stencil(src, dst, rows, cols) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }
}

/// Get CUDA backend info string.
#[pyfunction]
fn cuda_info() -> String {
    let mut info = String::from("SympleX CUDA Backend v1.0.0\n");
    if cfg!(feature = "cuda") {
        info.push_str("CUDA Feature: enabled (rebuild with --features cuda for GPU execution)\n");
    } else {
        info.push_str("CUDA Feature: not enabled (rebuild with --features cuda)\n");
    }
    if cuda_backend::CudaRuntime::is_available() {
        match cuda_backend::CudaRuntime::new(0) {
            Ok(rt) => {
                if let Ok(dev) = rt.device_info() {
                    info.push_str(&format!("GPU: {}\n", dev.name));
                    info.push_str(&format!("SM: {}\n", dev.sm_arch()));
                    info.push_str(&format!("Memory: {:.1} GB\n", dev.total_memory_bytes as f64 / 1e9));
                    info.push_str(&format!("SMs: {}\n", dev.num_sms));
                }
            }
            Err(e) => info.push_str(&format!("Error: {}\n", e)),
        }
    } else {
        info.push_str("GPU: not available (PTX generation still works)\n");
    }
    info.push_str("\nSupported Kernels:\n");
    info.push_str("  MatMul (tiled, shared memory, K-loop)\n");
    info.push_str("  Elementwise (add, sub, mul, div, min, max)\n");
    info.push_str("  FMA (fused multiply-add)\n");
    info.push_str("  Reduction (add, min, max)\n");
    info.push_str("  Conv2D (im2col GEMM)\n");
    info.push_str("  5-Point Stencil (2D, register rotation)\n");
    info
}

// ── Fused Stencil Compute — Single-buffer 5-point stencil with register rotation ──

/// Execute a fused 5-point stencil on a single contiguous 2D buffer.
///
/// Instead of passing 5 separate sliced array views (which causes 5 independent
/// memory streams totaling 2GB+ bandwidth), this function takes ONE contiguous
/// buffer and computes the stencil with register rotation:
///
///   out[i][j] = 0.2 * (src[i][j] + src[i-1][j] + src[i+1][j] + src[i][j-1] + src[i][j+1])
///
/// Register rotation: the center value (src[i][j]) loaded for position j is reused
/// as the west value (src[i][j-1]) for position j+1, eliminating one memory read
/// per element. Combined with row-pointer caching, this reduces memory reads from
/// 5 per element to effectively 3 per element.
///
/// Parameters:
///   src_ptr: pointer to source float32 array (rows x cols)
///   dst_ptr: pointer to destination float32 array (rows-2 x cols-2)
///   rows: number of rows in source array
///   cols: number of columns in source array
///
/// Uses rayon for parallel row processing.
#[pyfunction]
#[pyo3(signature = (src_ptr, dst_ptr, rows, cols))]
fn stencil_compute(src_ptr: usize, dst_ptr: usize, rows: i64, cols: i64) -> i64 {
    if rows < 3 || cols < 3 { return 0; }
    let rows = rows as usize;
    let cols = cols as usize;
    let out_rows = rows - 2;
    let out_cols = cols - 2;

    unsafe {
        let src = std::slice::from_raw_parts(src_ptr as *const f32, rows * cols);
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, out_rows * out_cols);

        // Parallelize over output rows using rayon
        dst.par_chunks_mut(out_cols)
            .enumerate()
            .for_each(|(out_i, row_dst)| {
                let i = out_i + 1; // source row index
                let row_center = &src[i * cols..];
                let row_north = &src[(i - 1) * cols..];
                let row_south = &src[(i + 1) * cols..];

                // Process the inner stencil with register rotation:
                // For the first element, we need all 5 values.
                // For subsequent elements, the "center" of j-1 becomes the "west" of j.
                let mut west = row_center[0]; // src[i][0] — will be west for j=1
                for j in 1..cols - 1 {
                    let center = row_center[j];
                    let north = row_north[j];
                    let south = row_south[j];
                    let east = row_center[j + 1];
                    // west = row_center[j-1], but we cached it from the previous iteration
                    row_dst[j - 1] = 0.2 * (center + north + south + west + east);
                    west = center; // rotate: current center becomes next iteration's west
                }
            });
    }
    0
}

/// Execute a fused 5-point stencil with multiple accumulators for better ILP.
///
/// Same as stencil_compute but unrolls the inner loop by 4 to expose
/// independent accumulation chains, allowing the CPU to overlap loads and FMA.
#[pyfunction]
#[pyo3(signature = (src_ptr, dst_ptr, rows, cols))]
fn stencil_compute_unrolled(src_ptr: usize, dst_ptr: usize, rows: i64, cols: i64) -> i64 {
    if rows < 3 || cols < 3 { return 0; }
    let rows = rows as usize;
    let cols = cols as usize;
    let out_rows = rows - 2;
    let out_cols = cols - 2;

    unsafe {
        let src = std::slice::from_raw_parts(src_ptr as *const f32, rows * cols);
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, out_rows * out_cols);

        dst.par_chunks_mut(out_cols)
            .enumerate()
            .for_each(|(out_i, row_dst)| {
                let i = out_i + 1;
                let row_center = &src[i * cols..];
                let row_north = &src[(i - 1) * cols..];
                let row_south = &src[(i + 1) * cols..];

                let mut j = 1;
                let j_end = cols - 1;
                let mut west = row_center[0];

                // Unrolled by 4: process 4 elements per iteration
                while j + 3 < j_end {
                    let c0 = row_center[j];     let c1 = row_center[j+1];
                    let c2 = row_center[j+2];   let c3 = row_center[j+3];

                    let n0 = row_north[j];      let n1 = row_north[j+1];
                    let n2 = row_north[j+2];    let n3 = row_north[j+3];

                    let s0 = row_south[j];      let s1 = row_south[j+1];
                    let s2 = row_south[j+2];    let s3 = row_south[j+3];

                    let e0 = row_center[j+1];   let e1 = row_center[j+2];
                    let e2 = row_center[j+3];   let e3 = row_center[j+4];

                    // west values come from register rotation
                    let w0 = west;
                    let w1 = c0;
                    let w2 = c1;
                    let w3 = c2;

                    row_dst[j-1]   = 0.2 * (c0 + n0 + s0 + w0 + e0);
                    row_dst[j]     = 0.2 * (c1 + n1 + s1 + w1 + e1);
                    row_dst[j+1]   = 0.2 * (c2 + n2 + s2 + w2 + e2);
                    row_dst[j+2]   = 0.2 * (c3 + n3 + s3 + w3 + e3);

                    west = c3;
                    j += 4;
                }

                // Handle remaining elements
                while j < j_end {
                    let center = row_center[j];
                    let north = row_north[j];
                    let south = row_south[j];
                    let east = row_center[j + 1];
                    row_dst[j - 1] = 0.2 * (center + north + south + west + east);
                    west = center;
                    j += 1;
                }
            });
    }
    0
}

// ── Tier 4: Composition / Orchestration Layer ────────────────────────────────
//
// Tier 4 is NOT a new execution engine — it's a smart scheduler that breaks
// general code into Tier 1–3 chunks. This function takes a trace (as opcode +
// operand pairs), runs the Rust Tier 4 planner (decompose → DAG → fusion →
// buffer plan → schedule), and returns a JSON schedule string.
//
// The Python side then dispatches each step to the appropriate existing
// backend (SIMD elementwise, fused vector, BLAS).

/// Plan a Tier 4 execution schedule for a trace.
///
/// Takes a list of (opcode, operands) pairs representing the trace,
/// decomposes into regions, builds a DAG, applies conservative fusion,
/// computes a buffer reuse plan, and returns a JSON schedule.
#[pyfunction]
fn tier4_plan(trace_ops: Vec<(u8, Vec<i64>)>) -> PyResult<String> {
    use symplex_engine::phase3_jit::tier4_compile;
    use symplex_engine::phase3_jit::tier4_validate_schedule;

    let schedule = tier4_compile(&trace_ops);

    // Validate the schedule before handing it to Python
    let (valid, warnings) = tier4_validate_schedule(&schedule);
    if !valid {
        return Ok(format!("{{\"error\": \"invalid_schedule\", \"warnings\": {}}}", warnings));
    }

    // Serialize schedule to JSON — include full step metadata for direct dispatch
    let steps_json: Vec<String> = schedule.steps.iter().map(|step| {
        let kind_str = match step.kind {
            p3::Tier4RegionKind::Elementwise => "elementwise",
            p3::Tier4RegionKind::Reduction => "reduction",
            p3::Tier4RegionKind::LinearAlgebra => "linear_algebra",
            p3::Tier4RegionKind::Stencil => "stencil",
            p3::Tier4RegionKind::Transcendental => "transcendental",
            p3::Tier4RegionKind::FmaChain => "fma_chain",
            p3::Tier4RegionKind::Scalar => "scalar",
            p3::Tier4RegionKind::Logical => "logical",
        };
        let input_slots_json: Vec<String> = step.input_slots.iter().map(|s| s.to_string()).collect();
        let output_slots_json: Vec<String> = step.output_slots.iter().map(|s| s.to_string()).collect();
        format!(
            "{{\"node_id\": {}, \"tier\": {}, \"op_desc\": \"{}\", \"kind\": \"{}\", \"input_slots\": [{}], \"output_slots\": [{}], \"instr_range\": [{}, {}], \"is_fused\": {}}}",
            step.node_id,
            step.tier,
            step.op_desc,
            kind_str,
            input_slots_json.join(", "),
            output_slots_json.join(", "),
            step.instr_range.0,
            step.instr_range.1,
            step.is_fused,
        )
    }).collect();

    let buffer_plan_json: Vec<String> = schedule.buffer_plan.buffer_lifetimes.iter().map(|(idx, size, first, last)| {
        format!("{{\"buffer\": {}, \"size_bytes\": {}, \"first_use\": {}, \"last_use\": {}}}", idx, size, first, last)
    }).collect();

    let slot_mapping_json: Vec<String> = schedule.buffer_plan.slot_to_buffer.iter().map(|(slot, buf)| {
        format!("\"{}\": {}", slot, buf)
    }).collect();

    Ok(format!(
        "{{\"steps\": [{}], \"fusion_applied\": {}, \"fused_node_count\": {}, \"estimated_cost\": {}, \"peak_memory_bytes\": {}, \"warnings\": {}, \"buffer_plan\": {{\"total_buffers\": {}, \"total_bytes\": {}, \"slot_mapping\": {{{}}}, \"lifetimes\": [{}]}}}}",
        steps_json.join(", "),
        schedule.fusion_applied,
        schedule.fused_node_count,
        schedule.estimated_cost,
        schedule.peak_memory_bytes,
        warnings,
        schedule.buffer_plan.total_buffers,
        schedule.buffer_plan.total_bytes,
        slot_mapping_json.join(", "),
        buffer_plan_json.join(", ")
    ))
}

// ── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn _symplex_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_trace, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_specialized, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    m.add_function(wrap_pyfunction!(detect_hardware, m)?)?;
    m.add_function(wrap_pyfunction!(micro_kernel_config, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(jit_compile, m)?)?;
    m.add_function(wrap_pyfunction!(jit_execute, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(jit_parallel_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(has_avx512, m)?)?;
    m.add_function(wrap_pyfunction!(has_avx2, m)?)?;
    m.add_function(wrap_pyfunction!(detect_isa, m)?)?;
    m.add_function(wrap_pyfunction!(vec_width, m)?)?;
    m.add_function(wrap_pyfunction!(num_cores, m)?)?;
    m.add_function(wrap_pyfunction!(jit_info, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_compile, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_compile_ssa, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_execute_int, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_execute_f64, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_bench_int, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_run_int, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_verify, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_code_size, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_dump_code, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_execute_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(trace_jit_execute, m)?)?;
    // SIMD elementwise f64
    m.add_function(wrap_pyfunction!(simd_elementwise_f64, m)?)?;
    // SIMD elementwise f32
    m.add_function(wrap_pyfunction!(simd_elementwise_f32, m)?)?;
    m.add_function(wrap_pyfunction!(simd_elementwise_isa, m)?)?;
    // SIMD reduction
    m.add_function(wrap_pyfunction!(simd_reduce_f32, m)?)?;
    m.add_function(wrap_pyfunction!(simd_reduce_f64, m)?)?;
    // Fused SIMD elementwise (single-pass multi-op chains)
    m.add_function(wrap_pyfunction!(simd_fused_elementwise_f32, m)?)?;
    m.add_function(wrap_pyfunction!(simd_fused_elementwise_f64, m)?)?;
    // CUDA backend
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_compile_ptx, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_stencil, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_info, m)?)?;
    // Fused stencil compute
    m.add_function(wrap_pyfunction!(stencil_compute, m)?)?;
    m.add_function(wrap_pyfunction!(stencil_compute_unrolled, m)?)?;
    // Tier 4 orchestration
    m.add_function(wrap_pyfunction!(tier4_plan, m)?)?;
    Ok(())
}
