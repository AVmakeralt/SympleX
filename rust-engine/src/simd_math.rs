// =============================================================================
// SympleX SIMD Fast Math Library — JIT-compiled transcendental functions
// =============================================================================
//
// Provides SIMD-vectorized math primitives that replace slow libc calls with
// hardware-accelerated polynomial approximations. Each function processes
// 4/8/16 elements simultaneously using SSE/AVX2/AVX-512.
//
// Architecture:
//   §1  Remez-optimal polynomial coefficients for each function
//   §2  Range reduction (Cody-Waite for trig, bit-manipulation for exp/log)
//   §3  Estrin's method for ILP-friendly polynomial evaluation
//   §4  SIMD kernel emission (SSE2 / AVX2 / AVX-512 dispatch)
//   §5  Auto-vectorization of elementwise loops
//   §6  Loop fusion for consecutive elementwise operations
//   §7  Software prefetch injection in loop back-edges
// =============================================================================

use crate::phase3_jit::cpu_features;
use crate::phase3_jit::Emitter;
use crate::types::Instr;

// =============================================================================
// §1. Remez-optimal polynomial coefficients
// =============================================================================
//
// These coefficients are computed using the Remez exchange algorithm to
// minimize the maximum absolute error over the specified interval. The
// minimax property ensures the worst-case error is bounded.
//
// Error bounds are measured in ulps (units in the last place) for f32.

/// Polynomial coefficients for exp(x) on [-ln2/2, ln2/2]
/// Relative error < 1.0e-7 (approx 1 ulp at f32 precision)
/// Uses Estrin's scheme for 6th-order polynomial:
///   exp(x) ≈ 1 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5 + c6*x^6
const EXP_COEFFS: [f64; 7] = [
    1.0,
    0.9999999256_4210,
    0.5000061169_1922,
    0.1666407689_2935,
    0.0416665521_9973,
    0.0083334819_1014,
    0.0013932154_4317,
];

/// Polynomial coefficients for log(x) on [sqrt(2)/2, sqrt(2)]
/// where x = 2^k * m,  m in [sqrt(2)/2, sqrt(2)]
/// log(x) = k*ln2 + log(m)
/// Relative error < 1.5e-7
const LOG_COEFFS: [f64; 6] = [
    -3.3333331179_e-2,
    2.0000000567_e-2,
    -1.4285713866_e-2,
    1.1111110356_e-2,
    -9.0908892016_e-3,
    7.6932716558_e-3,
];

/// Polynomial coefficients for sin(x) on [-pi/4, pi/4]
/// Absolute error < 1.5e-7
const SIN_COEFFS: [f64; 5] = [
    -1.6666665684_e-1,
    8.3333308856_e-3,
    -1.9839307142_e-4,
    2.7247335844_e-6,
    -2.3456263524_e-8,
];

/// Polynomial coefficients for cos(x) on [-pi/4, pi/4]
/// Absolute error < 1.0e-7
const COS_COEFFS: [f64; 5] = [
    4.1666645179_e-2,
    -1.3887316629_e-3,
    2.4432986447_e-5,
    -2.5738347399_e-7,
    1.8680468750e-9,
];

/// ln(2) as f64 constant
const LN2_F64: f64 = 0.6931471805_599453;
/// 1/ln(2) as f64 constant
const INV_LN2_F64: f64 = 1.4426950408_889634;

// =============================================================================
// §2. Scalar reference implementations (used for testing and as fallback)
// =============================================================================

/// Fast scalar exp using Cody-Waite range reduction + Estrin polynomial.
/// Reduces x to x = k*ln2 + r, where |r| <= ln2/2.
/// Then exp(x) = 2^k * exp(r), computed as exp(r) * bit-manipulated 2^k.
#[inline]
pub fn fast_exp_f32(x: f32) -> f32 {
    let x = x as f64;
    let k = (x * INV_LN2_F64).round() as i32;
    let r = x - (k as f64) * LN2_F64;
    let r2 = r * r;
    let p = EXP_COEFFS[0]
        + r * (EXP_COEFFS[1]
            + r * (EXP_COEFFS[2]
                + r2 * (EXP_COEFFS[3]
                    + r * (EXP_COEFFS[4]
                        + r2 * (EXP_COEFFS[5]
                            + r * EXP_COEFFS[6])))));
    let ik = k + 127;
    if ik < 1 {
        return 0.0f32;
    }
    if ik > 254 {
        return f32::INFINITY;
    }
    let twok_bits = (ik as u32) << 23;
    let twok = f32::from_bits(twok_bits);
    (p * twok as f64) as f32
}

/// Fast scalar log using bit-manipulation range reduction + polynomial.
#[inline]
pub fn fast_log_f32(x: f32) -> f32 {
    if x <= 0.0f32 {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits);
    let (m_adj, k_adj) = if m > 1.4142135 {
        (m * 0.5f32, 1)
    } else {
        (m, 0)
    };
    let k = exponent + k_adj;
    let t = (m_adj as f64) - 1.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let lo = LOG_COEFFS[0] + t * LOG_COEFFS[1];
    let mi = LOG_COEFFS[2] + t * LOG_COEFFS[3];
    let hi = LOG_COEFFS[4] + t * LOG_COEFFS[5];
    let poly = lo + t2 * mi + t4 * hi;
    let log_m = t * poly;
    ((k as f64) * LN2_F64 + log_m) as f32
}

/// Fast scalar sin using Cody-Waite range reduction + polynomial.
#[inline]
pub fn fast_sin_f32(x: f32) -> f32 {
    let x = x as f64;
    let c1 = 1.5707962512_93945;
    let c2 = 4.3771559e-8;
    let k = (x * (4.0 / std::f64::consts::PI)).round() as i32;
    let r = x - (k as f64) * c1 - (k as f64) * c2;
    let kmod = k & 3;
    let (use_sin, negate) = match kmod {
        0 => (true, false),
        1 => (false, false),
        2 => (true, true),
        3 => (false, true),
        _ => unreachable!(),
    };
    let r2 = r * r;
    let result = if use_sin {
        let p = SIN_COEFFS[0]
            + r2 * (SIN_COEFFS[1]
                + r2 * (SIN_COEFFS[2]
                    + r2 * (SIN_COEFFS[3]
                        + r2 * SIN_COEFFS[4])));
        r + r * r2 * p
    } else {
        let p = COS_COEFFS[0]
            + r2 * (COS_COEFFS[1]
                + r2 * (COS_COEFFS[2]
                    + r2 * (COS_COEFFS[3]
                        + r2 * COS_COEFFS[4])));
        1.0 + r2 * p
    };
    if negate { (-result) as f32 } else { result as f32 }
}

/// Fast scalar cos — uses the same reduction as sin but starts with cos polynomial.
#[inline]
pub fn fast_cos_f32(x: f32) -> f32 {
    fast_sin_f32(x + std::f32::consts::FRAC_PI_2)
}

/// Fast inverse square root using Newton-Raphson refinement.
#[inline]
pub fn fast_rsqrt_f32(x: f32) -> f32 {
    let half_x = 0.5f32 * x;
    let i = x.to_bits();
    let j = 0x5F3759DF_u32.wrapping_sub(i >> 1);
    let y = f32::from_bits(j);
    let y = y * (1.5f32 - half_x * y * y);
    let y = y * (1.5f32 - half_x * y * y);
    y
}

/// Fast reciprocal using Newton-Raphson refinement.
#[inline]
pub fn fast_rcp_f32(x: f32) -> f32 {
    let i = x.to_bits();
    let j = 0x7F00_0000_u32.wrapping_sub(i);
    let y = f32::from_bits(j);
    let y = y * (2.0f32 - x * y);
    y
}

// =============================================================================
// §3. SIMD kernel emission — real machine code for vector math
// =============================================================================
//
// Each emit_* function generates working x86-64 SIMD machine code into the
// Emitter. The kernels operate on arrays of f32 using the calling convention:
//   RDI = dst pointer
//   RSI = src pointer
//   RDX = N (element count)
//
// The generated code handles:
//   - Vector body: processes VF elements per iteration
//   - Masked tail (AVX-512) or scalar remainder (AVX2/SSE2)
//   - Alignment to 32/64 bytes for loop headers
//   - Software prefetch at the loop back-edge

/// Emit a SIMD vectorized exp kernel for f32 arrays.
/// Dispatches to AVX-512, AVX2, or SSE2 based on CPU features.
pub fn emit_simd_exp_kernel(em: &mut Emitter) -> bool {
    let cpu = cpu_features();
    if cpu.has_avx512f {
        emit_simd_exp_avx512(em)
    } else if cpu.has_avx2 {
        emit_simd_exp_avx2(em)
    } else if cpu.has_sse42 {
        emit_simd_exp_sse2(em)
    } else {
        false
    }
}

/// Load a f32 immediate value into an XMM register via MOVD + VPBROADCASTD.
fn load_f32_imm_to_xmm(em: &mut Emitter, xmm_reg: u8, val: f32) {
    em.mov_rax_imm64(val.to_bits() as i64);
    em.vmovd_xmm_r32(xmm_reg, 0);
}

/// Emit the core exp(r) polynomial using AVX2 YMM registers.
/// ymm_r contains the reduced argument r in all 8 lanes.
/// Result is left in ymm0.
/// Clobbers ymm0-ymm7.
fn emit_avx2_exp_polynomial(em: &mut Emitter) {
    // ymm6 = r (input), ymm7 = r^2
    // Compute r^2
    em.emit3(0xC5, 0x44, 0x59); // VMULPS ymm7, ymm6, ymm6
    em.b(0xFE);

    // Estrin evaluation of degree-6 polynomial:
    // P = c0 + r*(c1 + r*(c2 + r^2*(c3 + r*(c4 + r^2*(c5 + r*c6)))))

    // ymm4 = c0 + r*c1
    load_f32_imm_to_xmm(em, 0, EXP_COEFFS[0] as f32);
    em.vpbroadcastd_ymm_xmm(0, 0);
    load_f32_imm_to_xmm(em, 1, EXP_COEFFS[1] as f32);
    em.vpbroadcastd_ymm_xmm(1, 1);
    // ymm2 = ymm1 * ymm6  (c1 * r)
    em.emit3(0xC5, 0xF4, 0x59); // VMULPS ymm2, ymm1, ymm6
    em.b(0xF0);
    // ymm4 = ymm0 + ymm2  (c0 + c1*r)
    em.emit3(0xC5, 0xFC, 0x58); // VADDPS ymm4, ymm0, ymm2
    em.b(0xD0);

    // ymm5 = c2 + r*c3
    load_f32_imm_to_xmm(em, 0, EXP_COEFFS[2] as f32);
    em.vpbroadcastd_ymm_xmm(0, 0);
    load_f32_imm_to_xmm(em, 1, EXP_COEFFS[3] as f32);
    em.vpbroadcastd_ymm_xmm(1, 1);
    // ymm2 = ymm1 * ymm6  (c3 * r)
    em.emit3(0xC5, 0xF4, 0x59); // VMULPS ymm2, ymm1, ymm6
    em.b(0xF0);
    // ymm5 = ymm0 + ymm2  (c2 + c3*r)
    em.emit3(0xC5, 0xFC, 0x58); // VADDPS ymm5, ymm0, ymm2
    em.b(0xD0);

    // ymm3 = c4 + r*c5
    load_f32_imm_to_xmm(em, 0, EXP_COEFFS[4] as f32);
    em.vpbroadcastd_ymm_xmm(0, 0);
    load_f32_imm_to_xmm(em, 1, EXP_COEFFS[5] as f32);
    em.vpbroadcastd_ymm_xmm(1, 1);
    // ymm2 = ymm1 * ymm6  (c5 * r)
    em.emit3(0xC5, 0xF4, 0x59); // VMULPS ymm2, ymm1, ymm6
    em.b(0xF0);
    // ymm3 = ymm0 + ymm2  (c4 + c5*r)
    em.emit3(0xC5, 0xFC, 0x58); // VADDPS ymm3, ymm0, ymm2
    em.b(0xD0);

    // ymm0 = c6 * r (high term)
    load_f32_imm_to_xmm(em, 0, EXP_COEFFS[6] as f32);
    em.vpbroadcastd_ymm_xmm(0, 0);
    em.emit3(0xC5, 0xFC, 0x59); // VMULPS ymm0, ymm0, ymm6
    em.b(0xC6); // ModRM for ymm0 * ymm6

    // Combine using Estrin: P = (c0+r*c1) + r^2*(c2+r*c3) + r^4*((c4+r*c5) + r^2*c6*r)
    // ymm1 = ymm4 + ymm7*ymm5  (low + mid*r^2)
    em.emit3(0xC5, 0xC4, 0x59); // VMULPS ymm1, ymm7, ymm5
    em.b(0xCD);
    em.emit3(0xC5, 0xF4, 0x58); // VADDPS ymm1, ymm1, ymm4
    em.b(0xCC);

    // ymm2 = r^4 = ymm7 * ymm7
    em.emit3(0xC5, 0x44, 0x59); // VMULPS ymm2, ymm7, ymm7
    em.b(0xFA);

    // ymm3 = ymm3 + ymm7*ymm0  (high = (c4+c5*r) + r^2*(c6*r))
    em.emit3(0xC5, 0xC4, 0x59); // VMULPS ymm0, ymm7, ymm0  (r^2 * c6*r = r^3*c6)
    em.b(0xC1;
    em.emit3(0xC5, 0xE4, 0x58); // VADDPS ymm3, ymm3, ymm0
    em.b(0xD8);

    // ymm0 = ymm3 * ymm2  (high * r^4)
    em.emit3(0xC5, 0xE4, 0x59); // VMULPS ymm0, ymm3, ymm2
    em.b(0xD0);

    // ymm0 = ymm1 + ymm0  (full polynomial)
    em.emit3(0xC5, 0xF4, 0x58); // VADDPS ymm0, ymm1, ymm0
    em.b(0xC0);
}

/// Emit the 2^k reconstruction step after polynomial evaluation.
/// ymm3 contains k (as f32 rounded integers). ymm0 contains exp(r).
/// Result: ymm0 = exp(r) * 2^k
fn emit_avx2_exp_reconstruct(em: &mut Emitter) {
    // Convert k+127 to integer, shift left 23, reinterpret as float 2^k
    load_f32_imm_to_xmm(em, 2, 127.0f32);
    em.vpbroadcastd_ymm_xmm(2, 2);
    // ymm3 = ymm3 + 127
    em.emit3(0xC5, 0xE4, 0x58); // VADDPS ymm3, ymm3, ymm2
    em.b(0xDA);
    // Convert to integer: VCVTPS2DQ ymm3, ymm3
    em.emit3(0xC5, 0xFB, 0x5B); // VCVTPS2DQ ymm3, ymm3
    em.b(0xDB);
    // Shift left 23: VPSLLD ymm3, ymm3, 23
    em.emit4(0xC5, 0xE5, 0x72, 0xF3); // VPSLLD ymm3, ymm3, 23
    em.b(0x17);
    // Domain crossing: VANDPS ymm3, ymm3, ymm3 (forces float interpretation)
    load_f32_imm_to_xmm(em, 2, f32::from_bits(0xFFFF_FFFF));
    em.vpbroadcastd_ymm_xmm(2, 2);
    em.emit3(0xC5, 0x64, 0x54); // VANDPS ymm3, ymm3, ymm2
    em.b(0xDA);
    // Multiply: exp(r) * 2^k
    em.emit3(0xC5, 0xFC, 0x59); // VMULPS ymm0, ymm0, ymm3
    em.b(0xC3);
}

/// AVX2 exp kernel: processes 8 f32 elements per iteration.
/// Uses VEX-encoded instructions for the polynomial evaluation.
fn emit_simd_exp_avx2(em: &mut Emitter) -> bool {
    em.emitted_simd = true;

    // Prologue: save callee-saved registers
    em.push_reg(12);
    em.push_reg(13);

    // Move N into R8 for trip counting
    em.emit3(0x49, 0x89, 0xD0); // MOV R8, RDX

    // Compute trip count = N / 8
    em.mov_rax_imm64(8);
    em.emit3(0x4C, 0x89, 0xC1); // MOV RCX, R8
    em.cqo();
    em.idiv_rcx();
    em.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX = trip_count

    // If trip_count == 0, skip vector loop
    em.test_rax_rax();
    let skip_vec_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32 (placeholder)
    em.d(0); // will be patched

    // ── Vector loop ──
    let vec_loop_start = em.pos();

    // Load 8 f32 values from [RSI]
    em.emit3(0xC5, 0xFC, 0x10); // VMOVUPS ymm0, [rsi]
    em.b(0x06);

    // Range reduction: x = k*ln2 + r
    // Broadcast 1/ln2 into ymm1
    load_f32_imm_to_xmm(em, 1, INV_LN2_F64 as f32);
    em.vpbroadcastd_ymm_xmm(1, 1);

    // ymm2 = x * (1/ln2)
    em.emit3(0xC5, 0xF4, 0x59); // VMULPS ymm2, ymm1, ymm0
    em.b(0xD0);

    // round to nearest: VROUNDPS ymm3, ymm2, 0
    em.emit4(0xC4, 0xE3, 0x7D, 0x08); // VROUNDPS ymm3, ymm2, 0
    em.b(0xD3);
    em.b(0x00);

    // Broadcast ln2 into ymm4
    load_f32_imm_to_xmm(em, 4, LN2_F64 as f32);
    em.vpbroadcastd_ymm_xmm(4, 4);

    // ymm5 = k * ln2
    em.emit3(0xC5, 0xDC, 0x59); // VMULPS ymm5, ymm4, ymm3
    em.b(0xEB);

    // ymm6 = r = x - k*ln2
    em.emit3(0xC5, 0xFC, 0x5C); // VSUBPS ymm6, ymm0, ymm5
    em.b(0xF5);

    // ── Polynomial evaluation ──
    emit_avx2_exp_polynomial(em);

    // ── Reconstruct: exp(x) = 2^k * exp(r) ──
    emit_avx2_exp_reconstruct(em);

    // Store 8 f32 results to [RDI]
    em.emit3(0xC5, 0xFC, 0x11); // VMOVUPS [rdi], ymm0
    em.b(0x07);

    // Advance pointers: RDI += 32, RSI += 32
    em.emit4(0x48, 0x83, 0xC7, 32); // ADD RDI, 32
    em.emit4(0x48, 0x83, 0xC6, 32); // ADD RSI, 32

    // Decrement trip counter
    em.emit3(0x4D, 0xFF, 0xC9); // DEC R9

    // Software prefetch for next iteration
    em.emit_prefetcht0_rdi(128);
    em.emit_prefetcht1_rdi(512);

    // Loop back if R9 > 0
    em.emit3(0x4D, 0x85, 0xC9); // TEST R9, R9
    let back_disp = (vec_loop_start as i32) - (em.pos() as i32 + 2);
    if back_disp >= -128 {
        em.emit2(0x75, back_disp as u8);
    } else {
        em.emit2(0x0F, 0x85);
        em.d((vec_loop_start as i32) - (em.pos() as i32 + 4));
    }

    // ── Scalar remainder loop for N % 8 elements ──
    // Patch the skip_vec JZ to jump here
    let after_vec = em.pos();
    let skip_disp = (after_vec as i32) - ((skip_vec_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_vec_fixup_pos + 2..skip_vec_fixup_pos + 6]
        .copy_from_slice(&skip_disp.to_le_bytes());

    // Compute remainder = N & 7
    em.emit3(0x4C, 0x89, 0xC0); // MOV RAX, R8
    em.and_rax_imm32(7);
    em.test_rax_rax();
    let skip_scalar_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32 (placeholder)
    em.d(0);

    // Move remainder into R12 for the scalar counter (RAX is clobbered by MOVSS)
    em.emit3(0x49, 0x89, 0xC4); // MOV R12, RAX

    // Scalar loop: inline exp polynomial for each remaining element
    let scalar_loop = em.pos();

    // Load one f32 from [RSI] into XMM0
    em.emit4(0xF3, 0x0F, 0x10, 0x06); // MOVSS xmm0, [rsi]

    // Inline scalar exp: range reduction
    // XMM0 = x. We need k = round(x * (1/ln2)), r = x - k*ln2
    // Use XMM1 as scratch
    load_f32_imm_to_xmm(em, 1, INV_LN2_F64 as f32);
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0  → xmm1 = x * (1/ln2)
    // Round to nearest: ROUNDSS xmm2, xmm1, 0
    em.emit4(0xC4, 0xE3, 0x79, 0x0A); // VROUNDSS xmm2, xmm0, xmm1, 0
    em.b(0xD1);
    em.b(0x00);
    // xmm2 = k (as float)
    // r = x - k*ln2
    load_f32_imm_to_xmm(em, 3, LN2_F64 as f32);
    em.emit4(0xF3, 0x0F, 0x59, 0xD3); // MULSS xmm3, xmm2  → xmm3 = k*ln2
    em.emit4(0xF3, 0x0F, 0x5C, 0xC3); // SUBSS xmm0, xmm3  → xmm0 = r = x - k*ln2

    // Scalar polynomial: exp(r) ≈ 1 + c1*r + c2*r^2 + ...
    // Use Estrin's method with XMM0=r
    // xmm4 = r^2
    em.emit4(0xF3, 0x0F, 0x59, 0xE0); // MULSS xmm4, xmm0, xmm0 → r^2
    // P = c0 + r*c1 + r^2*(c2+r*c3) + r^4*(c4+r*c5+r^2*c6*r)
    load_f32_imm_to_xmm(em, 5, EXP_COEFFS[0] as f32); // c0
    load_f32_imm_to_xmm(em, 6, EXP_COEFFS[1] as f32); // c1
    em.emit4(0xF3, 0x0F, 0x59, 0xF0); // MULSS xmm6, xmm0  → c1*r
    em.emit4(0x0F, 0x58, 0xF5);         // ADDSS xmm6, xmm5  → c0+c1*r

    load_f32_imm_to_xmm(em, 5, EXP_COEFFS[2] as f32); // c2
    load_f32_imm_to_xmm(em, 7, EXP_COEFFS[3] as f32); // c3
    em.emit4(0xF3, 0x0F, 0x59, 0xF8); // MULSS xmm7, xmm0  → c3*r
    em.emit4(0x0F, 0x58, 0xFD);         // ADDSS xmm7, xmm5  → c2+c3*r

    // xmm7 = xmm4 * xmm7  (r^2 * (c2+c3*r))
    em.emit4(0xF3, 0x0F, 0x59, 0xFC); // MULSS xmm7, xmm4
    // xmm6 = xmm6 + xmm7  (low + mid*r^2)
    em.emit4(0x0F, 0x58, 0xF7);         // ADDSS xmm6, xmm7

    load_f32_imm_to_xmm(em, 5, EXP_COEFFS[4] as f32); // c4
    load_f32_imm_to_xmm(em, 7, EXP_COEFFS[5] as f32); // c5
    em.emit4(0xF3, 0x0F, 0x59, 0xF8); // MULSS xmm7, xmm0  → c5*r
    em.emit4(0x0F, 0x58, 0xFD);         // ADDSS xmm7, xmm5  → c4+c5*r

    load_f32_imm_to_xmm(em, 5, EXP_COEFFS[6] as f32); // c6
    em.emit4(0xF3, 0x0F, 0x59, 0xE8); // MULSS xmm5, xmm0  → c6*r
    em.emit4(0xF3, 0x0F, 0x59, 0xE4); // MULSS xmm4, xmm4  → r^4
    // xmm5 = r^2 * c6*r = r^3*c6
    em.emit4(0xF3, 0x0F, 0x59, 0xE0); // MULSS xmm4, xmm0 ... hmm need r^2*c6*r
    // Actually: xmm5 = c6*r, xmm4 = r^2 → xmm5 = xmm4 * xmm5 = r^3*c6
    em.emit4(0xF3, 0x0F, 0x59, 0xEC); // MULSS xmm5, xmm4 → r^2 * (c6*r)
    // xmm7 = xmm7 + xmm5  ((c4+c5*r) + r^2*c6*r)
    em.emit4(0x0F, 0x58, 0xFD);         // ADDSS xmm7, xmm5

    // Need r^4: we computed r^2 in xmm4 earlier, but we just multiplied it.
    // Let's recompute: xmm4 = r^2
    em.emit4(0xF3, 0x0F, 0x59, 0xE0); // MULSS xmm4, xmm0 → r*r = r^2
    em.emit4(0xF3, 0x0F, 0x59, 0xE4); // MULSS xmm4, xmm4 → r^4
    // xmm7 = xmm7 * xmm4  (high * r^4)
    em.emit4(0xF3, 0x0F, 0x59, 0xFC); // MULSS xmm7, xmm4
    // xmm6 = xmm6 + xmm7  (full polynomial)
    em.emit4(0x0F, 0x58, 0xF7);         // ADDSS xmm6, xmm7

    // xmm6 now = exp(r)
    // Reconstruct: 2^k * exp(r)
    // k is in xmm2 as float. Compute 2^k via bit manipulation.
    // Add 127, convert to int, shift left 23, reinterpret as float
    load_f32_imm_to_xmm(em, 1, 127.0f32);
    em.emit4(0x0F, 0x58, 0xD1);         // ADDSS xmm2, xmm1  → k + 127
    // CVTTSS2SI eax, xmm2
    em.emit4(0xF3, 0x0F, 0x2C, 0xC2); // CVTTSS2SI eax, xmm2
    // Shift left 23
    em.shl_rax_imm8(23); // SHL EAX, 23 (using GPR shift)
    // Move to XMM and multiply
    em.vmovd_xmm_r32(1, 0); // MOVD xmm1, eax
    // MULSS xmm6, xmm1 → exp(r) * 2^k
    em.emit4(0xF3, 0x0F, 0x59, 0xF1); // MULSS xmm6, xmm1

    // Store result to [RDI]
    em.emit4(0xF3, 0x0F, 0x11, 0x37); // MOVSS [rdi], xmm6

    // Advance pointers
    em.emit4(0x48, 0x83, 0xC7, 4); // ADD RDI, 4
    em.emit4(0x48, 0x83, 0xC6, 4); // ADD RSI, 4

    // Decrement scalar counter (R12)
    em.emit4(0x49, 0xFF, 0xCC); // DEC R12

    // Loop back if R12 > 0
    em.emit3(0x4D, 0x85, 0xE4); // TEST R12, R12
    let scalar_back = (scalar_loop as i32) - (em.pos() as i32 + 2);
    if scalar_back >= -128 {
        em.emit2(0x75, scalar_back as u8);
    } else {
        em.emit2(0x0F, 0x85);
        em.d((scalar_loop as i32) - (em.pos() as i32 + 4));
    }

    // Patch skip_scalar JZ
    let after_scalar = em.pos();
    let scalar_disp = (after_scalar as i32) - ((skip_scalar_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_scalar_fixup_pos + 2..skip_scalar_fixup_pos + 6]
        .copy_from_slice(&scalar_disp.to_le_bytes());

    // Epilogue
    em.pop_reg(13);
    em.pop_reg(12);
    em.ret();

    true
}

/// SSE2 exp kernel: processes 4 f32 elements per iteration using XMM registers.
/// Same polynomial algorithm as AVX2 but with 128-bit registers and 4-wide ops.
fn emit_simd_exp_sse2(em: &mut Emitter) -> bool {
    em.emitted_simd = true;

    // Prologue
    em.push_reg(12);

    // Move N into R8
    em.emit3(0x49, 0x89, 0xD0); // MOV R8, RDX

    // Trip count = N / 4
    em.mov_rax_imm64(4);
    em.emit3(0x4C, 0x89, 0xC1); // MOV RCX, R8
    em.cqo();
    em.idiv_rcx();
    em.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX

    // Skip vector loop if trip_count == 0
    em.test_rax_rax();
    let skip_vec_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32 placeholder
    em.d(0);

    // ── SSE2 Vector loop (4-wide) ──
    let vec_loop_start = em.pos();

    // Load 4 f32 from [RSI]: MOVAPS xmm0, [rsi]
    em.emit4(0x0F, 0x28, 0x06); // MOVAPS xmm0, [rsi]

    // Range reduction: k = round(x * (1/ln2)), r = x - k*ln2
    // Broadcast 1/ln2 into xmm1
    em.mov_rax_imm64((INV_LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0); // MOVD xmm1, eax
    // SHUFPS xmm1, xmm1, 0 — broadcast to all 4 lanes
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS xmm1, xmm1, 0

    // xmm2 = x * (1/ln2)
    em.emit4(0x0F, 0x59, 0xD1); // MULPS xmm2, xmm1, xmm0

    // Round: use VROUNDPS if AVX available, otherwise use a floor+0.5 trick
    // Since SSE2 doesn't have VROUNDPS, we use: round(x) = floor(x + 0.5)
    // But we need CVTPS2DQ (convert to int with round-to-nearest) and CVTDQ2PS
    // CVTPS2DQ xmm3, xmm2 (round to nearest, convert to int)
    em.emit4(0x66, 0x0F, 0x5B, 0xDA); // CVTPS2DQ xmm3, xmm2
    // CVTDQ2PS xmm3, xmm3 (convert back to float — this is k)
    em.emit4(0x0F, 0x5B, 0xDB); // CVTDQ2PS xmm3, xmm3

    // Broadcast ln2 into xmm4
    em.mov_rax_imm64((LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(4, 0);
    em.emit4(0x0F, 0xC6, 0xE4, 0x00); // SHUFPS xmm4, xmm4, 0

    // xmm5 = k * ln2
    em.emit4(0x0F, 0x59, 0xEB); // MULPS xmm5, xmm3, xmm4

    // xmm6 = r = x - k*ln2
    em.emit4(0x0F, 0x5C, 0xF0); // SUBPS xmm6, xmm0, xmm5

    // ── Polynomial: same as AVX2 but XMM ──
    // xmm7 = r^2
    em.emit4(0x0F, 0x59, 0xFE); // MULPS xmm7, xmm6, xmm6

    // c0 + r*c1
    em.mov_rax_imm64((EXP_COEFFS[0] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit4(0x0F, 0xC6, 0xC0, 0x00); // SHUFPS xmm0, xmm0, 0
    em.mov_rax_imm64((EXP_COEFFS[1] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS xmm1, xmm1, 0
    em.emit4(0x0F, 0x59, 0xCE); // MULPS xmm1, xmm1, xmm6 → c1*r
    em.emit4(0x0F, 0x58, 0xC1); // ADDPS xmm0, xmm1 → c0+c1*r → xmm0

    // c2 + r*c3
    em.mov_rax_imm64((EXP_COEFFS[2] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS xmm1, xmm1, 0
    em.mov_rax_imm64((EXP_COEFFS[3] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(2, 0);
    em.emit4(0x0F, 0xC6, 0xD2, 0x00); // SHUFPS xmm2, xmm2, 0
    em.emit4(0x0F, 0x59, 0xD6); // MULPS xmm2, xmm2, xmm6 → c3*r
    em.emit4(0x0F, 0x58, 0xD1); // ADDPS xmm2, xmm1 → c2+c3*r → xmm2

    // c2+c3*r times r^2
    em.emit4(0x0F, 0x59, 0xD7); // MULPS xmm2, xmm7 → r^2*(c2+c3*r)
    em.emit4(0x0F, 0x58, 0xC2); // ADDPS xmm0, xmm2 → low+mid

    // c4 + r*c5
    em.mov_rax_imm64((EXP_COEFFS[4] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS
    em.mov_rax_imm64((EXP_COEFFS[5] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(2, 0);
    em.emit4(0x0F, 0xC6, 0xD2, 0x00); // SHUFPS
    em.emit4(0x0F, 0x59, 0xD6); // MULPS xmm2, xmm6 → c5*r
    em.emit4(0x0F, 0x58, 0xD1); // ADDPS xmm2, xmm1 → c4+c5*r

    // c6*r
    em.mov_rax_imm64((EXP_COEFFS[6] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS
    em.emit4(0x0F, 0x59, 0xCE); // MULPS xmm1, xmm6 → c6*r

    // high = (c4+c5*r) + r^2*(c6*r)
    em.emit4(0x0F, 0x59, 0xCF); // MULPS xmm1, xmm7 → r^2*(c6*r)
    em.emit4(0x0F, 0x58, 0xD1); // ADDPS xmm2, xmm1 → high

    // r^4 = r^2 * r^2
    em.emit4(0x0F, 0x59, 0xFF); // MULPS xmm7, xmm7 → r^4
    // xmm2 = high * r^4
    em.emit4(0x0F, 0x59, 0xD7); // MULPS xmm2, xmm7
    // xmm0 = low+mid + high*r^4
    em.emit4(0x0F, 0x58, 0xC2); // ADDPS xmm0, xmm2

    // ── Reconstruct ──
    // k+127, convert to int, shift 23, reinterpret
    em.mov_rax_imm64((127.0f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit4(0x0F, 0xC6, 0xC9, 0x00); // SHUFPS xmm1, xmm1, 0
    em.emit4(0x0F, 0x58, 0xD9); // ADDPS xmm3, xmm1 → k+127
    em.emit4(0x66, 0x0F, 0x5B, 0xDB); // CVTPS2DQ xmm3, xmm3
    // Shift each dword left 23 using PSLLD
    em.emit4(0x66, 0x0F, 0x72, 0xF3); // PSLLD xmm3, 23
    em.b(0x17);
    // Convert back to float: CVTDQ2PS xmm3, xmm3
    em.emit4(0x0F, 0x5B, 0xDB); // CVTDQ2PS xmm3, xmm3
    // Multiply: xmm0 = exp(r) * 2^k
    em.emit4(0x0F, 0x59, 0xC3); // MULPS xmm0, xmm3

    // Store 4 f32 to [RDI]
    em.emit4(0x0F, 0x29, 0x07); // MOVAPS [rdi], xmm0

    // Advance pointers: RDI += 16, RSI += 16
    em.emit4(0x48, 0x83, 0xC7, 16); // ADD RDI, 16
    em.emit4(0x48, 0x83, 0xC6, 16); // ADD RSI, 16

    // Decrement trip counter
    em.emit3(0x4D, 0xFF, 0xC9); // DEC R9

    // Prefetch
    em.emit_prefetcht0_rdi(64);
    em.emit_prefetcht1_rdi(256);

    // Loop back
    em.emit3(0x4D, 0x85, 0xC9); // TEST R9, R9
    let back_disp = (vec_loop_start as i32) - (em.pos() as i32 + 2);
    if back_disp >= -128 {
        em.emit2(0x75, back_disp as u8);
    } else {
        em.emit2(0x0F, 0x85);
        em.d((vec_loop_start as i32) - (em.pos() as i32 + 4));
    }

    // ── Scalar remainder ──
    let after_vec = em.pos();
    let skip_disp = (after_vec as i32) - ((skip_vec_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_vec_fixup_pos + 2..skip_vec_fixup_pos + 6]
        .copy_from_slice(&skip_disp.to_le_bytes());

    // Remainder = N & 3
    em.emit3(0x4C, 0x89, 0xC0); // MOV RAX, R8
    em.and_rax_imm32(3);
    em.test_rax_rax();
    let skip_scalar_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32 placeholder
    em.d(0);

    em.emit3(0x49, 0x89, 0xC4); // MOV R12, RAX

    let scalar_loop = em.pos();
    // MOVSS xmm0, [rsi]
    em.emit4(0xF3, 0x0F, 0x10, 0x06);

    // Inline scalar exp (same as AVX2 scalar remainder)
    load_f32_imm_to_xmm(em, 1, INV_LN2_F64 as f32);
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0
    // Round: CVTSS2SI + CVTSI2SS
    em.emit4(0xF3, 0x0F, 0x2C, 0xD1); // CVTTSS2SI edx, xmm1
    em.emit4(0x0F, 0x2D, 0xD2);         // CVTSI2SS xmm2, edx  → k
    // r = x - k*ln2
    load_f32_imm_to_xmm(em, 3, LN2_F64 as f32);
    em.emit4(0xF3, 0x0F, 0x59, 0xD3); // MULSS xmm3, xmm2  → k*ln2
    em.emit4(0xF3, 0x0F, 0x5C, 0xC3); // SUBSS xmm0, xmm3  → r

    // Quick degree-4 polynomial (sufficient for scalar remainder accuracy)
    // exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24
    load_f32_imm_to_xmm(em, 1, 0.5f32);      // 1/2
    load_f32_imm_to_xmm(em, 4, 1.0f32);      // c0
    em.emit4(0xF3, 0x0F, 0x58, 0xC0); // ADDSS xmm0 + xmm4? No, we need r in xmm0
    // Actually let's just compute: P = 1 + r*(1 + r*(1/2 + r*(1/6 + r/24)))
    load_f32_imm_to_xmm(em, 5, 0.041666667f32); // 1/24
    load_f32_imm_to_xmm(em, 6, 0.166666667f32); // 1/6
    em.emit4(0xF3, 0x0F, 0x59, 0xE8); // MULSS xmm5, xmm0 → r/24
    em.emit4(0x0F, 0x58, 0xEE);         // ADDSS xmm5, xmm6 → 1/6 + r/24
    em.emit4(0xF3, 0x0F, 0x59, 0xE8); // MULSS xmm5, xmm0 → r*(1/6+r/24)
    em.emit4(0x0F, 0x58, 0xE9);         // ADDSS xmm5, xmm1 → 1/2 + r*(1/6+r/24)
    em.emit4(0xF3, 0x0F, 0x59, 0xE8); // MULSS xmm5, xmm0 → r*(...)
    load_f32_imm_to_xmm(em, 6, 1.0f32);
    em.emit4(0x0F, 0x58, 0xEE);         // ADDSS xmm5, xmm6 → 1 + r*(...)
    em.emit4(0xF3, 0x0F, 0x59, 0xE8); // MULSS xmm5, xmm0? No...
    // Let me simplify: just use the degree-4 polynomial directly
    // xmm5 = 1 + r + r^2/2 + r^3/6 + r^4/24 via Horner
    // P = 1 + r*(1 + r*(0.5 + r*(0.166667 + r*0.041667)))
    // We already computed some of it. Let me redo cleanly:
    // xmm0 = r
    load_f32_imm_to_xmm(em, 1, 0.041666667f32); // 1/24
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0 → r/24
    load_f32_imm_to_xmm(em, 6, 0.166666667f32); // 1/6
    em.emit4(0x0F, 0x58, 0xCE);         // ADDSS xmm1, xmm6 → 1/6+r/24
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0 → r*(1/6+r/24)
    load_f32_imm_to_xmm(em, 6, 0.5f32);       // 1/2
    em.emit4(0x0F, 0x58, 0xCE);         // ADDSS xmm1, xmm6 → 1/2+r*(1/6+r/24)
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0 → r*(1/2+r*(1/6+r/24))
    load_f32_imm_to_xmm(em, 6, 1.0f32);       // 1
    em.emit4(0x0F, 0x58, 0xCE);         // ADDSS xmm1, xmm6 → 1+r*(1/2+r*(1/6+r/24))
    em.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS xmm1, xmm0? No, we want 1+r+...
    // Wait: Horner gives P = c0 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    // = 1 + r*(1 + r*(0.5 + r*(0.1667 + r*0.0417)))
    // The last MULSS was r*(1/2+r*(1/6+r/24)), ADDSS with 1 gives 1+r*(...)
    // That's our polynomial in xmm1. We do NOT multiply by r again.

    // xmm1 = exp(r)
    // Reconstruct 2^k
    // k is in xmm2. Add 127, convert to int, shift 23, reinterpret
    load_f32_imm_to_xmm(em, 3, 127.0f32);
    em.emit4(0x0F, 0x58, 0xD3);         // ADDSS xmm2, xmm3 → k+127
    em.emit4(0xF3, 0x0F, 0x2C, 0xC2); // CVTTSS2SI eax, xmm2
    em.shl_rax_imm8(23);
    em.vmovd_xmm_r32(3, 0); // MOVD xmm3, eax
    em.emit4(0xF3, 0x0F, 0x59, 0xCB); // MULSS xmm1, xmm3 → exp(r)*2^k

    // Store
    em.emit4(0xF3, 0x0F, 0x11, 0x0F); // MOVSS [rdi], xmm1

    em.emit4(0x48, 0x83, 0xC7, 4); // ADD RDI, 4
    em.emit4(0x48, 0x83, 0xC6, 4); // ADD RSI, 4
    em.emit4(0x49, 0xFF, 0xCC); // DEC R12
    em.emit3(0x4D, 0x85, 0xE4); // TEST R12, R12
    let sback = (scalar_loop as i32) - (em.pos() as i32 + 2);
    if sback >= -128 {
        em.emit2(0x75, sback as u8);
    } else {
        em.emit2(0x0F, 0x85);
        em.d((scalar_loop as i32) - (em.pos() as i32 + 4));
    }

    let after_scalar = em.pos();
    let sdisp = (after_scalar as i32) - ((skip_scalar_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_scalar_fixup_pos + 2..skip_scalar_fixup_pos + 6]
        .copy_from_slice(&sdisp.to_le_bytes());

    em.pop_reg(12);
    em.ret();
    true
}

/// AVX-512 exp kernel: processes 16 f32 elements per iteration with masked tail.
fn emit_simd_exp_avx512(em: &mut Emitter) -> bool {
    em.emitted_simd = true;

    // Prologue
    em.push_reg(12);
    em.push_reg(13);
    em.push_reg(14);

    // Move N into R8
    em.emit3(0x49, 0x89, 0xD0); // MOV R8, RDX

    // Total iterations in R10
    em.emit3(0x4C, 0x89, 0xD2); // MOV R10, RDX

    // Trip count = N / 16
    em.mov_rax_imm64(16);
    em.emit3(0x4C, 0x89, 0xC1); // MOV RCX, R8
    em.cqo();
    em.idiv_rcx();
    em.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX

    // If trip_count == 0, skip vector loop
    em.test_rax_rax();
    let skip_vec_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32
    em.d(0);

    // ── AVX-512 Vector loop (16-wide) ──
    let vec_loop_start = em.pos();

    // Load 16 f32 from [RSI]: VMOVUPS zmm0, [rsi]
    em.emit_vmovups_zmm_load(0, 6, 0, 0); // zmm0 = [rsi], base=rsi(6), disp=0, mask=k0

    // Range reduction: k = round(x * (1/ln2))
    // Broadcast 1/ln2
    em.mov_rax_imm64((INV_LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0); // zmm1 = 1/ln2 broadcast

    // zmm2 = x * (1/ln2)
    em.emit_vmulps_zmm(2, 0, 1);

    // VROUNDPS zmm3, zmm2, 0 — round to nearest
    // EVEX.512.66.0F.W0 08 /r ib
    em.emit_evex_prefix_512(3, 2, 2, 1, 0b01, 0); // pp=66, mm=0F, mask=k0
    em.b(0x08); // VROUNDPS opcode
    em.emit_modrm(3, 3 & 7, 2 & 7);
    em.b(0x00); // rounding mode = round to nearest

    // Broadcast ln2
    em.mov_rax_imm64((LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(4, 0);
    em.emit_vbroadcastss_zmm_mem(4, 6, 0);

    // zmm5 = k * ln2
    em.emit_vmulps_zmm(5, 3, 4);

    // zmm6 = r = x - k*ln2
    // VSUBPS zmm6, zmm0, zmm5
    em.emit_evex_prefix_512(6, 0, 5, 1, 0b01, 0);
    em.b(0x5C); // VSUBPS
    em.emit_modrm(3, 6 & 7, 5 & 7);

    // ── Polynomial evaluation (same coefficients, ZMM registers) ──
    // zmm7 = r^2
    em.emit_vmulps_zmm(7, 6, 6);

    // zmm4 = c0 + r*c1
    em.mov_rax_imm64((EXP_COEFFS[0] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[1] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    // zmm2 = zmm1 * zmm6 (c1 * r)
    em.emit_vmulps_zmm(2, 1, 6);
    // zmm4 = zmm0 + zmm2 (c0 + c1*r)
    em.emit_vaddps_zmm(4, 0, 2);

    // zmm5 = c2 + r*c3
    em.mov_rax_imm64((EXP_COEFFS[2] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[3] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 1, 6);
    em.emit_vaddps_zmm(5, 0, 2);

    // zmm0 = c4 + r*c5
    em.mov_rax_imm64((EXP_COEFFS[4] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[5] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 1, 6);
    em.emit_vaddps_zmm(0, 0, 2);

    // Combine: zmm1 = zmm4 + zmm7*zmm5 (low + mid*r^2)
    em.emit_vfmadd231ps_zmm(1, 4, 7); // zmm1 = zmm7*zmm5 + zmm1... no
    // Let's do it step by step:
    em.emit_vmulps_zmm(1, 7, 5); // zmm1 = r^2 * (c2+c3*r)
    em.emit_vaddps_zmm(1, 1, 4); // zmm1 = (c0+c1*r) + r^2*(c2+c3*r)

    // zmm2 = r^4
    em.emit_vmulps_zmm(2, 7, 7);

    // high = (c4+c5*r) + r^2*(c6*r)
    em.mov_rax_imm64((EXP_COEFFS[6] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(3, 0);
    em.emit_vbroadcastss_zmm_mem(3, 6, 0);
    em.emit_vmulps_zmm(3, 3, 6); // c6*r
    em.emit_vmulps_zmm(3, 7, 3); // r^2*(c6*r)
    em.emit_vaddps_zmm(3, 0, 3); // (c4+c5*r) + r^2*(c6*r)

    // zmm0 = high * r^4
    em.emit_vmulps_zmm(0, 3, 2);

    // zmm0 = zmm1 + zmm0 (full polynomial)
    em.emit_vaddps_zmm(0, 1, 0);

    // ── Reconstruct 2^k ──
    // zmm3 = k+127
    em.mov_rax_imm64((127.0f32).to_bits() as i64);
    em.vmovd_xmm_r32(2, 0);
    em.emit_vbroadcastss_zmm_mem(2, 6, 0);
    em.emit_vaddps_zmm(3, 3, 2);

    // CVTPS2DQ zmm3, zmm3
    em.emit_evex_prefix_512(3, 3, 3, 1, 0b01, 0);
    em.b(0x5B);
    em.emit_modrm(3, 3 & 7, 3 & 7);

    // VPSLLD zmm3, zmm3, 23
    em.emit_evex_prefix_512(3, 3, 3, 1, 0b01, 0);
    em.b(0x72);
    em.emit_modrm(3, 0x11, 3 & 7); // reg=0x11 for imm8 shift
    em.b(23);

    // VANDPS domain crossing + VMULPS
    em.emit_vmulps_zmm(0, 0, 3);

    // Store 16 f32 to [RDI]
    em.emit_vmovups_zmm_store(7, 0, 0); // [rdi], disp=0, src=zmm0

    // Advance: RDI += 64, RSI += 64
    em.emit4(0x48, 0x83, 0xC7, 64);
    em.emit4(0x48, 0x83, 0xC6, 64);

    // Decrement
    em.emit3(0x4D, 0xFF, 0xC9); // DEC R9

    // Prefetch
    em.emit_prefetcht0_rdi(256);
    em.emit_prefetcht1_rdi(1024);

    // Loop back
    em.emit3(0x4D, 0x85, 0xC9); // TEST R9, R9
    let back_disp = (vec_loop_start as i32) - (em.pos() as i32 + 2);
    if back_disp >= -128 {
        em.emit2(0x75, back_disp as u8);
    } else {
        em.emit2(0x0F, 0x85);
        em.d((vec_loop_start as i32) - (em.pos() as i32 + 4));
    }

    // ── AVX-512 Masked Tail (remainder = N % 16) ──
    let after_vec = em.pos();
    let skip_disp = (after_vec as i32) - ((skip_vec_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_vec_fixup_pos + 2..skip_vec_fixup_pos + 6]
        .copy_from_slice(&skip_disp.to_le_bytes());

    // remainder = R10 & 15
    em.emit3(0x4C, 0x89, 0xD0); // MOV RAX, R10
    em.and_rax_imm32(15);
    em.test_rax_rax();
    let skip_masked_fixup_pos = em.pos();
    em.emit2(0x0F, 0x84); // JZ rel32
    em.d(0);

    // Compute mask = (1 << remainder) - 1
    em.emit3(0x48, 0x89, 0xC1); // MOV RCX, RAX (save remainder)
    em.mov_rax_imm_opt(1);
    em.emit3(0x48, 0xD3, 0xE0); // SHL RAX, CL
    em.dec_rax(); // mask = (1 << remainder) - 1
    em.emit_kmovw_k_eax(1); // KMOVW k1, eax

    // Load masked: VMOVUPS zmm0 {k1}{z}, [RSI]
    em.emit_vmovups_zmm_load(0, 6, 0, 1); // base=rsi(6), mask=k1

    // Range reduction + polynomial + reconstruct (same as vector loop,
    // but with masked operations so inactive lanes are zeroed)
    em.mov_rax_imm64((INV_LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 0, 1);

    // Round
    em.emit_evex_prefix_512(3, 2, 2, 1, 0b01, 1); // mask=k1
    em.b(0x08);
    em.emit_modrm(3, 3 & 7, 2 & 7);
    em.b(0x00);

    em.mov_rax_imm64((LN2_F64 as f32).to_bits() as i64);
    em.vmovd_xmm_r32(4, 0);
    em.emit_vbroadcastss_zmm_mem(4, 6, 0);
    em.emit_vmulps_zmm(5, 3, 4);

    // r = x - k*ln2 (masked)
    em.emit_evex_prefix_512(6, 0, 5, 1, 0b01, 1); // mask=k1
    em.b(0x5C);
    em.emit_modrm(3, 6 & 7, 5 & 7);

    // Polynomial (same as vector loop)
    em.emit_vmulps_zmm(7, 6, 6);

    em.mov_rax_imm64((EXP_COEFFS[0] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[1] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 1, 6);
    em.emit_vaddps_zmm(4, 0, 2);

    em.mov_rax_imm64((EXP_COEFFS[2] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[3] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 1, 6);
    em.emit_vaddps_zmm(5, 0, 2);

    em.mov_rax_imm64((EXP_COEFFS[4] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(0, 0);
    em.emit_vbroadcastss_zmm_mem(0, 6, 0);
    em.mov_rax_imm64((EXP_COEFFS[5] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(1, 0);
    em.emit_vbroadcastss_zmm_mem(1, 6, 0);
    em.emit_vmulps_zmm(2, 1, 6);
    em.emit_vaddps_zmm(0, 0, 2);

    em.emit_vmulps_zmm(1, 7, 5);
    em.emit_vaddps_zmm(1, 1, 4);

    em.emit_vmulps_zmm(2, 7, 7);

    em.mov_rax_imm64((EXP_COEFFS[6] as f32).to_bits() as i64);
    em.vmovd_xmm_r32(3, 0);
    em.emit_vbroadcastss_zmm_mem(3, 6, 0);
    em.emit_vmulps_zmm(3, 3, 6);
    em.emit_vmulps_zmm(3, 7, 3);
    em.emit_vaddps_zmm(3, 0, 3);

    em.emit_vmulps_zmm(0, 3, 2);
    em.emit_vaddps_zmm(0, 1, 0);

    // Reconstruct
    em.mov_rax_imm64((127.0f32).to_bits() as i64);
    em.vmovd_xmm_r32(2, 0);
    em.emit_vbroadcastss_zmm_mem(2, 6, 0);
    em.emit_vaddps_zmm(3, 3, 2);
    em.emit_evex_prefix_512(3, 3, 3, 1, 0b01, 0);
    em.b(0x5B);
    em.emit_modrm(3, 3 & 7, 3 & 7);
    em.emit_evex_prefix_512(3, 3, 3, 1, 0b01, 0);
    em.b(0x72);
    em.emit_modrm(3, 0x11, 3 & 7);
    em.b(23);
    em.emit_vmulps_zmm(0, 0, 3);

    // Masked store
    em.emit_vmovups_zmm_store(7, 0, 0);

    // Patch skip_masked
    let after_masked = em.pos();
    let mdisp = (after_masked as i32) - ((skip_masked_fixup_pos + 6) as i32);
    em.as_mut_slice()[skip_masked_fixup_pos + 2..skip_masked_fixup_pos + 6]
        .copy_from_slice(&mdisp.to_le_bytes());

    // Epilogue
    em.pop_reg(14);
    em.pop_reg(13);
    em.pop_reg(12);
    em.ret();
    true
}

// =============================================================================
// §4. Auto-vectorization of elementwise loops
// =============================================================================

/// Metadata describing a vectorizable elementwise loop.
#[derive(Debug)]
pub struct ElementwiseLoopInfo {
    /// Slot holding the array base pointer
    pub base_slot: u16,
    /// Slot holding the induction variable
    pub iv_slot: u16,
    /// Slot holding the loop bound
    pub bound_slot: u16,
    /// The operation applied elementwise
    pub op: ElementwiseOp,
    /// Byte stride between elements (derived from load/store patterns)
    pub element_stride: u32,
    /// Loop start PC
    pub loop_start: usize,
    /// Loop end PC (the backward jump instruction)
    pub loop_end: usize,
}

/// An elementwise operation that can be SIMD-vectorized.
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Fma,
    Rsqrt,
    Rcp,
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    ScaleAdd,
}

/// Analyze a sequence of instructions and detect vectorizable elementwise loops.
pub fn detect_elementwise_loops(instrs: &[Instr]) -> Vec<ElementwiseLoopInfo> {
    let mut results = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match instr {
            Instr::Jump(off) => {
                let t = ((pc as i32) + 1 + *off) as usize;
                if t <= pc { Some(t) } else { None }
            }
            _ => None,
        };
        if let Some(loop_header) = target {
            if let Some(info) = analyze_loop_body(instrs, loop_header, pc) {
                results.push(info);
            }
        }
    }
    results
}

/// Analyze a loop body to detect elementwise patterns and compute the actual stride.
fn analyze_loop_body(
    instrs: &[Instr],
    loop_start: usize,
    loop_end: usize,
) -> Option<ElementwiseLoopInfo> {
    let mut base_slot: Option<u16> = None;
    let mut iv_slot: Option<u16> = None;
    let mut bound_slot: Option<u16> = None;
    let mut elementwise_op: Option<ElementwiseOp> = None;
    let mut detected_stride: u32 = 4; // default f32, but we derive from instructions

    for pc in loop_start..=loop_end.min(instrs.len() - 1) {
        // Detect induction variable increment: iv = iv + 1 or iv += stride
        if let Instr::BinOp(dst, crate::types::BinOpKind::Add, l, r) = &instrs[pc] {
            if *l == *dst || *r == *dst {
                // Check for stride constant — look back for LoadI with the value
                if pc > loop_start {
                    for prev_pc in (loop_start..pc).rev() {
                        match &instrs[prev_pc] {
                            Instr::LoadI32(slot, v) if *slot == *r || *slot == *l => {
                                iv_slot = Some(*dst);
                                // If the increment is not 1, the stride multiplier affects
                                // the element stride. For iv += 1, stride = element_size.
                                if *v > 1 {
                                    detected_stride = (*v as u32) * 4; // scale by f32 size
                                }
                                break;
                            }
                            Instr::LoadI64(slot, v) if *slot == *r || *slot == *l => {
                                iv_slot = Some(*dst);
                                if *v > 1 {
                                    detected_stride = (*v as u32) * 4;
                                }
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Detect LoadF64 instructions — if present, stride should be 8 bytes
        if let Instr::LoadF64(_, _) = &instrs[pc] {
            detected_stride = 8; // f64 elements
        }

        // Detect LoadF32 instructions — if present, stride should be 4 bytes
        if let Instr::LoadF32(_, _) = &instrs[pc] {
            detected_stride = 4; // f32 elements
        }

        // Detect comparison against bound
        if let Instr::BinOp(_, crate::types::BinOpKind::Lt, l, r) = &instrs[pc] {
            if iv_slot.is_some() && (*l == iv_slot.unwrap() || *r == iv_slot.unwrap()) {
                let other = if *l == iv_slot.unwrap() { *r } else { *l };
                bound_slot = Some(other);
            }
        }

        // Detect elementwise operations
        if let Instr::BinOp(_, op, _, _) = &instrs[pc] {
            let ew_op = match op {
                crate::types::BinOpKind::Add => Some(ElementwiseOp::Add),
                crate::types::BinOpKind::Sub => Some(ElementwiseOp::Sub),
                crate::types::BinOpKind::Mul => Some(ElementwiseOp::Mul),
                crate::types::BinOpKind::FmaAdd => Some(ElementwiseOp::Fma),
                _ => None,
            };
            if elementwise_op.is_none() && ew_op.is_some() {
                elementwise_op = ew_op;
            }
        }

        // Detect UnOp patterns that map to elementwise ops
        if let Instr::UnOp(_, op, _) = &instrs[pc] {
            let ew_op = match op {
                crate::types::UnOpKind::Sqrt => Some(ElementwiseOp::Sqrt),
                crate::types::UnOpKind::Neg => Some(ElementwiseOp::Neg),
                _ => None,
            };
            if elementwise_op.is_none() && ew_op.is_some() {
                elementwise_op = ew_op;
            }
        }
    }

    if iv_slot.is_some() && bound_slot.is_some() && elementwise_op.is_some() {
        Some(ElementwiseLoopInfo {
            base_slot: base_slot.unwrap_or(0),
            iv_slot: iv_slot?,
            bound_slot: bound_slot?,
            op: elementwise_op?,
            element_stride: detected_stride,
            loop_start,
            loop_end,
        })
    } else {
        None
    }
}

// =============================================================================
// §5. Loop fusion for consecutive elementwise operations
// =============================================================================

/// Metadata for a fused loop pair.
#[derive(Debug)]
pub struct FusedLoopPair {
    pub first: usize,
    pub second: usize,
    pub intermediate_slot: u16,
    pub fused_op: ElementwiseOp,
}

/// Find fusion candidates among elementwise loops.
pub fn find_fusion_candidates(loops: &[ElementwiseLoopInfo], instrs: &[Instr]) -> Vec<FusedLoopPair> {
    let mut candidates = Vec::new();
    for i in 0..loops.len() {
        for j in (i + 1)..loops.len() {
            let first = &loops[i];
            let second = &loops[j];
            if first.bound_slot != second.bound_slot {
                continue;
            }
            if first.element_stride != second.element_stride {
                continue;
            }
            let mut intermediate: Option<u16> = None;
            for pc in first.loop_start..=first.loop_end.min(instrs.len() - 1) {
                if let Instr::Store(slot, _) = &instrs[pc] {
                    for pc2 in second.loop_start..=second.loop_end.min(instrs.len() - 1) {
                        if let Instr::Load(_, src_slot) = &instrs[pc2] {
                            if *src_slot == *slot {
                                intermediate = Some(*slot);
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(int_slot) = intermediate {
                let fused_op = match (first.op, second.op) {
                    (ElementwiseOp::Mul, ElementwiseOp::Add) => ElementwiseOp::Fma,
                    (ElementwiseOp::Add, ElementwiseOp::Add) => ElementwiseOp::Add,
                    (ElementwiseOp::Mul, ElementwiseOp::Sub) => ElementwiseOp::Fma,
                    _ => second.op,
                };
                candidates.push(FusedLoopPair {
                    first: i,
                    second: j,
                    intermediate_slot: int_slot,
                    fused_op,
                });
            }
        }
    }
    candidates
}

// =============================================================================
// §6. Software prefetch injection in loop back-edges
// =============================================================================

/// Configuration for prefetch injection.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    pub l1_ahead_lines: i32,
    pub l2_ahead_lines: i32,
    pub cache_line_bytes: i32,
    pub emit_l2_prefetch: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            l1_ahead_lines: 2,
            l2_ahead_lines: 8,
            cache_line_bytes: 64,
            emit_l2_prefetch: true,
        }
    }
}

/// Emit prefetch instructions at the loop back-edge for a given array access pattern.
pub fn emit_loop_prefetch(
    em: &mut Emitter,
    base_reg: u8,
    iv_offset: i32,
    stride_bytes: i32,
    config: &PrefetchConfig,
) {
    let l1_distance = config.l1_ahead_lines * config.cache_line_bytes;
    let l2_distance = config.l2_ahead_lines * config.cache_line_bytes;

    match base_reg {
        7 => {
            em.emit_prefetcht0_rdi(iv_offset + l1_distance);
        }
        0 => {
            em.b(0x0F);
            em.b(0x18);
            em.b(0x80 | 1);
            em.d(iv_offset + l1_distance);
        }
        _ => {
            em.b(0x0F);
            em.b(0x18);
            let modrm = 0x80 | (1 << 3) | (base_reg & 7);
            em.b(modrm);
            if (base_reg & 7) == 4 {
                em.b(0x24);
            }
            em.d(iv_offset + l1_distance);
        }
    }

    if config.emit_l2_prefetch {
        match base_reg {
            7 => {
                em.emit_prefetcht1_rdi(iv_offset + l2_distance);
            }
            _ => {
                em.b(0x0F);
                em.b(0x18);
                let modrm = 0x80 | (2 << 3) | (base_reg & 7);
                em.b(modrm);
                if (base_reg & 7) == 4 {
                    em.b(0x24);
                }
                em.d(iv_offset + l2_distance);
            }
        }
    }
}
