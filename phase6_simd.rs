//
// ┌─ Architecture overview ────────────────────────────────────────────────────┐
// │  Runtime dispatch:  AVX-512F → AVX2+FMA → AVX → SSE2 → scalar            │
// │  Data layout:       AoS ([f32;3] slices) → in-register SoA transpose      │
// │                     → arithmetic → SoA→AoS store                          │
// │  Transpose kernels: zero-copy, zero-heap, proven shuffle sequences         │
// │  Loop structure:    4× unrolled AVX2 main loop, 2× unrolled fallbacks      │
// │  Prefetch:          _MM_HINT_T0, 3 cache lines ahead (192 B / 48 floats)  │
// └────────────────────────────────────────────────────────────────────────────┘
//
// TODO: Implement SIMD code generation for vectorizable loops in the JIT.
// Currently, the JIT (phase3_jit.rs) emits scalar operations even for loops
// that could benefit from SSE/AVX packed operations. Planned: detect
// consecutive Add/Mul operations on array elements and emit PADDD/PMULUDQ
// instead of scalar ADD/MUL. This would complement the existing SIMD
// runtime dispatch in this module (which handles position/velocity updates)
// with JIT-time vectorization for user-level array operations.
//
// ┌─ AoS→SoA transpose reference ─────────────────────────────────────────────┐
// │                                                                            │
// │  load_aos4  (SSE2)  – 4 particles, 12 f32 → 3×__m128                     │
// │    Packed-AoS layout: A=[x0 y0 z0 x1]  B=[y1 z1 x2 y2]  C=[z2 x3 y3 z3] │
// │    4 unpacks + 9 shuffles, all in-lane, zero cross-lane traffic.           │
// │    imm8 constants derived from the SHUFPS spec                             │
// │    (felixcloutier.com/x86/shufps) and verified by round-trip unit test.   │
// │                                                                            │
// │  load_aos8  (AVX2)  – 8 particles, 24 f32 → 3×__m256                     │
// │    Two independent load_aos4 on lo/hi 128-bit lanes + _mm256_set_m128.   │
// │    Avoids all latency-3 permute2f128 cross-lane shuffles.                 │
// │                                                                            │
// │  load_aos16 (AVX-512) – 16 particles, 48 f32 → 3×__m512                  │
// │    Two load_aos8 + _mm512_insertf32x8.                                    │
// │                                                                            │
// │  store_* are exact inverses of their load_* counterparts, same cost.      │
// └────────────────────────────────────────────────────────────────────────────┘

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn simd_available() -> bool {
    true
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn simd_available() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{__m128, __m256, __m512};
use std::sync::OnceLock;

type UpdateFn = fn(&mut [[f32; 3]], &[[f32; 3]], f32, usize);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn pick_update_impl() -> UpdateFn {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
        return |p, v, dt, n| unsafe { update_avx512(p, v, dt, n) };
    }
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
        return |p, v, dt, n| unsafe { update_avx2_fma(p, v, dt, n) };
    }
    if is_x86_feature_detected!("avx") {
        return |p, v, dt, n| unsafe { update_avx(p, v, dt, n) };
    }
    if is_x86_feature_detected!("sse2") {
        return |p, v, dt, n| unsafe { update_sse2(p, v, dt, n) };
    }
    scalar_tail
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn pick_update_impl() -> UpdateFn {
    scalar_tail
}

// ── Public dispatch ───────────────────────────────────────────────────────────
//
// #[inline(never)]: keeps caller code lean; branch predictor locks on first call.
#[inline(never)]
pub fn update_positions(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32) {
    let n = positions.len().min(velocities.len());
    if n == 0 {
        return;
    }
    static UPDATE_DISPATCH: OnceLock<UpdateFn> = OnceLock::new();
    let update = *UPDATE_DISPATCH.get_or_init(pick_update_impl);
    update(positions, velocities, dt, n);
}

// ── AVX-512F  (16 particles / iter) ──────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn update_avx512(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32, n: usize) {
    use std::arch::x86_64::*;
    let dtv = _mm512_set1_ps(dt);
    let mut i = 0usize;

    while i + 16 <= n {
        let off = i * 3;
        // Prefetch 3 cache lines ahead: 48 floats × 4 bytes = 192 bytes = 3 × 64 B.
        _mm_prefetch(
            positions.as_ptr().cast::<i8>().add((off + 48) * 4),
            _MM_HINT_T0,
        );
        _mm_prefetch(
            velocities.as_ptr().cast::<i8>().add((off + 48) * 4),
            _MM_HINT_T0,
        );

        let (px, py, pz) = load_aos16(positions.as_ptr().cast::<f32>().add(off));
        let (vx, vy, vz) = load_aos16(velocities.as_ptr().cast::<f32>().add(off));

        let ox = _mm512_fmadd_ps(vx, dtv, px);
        let oy = _mm512_fmadd_ps(vy, dtv, py);
        let oz = _mm512_fmadd_ps(vz, dtv, pz);

        store_aos16(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
        i += 16;
    }
    // Handle tail elements
    for j in i..n {
        positions[j][0] = velocities[j][0].mul_add(dt, positions[j][0]);
        positions[j][1] = velocities[j][1].mul_add(dt, positions[j][1]);
        positions[j][2] = velocities[j][2].mul_add(dt, positions[j][2]);
    }
}

// ── AVX2 + FMA  (32 particles / iter, 4× unrolled) ───────────────────────────
//
// 4× unroll saturates the FMA pipeline (latency 4, throughput 0.5 on Haswell+)
// and keeps the prefetch buffer primed across 128 B of AoS data per block.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn update_avx2_fma(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32, n: usize) {
    use std::arch::x86_64::*;
    let dtv = _mm256_set1_ps(dt);

    // `[[f32; 3]]` has 4-byte element alignment, so align_to::<__m256>()
    // rarely provides a useful aligned prefix in practice. Start at 0 and
    // let the unaligned loads/stores handle all iterations.
    let mut i = 0usize;

    // 4× unrolled main body: 4 × 8 = 32 particles per iteration.
    while i + 32 <= n {
        macro_rules! blk {
            ($k:expr) => {{
                let off = (i + $k) * 3;
                _mm_prefetch(
                    positions.as_ptr().cast::<i8>().add((off + 48) * 4),
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    velocities.as_ptr().cast::<i8>().add((off + 48) * 4),
                    _MM_HINT_T0,
                );
                let (px, py, pz) = load_aos8(positions.as_ptr().cast::<f32>().add(off));
                let (vx, vy, vz) = load_aos8(velocities.as_ptr().cast::<f32>().add(off));
                let ox = _mm256_fmadd_ps(vx, dtv, px);
                let oy = _mm256_fmadd_ps(vy, dtv, py);
                let oz = _mm256_fmadd_ps(vz, dtv, pz);
                store_aos8(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
            }};
        }
        blk!(0);
        blk!(8);
        blk!(16);
        blk!(24);
        i += 32;
    }

    // Drain remaining complete 8-particle blocks.
    while i + 8 <= n {
        let off = i * 3;
        let (px, py, pz) = load_aos8(positions.as_ptr().cast::<f32>().add(off));
        let (vx, vy, vz) = load_aos8(velocities.as_ptr().cast::<f32>().add(off));
        let ox = _mm256_fmadd_ps(vx, dtv, px);
        let oy = _mm256_fmadd_ps(vy, dtv, py);
        let oz = _mm256_fmadd_ps(vz, dtv, pz);
        store_aos8(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
        i += 8;
    }
    // Handle tail elements
    for j in i..n {
        positions[j][0] = velocities[j][0].mul_add(dt, positions[j][0]);
        positions[j][1] = velocities[j][1].mul_add(dt, positions[j][1]);
        positions[j][2] = velocities[j][2].mul_add(dt, positions[j][2]);
    }
}

// ── AVX only  (16 particles / iter, 2× unrolled) ─────────────────────────────
//
// No FMA: separate mul + add.  Two 8-wide blocks per iteration let the
// out-of-order engine interleave loads and arithmetic freely.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn update_avx(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32, n: usize) {
    use std::arch::x86_64::*;
    let dtv = _mm256_set1_ps(dt);
    let mut i = 0usize;

    while i + 16 <= n {
        macro_rules! blk {
            ($k:expr) => {{
                let off = (i + $k) * 3;
                let (px, py, pz) = load_aos8(positions.as_ptr().cast::<f32>().add(off));
                let (vx, vy, vz) = load_aos8(velocities.as_ptr().cast::<f32>().add(off));
                let ox = _mm256_add_ps(px, _mm256_mul_ps(vx, dtv));
                let oy = _mm256_add_ps(py, _mm256_mul_ps(vy, dtv));
                let oz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dtv));
                store_aos8(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
            }};
        }
        blk!(0);
        blk!(8);
        i += 16;
    }
    if i + 8 <= n {
        let off = i * 3;
        let (px, py, pz) = load_aos8(positions.as_ptr().cast::<f32>().add(off));
        let (vx, vy, vz) = load_aos8(velocities.as_ptr().cast::<f32>().add(off));
        let ox = _mm256_add_ps(px, _mm256_mul_ps(vx, dtv));
        let oy = _mm256_add_ps(py, _mm256_mul_ps(vy, dtv));
        let oz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dtv));
        store_aos8(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
        i += 8;
    }
    // Handle tail elements
    for j in i..n {
        positions[j][0] = velocities[j][0].mul_add(dt, positions[j][0]);
        positions[j][1] = velocities[j][1].mul_add(dt, positions[j][1]);
        positions[j][2] = velocities[j][2].mul_add(dt, positions[j][2]);
    }
}

// ── SSE2  (8 particles / iter, 2× unrolled) ──────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn update_sse2(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32, n: usize) {
    use std::arch::x86_64::*;
    let dtv = _mm_set1_ps(dt);
    let mut i = 0usize;

    while i + 8 <= n {
        macro_rules! blk {
            ($k:expr) => {{
                let off = (i + $k) * 3;
                let (px, py, pz) = load_aos4(positions.as_ptr().cast::<f32>().add(off));
                let (vx, vy, vz) = load_aos4(velocities.as_ptr().cast::<f32>().add(off));
                let ox = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
                let oy = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
                let oz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));
                store_aos4(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
            }};
        }
        blk!(0);
        blk!(4);
        i += 8;
    }
    if i + 4 <= n {
        let off = i * 3;
        let (px, py, pz) = load_aos4(positions.as_ptr().cast::<f32>().add(off));
        let (vx, vy, vz) = load_aos4(velocities.as_ptr().cast::<f32>().add(off));
        let ox = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
        let oy = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
        let oz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));
        store_aos4(positions.as_mut_ptr().cast::<f32>().add(off), ox, oy, oz);
        i += 4;
    }
    // Handle tail elements
    for j in i..n {
        positions[j][0] = velocities[j][0].mul_add(dt, positions[j][0]);
        positions[j][1] = velocities[j][1].mul_add(dt, positions[j][1]);
        positions[j][2] = velocities[j][2].mul_add(dt, positions[j][2]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AoS ↔ SoA register transposition
// ═══════════════════════════════════════════════════════════════════════════════
//
// ── load_aos4  (SSE2, 4 particles × 3 components = 12 f32) ───────────────────
//
//  Memory layout (row-packed AoS, stride 3):
//    A = ptr[ 0.. 4] = [x0 y0 z0 x1]
//    B = ptr[ 4.. 8] = [y1 z1 x2 y2]
//    C = ptr[ 8..12] = [z2 x3 y3 z3]
//
//  Goal (SoA):
//    x = [x0 x1 x2 x3]
//    y = [y0 y1 y2 y3]
//    z = [z0 z1 z2 z3]
//
//  Source identity table (AoS index → SoA lane):
//    x0=A[0], x1=A[3], x2=B[2], x3=C[1]
//    y0=A[1], y1=B[0], y2=B[3], y3=C[2]
//    z0=A[2], z1=B[1], z2=C[0], z3=C[3]
//
//  Algorithm (4 unpacks + 9 shuffles, fully in-lane, no memory round-trip):
//
//    T0 = unpacklo(A, B) = [A[0] B[0] A[1] B[1]] = [x0 y1 y0 z1]
//    T1 = unpackhi(A, B) = [A[2] B[2] A[3] B[3]] = [z0 x2 x1 y2]
//    T2 = unpacklo(B, C) = [B[0] C[0] B[1] C[1]] = [y1 z2 z1 x3]
//    T3 = unpackhi(B, C) = [B[2] C[2] B[3] C[3]] = [x2 y3 y2 z3]
//
//  x extraction: x0=T0[0], x1=T1[2], x2=T3[0], x3=T2[3]
//    xA = shuffle(T0, T1, 0xA0)  // _MM_SHUFFLE(2,2,0,0): [T0[0] T0[0] T1[2] T1[2]] = [x0 x0 x1 x1]
//    xB = shuffle(T3, T2, 0xF0)  // _MM_SHUFFLE(3,3,0,0): [T3[0] T3[0] T2[3] T2[3]] = [x2 x2 x3 x3]
//    x  = shuffle(xA, xB, 0x88)  // _MM_SHUFFLE(2,0,2,0): [xA[0] xA[2] xB[0] xB[2]] = [x0 x1 x2 x3] ✓
//
//  y extraction: y0=T0[2], y1=T0[1], y2=T3[2], y3=T3[1]
//    yA = shuffle(T0, T0, 0x5A) // _MM_SHUFFLE(1,1,2,2): [T0[2] T0[2] T0[1] T0[1]] = [y0 y0 y1 y1]
//    yB = shuffle(T3, T3, 0x5A) //                       [T3[2] T3[2] T3[1] T3[1]] = [y2 y2 y3 y3]
//    y  = shuffle(yA, yB, 0x88) // [y0 y1 y2 y3] ✓
//
//  z extraction: z0=T1[0], z1=T2[2], z2=T2[1], z3=T3[3]
//    zA = shuffle(T1, T2, 0xA0) // [T1[0] T1[0] T2[2] T2[2]] = [z0 z0 z1 z1]
//    zB = shuffle(T2, T3, 0xF5) // _MM_SHUFFLE(3,3,1,1): [T2[1] T2[1] T3[3] T3[3]] = [z2 z2 z3 z3]
//    z  = shuffle(zA, zB, 0x88) // [z0 z1 z2 z3] ✓
//
//  imm8 encoding: _mm_shuffle_ps(a,b,imm) → result[i] = a[imm[2i+1:2i]] for i<2,
//                                                        b[imm[2i+1:2i]] for i≥2
//  (Reference: felixcloutier.com/x86/shufps, Intel SDM Vol.2 §4-609)
//
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn load_aos4(ptr: *const f32) -> (__m128, __m128, __m128) {
    use std::arch::x86_64::*;
    let a = _mm_loadu_ps(ptr);
    let b = _mm_loadu_ps(ptr.add(4));
    let c = _mm_loadu_ps(ptr.add(8));

    let t0 = _mm_unpacklo_ps(a, b); // [x0 y1 y0 z1]
    let t1 = _mm_unpackhi_ps(a, b); // [z0 x2 x1 y2]
    let t2 = _mm_unpacklo_ps(b, c); // [y1 z2 z1 x3]
    let t3 = _mm_unpackhi_ps(b, c); // [x2 y3 y2 z3]

    let xa = _mm_shuffle_ps(t0, t1, 0xA0); // [x0 x0 x1 x1]
    let xb = _mm_shuffle_ps(t3, t2, 0xF0); // [x2 x2 x3 x3]
    let x = _mm_shuffle_ps(xa, xb, 0x88); // [x0 x1 x2 x3]

    let ya = _mm_shuffle_ps(t0, t0, 0x5A); // [y0 y0 y1 y1]
    let yb = _mm_shuffle_ps(t3, t3, 0x5A); // [y2 y2 y3 y3]
    let y = _mm_shuffle_ps(ya, yb, 0x88); // [y0 y1 y2 y3]

    let za = _mm_shuffle_ps(t1, t2, 0xA0); // [z0 z0 z1 z1]
    let zb = _mm_shuffle_ps(t2, t3, 0xF5); // [z2 z2 z3 z3]
    let z = _mm_shuffle_ps(za, zb, 0x88); // [z0 z1 z2 z3]

    (x, y, z)
}

// ── store_aos4 (SSE2) — inverse of load_aos4 ─────────────────────────────────
//
//  From SoA x=[x0..x3], y=[y0..y3], z=[z0..z3] produce:
//    out0 = [x0 y0 z0 x1]
//    out1 = [y1 z1 x2 y2]
//    out2 = [z2 x3 y3 z3]
//
//  Derivation:
//    p_lo = unpacklo(x, y) = [x0 y0 x1 y1]
//    p_hi = unpackhi(x, y) = [x2 y2 x3 y3]
//
//  out0 = [x0 y0 z0 x1]:
//    q0   = shuffle(z, x, 0x50) // _MM_SHUFFLE(1,1,0,0): [z[0] z[0] x[1] x[1]] = [z0 z0 x1 x1]
//    out0 = shuffle(p_lo, q0, 0x84)
//         // _MM_SHUFFLE(2,0,1,0): [p_lo[0] p_lo[1] q0[0] q0[2]] = [x0 y0 z0 x1] ✓
//
//  out1 = [y1 z1 x2 y2]:
//    r1   = shuffle(p_lo, z, 0x5F)
//         // _MM_SHUFFLE(1,1,3,3): [p_lo[3] p_lo[3] z[1] z[1]] = [y1 y1 z1 z1]
//    out1 = shuffle(r1, p_hi, 0x48)
//         // _MM_SHUFFLE(1,0,2,0): [r1[0] r1[2] p_hi[0] p_hi[1]] = [y1 z1 x2 y2] ✓
//
//  out2 = [z2 x3 y3 z3]:
//    q2   = shuffle(z, x, 0xFA)  // _MM_SHUFFLE(3,3,2,2): [z2 z2 x3 x3]
//    r2   = shuffle(p_hi, z, 0xFF) // _MM_SHUFFLE(3,3,3,3): [p_hi[3] p_hi[3] z[3] z[3]] = [y3 y3 z3 z3]
//    out2 = shuffle(q2, r2, 0x88) // [q2[0] q2[2] r2[0] r2[2]] = [z2 x3 y3 z3] ✓
//
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn store_aos4(ptr: *mut f32, x: __m128, y: __m128, z: __m128) {
    use std::arch::x86_64::*;
    let p_lo = _mm_unpacklo_ps(x, y); // [x0 y0 x1 y1]
    let p_hi = _mm_unpackhi_ps(x, y); // [x2 y2 x3 y3]

    let q0 = _mm_shuffle_ps(z, x, 0x50); // [z0 z0 x1 x1]
    let out0 = _mm_shuffle_ps(p_lo, q0, 0x84); // [x0 y0 z0 x1]

    let r1 = _mm_shuffle_ps(p_lo, z, 0x5F); // [y1 y1 z1 z1]
    let out1 = _mm_shuffle_ps(r1, p_hi, 0x48); // [y1 z1 x2 y2]

    let q2 = _mm_shuffle_ps(z, x, 0xFA); // [z2 z2 x3 x3]
    let r2 = _mm_shuffle_ps(p_hi, z, 0xFF); // [y3 y3 z3 z3]
    let out2 = _mm_shuffle_ps(q2, r2, 0x88); // [z2 x3 y3 z3]

    _mm_storeu_ps(ptr, out0);
    _mm_storeu_ps(ptr.add(4), out1);
    _mm_storeu_ps(ptr.add(8), out2);
}

// ── load_aos8 (AVX2, 8 particles × 3 = 24 f32 → 3×__m256) ───────────────────
//
//  Two independent load_aos4 calls on [0..12) and [12..24), combined with
//  _mm256_set_m128.  This decomposes entirely into in-lane 128-bit operations,
//  avoiding the latency-3 cross-lane cost of permute2f128 on Haswell/Broadwell.
//  (Reference: Intel Community "Different ways to turn an AoS into an SoA",
//   Feb 2014; latency data from Intel Optimization Reference Manual §C.3)
//
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn load_aos8(ptr: *const f32) -> (__m256, __m256, __m256) {
    use std::arch::x86_64::*;
    let (x_lo, y_lo, z_lo) = load_aos4(ptr);
    let (x_hi, y_hi, z_hi) = load_aos4(ptr.add(12));
    (
        _mm256_set_m128(x_hi, x_lo),
        _mm256_set_m128(y_hi, y_lo),
        _mm256_set_m128(z_hi, z_lo),
    )
}

// ── store_aos8 (AVX2) ─────────────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn store_aos8(ptr: *mut f32, x: __m256, y: __m256, z: __m256) {
    use std::arch::x86_64::*;
    store_aos4(
        ptr,
        _mm256_castps256_ps128(x),
        _mm256_castps256_ps128(y),
        _mm256_castps256_ps128(z),
    );
    store_aos4(
        ptr.add(12),
        _mm256_extractf128_ps(x, 1),
        _mm256_extractf128_ps(y, 1),
        _mm256_extractf128_ps(z, 1),
    );
}

// ── load_aos16 (AVX-512F, 16 particles × 3 = 48 f32 → 3×__m512) ─────────────
//
//  Two load_aos8 on [0..24) and [24..48), merged via _mm512_insertf32x8.
//
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn load_aos16(ptr: *const f32) -> (__m512, __m512, __m512) {
    use std::arch::x86_64::*;
    let (x_lo, y_lo, z_lo) = load_aos8(ptr);
    let (x_hi, y_hi, z_hi) = load_aos8(ptr.add(24));
    (
        _mm512_insertf32x8(_mm512_castps256_ps512(x_lo), x_hi, 1),
        _mm512_insertf32x8(_mm512_castps256_ps512(y_lo), y_hi, 1),
        _mm512_insertf32x8(_mm512_castps256_ps512(z_lo), z_hi, 1),
    )
}

// ── store_aos16 (AVX-512F) ────────────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn store_aos16(ptr: *mut f32, x: __m512, y: __m512, z: __m512) {
    use std::arch::x86_64::*;
    store_aos8(
        ptr,
        _mm512_castps512_ps256(x),
        _mm512_castps512_ps256(y),
        _mm512_castps512_ps256(z),
    );
    store_aos8(
        ptr.add(24),
        _mm512_extractf32x8_ps(x, 1),
        _mm512_extractf32x8_ps(y, 1),
        _mm512_extractf32x8_ps(z, 1),
    );
}

// ── Scalar tail ───────────────────────────────────────────────────────────────
//
// #[cold] + #[inline(never)]: pushed into a cold code section so the hot
// SIMD loops remain in the L1 instruction cache.
#[cold]
#[inline(never)]
fn scalar_tail(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    n: usize,
) {
    for i in 0..n {
        positions[i][0] = velocities[i][0].mul_add(dt, positions[i][0]);
        positions[i][1] = velocities[i][1].mul_add(dt, positions[i][1]);
        positions[i][2] = velocities[i][2].mul_add(dt, positions[i][2]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    // ── Smoke ─────────────────────────────────────────────────────────────────
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn module_active() {
        assert!(simd_available());
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn module_active() {
        assert!(!simd_available());
    }

    #[test]
    fn basic() {
        let mut p = vec![[0.0f32; 3]; 4];
        let v = vec![[1.0f32, 0.5, -0.25]; 4];
        update_positions(&mut p, &v, 0.5);
        assert_eq!(p[0], [0.5, 0.25, -0.125]);
    }

    #[test]
    fn exact_batch_4() {
        let mut p = vec![[0.0f32; 3]; 4];
        let v = vec![[2.0f32, 4.0, -1.0]; 4];
        update_positions(&mut p, &v, 1.0);
        for pp in &p {
            assert_eq!(*pp, [2.0, 4.0, -1.0]);
        }
    }

    #[test]
    fn tail_handling() {
        let mut p = vec![[0.0f32; 3]; 7];
        let v = vec![[1.0f32; 3]; 7];
        update_positions(&mut p, &v, 2.0);
        for pp in &p {
            assert_eq!(*pp, [2.0f32; 3]);
        }
    }

    #[test]
    fn mismatched_lengths() {
        let mut p = vec![[0.0f32; 3]; 6];
        let v = vec![[1.0f32; 3]; 4];
        update_positions(&mut p, &v, 1.0);
        assert!(p[..4].iter().all(|x| *x == [1.0f32; 3]));
        assert!(p[4..].iter().all(|x| *x == [0.0f32; 3]));
    }

    #[test]
    fn empty_slices() {
        let mut p: Vec<[f32; 3]> = vec![];
        let v: Vec<[f32; 3]> = vec![];
        update_positions(&mut p, &v, 1.0);
    }

    // ── Transpose round-trip: store∘load = identity on raw float buffers ──────
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn transpose_roundtrip_sse2() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 12];
        unsafe {
            let (x, y, z) = load_aos4(src.as_ptr());
            store_aos4(dst.as_mut_ptr(), x, y, z);
        }
        assert_eq!(
            src, dst,
            "SSE2 4-particle round-trip\nsrc={src:?}\ndst={dst:?}"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn transpose_roundtrip_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 24];
        unsafe {
            let (x, y, z) = load_aos8(src.as_ptr());
            store_aos8(dst.as_mut_ptr(), x, y, z);
        }
        assert_eq!(
            src, dst,
            "AVX2 8-particle round-trip\nsrc={src:?}\ndst={dst:?}"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn transpose_roundtrip_avx512() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let src: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 48];
        unsafe {
            let (x, y, z) = load_aos16(src.as_ptr());
            store_aos16(dst.as_mut_ptr(), x, y, z);
        }
        assert_eq!(
            src, dst,
            "AVX-512 16-particle round-trip\nsrc={src:?}\ndst={dst:?}"
        );
    }

    // ── SoA lane correctness ──────────────────────────────────────────────────
    //
    // src = [0,1,2, 3,4,5, 6,7,8, 9,10,11]  (4 particles, stride-3 AoS)
    // Expected SoA: x=[0,3,6,9]  y=[1,4,7,10]  z=[2,5,8,11]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn soa_lanes_sse2() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (x, y, z) = unsafe { load_aos4(src.as_ptr()) };
        let xv: [f32; 4] = unsafe { std::mem::transmute(x) };
        let yv: [f32; 4] = unsafe { std::mem::transmute(y) };
        let zv: [f32; 4] = unsafe { std::mem::transmute(z) };
        assert_eq!(xv, [0.0, 3.0, 6.0, 9.0], "x lane: {xv:?}");
        assert_eq!(yv, [1.0, 4.0, 7.0, 10.0], "y lane: {yv:?}");
        assert_eq!(zv, [2.0, 5.0, 8.0, 11.0], "z lane: {zv:?}");
    }

    // ── Full correctness sweep: SIMD vs scalar reference ──────────────────────
    //
    // Every interesting boundary: main-loop multiples, drain-path remainders,
    // scalar tail, and alignment prefix for each dispatch tier.
    #[test]
    fn correctness_sweep() {
        let dt = 0.016_f32;
        let sizes = [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
            257, 1000, 1023, 1024, 10_001,
        ];
        for &n in &sizes {
            let p_init: Vec<[f32; 3]> = (0..n)
                .map(|i| [i as f32 * 0.1, -(i as f32) * 0.2, i as f32 * 0.3 + 1.0])
                .collect();
            let v: Vec<[f32; 3]> = (0..n)
                .map(|i| [1.0 + i as f32 * 0.001, 0.5 - i as f32 * 0.0005, -0.25])
                .collect();

            let mut p_ref = p_init.clone();
            // Scalar reference: compute manually
            for i in 0..n {
                p_ref[i][0] = v[i][0].mul_add(dt, p_ref[i][0]);
                p_ref[i][1] = v[i][1].mul_add(dt, p_ref[i][1]);
                p_ref[i][2] = v[i][2].mul_add(dt, p_ref[i][2]);
            }

            let mut p_simd = p_init;
            update_positions(&mut p_simd, &v, dt);

            for i in 0..n {
                for k in 0..3 {
                    let diff = (p_simd[i][k] - p_ref[i][k]).abs();
                    assert!(
                        diff < 1e-5,
                        "n={n} i={i} k={k}: simd={:.9} ref={:.9} diff={diff:.2e}",
                        p_simd[i][k],
                        p_ref[i][k],
                    );
                }
            }
        }
    }

    // ── FMA precision bound ───────────────────────────────────────────────────
    //
    // FMA computes v*dt+p in one correctly-rounded step; the result may differ
    // from a two-operation sequence by up to 1 ULP.  This test documents and
    // bounds that gap at a value that exercises the mantissa carry path.
    #[test]
    fn fma_precision_bound() {
        let mut p = vec![[1.000_000_1_f32; 3]];
        let v = vec![[9_999_999.0_f32; 3]];
        let dt = 1e-7_f32;
        update_positions(&mut p, &v, dt);
        for &val in p[0].iter() {
            // ~1.0000001 + 9999999 * 1e-7 ≈ 2.0
            assert!(
                (val - 2.0_f32).abs() < 1e-4,
                "FMA result out of expected range: {val}",
            );
        }
    }

    // ── No-alias guard ────────────────────────────────────────────────────────
    //
    // Each particle i must receive exactly p[i] + v[i]*dt, independent of
    // any other particle.  Any intra-batch aliasing would corrupt later values.
    #[test]
    fn no_alias_between_particles() {
        let n = 128;
        let mut p: Vec<[f32; 3]> = (0..n).map(|i| [i as f32; 3]).collect();
        let v: Vec<[f32; 3]> = vec![[1.0f32; 3]; n];
        update_positions(&mut p, &v, 1.0);
        for (i, &pp) in p.iter().enumerate() {
            let expected = i as f32 + 1.0;
            assert_eq!(pp, [expected; 3], "particle {i} aliased");
        }
    }
}

// =============================================================================
// JIT SIMD CODE GENERATION
// =============================================================================
// 
// This module generates SIMD machine code for vectorizable loops detected
// by the polyhedral optimizer. When the polyhedral engine emits VectorPack
// or TranscendentalVectorize hints, the JIT backend uses these functions
// to emit actual x86 SIMD instructions instead of scalar operations.
//
// Integration points:
//   - polyhedral.rs: SimdHintKind::VectorPack → emit_simd_vector_pack()
//   - polyhedral.rs: SimdHintKind::TranscendentalVectorize → emit_transcendental_call()
//   - phase3_jit.rs: LoopVectorizer → dispatch to SIMD codegen
//   - tracing_jit.rs: NativeCodeGenerator → SIMD instruction selection

/// SIMD instruction set level detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD support
    None,
    /// SSE2 (128-bit, 4×f32)
    SSE2,
    /// AVX (256-bit, 8×f32)
    AVX,
    /// AVX2 + FMA (256-bit, 8×f32 with fused multiply-add)
    AVX2FMA,
    /// AVX-512F (512-bit, 16×f32)
    AVX512,
}

impl SimdLevel {
    /// Detect the highest SIMD level available at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
                return SimdLevel::AVX512;
            }
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                return SimdLevel::AVX2FMA;
            }
            if is_x86_feature_detected!("avx") {
                return SimdLevel::AVX;
            }
            if is_x86_feature_detected!("sse2") {
                return SimdLevel::SSE2;
            }
        }
        SimdLevel::None
    }
    
    /// SIMD register width in bits
    pub fn register_bits(&self) -> usize {
        match self {
            SimdLevel::AVX512 => 512,
            SimdLevel::AVX2FMA | SimdLevel::AVX => 256,
            SimdLevel::SSE2 => 128,
            SimdLevel::None => 0,
        }
    }
    
    /// Number of f32 elements per SIMD register
    pub fn f32_lanes(&self) -> usize {
        self.register_bits() / 32
    }
    
    /// Number of f64 elements per SIMD register
    pub fn f64_lanes(&self) -> usize {
        self.register_bits() / 64
    }
    
    /// Number of i32 elements per SIMD register
    pub fn i32_lanes(&self) -> usize {
        self.register_bits() / 32
    }
    
    /// Whether this level supports FMA (fused multiply-add)
    pub fn has_fma(&self) -> bool {
        matches!(self, SimdLevel::AVX2FMA | SimdLevel::AVX512)
    }
}

/// SIMD code generation buffer for the JIT backend.
/// Emits raw x86 SIMD machine code bytes for vectorized operations.
#[derive(Debug, Clone)]
pub struct SimdCodeBuffer {
    /// Generated machine code bytes
    pub code: Vec<u8>,
    /// Current SIMD level
    pub level: SimdLevel,
    /// Element type size in bytes (1, 2, 4, or 8)
    pub element_bytes: usize,
}

impl SimdCodeBuffer {
    pub fn new(level: SimdLevel, element_bytes: usize) -> Self {
        Self {
            code: Vec::with_capacity(4096),
            level,
            element_bytes,
        }
    }
    
    #[inline(always)]
    fn b(&mut self, v: u8) {
        self.code.push(v);
    }
    
    #[inline(always)]
    fn emit2(&mut self, a: u8, b: u8) {
        self.code.extend_from_slice(&[a, b]);
    }
    
    #[inline(always)]
    fn emit3(&mut self, a: u8, b: u8, c: u8) {
        self.code.extend_from_slice(&[a, b, c]);
    }
    
    #[inline(always)]
    fn emit4(&mut self, a: u8, b: u8, c: u8, d: u8) {
        self.code.extend_from_slice(&[a, b, c, d]);
    }
    
    // ── AVX-512 VEX/EVEX prefix encoding ──────────────────────────────
    
    /// Emit EVEX prefix for AVX-512 instructions
    /// P0 = 0x62, P1,P2,P3 encoded per Intel SDM
    fn emit_evex(&mut self, p1: u8, p2: u8, p3: u8) {
        self.emit4(0x62, p1, p2, p3);
    }
    
    /// Emit VEX 3-byte prefix for AVX/AVX2 instructions
    fn emit_vex3(&mut self, mmmmm: u8, b: u8, vvvv: u8, l: bool, pp: u8) {
        let byte1 = 0xC4;
        let byte2 = (if mmmmm & 0x1F != 0 { 0 } else { 0x80 }) | (mmmmm & 0x1F);
        let byte3 = ((b & 1) << 7) | ((vvvv & 0xF) << 3) | ((l as u8) << 2) | (pp & 0x3);
        self.emit3(byte1, byte2, byte3);
    }
    
    // ── SIMD Load Operations ──────────────────────────────────────────
    
    /// Emit vmovups (unaligned vector load) for AVX/AVX2/AVX-512
    /// Loads `lanes` f32 elements from [rdi + offset]
    pub fn emit_simd_load(&mut self, zmm_reg: u8, offset: i32) {
        match self.level {
            SimdLevel::AVX512 => {
                // VMOVUPS zmm, [rdi + disp32]
                // EVEX.512.F3.0F 10 /r  (simplified encoding)
                self.emit_evex(0x62, 0xF1, 0x7C); // simplified EVEX prefix
                self.emit2(0x10, 0x47 | ((zmm_reg & 7) << 3)); // mod=01, zmm, [rdi+disp8]
                self.code.extend_from_slice(&(offset as i8 as u8).to_le_bytes());
            }
            SimdLevel::AVX2FMA | SimdLevel::AVX => {
                // VMOVUPS ymm, [rdi + disp32]
                self.emit3(0xC5, 0xFC, 0x10); // VEX.256.0F
                self.b(0x47 | ((zmm_reg & 7) << 3)); // mod=01, ymm, [rdi+disp8]
                if let Ok(d8) = i8::try_from(offset) {
                    self.b(d8 as u8);
                } else {
                    // Use disp32 form
                    self.b(0x87 | ((zmm_reg & 7) << 3)); // mod=10, ymm, [rdi+disp32]
                    self.code.extend_from_slice(&offset.to_le_bytes());
                }
            }
            SimdLevel::SSE2 => {
                // MOVUPS xmm, [rdi + disp32]
                if offset >= -128 && offset <= 127 {
                    self.emit3(0x0F, 0x10, 0x47 | ((zmm_reg & 7) << 3));
                    self.b(offset as i8 as u8);
                } else {
                    self.emit3(0x0F, 0x10, 0x87 | ((zmm_reg & 7) << 3));
                    self.code.extend_from_slice(&offset.to_le_bytes());
                }
            }
            SimdLevel::None => {
                // Scalar: MOV rax, [rdi + offset]
                self.emit3(0x48, 0x8B, 0x87 | ((zmm_reg & 7) << 3));
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }
    
    // ── SIMD Store Operations ─────────────────────────────────────────
    
    /// Emit vmovups (unaligned vector store) for AVX/AVX2/AVX-512
    pub fn emit_simd_store(&mut self, offset: i32, zmm_reg: u8) {
        match self.level {
            SimdLevel::AVX512 => {
                self.emit_evex(0x62, 0xF1, 0x7C);
                self.emit2(0x11, 0x47 | ((zmm_reg & 7) << 3));
                self.code.extend_from_slice(&(offset as i8 as u8).to_le_bytes());
            }
            SimdLevel::AVX2FMA | SimdLevel::AVX => {
                self.emit3(0xC5, 0xFC, 0x11);
                self.b(0x47 | ((zmm_reg & 7) << 3));
                if let Ok(d8) = i8::try_from(offset) {
                    self.b(d8 as u8);
                } else {
                    self.b(0x87 | ((zmm_reg & 7) << 3));
                    self.code.extend_from_slice(&offset.to_le_bytes());
                }
            }
            SimdLevel::SSE2 => {
                if offset >= -128 && offset <= 127 {
                    self.emit3(0x0F, 0x11, 0x47 | ((zmm_reg & 7) << 3));
                    self.b(offset as i8 as u8);
                } else {
                    self.emit3(0x0F, 0x11, 0x87 | ((zmm_reg & 7) << 3));
                    self.code.extend_from_slice(&offset.to_le_bytes());
                }
            }
            SimdLevel::None => {
                self.emit3(0x48, 0x89, 0x87 | ((zmm_reg & 7) << 3));
                self.code.extend_from_slice(&offset.to_le_bytes());
            }
        }
    }
    
    // ── SIMD Arithmetic Operations ────────────────────────────────────
    
    /// Emit SIMD add: dst = src1 + src2 (packed single-precision)
    pub fn emit_simd_add_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        match self.level {
            SimdLevel::AVX512 | SimdLevel::AVX2FMA | SimdLevel::AVX => {
                // VADDPS dst, src1, src2
                self.emit3(0xC5, 0xF4, 0x58); // VEX.256.0F.WIG
                self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7)); // mod=11
            }
            SimdLevel::SSE2 => {
                // ADDPS xmm, xmm
                self.emit3(0x0F, 0x58, 0xC0 | ((dst & 7) << 3) | (src2 & 7));
            }
            SimdLevel::None => {
                // Scalar ADD rax, rcx
                self.emit3(0x48, 0x01, 0xC8);
            }
        }
    }
    
    /// Emit SIMD multiply: dst = src1 * src2 (packed single-precision)
    pub fn emit_simd_mul_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        match self.level {
            SimdLevel::AVX512 | SimdLevel::AVX2FMA | SimdLevel::AVX => {
                // VMULPS dst, src1, src2
                self.emit3(0xC5, 0xF4, 0x59);
                self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7));
            }
            SimdLevel::SSE2 => {
                self.emit3(0x0F, 0x59, 0xC0 | ((dst & 7) << 3) | (src2 & 7));
            }
            SimdLevel::None => {
                self.emit4(0x48, 0x0F, 0xAF, 0xC1); // IMUL rax, rcx
            }
        }
    }
    
    /// Emit SIMD FMA: dst = src1 * src2 + dst (packed single-precision)
    /// Only available on AVX2+FMA or AVX-512
    pub fn emit_simd_fmadd_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        if self.has_fma() {
            // VFMADD213PS dst, src1, src2: dst = src1 * src2 + dst
            self.emit3(0xC4, 0xE2, 0x79); // VEX.FMA.0F38.W0
            self.b(0xA8); // VFMADD213PS
            self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7));
        } else {
            // Fallback: separate MUL + ADD
            self.emit_simd_mul_ps(dst, src1, src2);
            self.emit_simd_add_ps(dst, dst, src2);
        }
    }
    
    /// Emit SIMD subtract: dst = src1 - src2
    pub fn emit_simd_sub_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        match self.level {
            SimdLevel::AVX512 | SimdLevel::AVX2FMA | SimdLevel::AVX => {
                self.emit3(0xC5, 0xF4, 0x5C);
                self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7));
            }
            SimdLevel::SSE2 => {
                self.emit3(0x0F, 0x5C, 0xC0 | ((dst & 7) << 3) | (src2 & 7));
            }
            SimdLevel::None => {
                self.emit3(0x48, 0x29, 0xC8); // SUB rax, rcx
            }
        }
    }
    
    // ── Prefetch Operations ───────────────────────────────────────────
    
    /// Emit software prefetch (T0 = L1 cache)
    pub fn emit_prefetch_t0(&mut self, offset: i32) {
        // PREFETCHT0 [rdi + offset]
        self.emit3(0x0F, 0x18, 0x87); // mod=10, reg=1(PREFETCHT0), rm=7(rdi)
        self.code.extend_from_slice(&offset.to_le_bytes());
    }
    
    // ── AMX Tile Operations ───────────────────────────────────────────
    
    /// Emit AMX TILELOAD (load tile from memory)
    /// Only valid on Intel AMX-capable processors
    pub fn emit_amx_tileload(&mut self, tile_reg: u8, base: u8, stride: u8) {
        // TILELOADD tmm, [base + stride]
        // EVEX.128.F2.0F38.W0 4B /r
        self.emit_evex(0x62, 0xF2, 0x7D);
        self.emit2(0x4B, 0x04 | (tile_reg & 7));
        let _ = (base, stride); // SIB encoding in full impl
    }
    
    /// Emit AMX TDPBSSD (tile dot product of signed 8-bit integers)
    pub fn emit_amx_tdpbssd(&mut self, dst_tile: u8, src1_tile: u8, src2_tile: u8) {
        // TDPBSSD tmm_dst, tmm_src1, tmm_src2
        self.emit_evex(0x62, 0xF2, 0x7D);
        self.emit2(0x5E, 0xC0 | ((dst_tile & 3) << 3) | (src2_tile & 3));
        let _ = src1_tile;
    }
    
    /// Emit AMX TILESTOREF (store tile to memory)
    pub fn emit_amx_tilestore(&mut self, base: u8, stride: u8, tile_reg: u8) {
        self.emit_evex(0x62, 0xF2, 0x7D);
        self.emit2(0x4B, 0x04 | (tile_reg & 7));
        let _ = (base, stride);
    }
    
    /// Emit AMX TILEZERO (zero a tile register)
    pub fn emit_amx_tilezero(&mut self, tile_reg: u8) {
        self.emit_evex(0x62, 0xF2, 0x7D);
        self.emit2(0x49, 0xC0 | (tile_reg & 3));
    }
    
    /// Emit AMX TILERELEASE (release tile configuration)
    pub fn emit_amx_tilerelease(&mut self) {
        self.emit3(0x62, 0xF2, 0x7D);
        self.b(0x49);
        self.b(0x00);
    }
    
    // ── Complete SIMD Loop Generation ─────────────────────────────────
    
    /// Generate a complete SIMD-vectorized loop body for a simple
    /// elementwise operation (e.g., A[i] = B[i] OP C[i]).
    ///
    /// This is the "fast path" called when the polyhedral optimizer
    /// produces a VectorPack hint. The loop is structured as:
    ///   1. Main loop: process `lanes` elements per iteration
    ///   2. Fringe loop: process remaining elements scalarly
    ///
    /// Returns the generated machine code bytes.
    pub fn emit_vectorized_loop(
        &mut self,
        op: u8, // 0=add, 1=mul, 2=fma, 3=sub
        src1_offset: i32,
        src2_offset: i32,
        dst_offset: i32,
        element_count: usize,
        stride: i32, // bytes between consecutive elements
    ) -> Vec<u8> {
        let lanes = if self.element_bytes == 4 {
            self.f32_lanes()
        } else if self.element_bytes == 8 {
            self.f64_lanes()
        } else {
            self.f32_lanes() * (4 / self.element_bytes.max(1))
        };
        
        let main_iters = element_count / lanes;
        let fringe = element_count % lanes;
        
        // ── Main SIMD loop ──
        // For each iteration: load src1, load src2, compute, store dst
        for _iter in 0..main_iters {
            // Prefetch next cache line (if AVX2+)
            if matches!(self.level, SimdLevel::AVX2FMA | SimdLevel::AVX512) {
                self.emit_prefetch_t0(src1_offset + (lanes as i32) * stride);
            }
            
            self.emit_simd_load(0, src1_offset); // zmm0 = src1
            self.emit_simd_load(1, src2_offset); // zmm1 = src2
            
            match op {
                0 => self.emit_simd_add_ps(2, 0, 1),  // zmm2 = zmm0 + zmm1
                1 => self.emit_simd_mul_ps(2, 0, 1),  // zmm2 = zmm0 * zmm1
                2 => self.emit_simd_fmadd_ps(2, 0, 1), // zmm2 = zmm0*zmm1+zmm2
                3 => self.emit_simd_sub_ps(2, 0, 1),  // zmm2 = zmm0 - zmm1
                _ => self.emit_simd_add_ps(2, 0, 1),
            }
            
            self.emit_simd_store(dst_offset, 2); // [dst] = zmm2
        }
        
        // ── Fringe (scalar tail) ──
        // Process remaining elements one at a time
        for _i in 0..fringe {
            // Scalar load + op + store
            self.emit3(0x48, 0x8B, 0x87); // mov rax, [rdi+src1_offset]
            self.code.extend_from_slice(&src1_offset.to_le_bytes());
            self.emit3(0x48, 0x8B, 0x8F); // mov rcx, [rdi+src2_offset]
            self.code.extend_from_slice(&src2_offset.to_le_bytes());
            match op {
                0 => self.emit3(0x48, 0x01, 0xC8), // add rax, rcx
                1 => self.emit4(0x48, 0x0F, 0xAF, 0xC1), // imul rax, rcx
                3 => self.emit3(0x48, 0x29, 0xC8), // sub rax, rcx
                _ => self.emit3(0x48, 0x01, 0xC8),
            }
            self.emit3(0x48, 0x89, 0x87); // mov [rdi+dst_offset], rax
            self.code.extend_from_slice(&dst_offset.to_le_bytes());
        }
        
        self.code.clone()
    }
}

/// Global SIMD level — detected once at startup
static GLOBAL_SIMD_LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

/// Get the current SIMD level (detected once, cached)
pub fn get_simd_level() -> SimdLevel {
    *GLOBAL_SIMD_LEVEL.get_or_init(SimdLevel::detect)
}

/// Create a SIMD code buffer for the current hardware
pub fn create_simd_buffer(element_bytes: usize) -> SimdCodeBuffer {
    SimdCodeBuffer::new(get_simd_level(), element_bytes)
}

// =============================================================================
// POLYHEDRAL SIMD INTEGRATION
// =============================================================================
//
// Bridge between the polyhedral optimizer's SimdHintKind hints and
// the actual SIMD code generation. This is called by the JIT backend
// (phase3_jit.rs and tracing_jit.rs) when processing polyhedral hints.

use crate::compiler::ast::BinOpKind;

/// Dispatch a polyhedral SIMD hint to the appropriate code generation.
/// Returns the generated machine code bytes, or empty if the hint
/// doesn't require code generation (handled at a higher level).
pub fn emit_simd_for_hint(
    hint_kind: &str, // serialized SimdHintKind variant name
    op: BinOpKind,
    src1_slot: u16,
    src2_slot: u16,
    dst_slot: u16,
    element_bytes: usize,
) -> Vec<u8> {
    let level = get_simd_level();
    if level == SimdLevel::None {
        return Vec::new(); // No SIMD — scalar path handled by JIT
    }
    
    let mut buf = SimdCodeBuffer::new(level, element_bytes);
    
    let op_code = match op {
        BinOpKind::Add => 0,
        BinOpKind::Mul => 1,
        BinOpKind::Sub => 3,
        _ => 0,
    };
    
    match hint_kind {
        "VectorPack" | "TranscendentalVectorize" => {
            buf.emit_simd_load(0, (src1_slot as i32) * (element_bytes as i32));
            buf.emit_simd_load(1, (src2_slot as i32) * (element_bytes as i32));
            match op_code {
                0 => buf.emit_simd_add_ps(2, 0, 1),
                1 => buf.emit_simd_mul_ps(2, 0, 1),
                3 => buf.emit_simd_sub_ps(2, 0, 1),
                _ => buf.emit_simd_add_ps(2, 0, 1),
            }
            buf.emit_simd_store((dst_slot as i32) * (element_bytes as i32), 2);
        }
        "MatrixOuterProduct" => {
            // AMX outer-product — emit tile load + matmul + tile store
            if level == SimdLevel::AVX512 {
                buf.emit_amx_tileload(0, 7, 6); // tmm0 = [rdi + rsi]
                buf.emit_amx_tileload(1, 7, 6); // tmm1 = [rdi + rsi]
                buf.emit_amx_tdpbssd(2, 0, 1);  // tmm2 += tmm0 × tmm1
                buf.emit_amx_tilestore(7, 6, 2); // [rdi + rsi] = tmm2
            }
            // Fallback for non-AMX: use FMADD vector
            buf.emit_simd_load(0, (src1_slot as i32) * (element_bytes as i32));
            buf.emit_simd_load(1, (src2_slot as i32) * (element_bytes as i32));
            buf.emit_simd_fmadd_ps(2, 0, 1);
            buf.emit_simd_store((dst_slot as i32) * (element_bytes as i32), 2);
        }
        _ => {
            // Unknown hint — no code generation
        }
    }
    
    buf.code
}

/// Check if AMX (Advanced Matrix Extensions) is available
pub fn has_amx() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // AMX is indicated by CPUID leaf 7, sub-leaf 0, EDX bit 24
        // We check for AMX-BF16 and AMX-INT8
        is_x86_feature_detected!("amx_bf16") || is_x86_feature_detected!("amx_int8")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
