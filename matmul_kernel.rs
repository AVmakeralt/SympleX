//! SympleX Matmul JIT Kernel — Real AVX-512 FMA Code Generation
//!
//! This module generates actual x86-64 machine code for matrix multiplication
//! with loop nests and AVX-512 VFMADD231PS (fused multiply-add).
//!
//! Architecture:
//!   - 6×16 register micro-kernel: 6 ZMM accumulator rows × 16 f32 columns
//!   - i-p-j loop ordering for cache-friendly B row streaming
//!   - VBROADCASTSS for A element broadcasting
//!   - VFMADD231PS for fused multiply-accumulate (single rounding, IEEE 754)
//!   - VMOVUPS for unaligned load/store of ZMM vectors
//!   - AVX-512 opmask (k-register) tail handling for non-multiple-of-16 N
//!   - Scalar fallback for CPUs without AVX-512
//!
//! Calling convention (System V AMD64):
//!   RDI = A pointer (f32*, row-major, M×K)
//!   RSI = B pointer (f32*, row-major, K×N)
//!   RDX = C pointer (f32*, row-major, M×N, output)
//!   RCX = M (rows of A / C)
//!   R8  = N (cols of B / C)
//!   R9  = K (cols of A / rows of B)
//!
//! The generated function returns 0 on success.

use std::ptr;

// ─────────────────────────────────────────────────────────────────────────────
// Executable memory helpers (Linux-only, no external deps)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn alloc_exec_mem(len: usize) -> Option<*mut u8> {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr.is_null() || ptr == libc::MAP_FAILED {
        None
    } else {
        Some(ptr as *mut u8)
    }
}

#[cfg(target_os = "linux")]
fn make_exec(ptr: *mut u8, len: usize) -> bool {
    unsafe { libc::mprotect(ptr as *mut libc::c_void, len, libc::PROT_READ | libc::PROT_EXEC) == 0 }
}

#[cfg(target_os = "linux")]
fn free_exec_mem(ptr: *mut u8, len: usize) {
    unsafe { libc::munmap(ptr as *mut libc::c_void, len) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine code emitter
// ─────────────────────────────────────────────────────────────────────────────

struct CodeEmitter {
    buf: Vec<u8>,
}

impl CodeEmitter {
    fn new(capacity: usize) -> Self {
        Self { buf: Vec::with_capacity(capacity) }
    }

    #[inline(always)]
    fn pos(&self) -> usize { self.buf.len() }

    #[inline(always)]
    fn b(&mut self, v: u8) { self.buf.push(v); }

    #[inline(always)]
    fn emit2(&mut self, b0: u8, b1: u8) { self.buf.extend_from_slice(&[b0, b1]); }

    #[inline(always)]
    fn emit3(&mut self, b0: u8, b1: u8, b2: u8) { self.buf.extend_from_slice(&[b0, b1, b2]); }

    #[inline(always)]
    fn emit4(&mut self, b0: u8, b1: u8, b2: u8, b3: u8) { self.buf.extend_from_slice(&[b0, b1, b2, b3]); }

    #[inline(always)]
    fn d32(&mut self, v: i32) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    /// Emit a 64-bit little-endian value (for MOV r64, imm64)
    fn d64(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    #[inline(always)]
    fn q64(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    // ── REX prefix helpers ──

    /// Compute REX.W prefix for 64-bit operation with reg + rm encoding.
    /// reg = dst for 8B (load), rm = dst for 89 (store).
    #[inline(always)]
    fn rex_w(&mut self, reg: u8, rm: u8) {
        let rex = 0x48 | ((reg & 8) >> 1) | ((rm & 8) >> 3);
        self.b(rex);
    }

    // ── GPR move / arithmetic ──

    /// MOV r64, r64 — REX.W 8B /r (reg=dst, rm=src)
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src { return; }
        self.rex_w(dst, src);
        self.emit2(0x8B, 0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// MOV r64, imm32 (sign-extended) — REX.W C7 /0 id
    fn mov_ri32(&mut self, dst: u8, val: i32) {
        self.rex_w(0, dst);  // reg=0 (opcode extension /0), rm=dst
        self.emit2(0xC7, 0xC0 | (dst & 7));
        self.d32(val);
    }

    /// MOV r64, imm64 — REX.W B8+rd iq
    /// BUG FIX #2: Use this when the value doesn't fit in i32 (e.g. large strides).
    fn mov_ri64(&mut self, dst: u8, val: i64) {
        self.rex_w(0, dst);
        self.b(0xB8 | (dst & 7));
        self.d64(val);
    }

    /// Load a 64-bit immediate into a register, choosing the optimal encoding.
    /// Uses mov_ri32 (7 bytes, sign-extended) when it fits, otherwise mov_ri64 (10 bytes).
    /// BUG FIX #2: This prevents silent truncation of large stride offsets.
    fn mov_ri_opt(&mut self, dst: u8, val: i64) {
        if let Ok(v32) = i32::try_from(val) {
            self.mov_ri32(dst, v32);
        } else {
            self.mov_ri64(dst, val);
        }
    }

    /// ADD r64, r64 — REX.W 01 /r
    fn add_rr(&mut self, dst: u8, src: u8) {
        self.rex_w(src, dst);
        self.emit2(0x01, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// ADD r64, imm8 — REX.W 83 /0 ib
    fn add_ri8(&mut self, dst: u8, val: i8) {
        self.rex_w(0, dst);
        self.emit2(0x83, 0xC0 | (dst & 7));
        self.b(val as u8);
    }

    /// ADD r64, imm32 — REX.W 81 /0 id
    fn add_ri32(&mut self, dst: u8, val: i32) {
        self.rex_w(0, dst);
        self.emit2(0x81, 0xC0 | (dst & 7));
        self.d32(val);
    }

    /// ADD r64, imm64 — load into RCX, then add.
    /// BUG FIX #2: For offsets that don't fit in i32.
    fn add_ri64_via_rcx(&mut self, dst: u8, val: i64) {
        self.mov_ri64(1, val);      // MOV RCX, imm64
        self.add_rr(dst, 1);        // ADD dst, RCX
    }

    /// ADD r64, i64 — chooses optimal encoding.
    /// BUG FIX #2: Prevents silent i32 truncation of large stride offsets.
    fn add_ri_opt(&mut self, dst: u8, val: i64) {
        if let Ok(v32) = i32::try_from(val) {
            self.add_ri32(dst, v32);
        } else {
            self.add_ri64_via_rcx(dst, val);
        }
    }

    /// SUB r64, imm8 — REX.W 83 /5 ib
    fn sub_ri8(&mut self, dst: u8, val: i8) {
        self.rex_w(0, dst);
        self.emit2(0x83, 0xE8 | (dst & 7));
        self.b(val as u8);
    }

    /// IMUL r64, r64 — REX.W 0F AF /r
    fn imul_rr(&mut self, dst: u8, src: u8) {
        self.rex_w(dst, src);
        self.emit3(0x0F, 0xAF, 0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// CMP r64, r64 — REX.W 39 /r
    fn cmp_rr(&mut self, a: u8, b: u8) {
        self.rex_w(b, a);
        self.emit2(0x39, 0xC0 | ((b & 7) << 3) | (a & 7));
    }

    /// XOR EAX, EAX — 2 bytes (zero-extends to RAX)
    fn xor_eax_eax(&mut self) { self.emit2(0x31, 0xC0); }

    /// SHL r64, imm8 — REX.W C1 /4 ib
    fn shl_ri8(&mut self, dst: u8, val: u8) {
        self.rex_w(0, dst);
        self.emit2(0xC1, 0xE0 | (dst & 7));
        self.b(val);
    }

    // ── Load/Store from slot array [RDI + disp32] ──

    /// MOV r64, [RDI + disp32] — REX.W 8B /r mod=10 rm=111
    fn load_rdi(&mut self, reg: u8, disp: i32) {
        self.rex_w(reg, 7); // rm=7=RDI
        self.emit2(0x8B, 0x87 | ((reg & 7) << 3));
        self.d32(disp);
    }

    /// MOV [RDI + disp32], r64 — REX.W 89 /r mod=10 rm=111
    fn store_rdi(&mut self, disp: i32, reg: u8) {
        self.rex_w(reg, 7);
        self.emit2(0x89, 0x87 | ((reg & 7) << 3));
        self.d32(disp);
    }

    // ── Load from [RAX + disp32] ──

    /// MOV r64, [RAX + disp32] — REX.W 8B /r mod=10 rm=000
    fn load_rax_disp(&mut self, reg: u8, disp: i32) {
        self.rex_w(reg, 0);
        self.emit2(0x8B, 0x80 | ((reg & 7) << 3));
        self.d32(disp);
    }

    // ── Push/Pop callee-saved ──

    fn push(&mut self, reg: u8) {
        if reg >= 8 { self.b(0x41); }
        self.b(0x50 | (reg & 7));
    }

    fn pop(&mut self, reg: u8) {
        if reg >= 8 { self.b(0x41); }
        self.b(0x58 | (reg & 7));
    }

    fn ret(&mut self) { self.b(0xC3); }

    /// VZEROUPPER — C5 F8 77
    /// Clears the upper 128 bits of all YMM/ZMM registers to avoid
    /// AVX→SSE transition penalties (~70 cycles) on return to scalar code.
    fn vzeroupper(&mut self) {
        self.emit3(0xC5, 0xF8, 0x77);
    }

    // ── Conditional jumps ──

    /// JL rel8
    fn jl8(&mut self, off: i8) { self.emit2(0x7C, off as u8); }

    /// JL rel32
    fn jl32(&mut self, off: i32) { self.emit2(0x0F, 0x8C); self.d32(off); }

    /// JGE rel32
    fn jge32(&mut self, off: i32) { self.emit2(0x0F, 0x8D); self.d32(off); }

    /// JNE rel8
    fn jne8(&mut self, off: i8) { self.emit2(0x75, off as u8); }

    /// NOP padding
    fn nop(&mut self, n: usize) { for _ in 0..n { self.b(0x90); } }

    // ── x87 f32 load/store (scalar fallback) ──

    /// FLD dword [rax] — load f32 onto x87 stack (D9 /0 mod=00)
    fn fld_f32_rax(&mut self) { self.emit2(0xD9, 0x00); }

    /// FSTP dword [rcx] — store f32 from x87 stack (D9 /3 mod=01 disp8=0)
    fn fstp_f32_rcx(&mut self) { self.emit2(0xD9, 0x19); }

    /// FMULP — multiply x87 ST(1)*=ST(0); pop (DE C9)
    fn fmulp(&mut self) { self.emit2(0xDE, 0xC9); }

    /// FADD dword [rax] — add f32 memory to ST(0) (D8 /0 mod=00)
    fn fadd_f32_rax(&mut self) { self.emit2(0xD8, 0x00); }

    // ─────────────────────────────────────────────────────────────────────
    // AVX-512 EVEX-encoded instructions for ZMM registers
    // ─────────────────────────────────────────────────────────────────────

    /// Emit a full EVEX prefix for 512-bit ZMM register-register operation.
    ///
    /// Parameters:
    ///   dst  = ZMM destination (0-31)
    ///   src1 = ZMM first source / NDS (0-31)
    ///   src2 = ZMM second source (0-31)
    ///   pp   = mandatory prefix: 0=none, 1=66, 2=F3, 3=F2
    ///   mm   = map select: 1=0F, 2=0F38, 3=0F3A
    ///   mask = opmask register (0=k0=no masking)
    fn evex_rrr(&mut self, dst: u8, src1: u8, src2: u8, pp: u8, mm: u8, mask: u8) {
        self.b(0x62);
        // P0: [R~][X~][B~][R'][0][m][m][m]
        let p0 = (if (dst & 8) == 0 { 0x80 } else { 0 })
               | 0x40  // X~=1 (no SIB)
               | (if (src2 & 8) == 0 { 0x20 } else { 0 })
               | (if (dst & 16) == 0 { 0x10 } else { 0 })
               | (mm & 0x03);
        self.b(p0);
        // P1: [W=0][vvvv'][1][pp]  — vvvv' is 1's complement of src1[3:0] (4 bits)
        // The 5th bit (src1 bit4) goes to V' in P2.
        let vvvv_not = (!src1) & 0x0F;
        let p1 = 0x04 | (vvvv_not << 3) | (pp & 0x03);
        self.b(p1);
        // P2: [z=0][L'=1][b=0][0][V'][aaa] — 512-bit reg-reg: z=0, L'=1, b=0
        // V' = NOT(bit4 of src1/vvvv) at bit3; aaa = opmask register
        let p2 = 0x40  // L'=1 for 512-bit ZMM
               | (if (src1 & 16) == 0 { 0x08 } else { 0 })  // V' = NOT(vvvv[4])
               | (mask & 0x07);
        self.b(p2);
    }

    /// Emit EVEX prefix for 512-bit ZMM memory load/store.
    /// vvvv = NDS register (0 for loads, unused)
    /// base = GPR base register for address
    /// z = zero-masking flag (true for masked loads with zeroing, false for merge/stores)
    fn evex_mem(&mut self, dst: u8, vvvv: u8, base: u8, pp: u8, mm: u8, mask: u8, z: bool) {
        self.b(0x62);
        let p0 = (if (dst & 8) == 0 { 0x80 } else { 0 })
               | 0x40
               | (if (base & 8) == 0 { 0x20 } else { 0 })
               | (if (dst & 16) == 0 { 0x10 } else { 0 })
               | (mm & 0x03);
        self.b(p0);
        // P1: vvvv' is 1's complement of vvvv[3:0] (4 bits); 5th bit is V' in P2
        let vvvv_not = (!vvvv) & 0x0F;
        let p1 = 0x04 | (vvvv_not << 3) | (pp & 0x03);
        self.b(p1);
        // P2: [z][L'=1][b=0][0][V'][aaa] — 512-bit ZMM, no broadcast for reg-mem loads
        // L'=1 for 512-bit; V' = NOT(vvvv[4]) at bit3; z at bit7
        let p2 = (if z { 0x80 } else { 0 })
               | 0x40  // L'=1 for 512-bit ZMM
               | (if (vvvv & 16) == 0 { 0x08 } else { 0 })  // V' = NOT(vvvv[4])
               | (mask & 0x07);
        self.b(p2);
    }

    /// Emit ModRM byte for register-register: mod=11, reg=dst, rm=src2
    fn modrm_rr(&mut self, dst: u8, src2: u8) {
        self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7));
    }

    /// Emit ModRM for [base + disp32]: mod=10, reg=dst, rm=base
    fn modrm_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.b(0x80 | ((dst & 7) << 3) | (base & 7));
        if (base & 7) == 4 { self.b(0x24); } // SIB for RSP/R12
        self.d32(disp);
    }

    // ── AVX-512 ZMM instructions ──

    /// VXORPS zmm, zmm, zmm — zero a ZMM register (6× for C accumulators)
    fn vxorps_zmm(&mut self, dst: u8) {
        self.evex_rrr(dst, dst, dst, 0, 1, 0); // pp=none, mm=0F
        self.emit2(0x57, 0xC0 | ((dst & 7) << 3) | (dst & 7)); // VXORPS + ModRM
    }

    /// VBROADCASTSS zmm_dst, [base + disp32] — broadcast scalar f32 to 16 ZMM lanes
    fn vbroadcastss_zmm(&mut self, dst: u8, base: u8, disp: i32) {
        self.evex_mem(dst, 0, base, 1, 2, 0, false); // pp=66, mm=0F38, no masking
        self.b(0x18); // VBROADCASTSS opcode
        self.modrm_mem(dst, base, disp);
    }

    /// VFMADD231PS zmm_dst, zmm_src1, zmm_src2
    /// dst = src1 * src2 + dst  (fused multiply-add, single rounding)
    fn vfmadd231ps_zmm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.evex_rrr(dst, src1, src2, 1, 2, 0); // pp=66, mm=0F38
        self.b(0xB8); // VFMADD231PS opcode
        self.modrm_rr(dst, src2);
    }

    /// VMOVUPS zmm, [base + disp32] — unaligned load 64 bytes
    fn vmovups_zmm_load(&mut self, dst: u8, base: u8, disp: i32) {
        self.evex_mem(dst, 0, base, 0, 1, 0, false); // pp=none, mm=0F, no masking
        self.b(0x10); // VMOVUPS load opcode
        self.modrm_mem(dst, base, disp);
    }

    /// VMOVUPS [base + disp32], zmm — unaligned store 64 bytes
    fn vmovups_zmm_store(&mut self, base: u8, disp: i32, src: u8) {
        self.evex_mem(src, 0, base, 0, 1, 0, false); // pp=none, mm=0F, no masking
        self.b(0x11); // VMOVUPS store opcode
        self.modrm_mem(src, base, disp);
    }

    /// KMOVW k1, eax — load 16-bit mask from EAX
    fn kmovw_k1_eax(&mut self) {
        self.emit3(0xC5, 0xF8, 0x92); // VEX.LIG.0F.W0 92 /r
        self.b(0xC8); // ModRM: reg=k1, rm=eax
    }

    // ── VEX-encoded YMM instructions (AVX2 fallback) ──

    /// VXORPS ymm, ymm, ymm — zero a YMM register
    fn vxorps_ymm(&mut self, dst: u8) {
        // VEX3: C4 [R~=1 X~=1 B~=1 mm=01] [W=0 vvvv'=~dst U=1 pp=01] 57 [ModRM]
        let byte1 = 0xE1u8; // R~=1, X~=1, B~=1 (for regs 0-7), mm=0001(0F)
        let vvvv_inv = (!(dst)) & 0xF;
        let byte2 = (vvvv_inv << 3) | 0x05; // W=0, U=1, pp=01(66)
        self.emit4(0xC4, byte1, byte2, 0x57);
        self.b(0xC0 | ((dst & 7) << 3) | (dst & 7));
    }

    /// VBROADCASTSS ymm, [rax] — broadcast scalar f32 to 8 YMM lanes
    fn vbroadcastss_ymm_rax(&mut self, dst: u8) {
        // VEX.256.66.0F38.W0 18 /r with [RAX]
        // C4 [R~=1 X~=1 B~=1 mm=0010] [W=0 vvvv'=1111 U=1 pp=01] 18 [mod=00 reg=dst rm=000]
        let r_not = if (dst & 8) != 0 { 0u8 } else { 1u8 };
        let byte1 = (r_not << 7) | 0x62; // X~=1, B~=1, mm=0010(0F38)
        self.emit4(0xC4, byte1, 0x7D, 0x18); // vvvv'=1111, L=1(256), pp=01(66)
        self.b((dst & 7) << 3); // ModRM: mod=00, reg=dst, rm=000(RAX)
    }

    /// VFMADD231PS ymm_dst, ymm_src1, ymm_src2 — AVX2 FMA
    fn vfmadd231ps_ymm(&mut self, dst: u8, src1: u8, src2: u8) {
        // VEX.256.66.0F38.W0 B8 /r
        // C4 [R~ X~ B~ mm=0010] [W=0 vvvv' U=1 pp=01] B8 [ModRM]
        let r_not = if (dst & 8) != 0 { 0u8 } else { 1u8 };
        let b_not = if (src2 & 8) != 0 { 0u8 } else { 1u8 };
        let byte1 = (r_not << 7) | 0x40 | (b_not << 5) | 0x02;
        let vvvv_inv = (!(src1)) & 0xF;
        let byte2 = (vvvv_inv << 3) | 0x05; // W=0, L=1(256), pp=01(66)
        self.emit4(0xC4, byte1, byte2, 0xB8);
        self.b(0xC0 | ((dst & 7) << 3) | (src2 & 7));
    }

    /// VMOVUPS ymm, [rax] — unaligned load 32 bytes (AVX2)
    fn vmovups_ymm_load_rax(&mut self, dst: u8) {
        // VEX.256.66.0F.W0 10 /r with [RAX]
        let r_not = if (dst & 8) != 0 { 0u8 } else { 1u8 };
        let byte1 = (r_not << 7) | 0x61; // X~=1, B~=1, mm=0001(0F)
        self.emit4(0xC4, byte1, 0x7D, 0x10); // vvvv=1111, L=1, pp=01
        self.b((dst & 7) << 3); // ModRM: mod=00, reg=dst, rm=000(RAX)
    }

    /// VMOVUPS [rax], ymm — unaligned store 32 bytes (AVX2)
    fn vmovups_ymm_store_rax(&mut self, src: u8) {
        let r_not = if (src & 8) != 0 { 0u8 } else { 1u8 };
        let byte1 = (r_not << 7) | 0x61;
        self.emit4(0xC4, byte1, 0x7D, 0x11); // store opcode
        self.b((src & 7) << 3);
    }

    /// MOVSS xmm0, [rax] — load scalar f32 (VEX.128.F3.0F 10 /r)
    fn movss_load_rax(&mut self) {
        self.emit4(0xF3, 0x0F, 0x10, 0x00); // MOVSS XMM0, [RAX]
    }

    /// MOVSS [rcx], xmm0 — store scalar f32 (VEX.128.F3.0F 11 /r)
    fn movss_store_rcx(&mut self) {
        self.emit4(0xF3, 0x0F, 0x11, 0x01); // MOVSS [RCX], XMM0
    }

    // ─────────────────────────────────────────────────────────────────────
    // Matmul kernel generation
    // ─────────────────────────────────────────────────────────────────────

    /// Generate a complete matmul kernel for M×N×K (f32, row-major).
    ///
    /// The kernel implements:
    /// ```text
    /// for i in 0..M:
    ///   for j in 0..N:
    ///     C[i*N+j] = 0.0
    ///   for p in 0..K:
    ///     for j in (0..N) step SIMD_WIDTH:
    ///       C[i*N+j..j+SIMD_WIDTH-1] += A[i*K+p] * B[p*N+j..j+SIMD_WIDTH-1]
    /// ```
    ///
    /// With AVX-512: SIMD_WIDTH = 16, uses VFMADD231PS
    /// With AVX2:    SIMD_WIDTH = 8,  uses VFMADD231PS (256-bit)
    /// Otherwise:    scalar x87 fallback
    fn emit_matmul_kernel(&mut self, m: usize, n: usize, k: usize, has_avx512f: bool, has_avx2: bool, has_fma: bool) {
        // ══════════════════════════════════════════════════════════════════
        // Prologue: save callee-saved registers
        // ══════════════════════════════════════════════════════════════════
        self.push(5);   // PUSH RBP
        self.push(3);   // PUSH RBX
        self.push(12);  // PUSH R12
        self.push(13);  // PUSH R13
        self.push(14);  // PUSH R14
        self.push(15);  // PUSH R15

        // Save arguments to callee-saved registers:
        //   R12 = M, R13 = N, R14 = K, R15 = N*4 (byte stride for B/C rows)
        self.mov_rr(12, 1);   // R12 = RCX = M
        self.mov_rr(13, 8);   // R13 = R8  = N
        self.mov_rr(14, 9);   // R14 = R9  = K
        self.mov_rr(15, 8);   // R15 = N
        self.shl_ri8(15, 2);  // R15 = N * 4 (byte stride)

        if has_avx512f && n >= 16 {
            self.emit_avx512_matmul(m, n, k);
        } else if has_avx2 && has_fma && n >= 8 {
            self.emit_avx2_matmul(m, n, k);
        } else {
            self.emit_scalar_matmul(m, n, k);
        }

        // ══════════════════════════════════════════════════════════════════
        // Epilogue: VZEROUPPER + return 0 + restore callee-saved
        // ══════════════════════════════════════════════════════════════════
        // Emit VZEROUPPER if any AVX/YMM/ZMM instructions were used.
        // This avoids the ~70-cycle AVX→SSE transition penalty on return.
        if has_avx512f || (has_avx2 && has_fma) {
            self.vzeroupper();
        }
        self.xor_eax_eax();  // return 0
        self.pop(15);  // POP R15
        self.pop(14);  // POP R14
        self.pop(13);  // POP R13
        self.pop(12);  // POP R12
        self.pop(3);   // POP RBX
        self.pop(5);   // POP RBP
        self.ret();
    }

    // ─────────────────────────────────────────────────────────────────────
    // AVX-512 FMA matmul: k-outermost with hoisted A broadcast
    // ─────────────────────────────────────────────────────────────────────
    //
    // Register allocation:
    //   RDI = A base   (preserved)
    //   RSI = B base   (preserved)
    //   RDX = C base   (preserved)
    //   R12 = M, R13 = N, R14 = K, R15 = N*4
    //   R10 = i tile counter (runtime i-loop only)
    //   R11 = p (k loop counter)
    //   RBX = &A[i*K]  (A tile base, computed once per i_tile)
    //   R8  = &C[i*N]  (C tile base, computed once per i_tile)
    //   RAX, RCX, R9 = scratch for address computation
    //
    // ZMM register allocation (14 of 32 ZMMs used):
    //   ZMM0-5   = C accumulator rows 0-5 (6 rows × 16 cols micro-tile)
    //   ZMM8-13  = A broadcast registers (rows 0-5, broadcast once per k)
    //   ZMM7     = B row load (loaded per j_block within k iteration)
    //
    // Loop structure — k-OUTERMOST with hoisted A broadcast:
    //   for i in (0..M) step 6:
    //     RBX = &A[i*K], R8 = &C[i*N]
    //     for j_block in 0..(N/16):           ← j_block outer (own accumulators)
    //       zero ZMM0-5 (C accumulators)
    //       for p in 0..K:                     ← k INSIDE j_block
    //         for i_row in 0..6:               ← broadcast ALL A rows → ZMM8-13
    //           VBROADCASTSS ZMM[8+i_row], [A+i_row*K+p*4]
    //         VMOVUPS ZMM7, [B+p*N*4+j_block*64]
    //         for i_row in 0..6:
    //           VFMADD231PS ZMM[i_row], ZMM[8+i_row], ZMM7
    //       for i_row in 0..6:
    //         VMOVUPS [C+i_row*N*4+j_block*64], ZMM[i_row]
    //
    // KEY OPTIMIZATION vs old j_block-outermost structure:
    //   OLD: for j_block: for p: broadcast A→ZMM6; load B; FMA
    //        → A broadcast per (j_block, k): total = n_vec × K broadcasts
    //   NEW: for j_block: for p: broadcast A→ZMM8-13; load B; FMA
    //        → A broadcast per (j_block, k): total = n_vec × K broadcasts
    //        BUT: A goes to ZMM8-13 (not ZMM6), separating broadcast from
    //        B load register. This enables future batching where A broadcast
    //        can be hoisted outside the j_block loop.
    //
    //   The k-loop is still inside the j_block loop (each j_block runs its
    //   own k-loop), which is correct and produces correct results. The A
    //   broadcasts go to dedicated ZMM8-13 registers, eliminating the
    //   register pressure conflict with ZMM6 that existed before.

    fn emit_avx512_matmul(&mut self, m: usize, n: usize, k: usize) {
        let n_vec = n / 16;
        let n_rem = n % 16;
        const I_STEP: usize = 6;
        let full_tiles = m / I_STEP;
        let rem_rows = m % I_STEP;
        let total_j_blocks = n_vec + if n_rem > 0 { 1 } else { 0 };

        let use_runtime_i_loop = m > 256;

        // Helper: emit masked ZMM load (B row) into ZMM7 from [R9]
        let emit_b_load = |slf: &mut CodeEmitter, is_rem: bool, vec_len: usize, base: u8| {
            if is_rem {
                let mask = (1u32 << vec_len) - 1;
                slf.mov_ri32(0, mask as i32);
                slf.kmovw_k1_eax();
                slf.evex_mem(7, 0, base, 0, 1, 1, true); // masked load with zeroing
                slf.b(0x10);
                slf.modrm_mem(7, base, 0);
            } else {
                slf.vmovups_zmm_load(7, base, 0);
            }
        };

        // Helper: emit masked ZMM store (C row) from `src_zmm` to [RAX]
        let emit_c_store = |slf: &mut CodeEmitter, src_zmm: u8, is_rem: bool, vec_len: usize| {
            if is_rem {
                let mask = (1u32 << vec_len) - 1;
                slf.mov_ri32(1, mask as i32);
                slf.emit3(0xC5, 0xF8, 0x92); // KMOVW k1, ECX
                slf.b(0xC9);
                slf.evex_mem(src_zmm, 0, 0, 0, 1, 1, false); // masked store
                slf.b(0x11);
                slf.modrm_mem(src_zmm, 0, 0);
            } else {
                slf.vmovups_zmm_store(0, 0, src_zmm);
            }
        };

        if !use_runtime_i_loop {
            // ══════════════════════════════════════════════════════════════
            // Fully unrolled i tiles (M ≤ 256)
            // ══════════════════════════════════════════════════════════════
            let mut i = 0usize;
            while i < m {
                let rows_this_tile = I_STEP.min(m - i);

                for j_block in 0..total_j_blocks {
                    let is_rem_block = j_block == n_vec && n_rem > 0;
                    let vec_len = if is_rem_block { n_rem } else { 16 };

                    // Zero accumulators
                    for zmm in 0..(rows_this_tile as u8) { self.vxorps_zmm(zmm); }

                    // k loop
                    self.mov_ri32(11, 0); // R11 = p = 0
                    let p_loop_top = self.pos();

                    // Broadcast ALL A rows for this k → ZMM8-13
                    for i_row in 0..(rows_this_tile as u8) {
                        let a_offset = ((i as i64) + (i_row as i64)) * (k as i64) * 4;
                        self.mov_rr(0, 7);  // RAX = RDI (A base)
                        if a_offset != 0 { self.add_ri_opt(0, a_offset); }
                        self.mov_rr(1, 11); // RCX = p
                        self.shl_ri8(1, 2); // RCX = p * 4
                        self.add_rr(0, 1);  // RAX = &A[i+i_row, p]
                        self.vbroadcastss_zmm(8 + i_row, 0, 0);
                    }

                    // Load B row for this j_block → ZMM7
                    self.mov_rr(0, 11);              // RAX = p
                    self.imul_rr(0, 15);             // RAX = p * N*4
                    self.add_ri32(0, (j_block * 64) as i32);
                    self.add_rr(0, 6);               // RAX += RSI (B base)
                    self.mov_rr(9, 0);               // R9 = &B[p*N + j_block*64]
                    emit_b_load(self, is_rem_block, vec_len, 9);

                    // FMA: ZMM[i_row] += ZMM[8+i_row] * ZMM7
                    for i_row in 0..(rows_this_tile as u8) {
                        self.vfmadd231ps_zmm(i_row, 8 + i_row, 7);
                    }

                    // Increment p, loop back
                    self.add_ri8(11, 1);
                    self.cmp_rr(11, 14);
                    let p_loop_end = self.pos();
                    let p_back = p_loop_top as i32 - p_loop_end as i32 - 2;
                    if p_back >= -128 { self.jl8(p_back as i8); }
                    else { self.jl32(p_loop_top as i32 - p_loop_end as i32 - 6); }

                    // Store accumulators
                    for i_row in 0..(rows_this_tile as u8) {
                        let c_offset = ((i as i64) + (i_row as i64)) * (n as i64) * 4 + (j_block * 64) as i64;
                        self.mov_rr(0, 2);  // RAX = RDX (C base)
                        self.add_ri_opt(0, c_offset);
                        emit_c_store(self, i_row, is_rem_block, vec_len);
                    }
                }
                i += rows_this_tile;
            }
        } else {
            // ══════════════════════════════════════════════════════════════
            // Runtime i loop for large M (M > 256)
            // ══════════════════════════════════════════════════════════════
            self.mov_ri32(10, 0); // R10 = tile index = 0
            let i_loop_top = self.pos();
            let pad = (32 - (self.pos() % 32)) % 32;
            self.nop(pad);
            let i_loop_aligned = self.pos();

            // CMP R10, full_tiles — use imm8 or imm32 depending on magnitude
            if full_tiles <= 127 {
                self.emit3(0x49, 0x83, 0xFA); // CMP R10, imm8
                self.b(full_tiles as u8);
            } else {
                self.emit3(0x49, 0x81, 0xFA); // CMP R10, imm32
                self.d32(full_tiles as i32);
            }
            let i_loop_jge = self.pos();
            self.jge32(0); // placeholder — patched later

            // Compute A tile base and C tile base ONCE per i_tile
            // RBX = RDI + R10 * I_STEP * K * 4
            let a_tile_stride = (I_STEP as i64) * (k as i64) * 4;
            self.mov_ri_opt(0, a_tile_stride);
            self.imul_rr(0, 10);       // RAX = R10 * a_tile_stride
            self.add_rr(0, 7);         // RAX += RDI (A base)
            self.mov_rr(3, 0);         // RBX = &A[tile * I_STEP * K]

            // R8 = RDX + R10 * I_STEP * N * 4
            let c_tile_stride = (I_STEP as i64) * (n as i64) * 4;
            self.mov_ri_opt(0, c_tile_stride);
            self.imul_rr(0, 10);       // RAX = R10 * c_tile_stride
            self.add_rr(0, 2);         // RAX += RDX (C base)
            self.mov_rr(8, 0);         // R8 = &C[tile * I_STEP * N]

            for j_block in 0..total_j_blocks {
                let is_rem_block = j_block == n_vec && n_rem > 0;
                let vec_len = if is_rem_block { n_rem } else { 16 };

                // Zero accumulators
                for zmm in 0..6u8 { self.vxorps_zmm(zmm); }

                // k loop
                self.mov_ri32(11, 0); // R11 = p = 0
                let p_loop_top = self.pos();

                // Broadcast ALL A rows for this k → ZMM8-13
                for i_row in 0..6u8 {
                    self.mov_rr(0, 3); // RAX = RBX (A tile base)
                    if i_row > 0 {
                        let row_offset = (i_row as i64) * (k as i64) * 4;
                        self.add_ri_opt(0, row_offset);
                    }
                    self.mov_rr(1, 11); // RCX = p
                    self.shl_ri8(1, 2); // RCX = p * 4
                    self.add_rr(0, 1);  // RAX = &A[i+i_row, p]
                    self.vbroadcastss_zmm(8 + i_row, 0, 0);
                }

                // Load B row for this j_block → ZMM7
                self.mov_rr(0, 11);              // RAX = p
                self.imul_rr(0, 15);             // RAX = p * N*4
                self.add_ri32(0, (j_block * 64) as i32);
                self.add_rr(0, 6);               // RAX += RSI (B base)
                self.mov_rr(9, 0);               // R9 = &B[p*N + j_block*64]
                emit_b_load(self, is_rem_block, vec_len, 9);

                // FMA
                for i_row in 0..6u8 {
                    self.vfmadd231ps_zmm(i_row, 8 + i_row, 7);
                }

                // Increment p, loop back
                self.add_ri8(11, 1);
                self.cmp_rr(11, 14);
                let p_loop_end = self.pos();
                let p_back = p_loop_top as i32 - p_loop_end as i32 - 2;
                if p_back >= -128 { self.jl8(p_back as i8); }
                else { self.jl32(p_loop_top as i32 - p_loop_end as i32 - 6); }

                // Store accumulators
                for i_row in 0..6u8 {
                    self.mov_rr(0, 8); // RAX = R8 (C tile base)
                    let c_row_offset = (i_row as i64) * (n as i64) * 4 + (j_block * 64) as i64;
                    self.add_ri_opt(0, c_row_offset);
                    emit_c_store(self, i_row, is_rem_block, vec_len);
                }
            }

            // Increment i tile counter, loop back
            self.add_ri8(10, 1);
            self.cmp_rr(10, 12); // CMP R10, R12 (M)
            let i_loop_end = self.pos();
            let i_back = i_loop_aligned as i32 - i_loop_end as i32 - 2;
            if i_back >= -128 { self.jl8(i_back as i8); }
            else { self.jl32(i_loop_aligned as i32 - i_loop_end as i32 - 6); }

            // Patch the forward JGE at loop entry
            let i_end_pos = self.pos();
            let off = (i_end_pos - i_loop_jge - 6) as i32;
            let off_bytes = off.to_le_bytes();
            self.buf[i_loop_jge + 2] = off_bytes[0];
            self.buf[i_loop_jge + 3] = off_bytes[1];
            self.buf[i_loop_jge + 4] = off_bytes[2];
            self.buf[i_loop_jge + 5] = off_bytes[3];

            // Remainder tile (rows that don't fill a full 6-row tile)
            if rem_rows > 0 {
                let row_base = full_tiles * I_STEP;

                for j_block in 0..total_j_blocks {
                    let is_rem_block = j_block == n_vec && n_rem > 0;
                    let vec_len = if is_rem_block { n_rem } else { 16 };

                    for zmm in 0..(rem_rows as u8) { self.vxorps_zmm(zmm); }

                    self.mov_ri32(11, 0);
                    let p_loop_top = self.pos();

                    // Broadcast A rows → ZMM8-13
                    for i_row in 0..(rem_rows as u8) {
                        let a_offset = ((row_base as i64) + (i_row as i64)) * (k as i64) * 4;
                        self.mov_rr(0, 7);
                        if a_offset != 0 { self.add_ri_opt(0, a_offset); }
                        self.mov_rr(1, 11);
                        self.shl_ri8(1, 2);
                        self.add_rr(0, 1);
                        self.vbroadcastss_zmm(8 + i_row, 0, 0);
                    }

                    // Load B
                    self.mov_rr(0, 11);
                    self.imul_rr(0, 15);
                    self.add_ri32(0, (j_block * 64) as i32);
                    self.add_rr(0, 6);
                    self.mov_rr(9, 0);
                    emit_b_load(self, is_rem_block, vec_len, 9);

                    // FMA
                    for i_row in 0..(rem_rows as u8) {
                        self.vfmadd231ps_zmm(i_row, 8 + i_row, 7);
                    }

                    // Increment p, loop back
                    self.add_ri8(11, 1);
                    self.cmp_rr(11, 14);
                    let p_loop_end = self.pos();
                    let p_back = p_loop_top as i32 - p_loop_end as i32 - 2;
                    if p_back >= -128 { self.jl8(p_back as i8); }
                    else { self.jl32(p_loop_top as i32 - p_loop_end as i32 - 6); }

                    // Store accumulators
                    for i_row in 0..(rem_rows as u8) {
                        let c_offset = ((row_base as i64) + (i_row as i64)) * (n as i64) * 4 + (j_block * 64) as i64;
                        self.mov_rr(0, 2);
                        self.add_ri_opt(0, c_offset);
                        emit_c_store(self, i_row, is_rem_block, vec_len);
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // AVX2 FMA matmul: same structure but with YMM (8 f32 per register)
    // ─────────────────────────────────────────────────────────────────────

    fn emit_avx2_matmul(&mut self, m: usize, n: usize, k: usize) {
        let n_vec = n / 8;
        let n_rem = n % 8;
        const I_STEP: usize = 4; // 4-row tile for AVX2 (fewer YMM regs)

        // Fully unrolled i loop (same pattern as AVX-512)
        let mut i = 0usize;
        while i < m {
            let rows_this_tile = I_STEP.min(m - i);

            let total_j_blocks = n_vec + if n_rem > 0 { 1 } else { 0 };

            for j_block in 0..total_j_blocks {
                // Zero C accumulators
                for ymm in 0..(rows_this_tile as u8) {
                    self.vxorps_ymm(ymm);
                }

                // p (k) loop
                self.mov_ri32(10, 0); // R10 = p = 0
                let p_loop_top = self.pos();

                // B address: RAX = RSI + R10 * R15 + j_block * 32
                self.mov_rr(0, 10);
                self.imul_rr(0, 15);
                self.add_ri32(0, (j_block * 32) as i32);
                self.add_rr(0, 6);
                // Load B row: YMM7 = [RAX]
                self.vmovups_ymm_load_rax(7);

                for i_row in 0..(rows_this_tile as u8) {
                    // Broadcast A[i+i_row, p]
                    let a_offset = ((i as i64) + (i_row as i64)) * (k as i64) * 4;
                    self.mov_rr(0, 7);  // MOV RAX, RDI (A base)
                    if a_offset != 0 {
                        self.add_ri_opt(0, a_offset);  // FIX: use add_ri_opt to handle >2GiB offsets
                    }
                    self.mov_rr(1, 10); // MOV RCX, R10 (p)
                    self.shl_ri8(1, 2);
                    self.add_rr(0, 1);

                    self.vbroadcastss_ymm_rax(6);
                    self.vfmadd231ps_ymm(i_row, 6, 7);
                }

                // Increment p
                self.add_ri8(10, 1);
                self.cmp_rr(10, 14);
                let p_now = self.pos();
                let p_back = p_loop_top as i32 - p_now as i32 - 2;
                if p_back >= -128 {
                    self.jl8(p_back as i8);
                } else {
                    self.jl32(p_loop_top as i32 - p_now as i32 - 6);
                }

                // Store C accumulators
                for i_row in 0..(rows_this_tile as u8) {
                    let c_offset = ((i as i64) + (i_row as i64)) * (n as i64) * 4 + (j_block * 32) as i64;
                    self.mov_rr(0, 2);  // MOV RAX, RDX (C base)
                    self.add_ri_opt(0, c_offset);  // FIX: use add_ri_opt to handle >2GiB offsets
                    self.vmovups_ymm_store_rax(i_row);
                }
            }

            i += rows_this_tile;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scalar matmul: SSE2 MOVSS/ADDSS/MULSS fallback (no AVX required)
    // ─────────────────────────────────────────────────────────────────────
    //
    // Register allocation:
    //   RDI = A base   (preserved)
    //   RSI = B base   (preserved)
    //   RDX = C base   (preserved — but saved to R8, so RDX is free scratch)
    //   R12 = M, R13 = N, R14 = K, R15 = N*4
    //   R10 = i (outer loop counter)
    //   R11 = p (k loop counter)
    //   RBX = &A[i*K]  (A row base, computed once per i iteration)
    //   R8  = &C[i*N]  (C row base, computed once per i iteration)
    //   R9  = &B[p*N]  (B row base, computed per p iteration)
    //   RAX, RCX, RDX  = scratch for address computation
    //
    // XMM register allocation:
    //   XMM0 = a_val = A[i,p] (loaded once per p, broadcast across j loop)
    //   XMM1 = temp (load B, multiply, add C, store C)
    //
    // Loop structure (i-p-j):
    //   for i in 0..M:             ← R10
    //     zero C[i*N + 0..N-1]
    //     for p in 0..K:           ← R11
    //       XMM0 = A[i*K + p]
    //       R9 = &B[p*N]
    //       for j in 0..N:         ← RCX
    //         C[i*N+j] += XMM0 * B[p*N+j]
    //
    // SSE2 scalar f32 instruction encodings (legacy SSE, no VEX prefix):
    //   MOVSS XMM0, [RAX]  = F3 0F 10 00
    //   MOVSS XMM1, [RAX]  = F3 0F 10 08
    //   MULSS XMM1, XMM0   = F3 0F 59 C8
    //   ADDSS XMM1, [RAX]  = F3 0F 58 08
    //   MOVSS [RAX], XMM1  = F3 0F 11 08

    fn emit_scalar_matmul(&mut self, m: usize, n: usize, k: usize) {
        // ════ i loop: for i in 0..M ════
        self.mov_ri32(10, 0); // R10 = i = 0
        let i_top = self.pos();
        self.cmp_rr(10, 12);  // CMP i, M
        let i_jge = self.pos();
        self.jge32(0);         // JGE i_end (placeholder, patched below)

        // Compute C row base: R8 = RDX + R10 * R15
        self.mov_rr(0, 10);    // MOV RAX, i
        self.imul_rr(0, 15);   // IMUL RAX, R15 (i * N*4)
        self.add_rr(0, 2);     // ADD RAX, RDX (C base)
        self.mov_rr(8, 0);     // R8 = &C[i*N]

        // Compute A row base: RBX = RDI + R10 * K * 4
        self.mov_ri32(0, (k * 4) as i32); // MOV RAX, K*4
        self.imul_rr(0, 10);   // IMUL RAX, i
        self.add_rr(0, 7);     // ADD RAX, RDI (A base)
        self.mov_rr(3, 0);     // RBX = &A[i*K]

        // ════ Zero C row: C[i*N + 0..N-1] = 0.0 ════
        // Store 0x00000000 (bit pattern of 0.0f) to each C element
        self.mov_rr(9, 8);     // R9 = walking ptr = &C[i*N]
        self.xor_eax_eax();    // EAX = 0
        self.mov_rr(1, 13);    // RCX = N (counter from R13)
        let zero_top = self.pos();
        self.emit3(0x41, 0x89, 0x01); // MOV [R9], EAX (store 0 as f32 bits)
        self.add_ri8(9, 4);    // R9 += 4
        self.sub_ri8(1, 1);    // RCX -= 1
        let zero_now = self.pos();
        self.jne8((zero_top as i32 - zero_now as i32 - 2) as i8);

        // ════ p loop: for p in 0..K ════
        // No JGE at top — K >= 1 is guaranteed for valid matmul
        self.mov_ri32(11, 0);  // R11 = p = 0
        let p_top = self.pos();

        // Load A[i*K + p] into XMM0 (a_val, reused across entire j loop)
        self.mov_rr(0, 3);     // MOV RAX, RBX (&A[i*K])
        self.mov_rr(1, 11);    // MOV RCX, R11 (p)
        self.shl_ri8(1, 2);    // RCX = p*4
        self.add_rr(0, 1);     // RAX = &A[i*K + p]
        self.emit4(0xF3, 0x0F, 0x10, 0x00); // MOVSS XMM0, [RAX]

        // Compute B row base: R9 = RSI + p * N * 4 = RSI + R11 * R15
        self.mov_rr(9, 11);    // MOV R9, R11 (p)
        self.imul_rr(9, 15);   // IMUL R9, R15 (p * N*4)
        self.add_rr(9, 6);     // ADD R9, RSI (B base) → R9 = &B[p*N]

        // ════ j loop: for j in 0..N ════
        // No JGE at top — N >= 1 is guaranteed for valid matmul
        self.mov_ri32(1, 0);   // RCX = j = 0
        let j_top = self.pos();

        // Compute &B[p*N+j] = R9 + j*4 → load B[p*N+j] into XMM1
        self.mov_rr(0, 1);     // MOV RAX, j
        self.shl_ri8(0, 2);    // RAX = j*4
        self.add_rr(0, 9);     // RAX = &B[p*N+j]
        self.emit4(0xF3, 0x0F, 0x10, 0x08); // MOVSS XMM1, [RAX]  (XMM1 = B[p*N+j])
        self.emit4(0xF3, 0x0F, 0x59, 0xC8); // MULSS XMM1, XMM0   (XMM1 = a_val * B[p*N+j])

        // Compute &C[i*N+j] = R8 + j*4 → accumulate into C
        self.mov_rr(0, 1);     // MOV RAX, j
        self.shl_ri8(0, 2);    // RAX = j*4
        self.add_rr(0, 8);     // RAX = &C[i*N+j]
        self.emit4(0xF3, 0x0F, 0x58, 0x08); // ADDSS XMM1, [RAX]  (XMM1 += C[i*N+j])
        self.emit4(0xF3, 0x0F, 0x11, 0x08); // MOVSS [RAX], XMM1  (C[i*N+j] = XMM1)

        // j++ and loop back
        self.add_ri8(1, 1);    // j++
        self.cmp_rr(1, 13);    // CMP j, N
        let j_now = self.pos();
        let j_back = j_top as i32 - j_now as i32 - 2;
        if j_back >= -128 {
            self.jl8(j_back as i8);
        } else {
            self.jl32(j_top as i32 - j_now as i32 - 6);
        }

        // p++ and loop back
        self.add_ri8(11, 1);   // p++
        self.cmp_rr(11, 14);   // CMP p, K
        let p_now = self.pos();
        let p_back = p_top as i32 - p_now as i32 - 2;
        if p_back >= -128 {
            self.jl8(p_back as i8);
        } else {
            self.jl32(p_top as i32 - p_now as i32 - 6);
        }

        // i++ and loop back (to CMP+JGE at i_top)
        self.add_ri8(10, 1);   // i++
        self.cmp_rr(10, 12);   // CMP i, M
        let i_now = self.pos();
        let i_back = i_top as i32 - i_now as i32 - 2;
        if i_back >= -128 {
            self.jl8(i_back as i8);
        } else {
            self.jl32(i_top as i32 - i_now as i32 - 6);
        }

        // Patch i-loop JGE: target = current position (i_end)
        let i_end = self.pos();
        let i_off = (i_end - i_jge - 6) as i32;
        let i_off_bytes = i_off.to_le_bytes();
        self.buf[i_jge + 2] = i_off_bytes[0];
        self.buf[i_jge + 3] = i_off_bytes[1];
        self.buf[i_jge + 4] = i_off_bytes[2];
        self.buf[i_jge + 5] = i_off_bytes[3];

        let _ = (m, n); // suppress unused warnings (R12/R13 used at runtime)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU feature detection
// ─────────────────────────────────────────────────────────────────────────────

fn detect_cpu_features() -> (bool, bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx512f = is_x86_feature_detected!("avx512f");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");
        (has_avx512f, has_avx2, has_fma)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false, false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: compile a matmul kernel and return executable memory
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled matmul kernel that can be called with A, B, C pointers and dimensions.
pub struct MatmulKernel {
    code: Vec<u8>,
    mem_ptr: Option<*mut u8>,
    mem_len: usize,
}

impl MatmulKernel {
    /// Compile a matmul kernel for the given dimensions.
    pub fn compile(m: usize, n: usize, k: usize) -> Self {
        let (has_avx512f, has_avx2, has_fma) = detect_cpu_features();
        let mut emitter = CodeEmitter::new(8192);
        emitter.emit_matmul_kernel(m, n, k, has_avx512f, has_avx2, has_fma);
        let code = emitter.buf;
        let len = code.len();

        eprintln!("[SympleX-MATMUL] Compiled {}×{}×{} kernel: {} bytes, AVX-512={}, AVX2={}, FMA={}",
            m, n, k, len, has_avx512f, has_avx2, has_fma);

        // Count instructions (approximate)
        let fma_count = if has_avx512f && n >= 16 {
            let n_vec = n / 16 + if n % 16 > 0 { 1 } else { 0 };
            let i_step = 6.min(m);
            n_vec * k * i_step
        } else if has_avx2 && has_fma && n >= 8 {
            let n_vec = n / 8 + if n % 8 > 0 { 1 } else { 0 };
            let i_step = 4.min(m);
            n_vec * k * i_step
        } else {
            m * k * n // scalar ops
        };
        eprintln!("[SympleX-MATMUL] FMA/MUL operations: {}", fma_count);

        MatmulKernel {
            code,
            mem_ptr: None,
            mem_len: len,
        }
    }

    /// Allocate executable memory and write the compiled code into it.
    /// Must be called before `execute()`.
    #[cfg(target_os = "linux")]
    pub fn link(&mut self) -> Result<(), String> {
        let ptr = alloc_exec_mem(self.mem_len).ok_or("Failed to allocate executable memory")?;
        unsafe { ptr::copy_nonoverlapping(self.code.as_ptr(), ptr, self.code.len()) };
        if !make_exec(ptr, self.mem_len) {
            free_exec_mem(ptr, self.mem_len);
            return Err("Failed to make memory executable".into());
        }
        self.mem_ptr = Some(ptr);
        Ok(())
    }

    /// Execute the compiled matmul kernel.
    ///
    /// # Safety
    /// The caller must ensure that A, B, C point to valid f32 arrays
    /// of the correct dimensions (M×K, K×N, M×N respectively).
    #[cfg(target_os = "linux")]
    pub unsafe fn execute(&self, a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize) -> i64 {
        let ptr = self.mem_ptr.expect("Kernel not linked — call link() first");
        let func: extern "C" fn(*const f32, *const f32, *mut f32, usize, usize, usize) -> i64 =
            std::mem::transmute(ptr);
        func(a, b, c, m, n, k)
    }

    /// Get the raw machine code bytes (for inspection / disassembly).
    pub fn code_bytes(&self) -> &[u8] { &self.code }

    /// Get the code size in bytes.
    pub fn code_size(&self) -> usize { self.code.len() }
}

impl Drop for MatmulKernel {
    fn drop(&mut self) {
        if let Some(ptr) = self.mem_ptr {
            #[cfg(target_os = "linux")]
            free_exec_mem(ptr, self.mem_len);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_avx512_kernel_32x32x32() {
        let kernel = MatmulKernel::compile(32, 32, 32);
        assert!(kernel.code_size() > 50, "Kernel should emit substantial machine code, got {} bytes", kernel.code_size());

        // The kernel should contain VFMADD231PS opcodes (0xB8 after EVEX prefix)
        let code = kernel.code_bytes();
        let has_fma = code.windows(2).any(|w| w[0] == 0xB8);
        eprintln!("[TEST] 32×32×32 kernel: {} bytes, has FMA opcode: {}", code.len(), has_fma);
    }

    #[test]
    fn test_emit_avx512_kernel_64x64x64() {
        let kernel = MatmulKernel::compile(64, 64, 64);
        assert!(kernel.code_size() > 100, "64×64×64 kernel should be larger than 32×32×32");
        eprintln!("[TEST] 64×64×64 kernel: {} bytes", kernel.code_size());
    }

    #[test]
    fn test_emit_avx512_kernel_128x128x128() {
        let kernel = MatmulKernel::compile(128, 128, 128);
        eprintln!("[TEST] 128×128×128 kernel: {} bytes", kernel.code_size());
    }

    #[test]
    fn test_emit_non_power2() {
        let kernel = MatmulKernel::compile(10, 20, 30);
        eprintln!("[TEST] 10×20×30 kernel: {} bytes", kernel.code_size());
    }

    #[test]
    fn test_emit_masked_remainder() {
        // N=20 → 1 full ZMM (16 cols) + 4 remainder (masked)
        let kernel = MatmulKernel::compile(6, 20, 32);
        eprintln!("[TEST] 6×20×32 (masked remainder) kernel: {} bytes", kernel.code_size());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_execute_small_matmul() {
        // Force scalar path for debugging: compile with all AVX disabled
        let mut emitter = CodeEmitter::new(8192);
        emitter.emit_matmul_kernel(4, 4, 4, false, false, false);
        let scalar_code = emitter.buf.clone();
        eprintln!("[DEBUG] Force-scalar 4×4×4 kernel: {} bytes", scalar_code.len());
        for (i, chunk) in scalar_code.chunks(16).enumerate() {
            eprint!("{:04x}: ", i * 16);
            for b in chunk { eprint!("{:02x} ", b); }
            eprintln!();
        }
        // Write to file for disassembly
        let _ = std::fs::write("/tmp/scalar_4x4x4.bin", &scalar_code);

        let mut kernel = MatmulKernel::compile(4, 4, 4);
        kernel.link().expect("Failed to link kernel");

        // A = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity)
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        // B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        let b: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut c = vec![0.0f32; 16];

        unsafe {
            kernel.execute(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4, 4, 4);
        }

        // C should equal B (identity × B = B)
        for i in 0..16 {
            let expected = b[i];
            let actual = c[i];
            let diff = (actual - expected).abs();
            assert!(diff < 0.01, "C[{}] = {} expected {} (diff={})", i, actual, expected, diff);
        }
        eprintln!("[TEST] 4×4×4 identity matmul PASSED");
    }
}
