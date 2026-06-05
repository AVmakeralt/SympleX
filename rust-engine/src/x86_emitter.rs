//! x86-64 Machine Code Kernel Emitter for SympleX
//!
//! Uses iced-x86 for correct instruction encoding — eliminates ALL manual
//! VEX/EVEX prefix bugs that caused SIGSEGV/#UD with extended registers.
//!
//! Optimizations:
//!   Y: Multi-stream interleaved AVX-512 — 4 independent ZMM accumulator streams
//!   W: Multi-byte NOP stencils (0x0F 0x1F) replace single-byte 0x90 NOPs
//!   K+O: Software-pipelined load-compute interleaving
//!   B+U: Context invariant inlining — M, N, K baked as immediates
//!   S: 64-byte cache-line alignment for inner loop headers

use std::ptr;
use iced_x86::code_asm::*;

// ── Executable memory ──

unsafe fn alloc_exec(code: &[u8]) -> *mut u8 {
    let len = (code.len() + 4095) & !4095;
    let p = libc::mmap(
        ptr::null_mut(), len,
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0,
    );
    if p == libc::MAP_FAILED { return ptr::null_mut(); }
    ptr::copy_nonoverlapping(code.as_ptr(), p as *mut u8, code.len());
    p as *mut u8
}

unsafe fn free_exec(p: *mut u8, len: usize) {
    libc::munmap(p as *mut libc::c_void, (len + 4095) & !4095);
}

/// Emit multi-byte NOP padding of exactly `n` bytes.
pub fn emit_nop_padding(a: &mut CodeAssembler, n: usize) {
    let mut remaining = n;
    while remaining > 0 {
        match remaining {
            1 => { a.db(&[0x90]).unwrap(); remaining = 0; }
            2 => { a.db(&[0x66, 0x90]).unwrap(); remaining = 0; }
            3 => { a.db(&[0x0F, 0x1F, 0x00]).unwrap(); remaining = 0; }
            4 => { a.db(&[0x0F, 0x1F, 0x40, 0x00]).unwrap(); remaining = 0; }
            5 => { a.db(&[0x0F, 0x1F, 0x44, 0x00, 0x00]).unwrap(); remaining = 0; }
            6 => { a.db(&[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]).unwrap(); remaining = 0; }
            7 => { a.db(&[0x0F, 0x1F, 0x80]).unwrap(); a.dd(&[0]).unwrap(); remaining = 0; }
            8 => { a.db(&[0x0F, 0x1F, 0x84, 0x00]).unwrap(); a.dd(&[0]).unwrap(); remaining = 0; }
            9 => { a.db(&[0x66, 0x0F, 0x1F, 0x84, 0x00]).unwrap(); a.dd(&[0]).unwrap(); remaining = 0; }
            _ => { a.db(&[0x66, 0x0F, 0x1F, 0x84, 0x00]).unwrap(); a.dd(&[0]).unwrap(); remaining -= 9; }
        }
    }
}

/// Align to `alignment` boundary using multi-byte NOPs.
pub fn emit_align(a: &mut CodeAssembler, alignment: usize, current_len: usize) {
    let padding = (alignment - (current_len % alignment)) % alignment;
    if padding > 0 { emit_nop_padding(a, padding); }
}

// ── CompiledKernel ──

pub struct CompiledKernel {
    code: Vec<u8>,
    exec_ptr: *mut u8,
    m: usize, n: usize, k: usize,
}

impl CompiledKernel {
    pub fn exec_ptr(&self) -> *mut u8 { self.exec_ptr }

    pub fn exec_matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: i64, n: i64, k: i64) -> i64 {
        if self.exec_ptr.is_null() { return -1; }
        unsafe {
            let f: extern "C" fn(*const f32, *const f32, *mut f32, i64, i64, i64) -> i64 =
                std::mem::transmute(self.exec_ptr);
            f(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k)
        }
    }

    pub fn exec_elementwise(&self, dst: &mut [f32], a: &[f32], b: &[f32], n: i64) -> i64 {
        if self.exec_ptr.is_null() { return -1; }
        unsafe {
            let f: extern "C" fn(*mut f32, *const f32, *const f32, i64) -> i64 =
                std::mem::transmute(self.exec_ptr);
            f(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n)
        }
    }

    pub fn compiled_dims(&self) -> (usize, usize, usize) { (self.m, self.n, self.k) }

    pub fn exec_fused_matmul_bias_relu(
        &self, a: &[f32], b: &[f32], c: &mut [f32], bias: &[f32],
        m: i64, n: i64, k: i64
    ) -> i64 {
        if self.exec_ptr.is_null() { return -1; }
        unsafe {
            let f: extern "C" fn(*const f32, *const f32, *mut f32, i64, i64, i64, *const f32) -> i64 =
                std::mem::transmute(self.exec_ptr);
            f(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, bias.as_ptr())
        }
    }

    fn finalize(mut a: CodeAssembler, m: usize, n: usize, k: usize) -> Self {
        let code = a.assemble(0x10000).expect("iced-x86 assembly failed");
        let _ = std::fs::write("/tmp/symplex_kernel.bin", &code);
        let exec_ptr = unsafe { alloc_exec(&code) };
        CompiledKernel { code, exec_ptr, m, n, k }
    }
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        if !self.exec_ptr.is_null() {
            unsafe { free_exec(self.exec_ptr, self.code.len()); }
        }
    }
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

// ── ISA Level Detection ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ISALevel { SSE, AVX2, AVX512 }

impl std::fmt::Display for ISALevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ISALevel::SSE => write!(f, "SSE"),
            ISALevel::AVX2 => write!(f, "AVX2"),
            ISALevel::AVX512 => write!(f, "AVX-512"),
        }
    }
}

pub fn detect_isa_level() -> ISALevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { ISALevel::AVX512 }
        else if is_x86_feature_detected!("avx2") { ISALevel::AVX2 }
        else { ISALevel::SSE }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { ISALevel::SSE }
}

pub fn vector_width() -> usize {
    match detect_isa_level() {
        ISALevel::SSE => 1, ISALevel::AVX2 => 8, ISALevel::AVX512 => 16,
    }
}

// ── Kernel compilation ──
// All use iced-x86 CodeAssembler for correct VEX/EVEX encoding.
// Calling convention (System V AMD64): rdi=A, rsi=B, rdx=C, rcx=M, r8=N, r9=K

impl CompiledKernel {
    /// Compile SSE scalar matmul with register-resident accumulators (cache-friendly).
    ///
    /// Loop order: i → j → k (accumulator stays in XMM register for entire k-loop).
    /// This eliminates load-C/modify/store-C on every k iteration, reducing C
    /// memory traffic from 2*M*N*K to just 2*M*N writes. For a 64×64×64 matmul,
    /// this cuts L1D misses on C from ~262K to ~4K — a ~60x reduction.
    pub fn compile_matmul() -> Self {
        let mut a = CodeAssembler::new(64).unwrap();

        // Prologue
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();

        // R10=M, R11=K, R12=A, R13=B, R14=C, R15=N
        a.mov(r10, rcx).unwrap(); a.mov(r11, r9).unwrap();
        a.mov(r12, rdi).unwrap(); a.mov(r13, rsi).unwrap();
        a.mov(r14, rdx).unwrap(); a.mov(r15, r8).unwrap();

        // Tiled matmul: i-j-k with accumulator in XMM register
        // For each (i, j), the accumulator XMM0 holds the running dot product
        // across all k values — no C load/store inside the k-loop.
        a.xor(rdi, rdi).unwrap(); // rdi = i
        let mut m_i = a.create_label();
        let mut m_i_out = a.create_label();
        a.set_label(&mut m_i).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(m_i_out).unwrap();

        a.xor(r8, r8).unwrap(); // r8 = j
        let mut m_j = a.create_label();
        let mut m_j_out = a.create_label();
        a.set_label(&mut m_j).unwrap();
        a.cmp(r8, r15).unwrap(); a.jge(m_j_out).unwrap();

        // Zero the accumulator for this (i,j) element
        a.xorps(xmm0, xmm0).unwrap();

        // k-loop: accumulate A[i,k]*B[k,j] into XMM0
        a.xor(r9, r9).unwrap(); // r9 = k
        let mut m_k = a.create_label();
        let mut m_k_out = a.create_label();
        a.set_label(&mut m_k).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(m_k_out).unwrap();

        // Load A[i,k] — row-major, contiguous in k
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r11).unwrap(); a.add(rax, r9).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r12).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();

        // Load B[k,j] — stride N, but only one load per k
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, r8).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r13).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();

        // XMM0 += XMM1 * XMM2
        a.mulss(xmm2, xmm1).unwrap();
        a.addss(xmm0, xmm2).unwrap();

        a.add(r9, 1).unwrap(); a.jmp(m_k).unwrap();
        a.set_label(&mut m_k_out).unwrap();

        // Store C[i,j] — written ONCE after the entire k-loop
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, r8).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm0).unwrap();

        a.add(r8, 1).unwrap(); a.jmp(m_j).unwrap();
        a.set_label(&mut m_j_out).unwrap();
        a.add(rdi, 1).unwrap(); a.jmp(m_i).unwrap();
        a.set_label(&mut m_i_out).unwrap();

        a.xor(rax, rax).unwrap();
        a.pop(r15).unwrap(); a.pop(r14).unwrap(); a.pop(r13).unwrap();
        a.pop(r12).unwrap(); a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile elementwise kernel (SSE scalar f32, add/sub/mul/div/min/max)
    /// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    pub fn compile_elementwise(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap();
        a.mov(r10, rdi).unwrap(); a.mov(r11, rsi).unwrap();
        a.mov(r12, rdx).unwrap(); a.mov(rbx, rcx).unwrap(); // RBX = n

        // RCX = byte offset, RAX = n*4 byte limit
        a.xor(rcx, rcx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap();
        let mut lp = a.create_label();
        let mut lp_out = a.create_label();
        a.set_label(&mut lp).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(lp_out).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.movss(xmm0, dword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.movss(xmm1, dword_ptr(rdx)).unwrap();
        match op {
            0 => { a.addss(xmm0, xmm1).unwrap(); }
            1 => { a.subss(xmm0, xmm1).unwrap(); }
            2 => { a.mulss(xmm0, xmm1).unwrap(); }
            3 => { a.divss(xmm0, xmm1).unwrap(); }
            4 => { a.minss(xmm0, xmm1).unwrap(); }
            5 => { a.maxss(xmm0, xmm1).unwrap(); }
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.movss(dword_ptr(rdx), xmm0).unwrap();
        a.add(rcx, 4).unwrap(); a.jmp(lp).unwrap();
        a.set_label(&mut lp_out).unwrap();
        a.xor(rax, rax).unwrap(); a.pop(r12).unwrap(); a.pop(rbx).unwrap();
        a.pop(rbp).unwrap(); a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile AVX2-vectorized matmul with register-resident accumulators.
    ///
    /// Loop order: i → j_block(8-wide) → k — same structure as AVX-512 kernel.
    /// The j_block of C stays in YMM registers for the entire k-loop, eliminating
    /// load-C/modify/store-C on every k iteration. C memory traffic drops from
    /// 2*M*N*K*4B to 2*M*N*4B. For 256×256×256, this saves ~128MB of L1 traffic.
    ///
    /// Additionally uses 4 independent YMM accumulator streams to hide FMA latency
    /// (5-cycle on Skylake/Zen4). Each stream processes a separate 8-wide j_block,
    /// giving 32 floats of C resident in registers per i iteration.
    pub fn compile_matmul_avx2() -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();
        // R10=M, R11=K, R12=A, R13=B, R14=C, R15=N
        a.mov(r10, rcx).unwrap(); a.mov(r11, r9).unwrap();
        a.mov(r12, rdi).unwrap(); a.mov(r13, rsi).unwrap();
        a.mov(r14, rdx).unwrap(); a.mov(r15, r8).unwrap();

        // ── i-loop ──
        a.xor(rdi, rdi).unwrap(); // rdi = i
        let mut mi_loop = a.create_label();
        let mut mi_exit = a.create_label();
        a.set_label(&mut mi_loop).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(mi_exit).unwrap();

        // RDX = i*K*4 (byte offset into A's row)
        a.mov(rdx, rdi).unwrap();
        a.imul_2(rdx, r11).unwrap();
        a.shl(rdx, 2).unwrap();

        // ── j_block loop: process 32 columns of C at a time (4×YMM = 32 floats) ──
        // This gives 4 independent accumulator streams to hide FMA latency.
        a.xor(r8, r8).unwrap(); // r8 = j byte offset
        a.mov(rbx, r15).unwrap(); a.shl(rbx, 2).unwrap(); // RBX = N*4 (total C row bytes)

        let mut mj_block = a.create_label();
        let mut mj_block_exit = a.create_label();
        a.set_label(&mut mj_block).unwrap();
        a.mov(rax, rbx).unwrap(); a.sub(rax, 128).unwrap(); // 32 floats * 4 bytes = 128
        a.cmp(r8, rax).unwrap(); a.jge(mj_block_exit).unwrap();

        // Zero 4 independent YMM accumulators (32 floats of C)
        a.vxorps(ymm0, ymm0, ymm0).unwrap();
        a.vxorps(ymm1, ymm1, ymm1).unwrap();
        a.vxorps(ymm2, ymm2, ymm2).unwrap();
        a.vxorps(ymm3, ymm3, ymm3).unwrap();

        // k-loop: accumulate into YMM registers — NO C load/store inside
        a.xor(r9, r9).unwrap(); // r9 = k
        let mut mk_loop = a.create_label();
        let mut mk_exit = a.create_label();
        a.set_label(&mut mk_loop).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(mk_exit).unwrap();

        // Broadcast A[i,k]
        a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
        a.vbroadcastss(ymm4, dword_ptr(rax)).unwrap();

        // RBX_k = k*N*4
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();

        // Load 4×8 = 32 floats of B[k, j..j+32]
        a.lea(rcx, qword_ptr(r13 + rax)).unwrap(); a.add(rcx, r8).unwrap();
        a.vmovups(ymm5, ymmword_ptr(rcx)).unwrap();
        a.vmovups(ymm6, ymmword_ptr(rcx + 32)).unwrap();
        a.vmovups(ymm7, ymmword_ptr(rcx + 64)).unwrap();
        a.vmovups(ymm8, ymmword_ptr(rcx + 96)).unwrap();

        // 4 independent FMAs — hides 5-cycle FMA latency
        a.vfmadd231ps(ymm0, ymm4, ymm5).unwrap();
        a.vfmadd231ps(ymm1, ymm4, ymm6).unwrap();
        a.vfmadd231ps(ymm2, ymm4, ymm7).unwrap();
        a.vfmadd231ps(ymm3, ymm4, ymm8).unwrap();

        a.add(r9, 1).unwrap(); a.jmp(mk_loop).unwrap();
        a.set_label(&mut mk_exit).unwrap();

        // Store 32 floats of C — written ONCE after the entire k-loop
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm0).unwrap();
        a.vmovups(ymmword_ptr(rax + 32), ymm1).unwrap();
        a.vmovups(ymmword_ptr(rax + 64), ymm2).unwrap();
        a.vmovups(ymmword_ptr(rax + 96), ymm3).unwrap();

        a.add(r8, 128).unwrap(); a.jmp(mj_block).unwrap();
        a.set_label(&mut mj_block_exit).unwrap();

        // ── Remainder: 8-wide YMM blocks ──
        a.mov(rax, rbx).unwrap(); a.sub(rax, 32).unwrap();
        let mut mj_8 = a.create_label();
        let mut mj_8_exit = a.create_label();
        a.set_label(&mut mj_8).unwrap();
        a.cmp(r8, rax).unwrap(); a.jge(mj_8_exit).unwrap();

        a.vxorps(ymm0, ymm0, ymm0).unwrap();
        a.xor(r9, r9).unwrap();
        let mut mk8_loop = a.create_label();
        let mut mk8_exit = a.create_label();
        a.set_label(&mut mk8_loop).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(mk8_exit).unwrap();
        a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
        a.vbroadcastss(ymm4, dword_ptr(rax)).unwrap();
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r13).unwrap();
        a.vmovups(ymm5, ymmword_ptr(rax)).unwrap();
        a.vfmadd231ps(ymm0, ymm4, ymm5).unwrap();
        a.add(r9, 1).unwrap(); a.jmp(mk8_loop).unwrap();
        a.set_label(&mut mk8_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm0).unwrap();

        a.add(r8, 32).unwrap(); a.jmp(mj_8).unwrap();
        a.set_label(&mut mj_8_exit).unwrap();

        // ── Scalar tail ──
        let mut mj_sc = a.create_label();
        let mut mj_sc_exit = a.create_label();
        a.set_label(&mut mj_sc).unwrap();
        a.cmp(r8, rbx).unwrap(); a.jge(mj_sc_exit).unwrap();
        a.xorps(xmm0, xmm0).unwrap();
        a.xor(r9, r9).unwrap();
        let mut mks_loop = a.create_label();
        let mut mks_exit = a.create_label();
        a.set_label(&mut mks_loop).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(mks_exit).unwrap();
        a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r13).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();
        a.mulss(xmm2, xmm1).unwrap();
        a.addss(xmm0, xmm2).unwrap();
        a.add(r9, 1).unwrap(); a.jmp(mks_loop).unwrap();
        a.set_label(&mut mks_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm0).unwrap();
        a.add(r8, 4).unwrap(); a.jmp(mj_sc).unwrap();
        a.set_label(&mut mj_sc_exit).unwrap();

        // i++
        a.add(rdi, 1).unwrap(); a.jmp(mi_loop).unwrap();
        a.set_label(&mut mi_exit).unwrap();

        a.xor(rax, rax).unwrap();
        a.vzeroupper().unwrap();
        a.pop(r15).unwrap(); a.pop(r14).unwrap(); a.pop(r13).unwrap();
        a.pop(r12).unwrap(); a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile AVX-512 multi-stream interleaved matmul (Rule Y + K+O + B+U + W)
    pub fn compile_matmul_avx512(m: usize, n: usize, k: usize) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();

        // Prologue
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        // Stack alignment fix: after PUSH RBP, RSP ≡ 8 mod 16.
        // AND RSP, -16 aligns to 16-byte boundary for AVX-512 requirements.
        a.and(rsp, -16).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();
        a.sub(rsp, 64).unwrap();

        // Rule B+U: Bake constants
        a.mov(r12, rdi).unwrap(); // R12 = A
        a.mov(r13, rsi).unwrap(); // R13 = B
        a.mov(r14, rdx).unwrap(); // R14 = C
        let n4 = (n as u64).wrapping_mul(4);
        a.mov(r15, n4).unwrap();  // R15 = N*4

        // Zero C using AVX-512
        a.vxorps(zmm0, zmm0, zmm0).unwrap();
        let total_bytes = (m as u64) * (n as u64) * 4;
        a.xor(rsi, rsi).unwrap();
        let mut zvec_loop = a.create_label();
        let mut zvec_exit = a.create_label();
        a.set_label(&mut zvec_loop).unwrap();
        a.mov(rax, total_bytes).unwrap(); a.sub(rax, 64).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(zvec_exit).unwrap();
        a.lea(rax, qword_ptr(r14 + rsi)).unwrap();
        a.vmovups(zmmword_ptr(rax), zmm0).unwrap();
        a.add(rsi, 64).unwrap(); a.jmp(zvec_loop).unwrap();
        a.set_label(&mut zvec_exit).unwrap();

        // Scalar zero tail
        a.xorps(xmm0, xmm0).unwrap();
        a.mov(rax, total_bytes).unwrap();
        let mut zsc_loop = a.create_label();
        let mut zsc_exit = a.create_label();
        a.set_label(&mut zsc_loop).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(zsc_exit).unwrap();
        a.lea(rbx, qword_ptr(r14 + rsi)).unwrap();
        a.movss(dword_ptr(rbx), xmm0).unwrap();
        a.add(rsi, 4).unwrap(); a.jmp(zsc_loop).unwrap();
        a.set_label(&mut zsc_exit).unwrap();

        // Bake M, K
        a.mov(r10d, m as u32).unwrap();
        a.mov(r11d, k as u32).unwrap();

        // i-loop
        a.xor(rdi, rdi).unwrap();
        let mut i_loop = a.create_label();
        let mut i_exit = a.create_label();
        a.set_label(&mut i_loop).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(i_exit).unwrap();

        // RDX = i*K*4
        a.mov(rdx, rdi).unwrap();
        a.imul_2(rdx, r11).unwrap(); // RDX = i * K
        a.shl(rdx, 2).unwrap();       // RDX = i * K * 4

        let n64 = n / 64;
        a.xor(r8, r8).unwrap(); // j byte offset

        // j_block loop
        if n64 > 0 {
            let mut j_block_loop = a.create_label();

            a.set_label(&mut j_block_loop).unwrap();
            a.vxorps(zmm0, zmm0, zmm0).unwrap();
            a.vxorps(zmm1, zmm1, zmm1).unwrap();
            a.vxorps(zmm2, zmm2, zmm2).unwrap();
            a.vxorps(zmm3, zmm3, zmm3).unwrap();

            // k-loop
            a.xor(r9, r9).unwrap();
            let mut k_loop = a.create_label();
            let mut k_exit = a.create_label();
            a.set_label(&mut k_loop).unwrap();
            a.cmp(r9, r11).unwrap(); a.jge(k_exit).unwrap();

            // vbroadcastss ZMM8, A[i,k]
            a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
            a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
            a.vbroadcastss(zmm8, dword_ptr(rax)).unwrap();

            // RBX = k*N*4
            a.mov(rbx, r9).unwrap(); a.imul_2(rbx, r15).unwrap();

            // Rule K+O: interleaved load + FMA
            a.lea(rax, qword_ptr(r13 + rbx)).unwrap(); a.add(rax, r8).unwrap();
            a.vmovups(zmm4, zmmword_ptr(rax)).unwrap();
            a.vmovups(zmm5, zmmword_ptr(rax + 64)).unwrap();
            a.vfmadd231ps(zmm0, zmm8, zmm4).unwrap();
            a.vmovups(zmm6, zmmword_ptr(rax + 128)).unwrap();
            a.vfmadd231ps(zmm1, zmm8, zmm5).unwrap();
            a.vmovups(zmm7, zmmword_ptr(rax + 192)).unwrap();
            a.vfmadd231ps(zmm2, zmm8, zmm6).unwrap();
            a.vfmadd231ps(zmm3, zmm8, zmm7).unwrap();

            a.add(r9, 1).unwrap(); a.jmp(k_loop).unwrap();
            a.set_label(&mut k_exit).unwrap();

            // Store accumulators
            a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap();
            a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
            a.vmovups(zmmword_ptr(rax), zmm0).unwrap();
            a.vmovups(zmmword_ptr(rax + 64), zmm1).unwrap();
            a.vmovups(zmmword_ptr(rax + 128), zmm2).unwrap();
            a.vmovups(zmmword_ptr(rax + 192), zmm3).unwrap();

            a.add(r8, 256).unwrap();
            a.cmp(r8, ((n64 * 256) as u32) as i32).unwrap();
            a.jl(j_block_loop).unwrap();
        }

        // Remainder: 16-float ZMM blocks
        let n16_rem = (n % 64) / 16;
        if n16_rem > 0 {
            for j16 in 0..n16_rem {
                let j_offset = (n64 * 64 + j16 * 16) * 4;
                a.vxorps(zmm0, zmm0, zmm0).unwrap();
                a.xor(r9, r9).unwrap();
                let mut k16_loop = a.create_label();
                let mut k16_exit = a.create_label();
                a.set_label(&mut k16_loop).unwrap();
                a.cmp(r9, r11).unwrap(); a.jge(k16_exit).unwrap();
                a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
                a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
                a.vbroadcastss(zmm8, dword_ptr(rax)).unwrap();
                a.mov(rbx, r9).unwrap(); a.imul_2(rbx, r15).unwrap();
                a.add(rbx, j_offset as i32).unwrap(); a.add(rbx, r13).unwrap();
                a.vmovups(zmm4, zmmword_ptr(rbx)).unwrap();
                a.vfmadd231ps(zmm0, zmm8, zmm4).unwrap();
                a.add(r9, 1).unwrap(); a.jmp(k16_loop).unwrap();
                a.set_label(&mut k16_exit).unwrap();
                a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap();
                a.add(rax, j_offset as i32).unwrap(); a.add(rax, r14).unwrap();
                a.vmovups(zmmword_ptr(rax), zmm0).unwrap();
            }
        }

        // Scalar remainder
        let n_scalar_start = n64 * 64 + n16_rem * 16;
        let n_scalar = n % 16;
        if n_scalar > 0 {
            let j_scalar_byte = n_scalar_start * 4;
            a.xor(r8, r8).unwrap();
            a.mov(ecx, n_scalar as u32).unwrap();
            let mut scalar_j_loop = a.create_label();
            let mut scalar_j_exit = a.create_label();
            a.set_label(&mut scalar_j_loop).unwrap();
            a.cmp(r8d, ecx).unwrap(); a.jge(scalar_j_exit).unwrap();
            a.xorps(xmm0, xmm0).unwrap();
            a.xor(r9, r9).unwrap();
            let mut sk_loop = a.create_label();
            let mut sk_exit = a.create_label();
            a.set_label(&mut sk_loop).unwrap();
            a.cmp(r9, r11).unwrap(); a.jge(sk_exit).unwrap();
            a.mov(rax, r9).unwrap(); a.shl(rax, 2).unwrap();
            a.add(rax, rdx).unwrap(); a.add(rax, r12).unwrap();
            a.movss(xmm1, dword_ptr(rax)).unwrap();
            a.mov(rbx, r9).unwrap(); a.imul_2(rbx, r15).unwrap(); // RBX = k*N*4
            a.add(rbx, j_scalar_byte as i32).unwrap();              // RBX += n_scalar_start*4
            a.mov(rax, r8).unwrap(); a.shl(rax, 2).unwrap();       // RAX = j_scalar_index*4
            a.add(rbx, rax).unwrap();                                // RBX += j_scalar_index*4
            a.add(rbx, r13).unwrap();                                // RBX += B_base
            a.movss(xmm2, dword_ptr(rbx)).unwrap();
            a.mulss(xmm2, xmm1).unwrap();
            a.addss(xmm0, xmm2).unwrap();
            a.add(r9, 1).unwrap(); a.jmp(sk_loop).unwrap();
            a.set_label(&mut sk_exit).unwrap();
            a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); // RAX = i*N*4
            a.add(rax, j_scalar_byte as i32).unwrap();                // RAX += n_scalar_start*4
            a.mov(rbx, r8).unwrap(); a.shl(rbx, 2).unwrap();         // RBX = j_scalar_index*4
            a.add(rax, rbx).unwrap();                                  // RAX += j_scalar_index*4
            a.add(rax, r14).unwrap();                                  // RAX += C_base
            a.movss(dword_ptr(rax), xmm0).unwrap();
            a.add(r8, 1).unwrap(); a.jmp(scalar_j_loop).unwrap();
            a.set_label(&mut scalar_j_exit).unwrap();
        }

        // i++
        a.add(rdi, 1).unwrap(); a.jmp(i_loop).unwrap();
        a.set_label(&mut i_exit).unwrap();

        a.vzeroall().unwrap();
        a.add(rsp, 64).unwrap();
        a.xor(rax, rax).unwrap();
        a.pop(r15).unwrap(); a.pop(r14).unwrap(); a.pop(r13).unwrap();
        a.pop(r12).unwrap(); a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, m, n, k)
    }

    /// Compile fused MatMul + Bias + ReLU
    pub fn compile_fused_matmul_bias_relu() -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();
        // 7th arg (bias_ptr) at [rbp+16]
        a.mov(rax, qword_ptr(rbp + 16)).unwrap();
        a.mov(rbx, rax).unwrap(); // RBX = bias_ptr
        a.mov(r10, rcx).unwrap(); a.mov(r11, r9).unwrap();
        a.mov(r12, rdi).unwrap(); a.mov(r13, rsi).unwrap();
        a.mov(r14, rdx).unwrap(); a.mov(r15, r8).unwrap();

        // Zero C
        a.vxorps(ymm0, ymm0, ymm0).unwrap();
        a.mov(rdi, r10).unwrap(); a.imul_2(rdi, r15).unwrap(); a.xor(rsi, rsi).unwrap();
        let mut fz_loop = a.create_label();
        let mut fz_exit = a.create_label();
        a.set_label(&mut fz_loop).unwrap();
        a.mov(rax, rdi).unwrap(); a.shl(rax, 2).unwrap(); a.sub(rax, 32).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(fz_exit).unwrap();
        a.lea(rax, qword_ptr(r14 + rsi)).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm0).unwrap();
        a.add(rsi, 32).unwrap(); a.jmp(fz_loop).unwrap();
        a.set_label(&mut fz_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.shl(rax, 2).unwrap();
        let mut fzsc = a.create_label();
        let mut fzsc_exit = a.create_label();
        a.set_label(&mut fzsc).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(fzsc_exit).unwrap();
        a.lea(rcx, qword_ptr(r14 + rsi)).unwrap();
        a.movss(dword_ptr(rcx), xmm0).unwrap();
        a.add(rsi, 4).unwrap(); a.jmp(fzsc).unwrap();
        a.set_label(&mut fzsc_exit).unwrap();

        // Matmul
        a.xor(rdi, rdi).unwrap();
        let mut fi_loop = a.create_label();
        let mut fi_exit = a.create_label();
        a.set_label(&mut fi_loop).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(fi_exit).unwrap();
        a.xor(r9, r9).unwrap();
        let mut fk_loop = a.create_label();
        let mut fk_exit = a.create_label();
        a.set_label(&mut fk_loop).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(fk_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r11).unwrap(); a.add(rax, r9).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r12).unwrap();
        a.vbroadcastss(ymm0, dword_ptr(rax)).unwrap();
        a.xor(r8, r8).unwrap();
        a.mov(rcx, r15).unwrap(); a.shl(rcx, 2).unwrap();
        let mut fj_vec = a.create_label();
        let mut fj_vec_exit = a.create_label();
        a.set_label(&mut fj_vec).unwrap();
        a.mov(rax, rcx).unwrap(); a.sub(rax, 32).unwrap(); a.cmp(r8, rax).unwrap();
        a.jge(fj_vec_exit).unwrap();
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r13).unwrap();
        a.vmovups(ymm2, ymmword_ptr(rax)).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymm1, ymmword_ptr(rax)).unwrap();
        a.vfmadd231ps(ymm1, ymm0, ymm2).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm1).unwrap();
        a.add(r8, 32).unwrap(); a.jmp(fj_vec).unwrap();
        a.set_label(&mut fj_vec_exit).unwrap();
        // Scalar tail
        let mut fjsc = a.create_label();
        let mut fjsc_exit = a.create_label();
        a.set_label(&mut fjsc).unwrap();
        a.cmp(r8, rcx).unwrap(); a.jge(fjsc_exit).unwrap();
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r13).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();
        a.mulss(xmm2, xmm0).unwrap(); a.addss(xmm1, xmm2).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm1).unwrap();
        a.add(r8, 4).unwrap(); a.jmp(fjsc).unwrap();
        a.set_label(&mut fjsc_exit).unwrap();
        a.add(r9, 1).unwrap(); a.jmp(fk_loop).unwrap();
        a.set_label(&mut fk_exit).unwrap();

        // Bias + ReLU pass
        a.vxorps(ymm3, ymm3, ymm3).unwrap();
        a.xor(r8, r8).unwrap();
        a.mov(rcx, r15).unwrap(); a.shl(rcx, 2).unwrap();
        let mut fbj_vec = a.create_label();
        let mut fbj_exit = a.create_label();
        a.set_label(&mut fbj_vec).unwrap();
        a.mov(rax, rcx).unwrap(); a.sub(rax, 32).unwrap(); a.cmp(r8, rax).unwrap();
        a.jge(fbj_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymm1, ymmword_ptr(rax)).unwrap();
        a.lea(rax, qword_ptr(rbx + r8)).unwrap();
        a.vmovups(ymm2, ymmword_ptr(rax)).unwrap();
        a.vaddps(ymm1, ymm2, ymm1).unwrap();
        a.vmaxps(ymm1, ymm1, ymm3).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm1).unwrap();
        a.add(r8, 32).unwrap(); a.jmp(fbj_vec).unwrap();
        a.set_label(&mut fbj_exit).unwrap();
        // Scalar bias+ReLU tail
        let mut fbsc = a.create_label();
        let mut fbsc_exit = a.create_label();
        a.set_label(&mut fbsc).unwrap();
        a.cmp(r8, rcx).unwrap(); a.jge(fbsc_exit).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();
        a.lea(rax, qword_ptr(rbx + r8)).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();
        a.addss(xmm1, xmm2).unwrap();
        a.xorps(xmm2, xmm2).unwrap();
        a.maxss(xmm1, xmm2).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm1).unwrap();
        a.add(r8, 4).unwrap(); a.jmp(fbsc).unwrap();
        a.set_label(&mut fbsc_exit).unwrap();

        a.add(rdi, 1).unwrap(); a.jmp(fi_loop).unwrap();
        a.set_label(&mut fi_exit).unwrap();
        a.xor(rax, rax).unwrap();
        a.vzeroupper().unwrap();
        a.pop(r15).unwrap(); a.pop(r14).unwrap(); a.pop(r13).unwrap();
        a.pop(r12).unwrap(); a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    pub fn compile_matmul_best() -> Self {
        match detect_isa_level() {
            ISALevel::AVX512 => Self::compile_matmul_avx512(128, 128, 128),
            ISALevel::AVX2 => Self::compile_matmul_avx2(),
            ISALevel::SSE => Self::compile_matmul(),
        }
    }

    pub fn compile_matmul_avx512_sized(m: usize, n: usize, k: usize) -> Self {
        if m == 0 || n == 0 || k == 0 { return Self::compile_matmul_avx2(); }
        match detect_isa_level() {
            ISALevel::AVX512 => Self::compile_matmul_avx512(m, n, k),
            ISALevel::AVX2 => Self::compile_matmul_avx2(),
            ISALevel::SSE => Self::compile_matmul(),
        }
    }

    /// Compile vectorized elementwise kernel with AVX2 (f32, add/sub/mul/div/min/max)
    /// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    pub fn compile_elementwise_avx2(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap();
        a.mov(r10, rdi).unwrap(); a.mov(r11, rsi).unwrap();
        a.mov(r12, rdx).unwrap(); a.mov(rbx, rcx).unwrap(); // RBX = n

        // Vectorized loop: RCX = byte offset
        a.xor(rcx, rcx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap(); a.sub(rax, 32).unwrap();
        let mut vec_loop = a.create_label();
        let mut vec_exit = a.create_label();
        a.set_label(&mut vec_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(vec_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.vmovups(ymm0, ymmword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.vmovups(ymm1, ymmword_ptr(rdx)).unwrap();
        // Note: ymm0 = a (from r11), ymm1 = b (from r12)
        // AVX2 3-operand: dst = src1 op src2, so dst=a, src1=a, src2=b
        match op {
            0 => { a.vaddps(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vsubps(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vmulps(ymm0, ymm0, ymm1).unwrap(); }
            3 => { a.vdivps(ymm0, ymm0, ymm1).unwrap(); }
            4 => { a.vminps(ymm0, ymm0, ymm1).unwrap(); }
            5 => { a.vmaxps(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.vmovups(ymmword_ptr(rdx), ymm0).unwrap();
        a.add(rcx, 32).unwrap(); a.jmp(vec_loop).unwrap();
        a.set_label(&mut vec_exit).unwrap();

        // Scalar tail: RCX still byte offset
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap();
        let mut sc_loop = a.create_label();
        let mut sc_exit = a.create_label();
        a.set_label(&mut sc_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(sc_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.movss(xmm0, dword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.movss(xmm1, dword_ptr(rdx)).unwrap();
        match op {
            0 => { a.addss(xmm0, xmm1).unwrap(); }
            1 => { a.subss(xmm0, xmm1).unwrap(); }
            2 => { a.mulss(xmm0, xmm1).unwrap(); }
            3 => { a.divss(xmm0, xmm1).unwrap(); }
            4 => { a.minss(xmm0, xmm1).unwrap(); }
            5 => { a.maxss(xmm0, xmm1).unwrap(); }
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.movss(dword_ptr(rdx), xmm0).unwrap();
        a.add(rcx, 4).unwrap(); a.jmp(sc_loop).unwrap();
        a.set_label(&mut sc_exit).unwrap();

        a.vzeroupper().unwrap();
        a.xor(rax, rax).unwrap(); a.pop(r12).unwrap(); a.pop(rbx).unwrap();
        a.pop(rbp).unwrap(); a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile SSE2 scalar elementwise kernel for f64 (add/sub/mul/div/min/max)
    /// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    /// Calling convention: rdi=dst, rsi=a, rdx=b, rcx=n
    pub fn compile_elementwise_f64(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap();
        a.mov(r10, rdi).unwrap(); a.mov(r11, rsi).unwrap();
        a.mov(r12, rdx).unwrap(); a.mov(rbx, rcx).unwrap(); // RBX = n

        // RCX = byte offset (8 bytes per f64), RAX = n*8 byte limit
        a.xor(rcx, rcx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); // n*8
        let mut lp = a.create_label();
        let mut lp_out = a.create_label();
        a.set_label(&mut lp).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(lp_out).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.movsd_2(xmm0, qword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.movsd_2(xmm1, qword_ptr(rdx)).unwrap();
        match op {
            0 => { a.addsd(xmm0, xmm1).unwrap(); }
            1 => { a.subsd(xmm0, xmm1).unwrap(); }
            2 => { a.mulsd(xmm0, xmm1).unwrap(); }
            3 => { a.divsd(xmm0, xmm1).unwrap(); }
            4 => { a.minsd(xmm0, xmm1).unwrap(); }  // scalar f64 min
            5 => { a.maxsd(xmm0, xmm1).unwrap(); }  // scalar f64 max
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.movsd_2(qword_ptr(rdx), xmm0).unwrap();
        a.add(rcx, 8).unwrap(); a.jmp(lp).unwrap();
        a.set_label(&mut lp_out).unwrap();
        a.xor(rax, rax).unwrap(); a.pop(r12).unwrap(); a.pop(rbx).unwrap();
        a.pop(rbp).unwrap(); a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile AVX2 vectorized elementwise kernel for f64 (add/sub/mul/div/min/max)
    /// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    /// Processes 4 f64 per vector iteration.
    /// Calling convention: rdi=dst, rsi=a, rdx=b, rcx=n
    pub fn compile_elementwise_avx2_f64(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap();
        a.mov(r10, rdi).unwrap(); a.mov(r11, rsi).unwrap();
        a.mov(r12, rdx).unwrap(); a.mov(rbx, rcx).unwrap(); // RBX = n

        // Vectorized loop: RCX = byte offset
        a.xor(rcx, rcx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); a.sub(rax, 32).unwrap(); // n*8 - 32
        let mut vec_loop = a.create_label();
        let mut vec_exit = a.create_label();
        a.set_label(&mut vec_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(vec_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.vmovupd(ymm0, ymmword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.vmovupd(ymm1, ymmword_ptr(rdx)).unwrap();
        match op {
            0 => { a.vaddpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = a + b
            1 => { a.vsubpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = a - b
            2 => { a.vmulpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = a * b
            3 => { a.vdivpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = a / b
            4 => { a.vminpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = min(a, b)
            5 => { a.vmaxpd(ymm0, ymm0, ymm1).unwrap(); }  // dst = max(a, b)
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.vmovupd(ymmword_ptr(rdx), ymm0).unwrap();
        a.add(rcx, 32).unwrap(); a.jmp(vec_loop).unwrap();
        a.set_label(&mut vec_exit).unwrap();

        // Scalar tail: RCX still byte offset
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); // n*8
        let mut sc_loop = a.create_label();
        let mut sc_exit = a.create_label();
        a.set_label(&mut sc_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(sc_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.movsd_2(xmm0, qword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.movsd_2(xmm1, qword_ptr(rdx)).unwrap();
        match op {
            0 => { a.addsd(xmm0, xmm1).unwrap(); }  // a + b
            1 => { a.subsd(xmm0, xmm1).unwrap(); }  // a - b
            2 => { a.mulsd(xmm0, xmm1).unwrap(); }  // a * b
            3 => { a.divsd(xmm0, xmm1).unwrap(); }  // a / b
            4 => { a.minsd(xmm0, xmm1).unwrap(); }  // scalar f64 min
            5 => { a.maxsd(xmm0, xmm1).unwrap(); }  // scalar f64 max
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.movsd_2(qword_ptr(rdx), xmm0).unwrap();
        a.add(rcx, 8).unwrap(); a.jmp(sc_loop).unwrap();
        a.set_label(&mut sc_exit).unwrap();

        a.vzeroupper().unwrap();
        a.xor(rax, rax).unwrap(); a.pop(r12).unwrap(); a.pop(rbx).unwrap();
        a.pop(rbp).unwrap(); a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile best available elementwise kernel for f64
    pub fn compile_elementwise_best_f64(op: u8) -> Self {
        match detect_isa_level() {
            ISALevel::AVX512 => Self::compile_elementwise_avx512_f64(op),
            ISALevel::AVX2 => Self::compile_elementwise_avx2_f64(op),
            ISALevel::SSE => Self::compile_elementwise_f64(op),
        }
    }

    /// Compile AVX-512 vectorized elementwise kernel for f64 (add/sub/mul/div/min/max)
    /// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    /// Processes 8 f64 per vector iteration using ZMM registers.
    /// Calling convention: rdi=dst, rsi=a, rdx=b, rcx=n
    pub fn compile_elementwise_avx512_f64(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap();
        a.mov(r10, rdi).unwrap(); a.mov(r11, rsi).unwrap();
        a.mov(r12, rdx).unwrap(); a.mov(rbx, rcx).unwrap(); // RBX = n

        // Vectorized loop: RCX = byte offset
        a.xor(rcx, rcx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); a.sub(rax, 64).unwrap(); // n*8 - 64
        let mut vec_loop = a.create_label();
        let mut vec_exit = a.create_label();
        a.set_label(&mut vec_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(vec_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.vmovupd(zmm0, zmmword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.vmovupd(zmm1, zmmword_ptr(rdx)).unwrap();
        match op {
            0 => { a.vaddpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = a + b
            1 => { a.vsubpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = a - b
            2 => { a.vmulpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = a * b
            3 => { a.vdivpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = a / b
            4 => { a.vminpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = min(a, b)
            5 => { a.vmaxpd(zmm0, zmm0, zmm1).unwrap(); }  // dst = max(a, b)
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.vmovupd(zmmword_ptr(rdx), zmm0).unwrap();
        a.add(rcx, 64).unwrap(); a.jmp(vec_loop).unwrap();
        a.set_label(&mut vec_exit).unwrap();

        // Scalar tail: RCX still byte offset
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); // n*8
        let mut sc_loop = a.create_label();
        let mut sc_exit = a.create_label();
        a.set_label(&mut sc_loop).unwrap();
        a.cmp(rcx, rax).unwrap(); a.jge(sc_exit).unwrap();
        a.lea(rdx, qword_ptr(r11 + rcx)).unwrap();
        a.movsd_2(xmm0, qword_ptr(rdx)).unwrap();
        a.lea(rdx, qword_ptr(r12 + rcx)).unwrap();
        a.movsd_2(xmm1, qword_ptr(rdx)).unwrap();
        match op {
            0 => { a.addsd(xmm0, xmm1).unwrap(); }
            1 => { a.subsd(xmm0, xmm1).unwrap(); }
            2 => { a.mulsd(xmm0, xmm1).unwrap(); }
            3 => { a.divsd(xmm0, xmm1).unwrap(); }
            4 => { a.minsd(xmm0, xmm1).unwrap(); }
            5 => { a.maxsd(xmm0, xmm1).unwrap(); }
            _ => {}
        }
        a.lea(rdx, qword_ptr(r10 + rcx)).unwrap();
        a.movsd_2(qword_ptr(rdx), xmm0).unwrap();
        a.add(rcx, 8).unwrap(); a.jmp(sc_loop).unwrap();
        a.set_label(&mut sc_exit).unwrap();

        a.vzeroupper().unwrap();
        a.xor(rax, rax).unwrap(); a.pop(r12).unwrap(); a.pop(rbx).unwrap();
        a.pop(rbp).unwrap(); a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Execute elementwise f64 kernel: dst[i] = a[i] op b[i]
    pub fn exec_elementwise_f64(&self, dst: &mut [f64], a: &[f64], b: &[f64], n: i64) -> i64 {
        if self.exec_ptr.is_null() { return -1; }
        unsafe {
            let f: extern "C" fn(*mut f64, *const f64, *const f64, i64) -> i64 =
                std::mem::transmute(self.exec_ptr);
            f(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n)
        }
    }

    /// Compile best available elementwise kernel for f32
    pub fn compile_elementwise_best_f32(op: u8) -> Self {
        match detect_isa_level() {
            ISALevel::AVX2 | ISALevel::AVX512 => Self::compile_elementwise_avx2(op),
            ISALevel::SSE => Self::compile_elementwise(op),
        }
    }
}

// ── SIMD Elementwise Executor (cached kernels) ──────────────────────────────

use std::sync::Mutex;

/// Global cache for f64 elementwise kernels. Key = op byte (0-5).
static F64_ELEM_KERNELS: Mutex<Vec<Option<CompiledKernel>>> = Mutex::new(Vec::new());

/// Global cache for f32 elementwise kernels. Key = op byte (0-5).
static F32_ELEM_KERNELS: Mutex<Vec<Option<CompiledKernel>>> = Mutex::new(Vec::new());

/// Execute a cached f64 elementwise kernel over arrays.
/// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
/// Returns the elapsed time in seconds, or -1.0 on error.
pub fn simd_elementwise_f64(op: u8, dst_ptr: usize, a_ptr: usize, b_ptr: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }

    // Get or compile the kernel
    let mut kernels = F64_ELEM_KERNELS.lock().unwrap();
    let idx = op as usize;
    while kernels.len() <= idx {
        kernels.push(None);
    }

    if kernels[idx].is_none() {
        let kernel = CompiledKernel::compile_elementwise_best_f64(op);
        kernels[idx] = Some(kernel);
    }

    let kernel = kernels[idx].as_ref().unwrap();
    if kernel.exec_ptr.is_null() {
        return -1.0;
    }

    unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut f64, n);
        let a = std::slice::from_raw_parts(a_ptr as *const f64, n);
        let b = std::slice::from_raw_parts(b_ptr as *const f64, n);
        let start = std::time::Instant::now();
        kernel.exec_elementwise_f64(dst, a, b, n as i64);
        start.elapsed().as_secs_f64()
    }
}

/// Execute a cached f32 elementwise kernel over arrays.
/// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
/// Returns the elapsed time in seconds, or -1.0 on error.
pub fn simd_elementwise_f32(op: u8, dst_ptr: usize, a_ptr: usize, b_ptr: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }

    // Get or compile the kernel
    let mut kernels = F32_ELEM_KERNELS.lock().unwrap();
    let idx = op as usize;
    while kernels.len() <= idx {
        kernels.push(None);
    }

    if kernels[idx].is_none() {
        let kernel = CompiledKernel::compile_elementwise_best_f32(op);
        kernels[idx] = Some(kernel);
    }

    let kernel = kernels[idx].as_ref().unwrap();
    if kernel.exec_ptr.is_null() {
        return -1.0;
    }

    unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, n);
        let a = std::slice::from_raw_parts(a_ptr as *const f32, n);
        let b = std::slice::from_raw_parts(b_ptr as *const f32, n);
        let start = std::time::Instant::now();
        kernel.exec_elementwise(dst, a, b, n as i64);
        start.elapsed().as_secs_f64()
    }
}

// ── SIMD Reduction Kernels ───────────────────────────────────────────────────
//
// Reduction operations (sum, max, min) over a single input array, producing
// a scalar result.  These use unrolled YMM accumulators to saturate the
// adder ports, then a horizontal reduction phase to produce the final scalar.
//
// op: 0=sum, 1=max, 2=min
// Calling convention: rdi=data_ptr, rsi=n  →  returns f64 in xmm0

/// Global caches for reduction kernels
static F32_REDUCE_KERNELS: Mutex<Vec<Option<CompiledKernel>>> = Mutex::new(Vec::new());
static F64_REDUCE_KERNELS: Mutex<Vec<Option<CompiledKernel>>> = Mutex::new(Vec::new());

impl CompiledKernel {
    /// Compile AVX2 reduction kernel for f32.
    /// Uses 8 YMM accumulators for 8-wide unrolling (256 f32 per iteration).
    /// op: 0=sum, 1=max, 2=min
    pub fn compile_reduce_avx2_f32(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap();

        // rdi = data ptr, rsi = n
        a.mov(rbx, rsi).unwrap(); // rbx = n

        // Initialize 8 YMM accumulators
        // For max/min, use ±∞ so that any finite value wins over the initial.
        // For sum, zero is the correct identity.
        match op {
            0 => {
                // sum: zero-init
                a.vxorps(ymm0, ymm0, ymm0).unwrap();
                a.vxorps(ymm1, ymm1, ymm1).unwrap();
                a.vxorps(ymm2, ymm2, ymm2).unwrap();
                a.vxorps(ymm3, ymm3, ymm3).unwrap();
                a.vxorps(ymm4, ymm4, ymm4).unwrap();
                a.vxorps(ymm5, ymm5, ymm5).unwrap();
                a.vxorps(ymm6, ymm6, ymm6).unwrap();
                a.vxorps(ymm7, ymm7, ymm7).unwrap();
            }
            1 => {
                // max: init to -∞
                let neg_inf_bits: u32 = f32::NEG_INFINITY.to_bits();
                // Load -∞ into xmm0, then broadcast to all YMMs
                a.mov(eax, neg_inf_bits as u32).unwrap();
                a.vmovd(xmm0, eax).unwrap();
                a.vbroadcastss(ymm0, xmm0).unwrap();
                a.vmovapd(ymm1, ymm0).unwrap();
                a.vmovapd(ymm2, ymm0).unwrap();
                a.vmovapd(ymm3, ymm0).unwrap();
                a.vmovapd(ymm4, ymm0).unwrap();
                a.vmovapd(ymm5, ymm0).unwrap();
                a.vmovapd(ymm6, ymm0).unwrap();
                a.vmovapd(ymm7, ymm0).unwrap();
            }
            2 => {
                // min: init to +∞
                let pos_inf_bits: u32 = f32::INFINITY.to_bits();
                a.mov(eax, pos_inf_bits as u32).unwrap();
                a.vmovd(xmm0, eax).unwrap();
                a.vbroadcastss(ymm0, xmm0).unwrap();
                a.vmovapd(ymm1, ymm0).unwrap();
                a.vmovapd(ymm2, ymm0).unwrap();
                a.vmovapd(ymm3, ymm0).unwrap();
                a.vmovapd(ymm4, ymm0).unwrap();
                a.vmovapd(ymm5, ymm0).unwrap();
                a.vmovapd(ymm6, ymm0).unwrap();
                a.vmovapd(ymm7, ymm0).unwrap();
            }
            _ => {
                a.vxorps(ymm0, ymm0, ymm0).unwrap();
                a.vxorps(ymm1, ymm1, ymm1).unwrap();
                a.vxorps(ymm2, ymm2, ymm2).unwrap();
                a.vxorps(ymm3, ymm3, ymm3).unwrap();
                a.vxorps(ymm4, ymm4, ymm4).unwrap();
                a.vxorps(ymm5, ymm5, ymm5).unwrap();
                a.vxorps(ymm6, ymm6, ymm6).unwrap();
                a.vxorps(ymm7, ymm7, ymm7).unwrap();
            }
        }

        // Vectorized loop: process 64 f32 (8 × YMM) per iteration
        // rdx = byte offset
        a.xor(rdx, rdx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap(); a.sub(rax, 256).unwrap(); // n*4 - 256
        let mut vec_loop = a.create_label();
        let mut vec_exit = a.create_label();
        a.set_label(&mut vec_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jg(vec_exit).unwrap(); // jg (not jge) to include last full vector

        // Load 8 × 8 = 64 f32 per iteration (256 bytes), accumulate into 8 YMM regs
        match op {
            0 => { // sum: VADDPS accumulator
                a.vaddps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vaddps(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vaddps(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vaddps(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
                a.vaddps(ymm4, ymm4, ymmword_ptr(rdi + rdx + 128)).unwrap();
                a.vaddps(ymm5, ymm5, ymmword_ptr(rdi + rdx + 160)).unwrap();
                a.vaddps(ymm6, ymm6, ymmword_ptr(rdi + rdx + 192)).unwrap();
                a.vaddps(ymm7, ymm7, ymmword_ptr(rdi + rdx + 224)).unwrap();
            }
            1 => { // max: VMAXPS accumulator
                a.vmaxps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vmaxps(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vmaxps(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vmaxps(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
                a.vmaxps(ymm4, ymm4, ymmword_ptr(rdi + rdx + 128)).unwrap();
                a.vmaxps(ymm5, ymm5, ymmword_ptr(rdi + rdx + 160)).unwrap();
                a.vmaxps(ymm6, ymm6, ymmword_ptr(rdi + rdx + 192)).unwrap();
                a.vmaxps(ymm7, ymm7, ymmword_ptr(rdi + rdx + 224)).unwrap();
            }
            2 => { // min: VMINPS accumulator
                a.vminps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vminps(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vminps(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vminps(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
                a.vminps(ymm4, ymm4, ymmword_ptr(rdi + rdx + 128)).unwrap();
                a.vminps(ymm5, ymm5, ymmword_ptr(rdi + rdx + 160)).unwrap();
                a.vminps(ymm6, ymm6, ymmword_ptr(rdi + rdx + 192)).unwrap();
                a.vminps(ymm7, ymm7, ymmword_ptr(rdi + rdx + 224)).unwrap();
            }
            _ => {}
        }
        a.add(rdx, 256).unwrap(); // 8 × 32 bytes
        a.jmp(vec_loop).unwrap();
        a.set_label(&mut vec_exit).unwrap();

        // Remaining vector elements: process one YMM at a time
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap(); a.sub(rax, 32).unwrap(); // n*4 - 32
        let mut rem_loop = a.create_label();
        let mut rem_exit = a.create_label();
        a.set_label(&mut rem_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jg(rem_exit).unwrap(); // jg (not jge) to include last full vector
        match op {
            0 => { a.vaddps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            1 => { a.vmaxps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            2 => { a.vminps(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            _ => {}
        }
        a.add(rdx, 32).unwrap();
        a.jmp(rem_loop).unwrap();
        a.set_label(&mut rem_exit).unwrap();

        // Horizontal reduction of the 8 accumulators into ymm0
        match op {
            0 => {
                a.vaddps(ymm0, ymm0, ymm1).unwrap();
                a.vaddps(ymm0, ymm0, ymm2).unwrap();
                a.vaddps(ymm0, ymm0, ymm3).unwrap();
                a.vaddps(ymm0, ymm0, ymm4).unwrap();
                a.vaddps(ymm0, ymm0, ymm5).unwrap();
                a.vaddps(ymm0, ymm0, ymm6).unwrap();
                a.vaddps(ymm0, ymm0, ymm7).unwrap();
            }
            1 => {
                a.vmaxps(ymm0, ymm0, ymm1).unwrap();
                a.vmaxps(ymm0, ymm0, ymm2).unwrap();
                a.vmaxps(ymm0, ymm0, ymm3).unwrap();
                a.vmaxps(ymm0, ymm0, ymm4).unwrap();
                a.vmaxps(ymm0, ymm0, ymm5).unwrap();
                a.vmaxps(ymm0, ymm0, ymm6).unwrap();
                a.vmaxps(ymm0, ymm0, ymm7).unwrap();
            }
            2 => {
                a.vminps(ymm0, ymm0, ymm1).unwrap();
                a.vminps(ymm0, ymm0, ymm2).unwrap();
                a.vminps(ymm0, ymm0, ymm3).unwrap();
                a.vminps(ymm0, ymm0, ymm4).unwrap();
                a.vminps(ymm0, ymm0, ymm5).unwrap();
                a.vminps(ymm0, ymm0, ymm6).unwrap();
                a.vminps(ymm0, ymm0, ymm7).unwrap();
            }
            _ => {}
        }

        // Horizontal reduction within ymm0/xmm0 (BEFORE scalar tail):
        // ymm0 = [s7, s6, s5, s4, s3, s2, s1, s0]
        a.vperm2f128(ymm1, ymm0, ymm0, 0x01).unwrap();
        match op {
            0 => { a.vaddps(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vmaxps(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vminps(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }
        a.vshufps(ymm1, ymm0, ymm0, 0x4E).unwrap();
        match op {
            0 => { a.vaddps(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vmaxps(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vminps(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }
        a.vshufps(ymm1, ymm0, ymm0, 0xB1).unwrap();
        match op {
            0 => { a.vaddps(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vmaxps(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vminps(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }

        // Scalar tail: handle remaining 0-7 f32 elements (AFTER horizontal reduction)
        a.mov(rax, rbx).unwrap(); a.shl(rax, 2).unwrap(); // n*4
        let mut sc_loop = a.create_label();
        let mut sc_exit = a.create_label();
        a.set_label(&mut sc_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jge(sc_exit).unwrap();
        match op {
            0 => { a.vaddss(xmm0, xmm0, dword_ptr(rdi + rdx)).unwrap(); }
            1 => { a.vmaxss(xmm0, xmm0, dword_ptr(rdi + rdx)).unwrap(); }
            2 => { a.vminss(xmm0, xmm0, dword_ptr(rdi + rdx)).unwrap(); }
            _ => {}
        }
        a.add(rdx, 4).unwrap();
        a.jmp(sc_loop).unwrap();
        a.set_label(&mut sc_exit).unwrap();

        // Convert f32 result to f64 for return in xmm0
        a.cvtss2sd(xmm0, xmm0).unwrap();

        a.vzeroupper().unwrap();
        a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Compile AVX2 reduction kernel for f64.
    /// Uses 4 YMM accumulators for 4-wide unrolling (16 f64 per iteration).
    /// op: 0=sum, 1=max, 2=min
    pub fn compile_reduce_avx2_f64(op: u8) -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap();

        // rdi = data ptr, rsi = n
        a.mov(rbx, rsi).unwrap(); // rbx = n

        // Zero-initialize 4 YMM accumulators
        match op {
            0 => {
                // sum: zero-init
                a.vxorpd(ymm0, ymm0, ymm0).unwrap();
                a.vxorpd(ymm1, ymm1, ymm1).unwrap();
                a.vxorpd(ymm2, ymm2, ymm2).unwrap();
                a.vxorpd(ymm3, ymm3, ymm3).unwrap();
            }
            1 => {
                // max: init to -∞
                let neg_inf_bits: u64 = f64::NEG_INFINITY.to_bits();
                a.mov(rax, neg_inf_bits).unwrap();
                a.vmovq(xmm0, rax).unwrap();
                a.vbroadcastsd(ymm0, xmm0).unwrap();
                a.vmovapd(ymm1, ymm0).unwrap();
                a.vmovapd(ymm2, ymm0).unwrap();
                a.vmovapd(ymm3, ymm0).unwrap();
            }
            2 => {
                // min: init to +∞
                let pos_inf_bits: u64 = f64::INFINITY.to_bits();
                a.mov(rax, pos_inf_bits).unwrap();
                a.vmovq(xmm0, rax).unwrap();
                a.vbroadcastsd(ymm0, xmm0).unwrap();
                a.vmovapd(ymm1, ymm0).unwrap();
                a.vmovapd(ymm2, ymm0).unwrap();
                a.vmovapd(ymm3, ymm0).unwrap();
            }
            _ => {
                a.vxorpd(ymm0, ymm0, ymm0).unwrap();
                a.vxorpd(ymm1, ymm1, ymm1).unwrap();
                a.vxorpd(ymm2, ymm2, ymm2).unwrap();
                a.vxorpd(ymm3, ymm3, ymm3).unwrap();
            }
        }

        // Vectorized loop: process 16 f64 (4 × YMM) per iteration
        // rdx = byte offset
        a.xor(rdx, rdx).unwrap();
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); a.sub(rax, 128).unwrap(); // n*8 - 128
        let mut vec_loop = a.create_label();
        let mut vec_exit = a.create_label();
        a.set_label(&mut vec_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jg(vec_exit).unwrap(); // jg (not jge) to include last full vector

        match op {
            0 => {
                a.vaddpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vaddpd(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vaddpd(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vaddpd(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
            }
            1 => {
                a.vmaxpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vmaxpd(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vmaxpd(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vmaxpd(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
            }
            2 => {
                a.vminpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap();
                a.vminpd(ymm1, ymm1, ymmword_ptr(rdi + rdx + 32)).unwrap();
                a.vminpd(ymm2, ymm2, ymmword_ptr(rdi + rdx + 64)).unwrap();
                a.vminpd(ymm3, ymm3, ymmword_ptr(rdi + rdx + 96)).unwrap();
            }
            _ => {}
        }
        a.add(rdx, 128).unwrap(); // 4 × 32 bytes
        a.jmp(vec_loop).unwrap();
        a.set_label(&mut vec_exit).unwrap();

        // Remaining vector elements: process one YMM at a time
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap(); a.sub(rax, 32).unwrap();
        let mut rem_loop = a.create_label();
        let mut rem_exit = a.create_label();
        a.set_label(&mut rem_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jg(rem_exit).unwrap(); // jg (not jge) to include last full vector
        match op {
            0 => { a.vaddpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            1 => { a.vmaxpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            2 => { a.vminpd(ymm0, ymm0, ymmword_ptr(rdi + rdx)).unwrap(); }
            _ => {}
        }
        a.add(rdx, 32).unwrap();
        a.jmp(rem_loop).unwrap();
        a.set_label(&mut rem_exit).unwrap();

        // Horizontal reduction of the 4 accumulators into ymm0
        match op {
            0 => {
                a.vaddpd(ymm0, ymm0, ymm1).unwrap();
                a.vaddpd(ymm0, ymm0, ymm2).unwrap();
                a.vaddpd(ymm0, ymm0, ymm3).unwrap();
            }
            1 => {
                a.vmaxpd(ymm0, ymm0, ymm1).unwrap();
                a.vmaxpd(ymm0, ymm0, ymm2).unwrap();
                a.vmaxpd(ymm0, ymm0, ymm3).unwrap();
            }
            2 => {
                a.vminpd(ymm0, ymm0, ymm1).unwrap();
                a.vminpd(ymm0, ymm0, ymm2).unwrap();
                a.vminpd(ymm0, ymm0, ymm3).unwrap();
            }
            _ => {}
        }

        // Horizontal reduction within ymm0 for f64 (BEFORE scalar tail):
        a.vperm2f128(ymm1, ymm0, ymm0, 0x01).unwrap();
        match op {
            0 => { a.vaddpd(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vmaxpd(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vminpd(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }
        a.vshufpd(ymm1, ymm0, ymm0, 0x05).unwrap();
        match op {
            0 => { a.vaddpd(ymm0, ymm0, ymm1).unwrap(); }
            1 => { a.vmaxpd(ymm0, ymm0, ymm1).unwrap(); }
            2 => { a.vminpd(ymm0, ymm0, ymm1).unwrap(); }
            _ => {}
        }

        // Scalar tail (AFTER horizontal reduction)
        a.mov(rax, rbx).unwrap(); a.shl(rax, 3).unwrap();
        let mut sc_loop = a.create_label();
        let mut sc_exit = a.create_label();
        a.set_label(&mut sc_loop).unwrap();
        a.cmp(rdx, rax).unwrap(); a.jge(sc_exit).unwrap();
        match op {
            0 => { a.vaddsd(xmm0, xmm0, qword_ptr(rdi + rdx)).unwrap(); }
            1 => { a.vmaxsd(xmm0, xmm0, qword_ptr(rdi + rdx)).unwrap(); }
            2 => { a.vminsd(xmm0, xmm0, qword_ptr(rdi + rdx)).unwrap(); }
            _ => {}
        }
        a.add(rdx, 8).unwrap();
        a.jmp(sc_loop).unwrap();
        a.set_label(&mut sc_exit).unwrap();

        // Result is already f64 in xmm0
        a.vzeroupper().unwrap();
        a.pop(rbx).unwrap(); a.pop(rbp).unwrap();
        a.ret().unwrap();

        Self::finalize(a, 0, 0, 0)
    }

    /// Execute a cached f32 reduction kernel.
    /// Returns the reduced scalar value.
    pub fn exec_reduce_f32(&self, data: &[f32], n: i64) -> f64 {
        if self.exec_ptr.is_null() { return f64::NAN; }
        unsafe {
            let f: extern "C" fn(*const f32, i64) -> f64 =
                std::mem::transmute(self.exec_ptr);
            f(data.as_ptr(), n)
        }
    }

    /// Execute a cached f64 reduction kernel.
    /// Returns the reduced scalar value.
    pub fn exec_reduce_f64(&self, data: &[f64], n: i64) -> f64 {
        if self.exec_ptr.is_null() { return f64::NAN; }
        unsafe {
            let f: extern "C" fn(*const f64, i64) -> f64 =
                std::mem::transmute(self.exec_ptr);
            f(data.as_ptr(), n)
        }
    }
}

/// Execute a cached f32 reduction kernel over an array.
/// op: 0=sum, 1=max, 2=min
/// Returns the reduced scalar value, or NaN on error.
pub fn simd_reduce_f32(op: u8, data_ptr: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }

    let mut kernels = F32_REDUCE_KERNELS.lock().unwrap();
    let idx = op as usize;
    while kernels.len() <= idx {
        kernels.push(None);
    }

    if kernels[idx].is_none() {
        let kernel = CompiledKernel::compile_reduce_avx2_f32(op);
        kernels[idx] = Some(kernel);
    }

    let kernel = kernels[idx].as_ref().unwrap();
    if kernel.exec_ptr.is_null() {
        return f64::NAN;
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr as *const f32, n);
        kernel.exec_reduce_f32(data, n as i64)
    }
}

/// Execute a cached f64 reduction kernel over an array.
/// op: 0=sum, 1=max, 2=min
/// Returns the reduced scalar value, or NaN on error.
pub fn simd_reduce_f64(op: u8, data_ptr: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }

    let mut kernels = F64_REDUCE_KERNELS.lock().unwrap();
    let idx = op as usize;
    while kernels.len() <= idx {
        kernels.push(None);
    }

    if kernels[idx].is_none() {
        let kernel = CompiledKernel::compile_reduce_avx2_f64(op);
        kernels[idx] = Some(kernel);
    }

    let kernel = kernels[idx].as_ref().unwrap();
    if kernel.exec_ptr.is_null() {
        return f64::NAN;
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr as *const f64, n);
        kernel.exec_reduce_f64(data, n as i64)
    }
}

// ── BLIS-Style Cache-Blocked Matmul ──────────────────────────────────────
//
// Implements a 5-loop BLIS-style algorithm with:
//   - AVX2 6×16 micro-kernel (12 YMM accumulators, 12 FMAs per k-step)
//   - B-matrix packing (contiguous access in micro-kernel, eliminates stride)
//   - A-matrix packing (column-major micro-panels for sequential broadcast)
//   - L1/L2/L3 cache blocking (MC=64, NC=64, KC=256)
//   - Rayon parallelism on the outermost (i2) loop
//
// Packed layouts (critical for micro-kernel performance):
//   A micro-panel [MR rows × kc cols]: column-major within panel
//     Layout: a[0,0] a[1,0] ... a[MR-1,0] a[0,1] a[1,1] ... a[MR-1,kc-1]
//     Stride between k-steps = MR  (micro-kernel does a_ptr += MR)
//
//   B micro-panel [kc rows × NR cols]: row-major within panel, padded to NR
//     Layout: b[0,0] b[0,1] ... b[0,NR-1] b[1,0] ... b[kc-1,NR-1]
//     Stride between k-steps = NR  (micro-kernel does b_ptr += NR)

/// Micro-kernel row dimension (6 rows accumulated in YMM registers)
const BLIS_MR: usize = 6;
/// Micro-kernel column dimension (2 YMMs wide = 16 f32)
const BLIS_NR: usize = 16;
/// L2 cache block: rows of A
const BLIS_MC: usize = 128;
/// L2 cache block: cols of B
const BLIS_NC: usize = 128;
/// L2 cache block: shared dimension
const BLIS_KC: usize = 128;

/// AVX2 micro-kernel for MR×NR rank-1 update.
///
/// Accumulates into C[i:i+mr, j:j+nr] += A_panel * B_panel
/// using 12 YMM accumulators (6 rows × 2 YMM columns = 12 YMMs).
///
/// A_packed layout (column-major within panel):
///   For each k step k, BLIS_MR consecutive values starting at a_packed + k*BLIS_MR:
///   a[0,k], a[1,k], ..., a[BLIS_MR-1,k]
///
/// B_packed layout (row-major within panel, b_stride between k-rows):
///   For each k step k, BLIS_NR consecutive values starting at b_packed + k*b_stride:
///   b[k,0], b[k,1], ..., b[k,BLIS_NR-1]
///   b_stride must be >= BLIS_NR (padded so loads are always valid).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn micro_kernel_6x16(
    a_packed: *const f32,
    b_packed: *const f32,
    b_stride: usize, // stride between k-rows in packed B
    c: *mut f32,
    ldc: usize,
    kc: usize,
    mr: usize, // actual mr (may be < BLIS_MR at edges)
    nr: usize, // actual nr (may be < BLIS_NR at edges)
) {
    use std::arch::x86_64::*;

    // 12 YMM accumulators: acc[row][col_block]
    // row 0: acc[0], acc[1]    row 1: acc[2], acc[3]    row 2: acc[4], acc[5]
    // row 3: acc[6], acc[7]    row 4: acc[8], acc[9]    row 5: acc[10], acc[11]
    let mut acc = [_mm256_setzero_ps(); 12];

    for p in 0..kc {
        // Load 2 YMM from B row p (16 f32 contiguous, padded so loads are always valid)
        let b_row = b_packed.add(p * b_stride);
        let b0 = _mm256_loadu_ps(b_row);
        let b1 = _mm256_loadu_ps(b_row.add(8));

        // Broadcast each A row value and do 12 FMAs
        // A is stored column-major: a_packed + p*BLIS_MR has all MR row values for this k
        let a_row = a_packed.add(p * BLIS_MR);
        for row in 0..BLIS_MR {
            let a_val = _mm256_broadcast_ss(&*a_row.add(row));
            acc[row * 2] = _mm256_fmadd_ps(a_val, b0, acc[row * 2]);
            acc[row * 2 + 1] = _mm256_fmadd_ps(a_val, b1, acc[row * 2 + 1]);
        }
    }

    // Store accumulators to C with stride ldc — ADD to existing C values
    // (not overwrite!) so that multiple k-blocks accumulate correctly.
    for row in 0..mr {
        let c_row = c.add(row * ldc);
        if nr >= 16 {
            let c0 = _mm256_loadu_ps(c_row);
            let c1 = _mm256_loadu_ps(c_row.add(8));
            _mm256_storeu_ps(c_row, _mm256_add_ps(c0, acc[row * 2]));
            _mm256_storeu_ps(c_row.add(8), _mm256_add_ps(c1, acc[row * 2 + 1]));
        } else if nr >= 8 {
            let c0 = _mm256_loadu_ps(c_row);
            _mm256_storeu_ps(c_row, _mm256_add_ps(c0, acc[row * 2]));
            if nr > 8 {
                // Partial second YMM: add only the valid elements
                let acc1_arr: [f32; 8] = std::mem::transmute(acc[row * 2 + 1]);
                for j in 8..nr {
                    *c_row.add(j) += acc1_arr[j - 8];
                }
            }
        } else {
            // nr < 8: add element by element from first YMM
            let acc0_arr: [f32; 8] = std::mem::transmute(acc[row * 2]);
            for j in 0..nr {
                *c_row.add(j) += acc0_arr[j];
            }
        }
    }
}

/// Core BLIS-style blocked matmul body (shared between serial and parallel).
/// Processes a single MC-row block starting at row i2.
/// Uses raw pointers for C to allow sharing across rayon threads safely.
fn blis_process_block(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    _m: usize,
    n: usize,
    k_dim: usize,  // renamed from `k` to avoid conflict with iced-x86's k2 register
    i2: usize,
    mc: usize,
    packed_b: &mut [f32],
    packed_a: &mut [f32],
) {
    let has_avx2 = is_x86_feature_detected!("avx2");

    // packed_b layout: [kc][BLIS_NC] with stride BLIS_NC (>= BLIS_NR, multiple of BLIS_NR)
    // BLIS_NC = 64 which is a multiple of BLIS_NR = 16, so loads at j1 offsets are safe
    let b_stride = BLIS_NC;

    for j2 in (0..n).step_by(BLIS_NC) {
        let nc = std::cmp::min(BLIS_NC, n - j2);

        for kk2 in (0..k_dim).step_by(BLIS_KC) {
            let kc = std::cmp::min(BLIS_KC, k_dim - kk2);

            // Pack B[kk2:kk2+kc, j2:j2+nc] with stride b_stride = BLIS_NC
            // Layout: [kc][BLIS_NC] row-major, zero-padded beyond nc
            for p in 0..kc {
                let k_idx = kk2 + p;
                let row_start = p * b_stride;
                // Copy actual data
                for jj in 0..nc {
                    packed_b[row_start + jj] = b[k_idx * n + j2 + jj];
                }
                // Zero-pad remainder (only if nc < BLIS_NC)
                if nc < BLIS_NC {
                    for jj in nc..BLIS_NC {
                        packed_b[row_start + jj] = 0.0f32;
                    }
                }
            }

            // Pack A[i2:i2+mc, kk2:kk2+kc] in column-major order within micro-panels
            // Layout within micro-panel (at offset i1*kc in packed_a):
            //   for each k step p: BLIS_MR consecutive values
            //   packed_a[i1*kc + p*BLIS_MR + row] = A[i2+i1+row, kk2+p]
            for i1 in (0..mc).step_by(BLIS_MR) {
                let mr = std::cmp::min(BLIS_MR, mc - i1);
                let panel_base = i1 * kc;
                for p in 0..kc {
                    let col_start = panel_base + p * BLIS_MR;
                    // Copy actual rows
                    for row in 0..mr {
                        let i_idx = i2 + i1 + row;
                        packed_a[col_start + row] = a[i_idx * k_dim + kk2 + p];
                    }
                    // Zero-pad remainder (only if mr < BLIS_MR)
                    if mr < BLIS_MR {
                        for row in mr..BLIS_MR {
                            packed_a[col_start + row] = 0.0f32;
                        }
                    }
                }
            }

            // Micro-kernel loops
            for i1 in (0..mc).step_by(BLIS_MR) {
                let mr = std::cmp::min(BLIS_MR, mc - i1);

                for j1 in (0..nc).step_by(BLIS_NR) {
                    let nr = std::cmp::min(BLIS_NR, nc - j1);

                    let a_panel_ptr = packed_a.as_ptr().wrapping_add(i1 * kc);
                    // B micro-panel starts at column j1 within the packed NC-wide panel
                    let b_panel_ptr = packed_b.as_ptr().wrapping_add(j1);
                    let c_row_ptr = unsafe { c_ptr.add((i2 + i1) * n + j2 + j1) };

                    if has_avx2 {
                        unsafe {
                            micro_kernel_6x16(
                                a_panel_ptr,
                                b_panel_ptr,
                                b_stride, // stride between k-rows in packed B
                                c_row_ptr,
                                n, // ldc
                                kc,
                                mr,
                                nr,
                            );
                        }
                    } else {
                        // Scalar fallback
                        for i in 0..mr {
                            for j in 0..nr {
                                let mut sum = 0.0f32;
                                for p in 0..kc {
                                    let a_val = packed_a[i1 * kc + p * BLIS_MR + i];
                                    let b_val = packed_b[p * b_stride + j1 + j];
                                    sum += a_val * b_val;
                                }
                                unsafe {
                                    *c_ptr.add((i2 + i1 + i) * n + j2 + j1 + j) += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// BLIS-style cache-blocked matmul (serial version).
pub fn cache_blocked_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 { return; }

    // Pre-allocate packing buffers (reused across loops)
    // A pack needs extra space for zero-padding: round up MC to next MR multiple
    let a_pack_rows = ((BLIS_MC + BLIS_MR - 1) / BLIS_MR) * BLIS_MR;
    let b_pack_size = BLIS_KC * BLIS_NC;  // B panel [kc][BLIS_NC]
    let a_pack_size = a_pack_rows * BLIS_KC;   // A panel (rounded mc × kc)
    let mut packed_b = vec![0.0f32; b_pack_size];
    let mut packed_a = vec![0.0f32; a_pack_size];

    // Zero C
    for val in c.iter_mut().take(m * n) {
        *val = 0.0f32;
    }

    let c_ptr = c.as_mut_ptr();

    // Process all MC-row blocks serially
    for i2 in (0..m).step_by(BLIS_MC) {
        let mc = std::cmp::min(BLIS_MC, m - i2);
        blis_process_block(a, b, c_ptr, m, n, k, i2, mc, &mut packed_b, &mut packed_a);
    }
}

/// Parallel version of cache_blocked_matmul using rayon.
/// Parallelizes over the outermost loop (MC-row blocks).
pub fn parallel_cache_blocked_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    if m == 0 || n == 0 || k == 0 { return; }

    // Zero C first
    for val in c.iter_mut().take(m * n) {
        *val = 0.0f32;
    }

    let num_blocks = (m + BLIS_MC - 1) / BLIS_MC;

    // Convert pointer to usize for safe Send+Sync capture across rayon threads.
    // Safety: each thread writes to disjoint rows of C (different i2 blocks).
    let c_addr = c.as_mut_ptr() as usize;

    // Process each MC-row block in parallel (each thread gets its own packing buffers)
    (0..num_blocks).into_par_iter().for_each(|block_idx| {
        let i2 = block_idx * BLIS_MC;
        let mc = std::cmp::min(BLIS_MC, m - i2);
        let c_ptr = c_addr as *mut f32;

        // Per-thread packing buffers
        let a_pack_rows = ((BLIS_MC + BLIS_MR - 1) / BLIS_MR) * BLIS_MR;
        let b_pack_size = BLIS_KC * BLIS_NC;
        let a_pack_size = a_pack_rows * BLIS_KC;
        let mut packed_b = vec![0.0f32; b_pack_size];
        let mut packed_a = vec![0.0f32; a_pack_size];

        blis_process_block(a, b, c_ptr, m, n, k, i2, mc, &mut packed_b, &mut packed_a);
    });
}

pub fn parallel_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    if m == 0 || n == 0 || k == 0 { return; }

    // Use cache-blocked matmul when AVX2 is available for significant speedup
    if is_x86_feature_detected!("avx2") {
        parallel_cache_blocked_matmul(a, b, c, m, n, k);
        return;
    }

    // Scalar fallback
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        c_row.fill(0.0f32);
        let a_row = &a[i * k..(i + 1) * k];
        for kk in 0..k {
            let a_ik = a_row[kk];
            let b_row = &b[kk * n..(kk + 1) * n];
            for j in 0..n { c_row[j] += a_ik * b_row[j]; }
        }
    });
}

pub fn jit_parallel_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 { return; }

    // Use JIT AVX-512 kernel for large matrices where it excels
    let isa = detect_isa_level();
    if isa == ISALevel::AVX512 && n >= 64 {
        let kernel = CompiledKernel::compile_matmul_avx512(m, n, k);
        if !kernel.exec_ptr.is_null() {
            let _ = kernel.exec_matmul(a, b, c, m as i64, n as i64, k as i64);
            return;
        }
    }

    // Use cache-blocked matmul when AVX2 is available
    if is_x86_feature_detected!("avx2") {
        parallel_cache_blocked_matmul(a, b, c, m, n, k);
        return;
    }

    parallel_matmul(a, b, c, m, n, k);
}

// ── Fused Multi-Op SIMD Elementwise + Reduce Kernels ─────────────────────────
//
// These kernels execute a chain of elementwise operations in a SINGLE PASS
// over memory, keeping intermediate results in SIMD registers. This eliminates
// the catastrophic multi-pass pattern where each op creates a temporary array
// and reads/writes the entire dataset again.
//
// For a chain like x * 2.0 + 1.0 → sum:
//   Old: read x(800MB), mul, write temp(800MB), read temp, add, write temp2,
//        read temp2, reduce = ~4.8GB traffic
//   New: read x(800MB), mul in regs, add in regs, accumulate = 800MB traffic
//
// Up to 8 ops can be fused (limited by YMM register count for intermediates).
// The kernel auto-selects AVX2 when available, falls back to scalar.

/// Fused elementwise operation descriptor.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FusedOpDesc {
    /// Operation: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    pub op: u8,
    /// Left operand source: 0=input_array, 1=constant, 2=result_of_previous_op
    pub lhs_src: u8,
    /// Index into the source (input array index, constant index, or previous op result index)
    pub lhs_idx: u16,
    /// Right operand source: same encoding as lhs_src
    pub rhs_src: u8,
    /// Index into the source
    pub rhs_idx: u16,
}

/// Maximum number of ops in a fused chain (limited by YMM register count)
pub const MAX_FUSED_OPS: usize = 8;

/// Reduce operation codes
const REDUCE_SUM: u8 = 0;
const REDUCE_MAX: u8 = 1;
const REDUCE_MIN: u8 = 2;
const REDUCE_NONE: u8 = 255;

// ── f32 AVX2 core ──
//
// Supports ARBITRARY-LENGTH op chains — not limited to MAX_FUSED_OPS.
// Intermediates are stored in a Vec<__m256> allocated once outside the
// element loop.  For short chains (≤ MAX_FUSED_OPS) the compiler keeps
// intermediates in YMM registers.  For long chains (e.g., 700 ops for
// Mandelbrot) the compiler spills to the stack but arithmetic stays
// vectorised (8 f32 per YMM), which is dramatically faster than the
// scalar Rust fallback.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fused_elem_f32_avx2_core(
    ops: &[FusedOpDesc],
    input_ptrs: &[*const f32],
    constants: &[f32],
    n: usize,
    reduce_op: u8,
    dst_ptr: *mut f32,
) -> f64 {
    use std::arch::x86_64::*;

    let num_ops = ops.len();
    if num_ops == 0 || n == 0 { return 0.0; }

    let has_reduce = reduce_op != REDUCE_NONE;
    let zero = _mm256_setzero_ps();

    // 4 YMM accumulators for 4x-unrolled reduce
    // For max/min, initialize to ±∞ so any finite value wins.
    let init_val: __m256 = match reduce_op {
        REDUCE_MAX => _mm256_set1_ps(f32::NEG_INFINITY),
        REDUCE_MIN => _mm256_set1_ps(f32::INFINITY),
        _ => zero,
    };
    let mut acc0 = init_val;
    let mut acc1 = init_val;
    let mut acc2 = init_val;
    let mut acc3 = init_val;

    // Allocate intermediates ONCE — reused across all elements.
    // For short chains this fits in registers; for long chains the
    // compiler spills to stack, but every load/store is still 32 bytes.
    let mut intermediates: Vec<__m256> = vec![zero; num_ops];

    // Process 32 f32 per iteration (4 × 8-wide YMM)
    let n_vec4 = n & !31;
    let mut i = 0usize;

    while i < n_vec4 {
        for u in 0..4usize {
            let base = i + u * 8;

            for (op_idx, desc) in ops.iter().enumerate() {
                let lhs = match desc.lhs_src {
                    0 => {
                        let idx = desc.lhs_idx as usize;
                        if idx < input_ptrs.len() {
                            _mm256_loadu_ps(input_ptrs[idx].add(base))
                        } else {
                            zero
                        }
                    }
                    1 => {
                        let idx = desc.lhs_idx as usize;
                        if idx < constants.len() {
                            _mm256_set1_ps(*constants.get_unchecked(idx))
                        } else {
                            zero
                        }
                    }
                    2 => {
                        let idx = desc.lhs_idx as usize;
                        *intermediates.get_unchecked(idx)
                    }
                    _ => zero,
                };
                let rhs = match desc.rhs_src {
                    0 => {
                        let idx = desc.rhs_idx as usize;
                        if idx < input_ptrs.len() {
                            _mm256_loadu_ps(input_ptrs[idx].add(base))
                        } else {
                            zero
                        }
                    }
                    1 => {
                        let idx = desc.rhs_idx as usize;
                        if idx < constants.len() {
                            _mm256_set1_ps(*constants.get_unchecked(idx))
                        } else {
                            zero
                        }
                    }
                    2 => {
                        let idx = desc.rhs_idx as usize;
                        *intermediates.get_unchecked(idx)
                    }
                    _ => zero,
                };

                intermediates[op_idx] = match desc.op {
                    0 => _mm256_add_ps(lhs, rhs),
                    1 => _mm256_sub_ps(lhs, rhs),
                    2 => _mm256_mul_ps(lhs, rhs),
                    3 => _mm256_div_ps(lhs, rhs),
                    4 => _mm256_min_ps(lhs, rhs),
                    5 => _mm256_max_ps(lhs, rhs),
                    _ => lhs,
                };
            }

            let final_val = intermediates[num_ops - 1];

            if has_reduce {
                match reduce_op {
                    REDUCE_SUM => match u {
                        0 => acc0 = _mm256_add_ps(acc0, final_val),
                        1 => acc1 = _mm256_add_ps(acc1, final_val),
                        2 => acc2 = _mm256_add_ps(acc2, final_val),
                        _ => acc3 = _mm256_add_ps(acc3, final_val),
                    },
                    REDUCE_MAX => match u {
                        0 => acc0 = _mm256_max_ps(acc0, final_val),
                        1 => acc1 = _mm256_max_ps(acc1, final_val),
                        2 => acc2 = _mm256_max_ps(acc2, final_val),
                        _ => acc3 = _mm256_max_ps(acc3, final_val),
                    },
                    REDUCE_MIN => match u {
                        0 => acc0 = _mm256_min_ps(acc0, final_val),
                        1 => acc1 = _mm256_min_ps(acc1, final_val),
                        2 => acc2 = _mm256_min_ps(acc2, final_val),
                        _ => acc3 = _mm256_min_ps(acc3, final_val),
                    },
                    _ => {}
                }
            } else {
                _mm256_storeu_ps(dst_ptr.add(base), final_val);
            }
        }
        i += 32;
    }

    // Process remaining 8-element chunks
    let n_vec = n & !7;
    while i < n_vec {
        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs = match desc.lhs_src {
                0 => {
                    let idx = desc.lhs_idx as usize;
                    if idx < input_ptrs.len() { _mm256_loadu_ps(input_ptrs[idx].add(i)) } else { zero }
                }
                1 => {
                    let idx = desc.lhs_idx as usize;
                    if idx < constants.len() { _mm256_set1_ps(*constants.get_unchecked(idx)) } else { zero }
                }
                2 => {
                    let idx = desc.lhs_idx as usize;
                    *intermediates.get_unchecked(idx)
                }
                _ => zero,
            };
            let rhs = match desc.rhs_src {
                0 => {
                    let idx = desc.rhs_idx as usize;
                    if idx < input_ptrs.len() { _mm256_loadu_ps(input_ptrs[idx].add(i)) } else { zero }
                }
                1 => {
                    let idx = desc.rhs_idx as usize;
                    if idx < constants.len() { _mm256_set1_ps(*constants.get_unchecked(idx)) } else { zero }
                }
                2 => {
                    let idx = desc.rhs_idx as usize;
                    *intermediates.get_unchecked(idx)
                }
                _ => zero,
            };

            intermediates[op_idx] = match desc.op {
                0 => _mm256_add_ps(lhs, rhs),
                1 => _mm256_sub_ps(lhs, rhs),
                2 => _mm256_mul_ps(lhs, rhs),
                3 => _mm256_div_ps(lhs, rhs),
                4 => _mm256_min_ps(lhs, rhs),
                5 => _mm256_max_ps(lhs, rhs),
                _ => lhs,
            };
        }

        let final_val = intermediates[num_ops - 1];

        if has_reduce {
            match reduce_op {
                REDUCE_SUM => acc0 = _mm256_add_ps(acc0, final_val),
                REDUCE_MAX => acc0 = _mm256_max_ps(acc0, final_val),
                REDUCE_MIN => acc0 = _mm256_min_ps(acc0, final_val),
                _ => {}
            }
        } else {
            _mm256_storeu_ps(dst_ptr.add(i), final_val);
        }

        i += 8;
    }

    // Scalar tail — also uses Vec for arbitrary-length chains
    let mut scalar_intermediates: Vec<f32> = vec![0.0; num_ops];
    let mut scalar_acc: f64 = 0.0;
    let mut first_scalar = true;
    while i < n {
        scalar_intermediates.fill(0.0f32);

        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs: f32 = match desc.lhs_src {
                0 => {
                    let idx = desc.lhs_idx as usize;
                    if idx < input_ptrs.len() { *input_ptrs[idx].add(i) } else { 0.0 }
                }
                1 => *constants.get(desc.lhs_idx as usize).unwrap_or(&0.0),
                2 => scalar_intermediates[desc.lhs_idx as usize],
                _ => 0.0,
            };
            let rhs: f32 = match desc.rhs_src {
                0 => {
                    let idx = desc.rhs_idx as usize;
                    if idx < input_ptrs.len() { *input_ptrs[idx].add(i) } else { 0.0 }
                }
                1 => *constants.get(desc.rhs_idx as usize).unwrap_or(&0.0),
                2 => scalar_intermediates[desc.rhs_idx as usize],
                _ => 0.0,
            };

            scalar_intermediates[op_idx] = match desc.op {
                0 => lhs + rhs,
                1 => lhs - rhs,
                2 => lhs * rhs,
                3 => lhs / rhs,
                4 => lhs.min(rhs),
                5 => lhs.max(rhs),
                _ => lhs,
            };
        }

        let final_val = scalar_intermediates[num_ops - 1];

        if has_reduce {
            match reduce_op {
                REDUCE_SUM => scalar_acc += final_val as f64,
                REDUCE_MAX => scalar_acc = if first_scalar { final_val as f64 } else { scalar_acc.max(final_val as f64) },
                REDUCE_MIN => scalar_acc = if first_scalar { final_val as f64 } else { scalar_acc.min(final_val as f64) },
                _ => {}
            }
            first_scalar = false;
        } else {
            *dst_ptr.add(i) = final_val;
        }

        i += 1;
    }

    // Horizontal reduction of the 4 accumulators
    if has_reduce {
        // Combine acc0-acc3
        match reduce_op {
            REDUCE_SUM => {
                acc0 = _mm256_add_ps(acc0, acc1);
                acc0 = _mm256_add_ps(acc0, acc2);
                acc0 = _mm256_add_ps(acc0, acc3);
            }
            REDUCE_MAX => {
                acc0 = _mm256_max_ps(acc0, acc1);
                acc0 = _mm256_max_ps(acc0, acc2);
                acc0 = _mm256_max_ps(acc0, acc3);
            }
            REDUCE_MIN => {
                acc0 = _mm256_min_ps(acc0, acc1);
                acc0 = _mm256_min_ps(acc0, acc2);
                acc0 = _mm256_min_ps(acc0, acc3);
            }
            _ => {}
        }

        // Horizontal reduction within ymm0:
        // Step 1: swap high/low 128-bit lanes
        let hi = _mm256_permute2f128_ps(acc0, acc0, 0x01);
        acc0 = match reduce_op {
            REDUCE_SUM => _mm256_add_ps(acc0, hi),
            REDUCE_MAX => _mm256_max_ps(acc0, hi),
            REDUCE_MIN => _mm256_min_ps(acc0, hi),
            _ => acc0,
        };
        // Step 2: shuffle 64-bit halves within each lane
        let shuf = _mm256_shuffle_ps(acc0, acc0, 0x4E); // swap positions 0↔1, 2↔3
        acc0 = match reduce_op {
            REDUCE_SUM => _mm256_add_ps(acc0, shuf),
            REDUCE_MAX => _mm256_max_ps(acc0, shuf),
            REDUCE_MIN => _mm256_min_ps(acc0, shuf),
            _ => acc0,
        };
        // Step 3: shuffle 32-bit halves
        let shuf2 = _mm256_shuffle_ps(acc0, acc0, 0xB1); // swap positions 0↔1 within each 64-bit
        acc0 = match reduce_op {
            REDUCE_SUM => _mm256_add_ps(acc0, shuf2),
            REDUCE_MAX => _mm256_max_ps(acc0, shuf2),
            REDUCE_MIN => _mm256_min_ps(acc0, shuf2),
            _ => acc0,
        };

        // Extract the scalar from the low element of the low 128-bit lane
        let result = _mm_cvtss_f32(_mm256_castps256_ps128(acc0));
        return result as f64 + scalar_acc;
    }

    0.0
}

/// Scalar fallback for f32 fused elementwise.
/// Unlike the AVX2 path, this supports arbitrary-length op chains
/// (not limited to MAX_FUSED_OPS).
unsafe fn fused_elem_f32_scalar(
    ops: &[FusedOpDesc],
    input_ptrs: &[*const f32],
    constants: &[f32],
    n: usize,
    reduce_op: u8,
    dst_ptr: *mut f32,
) -> f64 {
    let num_ops = ops.len();
    let has_reduce = reduce_op != REDUCE_NONE;
    let mut acc: f64 = 0.0;
    let mut first = true;

    // Allocate intermediates once outside the loop — reused for each element.
    // This avoids O(n) heap allocations which would dominate runtime for
    // large arrays (e.g., 250K elements × 700 ops = 250K Vec allocations).
    let mut intermediates: Vec<f32> = vec![0.0; num_ops];

    for i in 0..n {
        // Clear intermediates for this element
        intermediates.fill(0.0);

        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs: f32 = match desc.lhs_src {
                0 => if (desc.lhs_idx as usize) < input_ptrs.len() { *input_ptrs[desc.lhs_idx as usize].add(i) } else { 0.0 },
                1 => *constants.get(desc.lhs_idx as usize).unwrap_or(&0.0),
                2 => if (desc.lhs_idx as usize) < num_ops { intermediates[desc.lhs_idx as usize] } else { 0.0 },
                _ => 0.0,
            };
            let rhs: f32 = match desc.rhs_src {
                0 => if (desc.rhs_idx as usize) < input_ptrs.len() { *input_ptrs[desc.rhs_idx as usize].add(i) } else { 0.0 },
                1 => *constants.get(desc.rhs_idx as usize).unwrap_or(&0.0),
                2 => if (desc.rhs_idx as usize) < num_ops { intermediates[desc.rhs_idx as usize] } else { 0.0 },
                _ => 0.0,
            };
            intermediates[op_idx] = match desc.op {
                0 => lhs + rhs,
                1 => lhs - rhs,
                2 => lhs * rhs,
                3 => lhs / rhs,
                4 => lhs.min(rhs),
                5 => lhs.max(rhs),
                _ => lhs,
            };
        }

        let val = intermediates[num_ops - 1];
        if has_reduce {
            match reduce_op {
                REDUCE_SUM => acc += val as f64,
                REDUCE_MAX => acc = if first { val as f64 } else { acc.max(val as f64) },
                REDUCE_MIN => acc = if first { val as f64 } else { acc.min(val as f64) },
                _ => {}
            }
            first = false;
        } else {
            *dst_ptr.add(i) = val;
        }
    }

    acc
}

/// Execute a fused chain of elementwise operations in a single pass for f32 arrays.
///
/// This is the key performance optimization: instead of writing intermediate
/// results to memory and reading them back, all ops are computed per-element
/// in SIMD registers.
///
/// # Arguments
/// * `ops` - Vec of (op, lhs_src, lhs_idx, rhs_src, rhs_idx) tuples
///   - op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
///   - lhs_src/rhs_src: 0=input_array, 1=constant, 2=previous_op_result
///   - lhs_idx/rhs_idx: index into the respective source
/// * `input_ptrs` - Raw pointers to input f32 arrays
/// * `constants` - f32 constant values
/// * `n` - Number of elements
/// * `reduce_op` - 0=sum, 1=max, 2=min, 255=no reduce (write to dst)
/// * `dst_ptr` - Output array pointer (used when reduce_op == 255)
///
/// # Returns
/// Reduced scalar value (if reduce) or 0.0 (if writing to dst)
pub fn simd_fused_elementwise_f32(
    ops: Vec<(u8, u8, u16, u8, u16)>,
    input_ptrs: Vec<usize>,
    constants: Vec<f32>,
    n: usize,
    reduce_op: u8,
    dst_ptr: usize,
) -> f64 {
    if n == 0 || ops.is_empty() { return 0.0; }

    let descs: Vec<FusedOpDesc> = ops.iter().map(|&(op, ls, li, rs, ri)| FusedOpDesc {
        op, lhs_src: ls, lhs_idx: li, rhs_src: rs, rhs_idx: ri,
    }).collect();
    let ptrs: Vec<*const f32> = input_ptrs.iter().map(|&p| p as *const f32).collect();
    let dst = dst_ptr as *mut f32;

    // Always use AVX2 when available — the kernel now supports
    // arbitrary-length op chains via Vec<__m256> intermediates.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                fused_elem_f32_avx2_core(&descs, &ptrs, &constants, n, reduce_op, dst)
            };
        }
    }

    // Scalar fallback for non-AVX2 systems
    unsafe { fused_elem_f32_scalar(&descs, &ptrs, &constants, n, reduce_op, dst) }
}

// ── f64 AVX2 core ──
//
// Supports ARBITRARY-LENGTH op chains — same approach as the f32 core.
// Vec<__m256d> intermediates allocated once, reused per element.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fused_elem_f64_avx2_core(
    ops: &[FusedOpDesc],
    input_ptrs: &[*const f64],
    constants: &[f64],
    n: usize,
    reduce_op: u8,
    dst_ptr: *mut f64,
) -> f64 {
    use std::arch::x86_64::*;

    let num_ops = ops.len();
    if num_ops == 0 || n == 0 { return 0.0; }

    let has_reduce = reduce_op != REDUCE_NONE;
    let zero = _mm256_setzero_pd();

    // For max/min, initialize to ±∞ so any finite value wins.
    let init_val: __m256d = match reduce_op {
        REDUCE_MAX => _mm256_set1_pd(f64::NEG_INFINITY),
        REDUCE_MIN => _mm256_set1_pd(f64::INFINITY),
        _ => zero,
    };
    let mut acc0 = init_val;
    let mut acc1 = init_val;
    let mut acc2 = init_val;
    let mut acc3 = init_val;

    // Allocate intermediates ONCE — reused across all elements.
    let mut intermediates: Vec<__m256d> = vec![zero; num_ops];

    // Process 16 f64 per iteration (4 × 4-wide YMM)
    let n_vec4 = n & !15;
    let mut i = 0usize;

    while i < n_vec4 {
        for u in 0..4usize {
            let base = i + u * 4;

            for (op_idx, desc) in ops.iter().enumerate() {
                let lhs = match desc.lhs_src {
                    0 => {
                        let idx = desc.lhs_idx as usize;
                        if idx < input_ptrs.len() { _mm256_loadu_pd(input_ptrs[idx].add(base)) } else { zero }
                    }
                    1 => {
                        let idx = desc.lhs_idx as usize;
                        if idx < constants.len() { _mm256_set1_pd(*constants.get_unchecked(idx)) } else { zero }
                    }
                    2 => {
                        let idx = desc.lhs_idx as usize;
                        *intermediates.get_unchecked(idx)
                    }
                    _ => zero,
                };
                let rhs = match desc.rhs_src {
                    0 => {
                        let idx = desc.rhs_idx as usize;
                        if idx < input_ptrs.len() { _mm256_loadu_pd(input_ptrs[idx].add(base)) } else { zero }
                    }
                    1 => {
                        let idx = desc.rhs_idx as usize;
                        if idx < constants.len() { _mm256_set1_pd(*constants.get_unchecked(idx)) } else { zero }
                    }
                    2 => {
                        let idx = desc.rhs_idx as usize;
                        *intermediates.get_unchecked(idx)
                    }
                    _ => zero,
                };

                intermediates[op_idx] = match desc.op {
                    0 => _mm256_add_pd(lhs, rhs),
                    1 => _mm256_sub_pd(lhs, rhs),
                    2 => _mm256_mul_pd(lhs, rhs),
                    3 => _mm256_div_pd(lhs, rhs),
                    4 => _mm256_min_pd(lhs, rhs),
                    5 => _mm256_max_pd(lhs, rhs),
                    _ => lhs,
                };
            }

            let final_val = intermediates[num_ops - 1];

            if has_reduce {
                match reduce_op {
                    REDUCE_SUM => match u {
                        0 => acc0 = _mm256_add_pd(acc0, final_val),
                        1 => acc1 = _mm256_add_pd(acc1, final_val),
                        2 => acc2 = _mm256_add_pd(acc2, final_val),
                        _ => acc3 = _mm256_add_pd(acc3, final_val),
                    },
                    REDUCE_MAX => match u {
                        0 => acc0 = _mm256_max_pd(acc0, final_val),
                        1 => acc1 = _mm256_max_pd(acc1, final_val),
                        2 => acc2 = _mm256_max_pd(acc2, final_val),
                        _ => acc3 = _mm256_max_pd(acc3, final_val),
                    },
                    REDUCE_MIN => match u {
                        0 => acc0 = _mm256_min_pd(acc0, final_val),
                        1 => acc1 = _mm256_min_pd(acc1, final_val),
                        2 => acc2 = _mm256_min_pd(acc2, final_val),
                        _ => acc3 = _mm256_min_pd(acc3, final_val),
                    },
                    _ => {}
                }
            } else {
                _mm256_storeu_pd(dst_ptr.add(base), final_val);
            }
        }
        i += 16;
    }

    // Remaining 4-element chunks
    let n_vec = n & !3;
    while i < n_vec {
        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs = match desc.lhs_src {
                0 => { let idx = desc.lhs_idx as usize; if idx < input_ptrs.len() { _mm256_loadu_pd(input_ptrs[idx].add(i)) } else { zero } }
                1 => { let idx = desc.lhs_idx as usize; if idx < constants.len() { _mm256_set1_pd(*constants.get_unchecked(idx)) } else { zero } }
                2 => { let idx = desc.lhs_idx as usize; *intermediates.get_unchecked(idx) }
                _ => zero,
            };
            let rhs = match desc.rhs_src {
                0 => { let idx = desc.rhs_idx as usize; if idx < input_ptrs.len() { _mm256_loadu_pd(input_ptrs[idx].add(i)) } else { zero } }
                1 => { let idx = desc.rhs_idx as usize; if idx < constants.len() { _mm256_set1_pd(*constants.get_unchecked(idx)) } else { zero } }
                2 => { let idx = desc.rhs_idx as usize; *intermediates.get_unchecked(idx) }
                _ => zero,
            };

            intermediates[op_idx] = match desc.op {
                0 => _mm256_add_pd(lhs, rhs),
                1 => _mm256_sub_pd(lhs, rhs),
                2 => _mm256_mul_pd(lhs, rhs),
                3 => _mm256_div_pd(lhs, rhs),
                4 => _mm256_min_pd(lhs, rhs),
                5 => _mm256_max_pd(lhs, rhs),
                _ => lhs,
            };
        }

        let final_val = intermediates[num_ops - 1];
        if has_reduce {
            match reduce_op {
                REDUCE_SUM => acc0 = _mm256_add_pd(acc0, final_val),
                REDUCE_MAX => acc0 = _mm256_max_pd(acc0, final_val),
                REDUCE_MIN => acc0 = _mm256_min_pd(acc0, final_val),
                _ => {}
            }
        } else {
            _mm256_storeu_pd(dst_ptr.add(i), final_val);
        }
        i += 4;
    }

    // Scalar tail for f64 — uses Vec for arbitrary-length chains
    let mut scalar_intermediates: Vec<f64> = vec![0.0; num_ops];
    let mut scalar_acc: f64 = 0.0;
    let mut first_scalar = true;
    while i < n {
        scalar_intermediates.fill(0.0f64);
        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs: f64 = match desc.lhs_src {
                0 => { let idx = desc.lhs_idx as usize; if idx < input_ptrs.len() { *input_ptrs[idx].add(i) } else { 0.0 } }
                1 => *constants.get(desc.lhs_idx as usize).unwrap_or(&0.0),
                2 => scalar_intermediates[desc.lhs_idx as usize],
                _ => 0.0,
            };
            let rhs: f64 = match desc.rhs_src {
                0 => { let idx = desc.rhs_idx as usize; if idx < input_ptrs.len() { *input_ptrs[idx].add(i) } else { 0.0 } }
                1 => *constants.get(desc.rhs_idx as usize).unwrap_or(&0.0),
                2 => scalar_intermediates[desc.rhs_idx as usize],
                _ => 0.0,
            };
            scalar_intermediates[op_idx] = match desc.op {
                0 => lhs + rhs, 1 => lhs - rhs, 2 => lhs * rhs, 3 => lhs / rhs,
                4 => lhs.min(rhs), 5 => lhs.max(rhs), _ => lhs,
            };
        }
        let val = scalar_intermediates[num_ops - 1];
        if has_reduce {
            match reduce_op {
                REDUCE_SUM => scalar_acc += val,
                REDUCE_MAX => scalar_acc = if first_scalar { val } else { scalar_acc.max(val) },
                REDUCE_MIN => scalar_acc = if first_scalar { val } else { scalar_acc.min(val) },
                _ => {}
            }
            first_scalar = false;
        } else {
            *dst_ptr.add(i) = val;
        }
        i += 1;
    }

    // Horizontal reduction
    if has_reduce {
        match reduce_op {
            REDUCE_SUM => {
                acc0 = _mm256_add_pd(acc0, acc1);
                acc0 = _mm256_add_pd(acc0, acc2);
                acc0 = _mm256_add_pd(acc0, acc3);
            }
            REDUCE_MAX => {
                acc0 = _mm256_max_pd(acc0, acc1);
                acc0 = _mm256_max_pd(acc0, acc2);
                acc0 = _mm256_max_pd(acc0, acc3);
            }
            REDUCE_MIN => {
                acc0 = _mm256_min_pd(acc0, acc1);
                acc0 = _mm256_min_pd(acc0, acc2);
                acc0 = _mm256_min_pd(acc0, acc3);
            }
            _ => {}
        }
        let hi = _mm256_permute2f128_pd(acc0, acc0, 0x01);
        acc0 = match reduce_op {
            REDUCE_SUM => _mm256_add_pd(acc0, hi),
            REDUCE_MAX => _mm256_max_pd(acc0, hi),
            REDUCE_MIN => _mm256_min_pd(acc0, hi),
            _ => acc0,
        };
        let shuf = _mm256_shuffle_pd(acc0, acc0, 0x05);
        acc0 = match reduce_op {
            REDUCE_SUM => _mm256_add_pd(acc0, shuf),
            REDUCE_MAX => _mm256_max_pd(acc0, shuf),
            REDUCE_MIN => _mm256_min_pd(acc0, shuf),
            _ => acc0,
        };
        // Extract low f64 from the 128-bit low half
        let low128 = _mm256_castpd256_pd128(acc0);
        let result = _mm_cvtsd_f64(low128);
        return result + scalar_acc;
    }

    0.0
}

/// Scalar fallback for f64 fused elementwise.
/// Used when AVX2 is not available. Supports arbitrary-length op chains.
unsafe fn fused_elem_f64_scalar(
    ops: &[FusedOpDesc],
    input_ptrs: &[*const f64],
    constants: &[f64],
    n: usize,
    reduce_op: u8,
    dst_ptr: *mut f64,
) -> f64 {
    let num_ops = ops.len();
    let has_reduce = reduce_op != REDUCE_NONE;
    let mut acc: f64 = 0.0;
    let mut first = true;

    let mut intermediates: Vec<f64> = vec![0.0; num_ops];

    for i in 0..n {
        intermediates.fill(0.0);
        for (op_idx, desc) in ops.iter().enumerate() {
            let lhs: f64 = match desc.lhs_src {
                0 => if (desc.lhs_idx as usize) < input_ptrs.len() { *input_ptrs[desc.lhs_idx as usize].add(i) } else { 0.0 },
                1 => *constants.get(desc.lhs_idx as usize).unwrap_or(&0.0),
                2 => if (desc.lhs_idx as usize) < num_ops { intermediates[desc.lhs_idx as usize] } else { 0.0 },
                _ => 0.0,
            };
            let rhs: f64 = match desc.rhs_src {
                0 => if (desc.rhs_idx as usize) < input_ptrs.len() { *input_ptrs[desc.rhs_idx as usize].add(i) } else { 0.0 },
                1 => *constants.get(desc.rhs_idx as usize).unwrap_or(&0.0),
                2 => if (desc.rhs_idx as usize) < num_ops { intermediates[desc.rhs_idx as usize] } else { 0.0 },
                _ => 0.0,
            };
            intermediates[op_idx] = match desc.op {
                0 => lhs + rhs,
                1 => lhs - rhs,
                2 => lhs * rhs,
                3 => lhs / rhs,
                4 => lhs.min(rhs),
                5 => lhs.max(rhs),
                _ => lhs,
            };
        }
        let val = intermediates[num_ops - 1];
        if has_reduce {
            match reduce_op {
                REDUCE_SUM => acc += val,
                REDUCE_MAX => acc = if first { val } else { acc.max(val) },
                REDUCE_MIN => acc = if first { val } else { acc.min(val) },
                _ => {}
            }
            first = false;
        } else {
            *dst_ptr.add(i) = val;
        }
    }
    acc
}

/// Execute a fused chain of elementwise operations in a single pass for f64 arrays.
pub fn simd_fused_elementwise_f64(
    ops: Vec<(u8, u8, u16, u8, u16)>,
    input_ptrs: Vec<usize>,
    constants: Vec<f64>,
    n: usize,
    reduce_op: u8,
    dst_ptr: usize,
) -> f64 {
    if n == 0 || ops.is_empty() { return 0.0; }

    let descs: Vec<FusedOpDesc> = ops.iter().map(|&(op, ls, li, rs, ri)| FusedOpDesc {
        op, lhs_src: ls, lhs_idx: li, rhs_src: rs, rhs_idx: ri,
    }).collect();
    let ptrs: Vec<*const f64> = input_ptrs.iter().map(|&p| p as *const f64).collect();
    let dst = dst_ptr as *mut f64;

    // Always use AVX2 when available — the kernel now supports
    // arbitrary-length op chains via Vec<__m256d> intermediates.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                fused_elem_f64_avx2_core(&descs, &ptrs, &constants, n, reduce_op, dst)
            };
        }
    }

    // Scalar fallback for non-AVX2 systems
    unsafe { fused_elem_f64_scalar(&descs, &ptrs, &constants, n, reduce_op, dst) }
}
