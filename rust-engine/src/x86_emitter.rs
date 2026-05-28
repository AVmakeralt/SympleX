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
    /// Compile SSE scalar matmul (baseline fallback)
    pub fn compile_matmul() -> Self {
        let mut a = CodeAssembler::new(64).unwrap();

        // Prologue
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();

        a.mov(r10, rcx).unwrap(); a.mov(r11, r9).unwrap();
        a.mov(r12, rdi).unwrap(); a.mov(r13, rsi).unwrap();
        a.mov(r14, rdx).unwrap(); a.mov(r15, r8).unwrap();

        // Zero C
        a.xorps(xmm0, xmm0).unwrap();
        a.xor(rdi, rdi).unwrap();
        let mut z_i = a.create_label();
        let mut z_i_out = a.create_label();
        a.set_label(&mut z_i).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(z_i_out).unwrap();
        a.xor(rsi, rsi).unwrap();
        let mut z_j = a.create_label();
        let mut z_j_out = a.create_label();
        a.set_label(&mut z_j).unwrap();
        a.cmp(rsi, r15).unwrap(); a.jge(z_j_out).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, rsi).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm0).unwrap();
        a.add(rsi, 1).unwrap(); a.jmp(z_j).unwrap();
        a.set_label(&mut z_j_out).unwrap();
        a.add(rdi, 1).unwrap(); a.jmp(z_i).unwrap();
        a.set_label(&mut z_i_out).unwrap();

        // Matmul: i-k-j
        a.xor(rdi, rdi).unwrap();
        let mut m_i = a.create_label();
        let mut m_i_out = a.create_label();
        a.set_label(&mut m_i).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(m_i_out).unwrap();
        a.xor(r9, r9).unwrap();
        let mut m_k = a.create_label();
        let mut m_k_out = a.create_label();
        a.set_label(&mut m_k).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(m_k_out).unwrap();
        // Load A[i,k]
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r11).unwrap(); a.add(rax, r9).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r12).unwrap();
        a.movss(xmm0, dword_ptr(rax)).unwrap();

        a.xor(r8, r8).unwrap();
        let mut m_j = a.create_label();
        let mut m_j_out = a.create_label();
        a.set_label(&mut m_j).unwrap();
        a.cmp(r8, r15).unwrap(); a.jge(m_j_out).unwrap();
        // Load C[i,j]
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, r8).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r14).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();
        // Load B[k,j]
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, r8).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r13).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();
        // C[i,j] += A[i,k] * B[k,j]
        a.mulss(xmm2, xmm0).unwrap();
        a.addss(xmm1, xmm2).unwrap();
        // Store C[i,j]
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.add(rax, r8).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm1).unwrap();
        a.add(r8, 1).unwrap(); a.jmp(m_j).unwrap();
        a.set_label(&mut m_j_out).unwrap();
        a.add(r9, 1).unwrap(); a.jmp(m_k).unwrap();
        a.set_label(&mut m_k_out).unwrap();
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

    /// Compile AVX2-vectorized matmul
    pub fn compile_matmul_avx2() -> Self {
        let mut a = CodeAssembler::new(64).unwrap();
        a.push(rbp).unwrap(); a.mov(rbp, rsp).unwrap();
        a.push(rbx).unwrap(); a.push(r12).unwrap(); a.push(r13).unwrap();
        a.push(r14).unwrap(); a.push(r15).unwrap();
        a.mov(r10, rcx).unwrap(); a.mov(r11, r9).unwrap();
        a.mov(r12, rdi).unwrap(); a.mov(r13, rsi).unwrap();
        a.mov(r14, rdx).unwrap(); a.mov(r15, r8).unwrap();

        // Zero C
        a.vxorps(ymm0, ymm0, ymm0).unwrap();
        a.mov(rdi, r10).unwrap(); a.imul_2(rdi, r15).unwrap();
        a.xor(rsi, rsi).unwrap();
        let mut zvec_loop = a.create_label();
        let mut zvec_exit = a.create_label();
        a.set_label(&mut zvec_loop).unwrap();
        a.mov(rax, rdi).unwrap(); a.shl(rax, 2).unwrap(); a.sub(rax, 32).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(zvec_exit).unwrap();
        a.lea(rax, qword_ptr(r14 + rsi)).unwrap();
        a.vmovups(ymmword_ptr(rax), ymm0).unwrap();
        a.add(rsi, 32).unwrap(); a.jmp(zvec_loop).unwrap();
        a.set_label(&mut zvec_exit).unwrap();
        // Scalar zero tail
        a.mov(rax, rdi).unwrap(); a.shl(rax, 2).unwrap();
        let mut zsc_loop = a.create_label();
        let mut zsc_exit = a.create_label();
        a.set_label(&mut zsc_loop).unwrap();
        a.cmp(rsi, rax).unwrap(); a.jge(zsc_exit).unwrap();
        a.lea(rbx, qword_ptr(r14 + rsi)).unwrap();
        a.movss(dword_ptr(rbx), xmm0).unwrap();
        a.add(rsi, 4).unwrap(); a.jmp(zsc_loop).unwrap();
        a.set_label(&mut zsc_exit).unwrap();

        // i-k-j matmul
        a.xor(rdi, rdi).unwrap();
        let mut mi_loop = a.create_label();
        let mut mi_exit = a.create_label();
        a.set_label(&mut mi_loop).unwrap();
        a.cmp(rdi, r10).unwrap(); a.jge(mi_exit).unwrap();
        a.xor(r9, r9).unwrap();
        let mut mk_loop = a.create_label();
        let mut mk_exit = a.create_label();
        a.set_label(&mut mk_loop).unwrap();
        a.cmp(r9, r11).unwrap(); a.jge(mk_exit).unwrap();

        // Broadcast A[i,k]
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r11).unwrap(); a.add(rax, r9).unwrap();
        a.shl(rax, 2).unwrap(); a.add(rax, r12).unwrap();
        a.vbroadcastss(ymm0, dword_ptr(rax)).unwrap();

        a.xor(r8, r8).unwrap();
        a.mov(rbx, r15).unwrap(); a.shl(rbx, 2).unwrap(); // RBX = N*4

        let mut mj_vec = a.create_label();
        let mut mj_vec_exit = a.create_label();
        a.set_label(&mut mj_vec).unwrap();
        a.mov(rax, rbx).unwrap(); a.sub(rax, 32).unwrap(); a.cmp(r8, rax).unwrap();
        a.jge(mj_vec_exit).unwrap();

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

        a.add(r8, 32).unwrap(); a.jmp(mj_vec).unwrap();
        a.set_label(&mut mj_vec_exit).unwrap();

        // Scalar tail
        let mut mj_sc = a.create_label();
        let mut mj_sc_exit = a.create_label();
        a.set_label(&mut mj_sc).unwrap();
        a.cmp(r8, rbx).unwrap(); a.jge(mj_sc_exit).unwrap();
        a.mov(rax, r9).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r13).unwrap();
        a.movss(xmm2, dword_ptr(rax)).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(xmm1, dword_ptr(rax)).unwrap();
        a.mulss(xmm2, xmm0).unwrap();
        a.addss(xmm1, xmm2).unwrap();
        a.mov(rax, rdi).unwrap(); a.imul_2(rax, r15).unwrap(); a.shl(rax, 2).unwrap();
        a.add(rax, r8).unwrap(); a.add(rax, r14).unwrap();
        a.movss(dword_ptr(rax), xmm1).unwrap();
        a.add(r8, 4).unwrap(); a.jmp(mj_sc).unwrap();
        a.set_label(&mut mj_sc_exit).unwrap();

        a.add(r9, 1).unwrap(); a.jmp(mk_loop).unwrap();
        a.set_label(&mut mk_exit).unwrap();
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

// ── Multi-threaded matmul ──

pub fn parallel_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    if m == 0 || n == 0 || k == 0 { return; }
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
    let isa = detect_isa_level();
    if isa == ISALevel::AVX512 && n >= 64 {
        let kernel = CompiledKernel::compile_matmul_avx512(m, n, k);
        if !kernel.exec_ptr.is_null() {
            let _ = kernel.exec_matmul(a, b, c, m as i64, n as i64, k as i64);
            return;
        }
    }
    parallel_matmul(a, b, c, m, n, k);
}
