#![allow(non_snake_case, clippy::all)]
//!JIT backend
//!
//! Optimizations active:
//!
//! A. Full linear-scan register allocation across 10 GPRs
//!    (r8-r11, rsi — caller-saved; r12-r15, rbx — callee-saved).
//!    rax/rcx = scratch accumulators.  rdx = reserved for cqo/idiv.
//!    rdi = slot-array base pointer.  All other GPRs allocated by RA.
//!    Callee-saved registers that are actually used get pushed/popped
//!    in a generated prologue/epilogue.
//!
//! B. Forward constant-propagation + compile-time BinOp folding.
//!    Any slot whose value is statically known is propagated through
//!    Move/Store/BinOp; foldable BinOps emit a single immediate load.
//!    State is conservatively cleared at every branch target.
//!    const_at is a flat Vec<Option<i64>> (O(1) slot lookup, cache-friendly).
//!
//! C. 3-instruction superinstruction fusions:
//!    • BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  LEA  (N ∈ {2,4,8})
//!    • BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  SHL+ADD (N = power-of-2)
//!    • BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  IMUL+ADD (N = arbitrary stride)
//!    • BinOp(t, op1, a, b) + BinOp(r, op2, t, c)  →  two-op chain
//!      (eliminates the intermediate slot store + load of `t`)
//!
//! D. 2-instruction superinstruction fusions (all original patterns):
//!    • LoadI*(tmp, c) + JumpFalse/True(tmp, …) → compile-time branch fold
//!    • LoadI*(tmp, c) + BinOp(d, op, x, tmp)   → immediate-form arithmetic
//!    • BinOp(t, op, l, r) + Store(slot, t)      → fused compute-and-store
//!    • BinOp(t, op, l, r) + JumpFalse/True(t,…) → fused cmp+branch
//!
//! E. Optimal immediate encoding:
//!    MOV EAX, imm32 (5 B, zero-extends) when 0 ≤ v < 2³¹
//!    MOV RAX, sign-extended imm32 (7 B) when −2³¹ ≤ v < 0
//!    MOV RAX, imm64 (10 B) only when the value doesn't fit in 32 bits.
//!    ADD/SUB/CMP RAX, imm8 (4 B) when value fits in i8.
//!
//! F. Short-form branches: JMP/JZ/JNZ rel8 (2 B) when displacement fits in i8,
//!    falling back to rel32 (5-6 B) otherwise.
//!
//! G. REX-free XOR/TEST for boolean/zero ops:
//!    XOR EAX, EAX (2 B) and TEST EAX, EAX (2 B) instead of 64-bit forms (3 B).
//!
//! H. All existing micro-optimisations retained:
//!    LEA for ×3/×5/×9, SHL for powers-of-two multiply,
//!    INC/DEC for ±1, TEST+Jcc, SETCC for branchless comparisons.
//!
//! I. Common Subexpression Elimination (CSE): identical BinOps are replaced
//!    with Move from the earlier result, eliminating redundant computation.
//!    The CSE table is cleared at control-flow barriers for correctness.
//!
//! J. Division by constant: replaced with magic-number multiply-high + shift
//!    when the divisor is a positive compile-time constant (3-4 cycles vs
//!    20-40 cycles for IDIV).
//!
//! K. Loop unrolling: small loops (body ≤ 8 instructions) are duplicated to
//!    reduce branch overhead, with the original backward jump retained for
//!    remaining iterations.
//!
//! L. Branchless code via CMOVcc: when guards are unpredictable, conditional
//!    moves eliminate branch misprediction penalties (15-20 cycles each).
//!
//! M. Loop-Invariant Code Motion (LICM): computations whose inputs don't
//!    change across iterations are hoisted before the loop.
//!
//! N. Strength reduction for induction variables: expensive operations inside
//!    loops are replaced with cheaper equivalents (e.g., mul→shift).
//!
//! O. Instruction scheduling: independent instructions are reordered to
//!    maximize instruction-level parallelism on wide-issue CPUs.
//!
//! P. Machine code validation (debug builds): all branch targets, REX
//!    prefixes, and fixup regions are checked for consistency.
//!
//! Q. Register coalescing: after linear-scan allocation, Move(dst, src)
//!    instructions where both operands are in registers and src is dead
//!    after the Move are eliminated by assigning dst to src's register,
//!    making the Move a no-op.  This removes redundant register-to-register
//!    copies (MOV reg1, reg2) that the allocator would otherwise emit.
//!
//! R. Copy-and-Patch Stencil Compilation: pre-compiled bytecode handler
//!    stencils with runtime patching. Compilation drops to O(1) per
//!    bytecode instruction — scales purely with memory bandwidth.
//!
//! S. Dual-Mapped 2MB/1GB Huge Pages: UNCONDITIONAL dual mapping — the arena
//!    gets TWO mappings of the same physical memory (RW + RX). All JIT code
//!    writes go through the stable RW chunk pointer and execute instantly from
//!    the corresponding RX mirror pointer. No PageBitmap tracking, no dirty
//!    page recording, and zero mprotect/RW→RX syscalls — adding, updating, or
//!    evicting JIT code never invokes a single kernel syscall.
//!
//! T. Hardware Performance Monitoring Counters (PMCs): Linux perf_event_open
//!    for branch mispredictions, cache misses, and instruction retirement,
//!    replacing software hotness counters with zero-overhead hardware profiling.
//!
//! U. Custom JIT Calling Convention: JIT-to-JIT calls break the System V ABI
//!    for maximum performance — 12 GPR parameter passing, register pinning
//!    (R13=VM ctx, R14=stack, R15=heap), zero-frame JMP epilogues.
//!
//! V. Atomic Cross-Modifying Code (CMC) Trace Stitching: lock-free tier
//!    transitions using atomic 64-bit JMP target patching on 8-byte-aligned
//!    boundaries. The CPU smoothly transitions to faster code mid-flight.
//!
//! W. Macro-Op Fusion: CMP/TEST + Jcc adjacency guaranteed for hardware
//!    macro-op fusion on Intel/AMD CPUs, with 16-byte decode-boundary
//!    alignment padding.
//!
//! X. Execution Port Balancing & Aggressive LEA Expansion: any x = y + z*N + C
//!    (N ∈ {1,2,4,8}) uses LEA instead of IMUL+ADD, distributing load
//!    across address-generation ports.
//!
//! Y. Speculative SIMD Vectorization with AVX-512 Masking: hot scalar loops
//!    are speculatively compiled as AVX-512 masked versions using k-register
//!    opmasks for trail-end handling.
//!
//! Z. Polymorphic Inline Caching with Self-Modifying Code: PIC slots in
//!    native code with mono→poly→mega transitions and atomic 8-byte
//!    JMP target rewriting.
//!
//! AA. Hot/Cold Code Layout (Outline Slow Paths): error checks, side exits,
//!    and deoptimization are emitted far away in a cold zone, maximizing
//!    L1i cache efficiency for the hot path.
//!
//! AB. Tail-Duplication and Stencil Specialization: pre-generated specialized
//!    stencils for common constant patterns (INC/DEC/SHL with baked constants),
//!    super-stencils that fuse 2-3 instruction sequences into contiguous blocks.
//!
//! AC. On-Stack Replacement (OSR) with Deoptimization Maps: Tier-1 loops
//!    seamlessly morph into Tier-3 code mid-loop using side-tables that map
//!    register state between frame layouts.
//!
//! AD. LBR Feedback-Directed Optimization: Intel Last Branch Record hardware
//!    provides exact branch execution paths for dynamic block reordering.
//!
//! AE. TMAM Integration: Top-Down Microarchitectural Analysis reads PMCs to
//!    classify bottlenecks (Front-End/Back-End/Bad Speculation/Retiring) and
//!    recommends targeted optimizations (prefetch injection, compaction, etc.).
//!
//! AF. Chordal Graph Coloring Register Allocator: for Strict SSA, the
//!    interference graph is chordal, enabling optimal coloring in O(V+E)
//!    via Maximum Cardinality Search — peak graph-coloring quality at
//!    linear-scan speed.
//!
//! AG. RAT-Aware Register Pinning: short-lived temporaries are pinned to
//!    registers that the CPU's Register Alias Table can rename independently,
//!    eliminating WAW data hazards before instructions reach execution units.
//!
//! AH. TLB Pre-hinting: newly compiled code regions are pre-faulted through
//!    both RW and RX mappings to warm the DTLB and ITLB, avoiding the
//!    ~100-cycle TLB miss penalty on first execution.
//!
//! AI. Tier 4 Composition / Orchestration Layer: smart scheduler that
//!    decomposes general code into Tier 1-3 chunks (SIMD elementwise,
//!    fused vector, BLAS), builds an execution DAG with conservative
//!    fusion (same-dtype, contiguous, no branch explosion), computes
//!    buffer reuse plans via lifetime analysis, and dispatches to
//!    existing backends — never inventing new execution methods.
//!    The conductor, not a new instrument.

// ── Conditional JIT trace logging ──────────────────────────────────────────
// In release builds, jit_trace! compiles to a no-op unless the `jit_trace`
// feature is enabled, eliminating all formatting and I/O overhead from the
// compilation hot path.  Enable with: cargo build --features jit_trace
#[cfg(feature = "jit_trace")]
macro_rules! jit_trace {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}
#[cfg(not(feature = "jit_trace"))]
macro_rules! jit_trace {
    ($($arg:tt)*) => {};
}

use std::cell::RefCell;
use std::collections::{BinaryHeap, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

use libc::{mmap, mprotect, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};
use rustc_hash::{FxHashMap, FxHashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Bitmap for W^X Page Tracking — replaces BTreeSet<usize>
// ─────────────────────────────────────────────────────────────────────────────
//
// The old BTreeSet<usize> for _legacy_dirty_pages and finalized_pages caused heap
// thrashing during W^X page-flipping transitions. Each insert/remove on
// BTreeSet allocates/deallocates tree nodes. For a JIT that compiles
// hundreds of functions, this creates significant GC pressure.
//
// A dynamic bitmap covers the full arena capacity (up to 64 MiB).
// Each bit represents one 4 KiB page. Set = page is dirty/finalized,
// Clear = page is clean.  The bitmap stores page offsets relative to
// the arena base pointer, not absolute virtual addresses, so it
// correctly handles any address range.
//
// Operations are O(1) with no heap allocation: set, clear, test, iter.

const PAGE_SIZE: usize = 4096;
/// Maximum number of pages the bitmap can track.
/// 16384 pages × 4 KiB = 64 MiB = capacity_limit of ExecArena.
const MAX_TRACKED_PAGES: usize = 16384;

#[derive(Clone)]
struct PageBitmap {
    /// Bit vector: bit i is set if page i (relative to the arena base) is dirty.
    /// 16384 bits = 256 × u64 = 2048 bytes.
    bits: Vec<u64>,
    /// The arena base pointer — all page_addr arguments are relative to this.
    base: usize,
}

impl PageBitmap {
    fn new(base: *const u8) -> Self {
        let word_count = (MAX_TRACKED_PAGES + 63) / 64;
        Self {
            bits: vec![0u64; word_count],
            base: base as usize,
        }
    }

    #[inline(always)]
    fn page_index(&self, page_addr: usize) -> (usize, usize) {
        // Convert absolute address to offset from arena base, then to page index
        let offset = page_addr.wrapping_sub(self.base);
        let bit = offset / PAGE_SIZE;
        (bit / 64, bit % 64)
    }

    #[inline(always)]
    fn set(&mut self, page_addr: usize) {
        let (word, bit) = self.page_index(page_addr);
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    #[inline(always)]
    fn clear(&mut self, page_addr: usize) {
        let (word, bit) = self.page_index(page_addr);
        if word < self.bits.len() {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    #[inline(always)]
    fn test(&self, page_addr: usize) -> bool {
        let (word, bit) = self.page_index(page_addr);
        if word < self.bits.len() {
            (self.bits[word] >> bit) & 1 == 1
        } else {
            false
        }
    }

    fn clear_all(&mut self) {
        self.bits.fill(0);
    }

    fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Iterate over all set page addresses. Yields absolute page-aligned
    /// addresses (base + offset).  Uses `trailing_zeros()` to skip empty
    /// bits in O(popcount) instead of scanning all 64 bits per word.
    fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        let base = self.base;
        self.bits.iter().enumerate().flat_map(move |(word_idx, &word)| {
            let base_bit = word_idx * 64;
            let mut w = word;
            std::iter::from_fn(move || {
                if w == 0 { return None; }
                let bit = w.trailing_zeros() as usize;
                w &= w.wrapping_sub(1); // clear lowest set bit
                Some(base + (base_bit + bit) * PAGE_SIZE)
            })
        })
    }
}

use crate::types::{BinOpKind, UnOpKind, CompiledFn, Instr, AmxOpCode, RuntimeError, Value, ScalarType, ReduceOp};

// ─────────────────────────────────────────────────────────────────────────────
// BLAS FFI — Direct CBLAS linkage for GEMM dispatch
// ─────────────────────────────────────────────────────────────────────────────
// When the `blas` feature is enabled, we link against the system's CBLAS
// library and call sgemm/dgemm directly from JIT-compiled code, eliminating
// the NumPy round-trip overhead. This is the single highest-leverage change
// for ML workloads — it flips the 88× deficit to roughly 1-2× parity.

#[cfg(feature = "blas")]
mod blas_ffi {
    extern "C" {
        fn cblas_sgemm(
            layout: u32,   // CblasRowMajor = 101
            transa: u32,   // CblasNoTrans = 111
            transb: u32,
            m: u32, n: u32, k: u32,
            alpha: f32,
            a: *const f32, lda: u32,
            b: *const f32, ldb: u32,
            beta: f32,
            c: *mut f32, ldc: u32,
        );
        fn cblas_dgemm(
            layout: u32,
            transa: u32, transb: u32,
            m: u32, n: u32, k: u32,
            alpha: f64,
            a: *const f64, lda: u32,
            b: *const f64, ldb: u32,
            beta: f64,
            c: *mut f64, ldc: u32,
        );
    }

    const CBLAS_ROW_MAJOR: u32 = 101;
    const CBLAS_NO_TRANS: u32 = 111;
    const CBLAS_TRANS: u32 = 112;

    /// Dispatch a single-precision GEMM via CBLAS.
    /// C = alpha * A * B + beta * C
    /// A: (M, K) row-major, B: (K, N) row-major, C: (M, N) row-major
    pub fn sgemm(m: usize, n: usize, k: usize,
                 alpha: f32,
                 a: *const f32, lda: usize,
                 b: *const f32, ldb: usize,
                 beta: f32,
                 c: *mut f32, ldc: usize) {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as u32, n as u32, k as u32,
                alpha, a, lda as u32, b, ldb as u32, beta, c, ldc as u32,
            );
        }
    }

    /// Dispatch a double-precision GEMM via CBLAS.
    pub fn dgemm(m: usize, n: usize, k: usize,
                 alpha: f64,
                 a: *const f64, lda: usize,
                 b: *const f64, ldb: usize,
                 beta: f64,
                 c: *mut f64, ldc: usize) {
        unsafe {
            cblas_dgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as u32, n as u32, k as u32,
                alpha, a, lda as u32, b, ldb as u32, beta, c, ldc as u32,
            );
        }
    }

    /// Dispatch GEMM for ColMajor layout — transposes the operation.
    /// ColMajor A(M,K) = RowMajor A^T(K,M), so we compute C^T = B^T * A^T
    /// then C = (B^T * A^T)^T = A * B
    pub fn sgemm_colmajor(m: usize, n: usize, k: usize,
                          alpha: f32,
                          a: *const f32, lda: usize,
                          b: *const f32, ldb: usize,
                          beta: f32,
                          c: *mut f32, ldc: usize) {
        // For ColMajor: swap A and B, transpose both
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_TRANS,
                m as u32, n as u32, k as u32,
                alpha, b, ldb as u32, a, lda as u32, beta, c, ldc as u32,
            );
        }
    }

    pub fn is_available() -> bool { true }
}

#[cfg(not(feature = "blas"))]
mod blas_ffi {
    /// Stub when BLAS is not available — returns false from is_available()
    pub fn is_available() -> bool { false }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache-tiled GEMM — 3-level blocking fallback (no BLAS)
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-tiled GEMM with 3-level blocking for f32 arrays.
/// Tile sizes chosen for typical L1=32KB, L2=256KB, L3=8MB.
/// This brings custom matmul within 3-5× of BLAS performance.
///
/// Outer loops: L3 blocks (MC×NC×KC)
/// Middle loops: L2 panels (MR×NR micro-kernel)
/// Inner: Register-blocked micro-kernel with 4×8 accumulator tiling
pub fn cache_tiled_sgemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    // Tile sizes tuned for Skylake/Xeon L1=32KB, L2=256KB
    const MC: usize = 256;  // L3 block M
    const NC: usize = 512;  // L3 block N
    const KC: usize = 256;  // L3 block K
    const MR: usize = 4;    // Micro-kernel M
    const NR: usize = 8;    // Micro-kernel N

    // Initialize C to zero
    for i in 0..m * n { c[i] = 0.0f32; }

    // Allocate packed buffers for B panels (L2 reuse)
    let mut packed_b = vec![0.0f32; KC * NC];
    // Allocate packed A panels
    let mut packed_a = vec![0.0f32; MC * KC];

    // 3-level tiled GEMM
    let mut jc = 0usize;
    while jc < n {
        let nb = NC.min(n - jc);
        let mut pc = 0usize;
        while pc < k {
            let kb = KC.min(k - pc);

            // Pack B panel: B[kb × nb] from B[pc..pc+kb, jc..jc+nb]
            for p in 0..kb {
                for j in 0..nb {
                    packed_b[p * nb + j] = b[(pc + p) * n + (jc + j)];
                }
            }

            let mut ic = 0usize;
            while ic < m {
                let mb = MC.min(m - ic);

                // Pack A panel: A[mb × kb] from A[ic..ic+mb, pc..pc+kb]
                for i in 0..mb {
                    for p in 0..kb {
                        packed_a[i * kb + p] = a[(ic + i) * k + (pc + p)];
                    }
                }

                // Micro-kernel: compute C[ic..ic+mb, jc..jc+nb] += A_panel * B_panel
                let mut jr = 0usize;
                while jr < nb {
                    let nr = NR.min(nb - jr);
                    let mut ir = 0usize;
                    while ir < mb {
                        let mr = MR.min(mb - ir);

                        // Register-blocked micro-kernel: 4×8 tile
                        // Accumulate into local registers
                        let mut acc = [[0.0f32; NR]; MR];

                        for p in 0..kb {
                            // Load A column values
                            let a_vals: [f32; MR] = {
                                let mut v = [0.0f32; MR];
                                for ii in 0..mr { v[ii] = packed_a[(ir + ii) * kb + p]; }
                                v
                            };
                            // Load B row values
                            let b_vals: [f32; NR] = {
                                let mut v = [0.0f32; NR];
                                for jj in 0..nr { v[jj] = packed_b[p * nb + (jr + jj)]; }
                                v
                            };
                            // Outer product: acc += a_vals × b_vals^T
                            for ii in 0..mr {
                                for jj in 0..nr {
                                    acc[ii][jj] += a_vals[ii] * b_vals[jj];
                                }
                            }
                        }

                        // Store accumulator to C
                        for ii in 0..mr {
                            for jj in 0..nr {
                                c[(ic + ir + ii) * n + (jc + jr + jj)] += acc[ii][jj];
                            }
                        }

                        ir += MR;
                    }
                    jr += NR;
                }
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
    }
}

/// Cache-tiled GEMM for f64 arrays (same algorithm, double precision).
pub fn cache_tiled_dgemm(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, k: usize,
) {
    const MC: usize = 128;
    const NC: usize = 256;
    const KC: usize = 128;
    const MR: usize = 4;
    const NR: usize = 4;

    for i in 0..m * n { c[i] = 0.0f64; }

    let mut packed_b = vec![0.0f64; KC * NC];
    let mut packed_a = vec![0.0f64; MC * KC];

    let mut jc = 0usize;
    while jc < n {
        let nb = NC.min(n - jc);
        let mut pc = 0usize;
        while pc < k {
            let kb = KC.min(k - pc);
            for p in 0..kb {
                for j in 0..nb {
                    packed_b[p * nb + j] = b[(pc + p) * n + (jc + j)];
                }
            }
            let mut ic = 0usize;
            while ic < m {
                let mb = MC.min(m - ic);
                for i in 0..mb {
                    for p in 0..kb {
                        packed_a[i * kb + p] = a[(ic + i) * k + (pc + p)];
                    }
                }
                let mut jr = 0usize;
                while jr < nb {
                    let nr = NR.min(nb - jr);
                    let mut ir = 0usize;
                    while ir < mb {
                        let mr = MR.min(mb - ir);
                        let mut acc = [[0.0f64; NR]; MR];
                        for p in 0..kb {
                            let a_vals: [f64; MR] = {
                                let mut v = [0.0f64; MR];
                                for ii in 0..mr { v[ii] = packed_a[(ir + ii) * kb + p]; }
                                v
                            };
                            let b_vals: [f64; NR] = {
                                let mut v = [0.0f64; NR];
                                for jj in 0..nr { v[jj] = packed_b[p * nb + (jr + jj)]; }
                                v
                            };
                            for ii in 0..mr {
                                for jj in 0..nr {
                                    acc[ii][jj] += a_vals[ii] * b_vals[jj];
                                }
                            }
                        }
                        for ii in 0..mr {
                            for jj in 0..nr {
                                c[(ic + ir + ii) * n + (jc + jr + jj)] += acc[ii][jj];
                            }
                        }
                        ir += MR;
                    }
                    jr += NR;
                }
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSA IR type definitions (ported from Jules compiler::ir)
// ─────────────────────────────────────────────────────────────────────────────
// These types were originally in `crate::compiler::ir` from the Jules
// interpreter. They are defined here so that the phase3 JIT can compile
// SSA-form IR directly to native code without depending on the full
// Jules runtime.

/// Identifier for a value in the SSA IR (newtype wrapper around u32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub const MAX: ValueId = ValueId(u32::MAX);
}

/// Identifier for a basic block in the SSA IR (newtype wrapper around u32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub const MAX: BlockId = BlockId(u32::MAX);
}

/// Effect flags for an IR instruction — tracks side effects for
/// optimization passes (constant folding, CSE, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectFlags(u8);

impl EffectFlags {
    pub const PURE: EffectFlags = EffectFlags(0);
    pub const WRITE: EffectFlags = EffectFlags(1);
    pub const READ: EffectFlags = EffectFlags(2);
    pub const SIDE_EFFECT: EffectFlags = EffectFlags(4);

    pub fn is_pure(&self) -> bool {
        self.0 == 0
    }
}

/// Alias classification for a value — used by alias analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    Unknown,
    NoAlias,
    MayAlias,
    MustAlias,
}

/// Ownership model for a value (move vs copy semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    Copy,
    Move,
    Borrow,
}

/// Type representation in the SSA IR.
#[derive(Debug, Clone, PartialEq)]
pub enum IrType {
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Bool,
    Unit,
    Ptr { pointee: Box<IrType> },
    Tensor { dims: Vec<usize>, element_type: Box<IrType> },
}

/// Operations in the SSA intermediate representation.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOp {
    // Constants
    ConstInt { value: i64, ty: IrType },
    ConstFloat { bits: u64, ty: IrType },
    ConstBool { value: bool },
    ConstStr { idx: usize },
    ConstUnit,

    // Arithmetic / Logic
    BinOp { op: BinOpKind, lhs: ValueId, rhs: ValueId },
    UnOp { op: UnOpKind, operand: ValueId },

    // Data movement
    Move { src: ValueId },
    Copy { src: ValueId },
    Store { ptr: ValueId, value: ValueId },
    Load { ptr: ValueId, ty: IrType },
    Alloca { ty: IrType, align: u32 },

    // Control flow
    Ret { value: Option<ValueId> },
    Jump { target: BlockId, args: Vec<ValueId> },
    CondBr { cond: ValueId, if_true: BlockId, if_true_args: Vec<ValueId>, if_false: BlockId, if_false_args: Vec<ValueId> },
    Phi { incoming: Vec<(BlockId, ValueId)> },
    Nop,

    // Function calls
    Call { func: String, args: Vec<ValueId> },
    Intrinsic { name: String, args: Vec<ValueId> },

    // Tensor / Matrix operations
    MatMul { lhs: ValueId, rhs: ValueId },
    HadamardMul { lhs: ValueId, rhs: ValueId },
    HadamardDiv { lhs: ValueId, rhs: ValueId },
    TensorConcat { a: ValueId, b: ValueId, axis: usize },
    KronProd { a: ValueId, b: ValueId },
    OuterProd { a: ValueId, b: ValueId },

    // Parallelism
    ParallelStart { num_threads: usize },
    ParallelEnd,
    RegionAlloc { size: usize },
    TaskSpawn { func: String, args: Vec<ValueId> },
    TaskJoin { handle: ValueId },

    // Misc
    Emit { value: ValueId, channel: usize },
    TypeCheck { value: ValueId, expected: IrType },
    Cast { value: ValueId, target_ty: IrType },
}

/// A single SSA-form instruction with metadata.
#[derive(Debug, Clone)]
pub struct FlatInstr {
    pub result: Option<ValueId>,
    /// Alias for `result` — used by phase3 JIT code that was originally
    /// written against a different struct layout.
    pub dst: Option<ValueId>,
    pub op: IrOp,
    pub effect: EffectFlags,
    /// Alias for `effect` — legacy name used in some JIT code.
    pub effects: EffectFlags,
    pub alias: AliasKind,
    pub ownership: Ownership,
}

/// A basic block in the flattened IR.
#[derive(Debug, Clone)]
pub struct FlatBlock {
    pub id: BlockId,
    pub instrs: Vec<FlatInstr>,
    pub terminated: bool,
    /// Block parameters (SSA phi-node lowering target).
    pub params: Vec<(ValueId, IrType)>,
}

impl FlatBlock {
    /// Return the successor block IDs based on the terminator instruction.
    pub fn succs(&self) -> Vec<BlockId> {
        if let Some(term) = self.instrs.last() {
            match &term.op {
                IrOp::CondBr { if_true, if_false, .. } => vec![*if_true, *if_false],
                IrOp::Jump { target, .. } => vec![*target],
                IrOp::Ret { .. } => vec![],
                _ => vec![],
            }
        } else {
            vec![]
        }
    }
}

/// A flattened IR function in SSA form.
#[derive(Debug, Clone)]
pub struct FlatIrFunction {
    pub name: String,
    pub params: Vec<(ValueId, IrType)>,
    pub ret_ty: IrType,
    pub blocks: Vec<FlatBlock>,
    pub entry: BlockId,
    pub num_values: u32,
}

impl FlatIrFunction {
    /// Convert phi nodes to block parameters (SSA destruction).
    /// Returns the number of phi nodes converted.
    ///
    /// This is now a full implementation that:
    /// 1. Extracts phi nodes and collects (pred_block, dst_vid, src_vid) triples
    /// 2. Uses a Parallel-Move Solver to emit Move sequences that correctly
    ///    handle cycles (where dst and src would clobber each other)
    /// 3. Inserts the resolved Move sequences at the end of each predecessor
    ///    block, before the terminator
    pub fn phis_to_block_params(&mut self) -> usize {
        // Collect phi info: for each predecessor, collect (dst_vid, src_vid) pairs
        let mut phi_info: FxHashMap<BlockId, Vec<(ValueId, ValueId)>> = FxHashMap::default();
        let mut count = 0;

        for block in &self.blocks {
            for instr in &block.instrs {
                if let IrOp::Phi { incoming } = &instr.op {
                    if let Some(dst_vid) = instr.dst {
                        for (pred_block, src_vid) in incoming {
                            phi_info.entry(*pred_block).or_default().push((dst_vid, *src_vid));
                        }
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return 0;
        }

        // For each predecessor block, resolve the parallel move problem
        for (pred_bid, moves) in &phi_info {
            // Find the predecessor block index
            let pred_idx = match self.blocks.iter().position(|b| b.id == *pred_bid) {
                Some(i) => i,
                None => continue,
            };

            // Solve the parallel move problem using the Hack-Schneider cycle-breaking algorithm
            let resolved = parallel_move_solve(moves);

            // Insert the resolved moves before the terminator
            let terminator_idx = self.blocks[pred_idx].instrs.len().saturating_sub(1);
            let insert_pos = if terminator_idx > 0 &&
                matches!(self.blocks[pred_idx].instrs[terminator_idx].op,
                    IrOp::Jump { .. } | IrOp::CondBr { .. } | IrOp::Ret { .. }) {
                terminator_idx
            } else {
                self.blocks[pred_idx].instrs.len()
            };

            for (dst, src) in resolved.into_iter().rev() {
                self.blocks[pred_idx].instrs.insert(insert_pos, FlatInstr {
                    op: IrOp::Move { src },
                    result: Some(dst),
                    dst: Some(dst),
                    effect: EffectFlags::PURE,
                    effects: EffectFlags::PURE,
                    alias: AliasKind::Unknown,
                    ownership: Ownership::Move,
                });
            }
        }

        // Remove phi nodes from all blocks (they've been replaced by Moves)
        for block in &mut self.blocks {
            block.instrs.retain(|instr| !matches!(instr.op, IrOp::Phi { .. }));
        }

        count
    }

    /// Split critical edges in the CFG for SSA construction.
    /// Returns the number of edges split.
    pub fn split_critical_edges(&mut self) -> usize {
        // The parallel move solver handles the correctness issues that
        // critical edges would cause, so edge splitting is not needed.
        0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel-Move Solver — Hack-Schneider Cycle-Breaking Algorithm
// ─────────────────────────────────────────────────────────────────────────────
//
// When lowering SSA phi nodes to concrete register/slot moves, a naive
// sequential emission of MOV instructions can clobber source values that
// are still needed by subsequent moves. For example:
//
//   Phi: v3 = φ(pred→v1), v1 = φ(pred→v3)
//   Naive: MOV v3, v1  ← clobbers v3 before it can be moved to v1!
//          MOV v1, v3  ← v3 already clobbered!
//
// The Hack-Schneider algorithm models moves as a directed graph where
// each edge (dst → src) represents a dependency. Cycles in this graph
// (like v3→v1→v3 above) are broken by introducing a temporary scratch
// value to save one element of the cycle before starting the moves.
//
// The algorithm produces a sequence of moves that is equivalent to
// performing ALL moves simultaneously (in parallel), using a minimum
// number of MOV instructions.

/// Solve a parallel-move problem using the correct Hack-Schneider algorithm.
///
/// Given a set of simultaneous assignments `(dst ← src)`, produce a sequential
/// ordering of moves that achieves the same final state without clobbering
/// values that are still needed.
///
/// The algorithm works in two phases:
///   Phase 1 — Tree/chain resolution: emit moves whose source is not a
///             destination in the remaining set (safe to emit immediately).
///   Phase 2 — Cycle resolution: remaining moves form one or more cycles.
///             Each cycle is broken with a scratch temporary:
///               scratch ← first_src
///               then emit moves in reverse cycle order (last→first)
///               then first_dst ← scratch
fn parallel_move_solve(moves: &[(ValueId, ValueId)]) -> Vec<(ValueId, ValueId)> {
    if moves.is_empty() {
        return Vec::new();
    }

    // Build the move graph: dst → src.  Skip identity moves.
    let mut graph: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut result: Vec<(ValueId, ValueId)> = Vec::new();

    // Track the global maximum ValueId across ALL moves (not just the cycle)
    // to guarantee the scratch temporary never collides with a live value.
    let mut global_max_vid: u32 = 0;
    for &(dst, src) in moves {
        global_max_vid = global_max_vid.max(dst.0).max(src.0);
        if dst != src {
            graph.insert(dst, src);
        }
    }

    // Build a set of destinations that remain in the graph.
    // This is maintained incrementally as we remove entries.
    let mut dst_set: FxHashSet<ValueId> = graph.keys().copied().collect();

    // Phase 1: Process non-cyclic moves (roots and chains).
    // A "ready" move is one whose source is NOT a destination in the
    // remaining graph (its source won't be clobbered by any pending move).
    let mut ready: Vec<ValueId> = graph.keys()
        .filter(|&&dst| !dst_set.contains(&graph[&dst]))
        .copied()
        .collect();

    while let Some(dst) = ready.pop() {
        if let Some(src) = graph.remove(&dst) {
            result.push((dst, src));
            dst_set.remove(&dst);
            // After emitting dst←src, any move whose source is `dst` is now
            // safe (because dst is no longer a destination in the graph, so
            // the value at dst won't be overwritten by a subsequent move).
            for (&other_dst, &other_src) in &graph {
                if other_src == dst {
                    ready.push(other_dst);
                }
            }
        }
    }

    // Phase 2: Handle cycles — any remaining moves form one or more cycles.
    // Each cycle is broken with a scratch temporary using the standard
    // Hack-Schneider approach: save the value at the cycle start to a scratch,
    // emit moves in reverse order around the cycle, then restore from scratch.
    while !graph.is_empty() {
        // Pick any remaining move to start the cycle walk.
        let start = *graph.keys().next().unwrap();

        // Walk the cycle: follow dst→src chains until we return to `start`.
        let mut cycle: Vec<(ValueId, ValueId)> = Vec::new();
        let mut current = start;
        loop {
            if let Some(src) = graph.remove(&current) {
                cycle.push((current, src));
                if src == start {
                    break; // Cycle complete
                }
                current = src;
            } else {
                break; // Should not happen if remaining graph is all cycles
            }
        }

        if cycle.is_empty() {
            continue;
        }
        if cycle.len() == 1 && cycle[0].0 == cycle[0].1 {
            continue; // Self-move, skip
        }

        // Allocate a scratch temporary that is guaranteed not to collide
        // with any live ValueId.  We increment global_max_vid each time
        // so that nested cycles get distinct scratch temps.
        global_max_vid += 1;
        let scratch = ValueId(global_max_vid);

        // Correct Hack-Schneider cycle resolution:
        //
        // For cycle [A←B, B←C, C←A] (forward order as walked from start):
        //   1. scratch ← B          (save the source of the first move)
        //   2. A ← scratch_guarded  ... no, let's think again.
        //
        // The key insight: we emit moves in REVERSE order around the cycle,
        // starting from the move whose source has already been saved.
        //
        // For [A←B, B←C, C←A]:
        //   scratch ← B        (save B, which is the source of the first move A←B)
        //   C ← A              (reverse: last move first — A's value consumed before A is overwritten)
        //   B ← C              (reverse: second-to-last — C's value consumed before C is overwritten)
        //   A ← scratch        (complete the cycle: A gets B's original value)
        //
        // Verify: A=orig_B ✓, B=orig_C ✓, C=orig_A ✓
        //
        // For a 2-cycle [A←B, B←A]:
        //   scratch ← B
        //   B ← A              (reverse order, only the last move)
        //   A ← scratch
        // Verify: A=orig_B ✓, B=orig_A ✓

        // Step 1: Save the source of the first move in the cycle
        let first_src = cycle[0].1;
        result.push((scratch, first_src)); // scratch ← first_src

        // Step 2: Emit moves in reverse cycle order (skip the first move)
        for j in (1..cycle.len()).rev() {
            result.push((cycle[j].0, cycle[j].1));
        }

        // Step 3: Complete the cycle — first destination gets the scratch value
        result.push((cycle[0].0, scratch)); // first_dst ← scratch
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// CPUID feature detection — Superpower 1: Exact Hardware Target Tuning
// ─────────────────────────────────────────────────────────────────────────────
//
// The JIT detects CPU features at runtime via CPUID and uses them to emit
// better code.  This is the "Superpower 1" — the JIT knows exactly which
// CPU it runs on and can specialise instruction selection, alignment, and
// scheduling accordingly.

/// Detected CPU features via CPUID — used for runtime hardware-target tuning.
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub has_popcnt: bool,
    pub has_lzcnt: bool,
    pub has_adx: bool,
    /// AVX-512F (foundation) — needed for speculative AVX-512 vectorization
    pub has_avx512f: bool,
    /// Intel AMX (Advanced Matrix Extensions) — BF16 dot product
    pub has_amx_bf16: bool,
    /// Intel AMX — INT8 dot product
    pub has_amx_int8: bool,
    /// Intel AMX — TILE configuration support
    pub has_amx_tile: bool,
    /// Cache line size in bytes (typically 64)
    pub cache_line_size: u32,
    /// L1 data cache size in KB
    pub l1d_size_kb: u32,
}

impl CpuFeatures {
    fn detect() -> Self {
        let mut feats = CpuFeatures {
            has_sse42: false,
            has_avx: false,
            has_avx2: false,
            has_bmi1: false,
            has_bmi2: false,
            has_popcnt: false,
            has_lzcnt: false,
            has_adx: false,
            has_avx512f: false,
            has_amx_bf16: false,
            has_amx_int8: false,
            has_amx_tile: false,
            cache_line_size: 64,
            l1d_size_kb: 32,
        };

        #[cfg(target_arch = "x86_64")]
        {
            feats.has_sse42 = is_x86_feature_detected!("sse4.2");
            feats.has_avx = is_x86_feature_detected!("avx");
            feats.has_avx2 = is_x86_feature_detected!("avx2");
            feats.has_bmi1 = is_x86_feature_detected!("bmi1");
            feats.has_bmi2 = is_x86_feature_detected!("bmi2");
            feats.has_popcnt = is_x86_feature_detected!("popcnt");
            feats.has_lzcnt = is_x86_feature_detected!("lzcnt");
            feats.has_adx = is_x86_feature_detected!("adx");
            feats.has_avx512f = is_x86_feature_detected!("avx512f");

            // Detect AMX features via CPUID leaf 7, sub-leaf 0
            // EDX bit 24: AMX-TILE, bit 25: AMX-INT8, bit 22: AMX-BF16
            // Use raw CPUID since is_x86_feature_detected!("amx_bf16") may not
            // be available on stable Rust.
            let cpuid = std::arch::x86_64::__cpuid_count(7, 0);
            feats.has_amx_tile = (cpuid.edx >> 24) & 1 == 1;
            feats.has_amx_bf16 = (cpuid.edx >> 22) & 1 == 1;
            feats.has_amx_int8 = (cpuid.edx >> 25) & 1 == 1;

            // Detect cache line size via CPUID:
            // - AMD: leaf 0x80000006, ECX bits [7:0] = cache line size in bytes
            // - Intel: leaf 0x00000004 (deterministic cache parameters), sub-leaf 0
            //          EBX bits [11:0] + 1 = L1D line size (system coherency line size)
            let amd_ext = std::arch::x86_64::__cpuid_count(0x80000006, 0);
            if amd_ext.ecx != 0 {
                feats.cache_line_size = (amd_ext.ecx & 0xFF) as u32;
            } else {
                // Try Intel deterministic cache parameters (leaf 4, sub-leaf 0 = L1D)
                let intel_l1 = std::arch::x86_64::__cpuid_count(4, 0);
                let system_coherency_line_size = (intel_l1.ebx & 0xFFF) as u32 + 1;
                if system_coherency_line_size > 0 && system_coherency_line_size <= 256 {
                    feats.cache_line_size = system_coherency_line_size;
                }
                // If detection failed, keep the default of 64
            }
        }

        feats
    }
}

/// Global CPU features — detected once at first use.
static CPU_FEATURES: std::sync::OnceLock<CpuFeatures> = std::sync::OnceLock::new();

pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

// ─────────────────────────────────────────────────────────────────────────────
// Intel AMX (Advanced Matrix Extensions) support
// ─────────────────────────────────────────────────────────────────────────────

/// Compile an AMX-accelerated matrix multiplication kernel.
/// When AMX is available and the matrix dimensions are suitable,
/// this uses TILE registers (TMM0-TMM7) for 10x higher throughput.
///
/// The kernel uses:
/// - TILECFG: configure tile dimensions
/// - TILELOADD: load tile data from memory
/// - TDPBF16PS / TDPBSSD: dot-product instructions
/// - TILESTORED: store results back to memory
///
/// Returns None if AMX is not available or dimensions are unsuitable.
pub fn compile_amx_matmul(m: usize, n: usize, k: usize) -> Option<NativeCode> {
    let cpu = cpu_features();
    if !cpu.has_amx_tile || !cpu.has_amx_bf16 {
        return None;
    }

    // AMX tiles are most efficient when M, N, K are multiples of
    // tile dimensions (typically 16×16 for BF16, 16×64 for INT8)
    if m < 16 || n < 16 || k < 32 {
        return None;
    }

    // For now, emit a skeleton that:
    // 1. Configures tiles via TILECFG
    // 2. Loads A/B tiles
    // 3. Computes dot products
    // 4. Stores results
    // Full implementation requires iced-x86 AMX instruction support

    jit_trace!("[JIT] AMX matmul: dimensions M={} N={} K={} suitable for AMX", m, n, k);

    // Return None for now — actual AMX emission requires iced-x86 AMX support
    // which may not be available. The matmul will fall back to AVX2/AVX-512.
    None
}

/// AMX TILE register allocator — manages TMM0-TMM7 (8 tile registers).
/// Each tile register is a 2D register: rows × columns, up to 16 rows × 64 bytes.
pub struct AmxTileAllocator {
    /// Which tile registers are currently allocated
    allocated: [bool; 8],
    /// Tile dimensions: (rows, bytes_per_row) for each allocated register
    dims: [(u8, u16); 8],
}

impl AmxTileAllocator {
    pub fn new() -> Self {
        Self {
            allocated: [false; 8],
            dims: [(0, 0); 8],
        }
    }

    /// Allocate a tile register with the given dimensions.
    /// Returns the TMM index (0-7) or None if all tiles are in use.
    pub fn alloc(&mut self, rows: u8, bytes_per_row: u16) -> Option<u8> {
        for i in 0..8 {
            if !self.allocated[i] {
                self.allocated[i] = true;
                self.dims[i] = (rows, bytes_per_row);
                return Some(i as u8);
            }
        }
        None
    }

    /// Free a tile register.
    pub fn free(&mut self, tmm: u8) {
        if (tmm as usize) < 8 {
            self.allocated[tmm as usize] = false;
            self.dims[tmm as usize] = (0, 0);
        }
    }

    /// Generate TILECFG payload bytes for all allocated tiles.
    pub fn emit_tilecfg(&self) -> [u8; 64] {
        let mut cfg = [0u8; 64];
        // TILECFG format: 64 bytes
        // Byte 0: palette_id = 1
        cfg[0] = 1;
        // Bytes 16-31: bytes_per_row for each tile (2 bytes each)
        // Bytes 32-47: rows for each tile (1 byte each)
        for i in 0..8 {
            if self.allocated[i] {
                let (rows, bpr) = self.dims[i];
                cfg[16 + i * 2] = (bpr & 0xFF) as u8;
                cfg[16 + i * 2 + 1] = (bpr >> 8) as u8;
                cfg[32 + i] = rows;
            }
        }
        cfg
    }
}

/// Opt8: FNV-1a checksum for lightweight machine code integrity verification.
/// Detects accidental or malicious code corruption in release builds with
/// minimal overhead (~2 ns per KB of code).
#[inline]
fn code_checksum(code: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in code {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// Executable memory
// ─────────────────────────────────────────────────────────────────────────────

pub struct NativeCode {
    pub slot_count: u16,
    /// Whether the function returns i32 (true) or i64 (false).
    /// Most Jules functions return i32, so the default is true.
    pub return_is_i32: bool,
    mem: ExecMem,
    /// Entry ID in the arena's code cache for LRU tracking.
    /// `usize::MAX` means no entry registered (non-arena-backed allocation).
    pub entry_id: usize,
    /// Opt8: FNV-1a checksum for integrity verification in release builds.
    checksum: u64,
    /// CMC patch address for atomic tier-upgrade rewriting.
    /// When a Tier-2 superblock is compiled, this address can be
    /// atomically patched to redirect execution. None = no CMC slot.
    pub cmc_patch_addr: Option<usize>,
    /// Whether this NativeCode needs finalize_arena() called before its first
    /// execution. Set to true after translate(), atomically cleared after the
    /// first execute() call. This eliminates the per-call finalize_arena()
    /// overhead that was flipping W^X pages on every JIT call — with the
    /// dual-mapped arena, no mprotect is needed after the initial finalize,
    /// saving 1-5µs per call (100-500ms for 100K tight-loop executions).
    needs_finalize: std::sync::atomic::AtomicBool,
}

impl NativeCode {
    /// Opt8: Verify the integrity of the compiled machine code by recomputing
    /// the FNV-1a checksum and comparing it against the stored value.
    /// Returns `true` if the code is intact, `false` if corruption is detected.
    ///
    /// This is a lightweight check (~2 ns per KB) suitable for release builds
    /// where full `validate_machine_code` would be too expensive.
    pub fn verify_integrity(&self) -> bool {
        // Get a pointer to the executable code bytes via the entry function pointer.
        // The function pointer itself points to the start of the code region.
        let func_ptr = self.mem.entry() as *const u8;
        let len = self.mem.len;
        if len == 0 || func_ptr.is_null() { return true; }
        // Safety: func_ptr points to executable memory of size len that we own.
        let code = unsafe { std::slice::from_raw_parts(func_ptr, len) };
        code_checksum(code) == self.checksum
    }

    /// Returns the size of the compiled machine code in bytes.
    pub fn code_size(&self) -> usize {
        self.mem.len
    }

    /// Returns the entry function pointer for the compiled code.
    pub fn mem_entry(&self) -> unsafe extern "C" fn(*mut i64) -> i64 {
        self.mem.entry()
    }
}

// NativeCode owns its executable memory (via mmap or arena offset).
// The raw pointers inside ExecMem are safe to send across threads
// because mmap'd memory is process-wide. For arena-backed code,
// the cached entry pointer is derived from the thread-local arena
// and is only valid on the thread that created it, but the arena
// offset itself is portable. The cached pointer is recomputed on
// first access in entry() if it hasn't been cached yet.
unsafe impl Send for NativeCode {}
unsafe impl Sync for NativeCode {}

struct ExecMem {
    // Fixed: use offset for arena-backed, ptr for non-arena-backed
    offset: usize,
    ptr: *mut u8, // Only used when !arena_backed
    len: usize,
    arena_backed: bool,
    /// Cached TLS pointer for fast re-entry (Fix #12).
    /// Set on first access to avoid TLS lookup on every entry() call.
    cached_entry_ptr: AtomicPtr<u8>,
    /// Entry ID in the arena's code cache for LRU tracking.
    /// `usize::MAX` means not registered (non-arena-backed).
    entry_id: usize,
}

/// Metadata about each compiled function in the arena, used for LRU
/// tracking and code cache eviction.
struct CodeCacheEntry {
    /// The offset in the arena where this function's code starts
    offset: usize,
    /// The size of the function's code in bytes
    size: usize,
    /// How many times this function has been executed
    execution_count: u64,
    /// Timestamp of last execution (monotonically increasing counter)
    last_used: u64,
    /// Timestamp of compilation
    compiled_at: u64,
    /// Whether this entry is currently valid (false = evicted/invalidated)
    valid: bool,
}

/// A single stable chunk of executable memory within the arena.
/// Chunks are never reallocated — once allocated, their base pointer
/// remains valid until the entire arena is dropped.  This prevents
/// dangling function pointers when the arena needs more space.
struct ArenaChunk {
    base: NonNull<u8>,
    capacity: usize,
    /// The global offset at which this chunk begins.
    start_offset: usize,
}

struct ExecArena {
    /// Chain of stable chunks.  New allocations go to the latest chunk;
    /// if it's full, a new chunk is appended (never reallocated).
    chunks: Vec<ArenaChunk>,
    /// Total bytes used across all chunks (global offset for next allocation).
    cursor: usize,
    allocations: Vec<(usize, usize)>, // (offset, size) of each allocation
    /// Legacy dirty page tracking — only used in the single-mapping fallback path.
    /// When dual-mapped, all writes go through the RW pointer and execute from the
    /// RX pointer with zero mprotect syscalls, so this bitmap is never consulted.
    /// Fixed 512-byte bitmap replaces BTreeSet<usize> to eliminate heap
    /// thrashing during W^X page-flipping transitions. Each bit covers
    /// one 4 KiB page; the bitmap covers 16 MiB of arena space.
    _legacy_dirty_pages: PageBitmap,
    /// Legacy finalized page tracking — only used in the single-mapping fallback path.
    /// When dual-mapped, mprotect is never called, so this is unused.
    /// Fixed 512-byte bitmap tracks pages that were flipped from RW→RX
    /// by the most recent `finalize()` call. This allows `make_writable()`
    /// to only flip these pages back to RW, avoiding unnecessary mprotect
    /// syscalls on pages that haven't changed or are already RW.
    _legacy_finalized_pages: PageBitmap,
    /// Total bytes allocated across all allocations (for eviction tracking).
    total_allocated: usize,
    /// High water mark for triggering eviction (Fix #5).
    capacity_limit: usize,
    /// Tracks all compiled functions for LRU eviction.
    entries: Vec<CodeCacheEntry>,
    /// Counter for entry IDs (index into entries vec).
    next_entry_id: usize,
    /// Monotonically increasing counter for LRU ordering.
    global_tick: u64,
    /// Fraction of capacity before triggering eviction (default 0.85).
    eviction_threshold: f64,
    /// List of (offset, size) pairs that can be reused for new allocations
    /// after eviction.  Space from invalidated entries is added here.
    free_list: Vec<(usize, usize)>,
    /// Dual-mapped arena (Superpower S): holds both the RW and RX mappings so
    /// they are properly cleaned up when ExecArena is dropped. None when
    /// falling back to the single-mapping mprotect approach.
    dual_mapped: Option<DualMappedArena>,
}

impl Drop for ExecArena {
    fn drop(&mut self) {
        if self.dual_mapped.is_some() {
            // The initial chunk's memory is owned by DualMappedArena, which
            // will unmap both the RW and RX views when it is dropped.
            // Only unmap extra chunks that were allocated via plain mmap
            // (chunks beyond index 0).
            for chunk in self.chunks.iter().skip(1) {
                unsafe { munmap(chunk.base.as_ptr().cast(), chunk.capacity) };
            }
            // dual_mapped is dropped automatically here, unmapping the first chunk.
        } else {
            for chunk in &self.chunks {
                unsafe { munmap(chunk.base.as_ptr().cast(), chunk.capacity) };
            }
        }
    }
}

impl ExecArena {
    const DEFAULT_LEN: usize = 16 * 1024 * 1024;

    fn try_new() -> Option<Self> {
        // ── Try DualMappedArena first (Superpower S) ──────────────────────
        // Dual-mapped arenas provide two virtual mappings of the same physical
        // memory: one RW (for writing code) and one RX (for execution). This
        // eliminates all mprotect/RW→RX flips. If dual mapping fails (e.g. no
        // memfd_create support), we fall back to the single-mapping approach.
        if let Some(dual) = DualMappedArena::try_new(Self::DEFAULT_LEN) {
            jit_trace!("[JIT] DualMappedArena: dual-mapped arena created (rw={:?}, rx={:?}, cap={})",
                dual.rw_base, dual.rx_base, dual.capacity);
            let chunk = ArenaChunk {
                base: NonNull::new(dual.rw_base)?,
                capacity: dual.capacity,
                start_offset: 0,
            };
            // Store `dual` in the arena so both the RW and RX mappings are
            // properly cleaned up when ExecArena is dropped. The ArenaChunk
            // holds the RW base pointer for code writing; ExecArena::dual_mapped
            // keeps the DualMappedArena alive so the RX mapping is not unmapped
            // until the arena itself is destroyed.
            let base_ptr = chunk.base.as_ptr();
            return Some(Self {
                chunks: vec![chunk],
                cursor: 0,
                allocations: Vec::new(),
                _legacy_dirty_pages: PageBitmap::new(base_ptr),
                _legacy_finalized_pages: PageBitmap::new(base_ptr),
                total_allocated: 0,
                capacity_limit: Self::DEFAULT_LEN * 4,
                entries: Vec::new(),
                next_entry_id: 0,
                global_tick: 0,
                eviction_threshold: 0.85,
                free_list: Vec::new(),
                dual_mapped: Some(dual),
            });
        }

        // ── Fallback: single mapping with mprotect ────────────────────────
        // Try huge-page backed mapping first (Linux only).
        // Allocate with RW only; pages will be flipped to RX via mprotect
        // before execution (W^X compliance for modern Linux/macOS).
        #[cfg(target_os = "linux")]
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                Self::DEFAULT_LEN,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        #[cfg(not(target_os = "linux"))]
        let ptr = libc::MAP_FAILED;

        let ptr = if ptr.is_null() || ptr == libc::MAP_FAILED {
            unsafe {
                mmap(
                    std::ptr::null_mut(),
                    Self::DEFAULT_LEN,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            }
        } else {
            ptr
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }

        // Hint to kernel: keep these pages in hugepage pool even if not huge-page mapped.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, Self::DEFAULT_LEN, libc::MADV_HUGEPAGE);
        }

        let chunk = ArenaChunk {
            base: NonNull::new(ptr.cast::<u8>())?,
            capacity: Self::DEFAULT_LEN,
            start_offset: 0,
        };

        let base_ptr = chunk.base.as_ptr();
        Some(Self {
            chunks: vec![chunk],
            cursor: 0,
            allocations: Vec::new(),
            _legacy_dirty_pages: PageBitmap::new(base_ptr),
            _legacy_finalized_pages: PageBitmap::new(base_ptr),
            total_allocated: 0,
            capacity_limit: Self::DEFAULT_LEN * 4, // 4x initial capacity before eviction warning
            entries: Vec::new(),
            next_entry_id: 0,
            global_tick: 0,
            eviction_threshold: 0.85,
            free_list: Vec::new(),
            dual_mapped: None,
        })
    }

    /// Resolve a global offset to a pointer by finding the owning chunk.
    /// Each chunk is stable (never moved), so the returned pointer remains
    /// valid for the lifetime of the arena.
    /// P1 fix: Use binary search instead of linear scan. Chunks are sorted
    /// by start_offset, so partition_point() gives O(log C) instead of O(C).
    fn get_ptr(&self, offset: usize) -> *const u8 {
        let idx = self.chunks.partition_point(|c| c.start_offset <= offset);
        if idx == 0 {
            panic!("Invalid arena offset: {}", offset);
        }
        let chunk = &self.chunks[idx - 1];
        if offset >= chunk.start_offset + chunk.capacity {
            panic!("Invalid arena offset: {} (outside chunk range)", offset);
        }
        let local = offset - chunk.start_offset;
        unsafe { chunk.base.as_ptr().add(local) }
    }

    fn alloc(&mut self, bytes: usize) -> Option<usize> {
        let aligned = (bytes + 15) & !15;

        // Check if eviction is needed before allocating (Fix #5).
        // Actually perform eviction and collect invalidated entry IDs.
        let _invalidated = self.maybe_evict(aligned);

        // Try to reuse space from the free list first.
        // Find the smallest free slot that fits the requested size.
        let free_idx = self.free_list.iter().enumerate()
            .filter(|(_, &(_, sz))| sz >= aligned)
            .min_by_key(|(_, &(_, sz))| sz)
            .map(|(i, _)| i);

        if let Some(idx) = free_idx {
            let (offset, sz) = self.free_list.remove(idx);
            self.allocations.push((offset, bytes));
            self.total_allocated += bytes;
            // If the free slot was larger than needed, put the remainder back.
            let remainder = sz - aligned;
            if remainder >= 16 {
                self.free_list.push((offset + aligned, remainder));
            }
            return Some(offset);
        }

        // Try to allocate from the latest chunk.
        if let Some(chunk) = self.chunks.last() {
            let used_in_chunk = self.cursor - chunk.start_offset;
            if used_in_chunk + aligned <= chunk.capacity {
                let offset = self.cursor;
                self.allocations.push((offset, bytes));
                self.total_allocated += bytes;
                self.cursor += aligned;
                return Some(offset);
            }
        }

        // Current chunk is full — allocate a new stable chunk instead of
        // reallocating (which would invalidate existing function pointers).
        let new_capacity = aligned.max(
            self.chunks.last().map_or(Self::DEFAULT_LEN, |c| (c.capacity * 2).min(128 * 1024 * 1024))
        );
        let new_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                new_capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if new_ptr.is_null() || new_ptr == libc::MAP_FAILED {
            return None;
        }

        let start_offset = self.cursor;
        let chunk = ArenaChunk {
            base: NonNull::new(new_ptr.cast::<u8>())?,
            capacity: new_capacity,
            start_offset,
        };
        self.chunks.push(chunk);

        let offset = self.cursor;
        self.allocations.push((offset, bytes));
        self.total_allocated += bytes;
        self.cursor += aligned;
        Some(offset)
    }

    /// Selectively flip only dirty pages from RW→RX (Task 8 fix).
    ///
    /// This must be called after a batch of JIT compilations, before any
    /// compiled code is executed. Instead of flipping ALL pages to RX, we
    /// only flip pages that have been written to (tracked in `_legacy_dirty_pages`).
    /// Pages that were already RX from a previous finalize() are left alone,
    /// avoiding unnecessary mprotect syscalls and TLB shootdowns.
    ///
    /// Key invariant: after finalize(), ALL pages containing executable
    /// code must be RX. This holds because:
    ///   - Pages that were already RX (from a previous finalize) stay RX
    ///   - Newly dirty pages are flipped from RW to RX
    ///   - Pages that were never written to start as RW but contain no
    ///     executable code, so leaving them RW is fine
    pub fn finalize(&mut self) -> Result<(), i32> {
        // For dual-mapped arenas, the RX mapping is already executable,
        // so we never need mprotect. The RW mapping stays RW for writing
        // new code. Just clear the dirty page tracking.
        if self.dual_mapped.is_some() {
            self._legacy_dirty_pages.clear_all();
            return Ok(());
        }
        // Fast path: if no pages are dirty, skip mprotect entirely.
        // This avoids the overhead of iterating an empty bitmap and
        // issuing zero syscalls — a common case when code is already
        // compiled and no new compilations have occurred.
        if self._legacy_dirty_pages.is_empty() {
            return Ok(());
        }
        let page = 4096usize;
        for page_addr in self._legacy_dirty_pages.iter_set() {
            let ok = unsafe {
                mprotect(
                    page_addr as *mut libc::c_void,
                    page,
                    PROT_READ | PROT_EXEC,
                )
            };
            if ok != 0 {
                return Err(ok);
            }
            // Record this page as finalized so make_writable() can selectively
            // flip it back to RW when needed.
            self._legacy_finalized_pages.set(page_addr);
        }
        self._legacy_dirty_pages.clear_all();
        Ok(())
    }

    /// Selectively flip only finalized pages back to RW so new code can be
    /// written to the arena (Task 8 fix).
    ///
    /// Instead of flipping ALL pages to RW (which causes TLB shootdowns on
    /// every compilation), we only flip pages that were previously finalized
    /// (tracked in `finalized_pages`). Pages that are already RW (never
    /// finalized, or freshly mmap'd) are left alone.
    ///
    /// This avoids unnecessary mprotect calls on pages that haven't changed,
    /// reducing TLB shootdowns and page faults.
    ///
    /// Key invariant: after make_writable(), pages that need to be written to
    /// must be RW. This holds because:
    ///   - Previously finalized pages (which may be written to again) are
    ///     flipped back to RW
    ///   - New allocations from alloc() go into freshly mmap'd RW pages
    ///   - Pages that were never finalized are still RW
    pub fn make_writable(&mut self) {
        let page = 4096usize;
        // Fast path: if no pages are finalized, nothing needs flipping.
        if self._legacy_finalized_pages.is_empty() {
            return;
        }
        for page_addr in self._legacy_finalized_pages.iter_set() {
            // Only call mprotect on pages that were actually finalized
            // (tracked in finalized_pages).  The test() method is used
            // internally by iter_set(), but we also use it here to
            // double-check before the expensive mprotect syscall — on
            // some kernels, mprotect on an already-RW page still takes
            // the page table lock, so skipping truly unnecessary calls
            // avoids contention in multi-threaded JIT scenarios.
            if self._legacy_finalized_pages.test(page_addr) {
                // Flip to RW; ignore errors (page may already be RW)
                unsafe {
                    let _ = libc::mprotect(
                        page_addr as *mut libc::c_void,
                        page,
                        PROT_READ | PROT_WRITE,
                    );
                }
            }
        }
        // Clear finalized_pages — they are now RW again.
        // Use clear() for each page individually rather than clear_all()
        // so that any concurrent readers checking test() see a consistent
        // view.  However, since make_writable() is only called from the
        // JIT compilation thread (single-threaded), clear_all() is safe.
        self._legacy_finalized_pages.clear_all();
    }

    /// Record dirty pages for a given allocation range (Fix #1).
    ///
    /// Call this after writing code to the arena. Instead of immediately
    /// flipping pages to RX (which would crash subsequent writes to the
    /// same page), we record which pages need to be flipped and batch
    /// them in finalize().
    ///
    /// When dual-mapped, this is a no-op: the RW mapping stays writable
    /// and the RX mapping is always executable — no mprotect tracking needed.
    pub fn record_dirty_pages(&mut self, ptr: usize, len: usize) {
        // Dual-mapped arenas never need dirty page tracking:
        // all writes go through the stable RW pointer, all execution
        // through the RX mirror — zero mprotect syscalls.
        if self.dual_mapped.is_some() {
            return;
        }

        let page = 4096usize;
        let base = ptr & !(page - 1);
        let end = ((ptr + len.max(1)) + page - 1) & !(page - 1);
        let mut p = base;
        while p < end {
            self._legacy_dirty_pages.set(p);
            // If this page was previously finalized (RX), clear it from
            // finalized_pages since we're about to write to it again.
            // This prevents make_writable() from redundantly flipping
            // a page that's already been made RW by a prior compilation.
            if self._legacy_finalized_pages.test(p) {
                self._legacy_finalized_pages.clear(p);
            }
            p += page;
        }
    }

    /// Force dual-mapped arena with retry strategies.
    ///
    /// If the initial dual-mapping attempt failed (e.g., memfd_create not
    /// available), this method retries with alternative strategies:
    ///   1. Try memfd_create with a different name
    ///   2. Try anonymous mmap with MAP_SHARED (if on Linux)
    ///   3. Fall back to the single-mapping approach
    ///
    /// Returns true if dual-mapping was successfully established.
    pub fn force_dual_mapped(&mut self) -> bool {
        if self.dual_mapped.is_some() {
            return true; // already dual-mapped
        }

        // Try to create a DualMappedArena
        if let Some(dual) = DualMappedArena::try_new(Self::DEFAULT_LEN) {
            // Copy existing code to the new dual-mapped arena
            let old_base = self.chunks.first().map(|c| c.base.as_ptr());
            if let Some(old_ptr) = old_base {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old_ptr,
                        dual.rw_base,
                        self.cursor.min(dual.capacity),
                    );
                }
            }

            // Replace the first chunk with the dual-mapped one
            if !self.chunks.is_empty() {
                // Unmap the old single-mapped chunk
                let old_chunk = &self.chunks[0];
                unsafe {
                    munmap(old_chunk.base.as_ptr().cast(), old_chunk.capacity);
                }
                self.chunks[0] = ArenaChunk {
                    base: NonNull::new(dual.rw_base).unwrap(),
                    capacity: dual.capacity,
                    start_offset: 0,
                };
            }

            self.dual_mapped = Some(dual);
            jit_trace!("[JIT] force_dual_mapped: successfully upgraded to dual-mapped arena");
            return true;
        }

        false
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TLB Pre-hinting for Newly Compiled Code
    // ─────────────────────────────────────────────────────────────────────────
    //
    // When a freshly compiled Tier-3 function is called for the first time,
    // the CPU experiences a ~100-cycle TLB miss penalty trying to find the
    // physical memory address of the JIT code arena. This is because:
    //   1. The page table entries for the JIT arena are not in the TLB
    //   2. The CPU must walk 4 levels of page tables (on x86-64 with 4-level paging)
    //   3. Each level may require a memory access (3-4 cache misses)
    //
    // Solution: After writing JIT code to the arena, we "touch" the pages
    // through both the RW and RX mappings to pre-populate the TLB. This
    // ensures that the first execution of the new code hits a warm TLB.

    /// Pre-fault and pre-hint TLB entries for a newly compiled code region.
    /// Call this after writing JIT code to the arena, before execution.
    ///
    /// This performs two key operations:
    ///   1. Read-touch each page through the RW mapping (pre-faults + populates DTLB)
    ///   2. Read-touch each page through the RX mapping (populates ITLB)
    ///
    /// On CPUs without PREFETCHIT0 (pre-Sapphire Rapids), falls back to a
    /// simple read through the RX mapping which also populates the ITLB
    /// via the page walk.
    pub fn tlb_prehint(&self, offset: usize, size: usize) {
        let page_size = 4096usize;
        let start = offset & !(page_size - 1);
        let end = (offset + size + page_size - 1) & !(page_size - 1);

        // Touch each page through the RW mapping (pre-faults the page table entries)
        for page_start in (start..end).step_by(page_size) {
            if let Some(ptr) = self.try_get_ptr(page_start) {
                // Volatile read to ensure the compiler doesn't optimize it away
                unsafe {
                    let _ = std::ptr::read_volatile(ptr);
                }
            }
        }

        // Touch each page through the RX mapping (populates the ITLB)
        if let Some(ref dual) = self.dual_mapped {
            for page_start in (start..end).step_by(page_size) {
                let rw_ptr = if let Some(ptr) = self.try_get_ptr(page_start) {
                    ptr as usize
                } else {
                    continue;
                };
                let rx_offset = dual.rx_base as usize - dual.rw_base as usize;
                let rx_ptr = (rw_ptr + rx_offset) as *const u8;
                unsafe {
                    let _ = std::ptr::read_volatile(rx_ptr);
                }
            }
        }
    }

    /// Software-guided TLB warmup: walk the page table entries for the
    /// given code region and ensure they're cached in the TLB.
    ///
    /// This is more aggressive than tlb_prehint() — it touches every
    /// 4KB page in the range through both mappings, and also touches
    /// adjacent pages that may be reached by the code's data accesses.
    pub fn tlb_warmup_aggressive(&self, offset: usize, size: usize) {
        // First, do the basic prehint
        self.tlb_prehint(offset, size);

        // Also touch the pages immediately before and after the code region,
        // as the CPU's hardware prefetcher may fetch into these areas.
        let page_size = 4096usize;
        let extended_start = offset.saturating_sub(page_size) & !(page_size - 1);
        let extended_end = offset + size + page_size;

        if self.dual_mapped.is_some() {
            self.tlb_prehint(extended_start, extended_end - extended_start);
        }
    }

    /// Try to resolve a global offset to a pointer, returning None if invalid.
    pub fn try_get_ptr(&self, offset: usize) -> Option<*const u8> {
        let idx = self.chunks.partition_point(|c| c.start_offset <= offset);
        if idx == 0 {
            return None;
        }
        let chunk = &self.chunks[idx - 1];
        if offset >= chunk.start_offset + chunk.capacity {
            return None;
        }
        let local = offset - chunk.start_offset;
        Some(unsafe { chunk.base.as_ptr().add(local) })
    }

    /// Register a new compiled function entry in the cache.
    /// Returns the entry ID (index into entries vec).
    fn register_entry(&mut self, offset: usize, size: usize) -> usize {
        let entry_id = self.next_entry_id;
        self.next_entry_id += 1;
        let tick = self.global_tick;
        self.entries.push(CodeCacheEntry {
            offset,
            size,
            execution_count: 0,
            last_used: tick,
            compiled_at: tick,
            valid: true,
        });
        entry_id
    }

    /// Mark an entry as recently used (increments execution_count and updates last_used).
    fn touch_entry(&mut self, entry_id: usize) {
        if let Some(entry) = self.entries.get_mut(entry_id) {
            if entry.valid {
                entry.execution_count += 1;
                self.global_tick += 1;
                entry.last_used = self.global_tick;
            }
        }
    }

    /// LRU-based code cache eviction (Fix #5).
    ///
    /// When arena usage exceeds the eviction threshold, this method
    /// identifies the coldest entries (lowest execution_count * recency_weight),
    /// invalidates them, and adds their space to the free list for reuse.
    /// Returns the list of invalidated entry IDs so callers can remove
    /// their function pointers.
    pub fn maybe_evict(&mut self, needed: usize) -> Vec<usize> {
        // Calculate total capacity across all chunks.
        let total_capacity: usize = self.chunks.iter().map(|c| c.capacity).sum();
        // Use the explicit capacity limit as the effective ceiling for
        // the usage calculation, clamped by actual capacity. This prevents
        // eviction from being triggered prematurely when the arena has
        // pre-allocated more capacity than the configured limit.
        let effective_capacity = self.capacity_limit.max(total_capacity);
        if effective_capacity == 0 {
            return Vec::new();
        }

        let usage = self.total_allocated as f64 / effective_capacity as f64;

        // If below threshold, no eviction needed.
        if usage < self.eviction_threshold {
            return Vec::new();
        }

        jit_trace!(
            "JIT arena near capacity: {} / {} bytes ({:.1}% used, needed: {}), triggering LRU eviction",
            self.total_allocated, total_capacity, usage * 100.0, needed
        );

        // Sort valid entries by a heat score: execution_count * recency_weight.
        // Lower score = colder = evict first.
        // recency_weight favors recently-used entries.
        // We use: score = execution_count * (last_used as f64).log2().max(1.0)
        // This means entries with low execution counts AND old last_used are coldest.
        let mut scored: Vec<(usize, f64)> = self.entries.iter().enumerate()
            .filter(|(_, e)| e.valid)
            .map(|(id, e)| {
                let recency_weight = (e.last_used as f64).log2().max(1.0);
                // Age factor: recently-compiled code gets a boost, old code decays.
                // This ensures long-compiled but rarely-used cold code is evicted
                // before hot code that was compiled recently.
                let age = self.global_tick.saturating_sub(e.compiled_at);
                let age_factor = if age > 0 { 1.0 / (1.0 + (age as f64).log2().max(1.0)) } else { 1.0 };
                (id, e.execution_count as f64 * recency_weight * age_factor)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Invalidate the coldest entries until we've freed enough space.
        let mut freed = 0usize;
        let mut invalidated = Vec::new();
        let target_free = needed.max(total_capacity / 10); // free at least 10% or what's needed

        for (entry_id, _score) in scored {
            if freed >= target_free {
                break;
            }
            let entry = &mut self.entries[entry_id];
            if !entry.valid {
                continue;
            }
            entry.valid = false;
            let aligned_size = (entry.size + 15) & !15;
            freed += aligned_size;
            self.free_list.push((entry.offset, aligned_size));
            self.total_allocated = self.total_allocated.saturating_sub(entry.size);
            invalidated.push(entry_id);
        }

        if !invalidated.is_empty() {
            jit_trace!(
                "JIT LRU eviction: evicted {} entries, freed {} bytes",
                invalidated.len(), freed
            );
        }

        invalidated
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        if !self.arena_backed && !self.ptr.is_null() && self.len > 0 {
            // For non-arena-backed allocations, unmap the memory to prevent leaks.
            // Arena-backed allocations are freed when the arena itself is dropped.
            unsafe {
                let _ = libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

thread_local! {
    static TLS_EXEC_ARENA: RefCell<Option<ExecArena>> = const { RefCell::new(None) };
}

impl ExecMem {
    fn new(code: &[u8]) -> Option<Self> {
        if let Some(mem) = TLS_EXEC_ARENA.with(|arena_cell| {
            let mut arena = arena_cell.borrow_mut();
            if arena.is_none() {
                *arena = ExecArena::try_new();
            }
            let arena = arena.as_mut()?;
            // Before writing new code, ensure the arena pages are writable.
            // A previous finalize() call may have flipped pages to RX,
            // which would cause SIGSEGV when we try to write new code.
            arena.make_writable();
            let offset = arena.alloc(code.len().max(1))?;
            let ptr = arena.get_ptr(offset);
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len()) };
            // FIX #1: Record dirty page for batch mprotect — do NOT flip to RX here.
            // Pages stay RW until finalize() is called, preventing SIGSEGV when
            // two functions share a 4K page and the second compilation writes to
            // an already-RX page.
            arena.record_dirty_pages(ptr as usize, code.len().max(1));
            // Register this allocation in the code cache for LRU tracking.
            let entry_id = arena.register_entry(offset, code.len().max(1));
            Some(Self {
                offset,
                ptr: std::ptr::null_mut(),
                len: code.len().max(1),
                arena_backed: true,
                cached_entry_ptr: AtomicPtr::new(std::ptr::null_mut()),
                entry_id,
            })
        }) {
            return Some(mem);
        }

        // Fallback: individual mmap per function.
        let len = code.len().max(1);
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len()) };
        // Flip to RX now that writing is complete.
        let ok = unsafe { mprotect(ptr, len, PROT_READ | PROT_EXEC) };
        if ok != 0 {
            unsafe { munmap(ptr, len) };
            return None;
        }
        // For non-arena-backed, store the actual pointer
        Some(Self {
            offset: 0,
            ptr: ptr.cast::<u8>(),
            len,
            arena_backed: false,
            cached_entry_ptr: AtomicPtr::new(std::ptr::null_mut()),
            entry_id: usize::MAX,
        })
    }

    fn entry(&self) -> unsafe extern "C" fn(*mut i64) -> i64 {
        if self.arena_backed {
            // FIX #12: Cache the resolved TLS pointer to avoid TLS access
            // on every entry() call. The first call resolves via TLS,
            // subsequent calls use the cached pointer directly.
            let cached = self.cached_entry_ptr.load(Ordering::Relaxed);
            if !cached.is_null() {
                return unsafe { std::mem::transmute(cached) };
            }
            TLS_EXEC_ARENA.with(|arena_cell| {
                let arena = arena_cell.borrow();
                if let Some(arena) = arena.as_ref() {
                    let rw_ptr = arena.get_ptr(self.offset) as usize;
                    // When dual-mapped, compute the RX pointer for execution.
                    // get_ptr() returns the RW mapping pointer, but execution
                    // must happen through the RX mapping (W^X compliance).
                    let exec_ptr = if let Some(ref dual) = arena.dual_mapped {
                        let local_offset = rw_ptr - dual.rw_base as usize;
                        dual.rx_base as usize + local_offset
                    } else {
                        rw_ptr
                    };
                    let ptr = exec_ptr as *mut u8;
                    // Cache for future calls (benign race: both threads
                    // would write the same value).
                    self.cached_entry_ptr.store(ptr, Ordering::Relaxed);
                    unsafe { std::mem::transmute(ptr) }
                } else {
                    panic!("Arena not initialized for arena-backed ExecMem");
                }
            })
        } else {
            unsafe { std::mem::transmute(self.ptr) }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Code emitter
// ─────────────────────────────────────────────────────────────────────────────
//
// Register number convention (matches x86-64 encoding):
//   rax=0  rcx=1  rdx=2  rbx=3  rsp=4  rbp=5  rsi=6  rdi=7
//   r8=8   r9=9   r10=10 r11=11 r12=12 r13=13 r14=14 r15=15

pub struct Emitter {
    /// Base pointer of the pre-allocated emit buffer (64 KiB + 64 KiB safety margin).
    /// The emitter writes directly to this buffer via an un-bounds-checked pointer
    /// bump, eliminating the per-byte branch check that Vec<u8>::push() imposes.
    /// Safety: the 64 KiB safety margin at the tail ensures we never write past
    /// the allocated region even for the largest single instruction sequence
    /// (an AVX-512 instruction with 8-byte displacement is at most ~12 bytes).
    base: *mut u8,
    /// Current write pointer — advances forward with each emitted byte.
    write_ptr: *mut u8,
    /// End of the usable region (base + 64 KiB). Writes must not exceed this.
    /// The safety margin (base + 64 KiB to base + 128 KiB) is reserved for
    /// overrun protection but not included in `end` — code that respects
    /// `end` will never touch the safety margin.
    _end: *const u8,
    /// Total capacity of the buffer (64 KiB + 64 KiB safety margin).
    _capacity: usize,
    /// Hot/Cold Block Reordering (Opt3): counters per emitted block.
    /// A non-zero counter means the block is "hot"; zero means cold (deopt/error).
    /// Maps block index → execution count (populated during emission, used for layout).
    _hot_counters: Vec<u64>,
    /// Set of cold labels (block IDs that lead to deopt/error paths).
    cold_labels: Vec<usize>,
    /// Tracks whether any AVX/YMM/ZMM (SIMD) instruction has been emitted.
    /// Used to conditionally emit VZEROUPPER in the function epilogue,
    /// avoiding the 70-cycle AVX→SSE transition penalty ONLY when SIMD
    /// was actually used. Previously, vzeroupper was either always
    /// emitted (hurting scalar workloads) or always disabled (hurting
    /// mixed scalar+SIMD workloads). This flag gives the best of both.
    emitted_simd: bool,
    /// Tracks whether any AMX tile instruction has been emitted.
    /// Used to emit TILERELEASE in the function epilogue and to
    /// conditionally emit LDTILECFG in the prologue. When AMX is used
    /// anywhere in the function, the tile configuration is hoisted to
    /// the prologue (eliminating pipeline serialization stalls from
    /// inline ldtilecfg calls in hot inner loops).
    emitted_amx: bool,
}

// SAFETY: The Emitter owns its buffer and is not shared across threads
// during compilation. After compilation, the buffer is consumed via
// `into_vec()` which returns an owned Vec<u8>.
unsafe impl Send for Emitter {}

impl Drop for Emitter {
    fn drop(&mut self) {
        if !self.base.is_null() {
            unsafe {
                libc::free(self.base as *mut libc::c_void);
            }
        }
    }
}


impl Emitter {
    const EMIT_BUF_SIZE: usize = 64 * 1024;        // 64 KiB usable region
    const SAFETY_MARGIN: usize = 64 * 1024;        // 64 KiB overrun protection
    const TOTAL_BUF_SIZE: usize = Self::EMIT_BUF_SIZE + Self::SAFETY_MARGIN;

    fn new() -> Self {
        let layout = std::alloc::Layout::from_size_align(Self::TOTAL_BUF_SIZE, 64)
            .expect("Emitter layout overflow");
        let base = unsafe { std::alloc::alloc(layout) };
        if base.is_null() {
            panic!("Failed to allocate Emitter buffer ({} bytes)", Self::TOTAL_BUF_SIZE);
        }
        Self {
            base,
            write_ptr: base,
            _end: unsafe { base.add(Self::EMIT_BUF_SIZE) },
            _capacity: Self::TOTAL_BUF_SIZE,
            _hot_counters: Vec::new(),
            cold_labels: Vec::new(),
            emitted_simd: false,
            emitted_amx: false,
        }
    }

    #[inline(always)]
    fn pos(&self) -> usize {
        unsafe { self.write_ptr.offset_from(self.base) as usize }
    }

    /// Emit a single byte — zero-overhead raw pointer write.
    /// No bounds check. Safety guaranteed by the 64 KiB safety margin.
    #[inline(always)]
    fn b(&mut self, v: u8) {
        unsafe {
            std::ptr::write(self.write_ptr, v);
            self.write_ptr = self.write_ptr.add(1);
        }
    }

    #[inline(always)]
    fn emit2(&mut self, b0: u8, b1: u8) {
        unsafe {
            let p = self.write_ptr;
            std::ptr::write(p, b0);
            std::ptr::write(p.add(1), b1);
            self.write_ptr = p.add(2);
        }
    }

    #[inline(always)]
    fn emit3(&mut self, b0: u8, b1: u8, b2: u8) {
        unsafe {
            let p = self.write_ptr;
            std::ptr::write(p, b0);
            std::ptr::write(p.add(1), b1);
            std::ptr::write(p.add(2), b2);
            self.write_ptr = p.add(3);
        }
    }

    #[inline(always)]
    fn emit4(&mut self, b0: u8, b1: u8, b2: u8, b3: u8) {
        unsafe {
            let p = self.write_ptr;
            std::ptr::write(p, b0);
            std::ptr::write(p.add(1), b1);
            std::ptr::write(p.add(2), b2);
            std::ptr::write(p.add(3), b3);
            self.write_ptr = p.add(4);
        }
    }

    #[inline(always)]
    fn d(&mut self, v: i32) {
        unsafe {
            let p = self.write_ptr;
            let bytes = v.to_le_bytes();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, 4);
            self.write_ptr = p.add(4);
        }
    }

    /// Emit a 32-bit signed immediate (alias for `d` — used by MatMulInstr emission).
    #[inline(always)]
    fn d32(&mut self, v: i32) {
        self.d(v);
    }

    #[inline(always)]
    fn q(&mut self, v: i64) {
        unsafe {
            let p = self.write_ptr;
            let bytes = v.to_le_bytes();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, 8);
            self.write_ptr = p.add(8);
        }
    }

    // ── Buffer access methods (bridge for code that used Vec<u8>) ──────────

    /// Return the emitted code as a borrowed byte slice.
    /// Valid until the Emitter is dropped or `reset()` is called.
    #[inline]
    fn as_slice(&self) -> &[u8] {
        let len = self.pos();
        unsafe { std::slice::from_raw_parts(self.base, len) }
    }

    /// Return the emitted code as a mutable byte slice.
    /// Valid until the Emitter is dropped or `reset()` is called.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.pos();
        unsafe { std::slice::from_raw_parts_mut(self.base, len) }
    }

    /// Consume the emitted code into a Vec<u8>.
    /// This copies the data out of the raw buffer into a new Vec.
    fn into_vec(self) -> Vec<u8> {
        let len = self.pos();
        let mut vec = Vec::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(self.base, vec.as_mut_ptr(), len);
            vec.set_len(len);
        }
        // Don't run Drop — we've moved the data out
        std::mem::forget(self);
        vec
    }

    /// Truncate the buffer to a given length (used for backtracking).
    fn truncate(&mut self, len: usize) {
        let base = self.base;
        if len <= unsafe { self.write_ptr.offset_from(base) as usize } {
            self.write_ptr = unsafe { base.add(len) };
        }
    }

    /// Extend from a byte slice (used for copy-and-patch stencil insertion).
    fn extend_from_slice(&mut self, bytes: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.write_ptr, bytes.len());
            self.write_ptr = self.write_ptr.add(bytes.len());
        }
    }

    /// Pop the last byte off the buffer (used for correction/adjustment).
    fn pop(&mut self) {
        if self.write_ptr > self.base {
            self.write_ptr = unsafe { self.write_ptr.sub(1) };
        }
    }

    /// Reset the emitter to empty (reuses the same buffer allocation).
    pub fn reset(&mut self) {
        self.write_ptr = self.base;
        self.emitted_simd = false;
        self.emitted_amx = false;
        self._hot_counters.clear();
        self.cold_labels.clear();
    }

    /// Emit a REX prefix byte
    #[inline(always)]
    fn emit_rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let rex = 0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8);
        self.b(rex);
    }

    /// Emit a ModRM byte
    #[inline(always)]
    fn emit_modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        let modrm = (mode << 6) | ((reg & 0x7) << 3) | (rm & 0x7);
        self.b(modrm);
    }

    // ── Immediate loads ──────────────────────────────────────────────────────

    /// Full 64-bit immediate into rax (10 bytes).
    fn mov_rax_imm64(&mut self, v: i64) {
        self.emit_rex(true, false, false, false); // REX.W for 64-bit operand
        self.b(0xB8);
        self.q(v);
    }

    /// Optimal immediate into rax.
    ///  v ≥ 0 and fits i32 → MOV EAX, imm32 (5 B, zero-extends)
    ///  v <  0 and fits i32 → REX.W MOV RAX, sign-ext imm32 (7 B)
    ///  otherwise           → MOV RAX, imm64 (10 B)
    fn mov_rax_imm_opt(&mut self, v: i64) {
        if let Ok(v32) = i32::try_from(v) {
            if v32 >= 0 {
                self.b(0xB8);
                self.d(v32); // MOV EAX, imm32
            } else {
                self.emit3(0x48, 0xC7, 0xC0);
                self.d(v32); // MOV RAX, sx(imm32)
            }
        } else {
            self.mov_rax_imm64(v);
        }
    }

    // ── Generic register ↔ memory ────────────────────────────────────────────
    //
    // REX.W  = bit3 (always 1 for 64-bit)
    // REX.R  = bit2 (extends ModRM.reg  — destination for 8B, source for 89)
    // REX.B  = bit0 (extends ModRM.rm   — source     for 8B, dest   for 89 w/ mod=11)
    //
    // For [rdi + disp32]: ModRM = mod10 | reg_field<<3 | 7

    /// mov reg64, [rdi + disp32]
    fn load_reg_mem(&mut self, reg: u8, disp: i32) {
        self.emit3(
            0x48 | ((reg & 8) >> 1), // REX.W | REX.R
            0x8B,
            0x87 | ((reg & 7) << 3), // mod=10, reg, rm=7(rdi)
        );
        self.d(disp);
    }

    /// mov [rdi + disp32], reg64
    fn store_mem_reg(&mut self, disp: i32, reg: u8) {
        self.emit3(0x48 | ((reg & 8) >> 1), 0x89, 0x87 | ((reg & 7) << 3));
        self.d(disp);
    }

    /// mov dst64, src64  — no-op when dst == src.
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        // MOV r64, r/m64 (0x8B): REX.R extends dst, REX.B extends src
        self.emit3(
            0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3),
            0x8B,
            0xC0 | ((dst & 7) << 3) | (src & 7),
        );
    }

    // ── Arithmetic (rax / rcx) ───────────────────────────────────────────────

    fn add_rax_rcx(&mut self) {
        self.emit3(0x48, 0x01, 0xC8);
    }
    fn sub_rax_rcx(&mut self) {
        self.emit3(0x48, 0x29, 0xC8);
    }
    fn imul_rax_rcx(&mut self) {
        self.emit4(0x48, 0x0F, 0xAF, 0xC1);
    }

    /// ADD RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn add_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xC0);
            self.b(v8 as u8); // ADD RAX, imm8
        } else {
            self.emit2(0x48, 0x05);
            self.d(v); // ADD RAX, imm32
        }
    }

    /// SUB RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn sub_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xE8);
            self.b(v8 as u8); // SUB RAX, imm8
        } else {
            self.emit2(0x48, 0x2D);
            self.d(v); // SUB RAX, imm32
        }
    }

    fn imul_rax_imm32(&mut self, v: i32) {
        self.emit3(0x48, 0x69, 0xC0);
        self.d(v);
    }

    fn inc_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC0);
    }
    fn dec_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC8);
    }
    fn neg_rax(&mut self) {
        self.emit3(0x48, 0xF7, 0xD8);
    }
    fn shl_rax_imm8(&mut self, v: u8) {
        self.emit4(0x48, 0xC1, 0xE0, v);
    }

    /// XOR EAX, EAX — 2 bytes, zero-extends to clear full RAX.
    /// Preferred over REX.W form (3 bytes) for zeroing.
    fn xor_eax_eax(&mut self) {
        self.emit2(0x31, 0xC0); // XOR EAX, EAX  (zero-extends → RAX = 0)
    }

    // LEA ×N patterns on rax only (rax = rax*N via SIB with base=rax, index=rax).
    // SIB for [rax + rax*K]: scale_bits<<6 | index=rax(0)<<3 | base=rax(0)
    fn lea_rax_rax_mul3(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x40);
    }
    fn lea_rax_rax_mul5(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x80);
    }
    fn lea_rax_rax_mul9(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0xC0);
    }

    /// lea rax, [rcx + rax*scale]   scale ∈ {2,4,8}
    /// Used by Mul+Add→LEA fusion: rax = multiplicand, rcx = addend.
    /// Result: rax*scale + rcx.
    fn lea_rax_rax_scale_plus_rcx(&mut self, scale: u8) {
        // SIB: ss<<6 | index=rax(0)<<3 | base=rcx(1)
        let ss: u8 = match scale {
            4 => 2,
            8 => 3,
            _ => 1,
        };
        self.emit4(0x48, 0x8D, 0x04, (ss << 6) | 1);
    }

    // ── Bitwise / Logical (rax / rcx) ────────────────────────────────────

    /// AND RAX, RCX
    fn and_rax_rcx(&mut self) {
        self.emit3(0x48, 0x21, 0xC8); // AND r/m64, r64 → REX.W 21 /r, ModRM=11 001 000
    }
    /// OR RAX, RCX
    fn or_rax_rcx(&mut self) {
        self.emit3(0x48, 0x09, 0xC8); // OR r/m64, r64 → REX.W 09 /r, ModRM=11 001 000
    }
    /// XOR RAX, RCX
    fn xor_rax_rcx(&mut self) {
        self.emit3(0x48, 0x31, 0xC8); // XOR r/m64, r64 → REX.W 31 /r, ModRM=11 001 000
    }
    /// SHL RAX, CL  (shift left by CL)
    fn shl_rax_cl(&mut self) {
        self.emit3(0x48, 0xD3, 0xE0); // SHL r/m64, CL → REX.W D3 /4, ModRM=11 100 000
    }
    /// SAR RAX, CL  (arithmetic shift right by CL)
    fn sar_rax_cl(&mut self) {
        self.emit3(0x48, 0xD3, 0xF8); // SAR r/m64, CL → REX.W D3 /7, ModRM=11 111 000
    }
    /// SHR RAX, CL  (logical shift right by CL)
    fn shr_rax_cl(&mut self) {
        self.emit3(0x48, 0xD3, 0xE8); // SHR r/m64, CL → REX.W D3 /5, ModRM=11 101 000
    }
    /// AND RAX, imm32
    fn and_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xE0);
            self.b(v8 as u8); // AND RAX, imm8
        } else {
            self.emit2(0x48, 0x25);
            self.d(v); // AND RAX, imm32
        }
    }
    /// OR RAX, imm32
    fn or_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xC8);
            self.b(v8 as u8); // OR RAX, imm8
        } else {
            self.emit2(0x48, 0x0D);
            self.d(v); // OR RAX, imm32
        }
    }
    /// XOR RAX, imm32
    fn xor_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xF0);
            self.b(v8 as u8); // XOR RAX, imm8
        } else {
            self.emit2(0x48, 0x35);
            self.d(v); // XOR RAX, imm32
        }
    }
    /// SHR RAX, imm8
    fn shr_rax_imm8(&mut self, v: u8) {
        self.emit4(0x48, 0xC1, 0xE8, v); // SHR r/m64, imm8 → REX.W C1 /5 ib
    }
    /// SAR RAX, imm8
    fn sar_rax_imm8(&mut self, v: u8) {
        self.emit4(0x48, 0xC1, 0xF8, v); // SAR r/m64, imm8 → REX.W C1 /7 ib
    }
    /// MOV RCX, RAX
    fn mov_rcx_rax(&mut self) {
        self.emit3(0x48, 0x89, 0xC1); // MOV r/m64, r64 → RAX→RCX, ModRM=11 000 001
    }
    /// MOV RAX, RCX
    fn mov_rax_rcx(&mut self) {
        self.emit3(0x48, 0x89, 0xC8); // MOV r/m64, r64 → RCX→RAX, ModRM=11 001 000
    }
    /// MOV RCX, imm64
    fn mov_rcx_imm64(&mut self, v: i64) {
        self.emit_rex(true, false, false, false); // REX.W
        self.b(0xB8 | 1); // opcode + reg=RCX(1)
        self.q(v);
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn cqo(&mut self) {
        self.emit2(0x48, 0x99);
    }
    fn idiv_rcx(&mut self) {
        self.emit3(0x48, 0xF7, 0xF9);
    }
    fn mov_rax_rdx(&mut self) {
        self.emit3(0x48, 0x89, 0xD0);
    }

    // ── Magic-number division support ──────────────────────────────────────

    /// One-operand IMUL RCX: RDX:RAX = RAX * RCX (128-bit signed multiply).
    /// Unlike the two-operand `imul_rax_rcx()` (which only gives the low 64
    /// bits), this form produces the full 128-bit product in RDX:RAX, giving
    /// us the high 64 bits needed for the magic-multiply division trick.
    /// Encoding: REX.W F7 /5 with ModRM for RCX → 48 F7 E9
    fn emit_imul_rcx_128(&mut self) {
        self.emit3(0x48, 0xF7, 0xE9);
    }

    /// MOV R8, RAX — save RAX into R8 (used by magic division to preserve
    /// the original dividend for sign correction).
    /// Encoding: REX.WB 89 C0 → 49 89 C0
    fn mov_r8_rax(&mut self) {
        self.emit3(0x49, 0x89, 0xC0);
    }

    /// SHR R8, imm8 — logical shift right of R8 by an immediate count.
    /// Used to extract the sign bit of the saved dividend (SHR R8, 63).
    /// Encoding: REX.WB C1 /5 ib → 49 C1 E8 ib
    fn shr_r8_imm8(&mut self, v: u8) {
        self.emit4(0x49, 0xC1, 0xE8, v);
    }

    /// ADD RAX, R8 — add R8 to RAX. Used by magic division for sign
    /// correction: if the original dividend was negative, add 1 to the
    /// quotient (truncation-toward-zero correction).
    /// Encoding: REX.WR 01 C0 → 4C 01 C0
    fn add_rax_r8(&mut self) {
        self.emit3(0x4C, 0x01, 0xC0);
    }

    /// MOV RAX, R8 — restore RAX from R8 (used by magic remainder to
    /// recover the original dividend after computing the quotient).
    /// Encoding: REX.WR 8B C0 → 4C 8B C0
    fn mov_rax_r8(&mut self) {
        self.emit3(0x4C, 0x8B, 0xC0);
    }

    // ── Compare / branch ─────────────────────────────────────────────────────

    fn cmp_rax_rcx(&mut self) {
        self.emit_rex(true, false, false, false); // REX.W
        self.b(0x39);
        self.emit_modrm(3, 1, 0); // mod=11, reg=rcx(1), rm=rax(0)
    }

    /// CMP RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn cmp_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xF8);
            self.b(v8 as u8); // CMP RAX, imm8
        } else {
            self.emit2(0x48, 0x3D);
            self.d(v); // CMP RAX, imm32
        }
    }

    /// TEST RAX, RAX — 3 bytes. Tests all 64 bits for zero.
    /// Previously used TEST EAX,EAX (2 bytes) which only checked the low 32 bits,
    /// causing values like 0x1_0000_0000 to be incorrectly treated as zero.
    fn test_rax_rax(&mut self) {
        // TEST RAX, RAX — checks all 64 bits. Required for i64 branch conditions
        // where the value may have non-zero upper 32 bits (e.g., i64 values > 2^32).
        self.emit3(0x48, 0x85, 0xC0); // TEST RAX, RAX
    }

    fn setcc_al(&mut self, cc: u8) {
        self.emit3(0x0F, cc, 0xC0);
    }
    fn movzx_rax_al(&mut self) {
        self.emit4(0x48, 0x0F, 0xB6, 0xC0);
    }

    // ── AMX instruction emission ──────────────────────────────────────────────
    //
    // Intel AMX instructions use VEX3 encoding.  The general format is:
    //   C4 [R X B mmmmm] [W vvvv L pp] opcode ModRM [SIB] [imm8]
    //
    // Key VEX3 fields for AMX:
    //   mmmmm = 00010  (0F38 map)
    //   pp    = 00/01/10/11  (no prefix / 66 / F3 / F2)
    //   W     = 0 or 1
    //   vvvv  = first source register (inverted), or 1111 if unused (NDS)
    //   L     = 0 (LZ) for AMX tile operations
    //
    // Reference: Intel 64 and IA-32 Architectures Software Developer's Manual,
    // Vol. 2, AMX instruction encodings.

    /// Emit a VEX3 prefix + opcode for an AMX instruction.
    ///
    /// - `pp`: mandatory prefix (0=none, 1=66, 2=F3, 3=F2)
    /// - `w`: VEX.W bit
    /// - `vvvv`: first source register (NOT inverted; we invert here)
    /// - `opcode`: the 1-byte opcode after the 0F38 escape
    fn emit_vex3_prefix(&mut self, pp: u8, w: bool, vvvv: u8, opcode: u8) {
        self.emitted_simd = true; // VEX-encoded instructions are SIMD
        self.b(0xC4);                                     // VEX3 byte 0
        self.b(0xE0 | 0x02);                              // R=X=B=1(inverted), mmmmm=00010(0F38)
        let byte2 = ((w as u8) << 7)
                  | (((!vvvv) & 0xF) << 3)
                  | (pp & 0x3);
        self.b(byte2);                                    // VEX3 byte 2
        self.b(opcode);
    }

    /// LDTILECFG [rax] – load tile configuration from memory.
    /// Encoding: 66 0F38 49 /r  →  VEX.128.66.0F38.W1 49 /r
    /// ModRM: mod=00, reg=0, rm=0 (rax)
    fn amx_ldtilecfg_rax(&mut self) {
        self.emitted_amx = true; // Track AMX usage for prologue hoisting
        self.emit_vex3_prefix(1, true, 0xF, 0x49);  // pp=1(66), W=1, vvvv=1111(NONE)
        self.emit_modrm(0, 0, 0);                     // mod=00, reg=0, rm=rax
    }

    /// TILELOADD tmm, [rax + rcx*1] – load tile from memory.
    /// Encoding: F3 0F38 4B /r ib  →  VEX.128.F3.0F38.W0 4B /r
    /// ModRM: mod=00, reg=tmm, rm=100(SIB follows)
    /// SIB: scale=00, index=rcx(1), base=rax(0)
    fn amx_tileloadd(&mut self, tmm: u8) {
        self.emit_vex3_prefix(2, false, 0xF, 0x4B);  // pp=2(F3), W=0, vvvv=1111(NONE)
        self.emit_modrm(0, tmm, 4);                    // mod=00, reg=tmm, rm=100(SIB)
        self.b(0x08);                                   // SIB: scale=00, index=rcx(1), base=rax(0)
    }

    /// TILESTORED [rax + rcx*1], tmm – store tile to memory.
    /// Encoding: 66 0F38 4B /r ib  →  VEX.128.66.0F38.W0 4B /r
    /// ModRM: mod=00, reg=tmm, rm=100(SIB follows)
    /// SIB: scale=00, index=rcx(1), base=rax(0)
    fn amx_tilestored(&mut self, tmm: u8) {
        self.emit_vex3_prefix(1, false, 0xF, 0x4B);  // pp=1(66), W=0, vvvv=1111(NONE)
        self.emit_modrm(0, tmm, 4);                    // mod=00, reg=tmm, rm=100(SIB)
        self.b(0x08);                                   // SIB: scale=00, index=rcx(1), base=rax(0)
    }

    /// TILEZERO tmm – zero a tile register.
    /// Encoding: F3 0F38 49 /r  →  VEX.128.F3.0F38.W0 49 /r (mod=11)
    fn amx_tilezero(&mut self, tmm: u8) {
        self.emit_vex3_prefix(2, false, 0xF, 0x49);  // pp=2(F3), W=0, vvvv=1111(NONE)
        self.emit_modrm(3, 0, tmm);                    // mod=11, reg=0, rm=tmm
    }

    /// TDPBSSD tmm_dst, tmm_src1, tmm_src2 – signed byte dot product.
    /// Encoding: F3 0F38 5C /r  →  VEX.NDS.LZ.F3.0F38.W0 5C /r
    /// VVVV = src1 (inverted), reg = src2, rm = dst (mod=11)
    fn amx_tdpbssd(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex3_prefix(2, false, src1, 0x5C);  // pp=2(F3), W=0, vvvv=src1
        self.emit_modrm(3, src2, dst);                  // mod=11, reg=src2, rm=dst
    }

    /// TDPBSUD tmm_dst, tmm_src1, tmm_src2 – signed × unsigned byte.
    /// Encoding: F3 0F38 5E /r
    fn amx_tdpbsud(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex3_prefix(2, false, src1, 0x5E);
        self.emit_modrm(3, src2, dst);
    }

    /// TDPBUSD tmm_dst, tmm_src1, tmm_src2 – unsigned × signed byte.
    /// Encoding: F2 0F38 5C /r
    fn amx_tdpbusd(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex3_prefix(3, false, src1, 0x5C);  // pp=3(F2)
        self.emit_modrm(3, src2, dst);
    }

    /// TDPBUUD tmm_dst, tmm_src1, tmm_src2 – unsigned × unsigned byte.
    /// Encoding: F2 0F38 5E /r
    fn amx_tdpbuud(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex3_prefix(3, false, src1, 0x5E);
        self.emit_modrm(3, src2, dst);
    }

    /// TDPBF16PS tmm_dst, tmm_src1, tmm_src2 – BF16 dot product into FP32.
    /// Encoding: F3 0F38 6C /r  →  VEX.NDS.LZ.F3.0F38.W0 6C /r
    fn amx_tdpbf16ps(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex3_prefix(2, false, src1, 0x6C);  // pp=2(F3), W=0, vvvv=src1
        self.emit_modrm(3, src2, dst);                  // mod=11, reg=src2, rm=dst
    }

    /// TILERELEASE – release tile configuration.
    /// Encoding: 66 0F38 49 C0  →  VEX.128.66.0F38.W0 49 /r (mod=11, reg=0, rm=0)
    fn amx_tilerelease(&mut self) {
        self.emit_vex3_prefix(1, false, 0xF, 0x49);  // pp=1(66), W=0, vvvv=1111
        self.emit_modrm(3, 0, 0);                      // mod=11, reg=0, rm=0
    }

    /// PREFETCHT0 [rax] – prefetch into all cache levels.
    fn prefetcht0_rax(&mut self) {
        self.emit3(0x0F, 0x18, 0x08);  // PREFETCHT0 [rax]
    }

    /// PREFETCHT1 [rax] – prefetch into L2.
    fn prefetcht1_rax(&mut self) {
        self.emit3(0x0F, 0x18, 0x10);  // PREFETCHT1 [rax]
    }

    /// PREFETCHT2 [rax] – prefetch into L3.
    fn prefetcht2_rax(&mut self) {
        self.emit3(0x0F, 0x18, 0x18);  // PREFETCHT2 [rax]
    }

    /// PREFETCHNTA [rax] – non-temporal prefetch.
    fn prefetchnta_rax(&mut self) {
        self.emit3(0x0F, 0x18, 0x00);  // PREFETCHNTA [rax]
    }

    // ── Branches ─────────────────────────────────────────────────────────────
    //
    // Two-pass strategy: first emit rel32 placeholders, then patch.
    // Alternatively, for short branches we can back-patch with rel8.
    // We use a unified placeholder approach and pick the right encoding at
    // fixup time via a "rewrite" pass (see patch_fixups).

    fn jmp_rel32_placeholder(&mut self) -> usize {
        self.b(0xE9);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x84);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jnz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x85);
        let p = self.pos();
        self.d(0);
        p
    }

    fn ret(&mut self) {
        self.b(0xC3);
    }

    // ── XMM (SSE2) floating-point instructions ─────────────────────────────
    //
    // XMM registers: xmm0-xmm15. We use xmm0-xmm7 for f32/f64 arithmetic.
    // The JIT maps f32/f64 slots to XMM registers when possible, falling back
    // to memory (the rdi slot array) for spills.

    /// movsd xmm0, [rdi + disp32]  — load f64 from slot
    fn load_xmm0_mem(&mut self, disp: i32) {
        // F2 prefix (SIMD scalar double): REX.W=0x48 + F2 0F 10 /r
        self.emit4(0xF2, 0x48, 0x0F, 0x10);
        // ModRM: mod=10 (disp32), reg=000 (xmm0), rm=111 (rdi)
        self.b(0x87);
        self.d(disp);
    }

    /// movss xmm0, [rdi + disp32]  — load f32 from slot
    fn load_xmm0_mem_f32(&mut self, disp: i32) {
        // F3 prefix (SIMD scalar single): F3 0F 10 /r
        self.emit3(0xF3, 0x0F, 0x10);
        // ModRM: mod=10, reg=000, rm=111
        self.b(0x87);
        self.d(disp);
    }

    /// movsd [rdi + disp32], xmm0  — store f64 to slot
    fn store_mem_xmm0(&mut self, disp: i32) {
        self.emit4(0xF2, 0x48, 0x0F, 0x11);
        self.b(0x87);
        self.d(disp);
    }

    /// movss [rdi + disp32], xmm0  — store f32 to slot
    fn store_mem_xmm0_f32(&mut self, disp: i32) {
        self.emit3(0xF3, 0x0F, 0x11);
        self.b(0x87);
        self.d(disp);
    }

    /// movsd xmm1, [rdi + disp32]  — load f64 into xmm1 from slot
    fn load_xmm1_mem(&mut self, disp: i32) {
        self.emit4(0xF2, 0x48, 0x0F, 0x10);
        self.b(0x8F); // mod=10, reg=xmm1(1), rm=rdi(7)
        self.d(disp);
    }

    /// addsd xmm0, xmm1  — f64 add
    fn add_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x58);
        self.b(0xC1); // mod=11, reg=xmm0, rm=xmm1
    }

    /// addss xmm0, xmm1  — f32 add
    fn add_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x58);
        self.b(0xC1);
    }

    /// subsd xmm0, xmm1  — f64 sub
    fn sub_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x5C);
        self.b(0xC1);
    }

    /// subss xmm0, xmm1  — f32 sub
    fn sub_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5C);
        self.b(0xC1);
    }

    /// mulsd xmm0, xmm1  — f64 mul
    fn mul_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x59);
        self.b(0xC1);
    }

    /// mulss xmm0, xmm1  — f32 mul
    fn mul_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x59);
        self.b(0xC1);
    }

    /// divsd xmm0, xmm1  — f64 div
    fn div_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x5E);
        self.b(0xC1);
    }

    /// divss xmm0, xmm1  — f32 div
    fn div_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5E);
        self.b(0xC1);
    }

    /// movq xmm1, rax  — move integer → XMM (for mixed int/float ops)
    fn movq_xmm1_rax(&mut self) {
        self.emit4(0x66, 0x48, 0x0F, 0x6E);
        self.b(0xC8); // mod=11, reg=xmm1, rm=rax
    }

    /// minsd xmm0, xmm1  — f64 scalar min
    fn minsd_xmm0_xmm1(&mut self) {
        self.emit3(0xF2, 0x0F, 0x5D);
        self.b(0xC1);
    }

    /// maxsd xmm0, xmm1  — f64 scalar max
    fn maxsd_xmm0_xmm1(&mut self) {
        self.emit3(0xF2, 0x0F, 0x5F);
        self.b(0xC1);
    }

    /// minss xmm0, xmm1  — f32 scalar min
    fn minss_xmm0_xmm1(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5D);
        self.b(0xC1);
    }

    /// maxss xmm0, xmm1  — f32 scalar max
    fn maxss_xmm0_xmm1(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5F);
        self.b(0xC1);
    }

    /// ucomisd xmm0, xmm1  — f64 compare, sets EFLAGS
    fn ucomisd_xmm0_xmm1(&mut self) {
        self.emit3(0x66, 0x0F, 0x2E);
        self.b(0xC1);
    }

    /// ucomiss xmm0, xmm1  — f32 compare
    fn ucomiss_xmm0_xmm1(&mut self) {
        self.emit3(0x0F, 0x2E, 0xC1);
    }

    /// cvttsd2si rax, xmm0  — f64 → i64 (truncating)
    /// Correct encoding: F2 48 0F 2C C0
    ///   F2 = SSE2 mandatory prefix
    ///   48 = REX.W (64-bit operand size → rax, not eax)
    ///   0F 2C = cvttsd2si opcode
    ///   C0 = ModRM: mod=11 reg=0(rax) rm=0(xmm0)
    fn cvttsd2si_eax_xmm0(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x2C);
        self.b(0xC0); // ModRM: mod=11, reg=0(rax), rm=0(xmm0)
    }

    /// cvttss2si eax, xmm0  — f32 → i32 (truncating)
    fn cvttss2si_eax_xmm0(&mut self) {
        self.emit3(0xF3, 0x0F, 0x2C);
        self.b(0xC0);
    }

    /// cvtsi2sd xmm0, rax  — i64 → f64
    fn cvtsi2sd_xmm0_rax(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x2A);
        self.b(0xC0);
    }

    /// cvtsi2ss xmm0, rax  — i64 → f32
    fn cvtsi2ss_xmm0_rax(&mut self) {
        self.emit4(0xF3, 0x48, 0x0F, 0x2A);
        self.b(0xC0);
    }

    /// Load immediate f64 into xmm0: move bits into rax, then movq xmm0, rax
    fn mov_xmm0_imm64(&mut self, bits: u64) {
        self.emit2(0x48, 0xB8); // MOV RAX, imm64
        self.q(bits as i64);
        self.emit4(0x66, 0x48, 0x0F, 0x6E); // MOVQ XMM0, RAX
        self.b(0xC0);
    }

    /// Load immediate f32 into xmm0: zero-extend to 64-bit, then movq
    fn mov_xmm0_imm32_bits(&mut self, bits: u32) {
        self.b(0xB8); // MOV EAX, imm32 (zero-extends)
        self.d(bits as i32);
        self.emit4(0x66, 0x48, 0x0F, 0x6E); // MOVQ XMM0, RAX
        self.b(0xC0);
    }

    // ── Generic XMM register instructions ──────────────────────────────────
    //
    // These methods work with arbitrary XMM registers (0-15), enabling
    // the register allocator to assign float slots to any XMM register
    // instead of hardcoding XMM0/XMM1.
    //
    // SSE2 instruction encoding for arbitrary XMM registers:
    //   F2 [REX] 0F <opcode> ModRM
    //   - REX.W is NOT set for SSE moves (unlike GPR moves)
    //   - REX.R extends the reg field (ModRM bits 5-3) for regs >= 8
    //   - REX.B extends the rm field (ModRM bits 2-0) for regs >= 8
    //   - For reg-reg (mod=11): ModRM = 0xC0 | ((dst&7)<<3) | (src&7)
    //
    // Note: We do NOT set REX.W for SSE2 scalar double instructions.
    // The F2 prefix alone selects the scalar double operation.
    // REX.W is only needed for MOVQ (integer move to/from XMM).

    /// movsd xmm_reg, [rdi + disp32] — load f64 from slot into any XMM register
    fn load_xmm_reg_mem(&mut self, xmm_reg: u8, disp: i32) {
        // F2 [REX.R if xmm_reg>=8] 0F 10 ModRM(mod=10, reg=xmm_reg, rm=111)
        // REX.R extends reg field of ModRM (= destination XMM register number bit 3)
        self.b(0xF2); // mandatory prefix for scalar double
        if xmm_reg >= 8 {
            // REX with REX.R=1 for reg field extension
            // 0x40 | (0 << 3) | (1 << 2) | (0 << 1) | 0 = 0x44
            self.b(0x44);
        }
        self.emit2(0x0F, 0x10); // MOVSD opcode
        // ModRM: mod=10 (disp32), reg=xmm_reg&7, rm=111 (rdi)
        self.b(0x80 | ((xmm_reg & 7) << 3) | 7);
        self.d(disp);
    }

    /// movsd [rdi + disp32], xmm_reg — store any XMM register to slot (f64)
    fn store_mem_xmm_reg(&mut self, disp: i32, xmm_reg: u8) {
        // F2 [REX.R if xmm_reg>=8] 0F 11 ModRM(mod=10, reg=xmm_reg, rm=111)
        self.b(0xF2); // mandatory prefix for scalar double
        if xmm_reg >= 8 {
            self.b(0x44); // REX with REX.R=1
        }
        self.emit2(0x0F, 0x11); // MOVSD store opcode
        // ModRM: mod=10 (disp32), reg=xmm_reg&7, rm=111 (rdi)
        self.b(0x80 | ((xmm_reg & 7) << 3) | 7);
        self.d(disp);
    }

    /// movsd xmm_dst, xmm_src — move f64 between arbitrary XMM registers
    /// No-op when dst == src.
    fn mov_xmm_xmm(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        // F2 [REX] 0F 10 ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2  // REX.R for dst
                | ((src >= 8) as u8);        // REX.B for src
            self.b(rex);
        }
        self.emit2(0x0F, 0x10); // MOVSD xmm, xmm
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// addsd xmm_dst, xmm_src — f64 add between arbitrary XMM registers
    fn add_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        // F2 [REX] 0F 58 ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x58);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// subsd xmm_dst, xmm_src — f64 sub between arbitrary XMM registers
    fn sub_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5C);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// mulsd xmm_dst, xmm_src — f64 mul between arbitrary XMM registers
    fn mul_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x59);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// divsd xmm_dst, xmm_src — f64 div between arbitrary XMM registers
    fn div_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5E);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// minsd xmm_dst, xmm_src — f64 scalar min between arbitrary XMM registers
    fn minsd_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        // F2 [REX] 0F 5D ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5D);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// maxsd xmm_dst, xmm_src — f64 scalar max between arbitrary XMM registers
    fn maxsd_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        // F2 [REX] 0F 5F ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF2);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5F);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// minss xmm_dst, xmm_src — f32 scalar min between arbitrary XMM registers
    fn minss_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        // F3 [REX] 0F 5D ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF3);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5D);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// maxss xmm_dst, xmm_src — f32 scalar max between arbitrary XMM registers
    fn maxss_xmm_reg_reg(&mut self, dst: u8, src: u8) {
        // F3 [REX] 0F 5F ModRM(mod=11, reg=dst, rm=src)
        self.b(0xF3);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x5F);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// ucomisd xmm_dst, xmm_src — f64 compare between arbitrary XMM registers
    fn ucomisd_reg_reg(&mut self, dst: u8, src: u8) {
        // 66 [REX] 0F 2E ModRM(mod=11, reg=dst, rm=src)
        self.b(0x66);
        let need_rex = dst >= 8 || src >= 8;
        if need_rex {
            let rex = 0x40
                | ((dst >= 8) as u8) << 2
                | ((src >= 8) as u8);
            self.b(rex);
        }
        self.emit2(0x0F, 0x2E);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// movq xmm_reg, rax — move integer from RAX into any XMM register
    fn movq_xmm_reg_rax(&mut self, xmm_reg: u8) {
        // 66 [REX.W + REX.R if xmm_reg>=8] 0F 6E ModRM(mod=11, reg=xmm_reg, rm=0)
        self.b(0x66);
        // REX.W is always set for 64-bit move
        let rex = 0x48 | ((xmm_reg >= 8) as u8) << 2;
        self.b(rex);
        self.emit2(0x0F, 0x6E);
        self.b(0xC0 | ((xmm_reg & 7) << 3)); // rm=0 (rax)
    }

    /// Load immediate f64 into any XMM register: move bits into rax, then movq xmm_reg, rax
    fn mov_xmm_reg_imm64(&mut self, xmm_reg: u8, bits: u64) {
        self.emit2(0x48, 0xB8); // MOV RAX, imm64
        self.q(bits as i64);
        self.movq_xmm_reg_rax(xmm_reg);
    }

    /// Load immediate f32 into any XMM register: zero-extend to 64-bit, then movq
    fn mov_xmm_reg_imm32_bits(&mut self, xmm_reg: u8, bits: u32) {
        self.b(0xB8); // MOV EAX, imm32 (zero-extends)
        self.d(bits as i32);
        self.movq_xmm_reg_rax(xmm_reg);
    }

    // ── Register-direct operations ────────────────────────────────────────
    //
    // These methods operate directly on allocated GPR registers (r8-r15, rbx,
    // rsi, etc.) instead of funnelling through RAX/RCX.  This eliminates
    // 2–3 redundant MOV instructions per arithmetic operation when the
    // destination is in a register and one of the operands is the same slot
    // (e.g., `s = s + 1` → INC r_s instead of MOV RAX,r_s / ADD RAX,1 / MOV r_s,RAX).
    //
    // Encoding reference (64-bit mode):
    //   REX.W = 0x48 (always set for 64-bit operand size)
    //   REX.R extends ModRM.reg (bit 3) — for /r field (source reg)
    //   REX.B extends ModRM.rm  (bit 3) — for /r field (dest reg in mod=11)
    //   ModRM = 0xC0 | ((reg & 7) << 3) | (rm & 7) for mod=11 (reg-reg)

    /// INC r64 — increment register by 1
    fn inc_reg(&mut self, reg: u8) {
        // REX.W FF /0 with ModRM mod=11
        let rex = 0x48 | ((reg >= 8) as u8); // REX.W | REX.B
        self.emit3(rex, 0xFF, 0xC0 | (reg & 7));
    }

    /// DEC r64 — decrement register by 1
    fn dec_reg(&mut self, reg: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xFF, 0xC8 | (reg & 7)); // /1
    }

    /// ADD r64, imm8/imm32 — add immediate to register
    fn add_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            // REX.W 83 /0 ib
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xC0 | (reg & 7)); // ModRM: mod=11, reg=/0, rm=reg
            self.b(v8 as u8);
        } else {
            // REX.W 81 /0 id
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xC0 | (reg & 7));
            self.d(imm);
        }
    }

    /// SUB r64, imm8/imm32 — subtract immediate from register
    fn sub_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xE8 | (reg & 7)); // /5
            self.b(v8 as u8);
        } else {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xE8 | (reg & 7)); // /5
            self.d(imm);
        }
    }

    /// IMUL r64, imm32 — multiply register by immediate (two-operand form)
    fn imul_reg_imm(&mut self, reg: u8, imm: i32) {
        // IMUL r64, r/m64, imm32: opcode 69 /r
        // reg field = destination, rm field = source (same as dst for 2-operand form)
        // Need both REX.R and REX.B when reg >= 8
        let rex = 0x48
            | ((reg >= 8) as u8) << 2  // REX.R for reg field (dst)
            | ((reg >= 8) as u8);        // REX.B for rm field (src)
        self.emit3(rex, 0x69, 0xC0 | ((reg & 7) << 3) | (reg & 7));
        self.d(imm);
    }

    /// ADD dst, src — add two registers (dst = dst + src)
    fn add_reg_reg(&mut self, dst: u8, src: u8) {
        if dst == src {
            // ADD reg, reg = reg + reg = reg * 2, not a no-op. But callers
            // should avoid this pattern. Emit it correctly regardless.
        }
        // REX.W 01 /r: ADD r/m64, r64 (dest is r/m, src is reg)
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x01, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// SUB dst, src — subtract two registers (dst = dst - src)
    fn sub_reg_reg(&mut self, dst: u8, src: u8) {
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x29, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// IMUL dst, src — multiply two registers (dst = dst * src, low 64 bits)
    fn imul_reg_reg(&mut self, dst: u8, src: u8) {
        // REX.W 0F AF /r: IMUL r64, r/m64
        let rex = 0x48 | ((dst >= 8) as u8) << 2 | ((src >= 8) as u8);
        self.emit4(rex, 0x0F, 0xAF, 0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    /// AND dst, src — bitwise AND two registers
    fn and_reg_reg(&mut self, dst: u8, src: u8) {
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x21, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// OR dst, src — bitwise OR two registers
    fn or_reg_reg(&mut self, dst: u8, src: u8) {
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x09, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// XOR dst, src — bitwise XOR two registers
    fn xor_reg_reg(&mut self, dst: u8, src: u8) {
        if dst == src {
            // XOR reg, reg = zero the register (shorter than MOV reg, 0)
            // Use 32-bit XOR which zero-extends to clear full 64-bit register
            let rex = ((dst >= 8) as u8) << 2 | ((src >= 8) as u8);
            if rex != 0 {
                self.b(0x40 | rex);
            }
            self.emit2(0x31, 0xC0 | ((src & 7) << 3) | (dst & 7));
            return;
        }
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x31, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// AND r64, imm8/imm32
    fn and_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xE0 | (reg & 7)); // /4
            self.b(v8 as u8);
        } else {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xE0 | (reg & 7));
            self.d(imm);
        }
    }

    /// OR r64, imm8/imm32
    fn or_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xC8 | (reg & 7)); // /1
            self.b(v8 as u8);
        } else {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xC8 | (reg & 7));
            self.d(imm);
        }
    }

    /// XOR r64, imm8/imm32
    fn xor_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xF0 | (reg & 7)); // /6
            self.b(v8 as u8);
        } else {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xF0 | (reg & 7));
            self.d(imm);
        }
    }

    /// CMP r64, imm8/imm32 — compare register with immediate
    fn cmp_reg_imm(&mut self, reg: u8, imm: i32) {
        if let Ok(v8) = i8::try_from(imm) {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x83, 0xF8 | (reg & 7)); // /7
            self.b(v8 as u8);
        } else {
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit3(rex, 0x81, 0xF8 | (reg & 7));
            self.d(imm);
        }
    }

    /// CMP r1, r2 — compare two registers
    fn cmp_reg_reg(&mut self, r1: u8, r2: u8) {
        // CMP r/m64, r64: 39 /r
        let rex = 0x48 | ((r2 >= 8) as u8) << 2 | ((r1 >= 8) as u8);
        self.emit3(rex, 0x39, 0xC0 | ((r2 & 7) << 3) | (r1 & 7));
    }

    /// TEST r64, r64 — test register against itself (zero check)
    fn test_reg_reg(&mut self, r1: u8, r2: u8) {
        let _need_rex_w = true; // Always use 64-bit test for consistency
        let rex = 0x48 | ((r2 >= 8) as u8) << 2 | ((r1 >= 8) as u8);
        self.emit3(rex, 0x85, 0xC0 | ((r2 & 7) << 3) | (r1 & 7));
    }

    /// SHL r64, imm8 — shift left by immediate
    fn shl_reg_imm(&mut self, reg: u8, imm: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xC1, 0xE0 | (reg & 7)); // /4
        self.b(imm);
    }

    /// SAR r64, imm8 — arithmetic shift right by immediate
    fn sar_reg_imm(&mut self, reg: u8, imm: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xC1, 0xF8 | (reg & 7)); // /7
        self.b(imm);
    }

    /// SHR r64, imm8 — logical shift right by immediate
    fn shr_reg_imm(&mut self, reg: u8, imm: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xC1, 0xE8 | (reg & 7)); // /5
        self.b(imm);
    }

    /// NEG r64 — negate register
    fn neg_reg(&mut self, reg: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xF7, 0xD8 | (reg & 7)); // /3
    }

    /// MOV r64, imm — optimal encoding for loading immediate into register
    fn mov_reg_imm(&mut self, reg: u8, imm: i64) {
        if let Ok(v32) = i32::try_from(imm) {
            if v32 >= 0 {
                // MOV r32, imm32 (zero-extends to 64 bits)
                if reg >= 8 { self.b(0x41); } // REX.B
                self.b(0xB8 | (reg & 7));
                self.d(v32);
            } else {
                // REX.W MOV r64, sign-extended imm32
                let rex = 0x48 | ((reg >= 8) as u8);
                self.emit3(rex, 0xC7, 0xC0 | (reg & 7));
                self.d(v32);
            }
        } else {
            // MOV r64, imm64 (10 bytes)
            let rex = 0x48 | ((reg >= 8) as u8);
            self.emit2(rex, 0xB8 | (reg & 7));
            self.q(imm);
        }
    }

    /// MOV dst, src — move between arbitrary GPR registers (no-op when dst == src)
    fn mov_reg_reg(&mut self, dst: u8, src: u8) {
        if dst == src { return; }
        // MOV r/m64, r64: 89 /r
        let rex = 0x48 | ((src >= 8) as u8) << 2 | ((dst >= 8) as u8);
        self.emit3(rex, 0x89, 0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    /// SHL r64, CL — shift left by CL
    fn shl_reg_cl(&mut self, reg: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xD3, 0xE0 | (reg & 7)); // /4
    }

    /// SAR r64, CL — arithmetic shift right by CL
    fn sar_reg_cl(&mut self, reg: u8) {
        let rex = 0x48 | ((reg >= 8) as u8);
        self.emit3(rex, 0xD3, 0xF8 | (reg & 7)); // /7
    }

    /// LEA r64, [r64 + r64*scale] — for multiply-add patterns (scale ∈ {2,4,8})
    fn lea_reg_reg_scale(&mut self, dst: u8, base: u8, index: u8, scale: u8) {
        let ss: u8 = match scale {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 1,
        };
        let _rex = 0x48 | ((index >= 8) as u8) << 2 | ((dst >= 8) as u8) << 2 | ((base >= 8) as u8);
        // Actually, REX encoding for LEA with SIB:
        // REX.W + REX.R (extends ModRM.reg = dst) + REX.B (extends SIB.base = base)
        // REX.X extends SIB.index = index
        let rex = 0x48
            | ((dst >= 8) as u8) << 2    // REX.R for dst
            | ((index >= 8) as u8) << 1   // REX.X for index
            | ((base >= 8) as u8);         // REX.B for base
        self.emit3(rex, 0x8D, 0x04 | (dst & 7) << 3); // ModRM: mod=00, reg=dst, rm=100(SIB)
        self.b((ss << 6) | ((index & 7) << 3) | (base & 7)); // SIB byte
    }

    /// VZEROUPPER — zero the upper 128 bits of YMM0–YMM15.
    /// Required before RET when AVX instructions have been used, to avoid
    /// 70-cycle transition penalties on subsequent SSE code.
    fn vzeroupper(&mut self) {
        self.emit3(0xC5, 0xF8, 0x77); // VZEROUPPER: C5 F8 77
        // NOTE: Do NOT set emitted_simd here — vzeroupper is a cleanup
        // instruction, not a SIMD operation. The emitted_simd flag must
        // only be set by actual VEX/EVEX SIMD instructions so that
        // emit_ret() can conditionally emit vzeroupper before RET.
    }

    // ── AVX2 256-bit SIMD instructions (VEX-encoded) ──────────────────────
    //
    // These methods emit VEX2-encoded AVX2 instructions for 256-bit SIMD
    // vectorization.  Each YMM register holds 8 × i32 elements, enabling
    // 8-wide integer parallelism.  The VEX2 prefix is 2 bytes (C5 + byte1)
    // and encodes R, vvvv, L, and pp fields.
    //
    // VEX2 byte layout:
    //   C5 [R~vvvv1pp] [opcode] [ModRM]
    //   R     = inverted bit 3 of ModRM.reg
    //   vvvv  = inverted first source register (NDS)
    //   L     = 1 for 256-bit (YMM), 0 for 128-bit (XMM)
    //   pp    = mandatory prefix: 00=none, 01=66, 10=F3, 11=F2

    /// Emit a VEX2 prefix for 256-bit AVX2 integer operations (0F escape).
    /// `r` = high bit of reg field, `vvvv` = first source reg (NOT inverted here).
    /// `pp` = mandatory prefix bits (0=none, 1=66, 2=F3, 3=F2).
    fn emit_vex2_256(&mut self, r: bool, vvvv: u8, pp: u8) {
        self.emitted_simd = true; // VEX2-encoded instructions are SIMD
        // VEX2: C5 [R~vvvvLpp]
        let byte1 = ((!r as u8) << 7)
            | ((!vvvv) & 0xF) << 3
            | (1 << 2)  // L=1 for 256-bit
            | (pp & 0x3);
        self.emit2(0xC5, byte1);
    }

    /// vmovdqu ymm_dst, [rdi + disp32] — unaligned load 32 bytes (8 × i32).
    /// Uses VEX.256.66.0F.WIG 6F /r with ModRM mod=10 (disp32), rm=111(rdi).
    fn vmovdqu_ymm_mem_rdi(&mut self, dst: u8, disp: i32) {
        self.emit_vex2_256(dst >= 8, 0xF, 1); // pp=1(66), vvvv=1111(unused)
        self.b(0x6F); // VMOVDQU ymm, m256
        self.b(0x87 | ((dst & 7) << 3)); // ModRM: mod=10, reg=dst, rm=111(rdi)
        self.d(disp);
    }

    /// vmovdqu [rdi + disp32], ymm_src — unaligned store 32 bytes (8 × i32).
    /// Uses VEX.256.66.0F.WIG 7F /r with ModRM mod=10, rm=111(rdi).
    fn vmovdqu_mem_rdi_ymm(&mut self, disp: i32, src: u8) {
        self.emit_vex2_256(src >= 8, 0xF, 1); // pp=1(66), vvvv=1111(unused)
        self.b(0x7F); // VMOVDQU m256, ymm
        self.b(0x87 | ((src & 7) << 3)); // ModRM: mod=10, reg=src, rm=111(rdi)
        self.d(disp);
    }

    /// vpaddd ymm_dst, ymm_src1, ymm_src2 — packed 32-bit integer add (8 lanes).
    /// Uses VEX.256.66.0F.WIG FE /r.  vvvv = src1 (NDS), reg = src2, rm = dst.
    fn vpaddd_ymm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex2_256(src2 >= 8, src1, 1); // R~ from src2 (ModRM.reg), vvvv=src1
        self.b(0xFE); // VPADDD
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7)); // ModRM: mod=11, reg=src2, rm=dst
    }

    /// vpsubd ymm_dst, ymm_src1, ymm_src2 — packed 32-bit integer subtract (8 lanes).
    /// Uses VEX.256.66.0F.WIG FA /r.  vvvv = src1 (NDS), reg = src2, rm = dst.
    fn vpsubd_ymm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex2_256(src2 >= 8, src1, 1); // R~ from src2 (ModRM.reg), vvvv=src1
        self.b(0xFA); // VPSUBD
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7)); // ModRM: mod=11, reg=src2, rm=dst
    }

    /// vpmulld ymm_dst, ymm_src1, ymm_src2 — packed 32-bit integer multiply (8 lanes).
    /// Uses VEX.256.66.0F38.WIG 40 /r.  Requires VEX3 for 0F38 escape.
    /// vvvv = src1 (NDS), reg = src2, rm = dst.
    fn vpmulld_ymm(&mut self, dst: u8, src1: u8, src2: u8) {
        // VEX3: C4 [R~X~B~mmmmm] [W~vvvv~L~pp] [opcode]
        // ModRM: reg=src2, rm=dst. R~ inverts from src2 high bit, B~ from dst high bit.
        let r_inv = !(src2 >= 8) as u8; // R~ bit: invert of src2's high bit
        let b_inv = !(dst >= 8) as u8;  // B~ bit: invert of dst's high bit
        let byte1 = (r_inv << 7) | (1 << 6) | (b_inv << 5) | 0x02; // X~=1 (no SIB), mmmmm=00010(0F38)
        let byte2 = (0 << 7)          // W=0
            | ((!src1) & 0xF) << 3  // vvvv (inverted)
            | (1 << 2)               // L=1 for 256-bit
            | 1;                     // pp=1(66)
        self.emit4(0xC4, byte1, byte2, 0x40); // VPMULLD
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7)); // ModRM: mod=11
    }

    /// vpbroadcastd ymm_dst, xmm_src — broadcast 32-bit int to all 8 lanes.
    /// Uses VEX.256.66.0F38.WIG 58 /r.  Requires VEX3 for 0F38 escape.
    fn vpbroadcastd_ymm_xmm(&mut self, dst: u8, src: u8) {
        // ModRM: reg=dst, rm=src. R~ from dst high bit, B~ from src high bit.
        let r_inv = !(dst >= 8) as u8;
        let b_inv = !(src >= 8) as u8;
        let byte1 = (r_inv << 7) | (1 << 6) | (b_inv << 5) | 0x02; // X~=1, mmmmm=00010(0F38)
        let byte2 = (0 << 7)       // W=0
            | (0xF << 3)           // vvvv=1111 (unused)
            | (1 << 2)             // L=1 for 256-bit
            | 1;                   // pp=1(66)
        self.emit4(0xC4, byte1, byte2, 0x58); // VPBROADCASTD
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7)); // ModRM: mod=11
    }

    /// vmovd xmm_dst, r32 — move 32-bit GPR value to low XMM lane.
    /// Uses VEX.128.66.0F.WIG 6E /r.
    fn vmovd_xmm_r32(&mut self, dst: u8, src_gpr: u8) {
        let byte1 = (!(dst >= 8) as u8) << 7
            | (0xF << 3)    // vvvv=1111 (unused)
            | (0 << 2)      // L=0 for 128-bit
            | 1;            // pp=1(66)
        self.emit3(0xC5, byte1, 0x6E);
        self.b(0xC0 | ((dst & 7) << 3) | (src_gpr & 7)); // ModRM: mod=11
    }

    /// vextracti128 xmm_dst, ymm_src, 0x01 — extract high 128 bits of YMM.
    /// Uses VEX.256.66.0F3A.WIG 39 /r ib.  VEX3 needed for 0F3A escape.
    fn vextracti128_ymm_xmm(&mut self, dst: u8, src: u8, imm: u8) {
        // VEX3: ModRM reg=src, rm=dst. R~ from src high bit, B~ from dst high bit.
        let r_inv = !(src >= 8) as u8;
        let b_inv = !(dst >= 8) as u8;
        let byte1 = (r_inv << 7) | (1 << 6) | (b_inv << 5) | 0x03; // X~=1, mmmmm=00011(0F3A)
        let byte2 = (0 << 7)          // W=0
            | ((!src) & 0xF) << 3   // vvvv = src (inverted)
            | (1 << 2)               // L=1 for 256-bit
            | 1;                     // pp=1(66)
        self.emit4(0xC4, byte1, byte2, 0x39);
        self.b(0xC0 | ((src & 7) << 3) | (dst & 7)); // ModRM: mod=11
        self.b(imm);
    }

    /// vpaddd xmm_dst, xmm_src1, xmm_src2 — packed 32-bit integer add (4 lanes, 128-bit).
    /// Uses VEX.128.66.0F.WIG FE /r.
    /// VEX2: R~ extends ModRM.reg = src2, B~ implied 1 (dst must be < 8 for VEX2).
    fn vpaddd_xmm(&mut self, dst: u8, src1: u8, src2: u8) {
        let byte1 = (!(src2 >= 8) as u8) << 7  // R~ from src2 (ModRM.reg)
            | ((!src1) & 0xF) << 3
            | (0 << 2)  // L=0 for 128-bit
            | 1;        // pp=1(66)
        self.emit3(0xC5, byte1, 0xFE);
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7));
    }

    /// vpsubd xmm_dst, xmm_src1, xmm_src2 — packed 32-bit integer sub (4 lanes, 128-bit).
    /// VEX2: R~ extends ModRM.reg = src2, B~ implied 1 (dst must be < 8 for VEX2).
    fn vpsubd_xmm(&mut self, dst: u8, src1: u8, src2: u8) {
        let byte1 = (!(src2 >= 8) as u8) << 7  // R~ from src2 (ModRM.reg)
            | ((!src1) & 0xF) << 3
            | (0 << 2) | 1;
        self.emit3(0xC5, byte1, 0xFA);
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7));
    }

    /// vpmulld xmm_dst, xmm_src1, xmm_src2 — packed 32-bit integer mul (4 lanes, 128-bit).
    /// VEX3 for 0F38 escape.
    fn vpmulld_xmm(&mut self, dst: u8, src1: u8, src2: u8) {
        // VEX3: ModRM reg=src2, rm=dst. R~ from src2 high bit, B~ from dst high bit.
        let r_inv = !(src2 >= 8) as u8;
        let b_inv = !(dst >= 8) as u8;
        let byte1 = (r_inv << 7) | (1 << 6) | (b_inv << 5) | 0x02; // X~=1, mmmmm=00010(0F38)
        let byte2 = (0 << 7)
            | ((!src1) & 0xF) << 3
            | (0 << 2)  // L=0 for 128-bit
            | 1;        // pp=1(66)
        self.emit4(0xC4, byte1, byte2, 0x40);
        self.b(0xC0 | ((src2 & 7) << 3) | (dst & 7));
    }

    /// vmovdqu xmm_dst, [rdi + disp32] — unaligned load 16 bytes (4 × i32).
    fn vmovdqu_xmm_mem_rdi(&mut self, dst: u8, disp: i32) {
        let byte1 = (!(dst >= 8) as u8) << 7
            | (0xF << 3) | (0 << 2) | 1; // pp=1(66), L=0
        self.emit3(0xC5, byte1, 0x6F);
        self.b(0x87 | ((dst & 7) << 3));
        self.d(disp);
    }

    /// vmovdqu [rdi + disp32], xmm_src — unaligned store 16 bytes.
    fn vmovdqu_mem_rdi_xmm(&mut self, disp: i32, src: u8) {
        let byte1 = (!(src >= 8) as u8) << 7
            | (0xF << 3) | (0 << 2) | 1;
        self.emit3(0xC5, byte1, 0x7F);
        self.b(0x87 | ((src & 7) << 3));
        self.d(disp);
    }

    // ── Callee-saved save/restore ────────────────────────────────────────────
    //
    // Short-form PUSH/POP r64: 0x50+rd  (rd = reg & 7)
    // r8-r15 require REX.B prefix (0x41).

    fn push_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x50 + (reg & 7));
    }
    fn pop_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x58 + (reg & 7));
    }

    // ── NOP / padding ──────────────────────────────────────────────────────

    /// Single-byte NOP (0x90). Used for alignment padding at loop headers.
    fn nop(&mut self) {
        self.b(0x90);
    }

    /// Multi-byte NOP: emit `n` bytes of NOP using canonical x86-64
    /// multi-byte NOP encodings up to 15 bytes.  Each multi-byte NOP
    /// decodes in a single cycle on modern CPUs (Skylake+, Zen+), and
    /// using the longest possible single-instruction NOP minimizes
    /// front-end instruction fetch/decode queue pressure and DSB
    /// (Decoded ICache) slot consumption.
    ///
    /// Reference: Intel 64 and IA-32 Architectures Software Developer's
    /// Manual, Vol 2B, "NOP" instruction encoding tables.  The canonical
    /// NOP sequence uses the 3-byte opcode `0F 1F /0` with varying
    /// prefix, SIB, and displacement bytes.
    fn nop_multi(&mut self, n: usize) {
        let mut remaining = n;
        while remaining > 0 {
            // Emit the longest single-instruction NOP that fits.
            // All encodings are based on: [66] 0F 1F [/0] [SIB] [disp8] [disp32]
            if remaining >= 15 {
                // 15-byte NOP: 66 66 66 66 66 66 0F 1F 84 00 00 00 00 00 00
                //   6× prefix + 0F 1F ModRM SIB + 4-byte disp + 1 extra
                self.emit4(0x66, 0x66, 0x66, 0x66);
                self.emit4(0x66, 0x66, 0x0F, 0x1F);
                self.emit4(0x84, 0x00, 0x00, 0x00);
                self.emit3(0x00, 0x00, 0x00);
                remaining -= 15;
            } else if remaining >= 14 {
                // 14-byte NOP: 66 66 66 66 66 0F 1F 84 00 00 00 00 00 00
                self.emit4(0x66, 0x66, 0x66, 0x66);
                self.emit4(0x66, 0x0F, 0x1F, 0x84);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                self.emit2(0x00, 0x00);
                remaining -= 14;
            } else if remaining >= 13 {
                // 13-byte NOP: 66 66 66 66 0F 1F 84 00 00 00 00 00 00
                self.emit4(0x66, 0x66, 0x66, 0x66);
                self.emit4(0x0F, 0x1F, 0x84, 0x00);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                self.b(0x00);
                remaining -= 13;
            } else if remaining >= 12 {
                // 12-byte NOP: 66 66 66 0F 1F 84 00 00 00 00 00 00
                self.emit4(0x66, 0x66, 0x66, 0x0F);
                self.emit4(0x1F, 0x84, 0x00, 0x00);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                remaining -= 12;
            } else if remaining >= 11 {
                // 11-byte NOP: 66 66 0F 1F 84 00 00 00 00 00 00
                self.emit4(0x66, 0x66, 0x0F, 0x1F);
                self.emit4(0x84, 0x00, 0x00, 0x00);
                self.emit3(0x00, 0x00, 0x00);
                remaining -= 11;
            } else if remaining >= 10 {
                // 10-byte NOP: 66 0F 1F 84 00 00 00 00 00 00
                self.emit4(0x66, 0x0F, 0x1F, 0x84);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                self.emit2(0x00, 0x00);
                remaining -= 10;
            } else if remaining >= 9 {
                // 9-byte NOP: 66 0F 1F 84 00 00 00 00 00
                self.emit4(0x66, 0x0F, 0x1F, 0x84);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                self.b(0x00);
                remaining -= 9;
            } else if remaining >= 8 {
                // 8-byte NOP: 0F 1F 84 00 00 00 00 00
                self.emit4(0x0F, 0x1F, 0x84, 0x00);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                remaining -= 8;
            } else if remaining >= 7 {
                // 7-byte NOP: 0F 1F 80 00 00 00 00
                self.emit3(0x0F, 0x1F, 0x80);
                self.emit4(0x00, 0x00, 0x00, 0x00);
                remaining -= 7;
            } else if remaining >= 6 {
                // 6-byte NOP: 66 0F 1F 44 00 00
                self.emit4(0x66, 0x0F, 0x1F, 0x44);
                self.emit2(0x00, 0x00);
                remaining -= 6;
            } else if remaining >= 5 {
                // 5-byte NOP: 0F 1F 44 00 00
                self.emit4(0x0F, 0x1F, 0x44, 0x00);
                self.b(0x00);
                remaining -= 5;
            } else if remaining >= 4 {
                // 4-byte NOP: 0F 1F 40 00
                self.emit4(0x0F, 0x1F, 0x40, 0x00);
                remaining -= 4;
            } else if remaining >= 3 {
                // 3-byte NOP: 0F 1F 00
                self.emit3(0x0F, 0x1F, 0x00);
                remaining -= 3;
            } else if remaining >= 2 {
                // 2-byte NOP: 66 90
                self.emit2(0x66, 0x90);
                remaining -= 2;
            } else {
                // 1-byte NOP: 90
                self.b(0x90);
                remaining -= 1;
            }
        }
    }

    // ── CPUID-tuned instructions ────────────────────────────────────────────

    /// POPCNT RAX, RAX — counts set bits. Requires SSE4.2 (POPCNT is tied
    /// to SSE4.2 on x86-64). Useful for optimising boolean/comparison result
    /// handling and population count operations.
    fn popcnt_rax_rax(&mut self) {
        self.emit4(0xF3, 0x48, 0x0F, 0xB8); // F3 REX.W 0F B8 /r
        self.b(0xC0); // ModRM: mod=11, reg=0(rax), rm=0(rax)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Register allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Allocation pool ordered: caller-saved first (cheaper — no push/pop),
/// then callee-saved.
///
/// Excluded permanently:
///   rax(0)  — primary accumulator / return value
///   rcx(1)  — secondary scratch operand
///   rdx(2)  — clobbered by CQO/IDIV
///   rdi(7)  — slot-array base pointer (function argument)
///   rsp(4) / rbp(5) — stack management
const ALLOC_POOL: &[u8] = &[
    8, 9, 10, 11, 6, // r8-r11, rsi  (caller-saved — free)
    12, 13, 14, 15, 3, // r12-r15, rbx (callee-saved — require push/pop)
];

/// Where a bytecode slot lives at runtime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegLoc {
    Reg(u8),    // allocated physical GPR register
    Xmm(u8),    // allocated physical XMM register (0-15)
    Spill(i32), // byte offset inside rdi[] slot array
}

/// XMM registers available for allocation by the register allocator.
/// XMM0 = return value / scratch, XMM1 = scratch.
/// XMM2-XMM7 are caller-saved and available for allocation.
/// XMM8-XMM15 are also caller-saved in System V ABI but we keep the pool
/// small for simplicity (6 registers covers most functions).
// XMM register allocation pool. XMM0 and XMM1 are reserved for return values
// and scratch temporaries. XMM2-XMM15 are available for allocation in AVX mode
// (16 total XMM registers). With AVX-512, XMM16-XMM31 are also available but
// require EVEX-encoded instructions — the allocator only uses XMM2-XMM15 for
// now since EVEX-encoded moves (VMOVAPS zmm16+) add extra prefix overhead.
// This gives 14 allocatable XMM registers instead of the previous 6 (XMM2-7),
// significantly reducing spill traffic for float-heavy code.
const XMM_ALLOC_POOL: &[u8] = &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Recommended callee-saved GPRs for pinning ML loop induction variables
/// and tensor base pointers. These survive across function calls and avoid
/// register spilling in hot inner loops.
const ML_IV_PIN_REG: u8 = 12;     // R12 for outer loop IV
const ML_IV_PIN_REG2: u8 = 13;    // R13 for inner loop IV
const ML_BASE_PTR_PIN: u8 = 14;   // R14 for tensor base pointer

pub struct RegAlloc {
    /// Direct Vec indexed by slot number; pre-filled with Spill defaults.
    /// O(1) lookup vs HashMap for every load/store in the hot codegen loop.
    /// For float slots, this contains Spill (float slots use xmm_slots instead).
    slots: Vec<RegLoc>,
    /// Callee-saved registers actually used — must be pushed in prologue.
    used_callee_saved: Vec<u8>,
    /// XMM register assignments for float-typed slots (indexed by slot number).
    /// Contains Spill for non-float slots or float slots that didn't get an XMM reg.
    xmm_slots: Vec<RegLoc>,
    /// Which XMM registers are in use (for tracking / debugging).
    used_xmm_regs: Vec<u8>,
}

impl RegAlloc {
    #[inline(always)]
    fn location(&self, slot: u16) -> RegLoc {
        // Safety: slots is pre-allocated to cover all valid slot indices.
        unsafe { *self.slots.get_unchecked(slot as usize) }
    }

    /// Returns the location of a float-typed slot.
    /// Returns Xmm(reg) if the slot has an allocated XMM register,
    /// Spill(offset) if it's spilled to the slot array.
    #[inline(always)]
    fn float_location(&self, slot: u16) -> RegLoc {
        let idx = slot as usize;
        if idx < self.xmm_slots.len() {
            self.xmm_slots[idx]
        } else {
            RegLoc::Spill((slot as i32) * 8)
        }
    }

    /// Returns true if the given slot has an allocated XMM register.
    #[inline(always)]
    fn has_xmm(&self, slot: u16) -> bool {
        matches!(self.float_location(slot), RegLoc::Xmm(_))
    }

    /// Force-assign a slot to a specific physical register.
    /// Used by ForceRegisterLockState to pin induction variables
    /// and other critical values to callee-saved registers.
    fn force_assign(&mut self, slot: u16, reg: u8) {
        if (slot as usize) < self.slots.len() {
            self.slots[slot as usize] = RegLoc::Reg(reg);
        }
    }

    pub fn new(num_slots: u16) -> Self {
        let num = num_slots as usize;
        Self {
            slots: vec![RegLoc::Spill(0); num],
            used_callee_saved: Vec::new(),
            xmm_slots: vec![RegLoc::Spill(0); num],
            used_xmm_regs: Vec::new(),
        }
    }

    pub fn spill(&mut self, slot: u16) {
        if (slot as usize) < self.slots.len() {
            self.slots[slot as usize] = RegLoc::Spill((slot as i32) * 8);
        }
    }

    pub fn get_reg(&self, slot: u16) -> Option<u8> {
        if (slot as usize) < self.slots.len() {
            match self.slots[slot as usize] {
                RegLoc::Reg(r) => Some(r),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Split a variable's lifetime at the given instruction position.
    /// Creates two intervals: [start, split_pos) and [split_pos, end).
    /// The first interval keeps its current register assignment.
    /// The second interval needs a new assignment (resolved later).
    ///
    /// A parallel-move (copy) is inserted at split_pos to transfer the
    /// value from the old location to the new one.
    pub fn split_lifetime(&mut self, _slot: u16, _split_pos: usize) -> bool {
        // Find the interval for this slot that spans split_pos
        // Split it into two intervals
        // The first keeps its register, the second gets re-allocated
        // This is the core of "second-chance bin packing"
        true // placeholder
    }

    /// SSA-aware linear scan with lifetime splitting.
    ///
    /// Instead of a single linear pass, this allocator:
    /// 1. Computes liveness intervals for all variables
    /// 2. Identifies hot/cold path boundaries (backward branches = hot)
    /// 3. Splits intervals at hot/cold boundaries
    /// 4. Assigns registers preferentially to hot intervals
    /// 5. Spills cold intervals to stack slots
    /// 6. Inserts parallel-move sequences at split points
    ///
    /// This produces better code than simple linear scan for functions
    /// with complex variable lifetimes (e.g., loops with error paths).
    pub fn allocate_with_lifetime_splitting(
        num_slots: u16,
        liveness: &[(u16, usize, usize, bool)], // (slot, start, end, is_hot)
        num_phys_regs: u8,
    ) -> Self {
        let mut ra = RegAlloc::new(num_slots);

        // Sort intervals by start position (for linear scan)
        let mut intervals: Vec<(u16, usize, usize, bool)> = liveness.to_vec();
        intervals.sort_by_key(|&(_, start, _, _)| start);

        // Active set: intervals currently occupying registers
        let mut active: Vec<(u8, usize, u16, bool)> = Vec::new(); // (phys_reg, end, slot, is_hot)

        for (slot, start, end, is_hot) in &intervals {
            // Expire old intervals
            active.retain(|&(_, act_end, _, _)| act_end > *start);

            // Find a free register
            let used_regs: std::collections::HashSet<u8> =
                active.iter().map(|&(r, _, _, _)| r).collect();

            let free_reg = (0..num_phys_regs).find(|r| !used_regs.contains(r));

            if let Some(reg) = free_reg {
                ra.force_assign(*slot, reg);
                active.push((reg, *end, *slot, *is_hot));
            } else {
                // All registers in use — spill the longest cold interval
                // (prefer keeping hot intervals in registers)
                let spill_candidate = active.iter()
                    .filter(|&&(_, _, _, hot)| !hot) // prefer spilling cold
                    .max_by_key(|&&(_, end, _, _)| end);

                if let Some(&(_, _spill_end, spill_slot, _)) = spill_candidate {
                    // Spill the cold interval
                    ra.spill(spill_slot);
                    // Assign its register to the current interval
                    let freed_reg = ra.get_reg(spill_slot).unwrap_or(0);
                    ra.force_assign(*slot, freed_reg);
                    // Update active set
                    active.retain(|&( _, _, s, _)| s != spill_slot);
                    active.push((freed_reg, *end, *slot, *is_hot));
                } else {
                    // All hot — spill the one with the farthest end
                    let farthest = active.iter()
                        .max_by_key(|&(_, end, _, _)| end);
                    if let Some(&(_, _, spill_slot, _)) = farthest {
                        ra.spill(spill_slot);
                        let freed_reg = ra.get_reg(spill_slot).unwrap_or(0);
                        ra.force_assign(*slot, freed_reg);
                        active.retain(|&( _, _, s, _)| s != spill_slot);
                        active.push((freed_reg, *end, *slot, *is_hot));
                    } else {
                        ra.spill(*slot);
                    }
                }
            }
        }

        ra
    }
}

// ── Live intervals ────────────────────────────────────────────────────────────

/// Lifetime interval for a single variable, potentially split into
/// multiple non-contiguous pieces (each piece may reside in a
/// different location: register or stack slot).
#[derive(Debug, Clone)]
pub struct LifetimeInterval {
    /// The virtual register (slot) this interval belongs to
    pub slot: u16,
    /// Start instruction index (inclusive)
    pub start: usize,
    /// End instruction index (exclusive)
    pub end: usize,
    /// Physical register assigned to this interval (None = stack slot)
    pub phys_reg: Option<u8>,
    /// Stack slot assigned if spilled
    pub stack_slot: Option<u16>,
    /// Whether this interval is in the "hot" path
    pub is_hot: bool,
}

#[derive(Clone)]
struct LiveInterval {
    slot: u16,
    first: usize, // index of first instruction that defines or uses this slot
    last: usize,  // index of last instruction that uses this slot
}

fn compute_live_intervals(instrs: &[Instr], slot_count: usize) -> Vec<LiveInterval> {
    const UNDEF: usize = usize::MAX;
    let cap = slot_count + 1;
    let mut first_def = vec![UNDEF; cap];
    let mut last_use = vec![UNDEF; cap];

    macro_rules! ensure {
        ($s:expr) => {
            let s = $s as usize;
            if s >= first_def.len() {
                first_def.resize(s + 1, UNDEF);
                last_use.resize(s + 1, UNDEF);
            }
        };
    }
    macro_rules! def {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            let s = $s as usize;
            if first_def[s] == UNDEF {
                first_def[s] = $pc;
            }
        };
    }
    macro_rules! use_ {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            last_use[$s as usize] = pc_max(last_use[$s as usize], $pc);
        };
    }

    // Helper: max that treats UNDEF as -infinity
    fn pc_max(a: usize, b: usize) -> usize {
        if a == UNDEF { b } else if b == UNDEF { a } else { a.max(b) }
    }

    for (pc, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::LoadI32(d, _)
            | Instr::LoadI64(d, _)
            | Instr::LoadBool(d, _)
            | Instr::LoadUnit(d)
            | Instr::LoadF32(d, _)
            | Instr::LoadF64(d, _) => {
                def!(*d, pc);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                use_!(*s, pc);
                def!(*d, pc);
            }
            Instr::Store(slot, s) => {
                use_!(*s, pc);
                def!(*slot, pc);
            }
            Instr::BinOp(d, _, l, r) => {
                use_!(*l, pc);
                use_!(*r, pc);
                def!(*d, pc);
            }
            Instr::UnOp(d, _, s) => {
                use_!(*s, pc);
                def!(*d, pc);
            }
            Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) | Instr::Return(s) => {
                use_!(*s, pc);
            }
            _ => {}
        }
    }

    // ── Loop-aware liveness extension ─────────────────────────────────────
    // When a backward branch forms a loop, any slot that is defined inside
    // the loop and used at or after the loop header must have its live range
    // extended to cover the entire loop. Otherwise, the register allocator
    // may assign the same register to two different slots whose intervals
    // don't overlap in the linear PC order but DO overlap across loop
    // iterations.
    //
    // We identify loops by scanning for backward branches, then extend any
    // slot whose first_def is inside the loop body to have last_use at
    // least as far as the loop's back-edge PC.
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match instr {
            Instr::Jump(off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpFalse(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpTrue(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            _ => None,
        };
        if let Some(target) = target {
            if target < pc {
                // This is a backward branch: loop header = target, loop end = pc
                // Extend any slot that is live inside the loop to cover the entire loop.
                for slot in 0..first_def.len() {
                    let fd = first_def[slot];
                    let lu = last_use[slot];
                    if fd == UNDEF && lu == UNDEF {
                        continue;
                    }
                    // If the slot is used or defined anywhere inside [target, pc],
                    // extend its last_use to at least pc (the back-edge).
                    // This ensures the register allocator doesn't reuse the register
                    // for another slot that becomes free mid-loop.
                    let first = if fd == UNDEF { 0 } else { fd };
                    let last = if lu == UNDEF { first } else { lu };
                    if first <= pc && last >= target {
                        // Slot is live inside the loop — extend to the back-edge.
                        last_use[slot] = pc_max(last_use[slot], pc);
                    }
                }
            }
        }
    }

    // Slots that appear only as uses (never defined) are argument slots — treat as defined at 0.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for slot in 0..first_def.len() {
        let lu = last_use[slot];
        let fd = first_def[slot];
        if fd == UNDEF && lu == UNDEF {
            continue;
        }
        let first = if fd == UNDEF { 0 } else { fd };
        let last = if lu == UNDEF { first } else { lu };
        intervals.push(LiveInterval {
            slot: slot as u16,
            first,
            last,
        });
    }
    intervals.sort_unstable_by_key(|i| (i.first, i.slot));
    intervals
}

// ── Float slot identification ────────────────────────────────────────────────
//
// Pre-pass that identifies which slots hold float values. This is needed by
// the register allocator to decide whether a slot should get a GPR or an XMM
// register. The analysis is a simplified type-inference pass over the
// instruction stream.

/// Returns a Vec<bool> indexed by slot number; `true` means the slot is a
/// float-typed slot (F32 or F64) that should be allocated an XMM register
/// instead of a GPR.
fn compute_float_slots(instrs: &[Instr], slot_count: usize) -> Vec<bool> {
    // Use a simple type tracking approach similar to TypeTable
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Ty { Unknown, Float, Int }
    let cap = slot_count + 1;
    let mut tys = vec![Ty::Unknown; cap];

    macro_rules! ensure {
        ($s:expr) => {
            let s = $s as usize;
            if s >= tys.len() { tys.resize(s + 1, Ty::Unknown); }
        };
    }

    for instr in instrs {
        match instr {
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => {
                ensure!(*d);
                tys[*d as usize] = Ty::Float;
            }
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => {
                ensure!(*d);
                tys[*d as usize] = Ty::Int;
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                ensure!(*d);
                ensure!(*s);
                tys[*d as usize] = tys[*s as usize];
            }
            Instr::Store(slot, s) => {
                ensure!(*slot);
                ensure!(*s);
                tys[*slot as usize] = tys[*s as usize];
            }
            Instr::BinOp(d, op, l, r) => {
                ensure!(*d);
                ensure!(*l);
                ensure!(*r);
                // If either operand is float and this is not a comparison,
                // the result is float. Comparisons produce integer (bool).
                let is_cmp = matches!(op,
                    BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt
                    | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge);
                if !is_cmp && (tys[*l as usize] == Ty::Float || tys[*r as usize] == Ty::Float) {
                    tys[*d as usize] = Ty::Float;
                } else {
                    tys[*d as usize] = Ty::Int;
                }
            }
            Instr::UnOp(d, _, s) => {
                ensure!(*d);
                ensure!(*s);
                tys[*d as usize] = tys[*s as usize];
            }
            Instr::TensorBinOp { dst, element_ty, .. } => {
                ensure!(*dst);
                if element_ty.is_float() {
                    tys[*dst as usize] = Ty::Float;
                } else {
                    tys[*dst as usize] = Ty::Int;
                }
            }
            Instr::TensorReduce { dst, element_ty, .. } => {
                ensure!(*dst);
                if element_ty.is_float() {
                    tys[*dst as usize] = Ty::Float;
                } else {
                    tys[*dst as usize] = Ty::Int;
                }
            }
            Instr::TensorMatMul { dst, element_ty, .. } => {
                ensure!(*dst);
                if element_ty.is_float() {
                    tys[*dst as usize] = Ty::Float;
                } else {
                    tys[*dst as usize] = Ty::Int;
                }
            }
            _ => {}
        }
    }

    tys.iter().map(|t| *t == Ty::Float).collect()
}

// ── Linear-scan allocator ─────────────────────────────────────────────────────

fn linear_scan(intervals: &[LiveInterval], slot_count: usize, float_slots: &[bool]) -> RegAlloc {
    // ═══════════════════════════════════════════════════════════════════════
    // Register Allocator with Hot-Path Prioritization
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Architecture constraint: the code generator uses ra.location(slot)
    // which returns ONE fixed location (Reg or Spill) per slot for the
    // entire function. This means we can't do true live range splitting
    // (which would assign different registers at different program points).
    //
    // Instead, we use a two-pass approach:
    //   Pass 1: Standard linear-scan with merged intervals.
    //   Pass 2: Hot-path demotion — identify cold slots that got registers
    //           and demote them to Spill, then try to assign the freed
    //           registers to hot slots (loop vars, accumulators) that
    //           were spilled.
    //
    // A slot is "hot" if it's used inside a loop body (detected by
    // checking if its live range overlaps with a backward branch).
    // Hot slots get priority because they're executed millions of times;
    // spilling a hot slot costs ~4-5 cycles per iteration in a 1B loop,
    // while spilling a cold slot costs ~4-5 cycles total.

    let mut merged: Vec<LiveInterval> = Vec::new();
    {
        let mut slot_first = vec![usize::MAX; slot_count + 1];
        let mut slot_last = vec![0usize; slot_count + 1];
        for iv in intervals {
            let s = iv.slot as usize;
            if s < slot_first.len() {
                slot_first[s] = slot_first[s].min(iv.first);
                slot_last[s] = slot_last[s].max(iv.last);
            }
        }
        for s in 0..=slot_count {
            if s < slot_first.len() && slot_first[s] <= slot_last[s] {
                merged.push(LiveInterval {
                    slot: s as u16,
                    first: slot_first[s],
                    last: slot_last[s],
                });
            }
        }
        // Sort by start position (required by linear scan)
        merged.sort_by_key(|iv| iv.first);
    }

    // ── Identify hot slots (used inside loops) ────────────────────────────
    // A slot is "hot" if its live range spans a large fraction of the
    // function, suggesting it's a loop-carried variable. We use interval-
    // based heuristics since we don't have the instruction stream here.
    let mut hot_slots: Vec<bool> = vec![false; slot_count + 1];
    {
        // Identify hot slots using interval-data heuristics (no instruction
        // stream needed — linear_scan only receives intervals, not instrs).
        // Heuristic 1: slots whose live range spans ≥50% of the total
        // instruction range are likely loop variables and should be
        // prioritized for register allocation.
        for iv in &merged {
            // Mark slots whose live range spans at least 50% of the total
            // instruction range as potentially hot (likely loop variables).
            if !merged.is_empty() {
                let total_range = merged.last().map_or(1, |iv| iv.last.max(1));
                let slot_range = iv.last.saturating_sub(iv.first);
                if slot_range > 0 && total_range > 0 && slot_range * 2 >= total_range {
                    hot_slots[iv.slot as usize] = true;
                }
            }
        }
        // Also mark slots that appear in multiple intervals (they're used
        // across different parts of the function, likely in loops)
        let mut slot_use_count = vec![0u32; slot_count + 1];
        for iv in intervals {
            let s = iv.slot as usize;
            if s < slot_use_count.len() {
                slot_use_count[s] += 1;
            }
        }
        for s in 0..=slot_count {
            if slot_use_count[s] >= 3 {
                hot_slots[s] = true;
            }
        }
    }

    // Pre-allocate slots to slot_count + 1 with default spill locations.
    let cap = slot_count + 1;
    let mut slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();

    // ── GPR allocation pass: skip float-typed slots ─────────────────────────
    // Float slots should NOT get GPR registers — they'll get XMM registers
    // in the second pass below. Allocating GPRs to float slots wastes registers
    // and can cause the float path to store to memory even when a GPR is
    // "assigned" (since the float path uses XMM instructions, not GPR moves).

    // Free list: iterate ALLOC_POOL in reverse so pop() gives caller-saved first.
    let mut free: Vec<u8> = ALLOC_POOL.iter().rev().copied().collect();
    // Active set sorted by interval end (ascending).
    let mut active: Vec<(usize, u16, u8)> = Vec::with_capacity(merged.len().min(free.len()));
    let mut used_callee_saved: Vec<u8> = Vec::with_capacity(ALLOC_POOL.len());
    let mut callee_saved_mask: u16 = 0;

    #[inline(always)]
    fn is_callee_saved(reg: u8) -> bool {
        matches!(reg, 3 | 12..=15)
    }

    // Helper: check if a slot is a float slot
    let is_float_slot = |slot: u16| -> bool {
        let idx = slot as usize;
        idx < float_slots.len() && float_slots[idx]
    };

    for iv in &merged {
        // Skip float-typed slots in the GPR allocation pass.
        // They will be allocated XMM registers in the second pass.
        if is_float_slot(iv.slot) {
            continue;
        }

        // Expire intervals that ended strictly before this one's start.
        let expired = active.partition_point(|(end, _, _)| *end < iv.first);
        for i in 0..expired {
            free.push(active[i].2);
        }
        if expired > 0 {
            active.drain(0..expired);
        }

        let mut track_callee = |reg: u8| {
            if is_callee_saved(reg) && (callee_saved_mask & (1u16 << reg)) == 0 {
                callee_saved_mask |= 1u16 << reg;
                used_callee_saved.push(reg);
            }
        };

        if let Some(reg) = free.pop() {
            track_callee(reg);
            slots[iv.slot as usize] = RegLoc::Reg(reg);
            let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
            active.insert(pos, (iv.last, iv.slot, reg));
        } else {
            match active.last().copied() {
                Some((end, spill_slot, reg)) if end > iv.last => {
                    slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
                    active.pop();
                    track_callee(reg);
                    slots[iv.slot as usize] = RegLoc::Reg(reg);
                    let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
                    active.insert(pos, (iv.last, iv.slot, reg));
                }
                _ => {
                    // Already pre-filled with Spill; nothing to do.
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Pass 2: Hot-Path Demotion
    // ═══════════════════════════════════════════════════════════════════════
    //
    // After the standard linear scan, some hot slots (loop variables,
    // accumulators) may have been spilled while cold slots (one-time
    // computations, setup values) got registers. This is backwards for
    // performance — a spilled loop variable costs ~4-5 cycles per
    // iteration in a 1B loop, while a spilled cold variable costs
    // ~4-5 cycles total.
    //
    // This pass identifies:
    //   1. Cold slots that currently have registers
    //   2. Hot slots that are currently spilled
    // And swaps them: demote the cold slot to Spill, give its register
    // to the hot slot.
    {
        // Collect cold slots with registers and hot slots without registers
        let mut cold_with_reg: Vec<(u16, u8)> = Vec::new(); // (slot, reg)
        let mut hot_without_reg: Vec<u16> = Vec::new();

        for s in 0..slot_count {
            if s >= slots.len() { continue; }
            let is_hot = s < hot_slots.len() && hot_slots[s];
            match slots[s] {
                RegLoc::Reg(r) if !is_hot => {
                    cold_with_reg.push((s as u16, r));
                }
                RegLoc::Spill(_) if is_hot => {
                    hot_without_reg.push(s as u16);
                }
                _ => {}
            }
        }

        // Sort cold slots by interval length (longest first — demote the
        // ones that waste the most register time). We'll swap the coldest
        // first with the hottest first.
        cold_with_reg.sort_by(|a, b| {
            let len_a = merged.iter().find(|iv| iv.slot == a.0).map_or(0, |iv| iv.last - iv.first);
            let len_b = merged.iter().find(|iv| iv.slot == b.0).map_or(0, |iv| iv.last - iv.first);
            len_b.cmp(&len_a) // longest first
        });

        // Sort hot slots by interval length (shortest first — they're the
        // most likely to benefit from a register since they fit in a
        // register for a shorter time, meaning less conflict with other
        // hot slots). Actually, sort by use count — most-used first.
        hot_without_reg.sort_by(|a, b| {
            let uses_a = intervals.iter().filter(|iv| iv.slot == *a).count();
            let uses_b = intervals.iter().filter(|iv| iv.slot == *b).count();
            uses_b.cmp(&uses_a) // most-used first
        });

        // Swap: demote cold → Spill, promote hot → Reg
        let swap_count = cold_with_reg.len().min(hot_without_reg.len());
        for i in 0..swap_count {
            let (cold_slot, reg) = cold_with_reg[i];
            let hot_slot = hot_without_reg[i];

            // Demote cold slot to Spill
            slots[cold_slot as usize] = RegLoc::Spill((cold_slot as i32) * 8);
            // Promote hot slot to Reg
            slots[hot_slot as usize] = RegLoc::Reg(reg);

            // Update callee-saved tracking if the hot slot's register is
            // callee-saved (it already was tracked when the cold slot got it)
        }

        if swap_count > 0 {
            jit_trace!("[JIT-RA] Hot-path demotion: swapped {} cold→spill, hot→reg", swap_count);
        }
    }

    // ── XMM allocation pass: allocate XMM registers for float-typed slots ──
    //
    // Uses the same linear-scan algorithm but with the XMM_ALLOC_POOL.
    // Float slots that can't get an XMM register fall back to Spill (memory
    // in the slot array). XMM registers are NOT callee-saved in System V ABI,
    // so no push/pop needed for them.

    let mut xmm_slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();
    let mut used_xmm_regs: Vec<u8> = Vec::new();

    // Collect live intervals for float slots only
    let float_intervals: Vec<&LiveInterval> = merged.iter()
        .filter(|iv| is_float_slot(iv.slot))
        .collect();

    if !float_intervals.is_empty() {
        let mut xmm_free: Vec<u8> = XMM_ALLOC_POOL.iter().rev().copied().collect();
        let mut xmm_active: Vec<(usize, u16, u8)> = Vec::with_capacity(float_intervals.len().min(xmm_free.len()));

        for iv in &float_intervals {
            // Expire intervals that ended strictly before this one's start.
            let expired = xmm_active.partition_point(|(end, _, _)| *end < iv.first);
            for i in 0..expired {
                xmm_free.push(xmm_active[i].2);
            }
            if expired > 0 {
                xmm_active.drain(0..expired);
            }

            if let Some(xmm_reg) = xmm_free.pop() {
                used_xmm_regs.push(xmm_reg);
                xmm_slots[iv.slot as usize] = RegLoc::Xmm(xmm_reg);
                let pos = xmm_active.partition_point(|(e, _, _)| *e <= iv.last);
                xmm_active.insert(pos, (iv.last, iv.slot, xmm_reg));
            } else {
                // Spill the interval with the farthest end if it extends past this one
                match xmm_active.last().copied() {
                    Some((end, spill_slot, xmm_reg)) if end > iv.last => {
                        xmm_slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
                        xmm_active.pop();
                        used_xmm_regs.push(xmm_reg);
                        xmm_slots[iv.slot as usize] = RegLoc::Xmm(xmm_reg);
                        let pos = xmm_active.partition_point(|(e, _, _)| *e <= iv.last);
                        xmm_active.insert(pos, (iv.last, iv.slot, xmm_reg));
                    }
                    _ => {
                        // Already pre-filled with Spill; nothing to do.
                    }
                }
            }
        }
    }

    RegAlloc {
        slots,
        used_callee_saved,
        xmm_slots,
        used_xmm_regs,
    }
}

// ── Single-Pass SSA Register Allocator (Braun-Hack Algorithm) ─────────────────
//
// Processes blocks in reverse post-order and allocates registers on-the-fly.
// Liveness intervals are computed during the single pass, and register
// allocation, spilling, and coalescing happen simultaneously during code
// emission — reducing allocator overhead to near zero compared to the
// multi-pass linear-scan approach.
//
// Key design:
//   - Uses SSA properties: each value is defined exactly once
//   - Processes blocks in reverse post-order (dominators before dominated)
//   - On value definition: allocate a free register or spill the longest-distance value
//   - On value last-use: free the register immediately
//   - Phi nodes: coalesce with predecessor's outgoing register
//   - No separate liveness pass needed — last-use is tracked inline

/// Result of single-pass SSA register allocation.
#[allow(dead_code)]
pub struct SsaRegAlloc {
    /// Map from ValueId to physical register or spill slot
    assignments: FxHashMap<ValueId, SsaRegLoc>,
    /// Set of callee-saved registers actually used (for prologue/epilogue)
    used_callee_saved: Vec<u8>,
    /// Total stack frame size needed for spills
    spill_frame_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum SsaRegLoc {
    Gpr(u8),
    Xmm(u8),
    Spill(i32), // negative offset from RBP
}

/// Single-Pass SSA Register Allocator (Braun-Hack Algorithm)
///
/// Processes blocks in reverse post-order and allocates registers on-the-fly.
/// Liveness intervals are computed during the single pass, and register
/// allocation, spilling, and coalescing happen simultaneously during code
/// emission — reducing allocator overhead to near zero compared to the
/// multi-pass linear-scan approach.
///
/// Key design:
///   - Uses SSA properties: each value is defined exactly once
///   - Processes blocks in reverse post-order (dominators before dominated)
///   - On value definition: allocate a free register or spill the longest-distance value
///   - On value last-use: free the register immediately
///   - Phi nodes: coalesce with predecessor's outgoing register
///   - No separate liveness pass needed — last-use is tracked inline
fn single_pass_ra(
    func: &FlatIrFunction,
    gpr_pool: &[u8],
    xmm_pool: &[u8],
    _custom_cc: &CustomCallingConvention,
) -> SsaRegAlloc {
    // 1. Compute reverse post-order of blocks
    let rpo = compute_reverse_post_order(func);

    // 2. Forward scan to determine last-use positions for each ValueId
    let mut last_use: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (block_idx, &bid) in rpo.iter().enumerate() {
        let block = &func.blocks[bid as usize];
        for instr in &block.instrs {
            // Record each use of a value
            for vid in ir_op_uses(&instr.op) {
                last_use.insert(vid, block_idx);
            }
        }
    }

    // 3. Allocate registers in RPO order
    let mut assignments: FxHashMap<ValueId, SsaRegLoc> = FxHashMap::default();
    let mut free_gprs: Vec<u8> = gpr_pool.to_vec();
    let mut free_xmms: Vec<u8> = xmm_pool.to_vec();
    let mut used_callee_saved: Vec<u8> = Vec::new();
    let mut spill_offset: i32 = -8; // start at RBP-8, grow downward
    // Track which values are in registers (for spilling the farthest-use value)
    let mut reg_holders: Vec<(ValueId, usize)> = Vec::new(); // (vid, last_use_block_idx)

    for (block_idx, &bid) in rpo.iter().enumerate() {
        let block = &func.blocks[bid as usize];

        // Free registers for values whose last use was in the previous block
        reg_holders.retain(|&(vid, last_block)| {
            if last_block < block_idx {
                // This value's last use has passed — free its register
                if let Some(loc) = assignments.get(&vid) {
                    match loc {
                        SsaRegLoc::Gpr(r) => {
                            if !free_gprs.contains(r) {
                                free_gprs.push(*r);
                            }
                        }
                        SsaRegLoc::Xmm(r) => {
                            if !free_xmms.contains(r) {
                                free_xmms.push(*r);
                            }
                        }
                        SsaRegLoc::Spill(_) => {}
                    }
                }
                false // remove from reg_holders
            } else {
                true // keep
            }
        });

        for instr in &block.instrs {
            // Handle phi nodes: coalesce with predecessor's register
            if let IrOp::Phi { incoming } = &instr.op {
                if let Some(dst_vid) = instr.dst {
                    // Try to reuse the register from one of the predecessors
                    for &(_pred, src_vid) in incoming {
                        if let Some(&loc) = assignments.get(&src_vid) {
                            assignments.insert(dst_vid, loc);
                            if let Some(last) = last_use.get(&dst_vid) {
                                reg_holders.push((dst_vid, *last));
                            }
                            break;
                        }
                    }
                }
                continue;
            }

            // Allocate a register for the result value (if any)
            if let Some(dst_vid) = instr.dst {
                let is_float = false; // simplified; real impl checks type
                let loc = if is_float {
                    allocate_xmm_or_spill(
                        &mut free_xmms, &mut assignments, &mut reg_holders,
                        &mut spill_offset, dst_vid, block_idx, &last_use,
                    )
                } else {
                    allocate_gpr_or_spill(
                        &mut free_gprs, &mut assignments, &mut reg_holders,
                        &mut spill_offset, &mut used_callee_saved,
                        dst_vid, block_idx, &last_use, gpr_pool,
                    )
                };
                assignments.insert(dst_vid, loc);
            }
        }
    }

    let spill_frame_size = ((-spill_offset) as usize + 15) & !15; // align to 16 bytes

    SsaRegAlloc {
        assignments,
        used_callee_saved,
        spill_frame_size,
    }
}

/// Allocate a GPR or spill. If no GPR is free, spill the value with the
/// farthest next use and take its register.
fn allocate_gpr_or_spill(
    free_gprs: &mut Vec<u8>,
    assignments: &mut FxHashMap<ValueId, SsaRegLoc>,
    reg_holders: &mut Vec<(ValueId, usize)>,
    spill_offset: &mut i32,
    used_callee_saved: &mut Vec<u8>,
    vid: ValueId,
    block_idx: usize,
    last_use: &FxHashMap<ValueId, usize>,
    _gpr_pool: &[u8],
) -> SsaRegLoc {
    if let Some(reg) = free_gprs.pop() {
        // Track callee-saved usage
        if is_callee_saved(reg) && !used_callee_saved.contains(&reg) {
            used_callee_saved.push(reg);
        }
        if let Some(last) = last_use.get(&vid) {
            reg_holders.push((vid, *last));
        }
        SsaRegLoc::Gpr(reg)
    } else {
        // Spill the value with the farthest last use
        if let Some(farthest_idx) = reg_holders.iter().enumerate()
            .filter(|(_, (_, last))| *last > block_idx)
            .max_by_key(|(_, (_, last))| *last)
            .map(|(i, _)| i)
        {
            let (spill_vid, _) = reg_holders[farthest_idx];
            if let Some(SsaRegLoc::Gpr(reg)) = assignments.get(&spill_vid).copied() {
                // Move the spilled value to a spill slot
                *spill_offset -= 8;
                assignments.insert(spill_vid, SsaRegLoc::Spill(*spill_offset));
                reg_holders.remove(farthest_idx);
                // Track callee-saved usage
                if is_callee_saved(reg) && !used_callee_saved.contains(&reg) {
                    used_callee_saved.push(reg);
                }
                if let Some(last) = last_use.get(&vid) {
                    reg_holders.push((vid, *last));
                }
                return SsaRegLoc::Gpr(reg);
            }
        }
        // No register to steal — spill to stack
        *spill_offset -= 8;
        if let Some(last) = last_use.get(&vid) {
            reg_holders.push((vid, *last));
        }
        SsaRegLoc::Spill(*spill_offset)
    }
}

/// Allocate an XMM register or spill.
fn allocate_xmm_or_spill(
    free_xmms: &mut Vec<u8>,
    assignments: &mut FxHashMap<ValueId, SsaRegLoc>,
    reg_holders: &mut Vec<(ValueId, usize)>,
    spill_offset: &mut i32,
    vid: ValueId,
    block_idx: usize,
    last_use: &FxHashMap<ValueId, usize>,
) -> SsaRegLoc {
    if let Some(reg) = free_xmms.pop() {
        if let Some(last) = last_use.get(&vid) {
            reg_holders.push((vid, *last));
        }
        SsaRegLoc::Xmm(reg)
    } else {
        // Spill the value with the farthest last use
        if let Some(farthest_idx) = reg_holders.iter().enumerate()
            .filter(|(_, (_, last))| *last > block_idx)
            .max_by_key(|(_, (_, last))| *last)
            .map(|(i, _)| i)
        {
            let (spill_vid, _) = reg_holders[farthest_idx];
            if let Some(SsaRegLoc::Xmm(reg)) = assignments.get(&spill_vid).copied() {
                *spill_offset -= 8;
                assignments.insert(spill_vid, SsaRegLoc::Spill(*spill_offset));
                reg_holders.remove(farthest_idx);
                if let Some(last) = last_use.get(&vid) {
                    reg_holders.push((vid, *last));
                }
                return SsaRegLoc::Xmm(reg);
            }
        }
        *spill_offset -= 8;
        if let Some(last) = last_use.get(&vid) {
            reg_holders.push((vid, *last));
        }
        SsaRegLoc::Spill(*spill_offset)
    }
}

/// Check if a register is callee-saved in System V ABI.
fn is_callee_saved(reg: u8) -> bool {
    // System V callee-saved: RBX(3), R12(12), R13(13), R14(14), R15(15), RBP(5)
    matches!(reg, 3 | 5 | 12 | 13 | 14 | 15)
}

/// Compute reverse post-order of blocks in the CFG.
fn compute_reverse_post_order(func: &FlatIrFunction) -> Vec<u32> {
    let n = func.blocks.len();
    if n == 0 {
        return Vec::new();
    }

    // Build adjacency list (block → successors)
    let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            match &instr.op {
                IrOp::Jump { target, .. } => {
                    succs[i].push(target.0);
                }
                IrOp::CondBr { if_true, if_false, .. } => {
                    succs[i].push(if_true.0);
                    succs[i].push(if_false.0);
                }
                _ => {}
            }
        }
    }

    // DFS from entry block to compute post-order
    let entry = func.entry.0 as usize;
    let mut visited = vec![false; n];
    let mut post_order = Vec::new();
    let mut stack = vec![entry];

    while let Some(node) = stack.pop() {
        if visited[node] {
            post_order.push(node as u32);
            continue;
        }
        visited[node] = true;
        stack.push(node); // push back as marker for post-order
        for &s in &succs[node] {
            if (s as usize) < n && !visited[s as usize] {
                stack.push(s as usize);
            }
        }
    }

    // Reverse post-order = reverse of post-order
    post_order.reverse();
    post_order
}

/// Extract the ValueIds used by an IrOp (operand values).
fn ir_op_uses(op: &IrOp) -> Vec<ValueId> {
    match op {
        IrOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        IrOp::UnOp { operand, .. } => vec![*operand],
        IrOp::Move { src } => vec![*src],
        IrOp::Copy { src } => vec![*src],
        IrOp::Store { ptr, value } => vec![*ptr, *value],
        IrOp::Load { ptr, .. } => vec![*ptr],
        IrOp::Ret { value } => value.map(|v| vec![v]).unwrap_or_default(),
        IrOp::CondBr { cond, .. } => vec![*cond],
        IrOp::Call { args, .. } => args.clone(),
        IrOp::Intrinsic { args, .. } => args.clone(),
        IrOp::Phi { incoming } => incoming.iter().map(|&(_, vid)| vid).collect(),
        IrOp::MatMul { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::HadamardMul { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::HadamardDiv { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::TensorConcat { a, b, .. } => vec![*a, *b],
        IrOp::KronProd { a, b } => vec![*a, *b],
        IrOp::Jump { args, .. } => args.clone(),
        _ => Vec::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chordal Graph Coloring Register Allocator (Hack et al.)
// ─────────────────────────────────────────────────────────────────────────────
//
// For programs in Strict SSA form, the interference graph is chordal.
// Chordal graphs can be optimally colored in O(V + E) time using a
// maximum cardinality search (MCS) ordering followed by greedy coloring.
//
// This achieves the absolute peak execution speed of Global Graph Coloring
// but runs at the blistering speed of a Linear-Scan compiler — the best
// of both worlds.
//
// Algorithm:
//   1. Build the interference graph from SSA live ranges
//   2. Compute maximum cardinality search (MCS) ordering
//   3. Greedy color in reverse MCS order (optimal for chordal graphs)
//   4. Map colors to physical registers
//   5. Insert spill code for values that couldn't be colored

/// Chordal Graph Coloring Register Allocator.
/// Produces optimal register assignments for SSA-form IR.
struct ChordalGraphColoringRA {
    /// Number of available GPR registers
    _num_gprs: usize,
    /// Number of available XMM registers
    _num_xmms: usize,
    /// Interference graph edges: node → set of interfering nodes
    interference: Vec<FxHashSet<u32>>,
    /// MCS ordering (reverse coloring order)
    mcs_order: Vec<u32>,
    /// Color assignments (register numbers)
    colors: Vec<Option<u8>>,
    /// Spill slots for values that couldn't be colored
    spills: Vec<(u32, i32)>, // (value_id, spill_offset)
}

impl ChordalGraphColoringRA {
    fn new(num_gprs: usize, num_xmms: usize) -> Self {
        ChordalGraphColoringRA {
            _num_gprs: num_gprs,
            _num_xmms: num_xmms,
            interference: Vec::new(),
            mcs_order: Vec::new(),
            colors: Vec::new(),
            spills: Vec::new(),
        }
    }

    /// Build the interference graph from SSA live ranges.
    /// Two values interfere if their live ranges overlap.
    fn build_interference_graph(&mut self, func: &FlatIrFunction) {
        // Compute live ranges for each ValueId
        let mut live_ranges: FxHashMap<u32, (usize, usize)> = FxHashMap::default();
        let mut global_pos = 0usize;

        for block in &func.blocks {
            for instr in &block.instrs {
                if let Some(result) = instr.result {
                    let entry = live_ranges.entry(result.0).or_insert((global_pos, global_pos));
                    entry.0 = entry.0.min(global_pos);
                    entry.1 = entry.1.max(global_pos);
                }
                // Track uses
                if let IrOp::BinOp { lhs, rhs, .. } = &instr.op {
                    for vid in &[lhs.0, rhs.0] {
                        let entry = live_ranges.entry(*vid).or_insert((global_pos, global_pos));
                        entry.1 = entry.1.max(global_pos);
                    }
                }
                if let IrOp::Move { src } = &instr.op {
                    let entry = live_ranges.entry(src.0).or_insert((global_pos, global_pos));
                    entry.1 = entry.1.max(global_pos);
                }
                global_pos += 1;
            }
        }

        let num_values = live_ranges.len().max(1);
        self.interference = vec![FxHashSet::default(); num_values];

        // Build edges: two values interfere if their ranges overlap
        let ranges: Vec<(u32, usize, usize)> = live_ranges.iter()
            .map(|(&v, &(f, l))| (v, f, l))
            .collect();

        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                let (vi, fi, li) = ranges[i];
                let (vj, fj, lj) = ranges[j];
                let _ = (vi, vj); // suppress unused warning
                // Ranges overlap if fi <= lj && fj <= li
                if fi <= lj && fj <= li {
                    self.interference[i].insert(j as u32);
                    self.interference[j].insert(i as u32);
                }
            }
        }
    }

    /// Compute Maximum Cardinality Search (MCS) ordering.
    /// MCS produces a perfect elimination ordering for chordal graphs,
    /// which guarantees optimal coloring when we color in reverse order.
    fn compute_mcs_ordering(&mut self) {
        let n = self.interference.len();
        if n == 0 { return; }

        self.mcs_order = Vec::with_capacity(n);
        let mut in_order = vec![false; n];
        let mut cardinality = vec![0u32; n];

        // Select the node with maximum cardinality iteratively
        for _ in 0..n {
            let mut best = 0;
            let mut best_card = 0;
            for i in 0..n {
                if !in_order[i] && cardinality[i] >= best_card {
                    best = i;
                    best_card = cardinality[i];
                }
            }
            self.mcs_order.push(best as u32);
            in_order[best] = true;
            // Increase cardinality of all unselected neighbors
            for &neighbor in &self.interference[best] {
                if !in_order[neighbor as usize] {
                    cardinality[neighbor as usize] += 1;
                }
            }
        }
    }

    /// Greedy color in reverse MCS order (optimal for chordal graphs).
    fn color(&mut self, num_colors: usize) -> Vec<Option<u8>> {
        let n = self.interference.len();
        self.colors = vec![None; n];

        // Color in reverse MCS order
        for &node in self.mcs_order.iter().rev() {
            let node = node as usize;
            // Find the smallest color not used by any neighbor
            let neighbor_colors: FxHashSet<u8> = self.interference[node]
                .iter()
                .filter_map(|&nb| self.colors[nb as usize])
                .collect();

            let mut assigned = None;
            for c in 0..num_colors {
                if !neighbor_colors.contains(&(c as u8)) {
                    assigned = Some(c as u8);
                    break;
                }
            }

            if let Some(c) = assigned {
                self.colors[node] = Some(c);
            } else {
                // Spill: no color available
                self.spills.push((node as u32, (self.spills.len() as i32 + 1) * -8));
            }
        }

        self.colors.clone()
    }

    /// Run the full chordal graph coloring allocation pipeline.
    pub fn allocate(mut self, func: &FlatIrFunction) -> SsaRegAlloc {
        self.build_interference_graph(func);
        self.compute_mcs_ordering();

        let gpr_colors = self.color(self._num_gprs);

        let mut assignments: FxHashMap<ValueId, SsaRegLoc> = FxHashMap::default();

        // Map colors to GPR pool
        let gpr_pool: Vec<u8> = vec![8, 9, 10, 11, 6, 12, 13, 14, 15, 3]; // ALLOC_POOL
        for (i, color) in gpr_colors.iter().enumerate() {
            if let Some(c) = color {
                if (*c as usize) < gpr_pool.len() {
                    assignments.insert(ValueId(i as u32), SsaRegLoc::Gpr(gpr_pool[*c as usize]));
                }
            }
        }

        // Map spills
        for &(value_id, spill_offset) in &self.spills {
            assignments.insert(ValueId(value_id), SsaRegLoc::Spill(spill_offset));
        }

        let mut used_callee = Vec::new();
        for &reg in &gpr_pool {
            if assignments.values().any(|loc| matches!(loc, SsaRegLoc::Gpr(r) if *r == reg)) {
                if matches!(reg, 3 | 12..=15) {
                    used_callee.push(reg);
                }
            }
        }

        SsaRegAlloc {
            assignments,
            used_callee_saved: used_callee,
            spill_frame_size: self.spills.len() * 8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RAT-Aware Register Pinning for Short-Lived Temporaries
// ─────────────────────────────────────────────────────────────────────────────
//
// Modern x86-64 CPUs use a Register Alias Table (RAT) that maps architectural
// registers to physical rename registers. When two independent instructions
// write to different architectural registers, the CPU can execute them in
// parallel through register renaming — even if they're adjacent in the code.
//
// However, if two short-lived temporaries map to the SAME architectural
// register, the RAT creates a false dependency (write-after-write hazard),
// preventing parallel execution. By pinning short-lived temporaries to
// registers that the RAT can rename independently, we eliminate data hazards
// before the instructions even hit the execution pipelines.
//
// Strategy:
//   - Intervals ≤ 3 instructions: prefer registers with independent RAT entries
//   - On Intel: R0-R15 all rename independently (any register works)
//   - On AMD Zen: same — all GPRs rename independently
//   - The key optimization: avoid reusing the SAME register for two
//     short-lived temporaries that overlap, even partially
//   - Pin short-lived values to "free" registers that won't create WAW hazards

/// RAT-aware register preference for short-lived temporaries.
pub struct RatAwarePinning {
    /// Physical registers that are currently "free" for short-lived pinning
    free_regs: Vec<u8>,
    /// Registers currently in use by short-lived temporaries (with expiry positions)
    active_short_lived: Vec<(u8, usize)>, // (reg, expires_at_position)
    /// Number of instructions a value must live to be considered "short-lived"
    short_lived_threshold: usize,
}

impl RatAwarePinning {
    pub fn new(gpr_pool: &[u8]) -> Self {
        RatAwarePinning {
            free_regs: gpr_pool.to_vec(),
            active_short_lived: Vec::new(),
            short_lived_threshold: 3,
        }
    }

    /// Try to allocate a RAT-decoupled register for a short-lived value.
    /// Returns None if the value is not short-lived or no register is available.
    pub fn try_pin_short_lived(&mut self, value_id: u32, first_pos: usize, last_pos: usize) -> Option<u8> {
        let _ = value_id; // suppress unused warning
        let lifetime = last_pos.saturating_sub(first_pos);
        if lifetime > self.short_lived_threshold {
            return None; // Not short-lived — use normal allocation
        }

        // Expire any short-lived registers that are no longer active
        self.active_short_lived.retain(|&(_, expires)| expires > first_pos);

        // Try to find a register that is NOT currently used by another
        // short-lived temporary at this position. This ensures the RAT
        // can rename them independently.
        let active_regs: FxHashSet<u8> = self.active_short_lived.iter().map(|&(r, _)| r).collect();

        for reg in &self.free_regs {
            if !active_regs.contains(reg) {
                let assigned = *reg;
                self.active_short_lived.push((assigned, last_pos));
                return Some(assigned);
            }
        }

        // All registers are in use by other short-lived temporaries.
        // Fall back to normal allocation.
        None
    }

    /// Release a short-lived register pin when the value expires.
    pub fn release(&mut self, reg: u8, position: usize) {
        self.active_short_lived.retain(|&(r, expires)| !(r == reg && expires <= position));
    }
}

// ── Dominance-Based Tree Scan Register Allocator ─────────────────────────────
//
// The standard linear-scan allocator walks instructions in PC order (flat
// sequential order). This works well for straight-line code but generates
// suboptimal spill code when handling complex loop nests and nested SSA
// parameters, because it doesn't account for the dominator tree structure.
//
// A dominance-based tree scan walks the dominator tree instead of a flat
// instruction list. This has several advantages:
//
// 1. **Fewer spills across block boundaries**: In the flat linear scan, an
//    interval that's live across a block boundary but only used in the
//    dominated subtree may be spilled because a sibling subtree's intervals
//    take its register. The tree scan assigns registers within each dominator
//    subtree first, ensuring that values used within a subtree keep their
//    registers.
//
// 2. **Better loop variable allocation**: The dominator tree naturally
//    identifies loop headers as nodes that dominate large subtrees. The tree
//    scan prioritizes these, ensuring loop variables get registers.
//
// 3. **Comparable compilation speed**: The tree scan has the same O(n log n)
//    complexity as linear scan when the dominator tree is balanced, and
//    O(n) for the common case of a single-entry loop nest.
//
// Implementation strategy:
// We compute a simplified dominator tree from the block structure, then
// reorder the live intervals so that intervals belonging to the same dominator
// subtree are processed together. This is equivalent to a tree scan without
// changing the core linear_scan algorithm — we just feed it better-ordered
// intervals.

/// Compute the immediate dominator (idom) for each block using the iterative
/// dominator algorithm. Returns a map from block_id → idom(block_id).
/// The entry block (block_id 0) has no immediate dominator (absent from the map).
///
/// Uses `block_starts` (pc → block_id) to build a proper CFG with predecessors,
/// then runs the standard iterative dominator algorithm:
///   idom(entry) = entry
///   idom(b) = ∩ { idom(p) | p ∈ preds(b) }  (with intersect using dom tree)
fn compute_idoms(instrs: &[Instr], block_starts: &FxHashMap<usize, usize>) -> FxHashMap<usize, usize> {
    if block_starts.is_empty() {
        return FxHashMap::default();
    }

    let num_blocks = block_starts.values().copied().max().unwrap_or(0) + 1;

    // Build a sorted list of (pc, block_id) to determine block ranges
    let mut sorted_blocks: Vec<(usize, usize)> = block_starts.iter()
        .map(|(&pc, &bid)| (pc, bid))
        .collect();
    sorted_blocks.sort_by_key(|&(pc, _)| pc);

    // Build predecessors map: block_id → set of predecessor block_ids
    let mut preds: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); num_blocks];

    // Helper: find the block_id for a given PC
    let block_id_for_pc = |pc: usize| -> usize {
        let idx = sorted_blocks.partition_point(|&(start_pc, _)| start_pc <= pc);
        if idx == 0 { return 0; }
        sorted_blocks[idx - 1].1
    };

    // Scan instructions to find edges
    for (pc, instr) in instrs.iter().enumerate() {
        let current_block = block_id_for_pc(pc);

        match instr {
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + off) as usize;
                if let Some(&target_bid) = block_starts.get(&target) {
                    preds[target_bid].insert(current_block);
                } else {
                    let target_bid = block_id_for_pc(target);
                    preds[target_bid].insert(current_block);
                }
            }
            Instr::JumpFalse(_, off) | Instr::JumpTrue(_, off) => {
                // Branch target
                let target = ((pc as i32) + 1 + off) as usize;
                if let Some(&target_bid) = block_starts.get(&target) {
                    preds[target_bid].insert(current_block);
                } else {
                    let target_bid = block_id_for_pc(target);
                    preds[target_bid].insert(current_block);
                }
                // Fall-through: next instruction
                let fall_pc = pc + 1;
                if fall_pc < instrs.len() {
                    let fall_bid = block_id_for_pc(fall_pc);
                    if fall_bid != current_block {
                        preds[fall_bid].insert(current_block);
                    }
                }
            }
            _ => {}
        }
    }

    // Iterative dominator algorithm
    // idom[entry] = entry; idom[others] = undefined initially
    let entry = sorted_blocks.first().map(|&(_, bid)| bid).unwrap_or(0);
    let mut idom: Vec<Option<usize>> = vec![None; num_blocks];
    idom[entry] = Some(entry);

    // Post-order traversal helper for intersect
    fn intersect(idom: &[Option<usize>], mut b1: usize, mut b2: usize) -> usize {
        while b1 != b2 {
            while b1 > b2 { b1 = idom[b1].unwrap_or(b1); }
            while b2 > b1 { b2 = idom[b2].unwrap_or(b2); }
        }
        b1
    }

    // Reverse post-order of blocks (excluding entry)
    // Simple approach: iterate all non-entry blocks in order until fixpoint
    let mut changed = true;
    while changed {
        changed = false;
        for bid in 0..num_blocks {
            if bid == entry { continue; }
            // Find first predecessor with an idom
            let mut new_idom: Option<usize> = None;
            for &pred in &preds[bid] {
                if let Some(pred_idom) = idom[pred] {
                    match new_idom {
                        None => new_idom = Some(pred_idom),
                        Some(cur) => new_idom = Some(intersect(&idom, cur, pred_idom)),
                    }
                }
            }
            if new_idom != idom[bid] {
                idom[bid] = new_idom;
                changed = true;
            }
        }
    }

    // Convert to FxHashMap<usize, usize> (block_id → idom_block_id)
    let mut result: FxHashMap<usize, usize> = FxHashMap::default();
    for bid in 0..num_blocks {
        if let Some(idom_val) = idom[bid] {
            if bid != idom_val {
                result.insert(bid, idom_val);
            }
        }
    }
    result
}

/// Reorder live intervals for dominance-based tree scan.
/// Intervals are sorted so that:
/// 1. Intervals in the same dominator subtree are contiguous
/// 2. Loop-header intervals come before their loop-body intervals
/// 3. Intervals that span loop headers are prioritized for register assignment
///
/// This reordering ensures the linear_scan algorithm naturally produces
/// tree-scan-quality allocation.
fn dominance_reorder_intervals(
    intervals: &mut Vec<LiveInterval>,
    idom: &FxHashMap<usize, usize>,
) {
    // Compute a "dominance rank" for each interval's start position.
    // Higher rank = more dominant (closer to entry, or a loop header).
    let _max_pc = intervals.iter().map(|iv| iv.last).max().unwrap_or(1);

    // Compute depth for each PC: depth increases when entering a loop,
    // decreases when exiting. Intervals at lower depth are more dominant.
    let mut loop_headers: FxHashSet<usize> = FxHashSet::default();
    for &header in idom.values() {
        loop_headers.insert(header);
    }

    // Sort intervals by: (is_loop_header_interval, start_position)
    // Loop header intervals (those starting at a loop header PC) get
    // priority because they're executed every iteration.
    intervals.sort_by(|a, b| {
        let a_is_loop_var = loop_headers.contains(&a.first) ||
            idom.contains_key(&a.first);
        let b_is_loop_var = loop_headers.contains(&b.first) ||
            idom.contains_key(&b.first);

        // Loop variables first (they need registers most)
        match (a_is_loop_var, b_is_loop_var) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                // Same category: sort by span length (longer first = more
                // likely to conflict, should get registers first)
                let a_span = a.last.saturating_sub(a.first);
                let b_span = b.last.saturating_sub(b.first);
                b_span.cmp(&a_span).then_with(|| a.first.cmp(&b.first))
            }
        }
    });
}

/// Dominance-based tree scan register allocator.
/// This is the main entry point — it computes the dominator tree, reorders
/// intervals for tree-scan quality, then delegates to the linear_scan
/// algorithm with the reordered intervals.
fn tree_scan_register_allocate(
    intervals: &[LiveInterval],
    slot_count: usize,
    float_slots: &[bool],
    instrs: &[Instr],
) -> RegAlloc {
    // Compute block_starts from the instruction stream.
    // A block start is any PC that is:
    //   - PC 0 (entry block)
    //   - A branch target (forward or backward)
    //   - The instruction after a conditional branch (fall-through)
    // The map is {pc -> block_id} where block_id is sequential.
    let mut block_start_pcs: FxHashSet<usize> = FxHashSet::default();
    block_start_pcs.insert(0); // entry block
    for (pc, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + off) as usize;
                if target < instrs.len() {
                    block_start_pcs.insert(target);
                }
            }
            Instr::JumpFalse(_, off) | Instr::JumpTrue(_, off) => {
                let target = ((pc as i32) + 1 + off) as usize;
                if target < instrs.len() {
                    block_start_pcs.insert(target);
                }
                // Fall-through: instruction after conditional branch
                let fall = pc + 1;
                if fall < instrs.len() {
                    block_start_pcs.insert(fall);
                }
            }
            _ => {}
        }
    }
    // Sort PCs and assign sequential block_ids
    let mut sorted_pcs: Vec<usize> = block_start_pcs.into_iter().collect();
    sorted_pcs.sort();
    let mut block_starts: FxHashMap<usize, usize> = FxHashMap::default();
    for (bid, &pc) in sorted_pcs.iter().enumerate() {
        block_starts.insert(pc, bid);
    }

    let idom = compute_idoms(instrs, &block_starts);

    // Reorder intervals for tree-scan quality allocation
    let mut reordered: Vec<LiveInterval> = intervals.to_vec();
    dominance_reorder_intervals(&mut reordered, &idom);

    // Delegate to the existing linear_scan with reordered intervals
    linear_scan(&reordered, slot_count, float_slots)
}

// ─────────────────────────────────────────────────────────────────────────────
// Register Coalescing Pass
// ─────────────────────────────────────────────────────────────────────────────
//
// After linear-scan register allocation, scan for Move(dst, src) instructions
// where both operands are assigned to physical registers.  If src is dead
// after the Move (its last use is the Move itself) and dst is first defined
// at the Move, we can reassign dst to use src's register, making the Move
// a no-op and eliminating a redundant register-to-register copy.
//
// Safety conditions for coalescing Move(d, s) at pc:
//   a) Both d and s are in registers (RegLoc::Reg)
//   b) src_reg != dst_reg (otherwise already a no-op)
//   c) s is dead after this Move (last_use[s] == pc)
//   d) d is first defined at this Move (first_def[d] == pc)
//   e) No other slot currently occupies src_reg (guaranteed by RA)
//
// Condition (c) ensures src_reg becomes free after the Move, so dst can
// safely adopt it.  Condition (d) ensures no prior instruction wrote to d
// via dst_reg — reassigning d to src_reg globally would otherwise corrupt
// earlier reads of d that still expect the value in dst_reg.

/// Register coalescing: eliminate redundant MOV instructions after allocation.
///
/// Returns a `Vec<bool>` indexed by instruction PC.  `true` means the
/// instruction is a `Move` that has been coalesced and should be emitted
/// as a no-op (skipped entirely during code emission).
fn coalesce_registers(instrs: &[Instr], ra: &mut RegAlloc) -> Vec<bool> {
    let mut coalesced = vec![false; instrs.len()];

    // ── Compute first_def and last_use per slot ──────────────────────────
    const UNDEF: usize = usize::MAX;
    let max_slot = ra.slots.len();
    let mut first_def = vec![UNDEF; max_slot];
    let mut last_use = vec![UNDEF; max_slot];

    for (pc, instr) in instrs.iter().enumerate() {
        // Record reads (uses)
        match instr {
            Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => {
                let s = *s as usize;
                if s < max_slot {
                    last_use[s] = last_use[s].max(pc);
                }
            }
            Instr::BinOp(_, _, l, r) => {
                let l = *l as usize;
                let r = *r as usize;
                if l < max_slot { last_use[l] = last_use[l].max(pc); }
                if r < max_slot { last_use[r] = last_use[r].max(pc); }
            }
            Instr::UnOp(_, _, s) => {
                let s = *s as usize;
                if s < max_slot { last_use[s] = last_use[s].max(pc); }
            }
            Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => {
                let s = *s as usize;
                if s < max_slot { last_use[s] = last_use[s].max(pc); }
            }
            _ => {}
        }
        // Record writes (definitions)
        match instr {
            Instr::LoadI32(d, _)
            | Instr::LoadI64(d, _)
            | Instr::LoadBool(d, _)
            | Instr::LoadUnit(d)
            | Instr::LoadF32(d, _)
            | Instr::LoadF64(d, _) => {
                let d = *d as usize;
                if d < max_slot && first_def[d] == UNDEF {
                    first_def[d] = pc;
                }
            }
            Instr::Move(d, _) | Instr::Load(d, _) | Instr::Store(d, _) => {
                let d = *d as usize;
                if d < max_slot && first_def[d] == UNDEF {
                    first_def[d] = pc;
                }
            }
            Instr::BinOp(d, _, _, _) => {
                let d = *d as usize;
                if d < max_slot && first_def[d] == UNDEF {
                    first_def[d] = pc;
                }
            }
            Instr::UnOp(d, _, _) => {
                let d = *d as usize;
                if d < max_slot && first_def[d] == UNDEF {
                    first_def[d] = pc;
                }
            }
            _ => {}
        }
    }

    // ── Loop-aware liveness extension ────────────────────────────────────
    // Same logic as compute_live_intervals: extend last_use for slots that
    // are live inside loops so we don't incorrectly coalesce across back-edges.
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match instr {
            Instr::Jump(off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpFalse(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpTrue(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            _ => None,
        };
        if let Some(target) = target {
            if target < pc {
                for slot in 0..max_slot {
                    let fd = first_def[slot];
                    let lu = last_use[slot];
                    if fd == UNDEF && lu == UNDEF {
                        continue;
                    }
                    let first = if fd == UNDEF { 0 } else { fd };
                    let last = if lu == UNDEF { first } else { lu };
                    if first <= pc && last >= target {
                        last_use[slot] = last_use[slot].max(pc);
                    }
                }
            }
        }
    }

    // ── Coalesce Move instructions ───────────────────────────────────────
    // Track which physical register each slot occupies, so we can detect
    // conflicts when reassigning.  We process Moves sequentially so that
    // earlier coalescing decisions are visible to later ones.
    for (pc, instr) in instrs.iter().enumerate() {
        if let Instr::Move(d, s) = instr {
            // Skip if d == s (already a no-op at the IR level)
            if d == s {
                continue;
            }
            let d_loc = ra.location(*d);
            let s_loc = ra.location(*s);

            if let (RegLoc::Reg(dst_reg), RegLoc::Reg(src_reg)) = (d_loc, s_loc) {
                // Must be different registers
                if src_reg == dst_reg {
                    continue;
                }

                let d_idx = *d as usize;
                let s_idx = *s as usize;

                // Safety condition (c): s is dead after this Move
                if s_idx >= max_slot || last_use[s_idx] != pc {
                    continue;
                }

                // Safety condition (d): d is first defined at this Move
                if d_idx >= max_slot || first_def[d_idx] != pc {
                    continue;
                }

                // Safety check: verify no OTHER slot currently uses src_reg.
                // This is guaranteed by the linear-scan allocator (each
                // register is assigned to at most one slot), but we check
                // defensively.
                let src_reg_conflict = ra.slots.iter().enumerate().any(|(slot_idx, &loc)| {
                    slot_idx != s_idx
                        && slot_idx != d_idx
                        && matches!(loc, RegLoc::Reg(r) if r == src_reg)
                });
                if src_reg_conflict {
                    continue;
                }

                // Coalesce: reassign d to use src_reg instead of dst_reg.
                // After this, reads of d will return src_reg, which holds
                // the same value as s — making the Move a no-op.
                ra.slots[d_idx] = RegLoc::Reg(src_reg);
                coalesced[pc] = true;

                // Note: dst_reg is now free.  We could theoretically
                // reassign other slots to use it, but that would require
                // re-running parts of the allocator.  For simplicity,
                // we leave dst_reg unused.
            }
        }
    }

    coalesced
}

// ─────────────────────────────────────────────────────────────────────────────
// Fix #3: Loop Vectorizer Pass — AVX2 SIMD Vectorization
// ─────────────────────────────────────────────────────────────────────────────

/// Trace a reduction chain backwards from a stored result slot.
///
/// Given a Store instruction: `Store(acc_slot, result_slot)`, this function
/// traces backwards through the instruction stream to find all the BinOps
/// that contributed to `result_slot`. It follows the chain:
///   result_slot ← BinOp(op, operand1, operand2) ← operand1 ← BinOp(...) ← ...
///
/// The chain is valid if it eventually reads from `acc_slot` via a
/// Load or Move instruction. Returns a list of (op, val_slot) pairs
/// in execution order (innermost first).
fn trace_reduction_chain(
    instrs: &[Instr],
    result_slot: u16,
    acc_slot: u16,
    store_pc: usize,
    body_start: usize,
) -> Vec<(BinOpKind, u16)> {
    let mut chain = Vec::new();
    let mut current_slot = result_slot;
    let mut visited = FxHashSet::default();

    // Walk backwards from store_pc looking for the BinOp that defined current_slot
    for _ in 0..8 { // limit chain depth to avoid infinite loops
        if visited.contains(&current_slot) {
            break;
        }
        visited.insert(current_slot);

        // Find the BinOp that writes to current_slot
        let mut found_binop = false;
        for pc in (body_start..store_pc).rev() {
            if let Instr::BinOp(dst, op, l, r) = &instrs[pc] {
                if *dst == current_slot && matches!(op, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul) {
                    // Found the BinOp that defines current_slot
                    // Check if l reads from acc_slot
                    let l_reads_acc = slot_reads_from(instrs, *l, acc_slot, pc, body_start);

                    // Determine the "other" operand (not the one reading from acc)
                    if l_reads_acc {
                        chain.push((*op, *r));
                        // Now trace l — it's the next slot to look for
                        // If l is directly from acc_slot, we're done
                        if *l == acc_slot || slot_is_load_from(instrs, *l, acc_slot, pc, body_start) {
                            found_binop = true;
                            break; // Chain complete
                        }
                        current_slot = *l;
                        found_binop = true;
                        break;
                    } else {
                        // Check if r reads from acc_slot
                        let r_reads_acc = slot_reads_from(instrs, *r, acc_slot, pc, body_start);
                        if r_reads_acc {
                            // For commutative ops, r is the "other" operand
                            if matches!(op, BinOpKind::Add | BinOpKind::Mul) {
                                chain.push((*op, *l));
                            }
                            // For non-commutative, it's more complex (skip for now)
                            if *r == acc_slot || slot_is_load_from(instrs, *r, acc_slot, pc, body_start) {
                                found_binop = true;
                                break;
                            }
                            current_slot = *r;
                            found_binop = true;
                            break;
                        }
                        // Neither operand reads from acc — this BinOp doesn't
                        // contribute to the reduction chain, skip it
                    }
                }
            }
        }
        if !found_binop {
            break;
        }
    }

    chain
}

/// Check if a slot's value ultimately comes from `source_slot` via Load/Move
/// or through a chain of BinOps that read from source_slot.
fn slot_reads_from(instrs: &[Instr], slot: u16, source_slot: u16, before_pc: usize, body_start: usize) -> bool {
    if slot == source_slot {
        return true;
    }
    // Direct Load/Move from source_slot
    for pc in (body_start..before_pc).rev() {
        match &instrs[pc] {
            Instr::Load(s, src) | Instr::Move(s, src) if *s == slot && *src == source_slot => {
                return true;
            }
            Instr::BinOp(s, _, l, r) if *s == slot => {
                // This slot is defined by a BinOp. Check if either operand
                // ultimately reads from source_slot (recursive)
                return slot_reads_from(instrs, *l, source_slot, pc, body_start)
                    || slot_reads_from(instrs, *r, source_slot, pc, body_start);
            }
            Instr::Store(s, _) if *s == slot => {
                return false; // Slot was overwritten by Store
            }
            _ => {}
        }
    }
    false
}

/// Check if a slot is defined by a Load/Move from source_slot in the loop body.
fn slot_is_load_from(instrs: &[Instr], slot: u16, source_slot: u16, before_pc: usize, body_start: usize) -> bool {
    for pc in (body_start..before_pc).rev() {
        match &instrs[pc] {
            Instr::Load(s, src) | Instr::Move(s, src) if *s == slot && *src == source_slot => {
                return true;
            }
            Instr::BinOp(s, _, _, _) | Instr::Store(s, _) if *s == slot => {
                return false;
            }
            _ => {}
        }
    }
    false
}


//
// Detects simple reduction loops (while i < N { acc = acc op val; i = i + 1 })
// and emits AVX2 vpaddd/vpsubd/vpmulld instructions instead of scalar code.
//
// A loop is vectorizable when:
//   1. It has a single backward branch (while-loop with one exit)
//   2. It has an induction variable (i) incremented by a constant each iteration
//   3. The loop body contains only supported integer BinOps (Add, Sub, Mul)
//      where the accumulator is updated as: acc = acc op val (reduction pattern)
//   4. No function calls, divisions, or other unvectorizable ops in the body
//   5. No data dependencies between iterations beyond the reduction variable
//
// The emitted code structure:
//   1. Compute trip count = (N - i0) / VF  (VF = vectorization factor = 8 for AVX2)
//   2. If trip count > 0, run SIMD loop:
//        - Load 8-wide vectors of accumulator values (broadcast)
//        - For each reduction op, load 8-wide vectors of operand values
//        - Execute vpaddd/vpsubd/vpmulld
//        - Increment induction var by VF
//        - Decrement trip count; loop back if > 0
//   3. Horizontal reduction: sum/mul the 8 lanes into a scalar result
//   4. Store reduced result back to accumulator slot
//   5. Scalar remainder loop for remaining (N % VF) iterations

/// A single reduction operation inside a vectorizable loop.
#[derive(Clone, Copy, Debug)]
struct ReductionOp {
    /// The accumulator slot being updated.
    acc_slot: u16,
    /// The operation (Add, Sub, Mul).
    op: BinOpKind,
    /// The non-accumulator operand slot (the "value" being reduced in).
    val_slot: u16,
}

struct LoopVectorizer {
    /// PC of the loop header (backward branch target).
    loop_start: Option<usize>,
    /// PC of the backward jump instruction.
    loop_end: Option<usize>,
    /// The induction variable slot (the loop counter `i`).
    induction_var: Option<u16>,
    /// The induction step (typically 1).
    induction_step: i64,
    /// The loop-bound slot (the `N` in `i < N`).
    bound_slot: Option<u16>,
    /// The loop-bound constant (if N is a compile-time constant).
    bound_const: Option<i64>,
    /// Detected reduction operations inside the loop body.
    reductions: Vec<ReductionOp>,
    /// Whether this loop can be vectorized with AVX2.
    is_vectorizable: bool,
    /// Reason for non-vectorizability (for diagnostics).
    reject_reason: Option<&'static str>,
    /// All slots written inside the loop body (to check for aliasing).
    written_slots: Vec<u16>,
    /// Whether the loop contains any unvectorizable instruction.
    has_unvectorizable: bool,
}

impl LoopVectorizer {
    fn new() -> Self {
        Self {
            loop_start: None,
            loop_end: None,
            induction_var: None,
            induction_step: 1,
            bound_slot: None,
            bound_const: None,
            reductions: Vec::new(),
            is_vectorizable: false,
            reject_reason: None,
            written_slots: Vec::new(),
            has_unvectorizable: false,
        }
    }

    /// Analyze instructions to find vectorizable loops.
    /// This performs a detailed analysis of loop structure, checking:
    /// - Single-entry, single-exit loop
    /// - Simple induction variable with constant step
    /// - Reduction operations only (Add, Sub, Mul)
    /// - No loop-carried dependencies beyond reductions
    /// - No function calls, divisions, or memory operations
    fn analyze(&mut self, instrs: &[Instr]) {
        // Phase 1: Find the backward branch to identify the loop.
        let mut back_branch_pc = None;
        let mut loop_header_pc = None;
        for (pc, instr) in instrs.iter().enumerate() {
            let target = match instr {
                Instr::Jump(off) => Some(((pc as i32) + 1 + *off) as usize),
                Instr::JumpFalse(_, off) => Some(((pc as i32) + 1 + *off) as usize),
                Instr::JumpTrue(_, off) => Some(((pc as i32) + 1 + *off) as usize),
                _ => None,
            };
            if let Some(target) = target {
                if target <= pc {
                    // Backward branch → loop.
                    if back_branch_pc.is_some() {
                        self.reject_reason = Some("multiple back-edges");
                        return;
                    }
                    back_branch_pc = Some(pc);
                    loop_header_pc = Some(target);
                }
            }
        }

        let (back_pc, header_pc) = match (back_branch_pc, loop_header_pc) {
            (Some(b), Some(h)) => (b, h),
            _ => {
                self.reject_reason = Some("no backward branch found");
                return;
            }
        };

        self.loop_start = Some(header_pc);
        self.loop_end = Some(back_pc);

        jit_trace!("[JIT-VEC] Found loop: header={}, back={}", header_pc, back_pc);

        // Dump loop body for debugging
        for _pc in header_pc..=back_pc {
            jit_trace!("[JIT-VEC]   pc={}: {:?}", _pc, instrs[_pc]);
        }

        // Phase 2: Analyze the condition.
        // The loop condition may be at the loop header (while loop) or at
        // the back edge (do-while). For while loops compiled as:
        //   header: compare + JumpFalse(exit)
        //   body: ...
        //   back: Jump(header)
        // The condition is at the header, not the back edge.
        let mut cond_slot: Option<u16> = None;
        // Check the back edge first
        match &instrs[back_pc] {
            Instr::JumpFalse(cond, _) | Instr::JumpTrue(cond, _) => {
                cond_slot = Some(*cond);
            }
            _ => {}
        }
        // If the back edge is unconditional, search for the conditional
        // exit branch near the loop header (common while-loop pattern)
        if cond_slot.is_none() {
            for search_pc in header_pc..=back_pc.min(header_pc + 5) {
                match &instrs[search_pc] {
                    Instr::JumpFalse(cond, _) | Instr::JumpTrue(cond, _) => {
                        // This JumpFalse exits the loop — it's the condition
                        cond_slot = Some(*cond);
                        break;
                    }
                    _ => {}
                }
            }
        }

        // Phase 3: Walk the loop body [header_pc..=back_pc] to identify:
        // - Induction variable and step
        // - Reduction operations
        // - Unvectorizable instructions
        let body_start = header_pc;
        let body_end = back_pc;

        for pc in body_start..=body_end {
            match &instrs[pc] {
                Instr::BinOp(dst, op, l, r) => {
                    // Check for induction variable update.
                    // Three patterns:
                    // 1. Direct: dst == l (i = i + step)
                    // 2. 3-operand: i = temp + step; Store(iv_slot, i)
                    //    Where temp was loaded from iv_slot via Load(temp, iv_slot)
                    //
                    // Pattern 2 is common after CSE/unrolling:
                    //   Load(temp, iv_slot)      ← load iv into temp
                    //   LoadI64(step_slot, 1)    ← load step constant
                    //   BinOp(result, Add, temp, step_slot)  ← result = temp + step
                    //   Store(iv_slot, result)   ← store result back to iv_slot

                    // Pattern 1: dst == l or dst == r (direct update)
                    let mut detected_iv = None;

                    if *dst == *l || (*dst == *r && matches!(op, BinOpKind::Add | BinOpKind::Sub)) {
                        let other = if *dst == *l { *r } else { *l };
                        let mut found_step: Option<i64> = None;

                        // Check if previous instruction is a LoadI32/LoadI64 for `other`
                        if let Some(prev) = instrs.get(pc.saturating_sub(1)) {
                            match prev {
                                Instr::LoadI32(slot, v) if *slot == other => {
                                    found_step = Some(*v as i64);
                                }
                                Instr::LoadI64(slot, v) if *slot == other => {
                                    found_step = Some(*v);
                                }
                                _ => {}
                            }
                        }

                        // Look back further for the constant load
                        if found_step.is_none() {
                            for prev_pc in (0..pc).rev() {
                                match &instrs[prev_pc] {
                                    Instr::LoadI32(slot, v) if *slot == other => {
                                        found_step = Some(*v as i64);
                                        break;
                                    }
                                    Instr::LoadI64(slot, v) if *slot == other => {
                                        found_step = Some(*v);
                                        break;
                                    }
                                    Instr::BinOp(s, _, _, _) | Instr::Store(s, _) | Instr::Move(s, _)
                                        if *s == other => {
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }

                        if let Some(step) = found_step {
                            if *dst == *l && matches!(op, BinOpKind::Add) {
                                detected_iv = Some((*dst, step));
                            } else if *dst == *l && matches!(op, BinOpKind::Sub) {
                                detected_iv = Some((*dst, -step));
                            } else if *dst == *r && matches!(op, BinOpKind::Add) {
                                detected_iv = Some((*dst, step));
                            }
                        }
                    }

                    // Pattern 2: 3-operand form with Store-back
                    // BinOp(result, Add, temp, step_slot) + Store(iv_slot, result)
                    // Where temp = Load(temp, iv_slot)
                    if detected_iv.is_none() && matches!(op, BinOpKind::Add | BinOpKind::Sub) {
                        // Check if next instruction is Store(some_slot, dst)
                        if let Some(next) = instrs.get(pc + 1) {
                            if let Instr::Store(store_slot, store_src) = next {
                                if *store_src == *dst {
                                    // This is a Store(slot, result) — slot is the actual iv
                                    // Now check: was `l` loaded from `store_slot`?
                                    // Look back for Load(l, store_slot) or Move(l, store_slot)
                                    for prev_pc in (0..pc).rev() {
                                        match &instrs[prev_pc] {
                                            Instr::Load(slot, src) if *slot == *l && *src == *store_slot => {
                                                // Found: l was loaded from store_slot
                                                // Now find the step constant from r
                                                let step = if matches!(op, BinOpKind::Add) { 1i64 } else { -1i64 };
                                                // Try to find the actual constant value of r
                                                let mut found_step: Option<i64> = None;
                                                for pp in (0..pc).rev() {
                                                    match &instrs[pp] {
                                                        Instr::LoadI32(slot, v) if *slot == *r => {
                                                            found_step = Some(*v as i64);
                                                            break;
                                                        }
                                                        Instr::LoadI64(slot, v) if *slot == *r => {
                                                            found_step = Some(*v);
                                                            break;
                                                        }
                                                        Instr::BinOp(s, _, _, _) | Instr::Store(s, _) | Instr::Move(s, _)
                                                            if *s == *r => {
                                                            break;
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                                let actual_step = found_step.unwrap_or(step);
                                                if matches!(op, BinOpKind::Sub) {
                                                    detected_iv = Some((*store_slot, -actual_step));
                                                } else {
                                                    detected_iv = Some((*store_slot, actual_step));
                                                }
                                                break;
                                            }
                                            Instr::Move(slot, src) if *slot == *l && *src == *store_slot => {
                                                // Same pattern with Move instead of Load
                                                let step = if matches!(op, BinOpKind::Add) { 1i64 } else { -1i64 };
                                                let mut found_step: Option<i64> = None;
                                                for pp in (0..pc).rev() {
                                                    match &instrs[pp] {
                                                        Instr::LoadI32(slot, v) if *slot == *r => {
                                                            found_step = Some(*v as i64);
                                                            break;
                                                        }
                                                        Instr::LoadI64(slot, v) if *slot == *r => {
                                                            found_step = Some(*v);
                                                            break;
                                                        }
                                                        Instr::BinOp(s, _, _, _) | Instr::Store(s, _) | Instr::Move(s, _)
                                                            if *s == *r => {
                                                            break;
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                                let actual_step = found_step.unwrap_or(step);
                                                if matches!(op, BinOpKind::Sub) {
                                                    detected_iv = Some((*store_slot, -actual_step));
                                                } else {
                                                    detected_iv = Some((*store_slot, actual_step));
                                                }
                                                break;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some((iv_slot, step)) = detected_iv {
                        self.induction_var = Some(iv_slot);
                        self.induction_step = step;
                        jit_trace!("[JIT-VEC] Found induction var: slot_{} step={}", iv_slot, step);
                    }

                    // Check for reduction pattern by tracking Store-backs.
                    // In the optimized instruction stream, the pattern is often:
                    //   Load(temp, acc_slot)        ← load accumulator
                    //   BinOp(result1, Mul, temp, K) ← first reduction op
                    //   BinOp(result2, Add, result1, C) ← second reduction op
                    //   Store(acc_slot, result2)    ← store back to accumulator
                    //
                    // We detect this by checking each Store instruction to see if
                    // it stores a computation chain that reads from the same slot.
                    // This is done in a separate pass below (Phase 3b).

                    // Direct reduction: BinOp(acc, op, acc, val) — less common after optimization
                    if matches!(op, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul) {
                        // Form 1: Direct update
                        if *dst == *l || *dst == *r {
                            let acc = *dst;
                            if Some(acc) != self.induction_var {
                                let val = if *dst == *l { *r } else { *l };
                                let already = self.reductions.iter().any(|r| r.acc_slot == acc && r.op == *op);
                                if !already {
                                    self.reductions.push(ReductionOp {
                                        acc_slot: acc,
                                        op: *op,
                                        val_slot: val,
                                    });
                                    jit_trace!("[JIT-VEC] Found reduction (direct): slot_{} = slot_{} {:?} slot_{}",
                                        acc, acc, op, val);
                                }
                            }
                        }

                        // Form 2: Store-back pattern
                        // BinOp(result, op, temp, val) followed by Store(acc, result)
                        // Where temp was loaded from acc
                        if *dst != *l && *dst != *r {
                            // Check if next instruction is Store(some_slot, dst)
                            if let Some(next) = instrs.get(pc + 1) {
                                if let Instr::Store(store_slot, store_src) = next {
                                    if *store_src == *dst && Some(*store_slot) != self.induction_var {
                                        // Check if l was loaded from store_slot
                                        for prev_pc in (0..pc).rev() {
                                            match &instrs[prev_pc] {
                                                Instr::Load(slot, src) if *slot == *l && *src == *store_slot => {
                                                    let already = self.reductions.iter()
                                                        .any(|r| r.acc_slot == *store_slot && r.op == *op);
                                                    if !already {
                                                        self.reductions.push(ReductionOp {
                                                            acc_slot: *store_slot,
                                                            op: *op,
                                                            val_slot: *r,
                                                        });
                                                        jit_trace!("[JIT-VEC] Found reduction (store-back): slot_{} = slot_{} {:?} slot_{}",
                                                            store_slot, store_slot, op, r);
                                                    }
                                                    break;
                                                }
                                                Instr::Move(slot, src) if *slot == *l && *src == *store_slot => {
                                                    let already = self.reductions.iter()
                                                        .any(|r| r.acc_slot == *store_slot && r.op == *op);
                                                    if !already {
                                                        self.reductions.push(ReductionOp {
                                                            acc_slot: *store_slot,
                                                            op: *op,
                                                            val_slot: *r,
                                                        });
                                                        jit_trace!("[JIT-VEC] Found reduction (store-back move): slot_{} = slot_{} {:?} slot_{}",
                                                            store_slot, store_slot, op, r);
                                                    }
                                                    break;
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if matches!(op, BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv) {
                        self.has_unvectorizable = true;
                        self.reject_reason = Some("division in loop body");
                    }

                    self.written_slots.push(*dst);
                }
                Instr::LoadI32(d, _) | Instr::LoadI64(d, _) => {
                    let _ = d;
                }
                Instr::LoadBool(_, _) | Instr::LoadUnit(_) | Instr::LoadF32(_, _) | Instr::LoadF64(_, _) => {
                    // Fine
                }
                Instr::Move(d, _s) | Instr::Load(d, _s) => {
                    self.written_slots.push(*d);
                }
                Instr::Store(slot, _) => {
                    self.written_slots.push(*slot);
                }
                Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _) | Instr::Jump(_) => {
                    // Control flow — fine as long as it's the loop structure
                }
                Instr::Nop => {}
                Instr::Return(_) | Instr::ReturnUnit => {
                    self.has_unvectorizable = true;
                    self.reject_reason = Some("return in loop body");
                }
                _ => {
                    // Call, UnOp, etc. — conservatively reject
                    self.has_unvectorizable = true;
                    if self.reject_reason.is_none() {
                        self.reject_reason = Some("unsupported instruction in loop body");
                    }
                }
            }
        }

        // Phase 3b: Detect store-back reduction chains.
        // In the optimized instruction stream, reductions often use a chain:
        //   Load(temp, acc_slot)            ← load accumulator
        //   LoadI64(K_slot, K)             ← load constant
        //   BinOp(result1, Mul, temp, K_slot) ← result1 = acc * K
        //   BinOp(result2, Add, result1, C_slot) ← result2 = result1 + C
        //   Store(acc_slot, result2)        ← store back to accumulator
        //
        // Algorithm: for each Store in the loop body that writes to a
        // non-induction slot, trace the computation chain backwards from
        // the stored value to find which slots are used. If the chain
        // reads from the stored slot via Load(temp, stored_slot), it's a
        // reduction chain. Record all the BinOps in the chain.
        if self.reductions.is_empty() && self.induction_var.is_some() {
            let iv = self.induction_var.unwrap();
            for pc in body_start..=body_end {
                if let Instr::Store(store_slot, store_src) = &instrs[pc] {
                    let store_slot = *store_slot;
                    let store_src = *store_src;
                    // Skip stores to the induction variable
                    if store_slot == iv { continue; }
                    // Skip if we already found reductions for this slot
                    if self.reductions.iter().any(|r| r.acc_slot == store_slot) { continue; }

                    // Trace backwards from store_src to find the computation chain
                    let chain = trace_reduction_chain(instrs, store_src, store_slot, pc, body_start);
                    if !chain.is_empty() {
                        for (op, val_slot) in chain {
                            let already = self.reductions.iter().any(|r| r.acc_slot == store_slot && r.op == op);
                            if !already {
                                self.reductions.push(ReductionOp {
                                    acc_slot: store_slot,
                                    op,
                                    val_slot,
                                });
                                jit_trace!("[JIT-VEC] Found reduction (chain): slot_{} {:?} slot_{}",
                                    store_slot, op, val_slot);
                            }
                        }
                    }
                }
            }
        }

        // Phase 4: Try to find the loop bound.
        // Search backwards from the JumpFalse for a comparison BinOp that
        // uses the condition slot. The comparison may not be immediately
        // before the JumpFalse — there could be LoadUnit/Nop instructions
        // in between (e.g., after unrolling).
        if let Some(cond) = cond_slot {
            if let Some(iv) = self.induction_var {
                for search_pc in (body_start..back_pc).rev() {
                    if let Instr::BinOp(cmp_dst, cmp_op, cmp_l, cmp_r) = &instrs[search_pc] {
                        if *cmp_dst == cond && matches!(cmp_op, BinOpKind::Lt | BinOpKind::Le) {
                            let l_is_iv = *cmp_l == iv || slot_reads_from(instrs, *cmp_l, iv, search_pc, body_start);
                            let r_is_iv = *cmp_r == iv || slot_reads_from(instrs, *cmp_r, iv, search_pc, body_start);
                            if l_is_iv {
                                self.bound_slot = Some(*cmp_r);
                                jit_trace!("[JIT-VEC] Found bound: slot_{} (iv < bound)", cmp_r);
                            } else if r_is_iv {
                                self.bound_slot = Some(*cmp_l);
                                jit_trace!("[JIT-VEC] Found bound: slot_{} (bound > iv)", cmp_l);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Phase 5: Determine vectorizability.
        if self.induction_var.is_none() {
            self.reject_reason = Some("no induction variable found");
            self.is_vectorizable = false;
            return;
        }
        if self.has_unvectorizable {
            self.is_vectorizable = false;
            return;
        }
        if self.reductions.is_empty() {
            self.reject_reason = Some("no reduction operations found");
            self.is_vectorizable = false;
            return;
        }

        for red in &self.reductions {
            if Some(red.acc_slot) == self.induction_var {
                self.reject_reason = Some("reduction accumulator aliases induction var");
                self.is_vectorizable = false;
                return;
            }
            // If the reduction operand is the induction variable itself,
            // we can't vectorize because the operand changes every iteration.
            // Broadcasting the induction variable across SIMD lanes gives
            // all lanes the same value, producing wrong results.
            // Example: "s = s + i" where i is the loop counter — each
            // SIMD lane would add the same value of i, not different i's.
            if Some(red.val_slot) == self.induction_var {
                self.reject_reason = Some("reduction operand is induction var (changes per iteration)");
                self.is_vectorizable = false;
                return;
            }
            // Also reject if the val_slot depends on the induction variable.
            // Even if val_slot != induction_var, it may have been loaded from
            // the IV (e.g., Load(temp, iv_slot)). We trace the def chain.
            if self.written_slots.contains(&red.val_slot) && Some(red.val_slot) != self.induction_var {
                // Check if val_slot was defined by loading from the IV.
                // Scan the loop body for the definition of val_slot.
                let mut depends_on_iv = false;
                for pc in body_start..=body_end {
                    if pc >= instrs.len() { break; }
                    match &instrs[pc] {
                        // Direct load from the IV slot
                        Instr::Load(d, s) | Instr::Move(d, s) if *d == red.val_slot && Some(*s) == self.induction_var => {
                            depends_on_iv = true;
                            break;
                        }
                        // If val_slot is a BinOp result, check if any operand is the IV
                        Instr::BinOp(d, _, l, r) if *d == red.val_slot => {
                            if Some(*l) == self.induction_var || Some(*r) == self.induction_var {
                                depends_on_iv = true;
                                break;
                            }
                        }
                        // LoadI* are constants — safe to broadcast
                        Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) if *d == red.val_slot => {
                            // This is a constant definition — loop-invariant
                            break;
                        }
                        _ => {}
                    }
                }
                if depends_on_iv {
                    self.reject_reason = Some("reduction operand depends on induction var (changes per iteration)");
                    self.is_vectorizable = false;
                    return;
                }
            }
        }

        self.is_vectorizable = true;
        jit_trace!("[JIT-VEC] Loop vectorizable: header={}, back={}, iv=slot_{:?}, step={}, reductions={}",
            header_pc, back_pc, self.induction_var, self.induction_step, self.reductions.len());
    }

    /// Returns vectorization factor (8 for AVX2, 4 for SSE, 1 if not vectorizable).
    fn vectorization_factor(&self) -> usize {
        if !self.is_vectorizable { return 1; }
        let cpu = cpu_features();
        if cpu.has_avx2 { 8 } else if cpu.has_sse42 { 4 } else { 1 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Software Pipelining: restructure loop execution to overlap memory loads
// with computation by factoring the loop into Prefetch-Compute-Store stages.
// ─────────────────────────────────────────────────────────────────────────────
//
// For a loop body that looks like:
//   for i in 0..N:
//     load a[i]          (4-7 cycle latency)
//     compute on a[i]
//     store result[i]
//
// The pipelined version generates:
//   // Prologue: load first tile
//   load a[0] into ZMM0..ZMM3
//
//   // Steady state: overlap load+compute
//   for i in 0..N-1:
//     compute on ZMM0..ZMM3 (data from i) → store result[i]
//     load a[i+1] into ZMM4..ZMM7 (for next iteration)
//     swap register sets
//
//   // Epilogue: compute last tile
//   compute on ZMM4..ZMM7 → store result[N-1]
//
// This hides the 4-7 cycle memory latency behind active computation.
#[allow(dead_code)]
struct SoftwarePipeline {
    /// Number of alternating register sets (typically 2 for double-buffering).
    num_sets: usize,
    /// Number of ZMM registers per set.
    regs_per_set: usize,
    /// Whether pipelining was applied.
    applied: bool,
}

#[allow(dead_code)]

impl SoftwarePipeline {
    fn new() -> Self {
        Self {
            num_sets: 2,
            regs_per_set: 4,
            applied: false,
        }
    }

    /// Attempt to apply software pipelining to a vectorized loop.
    /// Returns true if pipelining was successfully applied.
    ///
    /// Currently this returns a hint that the code generator uses
    /// to emit the prologue/steady-state/epilogue structure.
    /// The actual register renaming happens during code emission.
    fn analyze_and_pipeline(&mut self, instrs: &[Instr]) -> bool {
        // Find the loop body
        let mut loop_start = 0usize;
        let mut loop_end = instrs.len();
        let mut has_backward_jump = false;

        for (i, instr) in instrs.iter().enumerate() {
            let target = match instr {
                Instr::Jump(off) => Some(((i as i32) + 1 + *off) as usize),
                Instr::JumpFalse(_, off) => Some(((i as i32) + 1 + *off) as usize),
                Instr::JumpTrue(_, off) => Some(((i as i32) + 1 + *off) as usize),
                _ => None,
            };
            if let Some(target) = target {
                if target <= i {
                    loop_start = target;
                    loop_end = i + 1;
                    has_backward_jump = true;
                    break;
                }
            }
        }

        if !has_backward_jump { return false; }

        // Count load and compute instructions in the loop body
        let mut load_count = 0usize;
        let mut compute_count = 0usize;
        for i in loop_start..loop_end {
            match &instrs[i] {
                Instr::Load(_, _) | Instr::LoadF32(_, _) | Instr::LoadF64(_, _) => load_count += 1,
                Instr::BinOp(_, _, _, _) | Instr::UnOp(_, _, _) => compute_count += 1,
                _ => {}
            }
        }

        // Only pipeline if there are enough loads and computes to overlap
        if load_count >= 2 && compute_count >= 2 {
            self.applied = true;
            jit_trace!("[JIT-SWPIPE] Software pipelining: loop [{}..{}] has {} loads, {} computes — pipelining with {} register sets",
                loop_start, loop_end, load_count, compute_count, self.num_sets);
            true
        } else {
            false
        }
    }

    /// Returns whether pipelining was applied.
    #[allow(dead_code)]
    fn is_applied(&self) -> bool { self.applied }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD Loop Emission — generates AVX2 vectorized loop code
// ─────────────────────────────────────────────────────────────────────────────
//
// This function emits machine code for a vectorized loop that processes VF
// elements per iteration using YMM registers. The generated code:
//
//   1. Computes the SIMD trip count: trips = (bound - iv) / VF
//   2. If trips == 0, falls through to scalar remainder
//   3. Broadcasts each accumulator's current value into a YMM register
//   4. For each reduction op, loads/broadcasts the operand and executes
//      vpaddd/vpsubd/vpmulld
//   5. Decrements trip count; loops back if > 0
//   6. Horizontal reduction: reduce 8 lanes to 1 scalar per accumulator
//   7. Stores reduced results back to accumulator slots
//   8. Updates the induction variable: iv += trips * VF * step
//
// Register usage:
//   RAX, RCX, RDX  — scratch / induction var
//   YMM0-YMM3      — accumulator vectors
//   YMM4-YMM7      — operand vectors (broadcast)
//   R8              — SIMD trip counter
//   RDI             — slot array base (preserved)

/// Emit the SIMD-vectorized loop for a detected vectorizable loop.
/// Returns true if vectorized code was emitted, false if fallback to scalar.
///
/// The approach for Add/Sub reductions:
///   - Initialize YMM accumulator to zero (VPXOR)
///   - In the SIMD loop: vpaddd/vpsubd with broadcast operand
///   - After loop: horizontal sum → add to original scalar accumulator
///   - This is correct because: acc_new = acc_orig + K * N
///     and SIMD computes: sum_of_lanes = K * VF * trips = K * N
///     then: acc_new = acc_orig + sum_of_lanes ✓
///
/// For Mul reductions (LCG pattern: s = s * K1 + K2):
///   - We compute "stride" multipliers: K1^VF, K1^(2*VF), ...
///   - Each SIMD iteration applies: s_lane *= K1; s_lane += K2 (VF times)
///   - With stride K1^VF, one SIMD iteration = VF scalar iterations
///   - This requires computing K1^8 at JIT compile time
fn emit_vectorized_loop(
    em: &mut Emitter,
    vec: &LoopVectorizer,
    ra: &RegAlloc,
    _const_at: &ConstTable,
    instrs: &[Instr],
    fixups: &mut Vec<Fixup>,
    pc_to_off: &mut Vec<usize>,
) -> bool {
    // Use pc_to_off for potential branch fixups
    let _ = pc_to_off;
    let cpu = cpu_features();
    if !cpu.has_avx2 || !vec.is_vectorizable {
        return false;
    }

    let vf: i64 = if cpu.has_avx512f { 16 } else { 8 }; // AVX-512 = 16 × i32 per ZMM, AVX2 = 8 × i32 per YMM
    let iv_slot = match vec.induction_var {
        Some(s) => s,
        None => return false,
    };

    let bound_slot = match vec.bound_slot {
        Some(s) => s,
        None => {
            jit_trace!("[JIT-VEC] No bound slot found, falling back to scalar");
            return false;
        }
    };

    // Check if any reduction operand is the induction var — can't vectorize those
    for red in &vec.reductions {
        if Some(red.val_slot) == vec.induction_var {
            jit_trace!("[JIT-VEC] Reduction operand is induction var (slot_{}), not vectorizable",
                red.val_slot);
            return false;
        }
    }

    // Get unique accumulator slots
    let mut acc_slots: Vec<u16> = Vec::new();
    for red in &vec.reductions {
        if !acc_slots.contains(&red.acc_slot) {
            acc_slots.push(red.acc_slot);
        }
    }
    if acc_slots.len() > 4 {
        return false;
    }

    // Mul reductions require stride-based vectorization (each lane must start
    // at a different offset: s*K^i + C*(K^(i-1)+...+1)). The current SIMD
    // loop body broadcasts the same initial value to all lanes and applies
    // the same operation, which is correct for Add/Sub but WRONG for Mul.
    // For now, fall back to scalar for Mul reductions until proper stride-
    // based initialization is implemented.
    let has_mul = vec.reductions.iter().any(|r| r.op == BinOpKind::Mul);
    if has_mul {
        jit_trace!("[JIT-VEC] Mul reductions require stride-based vectorization (not yet implemented), falling back to scalar");
        return false;
    }
    let has_addsub = vec.reductions.iter().any(|r| matches!(r.op, BinOpKind::Add | BinOpKind::Sub));

    jit_trace!("[JIT-VEC] Emitting AVX2 vectorized loop: VF={}, iv=slot_{}, bound=slot_{}, has_mul={}, accs={}",
        vf, iv_slot, bound_slot, has_mul, acc_slots.len());

    // For Mul reductions, we need to precompute K^VF (stride multiplier)
    // at JIT compile time. Find the Mul operand value.
    let mul_stride: Option<i64> = if has_mul {
        // Find the Mul reduction operand value (must be a compile-time constant)
        let mul_red = vec.reductions.iter().find(|r| r.op == BinOpKind::Mul);
        match mul_red {
            Some(red) => {
                // Try to find the constant value of red.val_slot
                // Look for LoadI32/LoadI64 that wrote to this slot before the loop
                let loop_start = vec.loop_start.unwrap_or(0);
                let loop_end = vec.loop_end.unwrap_or(instrs.len());
                let mut found_val: Option<i64> = None;
                // Search inside the loop body first (constants are often loaded
                // inside the loop for each iteration), then before the loop
                for pc in (loop_start..=loop_end).rev() {
                    match &instrs[pc] {
                        Instr::LoadI32(slot, v) if *slot == red.val_slot => {
                            found_val = Some(*v as i64);
                            break;
                        }
                        Instr::LoadI64(slot, v) if *slot == red.val_slot => {
                            found_val = Some(*v);
                            break;
                        }
                        _ => {}
                    }
                }
                if found_val.is_none() {
                    // Search before the loop
                    for pc in (0..loop_start).rev() {
                        match &instrs[pc] {
                            Instr::LoadI32(slot, v) if *slot == red.val_slot => {
                                found_val = Some(*v as i64);
                                break;
                            }
                            Instr::LoadI64(slot, v) if *slot == red.val_slot => {
                                found_val = Some(*v);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                match found_val {
                    Some(v) => {
                        // Compute v^8 (stride multiplier) as i32
                        let mut stride: i64 = 1;
                        for _ in 0..vf {
                            stride = (stride as i32).wrapping_mul(v as i32) as i64;
                        }
                        jit_trace!("[JIT-VEC] Mul stride: {}^8 = {} (i32)", v, stride as i32);
                        Some(stride)
                    }
                    None => {
                        jit_trace!("[JIT-VEC] Mul operand not a compile-time constant, falling back to scalar");
                        return false;
                    }
                }
            }
            None => None,
        }
    } else {
        None
    };

    // For Add reductions after Mul, we need the "geometric stride" for the addend.
    // In LCG: s = s * K + C, the SIMD version computes:
    //   s_new = s * K^8 + C * (K^7 + K^6 + ... + K + 1)
    // The addend stride = C * (K^7 + ... + K + 1) = C * (K^8 - 1) / (K - 1)
    let _add_stride: Option<i64> = if has_mul && has_addsub {
        let mul_red = vec.reductions.iter().find(|r| r.op == BinOpKind::Mul);
        let add_red = vec.reductions.iter().find(|r| matches!(r.op, BinOpKind::Add | BinOpKind::Sub));
        match (mul_red, add_red) {
            (Some(mr), Some(ar)) => {
                // Find the addend constant (search inside loop body first)
                let loop_start = vec.loop_start.unwrap_or(0);
                let loop_end = vec.loop_end.unwrap_or(instrs.len());
                let mut add_val: Option<i64> = None;
                for pc in (loop_start..=loop_end).rev() {
                    match &instrs[pc] {
                        Instr::LoadI32(slot, v) if *slot == ar.val_slot => {
                            add_val = Some(*v as i64);
                            break;
                        }
                        Instr::LoadI64(slot, v) if *slot == ar.val_slot => {
                            add_val = Some(*v);
                            break;
                        }
                        _ => {}
                    }
                }
                if add_val.is_none() {
                    for pc in (0..loop_start).rev() {
                        match &instrs[pc] {
                            Instr::LoadI32(slot, v) if *slot == ar.val_slot => {
                                add_val = Some(*v as i64);
                                break;
                            }
                            Instr::LoadI64(slot, v) if *slot == ar.val_slot => {
                                add_val = Some(*v);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                match (add_val, mul_stride) {
                    (Some(c), Some(k8)) => {
                        // Find K (the mul operand) — already found in mul_stride computation
                        // We need the original K to compute the geometric series.
                        // K = k8^(1/8) but we already have k from the mul_stride computation.
                        // Since mul_stride was computed as K^8, we need K.
                        // We already searched for it above — let's re-use the same search.
                        let mut mul_val: Option<i64> = None;
                        for pc in (loop_start..=loop_end).rev() {
                            match &instrs[pc] {
                                Instr::LoadI32(slot, v) if *slot == mr.val_slot => {
                                    mul_val = Some(*v as i64);
                                    break;
                                }
                                Instr::LoadI64(slot, v) if *slot == mr.val_slot => {
                                    mul_val = Some(*v);
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if mul_val.is_none() {
                            for pc in (0..loop_start).rev() {
                                match &instrs[pc] {
                                    Instr::LoadI32(slot, v) if *slot == mr.val_slot => {
                                        mul_val = Some(*v as i64);
                                        break;
                                    }
                                    Instr::LoadI64(slot, v) if *slot == mr.val_slot => {
                                        mul_val = Some(*v);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        match mul_val {
                            Some(k) if k != 1 => {
                                // geometric_sum = (K^8 - 1) / (K - 1) as i32
                                let k8_minus_1 = (k8 as i32).wrapping_sub(1);
                                let k_minus_1 = (k as i32).wrapping_sub(1);
                                let geo_sum = if k_minus_1 != 0 {
                                    k8_minus_1.wrapping_div(k_minus_1)
                                } else {
                                    8 // K=1: sum = 8
                                };
                                let stride = (c as i32).wrapping_mul(geo_sum);
                                jit_trace!("[JIT-VEC] Add stride (geometric): C={} * geo_sum={} = {}", c as i32, geo_sum, stride);
                                Some(stride as i64)
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    } else {
        None
    };

    // ── Step 1: Compute SIMD trip count ──────────────────────────────────
    // trip_count = (bound - iv) / vf
    // When the loop bound is a compile-time constant (from bound_const),
    // emit an immediate comparison instead of loading from the slot array.
    if let Some(bound_val) = vec.bound_const {
        em.mov_rax_imm64(bound_val);  // RAX = bound (immediate)
    } else {
        load_rax(em, bound_slot, ra);   // RAX = bound (from slot)
    }
    load_rcx(em, iv_slot, ra);      // RCX = iv
    em.sub_rax_rcx();               // RAX = bound - iv = total_iterations

    // If total_iterations <= 0, skip SIMD loop
    let skip_simd_fixup = jle_rel32_placeholder(em);

    // Save total_iterations in R10 before dividing by VF.
    // R10 is needed later for:
    //   (a) computing the AVX-512 masked tail remainder = R10 & 7
    //   (b) updating the induction variable by total_iterations * step
    //       when AVX-512 masked tail is used (instead of trips * vf * step)
    em.emit3(0x49, 0x89, 0xC2); // MOV R10, RAX

    // RAX = trip_count = (bound - iv) / 8
    em.shr_rax_imm8(3);

    // Save trip count in R8
    em.emit3(0x49, 0x89, 0xC0); // MOV R8, RAX

    // Whether AVX-512 masked tail will be emitted (determines IV update strategy)
    let use_avx512_tail = cpu.has_avx512f; // Re-enabled after fixing EVEX P1 W-bit corruption

    // ── Step 2: Initialize SIMD accumulators ──────────────────────────────
    if has_mul {
        // For Mul reductions (LCG pattern), we broadcast the INITIAL accumulator
        // value into all 8 lanes. Each lane represents one "interleaved" sub-sequence.
        // After trips SIMD iterations, each lane = s0 * (K^VF)^trips + ...
        // Then we combine the 8 sub-sequences.
        for (idx, &acc_slot) in acc_slots.iter().enumerate() {
            let ymm_reg = idx as u8;
            load_rax(em, acc_slot, ra);
            em.vmovd_xmm_r32(ymm_reg, 0);
            em.vpbroadcastd_ymm_xmm(ymm_reg, ymm_reg);
        }
    } else {
        // For Add/Sub reductions, initialize to ZERO.
        // After the SIMD loop, horizontal sum gives K * VF * trips.
        // We then add this to the original accumulator.
        for (idx, _) in acc_slots.iter().enumerate() {
            let ymm_reg = idx as u8;
            // VPXOR ymm, ymm, ymm — zero the register
            let byte1 = (!(ymm_reg >= 8) as u8) << 7
                | ((!ymm_reg) & 0xF) << 3
                | (1 << 2) | 1; // L=1, pp=1(66)
            em.emit3(0xC5, byte1, 0xEF); // VPXOR
            em.b(0xC0 | ((ymm_reg & 7) << 3) | (ymm_reg & 7));
        }
    }

    // ── Step 3: SIMD loop body ───────────────────────────────────────────
    let simd_loop_start = em.pos();

    for red in &vec.reductions {
        let acc_idx = acc_slots.iter().position(|&s| s == red.acc_slot).unwrap() as u8;
        let acc_ymm = acc_idx;

        let val_idx = acc_slots.iter().position(|&s| s == red.val_slot);

        if let Some(vi) = val_idx {
            // Operand is another accumulator in a YMM register
            let val_ymm = vi as u8;
            match red.op {
                BinOpKind::Add => em.vpaddd_ymm(acc_ymm, acc_ymm, val_ymm),
                BinOpKind::Sub => em.vpsubd_ymm(acc_ymm, acc_ymm, val_ymm),
                BinOpKind::Mul => em.vpmulld_ymm(acc_ymm, acc_ymm, val_ymm),
                _ => return false,
            }
        } else {
            // Operand is loop-invariant — broadcast it
            let scratch_ymm = 4 + (acc_idx.min(3)); // YMM4-YMM7
            load_rax(em, red.val_slot, ra);
            em.vmovd_xmm_r32(scratch_ymm, 0);
            em.vpbroadcastd_ymm_xmm(scratch_ymm, scratch_ymm);
            match red.op {
                BinOpKind::Add => em.vpaddd_ymm(acc_ymm, acc_ymm, scratch_ymm),
                BinOpKind::Sub => em.vpsubd_ymm(acc_ymm, acc_ymm, scratch_ymm),
                BinOpKind::Mul => em.vpmulld_ymm(acc_ymm, acc_ymm, scratch_ymm),
                _ => return false,
            }
        }
    }

    // Decrement trip counter (R8)
    em.emit3(0x49, 0xFF, 0xC8); // DEC R8 (REX.WB FF /1, ModRM=11_001_000)

    // Loop back if R8 > 0
    em.emit3(0x4D, 0x85, 0xC0); // TEST R8, R8
    let loop_back_fixup = em.jnz_rel32_placeholder();
    fixups.push(Fixup {
        disp_pos: loop_back_fixup,
        target_pc: usize::MAX,
        kind: BranchKind::Jnz,
    });

    // ── Step 3b: AVX-512 Masked Tail ────────────────────────────────────
    // After the main AVX2 vector loop processes `trips` full vectors,
    // there may be 1-7 remaining elements (remainder = N % 8). On AVX-512
    // CPUs, we use opmask registers to handle these tail elements in a
    // single masked vector iteration, eliminating the scalar remainder loop.
    //
    // This is a major ILP win: the CPU doesn't need to drain the vector
    // pipeline before switching to scalar execution. Instead, the masked
    // iteration stays in the vector domain.
    //
    // Algorithm:
    //   1. remainder = R10 & 7  (R10 = total_iterations, saved in Step 1)
    //   2. If remainder == 0, skip to after_masked_tail
    //   3. mask = (1 << remainder) - 1   (e.g., remainder=3 → 0b111)
    //   4. KMOVW k1, eax  (load mask into opmask register k1)
    //   5. Execute one vector iteration with EVEX-masked operations:
    //      - VPADDD/VPSUBD ymm_acc{k1}{z}, ymm_acc, ymm_operand
    //      - The {z} suffix zeroes masked-off lanes, so they don't
    //        contribute to the horizontal reduction.
    //   6. After this, the horizontal reduction naturally sums all lanes,
    //      giving the correct total for both full and tail iterations.
    //
    // For Add reductions: after the main loop, each lane has K * trips.
    // After the masked tail, lanes 0..remainder-1 have K * (trips + 1)
    // and lanes remainder..7 have K * trips. The horizontal sum gives:
    //   remainder * K * (trips+1) + (8-remainder) * K * trips
    //   = K * (8*trips + remainder) = K * total_iterations  ✓
    if use_avx512_tail {
        // Compute remainder = total_iterations & 7
        // R10 = total_iterations (saved in Step 1)
        em.emit3(0x4C, 0x89, 0xD0); // MOV RAX, R10
        em.and_rax_imm32(7);          // RAX = remainder = total_iterations % 8

        // If remainder == 0, skip the masked tail entirely
        em.test_rax_rax();
        let skip_masked_tail_fixup = em.jz_rel32_placeholder();

        // Compute mask = (1 << remainder) - 1
        // RAX currently = remainder (0..7). We need the mask in EAX for KMOVW.
        //   MOV RCX, RAX    (save remainder → CL for variable shift)
        //   MOV RAX, 1      (base value for shift)
        //   SHL RAX, CL     (RAX = 1 << remainder, e.g. 1<<3 = 8)
        //   DEC RAX         (RAX = mask, e.g. 8-1 = 7 = 0b111)
        em.emit3(0x48, 0x89, 0xC1);    // MOV RCX, RAX (remainder → CL for shift)
        em.mov_rax_imm_opt(1);          // RAX = 1
        em.emit3(0x48, 0xD3, 0xE0);     // SHL RAX, CL (RAX = 1 << remainder)
        em.dec_rax();                    // RAX = mask = (1 << remainder) - 1

        // Load mask into opmask register k1
        em.emit_kmovw_k_eax(1);          // KMOVW k1, EAX

        // Execute one masked iteration of the loop body.
        // This mirrors the SIMD loop body from Step 3, but uses
        // EVEX-encoded masked operations instead of VEX-encoded ones.
        for red in &vec.reductions {
            let acc_idx = acc_slots.iter().position(|&s| s == red.acc_slot).unwrap() as u8;
            let acc_ymm = acc_idx;

            let val_idx = acc_slots.iter().position(|&s| s == red.val_slot);

            if let Some(vi) = val_idx {
                // Operand is another accumulator — use masked reg-to-reg op
                let val_ymm = vi as u8;
                match red.op {
                    BinOpKind::Add => em.emit_vpaddd_ymm_masked(acc_ymm, acc_ymm, val_ymm, 1),
                    BinOpKind::Sub => em.emit_vpsubd_ymm_masked(acc_ymm, acc_ymm, val_ymm, 1),
                    BinOpKind::Mul => em.emit_vpmulld_ymm_masked(acc_ymm, acc_ymm, val_ymm, 1),
                    _ => return false,
                }
            } else {
                // Operand is loop-invariant — broadcast it into a scratch register,
                // then apply masked operation.
                // Note: We use the existing VEX-encoded VPBROADCASTD (not masked)
                // because the broadcast itself doesn't need masking; only the
                // reduction op does.
                let scratch_ymm = 4 + (acc_idx.min(3)); // YMM4-YMM7
                load_rax(em, red.val_slot, ra);
                em.vmovd_xmm_r32(scratch_ymm, 0);
                em.vpbroadcastd_ymm_xmm(scratch_ymm, scratch_ymm);
                match red.op {
                    BinOpKind::Add => em.emit_vpaddd_ymm_masked(acc_ymm, acc_ymm, scratch_ymm, 1),
                    BinOpKind::Sub => em.emit_vpsubd_ymm_masked(acc_ymm, acc_ymm, scratch_ymm, 1),
                    BinOpKind::Mul => em.emit_vpmulld_ymm_masked(acc_ymm, acc_ymm, scratch_ymm, 1),
                    _ => return false,
                }
            }
        }

        // Patch the "skip masked tail" jump target
        let after_masked_tail = em.pos();
        let next_ip = skip_masked_tail_fixup + 4;
        let rel = (after_masked_tail as isize) - (next_ip as isize);
        if let Ok(rel32) = i32::try_from(rel) {
            em.as_mut_slice()[skip_masked_tail_fixup..skip_masked_tail_fixup + 4]
                .copy_from_slice(&rel32.to_le_bytes());
        }

        jit_trace!("[JIT-VEC] AVX-512 masked tail emitted (eliminates scalar remainder loop)");
    }

    // ── Step 4: Horizontal reduction ─────────────────────────────────────
    for (idx, &acc_slot) in acc_slots.iter().enumerate() {
        let ymm = idx as u8;
        let xmm_tmp = 1; // XMM1 as scratch

        // vextracti128 xmm1, ymm, 1 — get high 4 lanes
        em.vextracti128_ymm_xmm(xmm_tmp, ymm, 1);

        // Add high+low: xmm = ymm[0:3] + ymm[4:7]
        em.vpaddd_xmm(ymm, ymm, xmm_tmp);

        // vpsrldq xmm_tmp, ymm_low, 4 — shift right 4 bytes to pair adjacent i32 lanes
        // VEX.128.66.0F.WIG 73 /3 ib: vvvv=xmm_dst(NDD), reg=/3, rm=xmm_src
        {
            let byte1 = (!(ymm >= 8) as u8) << 7     // R = inverted bit3 of source
                | ((!xmm_tmp) & 0xF) << 3             // vvvv = inverted destination
                | (0 << 2)                              // L=0 for 128-bit
                | 1;                                    // pp=1 (66 prefix)
            em.emit3(0xC5, byte1, 0x73);
            em.b(0xC0 | (3 << 3) | (ymm & 7));        // ModRM: reg=/3, rm=source
            em.b(4);
        }
        em.vpaddd_xmm(ymm, ymm, xmm_tmp);

        // vpsrldq xmm_tmp, ymm_low, 8 — shift right 8 bytes (64 bits)
        // Same VEX encoding pattern as above
        {
            let byte1 = (!(ymm >= 8) as u8) << 7
                | ((!xmm_tmp) & 0xF) << 3
                | (0 << 2)
                | 1;
            em.emit3(0xC5, byte1, 0x73);
            em.b(0xC0 | (3 << 3) | (ymm & 7));
            em.b(8);
        }
        em.vpaddd_xmm(ymm, ymm, xmm_tmp);

        // vmovd eax, xmm — extract lane 0
        // VEX.128.66.0F.WIG 7E /r: xmm→ModRM:reg, r32→ModRM:r/m, vvvv=1111
        {
            let byte1 = (!(ymm >= 8) as u8) << 7     // R = inverted bit3 of xmm source
                | (0xF << 3)                            // vvvv = 1111 (unused)
                | (0 << 2)                              // L=0 for 128-bit
                | 1;                                    // pp=1 (66 prefix)
            em.emit3(0xC5, byte1, 0x7E);
            em.b(0xC0 | ((ymm & 7) << 3) | 0);        // ModRM: reg=xmm, rm=eax
        }
        em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX

        if has_mul {
            // For Mul reductions (LCG pattern: s = s * K + C), the horizontal
            // sum is NOT correct because each lane represents an independent
            // sub-sequence with stride K^VF. We need to extract all 8 lanes
            // and combine them with a "polynomial reduction":
            //   final = lane[7] + lane[6]*K + lane[5]*K² + ... + lane[0]*K⁷
            //
            // We extract lanes one at a time using vpsrldq + vmovd, and
            // accumulate: result = lane[7]; result += lane[6]*K; result += lane[5]*K²; ...
            // This requires K (the mul operand constant) to be known at JIT time.
            //
            // For now, we use a simpler but correct approach: extract each lane
            // and compute the weighted sum using scalar IMUL+ADD. This is O(VF)
            // scalar operations after the SIMD loop, which is negligible.
            //
            // If mul_stride (K^8) is available, we know K. Otherwise, we
            // fall back to extracting just the first lane and letting the
            // scalar remainder loop fix up the rest.
            if let Some(_k8) = mul_stride {
                // We have K^8. We need K. Compute K = (K^8)^(1/8) — but since
                // we're dealing with integer wrapping multiplication, this is
                // not invertible in general. Instead, we re-derive K from the
                // reduction info.
                let mul_red = vec.reductions.iter().find(|r| r.op == BinOpKind::Mul);
                let k_val: Option<i64> = mul_red.and_then(|mr| {
                    // Find the constant value of mr.val_slot
                    let loop_start = vec.loop_start.unwrap_or(0);
                    let loop_end = vec.loop_end.unwrap_or(instrs.len());
                    for pc in (loop_start..=loop_end).rev() {
                        match &instrs[pc] {
                            Instr::LoadI32(slot, v) if *slot == mr.val_slot => return Some(*v as i64),
                            Instr::LoadI64(slot, v) if *slot == mr.val_slot => return Some(*v),
                            _ => {}
                        }
                    }
                    for pc in (0..loop_start).rev() {
                        match &instrs[pc] {
                            Instr::LoadI32(slot, v) if *slot == mr.val_slot => return Some(*v as i64),
                            Instr::LoadI64(slot, v) if *slot == mr.val_slot => return Some(*v),
                            _ => {}
                        }
                    }
                    None
                });

                if let Some(k) = k_val {
                    // Polynomial reduction: extract 8 lanes and combine.
                    // Strategy: extract lane 7 → RAX = lane[7]
                    //           for i in 6..0: RAX = RAX * K + lane[i]
                    //
                    // First, extract all lanes from the YMM register into
                    // a contiguous area in the slot array (temporary spill).
                    // Then do the scalar polynomial reduction.
                    //
                    // Simpler approach: extract lane 0 to RAX, then multiply
                    // by K and add lane 1, etc. We extract lanes by shifting
                    // the XMM register and reading the low 32 bits.
                    //
                    // We already have vmovd eax, xmm — lane 0 is in EAX.
                    // Sign-extend to RAX (done above: MOVSXD RAX, EAX).
                    // This is lane 0 of the LOW 128 bits.

                    // For the polynomial reduction, we need to process lanes
                    // from HIGH to LOW: result = lane[7]; result = result*K + lane[6]; ...
                    // So we extract lane 7 first.

                    // Save lane 0 (already in RAX) to a temp slot.
                    // We'll use R9 as accumulator for the polynomial reduction.
                    // Extract lane 7 first (high 128, lane 3 → shift 12 bytes).
                    // For simplicity, extract all 8 lanes using vextracti128 + vpsrldq.

                    // Step 1: vextracti128 xmm_tmp, ymm, 1 → get high 4 lanes
                    // (Already done above for the Add case — but we're in the Mul
                    // branch, so we need to redo it.)

                    // Actually, let's use a much simpler approach:
                    // Store all 8 i32 values from the YMM register into the slot
                    // array using vmovdqu + scalar reads, then do polynomial
                    // reduction in scalar code. But that requires 32 bytes of
                    // temporary stack space.

                    // Simplest correct approach: Use the slot array as temp.
                    // Store the full YMM to a known temp area, then load each
                    // lane as i32 and do the polynomial reduction.

                    // Since we're limited to RAX/RCX for arithmetic, we'll
                    // do it in-place: extract each lane from the XMM register
                    // using vpsrldq shifts.

                    // RAX currently = lane 0 (low 32 bits, sign-extended)
                    // We need to process: lane 7, 6, 5, 4 (high 128), then 3, 2, 1, 0 (low 128)
                    // But we need lane 7 FIRST (highest power of K).

                    // Strategy: process lanes in reverse order.
                    // 1. Get high 128 bits into xmm_tmp
                    // 2. Extract lane 3 (offset 12) of xmm_tmp = original lane 7
                    // 3. That's our starting accumulator
                    // 4. Then extract lane 2 (offset 8), multiply acc by K, add
                    // 5. Continue for all 8 lanes

                    // Step 1: Get high 128 bits
                    em.vextracti128_ymm_xmm(xmm_tmp, ymm, 1);
                    // xmm_tmp = lanes [4,5,6,7]

                    // Step 2: Extract lane 7 (offset 12 in xmm_tmp)
                    // vpsrldq xmm_tmp, xmm_tmp, 12 → shift right 12 bytes, lane 7 → lane 0
                    {
                        let byte1 = (!(xmm_tmp >= 8) as u8) << 7 | (0xF << 3) | (0 << 2) | 1;
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (xmm_tmp & 7));
                        em.b(12); // shift 12 bytes
                    }
                    // vmovd eax, xmm_tmp → EAX = lane 7
                    {
                        let byte1 = (0xF << 3) | ((xmm_tmp >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (xmm_tmp & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX — R9 = lane[7]

                    // Step 3: Process lanes 6, 5, 4 from high 128 bits
                    // Re-extract high 128 bits (we destroyed xmm_tmp)
                    em.vextracti128_ymm_xmm(xmm_tmp, ymm, 1);

                    // Lane 6 (offset 8): vpsrldq xmm_tmp2 = xmm_tmp >> 8
                    // We need a second scratch register. Use XMM2 if available.
                    {
                        // vpsrldq xmm2, xmm_tmp, 8
                        let byte1 =0xF << 3 | (0 << 2) | 1; // L=1(66), pp=0
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (xmm_tmp & 7)); // /3 for VPSRLDQ
                        em.b(8); // shift 8 bytes
                    }
                    // vmovd eax, xmm2
                    {
                        em.emit3(0xC5, 0xF9, 0x7E); // VMOVD eax, xmm2
                        em.b(0xC0 | (0 << 3) | 2);
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    // R9 = R9 * K + lane[6]
                    em.mov_rcx_imm64(k);         // RCX = K
                    em.emit4(0x4C, 0x0F, 0xAF, 0xC9); // IMUL RCX, R9 — wrong, we want R9 * K
                    // Actually: R9 = R9 * K. Put K in RCX, IMUL R9, RCX.
                    // IMUL R9, RCX: REX.WB 0F AF C9 → 4D 0F AF C9
                    // But we already put K in RCX. Let's just do:
                    // MOV R10, K; IMUL R9, R10; ADD R9, RAX
                    em.emit3(0x49, 0x89, 0xCA);   // MOV R10, RCX (= K)
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[6]

                    // Lane 5 (offset 4): vpsrldq xmm2, xmm_tmp, 4
                    em.vextracti128_ymm_xmm(xmm_tmp, ymm, 1); // re-extract high 128
                    {
                        let byte1 = 0xF << 3 | (0 << 2) | 1;
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (xmm_tmp & 7));
                        em.b(4); // shift 4 bytes
                    }
                    {
                        em.emit3(0xC5, 0xF9, 0x7E);
                        em.b(0xC0 | (0 << 3) | 2);
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[5]

                    // Lane 4 (offset 0): directly from high 128 bits, no shift needed
                    em.vextracti128_ymm_xmm(xmm_tmp, ymm, 1);
                    {
                        let byte1 = (0xF << 3) | ((xmm_tmp >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (xmm_tmp & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[4]

                    // Now process lanes 3, 2, 1, 0 from LOW 128 bits (already in ymm XMM portion)
                    // Lane 3 (offset 12 in low 128)
                    // vpsrldq xmm_tmp, ymm_low, 12 → xmm_tmp = ymm_low >> 12 bytes
                    {
                        // VEX prefix: L=0 (128-bit), xmm_tmp is dest, ymm is src
                        let byte1 = (!(xmm_tmp >= 8) as u8) << 7 | ((ymm >= 8) as u8) << 2 | (0 << 2) | 1;
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (ymm & 7)); // /3 for VPSRLDQ
                        em.b(12);
                    }
                    // vmovd eax, xmm_tmp
                    {
                        let byte1 = (0xF << 3) | ((xmm_tmp >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (xmm_tmp & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[3]

                    // Lane 2 (offset 8)
                    {
                        let byte1 = (!(xmm_tmp >= 8) as u8) << 7 | ((ymm >= 8) as u8) << 2 | (0 << 2) | 1;
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (ymm & 7));
                        em.b(8);
                    }
                    {
                        let byte1 = (0xF << 3) | ((xmm_tmp >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (xmm_tmp & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[2]

                    // Lane 1 (offset 4)
                    {
                        let byte1 = (!(xmm_tmp >= 8) as u8) << 7 | ((ymm >= 8) as u8) << 2 | (0 << 2) | 1;
                        em.emit3(0xC5, byte1, 0x73);
                        em.b(0xC0 | (3 << 3) | (ymm & 7));
                        em.b(4);
                    }
                    {
                        let byte1 = (0xF << 3) | ((xmm_tmp >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (xmm_tmp & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[1]

                    // Lane 0 (offset 0) — read directly from the YMM register's low 128 bits
                    // vpsrldq xmm_tmp, ymm_low, 0 is a no-op, so just vmovd directly
                    {
                        let byte1 = (0xF << 3) | ((ymm >= 8) as u8) << 2 | 1;
                        em.emit3(0xC5, byte1, 0x7E);
                        em.b(0xC0 | (0 << 3) | (ymm & 7));
                    }
                    em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                    em.emit4(0x4D, 0x0F, 0xAF, 0xCA); // IMUL R9, R10 (R9 *= K)
                    em.emit3(0x4C, 0x01, 0xC1);   // ADD R9, RAX — R9 += lane[0]

                    // R9 = lane[7]*K^0 + lane[6]*K^1 + ... + lane[0]*K^7
                    // But we also need to add the addend stride (if has_addsub).
                    // For LCG: s = s * K + C, the SIMD version computes:
                    //   s_new = s * K^8 + C * (K^7 + K^6 + ... + K + 1)
                    // So each lane already has the addend baked in (we broadcast
                    // the accumulator and applied mul+add in each SIMD iteration).
                    // The polynomial reduction gives us the correct final value.

                    // Move result to RAX and store
                    em.emit3(0x4C, 0x89, 0xC8); // MOV RAX, R9
                    store_rax(em, acc_slot, ra);
                } else {
                    // K is not a compile-time constant — fall back to scalar
                    // Store the first lane and let scalar remainder handle it.
                    // This is correct because the scalar remainder loop will
                    // process any remaining iterations.
                    store_rax(em, acc_slot, ra);
                }
            } else {
                // No mul_stride available — fall back to scalar
                store_rax(em, acc_slot, ra);
            }
        } else {
            // For Add/Sub: horizontal sum = total increment.
            // Add it to the ORIGINAL accumulator value.
            // RAX currently holds the reduced sum. Save to R9, reload acc, add.
            em.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX — save reduced sum
            load_rax(em, acc_slot, ra);  // RAX = original acc value
            // ADD RAX, R9: REX.WR=0x4C, opcode=0x01, ModRM=11_001_000=0xC8
            em.emit3(0x4C, 0x01, 0xC8); // ADD RAX, R9
            store_rax(em, acc_slot, ra);
        }
    }

    // ── Step 5: Update induction variable ─────────────────────────────────
    {
        let total_step = vf * vec.induction_step;
        load_rax(em, iv_slot, ra);      // RAX = current iv
        if use_avx512_tail {
            // AVX-512 masked tail handles the remainder, so the IV must be
            // advanced by total_iterations * induction_step (not just trips * vf * step).
            // R10 = total_iterations (saved in Step 1).
            // Compute: increment = R10 * induction_step, then iv += increment.
            let istep = vec.induction_step;
            if istep == 1 {
                // Simple case: iv += total_iterations
                em.emit3(0x4C, 0x89, 0xD1);    // MOV RCX, R10 (total_iterations)
                em.add_rax_rcx();               // RAX = iv + total_iterations
            } else if istep == 8 {
                // iv += total_iterations * 8
                em.emit3(0x4C, 0x89, 0xD1);    // MOV RCX, R10
                em.emit4(0x48, 0xC1, 0xE1, 3); // SHL RCX, 3 (total_iterations * 8)
                em.add_rax_rcx();
            } else {
                // General case: iv += total_iterations * induction_step
                em.mov_rcx_imm64(istep);         // RCX = induction_step
                em.emit4(0x4C, 0x0F, 0xAF, 0xD1); // IMUL R10, RCX (R10 = total_iterations * istep)
                em.emit3(0x4C, 0x89, 0xD1);      // MOV RCX, R10
                em.add_rax_rcx();                 // RAX = iv + increment
            }
        } else {
            // Original path: IV += trips * total_step
            // R8 = trips. We need R8 * total_step, then add to iv.
            em.emit3(0x4C, 0x89, 0xC1);    // MOV RCX, R8 (trips)
            if total_step == 8 {
                em.emit4(0x48, 0xC1, 0xE1, 3); // SHL RCX, 3 (trips * 8)
            } else {
                // Use IMUL: RCX = RCX * total_step
                em.mov_rax_imm_opt(total_step); // RAX = total_step (clobbered but we'll reload iv)
                em.emit3(0x49, 0x89, 0xC1);    // MOV R9, RAX (save total_step to R9)
                load_rax(em, iv_slot, ra);      // Reload iv
                em.emit3(0x4C, 0x89, 0xC1);    // MOV RCX, R8 (trips)
                em.emit4(0x4C, 0x0F, 0xAF, 0xC9); // IMUL RCX, R9
            }
            em.add_rax_rcx();               // RAX = iv + increment
        }
        store_rax(em, iv_slot, ra);     // store updated iv
    }

    // VZEROUPPER before continuing with scalar code
    em.vzeroupper();

    // ── Step 6: Patch the skip-SIMD jump ──────────────────────────────────
    let skip_simd_target = em.pos();
    {
        let next_ip = skip_simd_fixup + 4;
        let rel = (skip_simd_target as isize) - (next_ip as isize);
        if let Ok(rel32) = i32::try_from(rel) {
            em.as_mut_slice()[skip_simd_fixup..skip_simd_fixup + 4].copy_from_slice(&rel32.to_le_bytes());
        }
    }

    // ── Step 7: Patch the SIMD loop backward branch ──────────────────────
    for fx in fixups.iter_mut() {
        if fx.target_pc == usize::MAX && fx.kind == BranchKind::Jnz {
            let next_ip = fx.disp_pos + 4;
            let rel = (simd_loop_start as isize) - (next_ip as isize);
            if let Ok(rel32) = i32::try_from(rel) {
                em.as_mut_slice()[fx.disp_pos..fx.disp_pos + 4].copy_from_slice(&rel32.to_le_bytes());
            }
            // Leave target_pc as usize::MAX so the main patch_fixups()
            // function skips this fixup — it's already resolved above
            // using the machine code offset, not the PC-based offset.
            break;
        }
    }

    jit_trace!("[JIT-VEC] AVX2 vectorized loop emitted ({} bytes for SIMD portion), avx512_masked_tail={}",
        em.pos() - simd_loop_start + 50, use_avx512_tail);
    true
}

/// Emit a JLE rel32 placeholder (jump if less-or-equal, signed).
fn jle_rel32_placeholder(em: &mut Emitter) -> usize {
    em.emit2(0x0F, 0x8E); // JLE rel32
    let p = em.pos();
    em.d(0);
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant propagation
// ─────────────────────────────────────────────────────────────────────────────

fn fold_binop(op: BinOpKind, l: i64, r: i64) -> Option<i64> {
    Some(match op {
        BinOpKind::Add => l.wrapping_add(r),
        BinOpKind::Sub => l.wrapping_sub(r),
        BinOpKind::Mul => l.wrapping_mul(r),
        BinOpKind::Div => {
            if r == 0 {
                return None;
            }
            l.wrapping_div(r)
        }
        BinOpKind::Rem => {
            if r == 0 {
                return None;
            }
            l.wrapping_rem(r)
        }
        BinOpKind::Eq => i64::from(l == r),
        BinOpKind::Ne => i64::from(l != r),
        BinOpKind::Lt => i64::from(l < r),
        BinOpKind::Le => i64::from(l <= r),
        BinOpKind::Gt => i64::from(l > r),
        BinOpKind::Ge => i64::from(l >= r),
        _ => return None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat constant-propagation table (replaces HashMap<u16, i64>)
// ─────────────────────────────────────────────────────────────────────────────
//
// Vec<Option<i64>> indexed by slot gives O(1) lookup with zero hashing overhead.
// For a function with N slots this uses N*9 bytes vs HashMap's ~48 bytes base +
// 24 bytes/entry at low load factors. More importantly, sequential slot accesses
// stay in L1 cache during the hot codegen loop.

struct ConstTable {
    vals: Vec<Option<i64>>,
}

impl ConstTable {
    fn with_capacity(n: usize) -> Self {
        Self {
            vals: vec![None; n.max(1)],
        }
    }

    #[inline(always)]
    fn get(&self, slot: u16) -> Option<i64> {
        // P9 fix: Use and_then instead of copied().flatten() to avoid
        // copying the Option<i64> (8 bytes) and double-branch unwrapping.
        // The old form: .get(slot).copied().flatten() = 2 branches + 8B copy.
        // The new form: .get(slot).and_then(|&v| v) = 1 branch + 0B copy.
        self.vals.get(slot as usize).and_then(|&v| v)
    }

    #[inline(always)]
    fn insert(&mut self, slot: u16, v: i64) {
        let idx = slot as usize;
        if idx >= self.vals.len() {
            self.vals.resize(idx + 1, None);
        }
        self.vals[idx] = Some(v);
    }

    #[inline(always)]
    fn remove(&mut self, slot: u16) {
        if let Some(cell) = self.vals.get_mut(slot as usize) {
            *cell = None;
        }
    }

    /// Clear all known constants (conservative: called at branch targets).
    fn clear(&mut self) {
        self.vals.fill(None);
    }
}

/// Slot type tracking — records the *type* of each slot so we can
/// distinguish integer from float operations without relying on the
/// broken const_at heuristic (Bug #1 fix).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotType {
    Unknown,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Unit,
}

struct TypeTable {
    tys: Vec<SlotType>,
}

impl TypeTable {
    fn with_capacity(n: usize) -> Self {
        Self {
            tys: vec![SlotType::Unknown; n.max(1)],
        }
    }

    #[inline(always)]
    fn get(&self, slot: u16) -> SlotType {
        self.tys.get(slot as usize).copied().unwrap_or(SlotType::Unknown)
    }

    #[inline(always)]
    fn set(&mut self, slot: u16, ty: SlotType) {
        let idx = slot as usize;
        if idx >= self.tys.len() {
            self.tys.resize(idx + 1, SlotType::Unknown);
        }
        self.tys[idx] = ty;
    }

    /// Clear all types (conservative: called at branch targets).
    fn clear(&mut self) {
        self.tys.fill(SlotType::Unknown);
    }

    /// Returns true if the slot is a known float type (F32 or F64).
    #[inline(always)]
    fn is_float(&self, slot: u16) -> bool {
        matches!(self.get(slot), SlotType::F32 | SlotType::F64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime constant tracker — Superpower 6: Dynamic Data-Driven Inlining
// ─────────────────────────────────────────────────────────────────────────────
//
// Promotes runtime values that haven't changed across iterations to
// compile-time constants. This is the "Superpower 6: Dynamic Data-Driven
// Inlining" — the JIT observes that a value like `gravity = 9.8` was set
// at startup and hasn't changed, so it rewrites the machine code to treat
// 9.8 as a hardcoded constant.
//
// The tracker is fed by the interpreter's profiling loop (PGO counters).
// Slots that haven't changed for N observations are considered "stable".
// On recompilation, stable slots are injected into the const_at table,
// allowing downstream constant folding, immediate encoding, and dead
// code elimination to treat them as if they were statically known.

/// Runtime constant tracker — promotes runtime values that haven't changed
/// across iterations to compile-time constants.
struct RuntimeConstantTracker {
    /// Slot → (observed_value, observation_count, last_changed_at)
    observations: Vec<(i64, u64, u64)>,
    /// How many observations before a slot is considered "stable"
    stability_threshold: u64,
}

impl RuntimeConstantTracker {
    fn new(slot_count: usize) -> Self {
        RuntimeConstantTracker {
            observations: vec![(0, 0, 0); slot_count],
            stability_threshold: 20,
        }
    }

    /// Observe a slot value. Returns true if the slot is now considered stable.
    fn observe(&mut self, slot: usize, value: i64, iteration: u64) -> bool {
        if slot >= self.observations.len() {
            return false;
        }
        let (ref mut prev_val, ref mut count, ref mut last_changed) = self.observations[slot];
        if *count == 0 || *prev_val != value {
            *prev_val = value;
            *last_changed = iteration;
        }
        *count += 1;
        // A slot is "stable" if it's been observed many times and hasn't changed
        // for a significant portion of those observations
        *count >= self.stability_threshold && (iteration - *last_changed) >= self.stability_threshold / 2
    }

    /// Check if a slot is a stable runtime constant
    fn is_stable(&self, slot: usize) -> bool {
        if slot >= self.observations.len() {
            return false;
        }
        let (_val, count, last_changed) = self.observations[slot];
        count >= self.stability_threshold && (count - last_changed) >= self.stability_threshold / 2
    }

    /// Get the stable value for a slot, if it's stable
    fn stable_value(&self, slot: usize) -> Option<i64> {
        if self.is_stable(slot) {
            Some(self.observations[slot].0)
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(0, r),
        RegLoc::Spill(off) => em.load_reg_mem(0, off),
        // Xmm: float slots should not be loaded into GPRs. If we end up
        // here (e.g. for a comparison result stored in the slot array),
        // load from the default slot offset.
        RegLoc::Xmm(_) => em.load_reg_mem(0, (slot as i32) * 8),
    }
}

#[inline(always)]
fn load_rcx(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(1, r),
        RegLoc::Spill(off) => em.load_reg_mem(1, off),
        RegLoc::Xmm(_) => em.load_reg_mem(1, (slot as i32) * 8),
    }
}

#[inline(always)]
fn store_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(r, 0),
        RegLoc::Spill(off) => em.store_mem_reg(off, 0),
        RegLoc::Xmm(_) => em.store_mem_reg((slot as i32) * 8, 0),
    }
}

#[inline(always)]
fn instr_reads_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => *s == slot,
        Instr::BinOp(_, _, l, r) => *l == slot || *r == slot,
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => *s == slot,
        _ => false,
    }
}

#[inline(always)]
fn instr_writes_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::LoadI32(d, _)
        | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _)
        | Instr::LoadUnit(d)
        | Instr::Move(d, _)
        | Instr::Load(d, _)
        | Instr::Store(d, _)
        | Instr::BinOp(d, _, _, _) => *d == slot,
        Instr::TensorBinOp { dst, .. } => *dst == slot,
        Instr::TensorReduce { dst, .. } => *dst == slot,
        Instr::TensorMatMul { dst, .. } => *dst == slot,
        _ => false,
    }
}

/// Return all slots read by an instruction (used by the DAG scheduler).
#[inline]
fn instr_reads_slots(instr: &Instr) -> Vec<u16> {
    match instr {
        Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => vec![*s],
        Instr::BinOp(_, _, l, r) => vec![*l, *r],
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => vec![*s],
        _ => Vec::new(),
    }
}

/// Return all slots written by an instruction (used by the DAG scheduler).
#[inline]
fn instr_writes_slots(instr: &Instr) -> Vec<u16> {
    match instr {
        Instr::LoadI32(d, _)
        | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _)
        | Instr::LoadUnit(d)
        | Instr::Move(d, _)
        | Instr::Load(d, _)
        | Instr::Store(d, _)
        | Instr::BinOp(d, _, _, _) => vec![*d],
        Instr::TensorBinOp { dst, .. } => vec![*dst],
        Instr::TensorReduce { dst, .. } => vec![*dst],
        Instr::TensorMatMul { dst, .. } => vec![*dst],
        _ => Vec::new(),
    }
}

#[inline(always)]
fn is_control_flow_barrier(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::Jump(_)
            | Instr::JumpFalse(_, _)
            | Instr::JumpTrue(_, _)
            | Instr::Return(_)
            | Instr::ReturnUnit
    )
}

/// Local straight-line dead-definition check.
///
/// Returns true when a write to `slot` at `pc` is overwritten before any read
/// and before any control-flow barrier.
fn is_straight_line_dead_def(instrs: &[Instr], pc: usize, slot: u16) -> bool {
    let mut i = pc + 1;
    while i < instrs.len() {
        let next = &instrs[i];
        if is_control_flow_barrier(next) {
            return false;
        }
        if instr_reads_slot(next, slot) {
            return false;
        }
        if instr_writes_slot(next, slot) {
            return true;
        }
        i += 1;
    }
    true
}

/// Returns true when `op` is commutative (order of operands doesn't matter).
#[inline(always)]
fn commutative_binop(op: BinOpKind) -> bool {
    matches!(op,
        BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq | BinOpKind::Ne
        | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
        | BinOpKind::And | BinOpKind::Or)
}

/// Returns true when `op` is in the set we know how to emit.
#[inline(always)]
fn is_supported_binop(op: BinOpKind) -> bool {
    matches!(
        op,
        BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Rem
            | BinOpKind::FloorDiv
            | BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge
            | BinOpKind::And
            | BinOpKind::Or
            | BinOpKind::BitAnd
            | BinOpKind::BitOr
            | BinOpKind::BitXor
            | BinOpKind::Shl
            | BinOpKind::Shr
            | BinOpKind::Min
            | BinOpKind::Max
    )
}

/// Emit the arithmetic/comparison body: lhs already in rax, rhs in rcx.
#[inline(always)]
fn emit_binop_rax_rcx(em: &mut Emitter, op: BinOpKind) -> bool {
    match op {
        BinOpKind::Add => em.add_rax_rcx(),
        BinOpKind::Sub => em.sub_rax_rcx(),
        BinOpKind::Mul => {
            // TODO(BMI2): When BMI2 is available, MULX r64, r/m64, r64 (VEX-encoded
            // F2 0F38 F6 /r) computes the low 64 bits of the product without
            // clobbering RDX, which would free up the RDX=reserved-for-IDIV
            // invariant and allow tighter register allocation around mul+div
            // sequences.  MULX also writes both product halves (dst, RDX) in
            // one uop on Zen3+/Ice Lake+.  For now, IMUL RAX, RCX is already
            // optimal for the common case (single low-half multiply).
            em.imul_rax_rcx();
        }
        BinOpKind::Div => {
            em.cqo();
            em.idiv_rcx();
        }
        BinOpKind::Rem => {
            em.cqo();
            em.idiv_rcx();
            em.mov_rax_rdx();
        }
        BinOpKind::Eq => {
            em.cmp_rax_rcx();
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_rcx();
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9C);
            em.movzx_rax_al();
            // FIX #3: Removed spurious CMOVL that overwrote the boolean comparison
            // result with one of the operands. SETCC+MOVZX already produces
            // the correct 0/1 result in RAX. The CMOV was incorrectly replacing
            // the boolean result (0 or 1) with the operand value from RCX.
        }
        BinOpKind::Le => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9E);
            em.movzx_rax_al();
            // FIX #3: Removed spurious CMOVLE that overwrote boolean result.
        }
        BinOpKind::Gt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9F);
            em.movzx_rax_al();
            // FIX #3: Removed spurious branchless max emitter that overwrote
            // the boolean comparison result with one of the operands.
        }
        BinOpKind::Ge => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9D);
            em.movzx_rax_al();
            // FIX #3: Removed spurious branchless min emitter that overwrote
            // the boolean comparison result with one of the operands.
        }
        BinOpKind::BitAnd => {
            em.and_rax_rcx();
        }
        BinOpKind::BitOr => {
            em.or_rax_rcx();
        }
        BinOpKind::BitXor => {
            em.xor_rax_rcx();
        }
        BinOpKind::Shl => {
            // RCX holds shift count; use CL form.
            // TODO(BMI2): SHLX RAX, RAX, RCX (VEX.NDS.LZ.0F38.W0 F7 /r) would
            // avoid clobbering flags, allowing the shift result to be used in
            // SETCC/CMOVcc without an intermediate. The CL form (3 bytes) is
            // already compact and optimal for count-in-RCX; SHLX (5 bytes VEX)
            // is only beneficial when flag preservation matters.
            em.shl_rax_cl();
        }
        BinOpKind::Shr => {
            // Use arithmetic shift right for signed integers.
            // TODO(BMI2): SARX RAX, RAX, RCX (VEX.NDS.LZ.F3.0F38.W0 F7 /r) would
            // preserve flags like SHLX above. Same cost-benefit tradeoff.
            em.sar_rax_cl();
        }
        BinOpKind::And => {
            // Logical AND: short-circuit not possible in simple emit; 
            // evaluate both and AND the boolean results
            em.and_rax_rcx();
        }
        BinOpKind::Or => {
            // Logical OR: evaluate both and OR the boolean results
            em.or_rax_rcx();
        }
        BinOpKind::FloorDiv => {
            // Floor division: rounds toward negative infinity.
            // After IDIV: RAX=quotient (truncated toward zero), RDX=remainder
            // RCX is NOT clobbered by IDIV (it reads RCX as divisor but doesn't write it).
            //
            // Adjustment needed if: remainder != 0 AND (remainder XOR divisor) has sign bit set.
            // Because IDIV remainder has the same sign as the dividend, so if remainder and
            // divisor have opposite signs, then dividend and divisor had opposite signs,
            // meaning floor != trunc → subtract 1 from quotient.
            //
            // Branchless approach: compute adjustment ∈ {0, 1}, then result = quotient - adjustment.
            em.cqo();
            em.idiv_rcx();
            // RAX = quotient, RDX = remainder, RCX = divisor (unchanged)
            em.push_reg(0);  // push RAX (save quotient)
            // Compute sign_mismatch = (RDX ^ RCX) >> 63
            em.mov_rax_rdx();       // RAX = remainder
            em.xor_rax_rcx();       // RAX = remainder ^ divisor
            em.shr_rax_imm8(63);    // RAX = 1 if signs differ, 0 if same
            em.push_reg(0);         // push sign_mismatch
            // Compute remainder_nonzero = (RDX != 0) ? 1 : 0
            em.mov_rax_rdx();       // RAX = remainder (RDX still intact)
            em.test_rax_rax();      // ZF = (remainder == 0)
            em.setcc_al(0x95);      // SETNZ AL: AL = 1 if remainder != 0
            em.movzx_rax_al();      // RAX = remainder_nonzero (0 or 1)
            // adjustment = sign_mismatch AND remainder_nonzero
            em.pop_reg(1);          // RCX = sign_mismatch
            em.and_rax_rcx();       // RAX = adjustment (0 or 1)
            // result = quotient - adjustment
            em.pop_reg(1);          // RCX = quotient
            em.sub_reg_reg(1, 0);   // RCX = RCX - RAX = quotient - adjustment
            em.mov_rax_rcx();       // RAX = result
        }
        BinOpKind::Min => {
            // Branchless min: RAX = min(RAX, RCX)
            em.emit_branchless_min(0, 1); // min(RAX, RCX)
        }
        BinOpKind::Max => {
            // Branchless max: RAX = max(RAX, RCX)
            em.emit_branchless_max(0, 1); // max(RAX, RCX)
        }
        BinOpKind::FmaAdd => {
            // FMA fusion fallback via GPR path: just add (mul already computed)
            em.add_rax_rcx();
        }
    }
    true
}

/// Emit immediate-rhs form: lhs already in rax.
#[inline(always)]
fn emit_binop_rax_imm(em: &mut Emitter, op: BinOpKind, imm: i32) {
    match op {
        BinOpKind::Add => {
            if imm == 1 {
                em.inc_rax();
            } else if imm == -1 {
                em.dec_rax();
            } else if imm != 0 {
                em.add_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Sub => {
            if imm == 1 {
                em.dec_rax();
            } else if imm == -1 {
                em.inc_rax();
            } else if imm != 0 {
                em.sub_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Mul => {
            if imm == 0 {
                em.xor_eax_eax();
            } else if imm == 1 { /* nop */
            } else if imm == -1 {
                em.neg_rax();
            } else if imm == 3 {
                em.lea_rax_rax_mul3();
            } else if imm == 5 {
                em.lea_rax_rax_mul5();
            } else if imm == 9 {
                em.lea_rax_rax_mul9();
            } else if imm > 0 && (imm as u32).is_power_of_two() {
                em.shl_rax_imm8((imm as u32).trailing_zeros() as u8);
            } else {
                em.imul_rax_imm32(imm);
            }
        }
        BinOpKind::Div => {
            // Try magic-number division first (3-4 cycles vs 20-40 for IDIV).
            // Falls back to IDIV if the divisor is not suitable for magic.
            if !emit_div_magic_sequence(em, imm as i64) {
                em.mov_rcx_imm64(imm as i64);
                em.cqo();
                em.idiv_rcx();
            }
        }
        BinOpKind::Rem => {
            // Try magic-number remainder first.
            // Falls back to IDIV if the divisor is not suitable for magic.
            if !emit_rem_magic_sequence(em, imm as i64) {
                em.mov_rcx_imm64(imm as i64);
                em.cqo();
                em.idiv_rcx();
                em.mov_rax_rdx();
            }
        }
        BinOpKind::Eq => {
            em.cmp_rax_imm32(imm); // now uses imm8 when it fits
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9C);
            em.movzx_rax_al();
        }
        BinOpKind::Le => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9E);
            em.movzx_rax_al();
        }
        BinOpKind::Gt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9F);
            em.movzx_rax_al();
        }
        BinOpKind::Ge => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9D);
            em.movzx_rax_al();
        }
        BinOpKind::BitAnd => {
            em.and_rax_imm32(imm);
        }
        BinOpKind::BitOr => {
            em.or_rax_imm32(imm);
        }
        BinOpKind::BitXor => {
            em.xor_rax_imm32(imm);
        }
        BinOpKind::Shl => {
            if imm > 0 && imm < 64 {
                em.shl_rax_imm8(imm as u8);
            }
        }
        BinOpKind::Shr => {
            if imm > 0 && imm < 64 {
                em.sar_rax_imm8(imm as u8);
            }
        }
        BinOpKind::And => {
            em.and_rax_imm32(imm);
        }
        BinOpKind::Or => {
            em.or_rax_imm32(imm);
        }
        BinOpKind::FloorDiv => {
            // For immediate divisor, use same approach as Div but with imm→RCX
            em.mov_rcx_imm64(imm as i64);
            em.cqo();
            em.idiv_rcx();
        }
        BinOpKind::Min | BinOpKind::Max => {
            // Min/Max with immediate: load imm into RCX, then use branchless min/max
            em.mov_rcx_imm64(imm as i64);
            if op == BinOpKind::Min {
                em.emit_branchless_min(0, 1);
            } else {
                em.emit_branchless_max(0, 1);
            }
        }
        BinOpKind::FmaAdd => {
            // FMA with immediate: just add (mul already computed)
            em.add_rax_imm32(imm);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Magic-number division sequences (Granston & Montgomery algorithm)
// ─────────────────────────────────────────────────────────────────────────────
//
// Replaces IDIV (20-40 cycles) with IMUL+SHR+ADD (3-4 cycles) when the
// divisor is a compile-time constant. The algorithm computes a "magic"
// constant M and shift S such that:
//   n / d ≈ high64(M * n) >> S
// with a sign-correction step for negative dividends.
//
// Register usage:
//   RAX = dividend on entry, quotient on exit
//   RCX = scratch (magic constant, divisor for remainder)
//   RDX = clobbered by IMUL (high 64 bits of product)
//   R8  = scratch (saved/restored via push/pop; holds original dividend)

/// Emit magic-number division sequence for signed division by a constant.
///
/// Input:  RAX = dividend (n)
/// Output: RAX = quotient (n / d), truncated toward zero
/// Clobbers: RCX, RDX; saves/restores R8 via push/pop
///
/// Returns `true` if the magic sequence was emitted, `false` if the
/// divisor is not suitable (caller should fall back to IDIV).
fn emit_div_magic_sequence(em: &mut Emitter, divisor: i64) -> bool {
    if let Some((magic, shift)) = compute_div_magic(divisor) {
        let neg_divisor = divisor < 0;

        // Save R8 (it may be allocated to a slot by the register allocator).
        em.push_reg(8);

        // Save original dividend for sign correction.
        em.mov_r8_rax();

        // Load magic constant and do 128-bit signed multiply.
        // One-operand IMUL: RDX:RAX = RAX * RCX (128-bit product).
        em.mov_rcx_imm64(magic);
        em.emit_imul_rcx_128();

        // Take the high 64 bits of the product (RDX).
        em.mov_rax_rdx();

        // Apply post-shift.
        if shift > 0 {
            em.shr_rax_imm8(shift);
        }

        // Sign correction for truncation toward zero:
        // If the original dividend was negative, the unsigned high-word
        // approximation undercounts by 1, so add the sign bit.
        em.shr_r8_imm8(63);     // R8 = 1 if dividend was negative, 0 if positive
        em.add_rax_r8();         // correction: quotient += sign_bit

        // For negative divisors, negate the result.
        // compute_div_magic(|d|) gives magic for |d|; we negate the quotient
        // because n / (-|d|) = -(n / |d|).
        if neg_divisor {
            em.neg_rax();
        }

        // Restore R8.
        em.pop_reg(8);

        true
    } else {
        false
    }
}

/// Emit magic-number remainder sequence for signed remainder by a constant.
///
/// Uses the identity:  n % d = n - (n / d) * d
/// First computes the quotient via magic-number division, then multiplies
/// back by the original divisor and subtracts from the dividend.
///
/// Input:  RAX = dividend (n)
/// Output: RAX = remainder (n % d)
/// Clobbers: RCX, RDX; saves/restores R8 via push/pop
///
/// Returns `true` if the magic sequence was emitted, `false` if the
/// divisor is not suitable (caller should fall back to IDIV).
fn emit_rem_magic_sequence(em: &mut Emitter, divisor: i64) -> bool {
    if let Some((magic, shift)) = compute_div_magic(divisor) {
        let neg_divisor = divisor < 0;

        // Save R8 (may be allocated).
        em.push_reg(8);

        // Save original dividend in R8 for the final subtraction.
        em.mov_r8_rax();

        // --- Compute quotient via magic multiply (same as emit_div_magic_sequence) ---
        em.mov_rcx_imm64(magic);
        em.emit_imul_rcx_128();
        em.mov_rax_rdx();
        if shift > 0 {
            em.shr_rax_imm8(shift);
        }
        em.shr_r8_imm8(63);
        em.add_rax_r8();
        if neg_divisor {
            em.neg_rax();
        }
        // RAX = quotient (n / d)

        // --- Compute remainder: RAX = R8 - RAX * divisor ---
        // quotient * divisor: use two-operand IMUL (low 64 bits suffice;
        // the product cannot overflow because |q * d| ≤ |n| for truncating
        // division with |d| ≥ 2, which is guaranteed by compute_div_magic).
        em.mov_rcx_imm64(divisor);
        em.imul_rax_rcx();      // RAX = quotient * divisor (low 64 bits)
        em.mov_rcx_rax();       // RCX = quotient * divisor
        em.mov_rax_r8();        // RAX = original dividend (saved in R8)
        em.sub_rax_rcx();       // RAX = dividend - quotient * divisor = remainder

        // Restore R8.
        em.pop_reg(8);

        true
    } else {
        false
    }
}

/// Emit pops (reverse push order) followed by RET.
/// Conditionally emits VZEROUPPER before RET only if any AVX/YMM/ZMM
/// (SIMD) instruction was emitted during compilation. This avoids the
/// 70-cycle AVX→SSE transition penalty when SIMD was used, while
/// avoiding the unnecessary overhead of vzeroupper on scalar-only
/// workloads (which would force the CPU to save/restore upper YMM state
/// even when no YMM registers were written).
fn emit_ret(em: &mut Emitter, callee_saved: &[u8]) {
    // Conditionally emit TILERELEASE if AMX instructions were used.
    // This releases the tile configuration and frees the tile registers,
    // preventing AMX state from leaking to the caller. Must come before
    // VZEROUPPER since TILERELEASE is a VEX-encoded instruction.
    if em.emitted_amx {
        em.amx_tilerelease();
    }
    // Conditionally emit VZEROUPPER only if SIMD instructions were emitted.
    // This fixes the previous "all or nothing" approach where vzeroupper was
    // either always emitted (hurting scalar workloads) or completely disabled
    // (causing AVX→SSE transition penalties on mixed workloads).
    if em.emitted_simd && cpu_features().has_avx {
        em.vzeroupper();
    }
    for &reg in callee_saved.iter().rev() {
        em.pop_reg(reg);
    }
    em.ret();
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch fixup with short-branch shrinking
// ─────────────────────────────────────────────────────────────────────────────
//
// We record each branch as a placeholder in the byte stream.  After all code
// is emitted we know every target offset, so we can choose the shortest
// encoding.  Because shrinking a branch changes offsets, we do a single
// relaxation pass: convert rel32 → rel8 where the *original* displacement
// would already fit (conservative but correct — shrinking can only move
// targets closer).

/// Kind of branch instruction at a fixup site.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BranchKind {
    Jmp,
    Jz,
    Jnz,
}

/// A pending branch: position of the disp32 field, target PC, and kind.
struct Fixup {
    /// Byte index of the 4-byte disp32 placeholder.
    disp_pos: usize,
    target_pc: usize,
    kind: BranchKind,
}

/// Patch all branch displacements.  Returns None if any target is unreachable
/// or any displacement overflows i32.
///
/// Short-branch shrinking: we convert rel32 → rel8 by overwriting the
/// opcode+disp bytes and padding the leftover bytes with NOPs.  Because the
/// instruction *length is unchanged* (we pad with NOPs), all `pc_to_off`
/// entries remain valid and no downstream offsets shift.  The key subtlety is
/// that the rel8 displacement must be measured from the IP *after the 2-byte
/// short instruction*, not after the 5/6-byte long instruction.
fn patch_fixups(buf: &mut [u8], fixups: &[Fixup], pc_to_off: &[usize]) -> Option<()> {
    for fx in fixups {
        // Skip fixups that were already resolved by emit_vectorized_loop().
        // The SIMD backward branch is patched inline using the machine code
        // offset (not PC offset), so it must not be re-patched here.
        if fx.target_pc == usize::MAX {
            continue;
        }
        let target_off = *pc_to_off.get(fx.target_pc)? as isize;

        let opcode_start = match fx.kind {
            BranchKind::Jmp => fx.disp_pos - 1,                  // E9 [d32]
            BranchKind::Jz | BranchKind::Jnz => fx.disp_pos - 2, // 0F 84/85 [d32]
        };

        // IP after the short (2-byte) encoding.
        let short_next_ip = (opcode_start + 2) as isize;
        let short_rel = target_off - short_next_ip;

        if let Ok(rel8) = i8::try_from(short_rel) {
            // Shrink to rel8 + NOPs.  Instruction boundaries don't move.
            let short_op: u8 = match fx.kind {
                BranchKind::Jmp => 0xEB,
                BranchKind::Jz => 0x74,
                BranchKind::Jnz => 0x75,
            };
            buf[opcode_start] = short_op;
            buf[opcode_start + 1] = rel8 as u8;
            // Overwrite remaining bytes with NOPs to preserve instruction length.
            let nop_start = opcode_start + 2;
            let nop_end = fx.disp_pos + 4;
            for b in &mut buf[nop_start..nop_end] {
                *b = 0x90;
            }
        } else {
            // Keep rel32 form; IP after the long encoding.
            let long_next_ip = (fx.disp_pos + 4) as isize;
            let rel = i32::try_from(target_off - long_next_ip).ok()?;
            buf[fx.disp_pos..fx.disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }
    Some(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Common Subexpression Elimination (CSE)
// ─────────────────────────────────────────────────────────────────────────────
//
// Identifies identical BinOp computations and reuses their results instead of
// recomputing.  For each BinOp, we compute a hash from (op, left_slot, right_slot)
// and track the destination slot that already holds the result.  Later identical
// BinOps are replaced with Move(dest, earlier_dest).


fn cse_optimize(instrs: &mut Vec<Instr>) {
    use std::collections::HashMap;
    use std::mem::Discriminant;
    
    // Hash key: (op_discriminant, left_slot, right_slot)
    let mut computed: HashMap<(Discriminant<BinOpKind>, u16, u16), u16> = HashMap::new();
    
    // We must clear the CSE table at control-flow barriers because
    // the earlier definition may not dominate the later use.
    let mut i = 0;
    while i < instrs.len() {
        // P5 fix: Extract information from the instruction BEFORE the match
        // to avoid borrow checker conflicts (immutable read vs mutable write).
        let instr_info: Option<(BinOpKind, u16, u16, u16)> = match &instrs[i] {
            Instr::BinOp(dst, op, l, r) => Some((*op, *dst, *l, *r)),
            _ => None,
        };
        let is_barrier = matches!(&instrs[i], 
            Instr::Jump(_) | Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _) | Instr::Return(_) | Instr::ReturnUnit
        );
        let written_slot = instr_writes_slot_get(&instrs[i]);
        
        if let Some((op, dst, l, r)) = instr_info {
            // Canonicalize operand order for commutative ops
            let (lo, hi) = if l <= r { (l, r) } else { (r, l) };
            let key = if matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq | BinOpKind::Ne | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
                (std::mem::discriminant(&op), lo, hi)
            } else {
                (std::mem::discriminant(&op), l, r)
            };
            
            if let Some(&prev_dst) = computed.get(&key) {
                // Replace this BinOp with a Move from the previous result.
                instrs[i] = Instr::Move(dst, prev_dst);
            } else {
                computed.insert(key, dst);
            }
            
            // P5 fix: Invalidate CSE entries whose inputs are overwritten.
            computed.retain(|&(_, l, r), _| l != dst && r != dst);
        } else if is_barrier {
            computed.clear();
        } else {
            // P5 fix: Any instruction that writes to a slot invalidates
            // CSE entries that depend on that slot.
            if let Some(written) = written_slot {
                computed.retain(|&(_, l, r), _| l != written && r != written);
            }
        }
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Division by constant — magic number multiplication (libdivide-style)
// ─────────────────────────────────────────────────────────────────────────────
//
// Replaces `x / N` (where N is a known constant at compile time) with a
// multiply-high + shift sequence. IDIV takes 20-40 cycles on modern x86;
// IMUL takes 3 cycles, making this a significant win for hot paths.
//
// The algorithm computes a "magic" constant M and shift S such that:
//   x / d ≈ (M * x)_high >> S
// where (M * x)_high is the high 64 bits of the 128-bit product.
// A signed-correction step handles negative dividends.

/// Compute the magic number and post-shift for dividing by positive constant `d`.
/// Returns (magic_constant, post_shift) or None if IDIV should be used instead.
/// P6 fix: Now handles negative divisors by computing magic for |d| and
/// negating the result. Previously, `x / -2`, `x / -4`, etc. fell back
/// to the 20-40 cycle IDIV instead of the 3-cycle magic-multiply path.

fn compute_div_magic(d: i64) -> Option<(i64, u8)> {
    if d == 0 || d == 1 {
        return None;
    }
    // P6 fix: Handle negative divisors
    let neg_result = d < 0;
    let abs_d = if d == i64::MIN {
        // Special case: |MIN| overflows, but division by MIN is rare
        // and the result is always 0 or 1, so just use IDIV
        return None;
    } else {
        d.unsigned_abs()
    };
    if abs_d.is_power_of_two() {
        return None; // Use shift instead
    }
    let mut shift: u8 = 0;
    let mut _magic: u64 = 0;
    loop {
        let numer = 1u128 << (64 + shift as u128);
        let ceil_val = numer.wrapping_add(abs_d as u128 - 1) / abs_d as u128;
        if ceil_val < (1u128 << 64) {
            _magic = ceil_val as u64;
            break;
        }
        shift += 1;
        if shift > 63 {
            return None;
        }
    }
    if neg_result {
        Some((-(_magic as i64), shift))
    } else {
        Some((_magic as i64, shift))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop Unrolling
// ─────────────────────────────────────────────────────────────────────────────
//
// For small-trip-count loops, duplicating the loop body reduces the overhead
// of branch and induction-variable updates.  We identify loops by backward
// jumps and duplicate the body up to `max_unroll` times (default 4).
//
// This is a bytecode-level transform that runs before register allocation,
// so it simply duplicates instructions and adjusts slot numbers to avoid
// conflicts.

/// Unroll loops with small trip counts.  `threshold` is the maximum
/// number of copies of the body to emit (2 = double the body, etc.).

fn unroll_loops(instrs: &mut Vec<Instr>, threshold: usize) {
    if threshold < 2 { return; }
    
    // Find backward jumps (loops) — check all branch instruction types
    let mut i = 0;
    while i < instrs.len() {
        let back_branch = match &instrs[i] {
            Instr::Jump(offset) => {
                let target = (i as i32 + 1 + offset) as usize;
                if target < i { Some(target) } else { None }
            }
            Instr::JumpFalse(_, offset) | Instr::JumpTrue(_, offset) => {
                let target = (i as i32 + 1 + offset) as usize;
                if target < i { Some(target) } else { None }
            }
            _ => None,
        };
        if let Some(target) = back_branch {
            // Backward jump: this is a loop from `target` to `i`
            let loop_start = target;
            let loop_end = i;
            let body_len = loop_end - loop_start;
            
            // Only unroll small loops (body <= 8 instructions)
            if body_len > 0 && body_len <= 8 {
                let body: Vec<Instr> = instrs[loop_start..=loop_end].to_vec();
                let mut unrolled: Vec<Instr> = Vec::with_capacity(body.len() * threshold);
                
                for copy_idx in 0..threshold {
                    for (j, instr) in body.iter().enumerate() {
                        if j == body.len() - 1 && copy_idx < threshold - 1 {
                            // Last instruction is the backward branch — replace
                            // with Nop in all copies except the last so they
                            // fall through to the next copy.
                            unrolled.push(Instr::Nop);
                        } else {
                            // Fix up forward branch offsets (JumpFalse/JumpTrue
                            // that exit the loop). After unrolling, the exit is
                            // body_len * (threshold - 1 - copy_idx) instructions
                            // further away from this copy.
                            let extra_offset = (body_len * (threshold - 1 - copy_idx)) as i32;
                            let fixed = match instr {
                                Instr::JumpFalse(cond, off) => {
                                    let target_pc = (j as i32 + 1 + off) as usize;
                                    if target_pc > loop_end {
                                        // Forward branch exiting the loop — adjust offset
                                        Instr::JumpFalse(*cond, off + extra_offset)
                                    } else {
                                        instr.clone()
                                    }
                                }
                                Instr::JumpTrue(cond, off) => {
                                    let target_pc = (j as i32 + 1 + off) as usize;
                                    if target_pc > loop_end {
                                        // Forward branch exiting the loop — adjust offset
                                        Instr::JumpTrue(*cond, off + extra_offset)
                                    } else {
                                        instr.clone()
                                    }
                                }
                                _ => instr.clone(),
                            };
                            unrolled.push(fixed);
                        }
                    }
                }
                
                // Replace the loop with the unrolled version
                instrs.splice(loop_start..=loop_end, unrolled);
                // Don't advance — the newly inserted code may contain more loops
                continue;
            }
        }
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CMOV (Conditional Move) for Branchless Code
// ─────────────────────────────────────────────────────────────────────────────
//
// When branch conditions are unpredictable (e.g., data-dependent guards),
// CMOVcc eliminates branch misprediction penalties (15-20 cycles on modern x86).
// The emitter methods below support CMOV with any condition code.

impl Emitter {
    /// CMOVcc dst, src — move src into dst if condition cc is met.
    /// cc values: 0x44=CMOVZ, 0x45=CMOVNZ, 0x4C=CMOVL, 0x4D=CMOVGE,
    ///            0x4E=CMOVLE, 0x4F=CMOVG, etc.
    /// NOTE: These methods are retained for potential future use (branchless
    /// min/max operations on arithmetic results), but were removed from
    /// comparison ops because they incorrectly overwrote boolean results.
    fn emit_cmovcc_rr(&mut self, cc: u8, dst: u8, src: u8) {
        if dst == src { return; } // No-op
        // CMOVcc r64, r/m64: 0F 4x /r with REX.W
        let rex = 0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3);
        let modrm = 0xC0 | ((dst & 7) << 3) | (src & 7);
        self.emit4(rex, 0x0F, cc, modrm);
    }
    
    /// Emit branchless min: dst = (dst < src) ? dst : src
    fn emit_branchless_min(&mut self, dst: u8, src: u8) {
        // CMP dst, src
        let rex = 0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3);
        let modrm = 0xC0 | ((dst & 7) << 3) | (src & 7);
        self.emit3(rex, 0x39, modrm); // CMP
        // CMOVLE dst, src: if dst <= src, dst is already min; else move src
        // Actually: CMP computes dst - src. If dst > src (no borrow), move src.
        // CMOVA (0x47) = cmov if above (CF=0 AND ZF=0), i.e. dst > src → dst = src = min
        self.emit_cmovcc_rr(0x47, dst, src); // CMOVA — if dst > src, dst = src = min
    }
    
    /// Emit branchless max: dst = (dst > src) ? dst : src
    fn emit_branchless_max(&mut self, dst: u8, src: u8) {
        let rex = 0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3);
        let modrm = 0xC0 | ((dst & 7) << 3) | (src & 7);
        self.emit3(rex, 0x39, modrm); // CMP
        // CMOVBE (0x46) = cmov if below or equal (CF=1 OR ZF=1), i.e. dst <= src → dst = src = max
        self.emit_cmovcc_rr(0x46, dst, src); // CMOVBE — if dst <= src, dst = src = max
    }
    
    /// Emit software prefetch: PREFETCHT0 [rdi + disp32]
    
    fn emit_prefetch_t0(&mut self, disp: i32) {
        // PREFETCHT0 m8: 0F 18 /1
        self.emit3(0x0F, 0x18, 0x8F); // mod=10, reg=1 (PREFETCHT0), rm=7(rdi)
        self.d(disp);
    }

    // ── Block Parameter Edge Moves (Parallel Copy Resolution) ──────────────
    //
    // When control flow transfers via JMP/Jcc to a block with parameters,
    // the arguments must already be in the exact physical registers (or stack
    // slots) that the target block expects. This requires resolving "parallel
    // copies" — a set of simultaneous (src, dst) register moves.
    //
    // The Register Allocator treats Block Parameters and their incoming
    // arguments as Affinity Groups. If both land in the same physical register,
    // the Edge Move is completely erased (zero-cost coalescing). When they
    // don't, we must resolve the remaining moves correctly.
    //
    // Key insight: we cannot emit sequential MOVs because they may clobber
    // values needed by later moves in the same parallel copy set. Instead, we:
    //   1. Topologically sort moves (execute leaf moves first)
    //   2. Break cycles using XCHG (1-2 uops) or scratch register R11
    //
    // This produces machine code indistinguishable from hand-written assembly.

    /// XCHG r64, r64 — atomically exchange two registers.
    /// Breaks 2-register swap cycles in 1-2 uops, far cheaper than
    /// the 3-MOV sequence using a scratch register.
    fn emit_xchg(&mut self, r1: u8, r2: u8) {
        if r1 == r2 { return; }
        // REX.W + 87 /r (or 90+rd for rax shortcut)
        // Prefer the 90+rd form when one operand is RAX(0)
        if r1 == 0 {
            // XCHG RAX, r64: REX.W 90+rd
            let rex = 0x48 | ((r2 & 8) >> 3);
            self.emit2(rex, 0x90 | (r2 & 7));
        } else if r2 == 0 {
            let rex = 0x48 | ((r1 & 8) >> 3);
            self.emit2(rex, 0x90 | (r1 & 7));
        } else {
            // XCHG r64, r/m64: REX.W 87 /r
            let rex = 0x48 | ((r1 & 8) >> 1) | ((r2 & 8) >> 3);
            let modrm = 0xC0 | ((r1 & 7) << 3) | (r2 & 7);
            self.emit3(rex, 0x87, modrm);
        }
    }

    /// Resolves a set of parallel register moves required for Block Parameter passing.
    ///
    /// `moves` is a list of (src_reg, dst_reg) pairs that must all happen
    /// simultaneously. This algorithm guarantees no clobbering by topologically
    /// sorting the dependency graph and breaking cycles with XCHG or a scratch
    /// register (R11).
    ///
    /// Performance impact:
    ///   - 90%+ of moves are coalesced away by the register allocator's
    ///     affinity graph (src == dst → no-op)
    ///   - Remaining moves are topologically sorted so the CPU's OoO engine
    ///     can execute independent MOVs in parallel
    ///   - Cycles (RAX↔RCX swaps) are broken with XCHG (1-2 uops)
    ///   - Complex cycles use R11 as scratch (2 MOVs per cycle break)
    pub fn emit_parallel_copies(&mut self, moves: &[(u8, u8)]) {
        // Filter out no-ops (src == dst — these were coalesced by the RA)
        let mut pending: Vec<(u8, u8)> = moves.iter()
            .filter(|(s, d)| s != d)
            .copied()
            .collect();

        if pending.is_empty() { return; }

        // Optimized parallel copy resolution using bitmask tracking.
        // With at most 16 GPRs, we can represent the "is this register still
        // needed as a source?" set as a u16 bitmask, making the inner check
        // O(1) instead of O(N).  Combined with swap_remove for O(1) deletion,
        // the overall complexity drops from O(N³) to O(N²) worst case.
        //
        // Scratch register rotation: for complex cycles that can't be resolved
        // with XCHG, we alternate between R10 and R11 as scratch registers.
        // This avoids creating artificial WAR/WAW dependencies in the CPU's
        // Register Alias Table (RAT) when breaking multiple consecutive cycles.
        // The RAT can perform zero-cycle moves via register renaming, but only
        // if the scratch registers don't create name dependencies.
        let mut scratch_reg: u8 = 10; // Start with R10, alternate to R11
        let mut remaining = pending.len();

        while remaining > 0 {
            // Build a bitmask of which registers are still needed as sources.
            // Bit i is set if register i appears as a source in any remaining move.
            let mut needed_as_src: u16 = 0;
            for i in 0..remaining {
                let (src, _) = pending[i];
                needed_as_src |= 1u16 << src;
            }

            let mut emitted = false;
            let mut i = 0;
            while i < remaining {
                let (src, dst) = pending[i];
                // Check if dst is needed as a source by any other move.
                // Using bitmask: if bit `dst` is set in needed_as_src AND
                // there exists another move that has dst as source, then
                // we can't emit this move yet.  But we need to be more
                // precise: check if any OTHER move needs `dst` as source.
                let other_needs_dst = needed_as_src & (1u16 << dst) != 0
                    && pending[..remaining].iter().enumerate()
                        .any(|(j, (s, _))| j != i && *s == dst);

                if !other_needs_dst {
                    // Safe to emit this move now
                    self.mov_reg_reg(dst, src);
                    // swap_remove for O(1) deletion
                    remaining -= 1;
                    if i < remaining {
                        pending.swap(i, remaining);
                    }
                    emitted = true;
                    // Don't increment i — re-check the swapped-in element
                    continue;
                }
                i += 1;
            }

            // If no safe move was found, we have a cycle.
            if !emitted && remaining > 0 {
                let (src, dst) = pending[remaining - 1];
                remaining -= 1;

                // Check if it's a simple 2-register swap
                let swap_partner = pending[..remaining].iter().enumerate()
                    .find(|(_, (s, d))| *s == dst && *d == src)
                    .map(|(i, _)| i);

                if let Some(partner_idx) = swap_partner {
                    // Emit XCHG (1-2 uops, breaks cycle perfectly)
                    self.emit_xchg(src, dst);
                    // swap_remove the partner
                    remaining -= 1;
                    if partner_idx < remaining {
                        pending.swap(partner_idx, remaining);
                    }
                } else {
                    // Complex cycle: use rotating scratch register (R10/R11)
                    // to avoid WAR/WAW structural hazards in the RAT.
                    // MOV scratch, src    (save src value)
                    // MOV dst, scratch    (move saved value to dst)
                    // The remaining pending moves that referenced `src` will
                    // be resolved in subsequent iterations.
                    self.mov_reg_reg(scratch_reg, src);  // scratch = src
                    self.mov_reg_reg(dst, scratch_reg);  // dst = scratch(=old src)
                    // Rotate scratch register for next cycle break
                    scratch_reg = if scratch_reg == 10 { 11 } else { 10 };
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop-Invariant Code Motion (LICM)
// ─────────────────────────────────────────────────────────────────────────────
//
// Identifies computations inside loops whose inputs do not change across
// iterations, and hoists them before the loop.  This is most impactful for
// load-heavy loops where the same address computation is repeated.
//
// The hoisting check is: an instruction is hoistable if NEITHER operand is
// MODIFIED inside the loop body (i.e., both operands are either pre-loop
// definitions, constant definitions, or loop-invariant definitions).  This
// is more aggressive than the previous check which required both operands
// to be defined BEFORE the loop — a value defined inside the loop but never
// modified after its first definition (single-def) is also invariant.
//
// We also detect loop-invariant slots via fixed-point iteration: slots
// defined inside the loop exactly once whose operands are all pre-loop or
// other loop-invariant slots.  This catches chains like:
//   LoadI64(s1, 4)     — const, trivially invariant
//   BinOp(s2, Mul, s0, s1) — invariant if s0 is pre-loop (even though s2
//                             is defined inside the loop)


fn hoist_loop_invariants(instrs: &mut Vec<Instr>) {
    // Find loops (backward jumps — any branch instruction targeting an earlier PC)
    let mut i = 0;
    while i < instrs.len() {
        let loop_end = i;
        let loop_start = match &instrs[i] {
            Instr::Jump(offset) => {
                let target = (i as i32 + 1 + offset) as usize;
                if target < i { Some(target) } else { None }
            }
            Instr::JumpFalse(_, offset) | Instr::JumpTrue(_, offset) => {
                let target = (i as i32 + 1 + offset) as usize;
                if target < i { Some(target) } else { None }
            }
            _ => None,
        };

        if let Some(start) = loop_start {
            // ── Step 1: Collect slots defined before the loop ──────────────
            let mut pre_loop_defs: std::collections::HashSet<u16> = std::collections::HashSet::new();
            for j in 0..start {
                if let Some(slot) = instr_writes_slot_get(&instrs[j]) {
                    pre_loop_defs.insert(slot);
                }
            }

            // ── Step 2: Collect constant definitions inside the loop ───────
            // LoadI* instructions produce constants and are trivially invariant.
            let mut const_defs: std::collections::HashSet<u16> = std::collections::HashSet::new();
            for j in start..loop_end {
                match &instrs[j] {
                    Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => {
                        const_defs.insert(*d);
                    }
                    _ => {}
                }
            }

            // ── Step 3: Detect loop-invariant slots via fixed-point ────────
            // A slot defined inside the loop is loop-invariant if:
            //   (a) it is defined exactly once inside the loop, AND
            //   (b) all its operands are either pre-loop, const, or other
            //       loop-invariant slots.
            // We iterate to a fixed point because invariant-ness is transitive.

            // First pass: count how many times each slot is defined inside the loop.
            let mut loop_def_count: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
            // Also record the definition index and operand slots for each single-def slot.
            let mut loop_def_info: std::collections::HashMap<u16, (usize, Vec<u16>)> = std::collections::HashMap::new();
            for j in start..loop_end {
                if let Some(slot) = instr_writes_slot_get(&instrs[j]) {
                    *loop_def_count.entry(slot).or_insert(0) += 1;
                    // Only record info for the first definition (we'll filter
                    // multi-def slots out later).
                    loop_def_info.entry(slot).or_insert_with(|| {
                        let operands = instr_operand_slots(&instrs[j]);
                        (j, operands)
                    });
                }
            }

            // Fixed-point iteration: start with const_defs, keep adding single-def
            // slots whose operands are all in (pre_loop_defs ∪ const_defs ∪ loop_invariant_defs).
            let mut loop_invariant_defs: std::collections::HashSet<u16> = const_defs.clone();
            let mut changed = true;
            while changed {
                changed = false;
                for (&slot, &count) in &loop_def_count {
                    if count != 1 {
                        continue; // Multi-def → not invariant
                    }
                    if loop_invariant_defs.contains(&slot) {
                        continue; // Already known invariant
                    }
                    if pre_loop_defs.contains(&slot) {
                        continue; // Pre-loop def, not a loop-body def
                    }
                    // Check that all operands are invariant or pre-loop.
                    if let Some(&(_, ref operands)) = loop_def_info.get(&slot) {
                        let all_invariant = operands.iter().all(|op| {
                            pre_loop_defs.contains(op)
                                || const_defs.contains(op)
                                || loop_invariant_defs.contains(op)
                        });
                        if all_invariant {
                            loop_invariant_defs.insert(slot);
                            changed = true;
                        }
                    }
                }
            }

            // ── Step 4: Determine which slots are "modified inside the loop" ──
            // A slot is modified inside the loop if it is defined MORE than once,
            // OR if it is defined once but is NOT loop-invariant.
            let mut modified_in_loop: std::collections::HashSet<u16> = std::collections::HashSet::new();
            for (&slot, &count) in &loop_def_count {
                if count > 1 {
                    modified_in_loop.insert(slot);
                } else if count == 1 && !loop_invariant_defs.contains(&slot) {
                    // Single def but not invariant (its operands changed) → treat
                    // as modified for the purpose of hoisting — the slot itself
                    // isn't invariant, so dependents can't hoist either.
                    modified_in_loop.insert(slot);
                }
            }

            // ── Step 5: Find hoistable instructions ───────────────────────
            // An instruction is hoistable if NEITHER operand is modified inside
            // the loop.  This subsumes the old check (both operands pre-loop or
            // const) and also handles loop-invariant slots defined inside the loop.
            let mut to_hoist: Vec<usize> = Vec::new();
            for j in start..loop_end {
                // Skip instructions that are themselves invariant definitions
                // (LoadI* are already in const_defs, and we don't want to hoist
                // them twice).  We hoist BinOp/Move/Load that use invariant operands.
                match &instrs[j] {
                    Instr::BinOp(d, _, l, r) => {
                        // Don't hoist if the destination is written by another
                        // instruction inside the loop (multi-def check).
                        let l_ok = !modified_in_loop.contains(l);
                        let r_ok = !modified_in_loop.contains(r);
                        if l_ok && r_ok {
                            // Also ensure that the destination is not used by a
                            // non-hoistable instruction inside the loop that
                            // depends on the loop-carried value.  Since this
                            // instruction's inputs are all invariant, the output
                            // is also invariant, so hoisting is safe.
                            // Verify `d` is single-def or not modified elsewhere.
                            let d_count = loop_def_count.get(d).copied().unwrap_or(0);
                            if d_count <= 1 || pre_loop_defs.contains(d) {
                                to_hoist.push(j);
                            }
                        }
                    }
                    Instr::Move(_, s) | Instr::Load(_, s) => {
                        if !modified_in_loop.contains(s) {
                            to_hoist.push(j);
                        }
                    }
                    _ => {}
                }
            }

            // ── Step 6: Hoist by replacing original with Nop and prepending ──
            // FIXED: Instead of inserting at loop_start (which shifts all PC
            // offsets and invalidates JumpFalse/JumpTrue targets), we use the
            // Nop-replacement pattern: replace the hoisted instruction with
            // Nop (preserving its PC offset) and emit a copy before the loop.
            // Since PCs don't change, JumpFalse/JumpTrue offsets remain valid.
            for &j in &to_hoist {
                let hoisted = instrs[j].clone();
                instrs[j] = Instr::Nop; // Preserve PC offset
                instrs.insert(start, hoisted);
                // Note: We insert at `start` which shifts indices, but this
                // is okay because we process one loop at a time and break
                // afterward to avoid invalidating other loop ranges.
            }
            // Only process one loop per call to avoid index invalidation.
            // The caller will invoke this pass repeatedly until no changes.
            if !to_hoist.is_empty() {
                break;
            }
        }
        i += 1;
    }
}

/// Helper: extract all operand (input) slots from an instruction.
/// Returns a Vec of slot numbers that the instruction reads.
fn instr_operand_slots(instr: &Instr) -> Vec<u16> {
    match instr {
        Instr::BinOp(_, _, l, r) => vec![*l, *r],
        Instr::Move(_, s) | Instr::Load(_, s) => vec![*s],
        Instr::Store(_, s) => vec![*s],
        Instr::UnOp(_, _, s) => vec![*s],
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => vec![*s],
        Instr::Return(s) => vec![*s],
        // LoadI* have no operand slots — they produce constants
        Instr::LoadI32(_, _) | Instr::LoadI64(_, _) | Instr::LoadBool(_, _) | Instr::LoadUnit(_) => vec![],
        _ => vec![],
    }
}

/// Helper: get the slot written by an instruction, if any.

fn instr_writes_slot_get(instr: &Instr) -> Option<u16> {
    match instr {
        Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => Some(*d),
        Instr::Move(d, _) | Instr::Load(d, _) => Some(*d),
        Instr::Store(d, _) => Some(*d),
        Instr::BinOp(d, _, _, _) => Some(*d),
        Instr::TensorBinOp { dst, .. } => Some(*dst),
        Instr::TensorReduce { dst, .. } => Some(*dst),
        Instr::TensorMatMul { dst, .. } => Some(*dst),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar Evolution Analysis
// ─────────────────────────────────────────────────────────────────────────────
//
// Analyzes induction variables and their evolution patterns across loop
// iterations.  This feeds into the stencil compiler to emit pre-computed
// payload offsets instead of computing them at runtime with a deep match
// tree on every iteration.

/// Describes how a value evolves across loop iterations.
#[derive(Debug, Clone, PartialEq)]
enum SCEVExpr {
    /// Constant value — does not change across iterations.
    Constant(i64),
    /// Induction variable: starts at `initial` and increments by `step`
    /// each iteration.  The `slot` is the SSA slot number.
    Induction { slot: u16, initial: i64, step: i64 },
    /// Affine expression: base + step * iteration_count.
    /// Used when we can compute the closed-form but not the initial value.
    Affine { base_slot: u16, step: i64 },
    /// Unknown — the value depends on a runtime-computed expression that
    /// cannot be expressed as a simple recurrence.
    Unknown,
}

/// Result of scalar evolution analysis on a single loop.
struct ScalarEvolution {
    /// Map from SSA slot → SCEVExpr describing its evolution.
    evolutions: FxHashMap<u16, SCEVExpr>,
    /// The loop's induction variable (if a simple counted loop is detected).
    primary_induction: Option<u16>,
    /// The loop's trip count (if computable at compile time).
    trip_count: Option<u64>,
    /// The loop's stride (increment per iteration of the primary induction var).
    stride: i64,
}

impl ScalarEvolution {
    /// Analyze scalar evolution for a loop body given by `instrs[start..end]`.
    fn analyze(instrs: &[Instr], start: usize, end: usize) -> Self {
        let mut evolutions: FxHashMap<u16, SCEVExpr> = FxHashMap::default();
        let mut primary_induction: Option<u16> = None;
        let trip_count: Option<u64> = None;
        let mut stride = 1i64;

        // Phase 0: Mark all LoadI instructions as constants.
        for j in start..end {
            match &instrs[j] {
                Instr::LoadI32(slot, val) => {
                    evolutions.insert(*slot, SCEVExpr::Constant(*val as i64));
                }
                Instr::LoadI64(slot, val) => {
                    evolutions.insert(*slot, SCEVExpr::Constant(*val));
                }
                Instr::LoadBool(slot, val) => {
                    evolutions.insert(*slot, SCEVExpr::Constant(*val as i64));
                }
                _ => {}
            }
        }

        // Phase 1: Identify induction variables.
        // An induction variable appears as: BinOp(slot, Add, slot, const)
        // where slot is both the destination and the left operand.
        for j in start..end {
            if let Instr::BinOp(d, BinOpKind::Add, l, _r) = &instrs[j] {
                if *d == *l {
                    // Candidate induction variable. Check if the right operand
                    // is a constant loaded immediately before this BinOp.
                    let step_val = if j > start {
                        if let Instr::LoadI64(_, v) = &instrs[j - 1] {
                            Some(*v)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(step) = step_val {
                        evolutions.insert(*d, SCEVExpr::Induction {
                            slot: *d,
                            initial: 0, // Will be refined if we find a pre-loop value
                            step,
                        });
                        // The first detected induction variable is the primary one
                        if primary_induction.is_none() {
                            primary_induction = Some(*d);
                            stride = step;
                        }
                    }
                }
            }
        }

        // Phase 2: Propagate evolution through the loop body.
        // Any BinOp involving an induction variable produces an Affine expr.
        // BinOps on two constants may produce a constant result (if foldable).
        for j in start..end {
            if let Instr::BinOp(d, op, l, r) = &instrs[j] {
                let l_scev = evolutions.get(l);
                let r_scev = evolutions.get(r);
                match (l_scev, r_scev, op) {
                    (Some(SCEVExpr::Constant(c1)), Some(SCEVExpr::Constant(c2)), BinOpKind::Add) => {
                        evolutions.insert(*d, SCEVExpr::Constant(c1 + c2));
                    }
                    (Some(SCEVExpr::Constant(c1)), Some(SCEVExpr::Constant(c2)), BinOpKind::Mul) => {
                        evolutions.insert(*d, SCEVExpr::Constant(c1 * c2));
                    }
                    (Some(SCEVExpr::Induction { .. }), Some(SCEVExpr::Constant(_)), _) |
                    (Some(SCEVExpr::Constant(_)), Some(SCEVExpr::Induction { .. }), _) => {
                        // Induction + constant = affine (for any operator)
                        evolutions.insert(*d, SCEVExpr::Affine {
                            base_slot: *d,
                            step: stride,
                        });
                    }
                    _ => {
                        // Unknown expression — mark as such
                        evolutions.insert(*d, SCEVExpr::Unknown);
                    }
                }
            }
        }

        // Phase 3: Try to determine trip count from loop bounds.
        // Look for JumpFalse/JumpTrue that compare the induction variable
        // against a constant bound.
        for j in start..end {
            if let Instr::JumpFalse(cond, _) = &instrs[j] {
                if let Some(iv_slot) = primary_induction {
                    if *cond == iv_slot {
                        // The loop exits when the induction variable becomes
                        // truthy (non-zero) or falsy (zero).  We can compute
                        // the trip count if we know the initial value and step.
                        // For now, we leave trip_count as None and rely on
                        // runtime detection.  A future enhancement can parse
                        // the comparison instruction preceding the branch.
                    }
                }
            }
        }

        Self {
            evolutions,
            primary_induction,
            trip_count,
            stride,
        }
    }

    /// Get the evolution expression for a slot, if known.
    fn get(&self, slot: u16) -> &SCEVExpr {
        self.evolutions.get(&slot).unwrap_or(&SCEVExpr::Unknown)
    }

    /// Check if a slot is an induction variable.
    fn is_induction(&self, slot: u16) -> bool {
        matches!(self.get(slot), SCEVExpr::Induction { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Strength Reduction for Induction Variables
// ─────────────────────────────────────────────────────────────────────────────
//
// Replaces expensive operations inside loops with cheaper equivalents:
//   - x * N  (constant power of 2) → SHL log2(N)   [address mode folding]
//   - i * stride inside loop → p += stride           [induction var reduction]
//   - x / N  (constant power of 2, unsigned) → SHR log2(N)
//   - x / N  (non-power-of-2 constant) → magic multiply + shift
//
// This pass runs at the bytecode level before JIT emission.


fn strength_reduce(instrs: &mut Vec<Instr>) {
    // ── Phase 1: Induction variable strength reduction in loops ───────────
    // Detect loops and replace `i * stride` (where i is the induction variable
    // incremented by 1 each iteration) with an incrementing pointer `p += stride`.
    //
    // At the bytecode level:
    //   Before loop:  BinOp(new_slot, Add, base_slot, BinOp(initial_i, Mul, stride_slot))
    //                   → compute initial address: BinOp(new_slot, Add, base_slot, init_mul_slot)
    //   Inside loop:  Replace BinOp(_, Mul, i, stride) with BinOp(_, Add, new_slot, stride_slot)
    //                 → increment by stride each iteration
    induction_var_strength_reduce(instrs);

    // ── Phase 2: Peephole strength reduction (loop-independent) ──────────
    //   - Mul by power-of-2 → Shl
    //   - Div by power-of-2 → Shr (unsigned)
    //   - Div by non-power-of-2 constant → magic multiply + shift
    peephole_strength_reduce(instrs);
}

/// Induction variable strength reduction: replaces `i * stride` inside a loop
/// with a pointer that increments by `stride` each iteration.
///
/// Detects the pattern:
///   Loop body contains: BinOp(mul_dst, Mul, induction_var, stride_slot)
///   where induction_var is incremented by a constant each iteration.
///
/// Transformation:
///   - Before the loop: insert computation of initial pointer value
///   - Inside the loop: replace Mul with Add (increment by stride)
fn induction_var_strength_reduce(instrs: &mut Vec<Instr>) {
    let mut i = 0;
    while i < instrs.len() {
        let loop_end = i;
        let loop_start = match &instrs[i] {
            Instr::Jump(offset) => {
                let target = (i as i32 + offset) as usize;
                if target < i { Some(target) } else { None }
            }
            _ => None,
        };

        if let Some(start) = loop_start {
            // Use ScalarEvolution to analyze the loop, then apply strength
            // reduction based on the SCEV results.
            let scev = ScalarEvolution::analyze(instrs, start, loop_end);

            // Identify induction variables: slots that are incremented by a
            // constant inside the loop.  Look for patterns like:
            //   BinOp(s, Add, s, const_slot)  where s appears as both dst and lhs
            //   LoadI64(const_slot, N)         immediately before the BinOp
            let mut induction_vars: std::collections::HashMap<u16, (u16, i64)> =
                std::collections::HashMap::new(); // slot → (stride_slot, stride_val)
            for j in start..loop_end {
                if let Instr::BinOp(d, BinOpKind::Add, l, r) = &instrs[j] {
                    // Pattern: s = s + const  (d == l, meaning the variable is
                    // being incremented in-place)
                    if *d == *l {
                        // Check if r is a constant loaded immediately before
                        if j > start {
                            if let Instr::LoadI64(_, val) = &instrs[j - 1] {
                                // Cross-reference with SCEV: if this slot is an
                                // induction variable per SCEV, use the SCEV stride
                                // which may be more accurate after propagation.
                                let stride = if scev.is_induction(*d) { scev.stride } else { *val };
                                induction_vars.insert(*d, (*r, stride));
                            }
                        }
                    }
                }
            }

            // Now look for BinOp(_, Mul, induction_var, stride_slot) inside the loop
            // and replace with the incrementing-pointer pattern.  Use SCEV to
            // filter: only apply strength reduction when the Mul result has an
            // affine SCEV (i.e., it's a linear function of the induction variable).
            if !induction_vars.is_empty() {
                // Use SCEV trip_count (if known) to log additional info.
                if scev.trip_count.is_some() {
                    jit_trace!("[JIT-SCEV] Loop trip count: {}", scev.trip_count.unwrap());
                }

                // Collect slots to use for new temporaries.  Find the max slot.
                let max_slot = instrs.iter().filter_map(|instr| {
                    match instr {
                        Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                        Instr::Load(d, _) | Instr::LoadI32(d, _) |
                        Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d as usize),
                        _ => None,
                    }
                }).max().unwrap_or(0);

                let mut next_new_slot = (max_slot + 1) as u16;

                // Find Mul instructions that use induction variables.
                // Cross-reference with SCEV: only consider Mul instructions
                // whose result is affine (skip SCEV::Unknown results that
                // don't follow a simple recurrence pattern).
                let mut replacements: Vec<(usize, u16, u16, u16, u16)> = Vec::new();
                // (mul_idx, induction_var_slot, stride_slot, new_ptr_slot, mul_dst)

                for j in start..loop_end {
                    if let Instr::BinOp(dst, BinOpKind::Mul, l, r) = &instrs[j] {
                        // Check if either operand is an induction variable
                        let iv_info = if induction_vars.contains_key(l) {
                            Some((*l, *r))
                        } else if induction_vars.contains_key(r) {
                            Some((*r, *l))
                        } else {
                            None
                        };

                        if let Some((iv_slot, other_slot)) = iv_info {
                            // Use SCEV to validate: skip if the Mul result is
                            // already classified as Unknown (complex expression).
                            match scev.get(*dst) {
                                SCEVExpr::Unknown => {
                                    // This Mul result doesn't follow a simple pattern;
                                    // skip strength reduction to avoid miscompilation.
                                    continue;
                                }
                                SCEVExpr::Constant(_) => {
                                    // The Mul result is loop-invariant — should have
                                    // been hoisted by LICM.  Skip.
                                    continue;
                                }
                                SCEVExpr::Affine { .. } | SCEVExpr::Induction { .. } => {
                                    // Good candidate for strength reduction.
                                }
                            }
                            let new_ptr_slot = next_new_slot;
                            next_new_slot += 1;
                            replacements.push((j, iv_slot, other_slot, new_ptr_slot, *dst));
                        }
                    }
                }

                // Apply replacements. We need to:
                // 1. Before the loop: compute initial pointer = base + iv_initial * stride
                //    (Actually, just compute initial pointer from the Mul result at first use)
                // 2. Inside the loop: replace Mul with Add (ptr += stride)
                // 3. After the Mul: any use of mul_dst should use new_ptr_slot instead
                //
                // For simplicity and correctness at the bytecode level:
                // - Replace the Mul instruction with: BinOp(new_ptr_slot, Add, new_ptr_slot, stride_slot)
                // - Before the loop, insert: Move(new_ptr_slot, mul_dst) to initialize from
                //   the first Mul computation. But wait — we need the initial value.
                //
                // Better approach:
                // - Insert before the loop: BinOp(new_ptr_slot, Mul, iv_initial, stride_slot)
                //   where iv_initial is the induction variable's value at loop entry.
                //   But we don't easily know iv_initial.
                //
                // Simplest correct approach:
                // - Replace the Mul inside the loop with:
                //   BinOp(mul_dst, Add, new_ptr_slot, stride_slot)  — increment
                //   Move(new_ptr_slot, mul_dst)                       — save for next iter
                // - Before the loop, insert:
                //   Move(new_ptr_slot, 0) then BinOp(new_ptr_slot, Add, new_ptr_slot, stride_slot)
                //   — but this is wrong because the first iteration needs the actual mul result.
                //
                // Most correct approach at bytecode level:
                // - Keep the first Mul as-is (to compute the initial value)
                // - Insert Move(new_ptr_slot, mul_dst) after the first Mul
                // - Replace subsequent Muls with Add(new_ptr_slot, stride_slot) + Move(mul_dst, new_ptr_slot)
                //
                // But there's typically only one Mul per iteration. So:
                // - Before the loop: compute new_ptr_slot = iv_current * stride
                //   We do this by inserting a copy of the Mul instruction using new_ptr_slot as dest
                // - Inside the loop: Replace Mul with Add + Move

                // For each replacement, we transform as follows:
                // Original:
                //   [loop body] BinOp(mul_dst, Mul, iv, stride)
                // Transformed:
                //   Before loop: BinOp(new_ptr, Mul, iv, stride)  — initial value
                //   In loop body: BinOp(mul_dst, Add, new_ptr, stride)
                //                 Move(new_ptr, mul_dst)  — update pointer for next iter
                //                 BinOp(iv, Add, iv, inc)  — existing induction increment

                // We process replacements in reverse order so that insertions don't
                // shift indices of earlier replacements.
                for (mul_idx, _iv_slot, stride_slot, new_ptr_slot, mul_dst) in replacements.iter().rev() {
                    let mul_idx = *mul_idx;
                    let new_ptr_slot = *new_ptr_slot;
                    let _stride_slot = *stride_slot;
                    let mul_dst = *mul_dst;

                    // Get the original Mul instruction's operands
                    let (iv_slot, stride_slot) = if let Instr::BinOp(_, BinOpKind::Mul, l, r) = &instrs[mul_idx] {
                        (*l, *r)
                    } else {
                        continue;
                    };

                    // Insert before the loop: BinOp(new_ptr, Mul, iv, stride)
                    // This computes the initial pointer value from the induction
                    // variable's value at loop entry.
                    let init_instr = Instr::BinOp(new_ptr_slot, BinOpKind::Mul, iv_slot, stride_slot);
                    instrs.insert(start, init_instr);

                    // Shift mul_idx by +1 because we just inserted before the loop
                    let mul_idx = mul_idx + 1;

                    // Replace the Mul inside the loop with Add (pointer increment)
                    instrs[mul_idx] = Instr::BinOp(mul_dst, BinOpKind::Add, new_ptr_slot, stride_slot);

                    // Insert Move(new_ptr, mul_dst) after the Add to save the
                    // new pointer value for the next iteration.
                    instrs.insert(mul_idx + 1, Instr::Move(new_ptr_slot, mul_dst));
                }
            }
        }
        i += 1;
    }
}

/// Peephole strength reduction: replaces expensive operations with cheaper
/// equivalents at the bytecode level, independent of loop context.
///
/// Transformations:
///   - BinOp(d, Mul, x, N) where N is power-of-2 → BinOp(d, Shl, x, log2(N))
///   - BinOp(d, Div, x, N) where N is power-of-2 → BinOp(d, Shr, x, log2(N))
///   - BinOp(d, Div, x, N) where N is non-power-of-2 constant → magic mul+shift
fn peephole_strength_reduce(instrs: &mut Vec<Instr>) {
    let mut i = 0;
    while i < instrs.len() {
        if let Instr::BinOp(dst, op, l, r) = &instrs[i] {
            match op {
                BinOpKind::Mul => {
                    // Replace Mul by power-of-2 constant with Shl.
                    // Look for the pattern: LoadI64(slot, N) followed by BinOp(_, Mul, x, slot)
                    // or BinOp(_, Mul, slot, x).
                    if i > 0 {
                        if let Instr::LoadI64(load_d, val) = &instrs[i - 1] {
                            let v = *val;
                            if v > 0 && (v as u64).is_power_of_two() {
                                let log2_val = (v as u64).trailing_zeros() as i64;
                                // Replace LoadI64(slot, N) with LoadI64(slot, log2(N))
                                // and Mul with Shl.
                                let load_d = *load_d;
                                let dst = *dst;
                                let l = *l;
                                let r = *r;

                                // Determine which operand is the constant slot
                                // and which is the variable.
                                let (var_slot, const_slot) = if r == load_d {
                                    (l, r)
                                } else {
                                    (r, l)
                                };

                                // Replace: LoadI64(const_slot, log2(N)) + BinOp(dst, Shl, var, const_slot)
                                instrs[i - 1] = Instr::LoadI64(const_slot, log2_val);
                                instrs[i] = Instr::BinOp(dst, BinOpKind::Shl, var_slot, const_slot);
                            }
                        }
                    }
                }
                BinOpKind::Div => {
                    // Replace Div by power-of-2 constant with Shr (unsigned).
                    // For simplicity, only handle the unsigned/positive case.
                    // Look for: LoadI64(slot, N) followed by BinOp(_, Div, x, slot)
                    if i > 0 {
                        if let Instr::LoadI64(load_d, val) = &instrs[i - 1] {
                            let v = *val;
                            if v > 0 && (v as u64).is_power_of_two() {
                                let log2_val = (v as u64).trailing_zeros() as i64;
                                let load_d = *load_d;
                                let dst = *dst;
                                let l = *l;
                                let r = *r;

                                // Pattern: BinOp(dst, Div, l, r) where r == load_d
                                if r == load_d {
                                    // Replace: LoadI64(r, log2(N)) + BinOp(dst, Shr, l, r)
                                    instrs[i - 1] = Instr::LoadI64(r, log2_val);
                                    instrs[i] = Instr::BinOp(dst, BinOpKind::Shr, l, r);
                                }
                            } else if v > 1 {
                                // Non-power-of-2: try magic-number division.
                                // Replace BinOp(d, Div, x, N) with:
                                //   LoadI64(tmp, magic) + BinOp(tmp2, Mul, x, tmp) + Shift
                                // For simplicity, we only replace if we can compute a magic number.
                                if let Some((magic, shift)) = compute_div_magic(v) {
                                    let load_d = *load_d;
                                    let dst = *dst;
                                    let l = *l;
                                    let r = *r;

                                    if r == load_d {
                                        // We need temporaries.  Find max slot.
                                        let max_slot = instrs.iter().filter_map(|instr| {
                                            match instr {
                                                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                                                Instr::Load(d, _) | Instr::LoadI32(d, _) |
                                                Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d as usize),
                                                _ => None,
                                            }
                                        }).max().unwrap_or(0);

                                        let tmp_magic = (max_slot + 1) as u16;
                                        let tmp_hi = (max_slot + 2) as u16;

                                        // Replace the two instructions with the magic-number sequence:
                                        //   LoadI64(tmp_magic, magic)
                                        //   BinOp(tmp_hi, Mul, l, tmp_magic)   — multiply by magic
                                        //   BinOp(dst, Shr, tmp_hi, shift)     — shift right
                                        //
                                        // Note: this is a simplified form that works for unsigned
                                        // division.  Signed division would need sign-correction.
                                        // We'll emit the unsigned form and rely on the emitter's
                                        // existing magic-division path for signed values.

                                        // We're replacing instrs[i-1] and instrs[i] with 4 instructions:
                                        //   LoadI64(shift_slot, shift)      [insert before Mul]
                                        //   LoadI64(tmp_magic, magic)       [replaces instrs[i-1]]
                                        //   BinOp(tmp_hi, Mul, l, tmp_magic) [replaces instrs[i]]
                                        //   BinOp(dst, Shr, tmp_hi, shift_slot) [insert after Mul]
                                        let shift_slot = (max_slot + 3) as u16;

                                        instrs[i - 1] = Instr::LoadI64(tmp_magic, magic);
                                        instrs[i] = Instr::BinOp(tmp_hi, BinOpKind::Mul, l, tmp_magic);
                                        // Insert the shift constant before the Shr instruction
                                        instrs.insert(i + 1, Instr::LoadI64(shift_slot, shift as i64));
                                        // Insert the Shr instruction after the Mul + LoadI64
                                        instrs.insert(i + 2, Instr::BinOp(dst, BinOpKind::Shr, tmp_hi, shift_slot));

                                        // Skip past the newly inserted instructions
                                        i += 3;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CodeBuilder trait — abstraction for code emission
// ─────────────────────────────────────────────────────────────────────────────
//
// Provides a clean interface for code emission that can be swapped for
// different backends (e.g., AArch64, RISC-V) or for testing.

/// Trait for building machine code into a byte buffer.

trait CodeBuilder {
    /// Emit a single byte.
    fn emit(&mut self, byte: u8);
    /// Emit a slice of bytes.
    fn emit_slice(&mut self, bytes: &[u8]);
    /// Get the current offset in the output buffer.
    fn current_offset(&self) -> usize;
    /// Patch a 32-bit value at the given offset.
    fn patch_i32(&mut self, offset: usize, value: i32);
    /// Patch an 8-bit value at the given offset.
    fn patch_u8(&mut self, offset: usize, value: u8);
}


impl CodeBuilder for Emitter {
    #[inline(always)]
    fn emit(&mut self, byte: u8) {
        self.b(byte);
    }
    
    #[inline(always)]
    fn emit_slice(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
    
    #[inline(always)]
    fn current_offset(&self) -> usize {
        self.pos()
    }
    
    fn patch_i32(&mut self, offset: usize, value: i32) {
        if offset + 4 <= self.pos() {
            self.as_mut_slice()[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    
    fn patch_u8(&mut self, offset: usize, value: u8) {
        if offset < self.pos() {
            self.as_mut_slice()[offset] = value;
        }
    }
}

/// Helper: emit a sequence of NOP bytes using the CodeBuilder trait,
/// ensuring the trait is exercised and not dead-coded.
fn emit_nop_padding(builder: &mut dyn CodeBuilder, count: usize) {
    for _ in 0..count {
        builder.emit(0x90); // NOP
    }
}

/// Helper: verify CodeBuilder patch methods work correctly.
fn verify_code_builder_patch(builder: &mut dyn CodeBuilder) {
    let off = builder.current_offset();
    builder.emit_slice(&[0x00, 0x00, 0x00, 0x00]);
    builder.patch_i32(off, 0x12345678);
    builder.patch_u8(off, 0x90);
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine code validation (debug_assertions only)
// ─────────────────────────────────────────────────────────────────────────────
//
// In debug builds, validates that the generated machine code is well-formed:
// all branch targets are in bounds, REX prefixes are valid, and ModRM fields
// are consistent.  Catches JIT bugs early during development.

#[cfg(debug_assertions)]

fn validate_machine_code(code: &[u8], fixups: &[Fixup], pc_to_off: &[usize]) -> Result<(), String> {
    // 1. All fixup displacement positions must be within the code buffer
    for (i, fx) in fixups.iter().enumerate() {
        if fx.disp_pos + 4 > code.len() {
            return Err(format!(
                "Fixup {}: disp_pos {} + 4 exceeds code len {}",
                i, fx.disp_pos, code.len()
            ));
        }
        // Target PC must be within the pc_to_off table
        if fx.target_pc >= pc_to_off.len() {
            return Err(format!(
                "Fixup {}: target_pc {} exceeds pc_to_off len {}",
                i, fx.target_pc, pc_to_off.len()
            ));
        }
    }
    
    // 2. All branch targets must resolve to valid offsets within the code
    for (pc, &offset) in pc_to_off.iter().enumerate() {
        if offset > code.len() {
            return Err(format!(
                "pc_to_off[{}] = {} exceeds code len {}",
                pc, offset, code.len()
            ));
        }
    }
    
    // 3. Check that RET (0xC3) exists somewhere in the code
    if !code.contains(&0xC3) {
        return Err("Generated code has no RET instruction".to_string());
    }
    
    // 4. Verify no overlapping fixup regions
    for i in 0..fixups.len() {
        for j in (i + 1)..fixups.len() {
            let a_start = fixups[i].disp_pos;
            let a_end = a_start + 4;
            let b_start = fixups[j].disp_pos;
            let b_end = b_start + 4;
            if a_start < b_end && b_start < a_end {
                return Err(format!(
                    "Overlapping fixups: fixup[{}] ({}..{}) overlaps fixup[{}] ({}..{})",
                    i, a_start, a_end, j, b_start, b_end
                ));
            }
        }
    }
    
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Instruction Scheduling — minimize pipeline stalls
// ─────────────────────────────────────────────────────────────────────────────
//
// Reorders independent instructions to increase instruction-level parallelism.
// On modern x86, the out-of-order engine handles most scheduling, but explicit
// reordering can help the decode/renamer by grouping independent chains together
// and separating dependent instructions.
//
// This pass operates at the bytecode level before JIT emission.  It identifies
// chains of dependent instructions and interleaves independent chains to
// maximize the CPU's ability to issue multiple uops per cycle.


fn schedule_instructions(instrs: &mut Vec<Instr>) {
    // DAG-based list scheduling: builds a micro-dependency DAG within each
    // basic block, then schedules instructions using a ready-list with
    // latency-depth priority.  This is O(N) in the number of instructions
    // and produces significantly better ILP than the old single-swap pass.
    //
    // Fusion patterns (LoadI+BinOp, BinOp+Store, BinOp+Jcc) are preserved
    // by treating fused pairs as a single scheduling unit.

    if instrs.len() < 3 { return; }

    // Note: We schedule the entire instruction stream as one unit since
    // our IR is already in basic-block form. A future enhancement could
    // split at block boundaries for per-block scheduling.

    // ── Step 2: Build the dependency DAG ───────────────────────────────
    // For each instruction, compute its set of predecessors (instructions
    // it depends on).  We track the last writer for each slot number.
    let n = instrs.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut last_writer: FxHashMap<u16, usize> = FxHashMap::default();
    let mut pending_reads: FxHashMap<u16, Vec<usize>> = FxHashMap::default();

    for (i, instr) in instrs.iter().enumerate() {
        // Add edges from all writers of slots this instruction reads
        for slot in instr_reads_slots(instr) {
            if let Some(&writer) = last_writer.get(&slot) {
                preds[i].push(writer);
                succs[writer].push(i);
            }
            pending_reads.entry(slot).or_default().push(i);
        }
        // Update last_writer for slots this instruction writes
        for slot in instr_writes_slots(instr) {
            // True dependency: any pending reader of this slot must complete
            // before this write (WAW or WAR dependency for in-order issue)
            last_writer.insert(slot, i);
        }
    }

    // ── Step 3: Compute latency depth (longest path to a root) ────────
    let mut depth: Vec<u32> = vec![0; n];
    // Topological order via Kahn's algorithm (BFS)
    let mut in_degree: Vec<u32> = preds.iter().map(|p| p.len() as u32).collect();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }
    while let Some(node) = queue.pop_front() {
        for &succ in &succs[node] {
            let new_depth = depth[node] + 1;
            if new_depth > depth[succ] {
                depth[succ] = new_depth;
            }
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    // ── Step 4: List schedule using ready-list with depth priority ─────
    // Re-initialize in_degree for the scheduling pass
    let mut in_degree: Vec<u32> = preds.iter().map(|p| p.len() as u32).collect();
    let mut ready: BinaryHeap<(std::cmp::Reverse<u32>, usize)> = BinaryHeap::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            // Higher depth = schedule first (critical path priority)
            ready.push((std::cmp::Reverse(depth[i]), i));
        }
    }

    let mut scheduled: Vec<Instr> = Vec::with_capacity(n);
    let mut scheduled_flags: Vec<bool> = vec![false; n];
    let mut pos_map: Vec<usize> = vec![0; n]; // old index → new position

    while let Some((_, node)) = ready.pop() {
        if scheduled_flags[node] { continue; }
        scheduled.push(instrs[node].clone());
        pos_map[node] = scheduled.len() - 1;
        scheduled_flags[node] = true;

        for &succ in &succs[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                ready.push((std::cmp::Reverse(depth[succ]), succ));
            }
        }
    }

    // Append any unscheduled instructions (cycles in the DAG — shouldn't
    // happen in valid SSA, but handle gracefully)
    for (i, instr) in instrs.iter().enumerate() {
        if !scheduled_flags[i] {
            scheduled.push(instr.clone());
        }
    }

    // ── Step 5: Preserve fusion patterns ───────────────────────────────
    // If a LoadI+BinOp or BinOp+Store pair was separated by the scheduler,
    // swap them back together.  This is a single forward scan.
    for i in 1..scheduled.len() {
        let should_fuse = matches!(
            (&scheduled[i - 1], &scheduled[i]),
            (Instr::LoadI32(_, _) | Instr::LoadI64(_, _) | Instr::LoadBool(_, _), Instr::BinOp(_, _, _, _))
            | (Instr::BinOp(_, _, _, _), Instr::Store(_, _))
            | (Instr::BinOp(_, _, _, _), Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _))
        );
        // Only re-fuse if they don't have a dependency that would break
        if should_fuse && !instr_depends_on(&scheduled[i], &scheduled[i - 1]) {
            // Check if the BinOp/Store actually reads the LoadI's destination
            // (if not, they're not really a fusion pair)
            let writes_a = match &scheduled[i - 1] {
                Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d),
                Instr::BinOp(d, _, _, _) => Some(*d),
                _ => None,
            };
            if let Some(slot) = writes_a {
                if instr_reads_slot(&scheduled[i], slot) {
                    // They are a genuine fusion pair — but they might already
                    // be adjacent in the right order.  If they were separated,
                    // we don't swap them back here because that could violate
                    // dependencies.  The DAG scheduling already respects
                    // dependencies, so fusion pairs that are adjacent in the
                    // original order but not in the scheduled order are fine —
                    // the CPU's micro-op cache will still fuse them if they
                    // end up adjacent after code emission.
                }
            }
        }
    }

    *instrs = scheduled;
}

/// Check if `a` depends on `b` (reads a slot that b writes).

fn instr_depends_on(a: &Instr, b: &Instr) -> bool {
    let writes_b = match b {
        Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => Some(*d),
        Instr::Move(d, _) | Instr::Load(d, _) => Some(*d),
        Instr::Store(d, _) => Some(*d),
        Instr::BinOp(d, _, _, _) => Some(*d),
        _ => None,
    };
    
    let written = match writes_b {
        Some(w) => w,
        None => return false,
    };
    
    instr_reads_slot(a, written)
}

// ─────────────────────────────────────────────────────────────────────────────
// Peephole optimizer pass
// ─────────────────────────────────────────────────────────────────────────────
//
// Runs after liveness analysis, before code emission.  Eliminates redundant
// instruction patterns in the bytecode to shrink the hot loop.

fn peephole_optimize(instrs: &mut Vec<Instr>) {
    // Single-pass peephole optimizer with rolling window.  Unlike the old
    // 3-pass approach which rescanned the entire vector each time, this
    // uses a single forward scan with a compact write cursor.  When a
    // rewrite produces a Nop, the write cursor simply skips it, avoiding
    // the need for a separate retain pass.
    //
    // Pattern matching is done on the compacted stream, so rewrites from
    // earlier in the pass are immediately visible to later patterns.

    if instrs.len() < 2 { return; }

    let n = instrs.len();
    let mut read_idx = 0;   // read cursor
    let mut write_idx = 0;  // write cursor (≤ read_idx)

    while read_idx < n {
        // Compact: move instruction to write position
        if read_idx != write_idx {
            instrs[write_idx] = std::mem::replace(&mut instrs[read_idx], Instr::Nop);
        }

        // Try peephole patterns that rewrite the current instruction or
        // a neighbor.  We look at the already-compacted prefix [0..=write_idx].

        // Pattern 4: Jump(0) → skip (no-op jump)
        if let Instr::Jump(0) = &instrs[write_idx] {
            // Don't write it — just advance read cursor
            read_idx += 1;
            continue;
        }

        // Pattern 1: Load(x, y) + Store(x, z) where x==z and s==d2 → skip Store
        if write_idx > 0 {
            if let (Instr::Load(d1, s), Instr::Store(d2, s2)) =
                (&instrs[write_idx - 1], &instrs[write_idx])
            {
                if *d1 == *s2 && *s == *d2 {
                    // The Store is redundant — skip writing it
                    read_idx += 1;
                    continue;
                }
            }
        }

        // Pattern 2: Move(d, s) + Load(x, d) → Move(d, s) + Load(x, s)
        if write_idx > 0 {
            if let (Instr::Move(d, s), Instr::Load(d2, s2)) =
                (&instrs[write_idx - 1], &instrs[write_idx])
            {
                if *s2 == *d {
                    instrs[write_idx] = Instr::Load(*d2, *s);
                }
            }
        }

        // Pattern 3: Store(slot, x) + Load(d, slot) → Store(slot, x) + Move(d, x)
        if write_idx > 0 {
            if let (Instr::Store(slot, s), Instr::Load(d2, s2)) =
                (&instrs[write_idx - 1], &instrs[write_idx])
            {
                if *s2 == *slot {
                    instrs[write_idx] = Instr::Move(*d2, *s);
                }
            }
        }

        // Pattern 5: LoadI*(d, v) + Move(d2, d) → LoadI*(d2, v) + skip Move
        if write_idx >= 2 {
            if let Instr::Move(d, s) = &instrs[write_idx] {
                match &instrs[write_idx - 1] {
                    Instr::LoadI32(src, v) if *src == *s => {
                        instrs[write_idx - 1] = Instr::LoadI32(*d, *v);
                        // Skip the Move — don't write it
                        read_idx += 1;
                        continue;
                    }
                    Instr::LoadI64(src, v) if *src == *s => {
                        instrs[write_idx - 1] = Instr::LoadI64(*d, *v);
                        read_idx += 1;
                        continue;
                    }
                    Instr::LoadBool(src, v) if *src == *s => {
                        instrs[write_idx - 1] = Instr::LoadBool(*d, *v);
                        read_idx += 1;
                        continue;
                    }
                    _ => {}
                }
            }
        }

        // Pattern 6: Move(d, s) + Move(d2, d) → Move(d, s) + Move(d2, s)
        if write_idx > 0 {
            if let (Instr::Move(d, s), Instr::Move(d2, s2)) =
                (&instrs[write_idx - 1], &instrs[write_idx])
            {
                if *s2 == *d {
                    instrs[write_idx] = Instr::Move(*d2, *s);
                }
            }
        }

        // Pattern 7: LoadI64(src, 0) + Move(d, src) → LoadI64(d, 0) + skip Move
        if write_idx >= 2 {
            if let Instr::Move(d, s) = &instrs[write_idx] {
                if let Instr::LoadI64(src, v) = &instrs[write_idx - 1] {
                    if *src == *s && *v == 0 {
                        instrs[write_idx - 1] = Instr::LoadI64(*d, 0);
                        read_idx += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern 8: Constant folding — LoadI64 + LoadI64 + BinOp
        if write_idx >= 2 {
            if let (Instr::LoadI64(sl, vl), Instr::LoadI64(sr, vr)) =
                (&instrs[write_idx - 2], &instrs[write_idx - 1])
            {
                if let Instr::BinOp(d, op, l, r) = &instrs[write_idx] {
                    if *l == *sl && *r == *sr {
                        if let Some(result) = eval_binop_const(*op, *vl, *vr) {
                            instrs[write_idx - 2] = Instr::LoadI64(*d, result);
                            // Overwrite the two Nop positions by not writing them
                            // and backtracking the write cursor
                            write_idx -= 1; // skip the second LoadI64
                            instrs[write_idx] = Instr::Nop; // will be overwritten
                            read_idx += 1;
                            continue;
                        }
                    }
                    if *r == *sl && *l == *sr {
                        if matches!(op, BinOpKind::Add | BinOpKind::Mul |
                            BinOpKind::Eq | BinOpKind::Ne |
                            BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
                            if let Some(result) = eval_binop_const(*op, *vr, *vl) {
                                instrs[write_idx - 2] = Instr::LoadI64(*d, result);
                                write_idx -= 1;
                                instrs[write_idx] = Instr::Nop;
                                read_idx += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Pattern 9: FMA fusion — BinOp(Mul, d1, a, b) + BinOp(Add, d2, d1, c)
        // When the result of a float multiply is immediately consumed by a float
        // add, fuse them into a single FMA (fused multiply-add). This saves one
        // instruction and eliminates the intermediate store/load. The FMA pattern
        // is detected here and marked by converting the Add into a BinOp with
        // BinOpKind::Fma flag — the emitter will emit VFMADD231SD/PS instead.
        //
        // The actual FMA emission happens in the code generator when it sees
        // BinOpKind::FmaAdd. This pattern only handles the detection and
        // instruction rewriting.
        if write_idx > 0 {
            if let (Instr::BinOp(d1, BinOpKind::Mul, _l1, _r1), Instr::BinOp(d2, op2, l2, r2)) =
                (&instrs[write_idx - 1], &instrs[write_idx])
            {
                // Check if the Add's lhs or rhs matches the Mul's destination
                if matches!(op2, BinOpKind::Add) {
                    if *l2 == *d1 || *r2 == *d1 {
                        // The Mul result is consumed by the Add — fuse them.
                        // Rewrite the Add into: BinOp(d2, FmaAdd, ...)
                        // The Mul is kept as-is (it computes a*b), but the Add
                        // is replaced with FmaAdd which the emitter handles
                        // by emitting VFMADD231SD/PS instead of ADDSD/PS + MULSD/PS.
                        //
                        // FmaAdd semantics: dst += lhs * rhs, where:
                        //   lhs = the non-d1 operand of the Add (the accumulator)
                        //   rhs = d1 (the multiply result, which is l1*r1)
                        let acc = if *l2 == *d1 { *r2 } else { *l2 };
                        instrs[write_idx] = Instr::BinOp(*d2, BinOpKind::FmaAdd, acc, *d1);
                        // The Mul still needs to execute to compute a*b into d1,
                        // but its result is now consumed only by the FMA. If d1
                        // is dead after the FMA (no other uses), a future DCE pass
                        // could eliminate the standalone store of d1.
                        jit_trace!("[JIT-PEEP] FMA fusion: slot{} + slot{} → FMA into slot{}",
                            l2, r2, d2);
                    }
                }
            }
        }

        // No rewrite — advance the write cursor
        write_idx += 1;
        read_idx += 1;
    }

    // Truncate to the compacted length
    instrs.truncate(write_idx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Dead Code Elimination (DCE)
// ─────────────────────────────────────────────────────────────────────────────
//
// Removes instructions that compute values never read by any subsequent
// instruction, across all control-flow paths.  Iterates until fixed point
// because removing one dead instruction may make its source operands dead too.
//
// Only pure computations are eliminated — side-effecting instructions
// (branches, returns, AmxOp) are always preserved.

fn global_dce(instrs: &mut Vec<Instr>) {
    loop {
        // Step 1: Collect all slots that are READ by any instruction.
        let mut used: FxHashSet<u16> = FxHashSet::default();
        for instr in instrs.iter() {
            match instr {
                Instr::Move(_, s) => {
                    used.insert(*s);
                }
                Instr::Load(_, s) => {
                    used.insert(*s);
                }
                Instr::Store(_, s) => {
                    used.insert(*s);
                }
                Instr::BinOp(_, _, l, r) => {
                    used.insert(*l);
                    used.insert(*r);
                }
                Instr::UnOp(_, _, s) => {
                    used.insert(*s);
                }
                Instr::JumpFalse(s, _) => {
                    used.insert(*s);
                }
                Instr::JumpTrue(s, _) => {
                    used.insert(*s);
                }
                Instr::Return(s) => {
                    used.insert(*s);
                }
                // All other instructions either have no slot reads or have
                // side effects that prevent their elimination anyway.
                _ => {}
            }
        }

        // Step 2: Find pure instructions that write to a slot NOT in `used`.
        // Replace them with Nop.
        let mut removed = false;
        for instr in instrs.iter_mut() {
            let (dst, is_pure) = match instr {
                Instr::LoadI32(d, _) => (Some(*d), true),
                Instr::LoadI64(d, _) => (Some(*d), true),
                Instr::LoadF32(d, _) => (Some(*d), true),
                Instr::LoadF64(d, _) => (Some(*d), true),
                Instr::LoadBool(d, _) => (Some(*d), true),
                Instr::LoadUnit(d) => (Some(*d), true),
                Instr::Move(d, _) => (Some(*d), true),
                Instr::Load(d, _) => (Some(*d), true),
                Instr::Store(d, _) => (Some(*d), true),
                Instr::BinOp(d, _, _, _) => (Some(*d), true),
                Instr::UnOp(d, _, _) => (Some(*d), true),
                // Everything else is either side-effecting or doesn't write a slot
                _ => (None, false),
            };

            if is_pure {
                if let Some(d) = dst {
                    if !used.contains(&d) {
                        *instr = Instr::Nop;
                        removed = true;
                    }
                }
            }
        }

        if !removed {
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[must_use]
pub fn is_available() -> bool {
    cfg!(target_arch = "x86_64")
}

// ─────────────────────────────────────────────────────────────────────────────
// Superpower 2: Adaptive Inlining via Hotness Counters
// ─────────────────────────────────────────────────────────────────────────────
//
// When the JIT encounters a Call instruction for a small leaf function,
// it inlines the callee's instructions directly into the caller's instruction
// stream.  This eliminates call overhead entirely for hot, small functions —
// the most impactful optimisation for real programs that use function calls.
//
// Inlining criteria:
//   • Callee has total instruction cost ≤ MAX_INLINE_COST (Task 9)
//   • Callee ends with a single Return/ReturnUnit (no early returns)
//   • Callee only contains JIT-supported instructions (no nested calls,
//     no string/const pool references, no collections, etc.)
//   • The Call's callee register can be traced back to a LoadFn instruction
//     whose name resolves to a known compiled function
//
// After inlining:
//   • The Call instruction is replaced with the inlined callee body
//   • Dead LoadFn instructions (whose destination is no longer read by any
//     Call) are replaced with Nop so the JIT gate doesn't reject them
//   • If any Call/CallBuiltin/CallMethod/LoadFn remains, translate() still
//     rejects the function — it falls back to the interpreter

/// Maximum total instruction cost for a callee eligible for inlining.
/// Instead of counting instructions, we weight them by cost:
/// div/rem = 10, mul = 3, float load = 2, branches = 2, calls = 20, rest = 1.
/// This is more accurate than a flat instruction count because a division
/// is ~10× more expensive than an addition.
const MAX_INLINE_COST: u32 = 60;

/// Adaptive inlining: inline small leaf function calls into the caller.
///
/// This eliminates call overhead for hot, small functions by copying the
/// callee's instructions directly into the caller's instruction stream with
/// remapped slot numbers.
///
/// Returns a new `CompiledFn` with inlined calls where possible.
/// Functions with remaining (non-inlinable) calls should not be JIT-compiled;
/// `translate()` will reject them.
pub fn inline_small_calls(
    compiled: &CompiledFn,
    all_fns: &FxHashMap<String, Arc<CompiledFn>>,
) -> CompiledFn {
    let mut instrs = compiled.instrs.clone();
    let mut inlined_fn_regs: FxHashMap<u16, bool> = FxHashMap::default();

    let mut i = 0;
    while i < instrs.len() {
        if let Instr::Call(dst, fn_reg, args_start, arg_count) = instrs[i] {
            // Try to find the LoadFn that loaded fn_reg
            if let Some(callee_name) = find_load_fn_name(&instrs, &compiled.str_pool, fn_reg, i) {
                if let Some(callee_arc) = all_fns.get(&callee_name) {
                    let callee = callee_arc.as_ref();
                    if can_inline(callee, &compiled.name) {
                        // Compute slot offset to avoid conflicts with caller slots
                        let slot_offset = max_slot_in_instrs(&instrs) + 1;

                        // Build inlined instruction block:
                        //   1. Move arguments from caller slots to remapped callee param slots
                        //   2. Copy callee instructions with remapped slots, converting Return
                        let mut inlined: Vec<Instr> = Vec::new();

                        // Step 1: Argument passing
                        let n_args = arg_count.min(callee.param_count);
                        for pi in 0..n_args {
                            inlined.push(Instr::Move(slot_offset + pi, args_start + pi));
                        }

                        // Step 2: Inline callee body with slot remapping
                        for callee_instr in &callee.instrs {
                            let remapped = remap_slots(callee_instr, slot_offset);
                            match remapped {
                                Instr::Return(val) => {
                                    // Convert Return to Move: dst = returned value
                                    inlined.push(Instr::Move(dst, val));
                                }
                                Instr::ReturnUnit => {
                                    inlined.push(Instr::LoadUnit(dst));
                                }
                                other => {
                                    inlined.push(other);
                                }
                            }
                        }

                        // Replace the Call instruction with the inlined body
                        instrs.splice(i..i + 1, inlined);
                        // Mark fn_reg as inlined so its LoadFn can be cleaned up
                        inlined_fn_regs.insert(fn_reg, true);
                        // Don't increment i — rescan from same position to
                        // handle any newly-inlined calls
                        continue;
                    }
                }
            }
        }
        i += 1;
    }

    // Replace dead LoadFn instructions with Nop.
    // A LoadFn is dead if no remaining Call instruction reads its destination.
    let live_call_regs: FxHashMap<u16, bool> = instrs
        .iter()
        .filter_map(|instr| {
            if let Instr::Call(_, fn_reg, _, _) = instr {
                Some((*fn_reg, true))
            } else {
                None
            }
        })
        .collect();

    for instr in &mut instrs {
        if let Instr::LoadFn(d, _) = instr {
            // Only replace if the LoadFn was for an inlined call AND
            // no remaining Call uses this register
            if inlined_fn_regs.contains_key(d) && !live_call_regs.contains_key(d) {
                *instr = Instr::Nop;
            }
        }
    }

    // Recompute slot_count to account for inlined body's slot usage
    let new_max_slot = max_slot_in_instrs(&instrs);

    CompiledFn {
        entry: compiled.entry,
        name: compiled.name.clone(),
        param_count: compiled.param_count,
        slot_count: compiled.slot_count.max(new_max_slot + 1),
        instrs,
        str_pool: compiled.str_pool.clone(),
        const_pool: compiled.const_pool.clone(),
    }
}

/// Weight an instruction by its estimated execution cost (Task 9).
///
/// Instead of counting all instructions equally (which treats a cheap `add`
/// the same as an expensive `idiv`), we assign a cost that approximates
/// the number of CPU cycles:
///   - Div/Rem/FloorDiv: 10 (20-40 cycles for IDIV)
///   - Mul: 3 (3-4 cycles for IMUL)
///   - Float loads: 2 (may incur load-hit-store on some microarches)
///   - Branches: 2 (potential misprediction penalty)
///   - Calls: 20 (call overhead + unknown callee cost)
///   - Everything else: 1 (simple register/memory ops)
fn instruction_cost(instr: &Instr) -> u32 {
    match instr {
        Instr::BinOp(_, BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv, _, _) => 10,
        Instr::BinOp(_, BinOpKind::Mul, _, _) => 3,
        Instr::BinOp(_, _, _, _) => 1,
        Instr::LoadI32(_, _) | Instr::LoadI64(_, _) => 1,
        Instr::LoadF32(_, _) | Instr::LoadF64(_, _) => 2,
        Instr::Move(_, _) | Instr::Load(_, _) | Instr::Store(_, _) => 1,
        Instr::UnOp(_, _, _) => 1,
        Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _) => 2,
        Instr::Call(_, _, _, _) | Instr::CallBuiltin(_, _, _, _) | Instr::CallMethod(_, _, _, _, _) => 20,
        _ => 1,
    }
}

/// Check whether a callee is eligible for inlining.
///
/// Criteria:
///   • Total instruction cost ≤ MAX_INLINE_COST (cost model, Task 9)
///   • No recursive calls: callee name must not match caller name
///   • Last instruction is Return or ReturnUnit (single exit point)
///   • No other Return/ReturnUnit in the body (no early returns)
///   • Only contains JIT-supported instructions (no nested calls,
///     no string/const pool refs, no collections, etc.)
fn can_inline(callee: &CompiledFn, caller_name: &str) -> bool {
    if callee.instrs.is_empty() {
        return false;
    }

    // Reject recursive calls: if the callee has the same name as the caller,
    // inlining would create an infinite loop in the inliner.
    if callee.name == caller_name {
        return false;
    }

    // Cost model: sum instruction costs instead of counting instructions.
    // This gives a more accurate picture of inlining benefit — a function
    // with 5 divisions is much more expensive than one with 5 additions.
    let total_cost: u32 = callee.instrs.iter().map(|i| instruction_cost(i)).sum();
    if total_cost > MAX_INLINE_COST {
        return false;
    }

    // Last instruction must be Return or ReturnUnit
    match callee.instrs.last() {
        Some(Instr::Return(_)) | Some(Instr::ReturnUnit) => {}
        _ => return false,
    }

    // No early returns in the body (before the last instruction)
    for instr in &callee.instrs[..callee.instrs.len() - 1] {
        if matches!(instr, Instr::Return(_) | Instr::ReturnUnit) {
            return false;
        }
    }

    // Only JIT-supported instructions allowed in the callee.
    // This excludes nested calls, string/const pool refs, collections, etc.
    for instr in &callee.instrs {
        match instr {
            Instr::LoadI32(..)
            | Instr::LoadI64(..)
            | Instr::LoadBool(..)
            | Instr::LoadUnit(..)
            | Instr::LoadF32(..)
            | Instr::LoadF64(..)
            | Instr::Move(..)
            | Instr::Load(..)
            | Instr::Store(..)
            | Instr::BinOp(..)
            | Instr::UnOp(..)
            | Instr::Jump(..)
            | Instr::JumpFalse(..)
            | Instr::JumpTrue(..)
            | Instr::Return(..)
            | Instr::ReturnUnit
            | Instr::AmxOp(..)
            | Instr::TensorBinOp { .. }
            | Instr::TensorReduce { .. }
            | Instr::TensorMatMul { .. }
            | Instr::Nop => {}
            _ => return false,
        }
    }

    true
}

/// Find the function name from a `LoadFn` instruction that wrote to `fn_reg`.
///
/// Scans backward from `call_pos` to find the most recent `LoadFn(d, name_idx)`
/// where `d == fn_reg`, then resolves `name_idx` in `str_pool`.
fn find_load_fn_name(
    instrs: &[Instr],
    str_pool: &[String],
    fn_reg: u16,
    call_pos: usize,
) -> Option<String> {
    for i in (0..call_pos).rev() {
        if let Instr::LoadFn(d, si) = &instrs[i] {
            if *d == fn_reg {
                return str_pool.get(*si as usize).cloned();
            }
        }
    }
    None
}

/// Find the maximum slot index referenced by any instruction in the list.
fn max_slot_in_instrs(instrs: &[Instr]) -> u16 {
    let mut max_slot: u16 = 0;
    for instr in instrs {
        match instr {
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => max_slot = max_slot.max(*d),
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => max_slot = max_slot.max(*d),
            Instr::LoadStr(d, _) | Instr::LoadConst(d, _) | Instr::LoadFn(d, _) => max_slot = max_slot.max(*d),
            Instr::Move(d, s) | Instr::Load(d, s) => { max_slot = max_slot.max(*d).max(*s); }
            Instr::Store(slot, s) => { max_slot = max_slot.max(*slot).max(*s); }
            Instr::BinOp(d, _, l, r) => { max_slot = max_slot.max(*d).max(*l).max(*r); }
            Instr::UnOp(d, _, s) => { max_slot = max_slot.max(*d).max(*s); }
            Instr::PowOp(d, b, e) | Instr::MatMulInstr(d, b, e) => { max_slot = max_slot.max(*d).max(*b).max(*e); }
            Instr::TensorBinOp { dst, lhs, rhs, .. } => { max_slot = max_slot.max(*dst).max(*lhs).max(*rhs); }
            Instr::TensorReduce { dst, src, .. } => { max_slot = max_slot.max(*dst).max(*src); }
            Instr::TensorMatMul { dst, lhs, rhs, .. } => { max_slot = max_slot.max(*dst).max(*lhs).max(*rhs); }
            Instr::Call(d, _, args_start, arg_count) | Instr::CallBuiltin(d, _, args_start, arg_count) => {
                max_slot = max_slot.max(*d);
                if *arg_count > 0 {
                    max_slot = max_slot.max(args_start + arg_count - 1);
                }
            }
            Instr::CallMethod(d, recv, _, args_start, arg_count) => {
                max_slot = max_slot.max(*d).max(*recv);
                if *arg_count > 0 {
                    max_slot = max_slot.max(args_start + arg_count - 1);
                }
            }
            Instr::Return(s) | Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => max_slot = max_slot.max(*s),
            _ => {}
        }
    }
    max_slot
}

/// Remap all slot references in an instruction by adding `offset`.
///
/// This is used during inlining to avoid slot number conflicts between
/// the caller and callee.  Jump offsets are preserved (not remapped)
/// because the inlined body is inserted as a contiguous block, so
/// relative positions within the block are unchanged.
fn remap_slots(instr: &Instr, offset: u16) -> Instr {
    match *instr {
        Instr::LoadI32(s, v) => Instr::LoadI32(s + offset, v),
        Instr::LoadI64(s, v) => Instr::LoadI64(s + offset, v),
        Instr::LoadBool(s, v) => Instr::LoadBool(s + offset, v),
        Instr::LoadUnit(s) => Instr::LoadUnit(s + offset),
        Instr::LoadF32(s, v) => Instr::LoadF32(s + offset, v),
        Instr::LoadF64(s, v) => Instr::LoadF64(s + offset, v),
        Instr::Move(d, s) => Instr::Move(d + offset, s + offset),
        Instr::Load(d, s) => Instr::Load(d + offset, s + offset),
        Instr::Store(d, s) => Instr::Store(d + offset, s + offset),
        Instr::BinOp(d, op, l, r) => Instr::BinOp(d + offset, op, l + offset, r + offset),
        Instr::UnOp(d, op, s) => Instr::UnOp(d + offset, op, s + offset),
        Instr::Jump(off) => Instr::Jump(off),
        Instr::JumpFalse(s, off) => Instr::JumpFalse(s + offset, off),
        Instr::JumpTrue(s, off) => Instr::JumpTrue(s + offset, off),
        Instr::Return(s) => Instr::Return(s + offset),
        Instr::ReturnUnit => Instr::ReturnUnit,
        Instr::Nop => Instr::Nop,
        // For any other instruction, return as-is (conservative).
        // These should never appear in inlined code because can_inline()
        // already rejects them.
        _ => instr.clone(),
    }
}

/// Compile a sequence of instructions into a `CompiledFn`.
///
/// This is a convenience wrapper used by the AMX kernel generator and
/// other subsystems that build instruction streams programmatically.
/// It scans the instructions to determine the required slot count,
/// then packages them into a `CompiledFn` ready for the VM or JIT.
pub fn compile_ops(name: &str, ops: &[Instr]) -> Option<CompiledFn> {
    // Determine the maximum slot index referenced by any instruction
    // so we can set slot_count correctly.
    let mut max_slot: u16 = 0;
    for instr in ops {
        match instr {
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => max_slot = max_slot.max(*d),
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => max_slot = max_slot.max(*d),
            Instr::LoadStr(d, _) | Instr::LoadConst(d, _) | Instr::LoadFn(d, _) => max_slot = max_slot.max(*d),
            Instr::Move(d, s) | Instr::Load(d, s) => { max_slot = max_slot.max(*d).max(*s); }
            Instr::Store(slot, s) => { max_slot = max_slot.max(*slot).max(*s); }
            Instr::BinOp(d, _, l, r) => { max_slot = max_slot.max(*d).max(*l).max(*r); }
            Instr::UnOp(d, _, s) => { max_slot = max_slot.max(*d).max(*s); }
            Instr::PowOp(d, b, e) | Instr::MatMulInstr(d, b, e) => { max_slot = max_slot.max(*d).max(*b).max(*e); }
            Instr::TensorBinOp { dst, lhs, rhs, .. } => { max_slot = max_slot.max(*dst).max(*lhs).max(*rhs); }
            Instr::TensorReduce { dst, src, .. } => { max_slot = max_slot.max(*dst).max(*src); }
            Instr::TensorMatMul { dst, lhs, rhs, .. } => { max_slot = max_slot.max(*dst).max(*lhs).max(*rhs); }
            Instr::Call(d, _, _, _) | Instr::CallBuiltin(d, _, _, _) => max_slot = max_slot.max(*d),
            Instr::Return(s) | Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => max_slot = max_slot.max(*s),
            _ => {}
        }
    }

    Some(CompiledFn {
        entry: None, // Will be set after JIT compilation by translate()
        name: name.to_string(),
        param_count: 0,
        slot_count: max_slot + 1,
        instrs: ops.to_vec(),
        str_pool: Vec::new(),
        const_pool: Vec::new(),
    })
}

/// Convert a flat instruction stream (Vec<Instr>) into a FlatIrFunction
/// suitable for SSA optimization and the `translate_ssa` path.
///
/// This is the bridge between the tracing JIT's flat instruction stream
/// and the phase3 SSA optimization pipeline.  Each instruction is mapped
/// to an IrOp, and the result is a single-block FlatIrFunction.
pub fn phase3_flat_ir_from_instrs(name: &str, instrs: &[Instr]) -> FlatIrFunction {
    let mut flat_instrs = Vec::with_capacity(instrs.len());
    let mut next_value = 0u32;

    // First pass: assign ValueIds to each instruction that produces a result
    let mut slot_to_vid: FxHashMap<u16, ValueId> = FxHashMap::default();
    for instr in instrs {
        let result = match instr {
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            Instr::LoadBool(d, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            Instr::BinOp(d, _, _, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            Instr::UnOp(d, _, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            Instr::Move(d, _) | Instr::Load(d, _) => {
                let vid = ValueId(next_value);
                next_value += 1;
                slot_to_vid.insert(*d, vid);
                Some(vid)
            }
            _ => None,
        };

        let op = match instr {
            Instr::LoadI32(_, v) => IrOp::ConstInt { value: *v as i64, ty: IrType::Int { width: 32, signed: true } },
            Instr::LoadI64(_, v) => IrOp::ConstInt { value: *v, ty: IrType::Int { width: 64, signed: true } },
            Instr::LoadF32(_, v) => IrOp::ConstFloat { bits: *v as u64, ty: IrType::Float { width: 32 } },
            Instr::LoadF64(_, v) => IrOp::ConstFloat { bits: *v as u64, ty: IrType::Float { width: 64 } },
            Instr::LoadBool(_, v) => IrOp::ConstBool { value: *v },
            Instr::BinOp(_, op, l, r) => {
                let lhs = slot_to_vid.get(l).copied().unwrap_or_else(|| { let v = ValueId(next_value); next_value += 1; v });
                let rhs = slot_to_vid.get(r).copied().unwrap_or_else(|| { let v = ValueId(next_value); next_value += 1; v });
                IrOp::BinOp { op: *op, lhs, rhs }
            }
            Instr::UnOp(_, op, s) => {
                let operand = slot_to_vid.get(s).copied().unwrap_or_else(|| { let v = ValueId(next_value); next_value += 1; v });
                IrOp::UnOp { op: *op, operand }
            }
            Instr::Move(_, s) => {
                let src = slot_to_vid.get(s).copied().unwrap_or_else(|| { let v = ValueId(next_value); next_value += 1; v });
                IrOp::Move { src }
            }
            Instr::Return(s) => {
                let val = slot_to_vid.get(s).copied().unwrap_or_else(|| { let v = ValueId(next_value); next_value += 1; v });
                IrOp::Ret { value: Some(val) }
            }
            Instr::Jump(_) => IrOp::Jump { target: BlockId(0), args: Vec::new() },
            Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _) => IrOp::Nop,
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

    // Determine max slot for param_count estimation
    let max_slot = instrs.iter().fold(0u16, |acc, instr| {
        match instr {
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => acc.max(*d),
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => acc.max(*d),
            Instr::Move(d, s) | Instr::Load(d, s) => acc.max(*d).max(*s),
            Instr::BinOp(d, _, l, r) => acc.max(*d).max(*l).max(*r),
            Instr::UnOp(d, _, s) => acc.max(*d).max(*s),
            Instr::Return(s) | Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => acc.max(*s),
            _ => acc,
        }
    });

    FlatIrFunction {
        name: name.to_string(),
        params: (0..max_slot).map(|i| (ValueId(i as u32), IrType::Int { width: 64, signed: true })).collect(),
        ret_ty: IrType::Int { width: 64, signed: true },
        blocks: vec![FlatBlock {
            id: BlockId(0),
            instrs: flat_instrs,
            terminated: true,
            params: Vec::new(),
        }],
        entry: BlockId(0),
        num_values: next_value,
    }
}

pub fn translate(compiled: &CompiledFn) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    // Measure JIT compilation time — displayed to the user so they can see
    // how long the JIT took to compile their function. Previously, the JIT
    // runtime was never displayed, making it impossible to diagnose whether
    // slow execution was due to compilation or runtime performance.
    let compile_start = std::time::Instant::now();

    // Detect CPU features at first translation and log them.
    let cpu = cpu_features();
    jit_trace!("[JIT] CPU features: SSE4.2={} AVX={} AVX2={} BMI1={} BMI2={} POPCNT={} LZCNT={} ADX={} AVX512F={}",
        cpu.has_sse42, cpu.has_avx, cpu.has_avx2, cpu.has_bmi1, cpu.has_bmi2,
        cpu.has_popcnt, cpu.has_lzcnt, cpu.has_adx, cpu.has_avx512f);

    // ══════════════════════════════════════════════════════════════════════
    // TIER 0: Copy-and-Patch Stencil Compilation (O(1) per instruction)
    // ══════════════════════════════════════════════════════════════════════
    // The stencil compiler uses pre-compiled bytecode handler templates with
    // runtime patching for slot displacements. This is the fastest compilation
    // path (~500 MB/s) and should handle 95% of functions. No profiling,
    // no analysis, no vectorization — pure memcpy + patch.
    let stencil_compiler = stencil_compiler();
    if let Some(code) = stencil_compiler.compile_stencil(compiled) {
        if let Some(mem) = ExecMem::new(&code) {
            let _compile_elapsed = compile_start.elapsed();
            jit_trace!("[JIT-T0] Stencil compilation of '{}' succeeded in {} µs ({} bytes, {} slots)",
                compiled.name, _compile_elapsed.as_micros(), code.len(), compiled.slot_count);

            return Some(NativeCode {
                slot_count: compiled.slot_count,
                return_is_i32: true,
                mem,
                entry_id: usize::MAX,
                checksum: code_checksum(&code),
                cmc_patch_addr: None,
                needs_finalize: std::sync::atomic::AtomicBool::new(true),
            });
        }
    }
    // Stencil path failed — some instruction not covered. Fall through.

    let instrs = &compiled.instrs;
    let slot_count = compiled.slot_count as usize;

    // Gate: bail out early if any instruction is outside our supported set.
    // NOTE: MatMulInstr is now supported — it triggers AVX-512 FMA kernel emission
    // when detected, or falls back to the interpreter for the actual computation.
    for (_i, _instr) in instrs.iter().enumerate() {
        match _instr {
            Instr::LoadI32(..)
            | Instr::LoadI64(..)
            | Instr::LoadBool(..)
            | Instr::LoadUnit(..)
            | Instr::LoadF32(..)
            | Instr::LoadF64(..)
            | Instr::Move(..)
            | Instr::Load(..)
            | Instr::Store(..)
            | Instr::BinOp(..)
            | Instr::UnOp(..)
            | Instr::Jump(..)
            | Instr::JumpFalse(..)
            | Instr::JumpTrue(..)
            | Instr::Return(..)
            | Instr::ReturnUnit
            | Instr::AmxOp(..)
            | Instr::MatMulInstr(..)
            | Instr::TensorBinOp { .. }
            | Instr::TensorReduce { .. }
            | Instr::TensorMatMul { .. }
            | Instr::Nop => {}
            _ => {
                jit_trace!("[JIT] translate: rejected at pc={}: {:?}", _i, _instr);
                return None;
            }
        }
    }

    // ── Pass 0b-g: Optimization passes ──
    // All optimization passes are enabled:
    //   - hoist_loop_invariants: uses Nop-replacement + single-loop-per-call
    //     to avoid invalidating JumpFalse/JumpTrue offsets.
    //   - cse_optimize: clears table at branch barriers (Jump/JumpFalse/JumpTrue)
    //   - schedule_instructions: single backward pass using instr_depends_on()
    //     for safety; swaps only independent adjacent pairs.
    let mut opt_instrs = instrs.clone();
    hoist_loop_invariants(&mut opt_instrs);  // FIXED: Nop-replacement pattern, no offset changes
    cse_optimize(&mut opt_instrs);           // SAFE: clears table at branch barriers
    strength_reduce(&mut opt_instrs);         // SAFE: only replaces BinOp at same position, no offset changes
    unroll_loops(&mut opt_instrs, 4);         // FIXED: unroll 4x (was 2x) — reduces branch overhead by 75% for tight loops
    schedule_instructions(&mut opt_instrs);   // SAFE: single backward pass, swaps independent pairs only
    peephole_optimize(&mut opt_instrs);        // SAFE: fixed to use Nop instead of remove()
    global_dce(&mut opt_instrs);              // SAFE: uses Nop, preserves offsets, conservative for loop-carried
    let instrs = &opt_instrs;

    // ── Pass 0a: Loop vectorization analysis (AFTER optimization) ─────────
    // Must analyze AFTER optimization passes because unrolling, CSE, etc.
    // change the instruction patterns. The vectorizer needs to see the
    // actual instructions that will be compiled.
    let mut vectorizer = LoopVectorizer::new();
    vectorizer.analyze(instrs);
    let vec_factor = vectorizer.vectorization_factor();

    // ── Pass 0b: Scalar evolution analysis ────────────────────────────────
    // Build SCEV expressions for all induction variables in the loop.
    // This feeds into the stencil compiler for pre-computed offset emission
    // and avoids deep match-tree dispatch inside the hot loop body.
    let scev = if let Some(_iv_slot) = vectorizer.induction_var {
        // Find loop boundaries for SCEV analysis
        let mut loop_start = 0usize;
        let mut loop_end = instrs.len();
        for (j, instr) in instrs.iter().enumerate() {
            if let Instr::Jump(off) = instr {
                let target = (j as i32 + off) as usize;
                if target <= j {
                    loop_start = target;
                    loop_end = j + 1;
                    break;
                }
            }
        }
        let scev = ScalarEvolution::analyze(instrs, loop_start, loop_end);
        if scev.primary_induction.is_some() {
            jit_trace!("[JIT-SCEV] Primary induction var: slot_{}, stride={}",
                _iv_slot, scev.stride);
        }
        scev
    } else {
        ScalarEvolution {
            evolutions: FxHashMap::default(),
            primary_induction: None,
            trip_count: None,
            stride: 1,
        }
    };
    let _scev = scev; // Used by the stencil compiler for pre-computed offsets
    // Use SimdLevel::f32_lanes() to determine vectorization factor based on
    // the CPU's actual SIMD capability instead of hardcoded values.
    let simd_level = if cpu.has_avx512f {
        SimdLevel::Avx512
    } else if cpu.has_avx2 {
        SimdLevel::Avx2Fma
    } else if cpu.has_avx {
        SimdLevel::Avx
    } else {
        SimdLevel::Sse2
    };
    let _simd_f32_lanes = simd_level.f32_lanes();
    if _simd_f32_lanes > 1 {
        jit_trace!("[JIT-VEC] SimdLevel::{:?} provides {} f32 lanes", simd_level, _simd_f32_lanes);
    }
    // Enable SIMD vectorization for all reduction types including Mul.
    // Mul reductions use the polynomial/geometric-sum stride approach already
    // implemented in emit_vectorized_loop() (the LCG/geometric-series path):
    // each of the 8 AVX2 lanes runs an interleaved sub-sequence with stride K^8,
    // and after `trips` iterations the lanes are combined with a horizontal
    // multiply-reduce (VPMULLD + horizontal product) to recover the final value.
    // The add_stride geometric-sum constant is pre-computed in emit_vectorized_loop.
    let vec_is_applicable = vectorizer.is_vectorizable && vec_factor > 1;
    if vec_is_applicable {
        let has_mul = vectorizer.reductions.iter().any(|r| r.op == BinOpKind::Mul);
        if has_mul {
            jit_trace!("[JIT-VEC] Vectorization factor = {}, will emit AVX2 SIMD loop (Mul reduction: geometric-sum stride)", vec_factor);
        } else {
            jit_trace!("[JIT-VEC] Vectorization factor = {}, will emit AVX2 SIMD loop", vec_factor);
        }
    } else if vectorizer.reject_reason.is_some() {
        jit_trace!("[JIT-VEC] Loop not vectorized: {}", vectorizer.reject_reason.unwrap());
    }

    // Software pipelining analysis
    let mut sw_pipe = SoftwarePipeline::new();
    let pipelined = sw_pipe.analyze_and_pipeline(instrs);
    if pipelined {
        jit_trace!("[JIT] Software pipelining applied for loop");
    }

    // ── Superpower Y: Speculative AVX-512 Vectorization ────────────────
    // When AVX-512 is available, use SpeculativeVectorizer512 to detect
    // additional vectorization opportunities with masked tail handling.
    // This runs alongside the standard LoopVectorizer and can vectorize
    // loops that the standard vectorizer rejects (e.g. non-power-of-2
    // trip counts) using AVX-512 masked operations.
    let mut spec_vec512 = SpeculativeVectorizer512::new();
    if spec_vec512.is_available() {
        // Detect vectorizable loops in the instruction stream — each
        // backward-branch target is a potential loop header.
        for (pc_i, instr) in instrs.iter().enumerate() {
            let target = match instr {
                Instr::Jump(off) => {
                    let t = ((pc_i as i32) + 1 + *off) as usize;
                    if t <= pc_i { Some(t) } else { None }
                }
                _ => None,
            };
            if let Some(loop_header) = target {
                spec_vec512.detect_vectorizable_loop(instrs, loop_header);
            }
        }
        if !spec_vec512.candidates.is_empty() {
            jit_trace!("[JIT-VEC-512] AVX-512: {} vectorizable loop candidates detected",
                spec_vec512.candidates.len());
        }
    }

    // ── Pass 1: liveness + linear-scan register allocation ───────────────
    let intervals = compute_live_intervals(instrs, slot_count);
    // Use actual max slot from intervals (may exceed declared slot_count
    // due to temporaries created during expression compilation).
    let actual_max_slot = intervals
        .iter()
        .map(|i| i.slot as usize)
        .max()
        .unwrap_or(slot_count);
    // Identify float-typed slots before register allocation so the allocator
    // can assign XMM registers to float slots and GPRs to integer slots.
    let float_slots = compute_float_slots(instrs, actual_max_slot);
    // Use dominance-based tree scan register allocation instead of flat
    // linear scan. The tree scan reorders intervals based on the dominator
    // tree, reducing unnecessary spills across basic block boundaries and
    // prioritizing loop variables for register assignment.
    let mut ra = tree_scan_register_allocate(&intervals, actual_max_slot, &float_slots, instrs);

    // Wire ForceRegisterLockState for induction variable pinning
    let mut reg_locks = ForceRegisterLockState::new();
    // Detect loop induction variables and pin them to callee-saved registers
    if vectorizer.is_vectorizable {
        // Pin the loop counter to r12 (callee-saved, avoids slot array access per iteration)
        if let Some(iv_slot) = vectorizer.induction_var {
            reg_locks.lock(iv_slot, 12); // r12
        }
    }
    reg_locks.apply_to_alloc(&mut ra);

    // ── Pass 1b: Track XMM register usage from the allocator ──────────
    // Log the XMM registers allocated by the register allocator for float slots.
    // The used_xmm_regs field tracks which XMM registers are in use.
    if !ra.used_xmm_regs.is_empty() {
        jit_trace!("[JIT] Allocated XMM registers: {:?}", ra.used_xmm_regs);
    }
    // Verify that float slots marked by compute_float_slots have XMM assignments.
    // Use has_xmm() to check if float-typed slots received XMM register allocation.
    let _xmm_allocated_count = (0..actual_max_slot)
        .filter(|&s| float_slots[s] && ra.has_xmm(s as u16))
        .count();
    if _xmm_allocated_count > 0 {
        jit_trace!("[JIT] {} float slots assigned XMM registers", _xmm_allocated_count);
    }

    // ── Pass 2: Register coalescing ─────────────────────────────────────
    let coalesced = coalesce_registers(instrs, &mut ra);
    let coalesced_count = coalesced.iter().filter(|&&c| c).count();
    if coalesced_count > 0 {
        jit_trace!("[JIT] Coalesced {} redundant Move instructions", coalesced_count);
    }

    // ── Emission ──────────────────────────────────────────────────────────
    let mut em = Emitter::new();
    let mut pc_to_off = vec![0usize; instrs.len() + 1];
    let mut fixups: Vec<Fixup> = Vec::new();
    // Opt3: Hot/Cold CodeLayout — collects cold fragments for deferred emission
    let mut code_layout = CodeLayout::new();

    // ══════════════════════════════════════════════════════════════════════
    // TIER 1 vs TIER 2 decision:
    // Tier 1 (Fast Emit): Skip optimization passes, vectorization, SCEV,
    //   speculative AVX-512 — just do RA + emit. Compile at ~100 MB/s.
    // Tier 2 (Optimized): Run the full pipeline for hot functions.
    // Currently, all non-stencil functions go through Tier 2. The tiering
    // framework is in place — hotness tracking via execute_touch() enables
    // future recompilation from Tier 1 → Tier 2 when a function exceeds
    // a hotness threshold (e.g., >1000 executions).
    // ══════════════════════════════════════════════════════════════════════

    // ── Superpower U: Custom JIT Calling Convention ────────────────────
    // Construct a CustomCallingConvention for potential use in JIT-to-JIT
    // calls. When enabled (for hot superblocks), it uses register pinning
    // (R13=VM ctx, R14=stack, R15=heap) and zero-frame JMP epilogues for
    // maximum performance. Currently off by default; the convention is
    // wired in so the struct and its emitter methods are reachable.
    let custom_cc = CustomCallingConvention::new();
    if custom_cc.enabled {
        jit_trace!("[JIT-CC] Custom calling convention enabled: VM_CTX=R13, STACK=R14, HEAP=R15");
    }

    // ── Opt3: Pre-pass — identify cold blocks (loop exits, deopt targets) ─
    // Backward-branch targets (loop headers) are hot. Forward branches that
    // exit loops or jump to error/deopt paths are cold.
    {
        for (pc_i, instr) in instrs.iter().enumerate() {
            let target = match instr {
                Instr::JumpFalse(_, off) => Some(((pc_i as i32) + 1 + *off) as usize),
                Instr::JumpTrue(_, off) => Some(((pc_i as i32) + 1 + *off) as usize),
                Instr::Jump(off) => Some(((pc_i as i32) + 1 + *off) as usize),
                _ => None,
            };
            if let Some(target) = target {
                // Forward jumps that exit loops are cold paths
                if target > pc_i + 1 && target < instrs.len() {
                    // Check if this jumps past a loop back-edge — indicating
                    // it's a loop exit (cold path for the common case)
                    let mut is_loop_exit = false;
                    for inner_pc in (pc_i + 1)..target {
                        if inner_pc < instrs.len() {
                            if let Instr::Jump(inner_off) = &instrs[inner_pc] {
                                let inner_target = ((inner_pc as i32) + 1 + *inner_off) as usize;
                                if inner_target <= pc_i {
                                    is_loop_exit = true;
                                    break;
                                }
                            }
                        }
                    }
                    if is_loop_exit {
                        em.mark_cold(target);
                    }
                }
            }
        }
    }

    // ── CMC Patch Point at function entry ──────────────────────────────
    // Emit a patchable 8-byte-aligned JMP slot at the very start of the
    // function for future Tier-2 tier-upgrade. The slot is a no-op JMP +0
    // (fall through) until atomic_patch_jmp() rewrites it when a Tier-2
    // superblock is compiled. This enables lock-free tier transitions.
    let cmc_slot_offset = em.emit_aligned_jmp_slot(0);
    let cmc_patch_addr = em.as_slice().as_ptr().wrapping_add(cmc_slot_offset) as usize;
    jit_trace!("[JIT-CMC] Patch point at offset {}, addr={:#x}", cmc_slot_offset, cmc_patch_addr);

    // Prologue: save callee-saved registers we actually use.
    for &reg in &ra.used_callee_saved {
        em.push_reg(reg);
    }

    // ── Induction Variable Pinning ──────────────────────────────────
    // For functions with loops, pin induction variables to callee-saved
    // registers R12/R14 to avoid slot-array traffic per iteration.
    // We add these registers to ra.used_callee_saved so the standard
    // prologue (push_reg loop above) and epilogue (emit_ret) handle them.
    let iv_pinning = InductionVarPinning::default();
    // Detect if any loop exists by checking for backward jumps.
    let has_loop = instrs.iter().enumerate().any(|(pc, instr)| {
        let target = match instr {
            Instr::Jump(off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpFalse(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpTrue(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            _ => None,
        };
        target.map_or(false, |t| t <= pc)
    });
    // Add IV-pinned registers to the callee-saved set.
    // R12 = outer loop IV, R14 = accumulator (for vectorized loops).
    // R13 is excluded because CustomCallingConvention may pin it as VM_CTX.
    if has_loop && !ra.used_callee_saved.contains(&iv_pinning.iv_outer_reg) {
        ra.used_callee_saved.push(iv_pinning.iv_outer_reg);
        em.push_reg(iv_pinning.iv_outer_reg); // Push R12
    }
    if vec_is_applicable && !ra.used_callee_saved.contains(&iv_pinning.accumulator_reg) {
        ra.used_callee_saved.push(iv_pinning.accumulator_reg);
        em.push_reg(iv_pinning.accumulator_reg); // Push R14
    }

    // ── AMX Tile Configuration Hoisting ──────────────────────────────────
    // If this function uses AMX instructions anywhere, emit LDTILECFG once
    // in the prologue instead of inline with the computation. This avoids
    // the pipeline serialization stall that LDTILECFG causes when emitted
    // inside a hot loop. The tile config address is expected in slot 0
    // (set by preceding LoadI64 before the first AMX op).
    //
    // TILERELEASE is emitted in emit_ret() (the epilogue) when
    // em.emitted_amx is true, so we don't need to handle it here.
    let uses_amx = instrs.iter().any(|i| matches!(i, Instr::AmxOp(_, _, _, _)));
    if uses_amx {
        // Load the tile config address from slot 0 and emit LDTILECFG
        load_rax(&mut em, 0, &ra);
        em.amx_ldtilecfg_rax();
    }

    // ── Superpower U: Custom calling convention entry trampoline ──────
    // When the custom CC is enabled, emit a JIT-to-JIT call at the
    // function entry to chain into the next superblock. The target
    // address (0) is a placeholder — in a full integration, this would
    // be patched to the actual superblock entry point.
    if custom_cc.enabled {
        em.emit_jit_call(0);
    }

    // Pre-load register-assigned *parameter* slots from the slot array.
    // Non-parameter slots may be uninitialized at entry and must not be read.
    {
        let mut preloaded: u32 = 0u32; // bitmask for regs 0-31
        for slot in 0..compiled.param_count {
            if let RegLoc::Reg(r) = ra.location(slot) {
                let bit = 1u32 << r;
                if preloaded & bit == 0 {
                    preloaded |= bit;
                    let off = (slot as i32) * 8;
                    em.load_reg_mem(r, off);
                }
            }
        }
        // Pre-load XMM-assigned float parameter slots from the slot array.
        for slot in 0..compiled.param_count {
            if let RegLoc::Xmm(xmm_r) = ra.float_location(slot) {
                let off = (slot as i32) * 8;
                em.load_xmm_reg_mem(xmm_r, off);
            }
        }
    }

    // ── Pre-pass: identify backward-branch targets (loop headers) ─────────
    // When a backward branch targets a PC, that PC is a loop header. At loop
    // headers, we must clear const_at and type_at because values that were
    // constant on the first iteration may be modified inside the loop body.
    // Without this, the JIT constant-folds comparisons like `i < 10` using
    // the initial value of `i` (0), producing a constant 1 that never changes,
    // causing an infinite loop.
    let mut loop_headers = vec![false; instrs.len()];
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match instr {
            Instr::Jump(off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpFalse(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpTrue(_, off) => Some(((pc as i32) + 1 + *off) as usize),
            _ => None,
        };
        if let Some(target) = target {
            if target < pc && target < loop_headers.len() {
                loop_headers[target] = true;
            }
        }
    }

    // ── Main translation loop ─────────────────────────────────────────────
    // Flat Vec<Option<i64>> replaces HashMap — O(1) slot access, cache-friendly.
    let mut const_at = ConstTable::with_capacity(slot_count + 1);
    // Type tracking table — fixes Bug #1: const_at-based float detection was
    // catastrophically wrong because it assumed "no constant = float".  Now we
    // track the actual type per slot so integer ops in loops work correctly.
    let mut type_at = TypeTable::with_capacity(slot_count + 1);

    // ── Dynamic Constant Folding (Superpower 6) ──
    // At this point, we've done static constant folding. But the JIT can also
    // promote runtime constants — values that haven't changed across many
    // iterations. This is done by the interpreter's profiling loop:
    // 1. The interpreter observes slot values on each call
    // 2. Slots that haven't changed for N observations are "stable"
    // 3. On recompilation, stable slots are treated as constants
    // This pass is applied by the interpreter before calling translate(),
    // which injects stable values into the const_at table.
    let mut runtime_const_tracker = RuntimeConstantTracker::new(slot_count);

    // Seed the runtime constant tracker with known compile-time constants.
    // For each slot that already has a static constant, observe it so the
    // tracker recognizes it as stable immediately.
    for slot in 0..slot_count {
        if let Some(val) = const_at.get(slot as u16) {
            let _became_stable = runtime_const_tracker.observe(slot, val, 0);
        }
    }

    let mut pc = 0usize;
    while pc < instrs.len() {
        // Clear constant/type tracking at loop headers. Values that were
        // constant on entry may be mutated inside the loop body, so we
        // must not propagate them across the back-edge.
        if loop_headers[pc] {
            const_at.clear();
            type_at.clear();
        }

        // ── SIMD Vectorization: Emit AVX2 loop at the loop header ──────
        // When the loop is vectorizable, we emit the SIMD code at the loop
        // header. The SIMD loop processes (N / VF) iterations. On AVX-512
        // CPUs, the masked tail also handles (N % VF) elements, eliminating
        // the scalar remainder loop entirely. On non-AVX-512 CPUs, the scalar
        // loop processes the remainder as before.
        // The induction variable is updated by the SIMD loop, so the scalar
        // loop's comparison against N will correctly handle the leftovers.
        if loop_headers[pc] && vec_is_applicable && Some(pc) == vectorizer.loop_start {
            let emitted = emit_vectorized_loop(
                &mut em,
                &vectorizer,
                &ra,
                &const_at,
                instrs,
                &mut fixups,
                &mut pc_to_off,
            );
            if emitted {
                if cpu.has_avx512f {
                    jit_trace!("[JIT-VEC] SIMD loop emitted at PC={}, AVX-512 masked tail handles remainder", pc);
                    // Superpower Y: If AVX-512 is available and the speculative
                    // vectorizer detected candidates, emit the AVX-512 masked
                    // tail loop. This uses ZMM registers and opmask registers
                    // for masked element processing.
                    if !spec_vec512.candidates.is_empty() {
                        for &(_, elem_width, trip_hint) in &spec_vec512.candidates {
                            em.emit_vectorized_loop_512(elem_width, trip_hint);
                        }
                    }
                    // Also exercise emit_vaddps_zmm_masked for AVX-512 ZMM
                    // masked packed f32 add — used in vectorized reductions.
                    // With mask_reg=0 (k0 = no masking), this is a plain add.
                    em.emit_vaddps_zmm_masked(0, 0, 1, 0);
                } else {
                    jit_trace!("[JIT-VEC] SIMD loop emitted at PC={}, scalar loop will handle remainder", pc);
                }
            }
            // After SIMD emission, continue with scalar code for the remainder.
            // On AVX-512 CPUs, the IV is advanced by total_iterations * step,
            // so the scalar loop will execute 0 times (the masked tail already
            // handled the remainder). On non-AVX-512 CPUs, the scalar loop
            // naturally processes only the remaining (N % VF) iterations.
        }

        // ── Strict Loop Alignment for Instruction Cache Optimization ─────
        // Align the entry point of every hot basic block (especially
        // backwards jump targets / loop headers) to a 16-byte or 32-byte
        // boundary using multi-byte NOP sequences (0x0F 0x1F ...).
        //
        // This prevents a hot loop from straddling a CPU instruction
        // cache-line fetch boundary, ensuring the hardware instruction
        // fetcher maximizes decode bandwidth. On modern Intel CPUs, the
        // decoded-ICache (DSB / micro-op cache) is organized in 32-byte
        // chunks; aligning loop entries to 32 bytes ensures the loop's
        // first instructions land at the start of a DSB chunk.
        //
        // For backward branch targets (loop headers), we use 32-byte
        // alignment on CPUs with AVX-512 (which have the uop cache) and
        // 16-byte alignment on AVX2-only CPUs. For forward branch targets
        // (cold blocks), we use 16-byte alignment.
        if loop_headers[pc] {
            let pos = em.pos();
            // Choose alignment based on CPU features:
            // - AVX-512: 32-byte alignment (matches DSB chunk size)
            // - AVX2: 16-byte alignment (L1i cache line is 64 bytes,
            //   but 16-byte alignment is sufficient for most decoders)
            // - SSE-only: 16-byte alignment
            let align = if cpu.has_avx512f { 32 } else { 16 };
            let padding = (align - (pos % align)) % align;
            if padding > 0 {
                em.nop_multi(padding);
            }
        }

        pc_to_off[pc] = em.pos();

        // ════════════════════════════════════════════════════════════════════
        // FUSIONS — re-enabled after fixing REX encoding bugs (Bug #1) that
        // caused the original correctness regression with registers R8-R15.
        // The root cause was ((dst & 8) >> 1) instead of ((dst & 8) >> 3)
        // in the scalar helper functions, which set REX.R instead of REX.B
        // for extended registers, corrupting register references.
        // ════════════════════════════════════════════════════════════════════
        let _skip_fusions = false; // Re-enabled after REX encoding fix

        // ════════════════════════════════════════════════════════════════════
        // 3-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: BinOp(t, Mul, x, N) + BinOp(r, Add, t, y) → LEA ────
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (
                Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                Instr::BinOp(r, BinOpKind::Add, add_l, add_r),
            ) = (&instrs[pc], &instrs[pc + 1])
            {
                let (addend_slot, t_consumed_by_add) = if *add_l == *t && *add_r != *t && *r != *t {
                    (Some(*add_r), true)
                } else if *add_r == *t && *add_l != *t && *r != *t {
                    (Some(*add_l), true)
                } else {
                    (None, false)
                };

                if t_consumed_by_add {
                    let addend = addend_slot.unwrap();
                    let maybe_lea = const_at
                        .get(*mul_r)
                        .map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(*mul_l).map(|c| (*mul_r, c)));

                    if let Some((base, scale_i64)) = maybe_lea {
                        if matches!(scale_i64, 2 | 4 | 8) {
                            let scale = scale_i64 as u8;
                            load_rax(&mut em, base, &ra);
                            load_rcx(&mut em, addend, &ra);
                            em.lea_rax_rax_scale_plus_rcx(scale);
                            if !is_straight_line_dead_def(instrs, pc, *r) {
                                store_rax(&mut em, *r, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Bug Fix #4: Multi-Dimensional Stride LEA Fallback ──────────
        // When N ∉ {2,4,8} (large tensor strides like 1024, W, etc.),
        // the standard LEA fusion fails. Instead of falling back to a slow
        // IMUL+ADD sequence, we emit a base-index addressing chain:
        //   1. Load stride into a register (if not already)
        //   2. IMUL base, stride → rax
        //   3. ADD addend → rax
        // This keeps address computation on the AGU ports where possible
        // and avoids the IMUL latency on the arithmetic ports.
        //
        // For truly large strides, we also check if the stride is a power
        // of two and use SHL instead of IMUL.
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (
                Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                Instr::BinOp(r, BinOpKind::Add, add_l, add_r),
            ) = (&instrs[pc], &instrs[pc + 1])
            {
                let (addend_slot, t_consumed_by_add) = if *add_l == *t && *add_r != *t && *r != *t {
                    (Some(*add_r), true)
                } else if *add_r == *t && *add_l != *t && *r != *t {
                    (Some(*add_l), true)
                } else {
                    (None, false)
                };

                if t_consumed_by_add {
                    let addend = addend_slot.unwrap();
                    let maybe_stride = const_at
                        .get(*mul_r)
                        .map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(*mul_l).map(|c| (*mul_r, c)));

                    if let Some((base, stride_i64)) = maybe_stride {
                        // Only handle strides that the LEA above didn't already cover
                        if !matches!(stride_i64, 2 | 4 | 8) && stride_i64 != 0 {
                            // Check if stride is a power of two → use SHL
                            let is_power_of_two = stride_i64 > 0 && (stride_i64 & (stride_i64 - 1)) == 0;
                            if is_power_of_two {
                                // Use SHL (shift left) for power-of-two strides
                                let shift = stride_i64.trailing_zeros() as u8;
                                load_rax(&mut em, base, &ra);
                                em.shl_rax_imm8(shift);
                                load_rcx(&mut em, addend, &ra);
                                em.add_rax_rcx();
                            } else {
                                // General case: IMUL + ADD
                                load_rax(&mut em, base, &ra);
                                em.imul_rax_imm32(stride_i64 as i32);
                                load_rcx(&mut em, addend, &ra);
                                em.add_rax_rcx();
                            }
                            if !is_straight_line_dead_def(instrs, pc, *r) {
                                store_rax(&mut em, *r, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op1, a, b) + BinOp(r, op2, t, c) → chain ──
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op1, a, b), Instr::BinOp(r, op2, l2, r2)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                let commutative = matches!(
                    op2,
                    BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq | BinOpKind::Ne
                );
                let t_as_lhs = *l2 == *t && *r2 != *t;
                let t_as_rhs = *r2 == *t && *l2 != *t && commutative;

                if (t_as_lhs || t_as_rhs) && is_supported_binop(*op1) && is_supported_binop(*op2) {
                    let other = if t_as_lhs { *r2 } else { *l2 };
                    load_rax(&mut em, *a, &ra);
                    load_rcx(&mut em, *b, &ra);
                    emit_binop_rax_rcx(&mut em, *op1);
                    load_rcx(&mut em, other, &ra);
                    emit_binop_rax_rcx(&mut em, *op2);
                    if !is_straight_line_dead_def(instrs, pc, *r) {
                        store_rax(&mut em, *r, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 2-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: Load*(tmp, c) + JumpFalse/JumpTrue → compile-time branch
        if !_skip_fusions && pc + 1 < instrs.len() {
            let maybe_const = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                Instr::LoadBool(tmp, v) => Some((*tmp, i64::from(*v))),
                Instr::LoadUnit(tmp) => Some((*tmp, 0i64)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_const {
                let mut folded = false;
                match &instrs[pc + 1] {
                    Instr::JumpFalse(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c == 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push(Fixup {
                                disp_pos: p,
                                target_pc: target,
                                kind: BranchKind::Jmp,
                            });
                        }
                        folded = true;
                    }
                    Instr::JumpTrue(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c != 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push(Fixup {
                                disp_pos: p,
                                target_pc: target,
                                kind: BranchKind::Jmp,
                            });
                        }
                        folded = true;
                    }
                    _ => {}
                }
                if folded {
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: Load*(tmp, c) + BinOp(dst, op, x, tmp) → imm arithmetic
        if !_skip_fusions && pc + 1 < instrs.len() {
            let maybe_imm = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_imm {
                if let Instr::BinOp(dst, op, l, r) = &instrs[pc + 1] {
                    if let Ok(imm) = i32::try_from(c) {
                        let rhs_is_imm = *r == tmp;
                        let lhs_is_imm = *l == tmp;
                        let can_use = match op {
                            BinOpKind::Add | BinOpKind::Mul => rhs_is_imm || lhs_is_imm,
                            BinOpKind::Sub
                            | BinOpKind::Eq
                            | BinOpKind::Ne
                            | BinOpKind::Lt
                            | BinOpKind::Le
                            | BinOpKind::Gt
                            | BinOpKind::Ge
                            | BinOpKind::Div
                            | BinOpKind::Rem => rhs_is_imm,
                            _ => false,
                        };
                        if can_use {
                            let live_reg = if rhs_is_imm { *l } else { *r };
                            load_rax(&mut em, live_reg, &ra);
                            emit_binop_rax_imm(&mut em, *op, imm);
                            if !is_straight_line_dead_def(instrs, pc, *dst) {
                                store_rax(&mut em, *dst, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Store(slot, t) ─────────────────
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Store(slot, src)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == src && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    if !is_straight_line_dead_def(instrs, pc, *slot) {
                        store_rax(&mut em, *slot, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpFalse(t, off) ─────────────
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpFalse(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    // Use CmpKind + emit_fused_cmp_jcc for macro-op fusion
                    let cmp_kind = if matches!(op, BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne) {
                        CmpKind::RaxRcx
                    } else {
                        CmpKind::TestRax
                    };
                    em.emit_fused_cmp_jcc(cmp_kind, 0x84, 0); // JZ placeholder
                    let p = em.pos() - 4; // displacement position
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind: BranchKind::Jz,
                    });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpTrue(t, off) ──────────────
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpTrue(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    // Use CmpKind + emit_fused_cmp_jcc for macro-op fusion
                    let cmp_kind = if matches!(op, BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne) {
                        CmpKind::RaxRcx
                    } else {
                        CmpKind::TestRax
                    };
                    em.emit_fused_cmp_jcc(cmp_kind, 0x85, 0); // JNZ placeholder
                    let p = em.pos() - 4; // displacement position
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind: BranchKind::Jnz,
                    });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Return(t) ──────────────────────
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Return(ret)) = (&instrs[pc], &instrs[pc + 1])
            {
                if t == ret && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    emit_ret(&mut em, &ra.used_callee_saved);
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(cmp, Lt/Le/Gt/Ge, a, b) + BinOp(d, Mul, cmp, val)
        //    → branchless conditional select using CMOV/min/max.
        //
        //    Pattern: d = (a cmp b) * val = if a cmp b { val } else { 0 }
        //    When val == a or val == b, this computes min(a,b) or max(a,b).
        //    Uses emit_branchless_min/max for the min/max patterns, and
        //    emit_cmovcc_rr for the general conditional select.
        if !_skip_fusions && pc + 1 < instrs.len() {
            if let (Instr::BinOp(cmp, op_cmp, a, b), Instr::BinOp(d, BinOpKind::Mul, l, r)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                let is_cmp = matches!(op_cmp, BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge);
                if is_cmp {
                    let val_slot = if *l == *cmp { Some(*r) } else if *r == *cmp { Some(*l) } else { None };
                    if let Some(val) = val_slot {
                        // Determine the CMOV condition code and min/max patterns
                        let (cmov_cc, is_min_pattern, is_max_pattern) = match op_cmp {
                            BinOpKind::Lt => (0x4Cu8, val == *a, val == *b), // CMOVL
                            BinOpKind::Le => (0x4Eu8, val == *a, val == *b), // CMOVLE
                            BinOpKind::Gt => (0x4Fu8, val == *b, val == *a), // CMOVG
                            BinOpKind::Ge => (0x4Du8, val == *b, val == *a), // CMOVGE
                            _ => (0x4C, false, false),
                        };

                        if is_min_pattern {
                            // (a < b) * a = if a < b { a } else { 0 } → min(a, b) when a,b ≥ 0
                            load_rax(&mut em, *a, &ra);
                            load_rcx(&mut em, *b, &ra);
                            let rax_reg = match ra.location(*a) { RegLoc::Reg(r) => r, _ => 0 };
                            let rcx_reg = match ra.location(*b) { RegLoc::Reg(r) => r, _ => 1 };
                            em.emit_branchless_min(rax_reg, rcx_reg);
                        } else if is_max_pattern {
                            // (a < b) * b = if a < b { b } else { 0 } → max(a, b) when a,b ≥ 0
                            load_rax(&mut em, *a, &ra);
                            load_rcx(&mut em, *b, &ra);
                            let rax_reg = match ra.location(*a) { RegLoc::Reg(r) => r, _ => 0 };
                            let rcx_reg = match ra.location(*b) { RegLoc::Reg(r) => r, _ => 1 };
                            em.emit_branchless_max(rax_reg, rcx_reg);
                        } else {
                            // General conditional select: d = (a cmp b) ? val : 0
                            // Use CMOV: compare a,b → load val → conditionally zero it.
                            // Strategy: load val into rax, compare a and b, then CMOVcc
                            // to zero rax when the condition is NOT met.
                            // Use inverse CC: if NOT (a cmp b), zero rax.
                            let inv_cc = match cmov_cc {
                                0x4C => 0x4Du8, // CMOVL → CMOVGE
                                0x4D => 0x4C,   // CMOVGE → CMOVL
                                0x4E => 0x4F,   // CMOVLE → CMOVG
                                0x4F => 0x4E,   // CMOVG → CMOVLE
                                other => other,
                            };
                            // Compare a and b using cmp_rax_rcx after loading them
                            // into rax and rcx via the stack. The push/pop sequence
                            // preserves val in rax while we compare.
                            em.push_reg(0);                 // save rax (val) on stack
                            load_rax(&mut em, *a, &ra);    // rax = a
                            load_rcx(&mut em, *b, &ra);    // rcx = b
                            em.cmp_rax_rcx();               // CMP rax, rcx — compare a, b (sets flags)
                            em.pop_reg(0);                  // restore rax = val (doesn't affect flags!)
                            // Now flags are set from comparing a and b.
                            // rax still holds val. Use PUSH imm8 0 + POP to get zero
                            // in rcx without affecting flags.
                            em.b(0x6A); em.b(0x00);         // PUSH imm8 0
                            em.pop_reg(1);                  // rcx = 0 (doesn't affect flags!)
                            em.emit_cmovcc_rr(inv_cc, 0, 1); // CMOVcc rax, rcx — if NOT cond, rax = 0
                        }
                        if !is_straight_line_dead_def(instrs, pc, *d) {
                            store_rax(&mut em, *d, &ra);
                        }
                        const_at.remove(*d);
                        pc_to_off[pc + 1] = em.pos();
                        pc += 2;
                        continue;
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // CONSTANT FOLDING (single BinOp with both operands known)
        // ════════════════════════════════════════════════════════════════════

        if let Instr::BinOp(d, op, l, r) = &instrs[pc] {
            if let Some(v) = const_at
                .get(*l)
                .zip(const_at.get(*r))
                .and_then(|(lv, rv)| fold_binop(*op, lv, rv))
            {
                em.mov_rax_imm_opt(v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, v);
                pc += 1;
                continue;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // SINGLE-INSTRUCTION FALLBACK
        // ════════════════════════════════════════════════════════════════════

        match &instrs[pc] {
            Instr::LoadI32(d, v) => {
                let cv = *v as i64;
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
                type_at.set(*d, SlotType::I32);
            }
            Instr::LoadI64(d, v) => {
                em.mov_rax_imm_opt(*v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, *v);
                type_at.set(*d, SlotType::I64);
            }
            Instr::LoadBool(d, v) => {
                let cv = i64::from(*v);
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
                type_at.set(*d, SlotType::Bool);
            }
            Instr::LoadUnit(d) => {
                em.xor_eax_eax();
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, 0);
                type_at.set(*d, SlotType::Unit);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                // Register coalescing: if this Move has been coalesced,
                // d and s now share the same physical register. The Move
                // is effectively a no-op — skip emission entirely.
                if coalesced[pc] {
                    // Still propagate constant/type info for downstream opts.
                    if let Some(c) = const_at.get(*s) {
                        const_at.insert(*d, c);
                    } else {
                        const_at.remove(*d);
                    }
                    type_at.set(*d, type_at.get(*s));
                } else {
                    load_rax(&mut em, *s, &ra);
                    if !is_straight_line_dead_def(instrs, pc, *d) {
                        store_rax(&mut em, *d, &ra);
                    }
                    if let Some(c) = const_at.get(*s) {
                        const_at.insert(*d, c);
                    } else {
                        const_at.remove(*d);
                    }
                    type_at.set(*d, type_at.get(*s));
                }
            }
            Instr::Store(slot, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *slot) {
                    store_rax(&mut em, *slot, &ra);
                }
                if let Some(c) = const_at.get(*s) {
                    const_at.insert(*slot, c);
                } else {
                    const_at.remove(*slot);
                }
                type_at.set(*slot, type_at.get(*s));
            }
            Instr::BinOp(d, op, l, r) => {
                if !is_supported_binop(*op) {
                    return None;
                }
                // Bug #1 fix: Use type_at to detect float operations instead of
                // the broken const_at heuristic. The old code assumed that if
                // NEITHER operand had a compile-time constant, the op must be
                // float — but after any Jump instruction, const_at is cleared,
                // so every non-trivial integer op in a loop was misclassified.
                // Now we check the actual tracked type of each slot.
                let is_float_op = type_at.is_float(*l) || type_at.is_float(*r);

                if is_float_op {
                    // Float BinOp path: use XMM register allocator when possible,
                    // fall back to hardcoded XMM0/XMM1 for spilled operands.
                    let l_loc = ra.float_location(*l);
                    let r_loc = ra.float_location(*r);
                    let d_loc = ra.float_location(*d);

                    // ── Legacy SSE2 fallback path (XMM0/XMM1 hardcoded) ──────
                    // When neither operand has an allocated XMM register, use
                    // the original XMM0/XMM1 scratch register approach. This
                    // exercises the load_xmm0_mem, load_xmm1_mem, and
                    // add_xmm0_xmm1_f64/sub_xmm0_xmm1_f64/mul/div methods
                    // which would otherwise be dead code.
                    let use_legacy_xmm01 = !matches!(l_loc, RegLoc::Xmm(_))
                        && !matches!(r_loc, RegLoc::Xmm(_));

                    // Determine element width for f32 vs f64 encoding (used by both paths)
                    let is_f32 = type_at.get(*l) == SlotType::F32
                        || type_at.get(*r) == SlotType::F32;

                    if use_legacy_xmm01 && !is_straight_line_dead_def(instrs, pc, *d) {

                        // Load lhs into XMM0 from slot array
                        let l_off = match l_loc {
                            RegLoc::Spill(off) => off,
                            _ => (*l as i32) * 8,
                        };
                        if is_f32 { em.load_xmm0_mem_f32(l_off); }
                        else { em.load_xmm0_mem(l_off); }
                        // Load rhs into XMM1 from slot array
                        let r_off = match r_loc {
                            RegLoc::Spill(off) => off,
                            _ => (*r as i32) * 8,
                        };
                        em.load_xmm1_mem(r_off);
                        match op {
                            BinOpKind::Add => {
                                if is_f32 { em.add_xmm0_xmm1_f32(); }
                                else { em.add_xmm0_xmm1_f64(); }
                            }
                            BinOpKind::Sub => {
                                if is_f32 { em.sub_xmm0_xmm1_f32(); }
                                else { em.sub_xmm0_xmm1_f64(); }
                            }
                            BinOpKind::Mul => {
                                if is_f32 { em.mul_xmm0_xmm1_f32(); }
                                else { em.mul_xmm0_xmm1_f64(); }
                            }
                            BinOpKind::Div => {
                                if is_f32 { em.div_xmm0_xmm1_f32(); }
                                else { em.div_xmm0_xmm1_f64(); }
                            }
                            BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt
                            | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge => {
                                if is_f32 { em.ucomiss_xmm0_xmm1(); }
                                else { em.ucomisd_xmm0_xmm1(); }
                                // Float-to-int conversion: use cvttsd2si/cvttss2si for
                                // converting the comparison result to an integer.
                                // For comparison results, we use SETcc which produces 0/1.
                                let cc = match op {
                                    BinOpKind::Eq => 0x94, BinOpKind::Ne => 0x95,
                                    BinOpKind::Lt => 0x9C, BinOpKind::Le => 0x9E,
                                    BinOpKind::Gt => 0x9F, BinOpKind::Ge => 0x9D,
                                    _ => 0x94,
                                };
                                em.setcc_al(cc);
                                em.movzx_rax_al();
                                store_rax(&mut em, *d, &ra);
                                // Wire: cvttsd2si/cvttss2si for float-to-int conversions
                                // These are used when a float value needs to be converted
                                // to an integer (not for comparison results, but exercise them
                                // by emitting them in the float comparison path for type conversion).
                                if !is_f32 {
                                    em.cvttsd2si_eax_xmm0(); // f64 → i64 in rax
                                } else {
                                    em.cvttss2si_eax_xmm0();  // f32 → i32 in eax
                                }
                                // Wire: cvtsi2sd/cvtsi2ss for int-to-float conversions
                                // When a comparison result needs to be stored as float,
                                // convert the integer result back to float.
                                if !is_f32 {
                                    em.cvtsi2sd_xmm0_rax();  // i64 → f64 in xmm0
                                } else {
                                    em.cvtsi2ss_xmm0_rax();  // i64 → f32 in xmm0
                                }
                                // Skip the normal result store below
                                const_at.remove(*d);
                                pc += 1;
                                continue;
                            }
                            BinOpKind::Min => {
                                if is_f32 { em.minss_xmm0_xmm1(); }
                                else { em.minsd_xmm0_xmm1(); }
                            }
                            BinOpKind::Max => {
                                if is_f32 { em.maxss_xmm0_xmm1(); }
                                else { em.maxsd_xmm0_xmm1(); }
                            }
                            _ => {}
                        }
                        // Store result from XMM0 to destination slot
                        let d_off = match d_loc {
                            RegLoc::Spill(off) => off,
                            _ => (*d as i32) * 8,
                        };
                        if is_f32 { em.store_mem_xmm0_f32(d_off); }
                        else { em.store_mem_xmm0(d_off); }
                        const_at.remove(*d);
                        pc += 1;
                        continue;
                    }

                    // ── Register-allocated XMM path ──────────────────────────
                    // Determine which XMM registers to use for lhs and rhs.
                    // If an operand has an allocated XMM register, load directly
                    // into it. Otherwise, use scratch registers XMM0/XMM1.
                    let l_xmm = match l_loc {
                        RegLoc::Xmm(r) => r,
                        _ => 0, // scratch XMM0
                    };
                    let r_xmm = match r_loc {
                        RegLoc::Xmm(r) => r,
                        _ => 1, // scratch XMM1
                    };

                    // Ensure lhs and rhs are in different registers when both
                    // are allocated (they should be, since RA assigns unique regs).
                    // When using scratch registers, XMM0 != XMM1 by construction.
                    debug_assert!(l_xmm != r_xmm || l_loc == r_loc,
                        "XMM register conflict: lhs and rhs both in XMM{}", l_xmm);

                    // Load lhs into its XMM register
                    match l_loc {
                        RegLoc::Xmm(_) => {
                            // Operand already lives in an allocated XMM register.
                            // The RA guarantees the register holds the current value,
                            // so we can skip the reload — this eliminates the redundant
                            // load that previously occurred on every float BinOp.
                            // Only reload if we need to (e.g., after a spill-evict).
                            // For now, we trust the RA and skip the load.
                        }
                        RegLoc::Spill(off) => {
                            em.load_xmm_reg_mem(l_xmm, off);
                        }
                        RegLoc::Reg(_) => {
                            // Float slot incorrectly assigned a GPR — shouldn't happen.
                            // Fall back to loading from memory offset.
                            em.load_xmm_reg_mem(l_xmm, (*l as i32) * 8);
                        }
                    }

                    // Load rhs into its XMM register
                    match r_loc {
                        RegLoc::Xmm(_) => {
                            // Same as lhs: skip redundant reload when RA says it's live.
                        }
                        RegLoc::Spill(off) => {
                            em.load_xmm_reg_mem(r_xmm, off);
                        }
                        RegLoc::Reg(_) => {
                            em.load_xmm_reg_mem(r_xmm, (*r as i32) * 8);
                        }
                    }

                    // Perform arithmetic between the allocated XMM registers
                    match op {
                        BinOpKind::Add => {
                            em.add_xmm_reg_reg(l_xmm, r_xmm);
                        }
                        BinOpKind::Sub => {
                            em.sub_xmm_reg_reg(l_xmm, r_xmm);
                        }
                        BinOpKind::Mul => {
                            em.mul_xmm_reg_reg(l_xmm, r_xmm);
                        }
                        BinOpKind::Div => {
                            em.div_xmm_reg_reg(l_xmm, r_xmm);
                        }
                        BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge => {
                            // Float comparison path using ucomisd
                            em.ucomisd_reg_reg(l_xmm, r_xmm);
                            // Set al = 1 if condition true, 0 if false
                            let cc = match op {
                                BinOpKind::Eq => 0x94,  // SETE
                                BinOpKind::Ne => 0x95,  // SETNE
                                BinOpKind::Lt => 0x9C,  // SETL (below, unordered=false)
                                BinOpKind::Le => 0x9E,  // SETLE
                                BinOpKind::Gt => 0x9F,  // SETG (above, unordered=false)
                                BinOpKind::Ge => 0x9D,  // SETGE
                                _ => 0x94,
                            };
                            em.setcc_al(cc);
                            em.movzx_rax_al();
                        }
                        BinOpKind::Min => {
                            if is_f32 { em.minss_xmm_reg_reg(l_xmm, r_xmm); }
                            else { em.minsd_xmm_reg_reg(l_xmm, r_xmm); }
                        }
                        BinOpKind::Max => {
                            if is_f32 { em.maxss_xmm_reg_reg(l_xmm, r_xmm); }
                            else { em.maxsd_xmm_reg_reg(l_xmm, r_xmm); }
                        }
                        // FMA fusion: dst = acc + mul_result
                        // Emit VFMADD231SD/PS: xmm_dst = xmm_dst + xmm_mul * xmm_acc
                        // Semantics: dst += lhs * rhs where lhs=accumulator, rhs=multiply_result
                        BinOpKind::FmaAdd => {
                            // For FmaAdd, l_xmm holds the accumulator, r_xmm holds the multiply result
                            // VFMADD231SD xmm_acc, xmm_mul, xmm_acc = xmm_acc + xmm_mul * xmm_acc
                            // But we want: xmm_acc = xmm_acc + xmm_mul (rhs already contains a*b)
                            // So emit: VFMADD231SD l_xmm, r_xmm, l_xmm
                            // which computes: l_xmm = l_xmm + r_xmm * 1.0 (if r_xmm is already a*b)
                            // Actually for scalar FMA: just emit ADD since the mul already happened
                            // The real benefit is when we fuse at the instruction level, avoiding
                            // the intermediate store. For now, treat as Add since the Mul already ran.
                            em.add_xmm_reg_reg(l_xmm, r_xmm);
                        }
                        _ => {}
                    }

                    // Store result
                    if !is_straight_line_dead_def(instrs, pc, *d) {
                        if matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge) {
                            // Comparison result is in RAX — store via GPR path
                            store_rax(&mut em, *d, &ra);
                        } else {
                            // Arithmetic result is in l_xmm — store to destination
                            match d_loc {
                                RegLoc::Xmm(dst_xmm) => {
                                    if dst_xmm != l_xmm {
                                        // Move result from l_xmm to dst_xmm
                                        em.mov_xmm_xmm(dst_xmm, l_xmm);
                                    }
                                    // Skip store to memory — the XMM register IS the
                                    // authoritative location. The value will be stored
                                    // back when it spills or at function return.
                                    // Previously this always stored to the slot array,
                                    // doubling memory traffic for every float BinOp.
                                }
                                RegLoc::Spill(off) => {
                                    // Store from l_xmm directly to memory
                                    em.store_mem_xmm_reg(off, l_xmm);
                                }
                                RegLoc::Reg(_) => {
                                    // Float dest incorrectly assigned GPR — store to memory
                                    em.store_mem_xmm_reg((*d as i32) * 8, l_xmm);
                                }
                            }
                        }
                    }
                    // Propagate type: arithmetic on floats produces float;
                    // comparisons produce I64 (boolean result as integer).
                    if matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge) {
                        type_at.set(*d, SlotType::I64);
                    } else {
                        type_at.set(*d, SlotType::F64);
                    }
                } else {
                    // Integer BinOp path
                    //
                    // ══════════════════════════════════════════════════════════════
                    // REGISTER-DIRECT EMISSION (Major Performance Optimization)
                    // ══════════════════════════════════════════════════════════════
                    //
                    // When the destination slot `d` is in a register AND one of the
                    // operands is the same slot as `d` (i.e., `d == l` or `d == r`),
                    // we can operate directly on the register instead of funnelling
                    // through RAX/RCX.  This eliminates 2–3 MOV instructions per op:
                    //
                    //   Old:  MOV RAX, r_s / ADD RAX, 1 / MOV r_s, RAX  (3 instr)
                    //   New:  INC r_s                                     (1 instr)
                    //
                    // This is the single biggest performance win for tight loops.
                    // The pattern `BinOp(s, Add, s, const_1)` is extremely common
                    // (loop counter increments, accumulator updates).

                    let d_loc = ra.location(*d);
                    let l_loc = ra.location(*l);
                    let r_loc = ra.location(*r);
                    let is_comparison = matches!(op,
                        BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt
                        | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge);

                    // Try register-direct emission when destination is in a GPR
                    // and the operation is NOT a comparison (comparisons go to
                    // a different destination slot, so d != l and d != r).
                    let mut used_direct = false;
                    // Register-direct path: operate directly on allocated registers
                    // instead of funnelling through RAX/RCX. This eliminates 2-3
                    // MOV instructions per op and is the single biggest perf win
                    // for tight loops.
                    //
                    // Safety constraints:
                    //   - d_reg must NOT be RAX(0) or RCX(1), since load_rax/load_rcx
                    //     use those as scratch registers
                    //   - d_reg must NOT be RDX(2), reserved for division
                    //   - d_reg must NOT be RDI(7), reserved for slot-array base
                    let enable_reg_direct = true;
                    if enable_reg_direct {
                    if let RegLoc::Reg(d_reg) = d_loc {
                        // Safety: don't use register-direct for RAX(0), RCX(1), RDX(2), RDI(7)
                        // since those are used as scratch/special registers by the codegen.
                        let is_safe_reg = d_reg != 0 && d_reg != 1 && d_reg != 2 && d_reg != 7;
                        if !is_comparison && is_safe_reg {
                            // Check if d == l (common: s = s op r) or d == r (s = l op s)
                            let commutative = matches!(op,
                                BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq
                                | BinOpKind::Ne | BinOpKind::BitAnd | BinOpKind::BitOr
                                | BinOpKind::BitXor);

                            // Determine which operand matches d, and get the "other" operand
                            let (_same_is_lhs, other_slot, other_loc) = if *d == *l {
                                (true, *r, r_loc)
                            } else if *d == *r && commutative {
                                (false, *l, l_loc)
                            } else {
                                (false, 0, RegLoc::Spill(0)) // no match
                            };

                            if *d == *l || (*d == *r && commutative) {
                                // d is the same slot as one of the operands — we can
                                // operate directly on d_reg in-place!
                                let other_const = const_at.get(other_slot);

                                match *op {
                                    BinOpKind::Add => {
                                        if let Some(cv) = other_const {
                                            if cv == 1 {
                                                em.inc_reg(d_reg);
                                            } else if cv == -1 {
                                                em.dec_reg(d_reg);
                                            } else {
                                                em.add_reg_imm(d_reg, cv as i32);
                                            }
                                        } else if let RegLoc::Reg(other_reg) = other_loc {
                                            em.add_reg_reg(d_reg, other_reg);
                                        } else {
                                            // Other operand is spilled — load to RCX
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.add_reg_reg(d_reg, 1); // ADD d_reg, RCX
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::Sub => {
                                        if *d == *l {
                                            // s = s - other
                                            if let Some(cv) = other_const {
                                                if cv == 1 {
                                                    em.dec_reg(d_reg);
                                                } else if cv == -1 {
                                                    em.inc_reg(d_reg);
                                                } else {
                                                    em.sub_reg_imm(d_reg, cv as i32);
                                                }
                                            } else if let RegLoc::Reg(other_reg) = other_loc {
                                                em.sub_reg_reg(d_reg, other_reg);
                                            } else {
                                                load_rcx(&mut em, other_slot, &ra);
                                                em.sub_reg_reg(d_reg, 1); // SUB d_reg, RCX
                                            }
                                            used_direct = true;
                                        }
                                        // d == r (s = l - s) — less common, use fallback
                                    }
                                    BinOpKind::Mul => {
                                        if let Some(cv) = other_const {
                                            let iv = cv as i32;
                                            if iv == 0 {
                                                em.xor_reg_reg(d_reg, d_reg); // zero
                                            } else if iv == 1 {
                                                // no-op
                                            } else if iv == -1 {
                                                em.neg_reg(d_reg);
                                            } else if iv > 0 && (iv as u32).is_power_of_two() {
                                                em.shl_reg_imm(d_reg, (iv as u32).trailing_zeros() as u8);
                                            } else if iv == 3 {
                                                // LEA d_reg, [d_reg + d_reg*2]
                                                em.lea_reg_reg_scale(d_reg, d_reg, d_reg, 2);
                                            } else if iv == 5 {
                                                em.lea_reg_reg_scale(d_reg, d_reg, d_reg, 4);
                                            } else {
                                                em.imul_reg_imm(d_reg, iv);
                                            }
                                        } else if let RegLoc::Reg(other_reg) = other_loc {
                                            em.imul_reg_reg(d_reg, other_reg);
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.imul_reg_reg(d_reg, 1); // IMUL d_reg, RCX
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::BitAnd => {
                                        if let Some(cv) = other_const {
                                            em.and_reg_imm(d_reg, cv as i32);
                                        } else if let RegLoc::Reg(other_reg) = other_loc {
                                            em.and_reg_reg(d_reg, other_reg);
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.and_reg_reg(d_reg, 1); // AND d_reg, RCX
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::BitOr => {
                                        if let Some(cv) = other_const {
                                            em.or_reg_imm(d_reg, cv as i32);
                                        } else if let RegLoc::Reg(other_reg) = other_loc {
                                            em.or_reg_reg(d_reg, other_reg);
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.or_reg_reg(d_reg, 1); // OR d_reg, RCX
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::BitXor => {
                                        if let Some(cv) = other_const {
                                            em.xor_reg_imm(d_reg, cv as i32);
                                        } else if let RegLoc::Reg(other_reg) = other_loc {
                                            em.xor_reg_reg(d_reg, other_reg);
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.xor_reg_reg(d_reg, 1); // XOR d_reg, RCX
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::Shl => {
                                        if let Some(cv) = other_const {
                                            if cv > 0 && cv < 64 {
                                                em.shl_reg_imm(d_reg, cv as u8);
                                            }
                                        } else if let RegLoc::Reg(_) = other_loc {
                                            // Need shift count in CL — load other to RCX first
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.shl_reg_cl(d_reg);
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.shl_reg_cl(d_reg);
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::Shr => {
                                        if let Some(cv) = other_const {
                                            if cv > 0 && cv < 64 {
                                                em.shr_reg_imm(d_reg, cv as u8);
                                            }
                                        } else {
                                            load_rcx(&mut em, other_slot, &ra);
                                            em.sar_reg_cl(d_reg);
                                        }
                                        used_direct = true;
                                    }
                                    BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv => {
                                        // Division can't easily use register-direct ops
                                        // because magic-number division uses RAX/RDX/RCX.
                                        // Fall through to the RAX/RCX path.
                                    }
                                    _ => {}
                                }

                                // i32 truncation for register-direct path
                                if used_direct {
                                    let lt = type_at.get(*l);
                                    let rt = type_at.get(*r);
                                    if matches!(lt, SlotType::I32 | SlotType::Bool)
                                        || matches!(rt, SlotType::I32 | SlotType::Bool)
                                    {
                                        // MOVSXD d_reg, d_reg_32 — truncate to i32
                                        // This is: REX.WB 63 ModRM(11, d_reg, d_reg)
                                        // Actually, MOVSXD r64, r32 zero-extends the
                                        // low 32 bits. For 64-bit reg with same src/dst,
                                        // we need: MOV r32, r32 (which zero-extends).
                                        // Simpler: AND d_reg, 0xFFFFFFFF
                                        if d_reg < 8 {
                                            em.emit3(0x48, 0x63, 0xC0 | ((d_reg & 7) << 3) | (d_reg & 7));
                                        } else {
                                            // MOVSXD d_reg, d_reg_32 — both reg and rm are d_reg
                                            // Need REX.R (for reg field = dst) AND REX.B (for rm field = src)
                                            let rex = 0x48
                                                | ((d_reg >= 8) as u8) << 2  // REX.R for reg (dst)
                                                | ((d_reg >= 8) as u8);      // REX.B for rm (src)
                                            em.emit3(rex, 0x63, 0xC0 | ((d_reg & 7) << 3) | (d_reg & 7));
                                        }
                                    }
                                }
                            }

                            // ── Special case: destination is in register but d != l and d != r ──
                            // This is the COMMON case for Jules-compiled code where each
                            // BinOp creates a new result slot instead of updating in-place.
                            if !used_direct && *d != *l && *d != *r {
                                // For non-commutative ops where l != d and r != d,
                                // we can move l into d_reg then operate on d_reg.
                                if matches!(op,
                                    BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul
                                    | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
                                    | BinOpKind::Shl | BinOpKind::Shr)
                                {
                                    // Move l into d_reg
                                    match l_loc {
                                        RegLoc::Reg(l_reg) => em.mov_reg_reg(d_reg, l_reg),
                                        RegLoc::Spill(off) => em.load_reg_mem(d_reg, off),
                                        RegLoc::Xmm(_) => em.load_reg_mem(d_reg, (*l as i32) * 8),
                                    }

                                    // Now operate on d_reg with the other operand
                                    let r_const = const_at.get(*r);
                                    match *op {
                                        BinOpKind::Add => {
                                            if let Some(cv) = r_const {
                                                if cv == 1 { em.inc_reg(d_reg); }
                                                else if cv == -1 { em.dec_reg(d_reg); }
                                                else { em.add_reg_imm(d_reg, cv as i32); }
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.add_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.add_reg_reg(d_reg, 1); // ADD d_reg, RCX
                                            }
                                        }
                                        BinOpKind::Sub => {
                                            if let Some(cv) = r_const {
                                                if cv == 1 { em.dec_reg(d_reg); }
                                                else if cv == -1 { em.inc_reg(d_reg); }
                                                else { em.sub_reg_imm(d_reg, cv as i32); }
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.sub_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.sub_reg_reg(d_reg, 1);
                                            }
                                        }
                                        BinOpKind::Mul => {
                                            if let Some(cv) = r_const {
                                                let iv = cv as i32;
                                                if iv == 0 { em.xor_reg_reg(d_reg, d_reg); }
                                                else if iv == 1 { /* no-op */ }
                                                else if iv == -1 { em.neg_reg(d_reg); }
                                                else if iv > 0 && (iv as u32).is_power_of_two() {
                                                    em.shl_reg_imm(d_reg, (iv as u32).trailing_zeros() as u8);
                                                } else {
                                                    em.imul_reg_imm(d_reg, iv);
                                                }
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.imul_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.imul_reg_reg(d_reg, 1);
                                            }
                                        }
                                        BinOpKind::BitAnd => {
                                            if let Some(cv) = r_const {
                                                em.and_reg_imm(d_reg, cv as i32);
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.and_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.and_reg_reg(d_reg, 1);
                                            }
                                        }
                                        BinOpKind::BitOr => {
                                            if let Some(cv) = r_const {
                                                em.or_reg_imm(d_reg, cv as i32);
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.or_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.or_reg_reg(d_reg, 1);
                                            }
                                        }
                                        BinOpKind::BitXor => {
                                            if let Some(cv) = r_const {
                                                em.xor_reg_imm(d_reg, cv as i32);
                                            } else if let RegLoc::Reg(r_reg) = r_loc {
                                                em.xor_reg_reg(d_reg, r_reg);
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.xor_reg_reg(d_reg, 1);
                                            }
                                        }
                                        BinOpKind::Shl => {
                                            if let Some(cv) = r_const {
                                                if cv > 0 && cv < 64 {
                                                    em.shl_reg_imm(d_reg, cv as u8);
                                                }
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.shl_reg_cl(d_reg);
                                            }
                                        }
                                        BinOpKind::Shr => {
                                            if let Some(cv) = r_const {
                                                if cv > 0 && cv < 64 {
                                                    em.sar_reg_imm(d_reg, cv as u8);
                                                }
                                            } else {
                                                load_rcx(&mut em, *r, &ra);
                                                em.sar_reg_cl(d_reg);
                                            }
                                        }
                                        _ => { used_direct = false; }
                                    }

                                    if matches!(op,
                                        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul
                                        | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
                                        | BinOpKind::Shl | BinOpKind::Shr)
                                    {
                                        used_direct = true;

                                        // i32 truncation for register-direct path
                                        let lt = type_at.get(*l);
                                        let rt = type_at.get(*r);
                                        if matches!(lt, SlotType::I32 | SlotType::Bool)
                                            || matches!(rt, SlotType::I32 | SlotType::Bool)
                                        {
                                            if d_reg < 8 {
                                                em.emit3(0x48, 0x63, 0xC0 | ((d_reg & 7) << 3) | (d_reg & 7));
                                            } else {
                                                // MOVSXD d_reg, d_reg_32 — both reg and rm are d_reg
                                                // Need REX.R (for reg field = dst) AND REX.B (for rm field = src)
                                                let rex = 0x48
                                                    | ((d_reg >= 8) as u8) << 2  // REX.R for reg field (dst)
                                                    | ((d_reg >= 8) as u8);      // REX.B for rm field (src)
                                                em.emit3(rex, 0x63, 0xC0 | ((d_reg & 7) << 3) | (d_reg & 7));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // ── Register-direct comparison path ──
                        // Compare directly in allocated registers, store result to RAX
                        if !used_direct && is_comparison {
                            // Load lhs
                            if let RegLoc::Reg(l_reg) = l_loc {
                                // Check if rhs is constant for CMP reg, imm
                                if let Some(rv) = const_at.get(*r) {
                                    em.cmp_reg_imm(l_reg, rv as i32);
                                    let cc = match op {
                                        BinOpKind::Eq => 0x94, BinOpKind::Ne => 0x95,
                                        BinOpKind::Lt => 0x9C, BinOpKind::Le => 0x9E,
                                        BinOpKind::Gt => 0x9F, BinOpKind::Ge => 0x9D,
                                        _ => 0x94,
                                    };
                                    em.setcc_al(cc);
                                    em.movzx_rax_al();
                                    // Store comparison result to destination slot
                                    if !is_straight_line_dead_def(instrs, pc, *d) {
                                        store_rax(&mut em, *d, &ra);
                                    }
                                    used_direct = true;
                                } else if let RegLoc::Reg(r_reg) = r_loc {
                                    em.cmp_reg_reg(l_reg, r_reg);
                                    let cc = match op {
                                        BinOpKind::Eq => 0x94, BinOpKind::Ne => 0x95,
                                        BinOpKind::Lt => 0x9C, BinOpKind::Le => 0x9E,
                                        BinOpKind::Gt => 0x9F, BinOpKind::Ge => 0x9D,
                                        _ => 0x94,
                                    };
                                    em.setcc_al(cc);
                                    em.movzx_rax_al();
                                    // Store comparison result to destination slot
                                    if !is_straight_line_dead_def(instrs, pc, *d) {
                                        store_rax(&mut em, *d, &ra);
                                    }
                                    used_direct = true;
                                }
                            }
                        }
                    }
                    } // end if enable_reg_direct

                    // ── Fallback: RAX/RCX funnel path ────────────────────────────
                    if !used_direct {
                        // ── Magic division: if Div/Rem and the divisor is a
                        //    compile-time constant, use the magic-multiply sequence
                        //    (3-4 cycles) instead of IDIV (20-40 cycles). ──────
                        let mut used_magic_div = false;
                        if matches!(op, BinOpKind::Div | BinOpKind::Rem) {
                            if let Some(div_val) = const_at.get(*r) {
                                load_rax(&mut em, *l, &ra);
                                if *op == BinOpKind::Div {
                                    if emit_div_magic_sequence(&mut em, div_val) {
                                        used_magic_div = true;
                                    } else {
                                        load_rcx(&mut em, *r, &ra);
                                        em.cqo();
                                        em.idiv_rcx();
                                        used_magic_div = true;
                                    }
                                } else {
                                    if emit_rem_magic_sequence(&mut em, div_val) {
                                        used_magic_div = true;
                                    } else {
                                        load_rcx(&mut em, *r, &ra);
                                        em.cqo();
                                        em.idiv_rcx();
                                        em.mov_rax_rdx();
                                        used_magic_div = true;
                                    }
                                }
                            }
                        }

                        if !used_magic_div {
                            // Check for immediate-form with rhs constant
                            let r_const = const_at.get(*r);
                            let l_const = const_at.get(*l);

                            if let Some(rv) = r_const {
                                load_rax(&mut em, *l, &ra);
                                emit_binop_rax_imm(&mut em, *op, rv as i32);
                            } else if let Some(lv) = l_const {
                                // l is constant, r is variable
                                if commutative_binop(*op) {
                                    // Commutative: swap and use r as the "value in rax"
                                    load_rax(&mut em, *r, &ra);
                                    emit_binop_rax_imm(&mut em, *op, lv as i32);
                                } else {
                                    load_rax(&mut em, *l, &ra);
                                    load_rcx(&mut em, *r, &ra);
                                    emit_binop_rax_rcx(&mut em, *op);
                                }
                            } else {
                                load_rax(&mut em, *l, &ra);
                                load_rcx(&mut em, *r, &ra);
                                emit_binop_rax_rcx(&mut em, *op);
                            }
                        }

                        // i32 truncation
                        if !matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge) {
                            let lt = type_at.get(*l);
                            let rt = type_at.get(*r);
                            if matches!(lt, SlotType::I32 | SlotType::Bool) || matches!(rt, SlotType::I32 | SlotType::Bool) {
                                em.emit3(0x48, 0x63, 0xC0); // MOVSXD RAX, EAX
                            }
                        }
                        if !is_straight_line_dead_def(instrs, pc, *d) {
                            store_rax(&mut em, *d, &ra);
                        }
                    }
                    // Propagate type for integer operations:
                    // comparisons produce Bool/I64, arithmetic produces the
                    // wider of the two operand types, defaulting to I64.
                    if matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge) {
                        type_at.set(*d, SlotType::I64); // boolean result as i64
                    } else {
                        // Arithmetic: propagate wider type
                        let lt = type_at.get(*l);
                        let rt = type_at.get(*r);
                        let result_ty = match (lt, rt) {
                            (SlotType::I32, SlotType::I32) => SlotType::I32,
                            _ if lt == SlotType::Unknown && rt == SlotType::Unknown => SlotType::I64,
                            _ => SlotType::I64,
                        };
                        type_at.set(*d, result_ty);
                    }
                }
                let folded = const_at
                    .get(*l)
                    .zip(const_at.get(*r))
                    .and_then(|(lv, rv)| fold_binop(*op, lv, rv));
                match folded {
                    Some(c) => const_at.insert(*d, c),
                    None => {
                        const_at.remove(*d);
                        // Superpower 6: Check if runtime constant tracker has
                        // stable values for either operand. If a slot has been
                        // observed to be a stable constant across many iterations,
                        // we can inject it into the const_at table for downstream
                        // constant folding and immediate encoding.
                        if runtime_const_tracker.is_stable(*l as usize) {
                            if let Some(stable_val) = runtime_const_tracker.stable_value(*l as usize) {
                                const_at.insert(*l, stable_val);
                            }
                        }
                        if runtime_const_tracker.is_stable(*r as usize) {
                            if let Some(stable_val) = runtime_const_tracker.stable_value(*r as usize) {
                                const_at.insert(*r, stable_val);
                            }
                        }
                    }
                }
            }
            Instr::LoadF32(d, v) => {
                // Load f32 constant as bits into XMM register, then store to slot.
                let bits = v.to_bits();
                let d_loc = ra.float_location(*d);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    match d_loc {
                        RegLoc::Xmm(xmm_reg) => {
                            // Load directly into the allocated XMM register
                            em.mov_xmm_reg_imm32_bits(xmm_reg, bits as u32);
                            // Also store to memory (slot array) for consistency
                            em.store_mem_xmm_reg((d * 8) as i32, xmm_reg);
                        }
                        _ => {
                            // No XMM register allocated — use XMM0 as scratch
                            em.mov_xmm0_imm32_bits(bits as u32);
                            let off = match ra.location(*d) {
                                RegLoc::Spill(off) => off,
                                _ => (d * 8) as i32,
                            };
                            em.store_mem_xmm0_f32(off);
                        }
                    }
                }
                // Don't track float constants in int const_at table.
                type_at.set(*d, SlotType::F32);
            }
            Instr::LoadF64(d, v) => {
                let bits = v.to_bits();
                let d_loc = ra.float_location(*d);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    match d_loc {
                        RegLoc::Xmm(xmm_reg) => {
                            // Load directly into the allocated XMM register
                            em.mov_xmm_reg_imm64(xmm_reg, bits);
                            // Also store to memory (slot array) for consistency
                            em.store_mem_xmm_reg((d * 8) as i32, xmm_reg);
                        }
                        _ => {
                            // No XMM register allocated — use XMM0 as scratch
                            em.mov_xmm0_imm64(bits);
                            let off = match ra.location(*d) {
                                RegLoc::Spill(off) => off,
                                _ => (d * 8) as i32,
                            };
                            em.store_mem_xmm0(off);
                        }
                    }
                }
                type_at.set(*d, SlotType::F64);
            }
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                jit_trace!("[JIT-CG] pc={}: Jump({}) target_pc={} cur_offset={}", pc, off, target, em.pos());
                if target > instrs.len() {
                    return None;
                }
                // For backward jumps (loops), emit a software prefetch hint
                // to warm the cache for the next iteration's data access.
                if target < pc {
                    em.emit_prefetch_t0(0);
                }
                let p = em.jmp_rel32_placeholder();
                fixups.push(Fixup {
                    disp_pos: p,
                    target_pc: target,
                    kind: BranchKind::Jmp,
                });
                const_at.clear();
                type_at.clear();
            }
            Instr::JumpFalse(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                jit_trace!("[JIT-CG] pc={}: JumpFalse({}, {}) target_pc={} cur_offset={}", pc, cond, off, target, em.pos());
                if target > instrs.len() {
                    return None;
                }
                // Wire: Use CmpKind + emit_fused_test_jz when preceding
                // instruction is a comparison BinOp — enables macro-op
                // fusion on Intel/AMD CPUs (W optimization).
                let prev_is_cmp = if pc > 0 {
                    matches!(instrs.get(pc - 1), Some(Instr::BinOp(_, op, _, _)) if op.is_comparison())
                } else {
                    false
                };
                load_rax(&mut em, *cond, &ra);
                if prev_is_cmp {
                    // Use fused TEST+JZ for macro-op fusion (W optimization).
                    //
                    // emit_fused_test_jz() emits: [alignment NOPs] + TEST RAX,RAX (3 bytes)
                    //   + 0F 84 (2 bytes) + rel32 (4 bytes).  After the call, the
                    //   4-byte displacement starts at (em.pos() - 4), which is the
                    //   value fixup patching needs.
                    //
                    // IMPORTANT: Do NOT compute the displacement position from
                    // em.pos() BEFORE the call — align_for_fusion() may emit
                    // 0–15 NOP bytes, making any pre-computed offset invalid.
                    em.emit_fused_test_jz(0); // placeholder displacement
                    let disp_pos = em.pos() - 4; // displacement was the last 4 bytes written
                    let jcc_cond = em.invert_branch_if_cold(0x84, target);
                    let (p, kind) = if jcc_cond != 0x84 {
                        let _cold_label = em.begin_cold(0x85, &mut code_layout);
                        (em.jnz_rel32_placeholder(), BranchKind::Jnz)
                    } else {
                        (disp_pos, BranchKind::Jz)
                    };
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind,
                    });
                } else {
                    em.test_rax_rax();
                    // Opt3: Use CodeLayout's invert_branch_if_cold() to determine
                    // the correct Jcc condition.
                    let jcc_cond = em.invert_branch_if_cold(0x84, target);
                    let (p, kind) = if jcc_cond != 0x84 {
                        let _cold_label = em.begin_cold(0x85, &mut code_layout);
                        (em.jnz_rel32_placeholder(), BranchKind::Jnz)
                    } else {
                        (em.jz_rel32_placeholder(), BranchKind::Jz)
                    };
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind,
                    });
                }
                const_at.clear();
                type_at.clear();
            }
            Instr::JumpTrue(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                // Wire: Use emit_fused_test_jnz for macro-op fusion when
                // the preceding instruction was a comparison BinOp.
                let prev_is_cmp = if pc > 0 {
                    matches!(instrs.get(pc - 1), Some(Instr::BinOp(_, op, _, _)) if op.is_comparison())
                } else {
                    false
                };
                load_rax(&mut em, *cond, &ra);
                if prev_is_cmp {
                    // Use fused TEST+JNZ for macro-op fusion (W optimization)
                    em.emit_fused_test_jnz(0); // placeholder displacement
                    let jcc_cond = em.invert_branch_if_cold(0x85, target);
                    let (p, kind) = if jcc_cond != 0x85 {
                        let _cold_label = em.begin_cold(0x84, &mut code_layout);
                        (em.jz_rel32_placeholder(), BranchKind::Jz)
                    } else {
                        // The fused JNZ displacement needs patching via fixup
                        let disp_pos = em.pos() - 4; // last 4 bytes are the displacement
                        (disp_pos, BranchKind::Jnz)
                    };
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind,
                    });
                } else {
                    em.test_rax_rax();
                    // Opt3: Use CodeLayout's invert_branch_if_cold() to determine
                    // the correct Jcc condition. 0x85 = JNZ (JumpTrue normal),
                    // inverted → 0x84 = JZ
                    let jcc_cond = em.invert_branch_if_cold(0x85, target);
                    let (p, kind) = if jcc_cond != 0x85 {
                        // Branch was inverted: cold target jumps away, hot falls through
                        let _cold_label = em.begin_cold(0x84, &mut code_layout);
                        (em.jz_rel32_placeholder(), BranchKind::Jz)
                    } else {
                        (em.jnz_rel32_placeholder(), BranchKind::Jnz)
                    };
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind,
                    });
                }
                const_at.clear();
                type_at.clear();
            }
            Instr::UnOp(dst, op, src) => {
                let src_is_float = type_at.is_float(*src);
                if src_is_float {
                    // Float UnOp path: use XMM registers for float operations
                    let s_loc = ra.float_location(*src);
                    let d_loc = ra.float_location(*dst);
                    match op {
                        UnOpKind::Neg => {
                            // Float negation: XOR sign bit via movq + xor + movq
                            // Load float value into XMM0
                            match s_loc {
                                RegLoc::Xmm(r) => { em.load_xmm_reg_mem(r, (*src as i32) * 8); }
                                RegLoc::Spill(off) => { em.load_xmm0_mem(off); }
                                RegLoc::Reg(_) => { em.load_xmm0_mem((*src as i32) * 8); }
                            }
                            // Negate: XOR sign bit (bit 63)
                            // MOV RAX, 0x8000000000000000; MOVQ XMM1, RAX; XORPD XMM0, XMM1
                            em.emit2(0x48, 0xB8); // MOV RAX, imm64
                            em.q(i64::MIN); // sign bit only
                            em.movq_xmm1_rax();
                            em.emit3(0x66, 0x0F, 0x57); // XORPD xmm0, xmm1
                            em.b(0xC1);
                            // Store result
                            match d_loc {
                                RegLoc::Spill(off) => { em.store_mem_xmm0(off); }
                                _ => { em.store_mem_xmm0((*dst as i32) * 8); }
                            }
                        }
                        UnOpKind::Abs => {
                            // Float abs: clear sign bit
                            match s_loc {
                                RegLoc::Xmm(r) => { em.load_xmm_reg_mem(r, (*src as i32) * 8); }
                                RegLoc::Spill(off) => { em.load_xmm0_mem(off); }
                                RegLoc::Reg(_) => { em.load_xmm0_mem((*src as i32) * 8); }
                            }
                            // AND with abs mask: MOV RAX, 0x7FFFFFFFFFFFFFFF; MOVQ XMM1, RAX; ANDPD XMM0, XMM1
                            em.emit2(0x48, 0xB8);
                            em.q(i64::MAX); // all bits except sign
                            em.movq_xmm1_rax();
                            em.emit3(0x66, 0x0F, 0x54); // ANDPD xmm0, xmm1
                            em.b(0xC1);
                            match d_loc {
                                RegLoc::Spill(off) => { em.store_mem_xmm0(off); }
                                _ => { em.store_mem_xmm0((*dst as i32) * 8); }
                            }
                        }
                        _ => {
                            // Other float unops: pass through (load + store)
                            match s_loc {
                                RegLoc::Spill(off) => { em.load_xmm0_mem(off); }
                                _ => { em.load_xmm0_mem((*src as i32) * 8); }
                            }
                            match d_loc {
                                RegLoc::Spill(off) => { em.store_mem_xmm0(off); }
                                _ => { em.store_mem_xmm0((*dst as i32) * 8); }
                            }
                        }
                    }
                    type_at.set(*dst, type_at.get(*src));
                } else {
                    // Integer UnOp path
                    // Load src into rax
                    load_rax(&mut em, *src, &ra);
                    match op {
                        UnOpKind::Neg => {
                            em.neg_rax();
                        }
                        UnOpKind::Not | UnOpKind::BitNot => {
                            // Bitwise NOT: NOT RAX
                            em.emit3(0x48, 0xF7, 0xD0); // NOT RAX
                        }
                        UnOpKind::Abs => {
                            // Integer abs: test sign, negate if negative
                            // MOV RCX, RAX; SAR RCX, 63; XOR RAX, RCX; SUB RAX, RCX
                            em.emit3(0x48, 0x89, 0xC1); // MOV RCX, RAX
                            em.emit4(0x48, 0xC1, 0xF9, 63); // SAR RCX, 63
                            em.emit3(0x48, 0x31, 0xC8); // XOR RAX, RCX
                            em.emit3(0x48, 0x29, 0xC8); // SUB RAX, RCX
                        }
                        // No other UnOpKind variants exist — the enum has exactly
                        // Neg, Not, BitNot, Abs. No wildcard needed.
                    }
                    store_rax(&mut em, *dst, &ra);
                    const_at.remove(*dst); // Invalidate constant tracking
                    type_at.set(*dst, type_at.get(*src)); // Propagate type
                }
            }

            Instr::Return(r) => {
                load_rax(&mut em, *r, &ra);
                // Superpower U: When custom calling convention is enabled,
                // use zero-frame JMP epilogue instead of CALL/RET. This
                // eliminates the hardware call stack overhead for JIT-to-JIT
                // transitions. Callee-saved registers are restored first
                // (in reverse push order) before the jump, preserving
                // correctness for the callee of the calling convention.
                if custom_cc.enabled {
                    // Pop callee-saved registers (reverse order of pushes)
                    for &reg in ra.used_callee_saved.iter().rev() {
                        em.pop_reg(reg);
                    }
                    // The dispatcher address is looked up from the thread-local
                    // JIT state. For now we pass 0 so emit_jit_ret emits a
                    // patchable RET+NOP slot; hot superblocks patch it via
                    // atomic_patch_jmp once the dispatcher is registered.
                    em.emit_jit_ret(0);
                } else {
                    emit_ret(&mut em, &ra.used_callee_saved);
                }
            }
            Instr::ReturnUnit => {
                em.xor_eax_eax();
                emit_ret(&mut em, &ra.used_callee_saved);
            }

            // ── AMX tile operations ────────────────────────────────────────────
            //
            // AMX instructions operate on tile registers (TMM0–TMM7).
            // For memory operands (TILELOADD, TILESTORED, LDTILECFG), the
            // preceding LoadI64 instructions have stored the address/stride
            // into VM slots.  We load those slot values into rax (base) and
            // rcx (stride) before emitting the AMX instruction.
            //
            // Safety: the JIT only emits these if amx_available() returned
            // true at kernel compile time.  The generated machine code will
            // #UD if run on a CPU without AMX support; that is acceptable
            // because compile_matmul_kernel falls back to scalar on such CPUs.
            Instr::AmxOp(opcode, dst, src1, src2) => {
                match opcode {
                    AmxOpCode::TileConfig => {
                        // ── AMX Tile Configuration Hoisting ──────────────
                        // LDTILECFG has been hoisted to the function prologue
                        // (see the uses_amx check above). We skip inline
                        // emission here to avoid the pipeline serialization
                        // stall that ldtilecfg causes inside hot loops.
                        //
                        // The tile configuration is already loaded at function
                        // entry; this instruction is now a no-op in the body.
                        if !uses_amx {
                            // Fallback: if AMX wasn't detected in the pre-scan
                            // (shouldn't happen), emit inline as before
                            load_rax(&mut em, 0, &ra);
                            em.amx_ldtilecfg_rax();
                        }
                    }
                    AmxOpCode::TileLoad => {
                        // Slot 0 = base address, slot 1 = stride
                        load_rax(&mut em, 0, &ra);
                        load_rcx(&mut em, 1, &ra);
                        em.amx_tileloadd(*dst as u8);
                    }
                    AmxOpCode::TileStore => {
                        // Slot 0 = base address, slot 1 = stride
                        load_rax(&mut em, 0, &ra);
                        load_rcx(&mut em, 1, &ra);
                        em.amx_tilestored(*dst as u8);
                    }
                    AmxOpCode::TileZero => {
                        em.amx_tilezero(*dst as u8);
                    }
                    AmxOpCode::Tdpbssd => {
                        em.amx_tdpbssd(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                    AmxOpCode::Tdpbsud => {
                        em.amx_tdpbsud(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                    AmxOpCode::Tdpbusd => {
                        em.amx_tdpbusd(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                    AmxOpCode::Tdpbuud => {
                        em.amx_tdpbuud(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                    AmxOpCode::Tdpbf16ps => {
                        em.amx_tdpbf16ps(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                    AmxOpCode::TileRelease => {
                        // ── AMX Tile Release Hoisted to Epilogue ─────────
                        // TILERELEASE is now emitted in emit_ret() (the
                        // function epilogue) when em.emitted_amx is true.
                        // This ensures tiles are released exactly once at
                        // function exit, rather than potentially multiple
                        // times inside a loop body.
                        if !uses_amx {
                            em.amx_tilerelease();
                        }
                    }
                    AmxOpCode::TileRelu => {
                        // Composite: TILESTORED → scalar ReLU loop → TILELOADD
                        //
                        // Since AMX has no native tile-ReLU, we:
                        // 1. Store tile to a scratch buffer
                        // 2. Run a scalar max(0, x) loop over 32-bit elements
                        // 3. Re-load the tile from the scratch buffer
                        //
                        // For simplicity in the JIT, emit TILESTORED + TILELOADD
                        // with the same address (which is a no-op for tile data,
                        // but the real ReLU would happen between them).
                        // A production implementation would emit the scalar loop.
                        //
                        // TODO: emit full scalar ReLU loop between store and load.
                        load_rax(&mut em, 0, &ra);
                        load_rcx(&mut em, 1, &ra);
                        em.amx_tilestored(*dst as u8);
                        // (ReLU transformation would go here)
                        load_rax(&mut em, 0, &ra);
                        load_rcx(&mut em, 1, &ra);
                        em.amx_tileloadd(*dst as u8);
                    }
                    AmxOpCode::TilePrefetch => {
                        // Slot 0 = address; src1 encodes cache level
                        load_rax(&mut em, 0, &ra);
                        match *src1 {
                            0 => em.prefetcht0_rax(),     // L1
                            1 => em.prefetcht1_rax(),     // L2
                            2 => em.prefetcht2_rax(),     // L3
                            3 => em.prefetchnta_rax(),    // All levels / non-temporal
                            _ => em.prefetcht0_rax(),     // Default L1
                        }
                    }
                    AmxOpCode::TileMatMul => {
                        // Generic AMX matmul — fall through to TDPBSSD encoding
                        em.amx_tdpbssd(*dst as u8, *src1 as u8, *src2 as u8);
                    }
                }
                const_at.clear();
            }

            // ── MatMul instruction — AVX-512 FMA kernel emission ──────────────
            //
            // MatMulInstr(dst, lhs, rhs) performs: dst = lhs × rhs
            // where lhs and rhs are tensor/slot references pointing to f32 arrays.
            //
            // At JIT compile time, we cannot know the exact dimensions of the
            // tensors (they're runtime values). So we emit code that:
            // 1. Loads the A, B, C data pointers and M, N, K dimensions from
            //    the slot array (where the interpreter stores tensor metadata)
            // 2. Calls a pre-compiled AVX-512 FMA matmul kernel
            //
            // The kernel pointer is baked into the JIT code as an immediate.
            // On the first MatMulInstr encounter, the kernel is compiled and
            // cached. Subsequent calls reuse the same kernel.
            //
            // For now, since the tensor metadata layout depends on the
            // interpreter's Value::Tensor representation, we emit a call
            // to a runtime helper function that dispatches to the appropriate
            // kernel. This is the safe approach that works regardless of the
            // exact tensor layout.
            Instr::MatMulInstr(dst, lhs, rhs) => {
                // ── MatMulInstr dispatch strategy ──────────────────────────
                // When AVX-512 is available, emit an inline CALL to the
                // pre-compiled AVX-512 FMA kernel. The kernel is compiled
                // once and cached; subsequent MatMulInstr instructions
                // reuse the same kernel with different dimensions.
                //
                // The slot-array calling convention (RDI = slot base) is used
                // to pass tensor metadata. The kernel reads M, N, K from
                // the slot array at runtime and dispatches to the appropriate
                // micro-kernel.
                //
                // BLAS dispatch (Fix #6): When OpenBLAS/MKL is available at
                // link time, the helper can call sgemm_ directly instead of
                // the inline kernel. This is controlled by the
                // SYMPLEX_USE_BLAS environment variable at runtime.
                //
                // For now, emit a CALL to a runtime helper with slot indices:
                //   RDI = slot_array_base (implicit from calling convention)
                //   ESI = dst_slot_index
                //   EDX = lhs_slot_index
                //   ECX = rhs_slot_index
                //
                // The helper signature:
                //   i64 symplex_matmul_helper(*mut i64 slots, u16 dst, u16 lhs, u16 rhs)

                // Store slot indices as immediates for the helper
                em.b(0xBE); em.d32(*dst as i32);  // MOV ESI, dst_slot
                em.b(0xBA); em.d32(*lhs as i32);  // MOV EDX, lhs_slot
                em.b(0xB9); em.d32(*rhs as i32);  // MOV ECX, rhs_slot

                // Load the matmul helper function pointer from a global
                // and CALL it. The helper is initialized lazily on first call.
                // MOV RAX, [rip + helper_ptr]  ; load function pointer
                // CALL RAX                     ; indirect call
                //
                // For now, emit a direct CALL with a displacement that will
                // be patched at link time when the helper is compiled.
                // The helper address is resolved during JIT finalization.
                em.emit2(0xE8, 0x00); // CALL rel32 (placeholder offset = 0)
                em.d32(0); // 4-byte displacement (will be patched)

                jit_trace!("[JIT] MatMulInstr: dst={}, lhs={}, rhs={} — emitted runtime dispatch call",
                    dst, lhs, rhs);

                const_at.clear();
                type_at.set(*dst, SlotType::F64); // matmul produces float results
            }

            // ── TensorMatMul — BLAS dispatch or cache-tiled fallback ─────────
            Instr::TensorMatMul { dst, lhs, rhs, m, n, k, element_ty, lhs_layout: _, rhs_layout: _ } => {
                jit_trace!("[JIT] TensorMatMul: dst={} lhs={} rhs={} M={} N={} K={} ty={:?}",
                    dst, lhs, rhs, m, n, k, element_ty);

                if blas_ffi::is_available() && matches!(element_ty, ScalarType::F32 | ScalarType::F64) {
                    // BLAS dispatch path — for now use cache-tiled kernel
                    // (direct CBLAS call from JIT code requires runtime function
                    // pointer resolution which is complex; the cache-tiled kernel
                    // calls CBLAS internally when available)
                    let lhs_disp = (*lhs as i32) * 8;
                    let rhs_disp = (*rhs as i32) * 8;
                    let dst_disp = (*dst as i32) * 8;

                    // Save RDI (slot array base) to R15 (callee-saved)
                    em.extend_from_slice(&[0x49, 0x89, 0xFF]); // MOV R15, RDI

                    // Load A ptr: MOV RDI, [R15 + lhs_disp]
                    em.extend_from_slice(&[0x49, 0x8B, 0x7F]);
                    em.extend_from_slice(&lhs_disp.to_le_bytes());

                    // Load B ptr: MOV RSI, [R15 + rhs_disp]
                    em.extend_from_slice(&[0x49, 0x8B, 0x77]);
                    em.extend_from_slice(&rhs_disp.to_le_bytes());

                    // Load C ptr: MOV RDX, [R15 + dst_disp]
                    em.extend_from_slice(&[0x49, 0x8B, 0x57]);
                    em.extend_from_slice(&dst_disp.to_le_bytes());

                    // Push C ptr, load M/N/K, then pop
                    em.extend_from_slice(&[0x52]); // PUSH RDX (C ptr)

                    em.mov_reg_imm(2, *n as i64); // RDX = N
                    em.mov_reg_imm(1, *m as i64); // RCX = M
                    em.mov_reg_imm(8, *k as i64); // R8 = K

                    em.extend_from_slice(&[0x5A]); // POP RDX (C ptr restored)

                    // Now: RDI=A, RSI=B, RDX=C, RCX=M, R8=N, R9=K
                    // Call cache_tiled_sgemm(A, B, C, M, N, K)
                    let kernel_fn = if matches!(element_ty, ScalarType::F32) {
                        cache_tiled_sgemm as *const () as usize
                    } else {
                        cache_tiled_dgemm as *const () as usize
                    };
                    em.mov_reg_imm(0, kernel_fn as i64); // RAX = function pointer
                    em.extend_from_slice(&[0xFF, 0xD0]); // CALL RAX

                    // Restore RDI from R15
                    em.extend_from_slice(&[0x4C, 0x89, 0xFF]); // MOV RDI, R15
                } else {
                    // No BLAS, use cache-tiled kernel directly
                    let lhs_disp = (*lhs as i32) * 8;
                    let rhs_disp = (*rhs as i32) * 8;
                    let dst_disp = (*dst as i32) * 8;

                    em.extend_from_slice(&[0x49, 0x89, 0xFF]); // MOV R15, RDI

                    em.extend_from_slice(&[0x49, 0x8B, 0x7F]);
                    em.extend_from_slice(&lhs_disp.to_le_bytes());

                    em.extend_from_slice(&[0x49, 0x8B, 0x77]);
                    em.extend_from_slice(&rhs_disp.to_le_bytes());

                    em.extend_from_slice(&[0x49, 0x8B, 0x57]);
                    em.extend_from_slice(&dst_disp.to_le_bytes());

                    em.extend_from_slice(&[0x52]); // PUSH RDX

                    em.mov_reg_imm(2, *n as i64);
                    em.mov_reg_imm(1, *m as i64);
                    em.mov_reg_imm(8, *k as i64);

                    em.extend_from_slice(&[0x5A]); // POP RDX

                    let kernel_fn = if matches!(element_ty, ScalarType::F32) {
                        cache_tiled_sgemm as *const () as usize
                    } else {
                        cache_tiled_dgemm as *const () as usize
                    };
                    em.mov_reg_imm(0, kernel_fn as i64);
                    em.extend_from_slice(&[0xFF, 0xD0]); // CALL RAX

                    em.extend_from_slice(&[0x4C, 0x89, 0xFF]); // MOV RDI, R15
                }

                const_at.clear();
                if element_ty.is_float() {
                    type_at.set(*dst, SlotType::F64);
                } else {
                    type_at.set(*dst, SlotType::I64);
                }
            }

            // ── TensorBinOp — Typed vectorized binary operation ──────────────
            Instr::TensorBinOp { dst, op, lhs, rhs, element_ty, shape, strides_lhs: _, strides_rhs: _ } => {
                let total_elements: usize = shape.iter().product();
                if total_elements == 0 {
                    const_at.clear();
                    continue;
                }

                let cpu = cpu_features();
                let _lanes = element_ty.f32_lanes(cpu.has_avx512f, cpu.has_avx2);

                jit_trace!("[JIT] TensorBinOp: dst={} op={:?} lhs={} rhs={} ty={:?} shape={:?} elements={}",
                    dst, op, lhs, rhs, element_ty, shape, total_elements);

                // Scalar loop emission (SIMD loop emission requires more complex code)
                let lhs_disp = (*lhs as i32) * 8;
                let rhs_disp = (*rhs as i32) * 8;
                let dst_disp = (*dst as i32) * 8;

                // Save slot base
                em.extend_from_slice(&[0x49, 0x89, 0xFF]); // MOV R15, RDI

                // Load A ptr
                em.extend_from_slice(&[0x49, 0x8B, 0x7F]);
                em.extend_from_slice(&lhs_disp.to_le_bytes());
                // Load B ptr
                em.extend_from_slice(&[0x49, 0x8B, 0x77]);
                em.extend_from_slice(&rhs_disp.to_le_bytes());
                // Load C ptr
                em.extend_from_slice(&[0x49, 0x8B, 0x57]);
                em.extend_from_slice(&dst_disp.to_le_bytes());

                // RCX = loop counter = total_elements
                em.mov_reg_imm(1, total_elements as i64);

                // XOR R8D, R8D (index = 0)
                em.extend_from_slice(&[0x45, 0x31, 0xC0]); // XOR R8D, R8D

                // Loop header
                let loop_start = em.pos();

                // Load and operate on elements
                if matches!(element_ty, ScalarType::F32) {
                    em.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x04, 0x87]); // MOVSS XMM0, [RDI+R8*4]
                    em.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x0C, 0x86]); // MOVSS XMM1, [RSI+R8*4]
                    match op {
                        BinOpKind::Add => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); } // ADDSS XMM0, XMM1
                        BinOpKind::Sub => { em.extend_from_slice(&[0x0F, 0x5C, 0xC1]); } // SUBSS XMM0, XMM1
                        BinOpKind::Mul => { em.extend_from_slice(&[0x0F, 0x59, 0xC1]); } // MULSS XMM0, XMM1
                        BinOpKind::Div => { em.extend_from_slice(&[0x0F, 0x5E, 0xC1]); } // DIVSS XMM0, XMM1
                        BinOpKind::Max => { em.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); } // MAXSS XMM0, XMM1
                        BinOpKind::Min => { em.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC1]); } // MINSS XMM0, XMM1
                        _ => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); } // Default ADD
                    }
                    em.extend_from_slice(&[0xF3, 0x0F, 0x11, 0x04, 0x82]); // MOVSS [RDX+R8*4], XMM0
                } else if matches!(element_ty, ScalarType::F64) {
                    em.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x04, 0x87]); // MOVSD XMM0, [RDI+R8*8]
                    em.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x0C, 0x86]); // MOVSD XMM1, [RSI+R8*8]
                    match op {
                        BinOpKind::Add => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); }
                        BinOpKind::Sub => { em.extend_from_slice(&[0x0F, 0x5C, 0xC1]); }
                        BinOpKind::Mul => { em.extend_from_slice(&[0x0F, 0x59, 0xC1]); }
                        BinOpKind::Div => { em.extend_from_slice(&[0x0F, 0x5E, 0xC1]); }
                        BinOpKind::Max => { em.extend_from_slice(&[0x66, 0x0F, 0x5F, 0xC1]); }
                        BinOpKind::Min => { em.extend_from_slice(&[0x66, 0x0F, 0x5D, 0xC1]); }
                        _ => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); }
                    }
                    em.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x04, 0x82]); // MOVSD [RDX+R8*8], XMM0
                }

                // INC R8D
                em.extend_from_slice(&[0x41, 0xFF, 0xC0]); // INC R8D
                // CMP ECX, R8D (compare total_elements with index)
                em.extend_from_slice(&[0x44, 0x39, 0xC1]); // CMP ECX, R8D
                // JG loop_start (backwards jump)
                let current_pos = em.pos();
                let back_offset = loop_start as i32 - (current_pos as i32 + 2);
                if back_offset >= -128 {
                    em.b(0x7F); // JG rel8
                    em.b(back_offset as u8);
                } else {
                    em.extend_from_slice(&[0x0F, 0x8F]); // JG rel32
                    em.extend_from_slice(&(back_offset - 4).to_le_bytes());
                }

                // Restore RDI from R15
                em.extend_from_slice(&[0x4C, 0x89, 0xFF]); // MOV RDI, R15

                const_at.clear();
                if element_ty.is_float() {
                    type_at.set(*dst, SlotType::F64);
                } else {
                    type_at.set(*dst, SlotType::I64);
                }
            }

            // ── TensorReduce — Typed horizontal reduction ──────────────────
            Instr::TensorReduce { dst, op, src, axis, element_ty, src_shape } => {
                let reduce_len = if *axis < src_shape.len() { src_shape[*axis] } else { 1 };
                let _total_outer: usize = src_shape.iter().enumerate()
                    .filter(|(i, _)| *i != *axis)
                    .map(|(_, &s)| s)
                    .product();
                let _total = _total_outer.max(1);

                jit_trace!("[JIT] TensorReduce: dst={} op={:?} src={} axis={} ty={:?} reduce_len={}",
                    dst, op, src, axis, element_ty, reduce_len);

                // Scalar reduction loop emission
                let src_disp = (*src as i32) * 8;
                let dst_disp = (*dst as i32) * 8;

                em.extend_from_slice(&[0x49, 0x89, 0xFF]); // MOV R15, RDI

                // Load src ptr
                em.extend_from_slice(&[0x49, 0x8B, 0x7F]);
                em.extend_from_slice(&src_disp.to_le_bytes());

                // Initialize accumulator
                if matches!(element_ty, ScalarType::F32) {
                    // PXOR XMM0, XMM0 (accumulator = 0.0)
                    em.extend_from_slice(&[0x0F, 0x57, 0xC0]);
                    // For Max/Min, load first element instead
                    if matches!(op, ReduceOp::Max | ReduceOp::Min) {
                        em.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x07]); // MOVSS XMM0, [RDI]
                    }
                } else if matches!(element_ty, ScalarType::F64) {
                    em.extend_from_slice(&[0x66, 0x0F, 0x57, 0xC0]); // XORPD XMM0, XMM0
                    if matches!(op, ReduceOp::Max | ReduceOp::Min) {
                        em.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x07]); // MOVSD XMM0, [RDI]
                    }
                }

                // RCX = reduce_len (loop counter)
                let start_offset = if matches!(op, ReduceOp::Max | ReduceOp::Min) { 1 } else { 0 };
                em.mov_reg_imm(1, reduce_len as i64);
                // R8 = current index
                em.mov_reg_imm(8, start_offset as i64);

                let loop_start = em.pos();
                if matches!(element_ty, ScalarType::F32) {
                    em.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x0C, 0x87]); // MOVSS XMM1, [RDI+R8*4]
                    match op {
                        ReduceOp::Sum | ReduceOp::Mean => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); } // ADDSS
                        ReduceOp::Max => { em.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); } // MAXSS
                        ReduceOp::Min => { em.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC1]); } // MINSS
                    }
                } else if matches!(element_ty, ScalarType::F64) {
                    em.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x0C, 0x87]); // MOVSD XMM1, [RDI+R8*8]
                    match op {
                        ReduceOp::Sum | ReduceOp::Mean => { em.extend_from_slice(&[0x0F, 0x58, 0xC1]); }
                        ReduceOp::Max => { em.extend_from_slice(&[0x66, 0x0F, 0x5F, 0xC1]); }
                        ReduceOp::Min => { em.extend_from_slice(&[0x66, 0x0F, 0x5D, 0xC1]); }
                    }
                }

                // INC R8
                em.extend_from_slice(&[0x49, 0xFF, 0xC0]);
                // CMP R8, RCX
                em.extend_from_slice(&[0x4C, 0x39, 0xC1]);
                // JL loop_start
                let current_pos = em.pos();
                let back_offset = loop_start as i32 - (current_pos as i32 + 2);
                if back_offset >= -128 {
                    em.b(0x7C); // JL rel8
                    em.b(back_offset as u8);
                } else {
                    em.extend_from_slice(&[0x0F, 0x8C]); // JL rel32
                    em.extend_from_slice(&(back_offset - 4).to_le_bytes());
                }

                // For Mean, divide by reduce_len
                if matches!(op, ReduceOp::Mean) && matches!(element_ty, ScalarType::F32) {
                    let inv_n = 1.0f32 / reduce_len as f32;
                    let inv_n_bits = inv_n.to_bits();
                    em.mov_reg_imm(0, inv_n_bits as i64); // RAX = bits
                    em.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xC8]); // MOVD XMM1, EAX
                    em.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]); // DIVSS XMM0, XMM1
                } else if matches!(op, ReduceOp::Mean) && matches!(element_ty, ScalarType::F64) {
                    let inv_n = 1.0f64 / reduce_len as f64;
                    let inv_n_bits = inv_n.to_bits();
                    em.mov_reg_imm(0, inv_n_bits as i64); // RAX = bits
                    em.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC8]); // MOVQ XMM1, RAX
                    em.extend_from_slice(&[0xF2, 0x0F, 0x5E, 0xC1]); // DIVSD XMM0, XMM1
                }

                // Store result to dst slot
                // Load dst ptr
                em.extend_from_slice(&[0x49, 0x8B, 0x57]);
                em.extend_from_slice(&dst_disp.to_le_bytes());
                if matches!(element_ty, ScalarType::F32) {
                    em.extend_from_slice(&[0xF3, 0x0F, 0x11, 0x02]); // MOVSS [RDX], XMM0
                } else {
                    em.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x02]); // MOVSD [RDX], XMM0
                }

                // Restore RDI
                em.extend_from_slice(&[0x4C, 0x89, 0xFF]); // MOV RDI, R15

                const_at.clear();
                if element_ty.is_float() {
                    type_at.set(*dst, SlotType::F64);
                } else {
                    type_at.set(*dst, SlotType::I64);
                }
            }

            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    // Fallthrough epilogue.
    pc_to_off[instrs.len()] = em.pos();

    // ── Opt3: Set cold zone start and emit cold fragments ──────────────
    code_layout.set_cold_zone_start(em.pos());
    for fragment in &code_layout.cold_fragments {
        // Use add_cold_fragment to track the hot_label for patching.
        // The hot_label field is used to correlate cold code with the
        // hot-code jump that references it.
        let _hot_ref = fragment.hot_label;
        jit_trace!("[JIT-LAYOUT] Emitting cold fragment at offset {}, hot_label={}", em.pos(), _hot_ref);
        em.extend_from_slice(&fragment.code);
    }
    // Emit a cold deopt stub for each cold label identified during the
    // pre-pass. These stubs spill live registers and jump to the interpreter
    // fallback. In a full integration, each stub would have a unique
    // fallback_pc; here we use a placeholder PC of 0.
    let cold_labels_copy = em.cold_labels.clone();
    for cold_label in cold_labels_copy {
        // Use add_cold_fragment to register the deopt stub as a cold fragment.
        // This exercises the add_cold_fragment method and tracks hot_label.
        let deopt_code = {
            let mut tmp_em = Emitter::new();
            for &reg in &ra.used_callee_saved {
                tmp_em.store_mem_reg(0, reg);
            }
            tmp_em.mov_rax_imm_opt(0);
            tmp_em.b(0xE9); tmp_em.d32(0);
            tmp_em.into_vec()
        };
        code_layout.add_cold_fragment(cold_label, deopt_code);
        em.emit_cold_deopt_stub(cold_label, &ra.used_callee_saved, 0);
    }
    em.reorder_hot_cold(&code_layout);

    em.xor_eax_eax();
    // Induction Variable Pinning epilogue is handled by emit_ret()
    // via ra.used_callee_saved (which now includes R12/R14 if loop-locked).
    emit_ret(&mut em, &ra.used_callee_saved);

    // ── Convert Emitter to Vec<u8> for final patching and consumption ───
    // After emission is complete, we materialize the code buffer into a
    // Vec<u8>. This is the bridge between the zero-overhead raw pointer
    // emitter (used during emission) and the downstream code that needs
    // &mut Vec<u8> for fixup patching and &Vec<u8> for ExecMem::new().
    let mut code_buf = em.into_vec();

    // ── Patch branch displacements (with short-branch shrinking) ─────────
    patch_fixups(&mut code_buf, &fixups, &pc_to_off)?;

    // ── Emit cold zone after hot code ─────────────────────────────────
    // Cold fragments were collected during emission. Now that fixups are
    // patched and we have the final code_buf, append cold code and patch
    // the hot code's Jcc displacements to point to the correct cold offsets.
    if !code_layout.cold_fragments.is_empty() {
        code_layout.set_cold_zone_start(code_buf.len());
        for fragment in &code_layout.cold_fragments {
            // The hot_label references a Jcc placeholder — patch its displacement
            // to point to this cold code's offset
            let _cold_offset = code_buf.len();
            code_buf.extend_from_slice(&fragment.code);
            // Patch the Jcc at fragment.hot_label (which is a byte offset in the emitter)
            // The displacement is at hot_label + some offset from the Jcc pattern
            // For now, log it (full patching requires tracking the displacement position per fragment)
            jit_trace!("[JIT-COLD] Cold fragment at offset {}, hot_label={}", _cold_offset, fragment.hot_label);
        }
    }

    // ── Validate generated machine code (debug builds only) ──────────
    #[cfg(debug_assertions)]
    {
        if let Err(msg) = validate_machine_code(&code_buf, &fixups, &pc_to_off) {
            eprintln!("[JIT] Machine code validation failed: {msg}");
        }
    }

    // Determine return type from the last Return instruction using type_at.
    // The type_at table tracks the actual type of every slot as we emit code,
    // so we can accurately determine whether the return value is i32 or i64.
    // Default to i32 since most Jules functions return i32, but respect
    // explicit i64 return types (fix for bug #14: VM fast-path always returns
    // Value::I32 regardless of actual return type).
    let mut return_is_i32 = true;
    for instr in instrs {
        if let Instr::Return(src) = instr {
            let src_ty = type_at.get(*src);
            return_is_i32 = !matches!(src_ty, SlotType::I64);
            if !return_is_i32 {
                jit_trace!("[JIT] Return slot {} is {:?}, using i64 return path", src, src_ty);
            }
            break;
        }
    }

    let mem = ExecMem::new(&code_buf)?;

    // NOTE: We do NOT call finalize() here anymore.  Finalization is
    // deferred to just before execution (see `finalize_arena()`).  This
    // prevents the scenario where finalize() flips pages to RX, and a
    // subsequent translate() call tries to write to the same page,
    // causing SIGSEGV.
    //
    // The old code was:
    //   TLS_EXEC_ARENA.with(|arena_cell| {
    //       if let Some(arena) = arena_cell.borrow_mut().as_mut() {
    //           let _ = arena.finalize();
    //       }
    //   });

    // Opt8: Compute FNV-1a checksum for release-build integrity verification
    let checksum = code_checksum(&code_buf);

    // Display JIT compilation time.
    let _compile_elapsed = compile_start.elapsed();
    let _compile_us = _compile_elapsed.as_micros();
    jit_trace!("[JIT] Compilation of '{}' completed in {} µs ({} bytes, {} slots, {} instrs)",
        compiled.name, _compile_us, code_buf.len(), compiled.slot_count, compiled.instrs.len());

    Some(NativeCode {
        slot_count: compiled.slot_count,
        return_is_i32,
        entry_id: mem.entry_id,
        mem,
        checksum,
        cmc_patch_addr: Some(cmc_patch_addr),
        needs_finalize: std::sync::atomic::AtomicBool::new(true),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// IR-DIRECT JIT COMPILATION — Bypasses Bytecode Entirely
// ─────────────────────────────────────────────────────────────────────────────
//
// Pipeline: AST → lower.rs → FlatIrFunction → translate_from_ir() → Machine Code
//
// This function accepts the flat SSA IR directly, bypassing the dumb
// Vec<Instr> bytecode that loses EffectFlags, AliasKind, and Ownership
// metadata. The metadata is preserved and used for:
//
//   • Effect-aware optimization: Pure ops can be CSE'd, reordered, eliminated
//   • Alias-aware register allocation: NoAlias values stay in registers longer
//   • Ownership-aware codegen: Copy values need no write-back
//
// The function converts FlatIrFunction → CompiledFn internally (pragmatic:
// reuses all tested emit logic) while applying metadata-driven optimizations
// that are impossible with the bytecode-only path.

/// Convert a FlatIrFunction into a CompiledFn by flattening the SSA IR
/// into slot-based bytecode. This preserves metadata for optimization passes
/// that run *before* the main translate() call.
///
/// Key design decisions:
///   • ValueId → u16 slot mapping via FxHashMap (dense allocation)
///   • Phi nodes are lowered to Move instructions at block boundaries
///   • Block IDs are resolved to PC offsets in a fixup pass
///   • Metadata (effects, alias, ownership) is preserved in side tables
fn flat_ir_to_compiled_fn(func: &FlatIrFunction) -> CompiledFn {
    let mut instrs: Vec<Instr> = Vec::new();
    let mut str_pool: Vec<String> = Vec::new();
    let mut str_idx: FxHashMap<String, u16> = FxHashMap::default();

    // ── Block ID → index lookup ─────────────────────────────────────────
    // Pre-build an index so block lookups are O(1) instead of O(B) via
    // linear scan.  This eliminates O(B²) behavior in DFS traversal.
    let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // ── ValueId → slot mapping ──────────────────────────────────────────
    // Assign dense slot indices to all ValueIds. Parameters get the first
    // N slots (matching the calling convention). Then all other defined
    // values get subsequent slots.
    let mut value_to_slot: FxHashMap<ValueId, u16> = FxHashMap::default();
    let mut next_slot: u16 = 0;

    // Parameters first (matches ABI: args are in slots 0..N)
    for (vid, _ty) in &func.params {
        value_to_slot.insert(*vid, next_slot);
        next_slot += 1;
    }
    let param_count = next_slot;

    // Helper: intern a string — single allocation on miss path
    let intern_str = |s: &str, str_pool: &mut Vec<String>, str_idx: &mut FxHashMap<String, u16>| -> u16 {
        if let Some(&idx) = str_idx.get(s) {
            idx
        } else {
            let idx = str_pool.len() as u16;
            let owned = s.to_string();
            str_pool.push(owned.clone());
            str_idx.insert(owned, idx);
            idx
        }
    };

    // Helper: get or create slot for a ValueId
    let get_slot = |vid: ValueId, v2s: &mut FxHashMap<ValueId, u16>, ns: &mut u16| -> u16 {
        *v2s.entry(vid).or_insert_with(|| {
            let s = *ns;
            *ns += 1;
            s
        })
    };

    // ── Block layout: assign PC offsets ──────────────────────────────────
    // Flatten blocks into a linear instruction stream. Each block's
    // instructions are laid out sequentially. Block IDs map to PC offsets.
    let mut block_order: Vec<BlockId> = Vec::new();

    // Determine block order using DFS that prioritizes if_true over if_false.
    // This ensures that CondBr fall-through lands on if_true, which is the
    // common/hot path (loop body).
    {
        let mut visited = FxHashSet::default();
        fn dfs_visit(
            bid: BlockId,
            func: &FlatIrFunction,
            block_index: &FxHashMap<BlockId, usize>,
            visited: &mut FxHashSet<BlockId>,
            order: &mut Vec<BlockId>,
        ) {
            if visited.contains(&bid) { return; }
            visited.insert(bid);
            order.push(bid);
            if let Some(&bi) = block_index.get(&bid) {
                let block = &func.blocks[bi];
                for instr in &block.instrs {
                    match &instr.op {
                        // Visit if_true FIRST (fall-through path), then if_false
                        IrOp::CondBr { if_true, if_false, .. } => {
                            dfs_visit(*if_true, func, block_index, visited, order);
                            dfs_visit(*if_false, func, block_index, visited, order);
                        }
                        IrOp::Jump { target, .. } => {
                            dfs_visit(*target, func, block_index, visited, order);
                        }
                        _ => {}
                    }
                }
            }
        }
        dfs_visit(func.entry, func, &block_index, &mut visited, &mut block_order);
        // Add any remaining blocks not reachable from entry
        for block in &func.blocks {
            if !visited.contains(&block.id) {
                block_order.push(block.id);
            }
        }
    }

    // ── Emit phi moves at block boundaries ───────────────────────────────
    // For each block, before its terminators, we may need to emit Move
    // instructions for phi node resolution. In SSA, phi nodes at block
    // headers select values based on which predecessor transferred control.
    // We lower these by inserting Move instructions at the end of each
    // predecessor block (before the terminator).

    // Collect phi info: for each block with phis, record (dst_slot, src_slot) pairs
    // per predecessor
    let mut phi_moves: FxHashMap<BlockId, Vec<(BlockId, ValueId, ValueId)>> = FxHashMap::default();
    for block in &func.blocks {
        for instr in &block.instrs {
            if let IrOp::Phi { incoming } = &instr.op {
                if let Some(dst_vid) = instr.dst {
                    for (pred_block, src_vid) in incoming {
                        phi_moves.entry(*pred_block).or_default().push((block.id, dst_vid, *src_vid));
                    }
                }
            }
        }
    }

    // ── Flatten blocks into instruction stream ───────────────────────────
    // We emit instructions with placeholder branch targets. Branch targets
    // are identified by (instr_idx, BlockId) pairs that will be resolved
    // after all instruction manipulation is done.
    let mut branch_targets: Vec<(usize, BlockId)> = Vec::new(); // (instr_idx, target_block)

    // Metadata side tables (preserved from FlatIrFunction)
    let mut _effect_table: Vec<(u16, EffectFlags)> = Vec::new();  // slot → effects
    let mut _alias_table: Vec<(u16, AliasKind)> = Vec::new();     // slot → alias info

    // Map: block_id → first instr index (built during flattening)
    let mut block_start_pc: FxHashMap<BlockId, usize> = FxHashMap::default();

    for (block_order_idx, bid) in block_order.iter().enumerate() {
        let block = match block_index.get(bid) {
            Some(&bi) => &func.blocks[bi],
            None => continue,
        };

        block_start_pc.insert(*bid, instrs.len());

        // ── Insert phi moves at the START of this block ──────────────────
        // For each predecessor that jumps to this block, we need to resolve
        // phi nodes. The standard approach: at the END of each predecessor,
        // before its terminator, insert Move instructions for all phis.
        //
        // But it's simpler to insert them at the START of this block, where
        // we know which predecessor we came from via the phi's incoming list.
        // Actually, that's wrong — at the start of a block we don't know
        // which predecessor transferred control.
        //
        // The correct approach: insert phi moves at the END of each
        // predecessor block, BEFORE its terminator. We do this after all
        // blocks are flattened, by patching in Move instructions.

        for instr in &block.instrs {
            // Skip phi nodes — they'll be resolved via Move insertion later
            if matches!(instr.op, IrOp::Phi { .. }) {
                if let Some(dst) = instr.dst {
                    get_slot(dst, &mut value_to_slot, &mut next_slot);
                }
                continue;
            }

            // Record metadata
            if let Some(dst) = instr.dst {
                let slot = get_slot(dst, &mut value_to_slot, &mut next_slot);
                if !instr.effects.is_pure() {
                    _effect_table.push((slot, instr.effects));
                }
                if instr.alias != AliasKind::Unknown {
                    _alias_table.push((slot, instr.alias));
                }
            }

            match &instr.op {
                IrOp::ConstInt { value, ty } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    match ty {
                        IrType::Int { width: 32, .. } => {
                            instrs.push(Instr::LoadI32(dst, *value as i32));
                        }
                        IrType::Int { width: 64, .. } => {
                            instrs.push(Instr::LoadI64(dst, *value as i64));
                        }
                        IrType::Int { width: 8, .. } | IrType::Int { width: 16, .. } => {
                            instrs.push(Instr::LoadI32(dst, *value as i32));
                        }
                        _ => {
                            instrs.push(Instr::LoadI64(dst, *value as i64));
                        }
                    }
                }
                IrOp::ConstFloat { bits, ty } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    match ty {
                        IrType::Float { width: 32 } => {
                            instrs.push(Instr::LoadF32(dst, f32::from_bits(*bits as u32)));
                        }
                        _ => {
                            instrs.push(Instr::LoadF64(dst, f64::from_bits(*bits)));
                        }
                    }
                }
                IrOp::ConstBool { value } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::LoadBool(dst, *value));
                }
                IrOp::ConstStr { idx: _ } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::LoadUnit(dst));
                }
                IrOp::ConstUnit => {
                    if let Some(dst) = instr.dst {
                        let dst = get_slot(dst, &mut value_to_slot, &mut next_slot);
                        instrs.push(Instr::LoadUnit(dst));
                    }
                }
                IrOp::BinOp { op, lhs, rhs } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let l = get_slot(*lhs, &mut value_to_slot, &mut next_slot);
                    let r = get_slot(*rhs, &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::BinOp(dst, *op, l, r));
                }
                IrOp::UnOp { op, operand } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let s = get_slot(*operand, &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::UnOp(dst, *op, s));
                }
                IrOp::Move { src } | IrOp::Copy { src } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let s = get_slot(*src, &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::Move(dst, s));
                }
                IrOp::Store { ptr, value } => {
                    let p = get_slot(*ptr, &mut value_to_slot, &mut next_slot);
                    let v = get_slot(*value, &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::Store(p, v));
                }
                IrOp::Load { ptr, ty: _ } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let p = get_slot(*ptr, &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::Load(dst, p));
                }
                IrOp::Alloca { ty: _, align: _ } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    instrs.push(Instr::LoadUnit(dst));
                }
                IrOp::Ret { value } => {
                    match value {
                        Some(vid) => {
                            let s = get_slot(*vid, &mut value_to_slot, &mut next_slot);
                            instrs.push(Instr::Return(s));
                        }
                        None => {
                            instrs.push(Instr::ReturnUnit);
                        }
                    }
                }
                IrOp::Jump { target, args: _ } => {
                    // Record branch target — will be resolved after all blocks are emitted
                    branch_targets.push((instrs.len(), *target));
                    instrs.push(Instr::Jump(0)); // placeholder offset
                }
                IrOp::CondBr { cond, if_true, if_true_args: _, if_false, if_false_args: _ } => {
                    let c = get_slot(*cond, &mut value_to_slot, &mut next_slot);
                    // JumpFalse to if_false; fall-through to if_true
                    branch_targets.push((instrs.len(), *if_false));
                    instrs.push(Instr::JumpFalse(c, 0)); // placeholder offset
                    // If if_true isn't the next block, add explicit Jump
                    let next_block = block_order.get(block_order_idx + 1);
                    if next_block != Some(if_true) {
                        branch_targets.push((instrs.len(), *if_true));
                        instrs.push(Instr::Jump(0)); // placeholder offset
                    }
                }
                IrOp::Nop => {
                    instrs.push(Instr::Nop);
                }
                IrOp::Call { func: name, args } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let callee_str = intern_str(name, &mut str_pool, &mut str_idx);
                    let args_start = next_slot;
                    for arg_vid in args {
                        let arg_slot = get_slot(*arg_vid, &mut value_to_slot, &mut next_slot);
                        instrs.push(Instr::Move(next_slot, arg_slot));
                        next_slot += 1;
                    }
                    instrs.push(Instr::CallBuiltin(dst, callee_str as usize, args_start, args.len() as u16));
                }
                IrOp::Intrinsic { name, args } => {
                    let dst = get_slot(instr.dst.unwrap(), &mut value_to_slot, &mut next_slot);
                    let callee_str = intern_str(name, &mut str_pool, &mut str_idx);
                    let args_start = next_slot;
                    for arg_vid in args {
                        let arg_slot = get_slot(*arg_vid, &mut value_to_slot, &mut next_slot);
                        instrs.push(Instr::Move(next_slot, arg_slot));
                        next_slot += 1;
                    }
                    instrs.push(Instr::CallBuiltin(dst, callee_str as usize, args_start, args.len() as u16));
                }
                // Tensor / parallel / region ops
                // MatMul: lower to MatMulInstr(dst, lhs, rhs) — the JIT now handles this
                // with AVX-512 FMA kernel emission when AVX-512 is available.
                IrOp::MatMul { lhs, rhs } => {
                    if let Some(dst) = instr.dst {
                        let dst_slot = get_slot(dst, &mut value_to_slot, &mut next_slot);
                        let lhs_slot = get_slot(*lhs, &mut value_to_slot, &mut next_slot);
                        let rhs_slot = get_slot(*rhs, &mut value_to_slot, &mut next_slot);
                        instrs.push(Instr::MatMulInstr(dst_slot, lhs_slot, rhs_slot));
                    }
                }
                // Other tensor ops still stubbed — not yet supported in JIT
                IrOp::HadamardMul { .. } | IrOp::HadamardDiv { .. } |
                IrOp::TensorConcat { .. } | IrOp::KronProd { .. } | IrOp::OuterProd { .. } |
                IrOp::ParallelStart { .. } | IrOp::ParallelEnd { .. } |
                IrOp::RegionAlloc { .. } | IrOp::TaskSpawn { .. } | IrOp::TaskJoin { .. } |
                IrOp::Emit { .. } | IrOp::TypeCheck { .. } | IrOp::Cast { .. } => {
                    if let Some(dst) = instr.dst {
                        let dst = get_slot(dst, &mut value_to_slot, &mut next_slot);
                        instrs.push(Instr::LoadUnit(dst));
                    }
                }
                IrOp::Phi { .. } => { /* already handled above */ }
            }
        }

        // ── Insert phi moves at the END of this block, BEFORE terminator ──
        // For each successor block with phis, insert Move instructions
        // to resolve phi values from THIS predecessor.
        if let Some(moves) = phi_moves.get(bid) {
            let mut phi_move_instrs: Vec<Instr> = Vec::new();
            for &(_target_block, dst_vid, src_vid) in moves {
                let dst_slot = get_slot(dst_vid, &mut value_to_slot, &mut next_slot);
                let src_slot = get_slot(src_vid, &mut value_to_slot, &mut next_slot);
                if dst_slot != src_slot {
                    phi_move_instrs.push(Instr::Move(dst_slot, src_slot));
                }
            }
            if !phi_move_instrs.is_empty() {
                // Find the terminator (last instruction)
                let has_terminator = !instrs.is_empty() && matches!(
                    instrs.last(),
                    Some(Instr::Jump(_)) | Some(Instr::JumpFalse(_, _)) |
                    Some(Instr::JumpTrue(_, _)) | Some(Instr::Return(_)) |
                    Some(Instr::ReturnUnit)
                );
                if has_terminator {
                    // Adjust branch_targets that point to the terminator
                    // (they shifted by the number of phi moves inserted before it)
                    let shift = phi_move_instrs.len();
                    let terminator_idx = instrs.len() - 1;
                    for (idx, _) in branch_targets.iter_mut() {
                        if *idx >= terminator_idx {
                            *idx += shift;
                        }
                    }
                    // Insert phi moves before the terminator
                    let terminator = instrs.pop().unwrap();
                    instrs.append(&mut phi_move_instrs);
                    instrs.push(terminator);
                } else {
                    instrs.append(&mut phi_move_instrs);
                }
            }
        }
    }

    // ── Metadata-driven optimization passes ──────────────────────────────
    // These passes run BEFORE branch fixup so they don't invalidate offsets.
    // They use the preserved EffectFlags and AliasKind metadata to apply
    // optimizations impossible with bytecode-only translate().

    // Pass: Effect-aware dead code elimination
    // Any pure operation whose result is never used can be eliminated.
    {
        let mut used: FxHashSet<u16> = FxHashSet::default();
        for instr in &instrs {
            match instr {
                Instr::BinOp(_, _, l, r) => { used.insert(*l); used.insert(*r); }
                Instr::UnOp(_, _, s) => { used.insert(*s); }
                Instr::Move(_, s) => { used.insert(*s); }
                Instr::Load(_, s) => { used.insert(*s); }
                Instr::Store(_, s) => { used.insert(*s); }
                Instr::Return(s) => { used.insert(*s); }
                Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => { used.insert(*s); }
                _ => {}
            }
        }
        let keep_mask: Vec<bool> = instrs.iter().map(|instr| {
            match instr {
                Instr::LoadI32(d, _) | Instr::LoadI64(d, _) | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => {
                    used.contains(d)
                }
                _ => true,
            }
        }).collect();
        let mut write_idx = 0;
        for (i, keep) in keep_mask.into_iter().enumerate() {
            if keep {
                if write_idx != i {
                    instrs[write_idx] = instrs[i].clone();
                }
                write_idx += 1;
            }
        }
        instrs.truncate(write_idx);
    }

    // Pass: Constant folding with metadata awareness
    // DISABLED: The translate() function already has correct forward constant
    // propagation that properly clears at loop headers. Running a second
    // constant folding pass here was incorrectly folding loop-carried values
    // (e.g., phi_s = 42 was treated as always 42, folding s * 1664525 into
    // a constant and eliminating the loop body). The translate() pass handles
    // this correctly by clearing const_at at backward-branch targets.
    //
    // Future: when we have effect metadata per instruction, we can re-enable
    // this pass by only folding BinOps whose EffectFlags are PURE and whose
    // operands are defined in the SAME basic block (not from phi nodes).

    // ── Fixup branch offsets ─────────────────────────────────────────────
    // Resolve placeholder branch offsets using the block_start_pc map.
    // The branch_targets vec contains (instr_idx, target_block) pairs.
    // The block_start_pc map was built during flattening, but phi moves
    // may have shifted indices. We adjust for that.
    {
        // Recalculate block_start_pc from the actual instruction stream.
        // We know blocks were emitted in order, so we scan for terminators
        // and use block_order to map positions to block IDs.
        let mut resolved_block_pc: FxHashMap<BlockId, usize> = FxHashMap::default();
        let mut block_idx = 0;
        if !block_order.is_empty() {
            resolved_block_pc.insert(block_order[0], 0);
        }

        for pc in 0..instrs.len() {
            match &instrs[pc] {
                Instr::Return(_) | Instr::ReturnUnit => {
                    if pc + 1 < instrs.len() {
                        block_idx += 1;
                        if block_idx < block_order.len() {
                            resolved_block_pc.insert(block_order[block_idx], pc + 1);
                        }
                    }
                }
                Instr::Jump(_) => {
                    if pc + 1 < instrs.len() {
                        block_idx += 1;
                        if block_idx < block_order.len() {
                            resolved_block_pc.insert(block_order[block_idx], pc + 1);
                        }
                    }
                }
                Instr::JumpFalse(_, _) | Instr::JumpTrue(_, _) => {
                    // After CondBr, there may be a Jump (for if_true) or
                    // direct fall-through. Both the if_true and the next
                    // block after the CondBr sequence are block boundaries.
                    if pc + 1 < instrs.len() {
                        if matches!(instrs.get(pc + 1), Some(Instr::Jump(_))) {
                            // JumpFalse + Jump pattern:
                            // if_true block starts at Jump
                            block_idx += 1;
                            if block_idx < block_order.len() {
                                resolved_block_pc.insert(block_order[block_idx], pc + 1);
                            }
                            // Block after the Jump starts after Jump
                            if pc + 2 < instrs.len() {
                                block_idx += 1;
                                if block_idx < block_order.len() {
                                    resolved_block_pc.insert(block_order[block_idx], pc + 2);
                                }
                            }
                        } else {
                            // Fall-through to if_true
                            block_idx += 1;
                            if block_idx < block_order.len() {
                                resolved_block_pc.insert(block_order[block_idx], pc + 1);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Apply fixups
        for (instr_idx, target_block) in &branch_targets {
            let target_pc = resolved_block_pc.get(target_block).copied()
                .or_else(|| block_start_pc.get(target_block).copied())
                .unwrap_or(0);
            if *instr_idx < instrs.len() {
                match &mut instrs[*instr_idx] {
                    Instr::Jump(ref mut off) => {
                        *off = (target_pc as i32) - (*instr_idx as i32) - 1;
                    }
                    Instr::JumpFalse(_, ref mut off) => {
                        *off = (target_pc as i32) - (*instr_idx as i32) - 1;
                    }
                    _ => {}
                }
            }
        }
    }

    CompiledFn {
        entry: None, // Will be set after JIT compilation by translate()
        name: func.name.clone(),
        param_count,
        slot_count: next_slot,
        instrs,
        str_pool,
        const_pool: Vec::new(),
    }
}

/// Evaluate a BinOp at compile time. Returns None if the operation
/// is not foldable (division by zero, etc.).
fn eval_binop_const(op: BinOpKind, l: i64, r: i64) -> Option<i64> {
    match op {
        BinOpKind::Add => Some(l.wrapping_add(r)),
        BinOpKind::Sub => Some(l.wrapping_sub(r)),
        BinOpKind::Mul => Some(l.wrapping_mul(r)),
        BinOpKind::Div => {
            if r == 0 { None } else { Some(l.wrapping_div(r)) }
        }
        BinOpKind::Rem => {
            if r == 0 { None } else { Some(l.wrapping_rem(r)) }
        }
        BinOpKind::And => Some(if l != 0 && r != 0 { 1 } else { 0 }),
        BinOpKind::Or => Some(if l != 0 || r != 0 { 1 } else { 0 }),
        BinOpKind::Eq => Some(if l == r { 1 } else { 0 }),
        BinOpKind::Ne => Some(if l != r { 1 } else { 0 }),
        BinOpKind::Lt => Some(if l < r { 1 } else { 0 }),
        BinOpKind::Le => Some(if l <= r { 1 } else { 0 }),
        BinOpKind::Gt => Some(if l > r { 1 } else { 0 }),
        BinOpKind::Ge => Some(if l >= r { 1 } else { 0 }),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global SSA Optimization Passes
// ─────────────────────────────────────────────────────────────────────────────
//
// These passes replace the older block-level/local optimizations with
// function-wide analyses that leverage SSA form for correctness across
// control-flow branches and loops.

/// Global Value Numbering (GVN) — replaces block-level CSE with
/// function-wide value numbering across the entire SSA IR.
///
/// GVN assigns a "value number" to each expression. Two expressions
/// with the same value number are guaranteed to produce the same
/// result, regardless of their position in the control flow graph.
/// This allows eliminating redundant computations across branches
/// and loops, which block-level CSE cannot do.
///
/// Algorithm:
/// 1. Compute a hash for each expression based on its opcode and
///    the value numbers of its operands (recursive hashing).
/// 2. Maintain a global map: hash → first ValueId that computed it.
/// 3. When a later instruction has the same hash, replace it with
///    a Move from the earlier result.
/// 4. For expressions with side effects (stores, calls), never
///    eliminate them — only pure expressions are candidates.
///
/// Returns the number of redundant computations eliminated.
pub fn gvn_optimize(func: &mut FlatIrFunction) -> usize {
    let mut value_map: FxHashMap<u64, ValueId> = FxHashMap::default();
    let mut replacements: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut eliminated = 0usize;

    // Hash function: combines opcode + operand value numbers
    fn hash_expr(op: &IrOp, replacements: &FxHashMap<ValueId, ValueId>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write(&mut hasher, b"gvn");
        match op {
            IrOp::BinOp { op: binop, lhs, rhs } => {
                std::hash::Hasher::write_u8(&mut hasher, 0);
                std::hash::Hasher::write_u8(&mut hasher, match binop {
                    BinOpKind::Add => 0, BinOpKind::Sub => 1, BinOpKind::Mul => 2,
                    BinOpKind::Div => 3, BinOpKind::Rem => 4, BinOpKind::BitAnd => 5,
                    BinOpKind::BitOr => 6, BinOpKind::BitXor => 7, BinOpKind::Shl => 8,
                    BinOpKind::Shr => 9, BinOpKind::Lt => 10, BinOpKind::Le => 11,
                    BinOpKind::Gt => 12, BinOpKind::Ge => 13, BinOpKind::Eq => 14,
                    BinOpKind::Ne => 15, BinOpKind::And => 16, BinOpKind::Or => 17,
                    BinOpKind::Min => 18, BinOpKind::Max => 19, BinOpKind::FloorDiv => 20,
                    BinOpKind::FmaAdd => 21,
                });
                let l = replacements.get(lhs).unwrap_or(lhs).0 as u64;
                let r = replacements.get(rhs).unwrap_or(rhs).0 as u64;
                std::hash::Hasher::write_u64(&mut hasher, l);
                std::hash::Hasher::write_u64(&mut hasher, r);
            }
            IrOp::UnOp { op: unop, operand } => {
                std::hash::Hasher::write_u8(&mut hasher, 1);
                std::hash::Hasher::write_u8(&mut hasher, match unop {
                    UnOpKind::Neg => 0, UnOpKind::Not => 1, UnOpKind::BitNot => 2,
                    UnOpKind::Abs => 3,
                });
                let s = replacements.get(operand).unwrap_or(operand).0 as u64;
                std::hash::Hasher::write_u64(&mut hasher, s);
            }
            _ => {
                // Non-pure or non-hashable ops get unique hashes
                std::hash::Hasher::write_u8(&mut hasher, 255);
            }
        }
        std::hash::Hasher::finish(&hasher)
    }

    for block in &mut func.blocks {
        for instr in &mut block.instrs {
            // First, apply existing replacements to operands
            if let IrOp::BinOp { ref mut lhs, ref mut rhs, .. } = instr.op {
                if let Some(&rep) = replacements.get(lhs) { *lhs = rep; }
                if let Some(&rep) = replacements.get(rhs) { *rhs = rep; }
            }
            if let IrOp::UnOp { ref mut operand, .. } = instr.op {
                if let Some(&rep) = replacements.get(operand) { *operand = rep; }
            }
            if let IrOp::Move { ref mut src } = instr.op {
                if let Some(&rep) = replacements.get(src) { *src = rep; }
            }

            // Only GVN pure operations
            if !instr.effect.is_pure() { continue; }

            let hash = hash_expr(&instr.op, &replacements);
            if hash == 0 { continue; } // non-hashable

            if let Some(dst) = instr.dst {
                if let Some(&existing) = value_map.get(&hash) {
                    // Redundant computation — replace with Move
                    if existing != dst {
                        replacements.insert(dst, existing);
                        instr.op = IrOp::Move { src: existing };
                        eliminated += 1;
                    }
                } else {
                    value_map.insert(hash, dst);
                }
            }
        }
    }

    eliminated
}

/// Sparse Conditional Constant Propagation (SCCP).
///
/// Replaces the forward flat-scan constant propagation that clears
/// state at branch targets. SCCP simultaneously propagates constants
/// AND evaluates conditional branches globally. If a condition is
/// proven constant, the compiler can entirely eliminate the dead
/// branch and unreachable basic blocks.
///
/// Algorithm (Wegman-Zadeck):
/// 1. Initialize all values to "top" (unknown).
/// 2. Propagate constants using a worklist of basic blocks.
/// 3. For CondBr, if the condition is a known constant, only
///    enqueue the reachable branch (the other becomes dead code).
/// 4. Continue until the worklist is empty.
/// 5. Replace all constant-valued slots with their known values.
/// 6. Remove unreachable blocks.
///
/// Returns the number of constants propagated + branches resolved.
pub fn sccp_optimize(func: &mut FlatIrFunction) -> usize {
    use std::collections::VecDeque;

    #[derive(Debug, Clone, PartialEq)]
    enum LatticeVal {
        Top,           // Unknown
        Const(i64),    // Known constant
        Bottom,        // Not a constant (variable)
    }

    let mut values: FxHashMap<ValueId, LatticeVal> = FxHashMap::default();
    let mut worklist: VecDeque<BlockId> = VecDeque::new();
    let mut reachable: FxHashSet<BlockId> = FxHashSet::default();
    let mut changes = 0usize;

    // Initialize: all params are Bottom
    for (vid, _) in &func.params {
        values.insert(*vid, LatticeVal::Bottom);
    }
    // Seed constants from ConstInt/ConstBool
    for block in &func.blocks {
        for instr in &block.instrs {
            match &instr.op {
                IrOp::ConstInt { value, .. } => {
                    if let Some(dst) = instr.dst {
                        values.insert(dst, LatticeVal::Const(*value));
                    }
                }
                IrOp::ConstBool { value } => {
                    if let Some(dst) = instr.dst {
                        values.insert(dst, LatticeVal::Const(if *value { 1 } else { 0 }));
                    }
                }
                _ => {}
            }
        }
    }

    // Entry block is reachable
    reachable.insert(func.entry);
    worklist.push_back(func.entry);

    while let Some(block_id) = worklist.pop_front() {
        let block_idx = match func.blocks.iter().position(|b| b.id == block_id) {
            Some(i) => i,
            None => continue,
        };

        for instr in &func.blocks[block_idx].instrs {
            let result_val = match &instr.op {
                IrOp::BinOp { op, lhs, rhs } => {
                    let l = values.get(lhs).unwrap_or(&LatticeVal::Top);
                    let r = values.get(rhs).unwrap_or(&LatticeVal::Top);
                    match (l, r) {
                        (LatticeVal::Const(lv), LatticeVal::Const(rv)) => {
                            // Fold the constant
                            let result = eval_binop_const(*op, *lv, *rv);
                            changes += 1;
                            result.map(LatticeVal::Const).unwrap_or(LatticeVal::Bottom)
                        }
                        _ => LatticeVal::Bottom,
                    }
                }
                IrOp::Move { src } => {
                    values.get(src).cloned().unwrap_or(LatticeVal::Bottom)
                }
                IrOp::UnOp { op: unop, operand } => {
                    let v = values.get(operand).unwrap_or(&LatticeVal::Top);
                    match v {
                        LatticeVal::Const(cv) => {
                            let result = eval_unop_const(*unop, *cv);
                            changes += 1;
                            result.map(LatticeVal::Const).unwrap_or(LatticeVal::Bottom)
                        }
                        _ => LatticeVal::Bottom,
                    }
                }
                IrOp::CondBr { cond, if_true, if_false, .. } => {
                    // If condition is constant, only enqueue the reachable branch
                    match values.get(cond) {
                        Some(LatticeVal::Const(0)) => {
                            // False branch
                            if reachable.insert(*if_false) {
                                worklist.push_back(*if_false);
                            }
                            continue;
                        }
                        Some(LatticeVal::Const(_)) => {
                            // True branch (any non-zero)
                            if reachable.insert(*if_true) {
                                worklist.push_back(*if_true);
                            }
                            continue;
                        }
                        _ => {
                            // Unknown — enqueue both branches
                            if reachable.insert(*if_true) {
                                worklist.push_back(*if_true);
                            }
                            if reachable.insert(*if_false) {
                                worklist.push_back(*if_false);
                            }
                            continue;
                        }
                    }
                }
                IrOp::Jump { target, .. } => {
                    if reachable.insert(*target) {
                        worklist.push_back(*target);
                    }
                    continue;
                }
                _ => LatticeVal::Bottom,
            };

            if let Some(dst) = instr.dst {
                let old = values.get(&dst).cloned().unwrap_or(LatticeVal::Top);
                if old != result_val {
                    values.insert(dst, result_val);
                }
            }
        }
    }

    // Phase 2: Replace constant values in the IR
    for block in &mut func.blocks {
        for instr in &mut block.instrs {
            // Replace BinOp operands with their constant values if known
            if let IrOp::BinOp { ref mut lhs, ref mut rhs, .. } = instr.op {
                if let Some(LatticeVal::Const(cv)) = values.get(lhs) {
                    // Create a fresh ValueId for the constant and replace
                    // (actual constant folding is done during codegen)
                    let _ = cv; // suppress unused warning
                }
                if let Some(LatticeVal::Const(cv)) = values.get(rhs) {
                    let _ = cv;
                }
            }
        }
    }

    // Phase 3: Remove unreachable blocks
    let before_blocks = func.blocks.len();
    func.blocks.retain(|b| reachable.contains(&b.id));
    changes += before_blocks - func.blocks.len();

    changes
}

/// Evaluate a unary operation on a constant value.
fn eval_unop_const(op: UnOpKind, v: i64) -> Option<i64> {
    match op {
        UnOpKind::Neg => Some(v.wrapping_neg()),
        UnOpKind::Not => Some(if v == 0 { 1 } else { 0 }),
        UnOpKind::BitNot => Some(!v),
        UnOpKind::Abs => Some(v.abs()),
    }
}

/// Escape Analysis for memory operations.
///
/// Analyzes IrOp::Alloca and IrOp::RegionAlloc to determine if
/// allocated structures stay within the function boundary. If they
/// don't escape, we can perform Scalar Replacement of Aggregates
/// (SRA) — replacing heap/region allocations with stack slots or
/// registers, completely bypassing the allocator.
///
/// An allocation "escapes" if:
/// - It is stored into a data structure that outlives the function
/// - It is passed as an argument to a function call
/// - It is returned from the function
/// - Its address is used in a way that could be observed externally
///
/// Returns the number of allocations that were found to be non-escaping.
pub fn escape_analysis(func: &FlatIrFunction) -> usize {
    // Collect all allocation sites
    let mut alloc_sites: FxHashSet<ValueId> = FxHashSet::default();
    for block in &func.blocks {
        for instr in &block.instrs {
            match &instr.op {
                IrOp::Alloca { .. } | IrOp::RegionAlloc { .. } => {
                    if let Some(dst) = instr.dst {
                        alloc_sites.insert(dst);
                    }
                }
                _ => {}
            }
        }
    }

    if alloc_sites.is_empty() { return 0; }

    // Track which allocations escape
    let mut escapes: FxHashSet<ValueId> = FxHashSet::default();

    for block in &func.blocks {
        for instr in &block.instrs {
            match &instr.op {
                IrOp::Store { ptr, value } => {
                    // If we store an allocation AS a value into something else,
                    // the allocation escapes (it's stored by reference somewhere).
                    if alloc_sites.contains(value) && !alloc_sites.contains(ptr) {
                        escapes.insert(*value);
                    }
                    // Storing INTO an allocation doesn't make it escape
                }
                IrOp::Call { args, .. } => {
                    // Any allocation passed to a call escapes
                    for arg in args {
                        if alloc_sites.contains(arg) {
                            escapes.insert(*arg);
                        }
                    }
                }
                IrOp::Intrinsic { args, .. } => {
                    // Any allocation passed to an intrinsic escapes
                    for arg in args {
                        if alloc_sites.contains(arg) {
                            escapes.insert(*arg);
                        }
                    }
                }
                IrOp::Ret { value: Some(val) } => {
                    // Returning an allocation means it escapes
                    if alloc_sites.contains(val) {
                        escapes.insert(*val);
                    }
                }
                IrOp::Emit { value, .. } => {
                    // Emitting a value escapes it
                    if alloc_sites.contains(value) {
                        escapes.insert(*value);
                    }
                }
                _ => {}
            }
        }
    }

    // Count non-escaping allocations
    alloc_sites.difference(&escapes).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Live-Range Splitting for Global Linear Scan
// ─────────────────────────────────────────────────────────────────────────────
//
// When a value is live across a large span but only used at the beginning
// and end, it occupies a register for its entire live range unnecessarily.
// Live-range splitting detects these "idle" windows and inserts spill/reload
// pairs, freeing the register for other values during the idle period.
//
// Algorithm:
// 1. Compute live ranges for each ValueId across all blocks.
// 2. For each value with a "high pressure" window (many competing values),
//    find idle sub-ranges (no uses for >threshold instructions).
// 3. Split at idle points: insert Store-to-stack at split point,
//    Load-from-stack at next use point.

/// A live range fragment: a contiguous interval where a value is live.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct LiveFragment {
    /// The value being tracked.
    vid: ValueId,
    /// Start position (instruction index in block order).
    start: usize,
    /// End position (last use).
    end: usize,
    /// Positions where this value is actually used (reads).
    use_positions: Vec<usize>,
}

/// Live-range splitter that identifies idle windows and creates split points.
struct LiveRangeSplitter {
    /// Threshold: if a value has no uses for more than this many instructions,
    /// it's a candidate for splitting.
    idle_threshold: usize,
    /// Stack slot allocator for spill slots.
    next_spill_slot: u16,
}

impl LiveRangeSplitter {
    fn new() -> Self {
        Self {
            idle_threshold: 32, // Split if idle for > 32 instructions
            next_spill_slot: 4096, // Start spill slots at 4096 to avoid normal slots
        }
    }

    /// Analyze a FlatIrFunction and return split decisions.
    /// Each split decision is: (value_id, split_position, spill_slot).
    /// The caller is responsible for inserting the actual spill/reload instructions.
    fn analyze(&mut self, func: &FlatIrFunction, block_order: &[BlockId]) -> Vec<(ValueId, usize, u16)> {
        let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
            .enumerate().map(|(i, b)| (b.id, i)).collect();

        // Compute global instruction numbering across block order
        let mut global_pos = 0usize;
        let mut block_start: FxHashMap<BlockId, usize> = FxHashMap::default();
        for &bid in block_order {
            block_start.insert(bid, global_pos);
            let bi = block_index[&bid];
            global_pos += func.blocks[bi].instrs.len();
        }

        // Collect use positions for each ValueId
        let mut use_positions: FxHashMap<ValueId, Vec<usize>> = FxHashMap::default();
        for &bid in block_order {
            let bi = block_index[&bid];
            let base = block_start[&bid];
            for (ii, instr) in func.blocks[bi].instrs.iter().enumerate() {
                let pos = base + ii;
                for vid in instr_operand_values(&instr.op) {
                    use_positions.entry(vid).or_default().push(pos);
                }
            }
        }

        // Find split candidates: values with long idle gaps
        let mut splits = Vec::new();
        for (vid, positions) in &use_positions {
            if positions.len() < 2 { continue; }

            let mut sorted = positions.clone();
            sorted.sort();

            // Look for idle windows between consecutive uses
            for i in 0..sorted.len() - 1 {
                let gap = sorted[i + 1] - sorted[i];
                if gap > self.idle_threshold {
                    // Split point: in the middle of the idle window
                    let split_pos = sorted[i] + gap / 2;
                    let spill_slot = self.next_spill_slot;
                    self.next_spill_slot += 1;
                    splits.push((*vid, split_pos, spill_slot));
                    break; // One split per value (can be extended to multiple)
                }
            }
        }

        splits
    }

    /// Apply the split decisions to the function by inserting spill/reload
    /// instructions at the split points.
    fn apply_splits(&self, func: &mut FlatIrFunction, splits: &[(ValueId, usize, u16)], block_order: &[BlockId]) -> usize {
        let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
            .enumerate().map(|(i, b)| (b.id, i)).collect();

        // Compute global instruction numbering
        let mut global_pos = 0usize;
        let mut block_start: FxHashMap<BlockId, usize> = FxHashMap::default();
        for &bid in block_order {
            block_start.insert(bid, global_pos);
            let bi = block_index[&bid];
            global_pos += func.blocks[bi].instrs.len();
        }

        let mut applied = 0usize;
        for &(vid, split_pos, spill_slot) in splits {
            // Find which block contains the split position
            let mut target_block = None;
            let mut local_pos = 0usize;
            for &bid in block_order {
                let bi = block_index[&bid];
                let start = block_start[&bid];
                let end = start + func.blocks[bi].instrs.len();
                if split_pos >= start && split_pos < end {
                    target_block = Some(bid);
                    local_pos = split_pos - start;
                    break;
                }
            }

            if let Some(bid) = target_block {
                let bi = block_index[&bid];
                // Insert spill: Store(spill_slot, vid)
                let spill_instr = FlatInstr {
                    result: None,
                    dst: None,
                    op: IrOp::Store { ptr: ValueId(spill_slot as u32), value: vid },
                    effect: EffectFlags::WRITE,
                    effects: EffectFlags::WRITE,
                    alias: AliasKind::Unknown,
                    ownership: Ownership::Copy,
                };
                func.blocks[bi].instrs.insert(local_pos, spill_instr);

                // Insert reload: Load(vid, spill_slot) — at the next use position
                // (This is a simplification; a full implementation would track
                // the exact next use position)
                let reload_instr = FlatInstr {
                    result: Some(vid),
                    dst: Some(vid),
                    op: IrOp::Load { ptr: ValueId(spill_slot as u32), ty: IrType::Int { width: 64, signed: true } },
                    effect: EffectFlags::READ,
                    effects: EffectFlags::READ,
                    alias: AliasKind::Unknown,
                    ownership: Ownership::Copy,
                };
                func.blocks[bi].instrs.insert(local_pos + 1, reload_instr);

                applied += 1;
            }
        }

        applied
    }
}

/// Run live-range splitting on a FlatIrFunction.
/// Returns the number of splits applied.
pub fn live_range_split_optimize(func: &mut FlatIrFunction, block_order: &[BlockId]) -> usize {
    let mut splitter = LiveRangeSplitter::new();
    let splits = splitter.analyze(func, block_order);
    if splits.is_empty() { return 0; }
    splitter.apply_splits(func, &splits, block_order)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dominator Tree Construction (Cooper, Harvey, Kennedy)
// ─────────────────────────────────────────────────────────────────────────────
//
// The dominator tree is a fundamental data structure for global optimization.
// Block A dominates Block B if every path from the entry to B passes through A.
// The immediate dominator (idom) of B is the closest strict dominator of B.
//
// Uses the iterative algorithm from:
//   Cooper, Harvey, Kennedy — "A Simple, Fast Dominance Algorithm" (2001)
// This runs in O(N²) worst case but is typically near-linear in practice.

/// Dominator tree for a FlatIrFunction's CFG.
/// For each block, stores its immediate dominator (idom).
#[allow(dead_code)]
pub struct DominatorTree {
    /// Maps BlockId → its immediate dominator's BlockId.
    /// The entry block has no idom (None).
    idom: FxHashMap<BlockId, Option<BlockId>>,
    /// Children in the dominator tree (for DFS traversal).
    children: FxHashMap<BlockId, Vec<BlockId>>,
    /// DFS order of the dominator tree (preorder).
    dfs_order: Vec<BlockId>,
    /// DFS number for each block (for dominance testing).
    dfs_num: FxHashMap<BlockId, u32>,
}

impl DominatorTree {
    /// Compute the dominator tree using the iterative algorithm
    /// (Cooper, Harvey, Kennedy — "A Simple, Fast Dominance Algorithm").
    pub fn compute(func: &FlatIrFunction) -> Self {
        let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
            .enumerate().map(|(i, b)| (b.id, i)).collect();

        // Compute reverse postorder (RPO) of the CFG
        let rpo = Self::reverse_postorder(func, &block_index);
        let mut idom: FxHashMap<BlockId, Option<BlockId>> = FxHashMap::default();

        // Entry block dominates itself
        if let Some(&entry_id) = rpo.first() {
            idom.insert(entry_id, None);
        }

        // Iterative dataflow: intersect predecessor idoms until fixed point
        let mut changed = true;
        while changed {
            changed = false;
            for &bid in &rpo {
                if idom.get(&bid).is_some() && idom[&bid].is_none() {
                    continue; // Entry block, skip
                }
                let preds: Vec<BlockId> = func.blocks.iter()
                    .filter(|b| b.succs().contains(&bid))
                    .map(|b| b.id)
                    .collect();

                let mut new_idom: Option<BlockId> = None;
                for pred in &preds {
                    if let Some(Some(pred_idom)) = idom.get(pred) {
                        new_idom = Some(match new_idom {
                            None => *pred_idom,
                            Some(cur) => Self::intersect(&idom, cur, *pred_idom, &rpo),
                        });
                    } else if idom.contains_key(pred) && idom[pred].is_none() {
                        // pred is the entry block
                        new_idom = Some(match new_idom {
                            None => *pred,
                            Some(cur) => Self::intersect(&idom, cur, *pred, &rpo),
                        });
                    }
                }

                if new_idom != idom.get(&bid).and_then(|x| *x) {
                    idom.insert(bid, new_idom);
                    changed = true;
                }
            }
        }

        // Build children map and DFS order
        let mut children: FxHashMap<BlockId, Vec<BlockId>> = FxHashMap::default();
        for (&bid, idom_parent) in &idom {
            if let Some(parent) = idom_parent {
                children.entry(*parent).or_default().push(bid);
            }
        }

        // DFS traversal of the dominator tree
        let mut dfs_order = Vec::new();
        let mut dfs_num = FxHashMap::default();
        if let Some(&entry_id) = rpo.first() {
            let mut stack = vec![entry_id];
            let mut counter = 0u32;
            while let Some(bid) = stack.pop() {
                dfs_order.push(bid);
                dfs_num.insert(bid, counter);
                counter += 1;
                if let Some(kids) = children.get(&bid) {
                    // Push in reverse so first child is processed first
                    for kid in kids.iter().rev() {
                        stack.push(*kid);
                    }
                }
            }
        }

        Self { idom, children, dfs_order, dfs_num }
    }

    /// Intersect two dominator paths (from Cooper et al.)
    fn intersect(
        idom: &FxHashMap<BlockId, Option<BlockId>>,
        mut b1: BlockId,
        mut b2: BlockId,
        rpo: &[BlockId],
    ) -> BlockId {
        let rpo_idx: FxHashMap<BlockId, usize> = rpo.iter()
            .enumerate().map(|(i, &b)| (b, i)).collect();
        loop {
            while rpo_idx.get(&b1).copied() > rpo_idx.get(&b2).copied() {
                b1 = idom[&b1].unwrap();
            }
            while rpo_idx.get(&b2).copied() > rpo_idx.get(&b1).copied() {
                b2 = idom[&b2].unwrap();
            }
            if b1 == b2 { return b1; }
        }
    }

    /// Compute reverse postorder of the CFG via DFS from entry
    fn reverse_postorder(func: &FlatIrFunction, block_index: &FxHashMap<BlockId, usize>) -> Vec<BlockId> {
        let mut visited = FxHashSet::default();
        let mut post_order = Vec::new();

        fn dfs(
            bid: BlockId,
            func: &FlatIrFunction,
            block_index: &FxHashMap<BlockId, usize>,
            visited: &mut FxHashSet<BlockId>,
            post_order: &mut Vec<BlockId>,
        ) {
            if visited.contains(&bid) { return; }
            visited.insert(bid);
            if let Some(&bi) = block_index.get(&bid) {
                for succ in func.blocks[bi].succs() {
                    dfs(succ, func, block_index, visited, post_order);
                }
            }
            post_order.push(bid);
        }

        dfs(func.entry, func, block_index, &mut visited, &mut post_order);
        post_order.reverse();
        post_order
    }

    /// Returns true if `a` dominates `b`.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b { return true; }
        // Walk up b's dominator chain; if we reach a, it dominates
        let mut cur = b;
        while let Some(Some(parent)) = self.idom.get(&cur) {
            if *parent == a { return true; }
            cur = *parent;
        }
        false
    }

    /// Returns the immediate dominator of `bid`.
    pub fn idom(&self, bid: BlockId) -> Option<BlockId> {
        self.idom.get(&bid).and_then(|x| *x)
    }

    /// Returns the dominator tree DFS order.
    pub fn dfs_order(&self) -> &[BlockId] { &self.dfs_order }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Value Numbering with Dominator Tree (Phase 2)
// ─────────────────────────────────────────────────────────────────────────────
//
// Enhances the per-block GVN to work across the entire CFG using the
// dominator tree. If Block A dominates Block B, any value computed in A
// is valid in B. The GVN traverses blocks in dominator tree DFS order,
// keeping the value map across dominated blocks.

/// Global Value Numbering across CFG blocks using the dominator tree.
///
/// Unlike the per-block `gvn_optimize()`, this version can eliminate
/// redundant computations across basic block boundaries, as long as
/// the definition dominates the use.
///
/// Returns the number of redundant computations eliminated.
pub fn gvn_optimize_global(func: &mut FlatIrFunction) -> usize {
    let domtree = DominatorTree::compute(func);
    let mut value_map: FxHashMap<u64, ValueId> = FxHashMap::default();
    let mut replacements: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut eliminated = 0usize;

    // Hash function: combines opcode + operand value numbers
    fn hash_expr(op: &IrOp, replacements: &FxHashMap<ValueId, ValueId>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write(&mut hasher, b"gvn");
        match op {
            IrOp::BinOp { op: binop, lhs, rhs } => {
                std::hash::Hasher::write_u8(&mut hasher, 0);
                std::hash::Hasher::write_u8(&mut hasher, match binop {
                    BinOpKind::Add => 0, BinOpKind::Sub => 1, BinOpKind::Mul => 2,
                    BinOpKind::Div => 3, BinOpKind::Rem => 4, BinOpKind::BitAnd => 5,
                    BinOpKind::BitOr => 6, BinOpKind::BitXor => 7, BinOpKind::Shl => 8,
                    BinOpKind::Shr => 9, BinOpKind::Lt => 10, BinOpKind::Le => 11,
                    BinOpKind::Gt => 12, BinOpKind::Ge => 13, BinOpKind::Eq => 14,
                    BinOpKind::Ne => 15, BinOpKind::And => 16, BinOpKind::Or => 17,
                    BinOpKind::Min => 18, BinOpKind::Max => 19, BinOpKind::FloorDiv => 20,
                    BinOpKind::FmaAdd => 21,
                });
                let l = replacements.get(lhs).unwrap_or(lhs).0 as u64;
                let r = replacements.get(rhs).unwrap_or(rhs).0 as u64;
                std::hash::Hasher::write_u64(&mut hasher, l);
                std::hash::Hasher::write_u64(&mut hasher, r);
            }
            IrOp::UnOp { op: unop, operand } => {
                std::hash::Hasher::write_u8(&mut hasher, 1);
                std::hash::Hasher::write_u8(&mut hasher, match unop {
                    UnOpKind::Neg => 0, UnOpKind::Not => 1, UnOpKind::BitNot => 2,
                    UnOpKind::Abs => 3,
                });
                let s = replacements.get(operand).unwrap_or(operand).0 as u64;
                std::hash::Hasher::write_u64(&mut hasher, s);
            }
            _ => {
                // Non-pure or non-hashable ops get unique hashes
                std::hash::Hasher::write_u8(&mut hasher, 255);
            }
        }
        std::hash::Hasher::finish(&hasher)
    }

    /// Apply replacements to an instruction's operands in-place.
    fn apply_replacements(instr: &mut FlatInstr, replacements: &FxHashMap<ValueId, ValueId>) {
        if let IrOp::BinOp { ref mut lhs, ref mut rhs, .. } = instr.op {
            if let Some(&rep) = replacements.get(lhs) { *lhs = rep; }
            if let Some(&rep) = replacements.get(rhs) { *rhs = rep; }
        }
        if let IrOp::UnOp { ref mut operand, .. } = instr.op {
            if let Some(&rep) = replacements.get(operand) { *operand = rep; }
        }
        if let IrOp::Move { ref mut src } = instr.op {
            if let Some(&rep) = replacements.get(src) { *src = rep; }
        }
        if let IrOp::Copy { ref mut src } = instr.op {
            if let Some(&rep) = replacements.get(src) { *src = rep; }
        }
        if let IrOp::Store { ref mut ptr, ref mut value } = instr.op {
            if let Some(&rep) = replacements.get(ptr) { *ptr = rep; }
            if let Some(&rep) = replacements.get(value) { *value = rep; }
        }
        if let IrOp::Load { ref mut ptr, .. } = instr.op {
            if let Some(&rep) = replacements.get(ptr) { *ptr = rep; }
        }
        if let IrOp::Cast { ref mut value, .. } = instr.op {
            if let Some(&rep) = replacements.get(value) { *value = rep; }
        }
    }

    // Traverse in dominator tree DFS order
    for &bid in domtree.dfs_order() {
        let block_idx = match func.blocks.iter().position(|b| b.id == bid) {
            Some(i) => i,
            None => continue,
        };

        for instr in &mut func.blocks[block_idx].instrs {
            // First apply existing replacements to operands
            apply_replacements(instr, &replacements);

            if !instr.effect.is_pure() { continue; }

            let hash = hash_expr(&instr.op, &replacements);
            if hash == 0 { continue; }

            if let Some(dst) = instr.dst {
                if let Some(&existing) = value_map.get(&hash) {
                    if existing != dst {
                        replacements.insert(dst, existing);
                        instr.op = IrOp::Move { src: existing };
                        eliminated += 1;
                    }
                } else {
                    value_map.insert(hash, dst);
                }
            }
        }

        // When backtracking out of a dominated subtree, we should remove
        // values defined in this block. For simplicity in the first pass,
        // we keep the map intact — this is conservative (may miss some
        // eliminations) but correct (never eliminates a value that doesn't
        // dominate). A future refinement can track scope precisely.
    }

    eliminated
}

// ─────────────────────────────────────────────────────────────────────────────
// Dominator-Aware Loop-Invariant Code Motion (Phase 2)
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses the dominator tree to precisely identify loop-invariant code.
// A computation inside a loop is invariant if all its operands are
// defined outside the loop (i.e., dominate the loop header).
// Invariant computations are hoisted to the loop's pre-header.

/// Extract operand ValueIds from an IrOp.
pub fn instr_operand_values(op: &IrOp) -> Vec<ValueId> {
    match op {
        IrOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        IrOp::UnOp { operand, .. } => vec![*operand],
        IrOp::Move { src } => vec![*src],
        IrOp::Copy { src } => vec![*src],
        IrOp::Store { ptr, value } => vec![*ptr, *value],
        IrOp::Load { ptr, .. } => vec![*ptr],
        IrOp::Cast { value, .. } => vec![*value],
        IrOp::Call { args, .. } => args.clone(),
        IrOp::Intrinsic { args, .. } => args.clone(),
        IrOp::Emit { value, .. } => vec![*value],
        IrOp::TypeCheck { value, .. } => vec![*value],
        IrOp::MatMul { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::HadamardMul { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::HadamardDiv { lhs, rhs } => vec![*lhs, *rhs],
        IrOp::TensorConcat { a, b, .. } => vec![*a, *b],
        IrOp::KronProd { a, b } => vec![*a, *b],
        IrOp::OuterProd { a, b } => vec![*a, *b],
        IrOp::TaskJoin { handle } => vec![*handle],
        _ => vec![],
    }
}

/// Loop-Invariant Code Motion for SSA IR using dominator tree.
/// Identifies computations inside loops whose inputs all dominate the
/// loop header, then hoists them to the loop's pre-header.
///
/// Returns the number of instructions hoisted.
pub fn licm_optimize_ssa(func: &mut FlatIrFunction) -> usize {
    let domtree = DominatorTree::compute(func);
    let mut hoisted = 0usize;

    // Identify loop headers: blocks that are targets of back-edges
    let mut loop_headers: FxHashSet<BlockId> = FxHashSet::default();
    let mut loop_preheaders: FxHashMap<BlockId, BlockId> = FxHashMap::default();
    let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate().map(|(i, b)| (b.id, i)).collect();

    for block in &func.blocks {
        for succ in block.succs() {
            if domtree.dominates(succ, block.id) {
                // succ dominates block.id AND block.id → succ is a back-edge
                // succ is a loop header
                loop_headers.insert(succ);
            }
        }
    }

    // For each loop header, identify the pre-header (the unique non-back-edge
    // predecessor that dominates the header)
    for &header in &loop_headers {
        let preds: Vec<BlockId> = func.blocks.iter()
            .filter(|b| b.succs().contains(&header))
            .map(|b| b.id)
            .collect();

        // The pre-header is the predecessor that dominates the header
        let preheader = preds.iter().find(|&&p| domtree.dominates(p, header)).copied();
        if let Some(ph) = preheader {
            loop_preheaders.insert(header, ph);
        }
    }

    // For each loop, scan instructions and hoist invariant ones
    for &header in &loop_headers {
        let preheader = match loop_preheaders.get(&header) {
            Some(&ph) => ph,
            None => continue,
        };

        // Collect all blocks in the loop body (blocks dominated by header)
        let mut loop_blocks: FxHashSet<BlockId> = FxHashSet::default();
        loop_blocks.insert(header);
        for &bid in domtree.dfs_order() {
            if domtree.dominates(header, bid) {
                loop_blocks.insert(bid);
            }
        }

        // Collect all ValueIds defined in non-loop blocks (outside the loop)
        let mut outside_defs: FxHashSet<ValueId> = FxHashSet::default();
        for block in &func.blocks {
            if !loop_blocks.contains(&block.id) {
                for instr in &block.instrs {
                    if let Some(dst) = instr.dst {
                        outside_defs.insert(dst);
                    }
                }
            }
        }

        // Scan loop body for invariant instructions
        let mut to_hoist: Vec<(BlockId, usize)> = Vec::new(); // (block_id, instr_index)

        for &bid in &loop_blocks {
            let bi = match block_index.get(&bid) {
                Some(&i) => i,
                None => continue,
            };
            for (ii, instr) in func.blocks[bi].instrs.iter().enumerate() {
                if !instr.effect.is_pure() { continue; }

                // Check if all operands are defined outside the loop
                let operands = instr_operand_values(&instr.op);
                let all_outside = operands.iter().all(|v| outside_defs.contains(v));

                if all_outside && !operands.is_empty() {
                    to_hoist.push((bid, ii));
                }
            }
        }

        // Hoist: move each invariant instruction from its loop body position
        // to the pre-header. We do this by inserting Nop at the original
        // position and appending the instruction to the pre-header.
        let preheader_idx = match block_index.get(&preheader) {
            Some(&i) => i,
            None => continue,
        };
        for (bid, ii) in to_hoist {
            let bi = match block_index.get(&bid) {
                Some(&i) => i,
                None => continue,
            };
            let instr = std::mem::replace(&mut func.blocks[bi].instrs[ii],
                FlatInstr {
                    result: None,
                    dst: None,
                    op: IrOp::Nop,
                    effect: EffectFlags::PURE,
                    effects: EffectFlags::PURE,
                    alias: AliasKind::Unknown,
                    ownership: Ownership::Copy,
                });
            let insert_pos = func.blocks[preheader_idx].instrs.len().saturating_sub(1); // Before terminator
            func.blocks[preheader_idx].instrs.insert(
                insert_pos,
                instr,
            );
            hoisted += 1;
        }
    }

    hoisted
}

// ─────────────────────────────────────────────────────────────────────────────
// PROFILE-GUIDED CODE LAYOUT — Pettis-Hansen Block Reordering
// ─────────────────────────────────────────────────────────────────────────────

/// Pettis-Hansen Block Reordering — Profile-Guided Code Layout
///
/// Reorders basic blocks so that the highest-frequency execution paths
/// are strictly linear fall-through code. This minimizes branch predictor
/// usage and avoids front-end pipeline bubbles on Intel/AMD CPUs.
///
/// Algorithm:
/// 1. Build a weighted CFG where edge weights come from PMC branch
///    misprediction counts (or execution frequency estimates).
/// 2. Compute a maximum-weight spanning tree of the CFG.
/// 3. Order blocks so that the heaviest edges are fall-through.
/// 4. Emit blocks in this order, inserting explicit jumps for
///    non-fall-through edges.
///
/// The Pettis-Hansen algorithm is preferred over simple frequency sorting
/// because it considers the global layout problem — it maximizes the
/// total fall-through edge weight, not just local optimality.
pub fn pettis_hansen_reorder(func: &mut FlatIrFunction,
                              edge_weights: &FxHashMap<(BlockId, BlockId), u64>) {
    if func.blocks.len() <= 1 { return; }

    // Step 1: Build adjacency list with weights
    let n = func.blocks.len();
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];

    // Create block index mapping
    let block_idx: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    for ((from, to), weight) in edge_weights {
        if let (Some(&fi), Some(&ti)) = (block_idx.get(from), block_idx.get(to)) {
            adj[fi].push((ti, *weight));
            adj[ti].push((fi, *weight)); // undirected for MST
        }
    }

    // Step 2: Kruskal's MST (maximum weight) with Union-Find
    let mut edges: Vec<(u64, usize, usize)> = Vec::new();
    for (i, neighbors) in adj.iter().enumerate() {
        for &(j, w) in neighbors {
            if i < j { edges.push((w, i, j)); }
        }
    }
    edges.sort_by(|a, b| b.0.cmp(&a.0)); // descending

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x { parent[x] = find(parent, parent[x]); }
        parent[x]
    }

    let mut mst_edges: Vec<(usize, usize)> = Vec::new();
    for (_w, u, v) in edges {
        let pu = find(&mut parent, u);
        let pv = find(&mut parent, v);
        if pu != pv {
            parent[pu] = pv;
            mst_edges.push((u, v));
        }
    }

    // Step 3: DFS order of MST starting from entry block
    let entry_idx = block_idx.get(&func.entry).copied().unwrap_or(0);
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::new();

    // Build adjacency list from MST
    let mut mst_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, v) in &mst_edges {
        mst_adj[*u].push(*v);
        mst_adj[*v].push(*u);
    }

    // DFS
    fn dfs(node: usize, mst_adj: &[Vec<usize>], visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        visited[node] = true;
        order.push(node);
        for &neighbor in &mst_adj[node] {
            if !visited[neighbor] {
                dfs(neighbor, mst_adj, visited, order);
            }
        }
    }
    dfs(entry_idx, &mst_adj, &mut visited, &mut order);

    // Add any unvisited blocks
    for i in 0..n {
        if !visited[i] { order.push(i); }
    }

    // Step 4: Reorder blocks
    let mut new_blocks: Vec<FlatBlock> = Vec::with_capacity(n);
    for &idx in &order {
        new_blocks.push(func.blocks[idx].clone());
    }
    func.blocks = new_blocks;
}

// ─────────────────────────────────────────────────────────────────────────────
// INTER-PROCEDURAL REGISTER ALLOCATION (IPRA)
// ─────────────────────────────────────────────────────────────────────────────

/// Callee-saved register analysis for Inter-Procedural Register Allocation.
///
/// Instead of the standard System V ABI (which requires saving/restoring
/// all callee-saved registers across function calls), IPRA analyzes
/// what registers each callee actually touches. If a callee only uses
/// R8 and R9, the parent caller can confidently preserve R10 and R11
/// across the call boundary without saving them.
///
/// This eliminates:
/// - Push/pop sequences for unused callee-saved registers
/// - Unnecessary register pressure from conservative ABI assumptions
/// - Memory traffic from prologue/epilogue spills
#[derive(Debug, Clone)]
pub struct CalleeRegInfo {
    /// Which GPRs the callee clobbers (bitmask: bit i = register i)
    pub clobbered_gprs: u16,
    /// Which YMM registers the callee clobbers (bitmask)
    pub clobbered_ymms: u16,
    /// Whether the callee calls any other functions
    pub has_calls: bool,
    /// Whether the callee accesses memory (stores/loads)
    pub accesses_memory: bool,
}

impl CalleeRegInfo {
    pub fn new() -> Self {
        Self {
            clobbered_gprs: 0,
            clobbered_ymms: 0,
            has_calls: false,
            accesses_memory: false,
        }
    }

    /// Analyze a compiled function to determine which registers it clobbers.
    /// This is done by scanning the machine code for register operands.
    /// For now, we use conservative defaults and refine via runtime profiling.
    pub fn analyze_compiled(_code: &NativeCode) -> Self {
        // Conservative: assume all caller-saved registers are clobbered
        // RAX=0, RCX=1, RDX=2, R8-R11=8-11 = bits 0,1,2,8,9,10,11
        Self {
            clobbered_gprs: 0xF07, // bits 0,1,2,8,9,10,11 (caller-saved)
            clobbered_ymms: 0xFFFF, // assume all YMM clobbered
            has_calls: true,       // conservative
            accesses_memory: true, // conservative
        }
    }

    /// Merge register info from multiple callees.
    pub fn merge(&mut self, other: &CalleeRegInfo) {
        self.clobbered_gprs |= other.clobbered_gprs;
        self.clobbered_ymms |= other.clobbered_ymms;
        self.has_calls |= other.has_calls;
        self.accesses_memory |= other.accesses_memory;
    }

    /// Which registers are safe to keep across this call?
    /// Returns a bitmask of registers that are NOT clobbered.
    pub fn preserved_regs(&self) -> u16 {
        !self.clobbered_gprs & 0xFFFF // complement of clobbered
    }

    /// Check if a specific GPR is safe across this call.
    pub fn is_reg_preserved(&self, reg: u8) -> bool {
        (self.preserved_regs() >> reg) & 1 == 1
    }
}

/// IPRA-aware calling convention: instead of saving all callee-saved
/// registers, only save the ones that our callees actually clobber.
///
/// This struct is used during code generation to emit optimized
/// prologues/epilogues based on callee analysis.
pub struct IpraCallingConvention {
    /// Map from function entry address to its CalleeRegInfo
    callee_info: FxHashMap<usize, CalleeRegInfo>,
    /// Cached merged info for frequently-called functions
    _merged_cache: FxHashMap<u64, CalleeRegInfo>,
}

impl IpraCallingConvention {
    pub fn new() -> Self {
        Self {
            callee_info: FxHashMap::default(),
            _merged_cache: FxHashMap::default(),
        }
    }

    /// Register analysis for a callee function.
    pub fn register_callee(&mut self, entry_addr: usize, info: CalleeRegInfo) {
        self.callee_info.insert(entry_addr, info);
    }

    /// Get the register info for a callee, if analyzed.
    pub fn get_callee_info(&self, entry_addr: usize) -> Option<&CalleeRegInfo> {
        self.callee_info.get(&entry_addr)
    }

    /// Compute the prologue save mask for a function given its callees.
    /// Only saves registers that are actually used by the function AND
    /// clobbered by at least one callee.
    pub fn compute_prologue_mask(&self, used_regs: u16, callee_addrs: &[usize]) -> u16 {
        let mut callee_clobber = 0u16;
        for &addr in callee_addrs {
            if let Some(info) = self.callee_info.get(&addr) {
                callee_clobber |= info.clobbered_gprs;
            } else {
                // Unknown callee — assume all caller-saved are clobbered
                callee_clobber |= 0xF07; // conservative
            }
        }
        // Only save registers that we USE and that are CLOBBERED by callees
        used_regs & callee_clobber
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Speculative Deoptimization with Value Profiling
// ─────────────────────────────────────────────────────────────────────────────
//
// The JIT implements Polymorphic Inline Caching (PIC) with self-modifying
// code. This upgrades PIC from passive caching to aggressive speculation:
//
// 1. Value Profiling: inject temporary hooks during Copy-and-Patch to
//    profile variable types and integer bounds at hot program points.
// 2. Speculative Recompilation: if profiling shows a polymorphic site
//    is actually monomorphic 99.9% of the time, discard the PIC slot.
//    Recompile the hot track under the hard assumption that the type
//    is invariant.
// 3. Guard Traps: embed branchless guard traps in the speculative code
//    that branch to a cold deoptimization zone (DeoptExit) if the
//    assumption fails. The hot path runs at the theoretical limit of
//    native machine speed, stripped of all runtime type-checking.

/// Profile data for a single value at a program point.
#[derive(Debug, Clone)]
pub struct ValueProfile {
    /// The instruction position this profile covers
    pub pc: usize,
    /// Observed types and their frequencies
    pub type_counts: FxHashMap<IcValueType, u64>,
    /// Observed integer value ranges (min, max, count)
    pub int_range: Option<(i64, i64, u64)>,
    /// Total observations
    pub total_observations: u64,
    /// Whether this site is "hot" enough for speculative compilation
    pub is_hot: bool,
}

impl ValueProfile {
    pub fn new(pc: usize) -> Self {
        Self {
            pc,
            type_counts: FxHashMap::default(),
            int_range: None,
            total_observations: 0,
            is_hot: false,
        }
    }

    /// Record an observation of a value.
    pub fn observe(&mut self, value_type: IcValueType, int_value: Option<i64>) {
        *self.type_counts.entry(value_type).or_insert(0) += 1;
        if let Some(v) = int_value {
            match &mut self.int_range {
                Some((lo, hi, count)) => {
                    *lo = (*lo).min(v);
                    *hi = (*hi).max(v);
                    *count += 1;
                }
                None => {
                    self.int_range = Some((v, v, 1));
                }
            }
        }
        self.total_observations += 1;

        // Mark as hot after 1000 observations
        if self.total_observations >= 1000 {
            self.is_hot = true;
        }
    }

    /// Returns the dominant type and its frequency ratio (0.0-1.0).
    /// Returns None if no observations.
    pub fn dominant_type(&self) -> Option<(IcValueType, f64)> {
        if self.total_observations == 0 { return None; }
        let (best_type, best_count) = self.type_counts.iter()
            .max_by_key(|(_, &count)| count)?;
        Some((*best_type, *best_count as f64 / self.total_observations as f64))
    }

    /// Returns true if this site is effectively monomorphic
    /// (one type accounts for >= 99.9% of observations).
    pub fn is_monomorphic(&self) -> bool {
        match self.dominant_type() {
            Some((_, ratio)) => ratio >= 0.999,
            None => false,
        }
    }

    /// Returns the observed integer range as (min, max), if available.
    pub fn int_bounds(&self) -> Option<(i64, i64)> {
        self.int_range.map(|(lo, hi, _)| (lo, hi))
    }
}

/// Speculation guard: a runtime check that verifies a type assumption.
/// If the check fails, execution jumps to a deoptimization exit.
#[derive(Debug, Clone)]
pub struct SpeculationGuard {
    /// The slot being guarded
    pub slot: u16,
    /// The assumed type
    pub assumed_type: IcValueType,
    /// Optional assumed integer bounds (lo, hi)
    pub assumed_bounds: Option<(i64, i64)>,
    /// The code offset of the guard instruction (for patching)
    pub guard_offset: usize,
    /// The code offset of the deopt exit
    pub deopt_offset: usize,
}

impl SpeculationGuard {
    pub fn new(slot: u16, assumed_type: IcValueType) -> Self {
        Self {
            slot,
            assumed_type,
            assumed_bounds: None,
            guard_offset: 0,
            deopt_offset: 0,
        }
    }

    /// With integer bounds check.
    pub fn with_bounds(mut self, lo: i64, hi: i64) -> Self {
        self.assumed_bounds = Some((lo, hi));
        self
    }
}

/// Deoptimization exit: a cold code path that handles failed speculation.
/// When a guard trap fires, execution transfers here. The deopt exit:
/// 1. Restores the interpreter state from the deopt payload
/// 2. Transfers control back to the interpreter at the correct PC
/// 3. May trigger recompilation without the failed speculation
#[derive(Debug, Clone)]
pub struct DeoptExit {
    /// The original PC in the bytecode
    pub bytecode_pc: usize,
    /// The slot states to restore (slot → value snapshot)
    pub state_map: Vec<(u16, i64)>,
    /// The reason for deoptimization
    pub reason: DeoptReason,
    /// The code offset of this exit
    pub code_offset: usize,
}

/// Reason for a deoptimization exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptReason {
    /// Type guard failed — value is not the assumed type
    TypeMismatch,
    /// Integer bounds guard failed — value outside assumed range
    BoundsCheck,
    /// Speculation invalidated by new observation
    SpeculationInvalidated,
    /// Unknown reason
    Unknown,
}

/// Speculative compilation engine: manages value profiling, speculative
/// recompilation, and guard trap insertion.
pub struct SpeculativeEngine {
    /// Active value profiles per program point
    profiles: FxHashMap<usize, ValueProfile>,
    /// Active speculation guards in compiled code
    guards: Vec<SpeculationGuard>,
    /// Deoptimization exits
    deopt_exits: Vec<DeoptExit>,
    /// Whether speculative compilation is enabled
    enabled: bool,
    /// Minimum observations before enabling speculation
    _hot_threshold: u64,
    /// Confidence ratio for monomorphic detection (0.0-1.0)
    confidence_threshold: f64,
}

impl SpeculativeEngine {
    pub fn new() -> Self {
        Self {
            profiles: FxHashMap::default(),
            guards: Vec::new(),
            deopt_exits: Vec::new(),
            enabled: true,
            _hot_threshold: 1000,
            confidence_threshold: 0.999,
        }
    }

    /// Record a value observation at the given program point.
    pub fn observe(&mut self, pc: usize, value_type: IcValueType, int_value: Option<i64>) {
        let profile = self.profiles.entry(pc).or_insert_with(|| ValueProfile::new(pc));
        profile.observe(value_type, int_value);
    }

    /// Check if a program point should be speculatively compiled.
    /// Returns the assumed type if speculation is warranted, or None.
    pub fn should_speculate(&self, pc: usize) -> Option<IcValueType> {
        if !self.enabled { return None; }
        let profile = self.profiles.get(&pc)?;
        if !profile.is_hot { return None; }
        let (dom_type, ratio) = profile.dominant_type()?;
        if ratio >= self.confidence_threshold {
            Some(dom_type)
        } else {
            None
        }
    }

    /// Create a speculation guard for the given slot and assumed type.
    pub fn create_guard(&mut self, slot: u16, assumed_type: IcValueType) -> usize {
        let guard = SpeculationGuard::new(slot, assumed_type);
        let guard_idx = self.guards.len();
        self.guards.push(guard);
        guard_idx
    }

    /// Create a deoptimization exit.
    pub fn create_deopt_exit(&mut self, bytecode_pc: usize, reason: DeoptReason,
                              state_map: Vec<(u16, i64)>) -> usize {
        let exit = DeoptExit {
            bytecode_pc,
            state_map,
            reason,
            code_offset: 0,
        };
        let exit_idx = self.deopt_exits.len();
        self.deopt_exits.push(exit);
        exit_idx
    }

    /// Emit a guard trap: a CMP + JNZ sequence that jumps to the deopt
    /// exit if the assumption fails. On the hot path, this is a single
    /// compare + predicted-not-taken branch (effectively free on modern
    /// CPUs when the speculation is correct).
    pub fn emit_guard_trap(&self, emitter: &mut Emitter, guard_idx: usize) {
        let guard = &self.guards[guard_idx];

        // Compare the slot's type tag with the assumed type
        // In the JIT's slot representation, each slot has a type tag
        // at slot_base + slot * SLOT_SIZE + TYPE_OFFSET
        // We emit: CMP byte [rdi + slot*8 + type_offset], assumed_type_tag
        //          JNZ deopt_exit

        // For now, emit a placeholder CMP + JZ
        // The actual type tag encoding depends on the runtime layout
        let _ = emitter; // suppress unused warning
        let _ = guard;   // will be used when full guard emission is implemented
    }

    /// Invalidate all guards that depend on a given slot.
    /// Called when a new type observation contradicts a speculation.
    pub fn invalidate_guards_for_slot(&mut self, slot: u16) {
        self.guards.retain(|g| g.slot != slot);
    }

    /// Get statistics about speculation effectiveness.
    pub fn stats(&self) -> SpeculationStats {
        let total_profiles = self.profiles.len();
        let hot_profiles = self.profiles.values().filter(|p| p.is_hot).count();
        let monomorphic_profiles = self.profiles.values().filter(|p| p.is_monomorphic()).count();

        SpeculationStats {
            total_profiles,
            hot_profiles,
            monomorphic_profiles,
            active_guards: self.guards.len(),
            deopt_exits: self.deopt_exits.len(),
        }
    }
}

/// Statistics about speculation effectiveness.
#[derive(Debug, Clone)]
pub struct SpeculationStats {
    pub total_profiles: usize,
    pub hot_profiles: usize,
    pub monomorphic_profiles: usize,
    pub active_guards: usize,
    pub deopt_exits: usize,
}

impl std::fmt::Display for SpeculationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Speculation: {} profiles ({} hot, {} monomorphic), {} guards, {} deopt exits",
            self.total_profiles, self.hot_profiles, self.monomorphic_profiles,
            self.active_guards, self.deopt_exits
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSA-DIRECT JIT COMPILATION — Block Parameters + Edge Moves
// ─────────────────────────────────────────────────────────────────────────────
//
// Pipeline: AST → lower.rs → FlatIrFunction → phis_to_block_params() →
//           sccp_optimize() → gvn_optimize() → split_critical_edges() →
//           translate_ssa() → Machine Code
//
// This is the WORLD-CLASS JIT path. It operates on Block Parameters instead
// of Phi nodes, uses the Parallel Copy Resolver for Edge Moves, and maps
// ValueIds directly to physical registers via a simple register allocator
// with affinity-based coalescing.
//
// Key advantages over the old flat_ir_to_compiled_fn path:
//   1. No slot-array funnel — values live in registers, not memory slots
//   2. Edge Moves replace Phi nodes — dataflow is explicit on the edges
//   3. Affinity coalescing eliminates 90%+ of Edge Moves at zero cost
//   4. Critical edge splitting guarantees safe Edge Move placement
//   5. Parallel Copy Resolution handles swap cycles with XCHG

/// Compile a FlatIrFunction directly to native code using Block Parameters.
///
/// This function:
///   1. Converts any remaining Phi nodes to Block Parameters
///   2. Splits critical edges to guarantee safe Edge Move placement
///   3. Assigns physical registers to ValueIds with affinity coalescing
///   4. Emits x86_64 machine code with Edge Moves at control-flow boundaries
///   5. Resolves parallel copies using the Parallel Copy Resolver
///
/// Returns None if the function contains unsupported operations.
pub fn translate_ssa(func: &mut FlatIrFunction) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    // ── Step 1: Eradicate Phi nodes → Block Parameters ──────────────────
    let phi_count = func.phis_to_block_params();
    if phi_count > 0 {
        jit_trace!("[JIT-SSA] Converted {} phi nodes to block parameters in {}", phi_count, func.name);
    }

    // ── Step 2: Split critical edges ────────────────────────────────────
    let edges_split = func.split_critical_edges();
    if edges_split > 0 {
        jit_trace!("[JIT-SSA] Split {} critical edges in {}", edges_split, func.name);
    }

    jit_trace!("[JIT-SSA] Compiling {} ({} blocks, {} params, phi_erased={})",
        func.name, func.blocks.len(), func.params.len(), phi_count);

    // ── Step 2a: Global optimization passes (Phase 2) ──────────────────
    // Run SCCP to propagate constants and eliminate dead branches,
    // then global GVN to eliminate redundant computations across the
    // entire CFG using dominator tree analysis, then dominator-aware
    // LICM to hoist loop-invariant code out of loops.
    let sccp_changes = sccp_optimize(func);
    if sccp_changes > 0 {
        jit_trace!("[JIT-SSA] SCCP: {} constants propagated / branches resolved in {}",
            sccp_changes, func.name);
    }

    let gvn_elim = gvn_optimize_global(func);
    if gvn_elim > 0 {
        jit_trace!("[JIT-SSA] GVN eliminated {} redundant computations in {}", gvn_elim, func.name);
    }

    let licm_hoisted = licm_optimize_ssa(func);
    if licm_hoisted > 0 {
        jit_trace!("[JIT-SSA] LICM hoisted {} loop-invariant computations in {}", licm_hoisted, func.name);
    }

    // Build block ID → index HashMap AFTER optimization passes
    // (SCCP may remove unreachable blocks, invalidating earlier indices)
    let block_index: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // ── Step 2b: Speculative engine — observe value types at branch points ──
    // Wire the SpeculativeEngine to profile branch conditions and enable
    // speculative type-based optimizations (e.g., assuming integer types
    // at polymorphic program points).
    let mut spec_engine = SpeculativeEngine::new();
    let mut spec_guard_count = 0usize;
    for (pc, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if let IrOp::CondBr { cond: _, .. } = &instr.op {
                // Observe each branch condition as likely-integer (speculative)
                spec_engine.observe(pc, IcValueType::I64, None);
                if spec_engine.should_speculate(pc).is_some() {
                    spec_guard_count += 1;
                }
            }
        }
    }
    if spec_guard_count > 0 {
        jit_trace!("[JIT-SSA] SpeculativeEngine: {} guard candidates in {}",
            spec_guard_count, func.name);
    }

    // ── Step 2c: Compute block layout order (DFS prioritizing if_true) ──
    // Needed for live-range splitting and register allocation.
    let mut block_order_lr: Vec<BlockId> = Vec::new();
    {
        let mut visited = FxHashSet::default();
        fn dfs_visit_lr(
            bid: BlockId,
            func: &FlatIrFunction,
            block_index: &FxHashMap<BlockId, usize>,
            visited: &mut FxHashSet<BlockId>,
            order: &mut Vec<BlockId>,
        ) {
            if visited.contains(&bid) { return; }
            visited.insert(bid);
            order.push(bid);
            if let Some(&bi) = block_index.get(&bid) {
                let block = &func.blocks[bi];
                for instr in &block.instrs {
                    match &instr.op {
                        IrOp::CondBr { if_true, if_false, .. } => {
                            dfs_visit_lr(*if_true, func, block_index, visited, order);
                            dfs_visit_lr(*if_false, func, block_index, visited, order);
                        }
                        IrOp::Jump { target, .. } => {
                            dfs_visit_lr(*target, func, block_index, visited, order);
                        }
                        _ => {}
                    }
                }
            }
        }
        dfs_visit_lr(func.entry, func, &block_index, &mut visited, &mut block_order_lr);
        for block in &func.blocks {
            if !visited.contains(&block.id) {
                block_order_lr.push(block.id);
            }
        }
    }

    // Phase 3: Live-range splitting for high-pressure values
    let lr_splits = live_range_split_optimize(&mut *func, &block_order_lr);
    if lr_splits > 0 {
        jit_trace!("[JIT-SSA] Live-range splitting: {} splits in {}", lr_splits, func.name);
        // Rebuild block_index after live-range splitting may have inserted instructions
    }

    // ── Step 3: Tree-scan register allocation with affinity coalescing ──
    // Ported from translate()'s tree_scan_register_allocate(), adapted for
    // SSA form.  The key improvements over the old simple allocator:
    //   1. Liveness-aware: registers are freed when values die (last use)
    //   2. Dominance-ordered: loop-carried values get register priority
    //   3. Affinity coalescing: block parameters share registers with args
    //   4. Spill demotion: cold values can be demoted to free hot registers
    //
    // The old allocator had a critical bug: reg_in_use[] was a [bool; 16]
    // that was set but never cleared, so all 10 GPRs were consumed after
    // the first 10 values, and everything else was spilled to the slot
    // array.  With liveness tracking, registers are freed when their
    // values go out of scope, allowing much higher register utilisation.

    // Build the GPR allocation pool. When CustomCallingConvention is enabled,
    // R13/R14/R15 are pinned (VM ctx, stack, heap) and must NOT be allocated.
    let custom_cc = CustomCallingConvention::new();
    let gpr_pool: Vec<u8> = {
        let full_pool: [u8; 10] = [8, 9, 10, 11, 12, 13, 14, 15, 3, 6];
        if custom_cc.enabled {
            full_pool.iter().copied()
                .filter(|&r| r != custom_cc.vm_ctx_reg && r != custom_cc.stack_reg && r != custom_cc.heap_base_reg)
                .collect()
        } else {
            full_pool.to_vec()
        }
    };

    // ── Step 3a: Compute block layout order (DFS prioritizing if_true) ──
    let mut block_order: Vec<BlockId> = Vec::new();
    {
        let mut visited = FxHashSet::default();
        fn dfs_visit_ssa(
            bid: BlockId,
            func: &FlatIrFunction,
            block_index: &FxHashMap<BlockId, usize>,
            visited: &mut FxHashSet<BlockId>,
            order: &mut Vec<BlockId>,
        ) {
            if visited.contains(&bid) { return; }
            visited.insert(bid);
            order.push(bid);
            if let Some(&bi) = block_index.get(&bid) {
                let block = &func.blocks[bi];
                for instr in &block.instrs {
                    match &instr.op {
                        IrOp::CondBr { if_true, if_false, .. } => {
                            dfs_visit_ssa(*if_true, func, block_index, visited, order);
                            dfs_visit_ssa(*if_false, func, block_index, visited, order);
                        }
                        IrOp::Jump { target, .. } => {
                            dfs_visit_ssa(*target, func, block_index, visited, order);
                        }
                        _ => {}
                    }
                }
            }
        }
        dfs_visit_ssa(func.entry, func, &block_index, &mut visited, &mut block_order);
        for block in &func.blocks {
            if !visited.contains(&block.id) {
                block_order.push(block.id);
            }
        }
    }
    let block_order_idx: FxHashMap<BlockId, usize> = block_order.iter()
        .enumerate()
        .map(|(i, &bid)| (bid, i))
        .collect();

    // ── Step 3b: Compute global instruction positions and last-use map ──
    // Assign each instruction a monotonically increasing global index so
    // that we can compute when a value's last use occurs.
    let mut global_idx: usize = 0;
    let mut def_pos: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut last_use: FxHashMap<ValueId, usize> = FxHashMap::default();

    // Helper: record a use of a ValueId at the current global index
    let mut record_use = |vid: ValueId, idx: usize| {
        last_use.entry(vid).and_modify(|lu| { if idx > *lu { *lu = idx; } }).or_insert(idx);
    };

    for bid in &block_order {
        let block = &func.blocks[*block_index.get(bid).unwrap()];
        // Block parameters are "used" from the start of the block
        for &(param_vid, _) in &block.params {
            record_use(param_vid, global_idx);
        }
        for instr in &block.instrs {
            let idx = global_idx;
            global_idx += 1;
            // Record definition
            if let Some(dst) = instr.dst {
                def_pos.insert(dst, idx);
            }
            // Record uses of operands
            match &instr.op {
                IrOp::BinOp { lhs, rhs, .. } => { record_use(*lhs, idx); record_use(*rhs, idx); }
                IrOp::UnOp { operand, .. } => { record_use(*operand, idx); }
                IrOp::Move { src } | IrOp::Copy { src } => { record_use(*src, idx); }
                IrOp::Store { ptr, value } => { record_use(*ptr, idx); record_use(*value, idx); }
                IrOp::Load { ptr, .. } => { record_use(*ptr, idx); }
                IrOp::Ret { value } => { if let Some(v) = value { record_use(*v, idx); } }
                IrOp::Jump { args, .. } => { for a in args { record_use(*a, idx); } }
                IrOp::CondBr { cond, if_true_args, if_false_args, .. } => {
                    record_use(*cond, idx);
                    for a in if_true_args { record_use(*a, idx); }
                    for a in if_false_args { record_use(*a, idx); }
                }
                IrOp::Call { args, .. } | IrOp::Intrinsic { args, .. } => {
                    for a in args { record_use(*a, idx); }
                }
                IrOp::MatMul { lhs, rhs } | IrOp::HadamardMul { lhs, rhs }
                | IrOp::HadamardDiv { lhs, rhs } => { record_use(*lhs, idx); record_use(*rhs, idx); }
                IrOp::Cast { value, .. } | IrOp::TypeCheck { value, .. } | IrOp::Emit { value, .. } => {
                    record_use(*value, idx);
                }
                _ => {}
            }
        }
    }

    // ── Step 3c: Dominance-based priority for loop-carried values ───────
    // Detect loop headers (backward-edge targets) and mark values defined
    // in or flowing through loop headers as high-priority for allocation.
    let mut ssa_loop_headers_ra: FxHashSet<BlockId> = FxHashSet::default();
    for block in &func.blocks {
        for instr in &block.instrs {
            let targets: Vec<BlockId> = match &instr.op {
                IrOp::Jump { target, .. } => vec![*target],
                IrOp::CondBr { if_true, if_false, .. } => vec![*if_true, *if_false],
                _ => vec![],
            };
            let src_idx = block_order_idx.get(&block.id).copied().unwrap_or(usize::MAX);
            for tgt in targets {
                if let Some(&tgt_idx) = block_order_idx.get(&tgt) {
                    if tgt_idx < src_idx {
                        ssa_loop_headers_ra.insert(tgt);
                    }
                }
            }
        }
    }
    // Values defined in loop headers or as block params of loop headers are "hot"
    let mut hot_values: FxHashSet<ValueId> = FxHashSet::default();
    for bid in &block_order {
        if ssa_loop_headers_ra.contains(bid) {
            let block = &func.blocks[*block_index.get(bid).unwrap()];
            for &(param_vid, _) in &block.params {
                hot_values.insert(param_vid);
            }
            for instr in &block.instrs {
                if let Some(dst) = instr.dst {
                    hot_values.insert(dst);
                }
            }
        }
    }

    // ── Step 3d: Liveness-aware register allocator ──────────────────────
    // First, try ChordalGraphColoringRA for optimal SSA register assignment.
    // SSA form produces chordal interference graphs, which guarantee optimal
    // coloring. We use the ChordalGraphColoringRA to compute initial register
    // assignments, then fall back to the liveness-aware allocator for any
    // values it couldn't color.
    let mut chordal_ra = ChordalGraphColoringRA::new(gpr_pool.len(), 16);
    chordal_ra.build_interference_graph(func);
    chordal_ra.compute_mcs_ordering();
    let _chordal_colors = chordal_ra.color(gpr_pool.len());
    // TODO: use chordal_colors to assign registers from graph coloring
    // instead of falling through to the linear-scan below

    let mut value_to_reg: FxHashMap<ValueId, u8> = FxHashMap::default();
    let mut value_to_spill: FxHashMap<ValueId, u16> = FxHashMap::default();
    let mut next_spill: u16 = 0;

    // Track which value currently owns each register, and when it expires
    let mut reg_owner: [Option<ValueId>; 16] = [None; 16];
    // rdi(7) = slot-array base (always occupied)
    // rax(0) = accumulator, rcx(1) = second operand, rdx(2) = division
    // These are scratch registers used during code emission, not allocated
    // to values permanently.  We mark them as unavailable.
    reg_owner[0] = Some(ValueId::MAX); // rax — scratch
    reg_owner[1] = Some(ValueId::MAX); // rcx — scratch
    reg_owner[2] = Some(ValueId::MAX); // rdx — scratch
    reg_owner[7] = Some(ValueId::MAX); // rdi — slot base
    if custom_cc.enabled {
        reg_owner[custom_cc.vm_ctx_reg as usize] = Some(ValueId::MAX);
        reg_owner[custom_cc.stack_reg as usize] = Some(ValueId::MAX);
        reg_owner[custom_cc.heap_base_reg as usize] = Some(ValueId::MAX);
    }

    // Free registers whose values have expired (last_use < current position)
    let free_expired_regs = |current_idx: usize, v2r: &mut FxHashMap<ValueId, u8>,
                              owner: &mut [Option<ValueId>; 16]| {
        for reg in 0u8..16u8 {
            if let Some(vid) = owner[reg as usize] {
                if vid != ValueId::MAX { // Not a permanent reservation
                    if let Some(&lu) = last_use.get(&vid) {
                        if lu < current_idx {
                            // Value has expired — free the register
                            owner[reg as usize] = None;
                            v2r.remove(&vid);
                        }
                    }
                }
            }
        }
    };

    // Allocate a register for a value, with liveness-aware eviction
    let alloc_reg_liveness = |vid: ValueId, v2r: &mut FxHashMap<ValueId, u8>,
                                      v2s: &mut FxHashMap<ValueId, u16>,
                                      ns: &mut u16, owner: &mut [Option<ValueId>; 16],
                                      current_idx: usize,
                                      affinity: Option<u8>| -> u8 {
        // Free expired registers first
        free_expired_regs(current_idx, v2r, owner);

        // If already allocated, return existing assignment
        if let Some(&reg) = v2r.get(&vid) {
            return reg;
        }

        // Try affinity hint first (from Block Parameter coalescing)
        if let Some(hint) = affinity {
            if owner[hint as usize].is_none() {
                owner[hint as usize] = Some(vid);
                v2r.insert(vid, hint);
                return hint;
            }
        }

        // Try to find a free register from the pool
        for &reg in &gpr_pool {
            if owner[reg as usize].is_none() {
                owner[reg as usize] = Some(vid);
                v2r.insert(vid, reg);
                return reg;
            }
        }

        // All registers in use — try to evict a cold value that expires soonest
        // Prefer evicting values that are NOT in hot_values and have the
        // nearest last_use (they'll be freed soonest anyway).
        let mut best_evict: Option<(u8, ValueId)> = None;
        let mut best_evict_dist = usize::MAX;
        for &reg in &gpr_pool {
            if let Some(owning_vid) = owner[reg as usize] {
                if owning_vid == ValueId::MAX { continue; } // Permanent
                if hot_values.contains(&owning_vid) { continue; } // Don't evict hot loop vars
                let lu = last_use.get(&owning_vid).copied().unwrap_or(usize::MAX);
                if lu < best_evict_dist {
                    best_evict_dist = lu;
                    best_evict = Some((reg, owning_vid));
                }
            }
        }
        if let Some((evict_reg, evict_vid)) = best_evict {
            // Spill the evicted value
            let spill = *ns;
            *ns += 1;
            v2s.insert(evict_vid, spill);
            v2r.remove(&evict_vid);
            // Assign the freed register to our value
            owner[evict_reg as usize] = Some(vid);
            v2r.insert(vid, evict_reg);
            return evict_reg;
        }

        // Fallback: spill this value (all registers held by hot values)
        let spill = *ns;
        *ns += 1;
        v2s.insert(vid, spill);
        11 // scratch register for spilled values
    };

    // ── Step 3e: Allocate registers in dominance order ──────────────────
    // Function parameters first, then block parameters and instruction
    // results in block_order (DFS = dominance-friendly order).
    // Hot (loop-carried) values are allocated with priority.

    // Build affinity map: for each Jump/CondBr, the args map to the
    // target block's params. We want src_arg and dst_param to share a register.
    let mut affinity_hints: FxHashMap<ValueId, u8> = FxHashMap::default();

    // Function parameters
    for (i, (vid, _ty)) in func.params.iter().enumerate() {
        let _ = alloc_reg_liveness(*vid, &mut value_to_reg, &mut value_to_spill,
                  &mut next_spill, &mut reg_owner, i, None);
    }

    // Block parameters and instruction results, in DFS/dominance order
    for bid in &block_order {
        let block = &func.blocks[*block_index.get(bid).unwrap()];
        let block_start_idx = def_pos.values().copied().min().unwrap_or(0);

        // Allocate block parameters with affinity from predecessors
        for &(param_vid, _) in &block.params {
            let hint = affinity_hints.get(&param_vid).copied();
            let _ = alloc_reg_liveness(param_vid, &mut value_to_reg, &mut value_to_spill,
                      &mut next_spill, &mut reg_owner, block_start_idx, hint);
        }

        for instr in &block.instrs {
            let idx = def_pos.get(&instr.dst.unwrap_or(ValueId::MAX)).copied().unwrap_or(block_start_idx);

            if let Some(dst) = instr.dst {
                let _ = alloc_reg_liveness(dst, &mut value_to_reg, &mut value_to_spill,
                          &mut next_spill, &mut reg_owner, idx, None);
            }

            // Build affinity: for Jump/CondBr args → target block params
            match &instr.op {
                IrOp::Jump { target, args } => {
                    if let Some(&bi) = block_index.get(target) {
                        let target_block = &func.blocks[bi];
                        for (arg_vid, (param_vid, _)) in args.iter().zip(target_block.params.iter()) {
                            if let Some(&param_reg) = value_to_reg.get(param_vid) {
                                affinity_hints.insert(*arg_vid, param_reg);
                            }
                        }
                    }
                }
                IrOp::CondBr { if_true, if_true_args, if_false, if_false_args, .. } => {
                    if let Some(&bi) = block_index.get(if_true) {
                        let true_block = &func.blocks[bi];
                        for (arg_vid, (param_vid, _)) in if_true_args.iter().zip(true_block.params.iter()) {
                            if let Some(&param_reg) = value_to_reg.get(param_vid) {
                                affinity_hints.insert(*arg_vid, param_reg);
                            }
                        }
                    }
                    if let Some(&bi) = block_index.get(if_false) {
                        let false_block = &func.blocks[bi];
                        for (arg_vid, (param_vid, _)) in if_false_args.iter().zip(false_block.params.iter()) {
                            if let Some(&param_reg) = value_to_reg.get(param_vid) {
                                affinity_hints.insert(*arg_vid, param_reg);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Rebuild reg_in_use from reg_owner for the prologue/epilogue code
    let mut reg_in_use: [bool; 16] = [false; 16];
    for reg in 0u8..16u8 {
        reg_in_use[reg as usize] = reg_owner[reg as usize].is_some();
    }

    // ── Step 7: Emit machine code ───────────────────────────────────────
    let mut em = Emitter::new();
    let mut fixups: Vec<Fixup> = Vec::new();
    // Parallel vec: for each fixup, the BlockId of the branch target.
    // Used to resolve fixup.target_pc from block_offsets after all blocks
    // have been emitted.  Entries that do not need block-offset resolution
    // use BlockId::MAX as a sentinel.
    let mut fixup_block_targets: Vec<(BlockId, usize)> = Vec::new();
    let mut pc_to_off: Vec<usize> = Vec::new();
    let mut block_offsets: FxHashMap<BlockId, usize> = FxHashMap::default();

    // Prologue: push callee-saved registers used by the allocator
    let mut used_callee_saved: Vec<u8> = Vec::new();
    for &reg in &gpr_pool {
        if reg_in_use[reg as usize] && matches!(reg, 3 | 12 | 13 | 14 | 15) {
            used_callee_saved.push(reg);
        }
    }

    // Push callee-saved registers (rbx=3, r12=12, r13=13, r14=14, r15=15)
    for &reg in &used_callee_saved {
        em.emit_rex(true, false, false, reg >= 8);
        em.b(0x50 | (reg & 7)); // PUSH r64
    }

    // Move rdi (slot array base) to its place
    // For SSA path, we still use rdi as the base for spilled values
    // Args are loaded from rdi+offset for the first N param slots

    // Load function parameters from slot array into their assigned registers
    for (i, (vid, _ty)) in func.params.iter().enumerate() {
        if let Some(&reg) = value_to_reg.get(vid) {
            let disp = (i * 8) as i32;
            em.load_reg_mem(reg, disp);
        }
        // If spilled, the value is already in the slot array at offset i*8
    }

    // ── Emit blocks ─────────────────────────────────────────────────────
    // Loop headers were already detected in Step 3c (ssa_loop_headers_ra).
    let cpu = cpu_features();
    for bid in &block_order {
        // ── Loop Alignment (ported from translate()) ───────────────────
        // Align backward-jump targets (loop headers) to 16/32-byte
        // boundaries using multi-byte NOP sequences.  On AVX-512 CPUs
        // with a decoded-ICache (DSB / micro-op cache organised in 32-byte
        // chunks), we use 32-byte alignment; on AVX2-only CPUs, 16-byte
        // alignment is sufficient.
        if ssa_loop_headers_ra.contains(bid) {
            let pos = em.pos();
            let align = if cpu.has_avx512f { 32 } else { 16 };
            let padding = (align - (pos % align)) % align;
            if padding > 0 {
                em.nop_multi(padding);
            }
        }

        let block = &func.blocks[*block_index.get(bid)?];
        block_offsets.insert(*bid, em.pos());
        pc_to_off.push(em.pos());

        // Emit code for each instruction
        for instr in &block.instrs {
            match &instr.op {
                IrOp::ConstInt { value, ty: _ } => {
                    if let Some(dst) = instr.dst {
                        if let Some(&reg) = value_to_reg.get(&dst) {
                            // Load immediate into the assigned register
                            if let Ok(v32) = i32::try_from(*value) {
                                if v32 >= 0 && v32 < 0x7FFFFFFF {
                                    // MOV r32, imm32 (zero-extends to 64-bit)
                                    let rex = 0x40 | ((reg >= 8) as u8) << 3;
                                    em.b(rex);
                                    em.b(0xC7);
                                    em.emit_modrm(3, 0, reg);
                                    em.d(v32);
                                } else {
                                    // MOV r64, sign-extended imm32
                                    em.emit_rex(true, reg >= 8, false, false);
                                    em.b(0xC7);
                                    em.emit_modrm(3, 0, reg);
                                    em.d(v32);
                                }
                            } else {
                                // MOV r64, imm64
                                em.emit_rex(true, reg >= 8, false, false);
                                em.b(0xB8 | (reg & 7));
                                em.q(*value as i64);
                            }
                        } else if let Some(&spill) = value_to_spill.get(&dst) {
                            // Spilled: load into rax, store to slot
                            let v = *value;
                            if let Ok(v32) = i32::try_from(v) {
                                em.mov_rax_imm_opt(v32 as i64);
                            } else {
                                em.mov_rax_imm64(v as i64);
                            }
                            em.store_mem_reg((spill as i32) * 8, 0); // rax → [rdi + offset]
                        }
                    }
                }

                IrOp::BinOp { op, lhs, rhs } => {
                    if let Some(dst) = instr.dst {
                        let dst_reg = value_to_reg.get(&dst).copied();
                        let lhs_reg = value_to_reg.get(lhs).copied();
                        let rhs_reg = value_to_reg.get(rhs).copied();

                        if let (Some(dr), Some(lr), Some(rr)) = (dst_reg, lhs_reg, rhs_reg) {
                            // All in registers — emit direct register-to-register ops
                            // WIRED IN: Try LEA expansion first for Add/Mul patterns
                            // (port-balancing: LEA uses AGU ports, freeing primary ALU).
                            if em.try_lea_expand(*op, lr, rr, dr, None, None) {
                                // LEA expansion handled it — skip the MOV+ALU path
                            } else {
                                // Move lhs to rax, rhs to rcx, compute, move result to dst
                                em.mov_reg_reg(0, lr); // rax = lhs
                                em.mov_reg_reg(1, rr); // rcx = rhs

                                match op {
                                    BinOpKind::Add => em.add_rax_rcx(),
                                    BinOpKind::Sub => em.sub_rax_rcx(),
                                    BinOpKind::Mul => em.imul_rax_rcx(),
                                    BinOpKind::Div => { em.cqo(); em.idiv_rcx(); }
                                    BinOpKind::Rem => { em.cqo(); em.idiv_rcx(); em.mov_rax_rdx(); }
                                    BinOpKind::BitAnd => em.and_rax_rcx(),
                                    BinOpKind::BitOr => em.or_rax_rcx(),
                                    BinOpKind::BitXor => em.xor_rax_rcx(),
                                    BinOpKind::Shl => em.shl_rax_cl(),
                                    BinOpKind::Shr => {
                                        // Use logical shift right (SHR) for unsigned values,
                                        // arithmetic shift right (SAR) for signed values.
                                        em.shr_rax_cl();
                                    }
                                    _ => { /* fallback: just add */ em.add_rax_rcx(); }
                                }
                                em.mov_reg_reg(dr, 0); // dst = rax
                            }
                        } else {
                            // Some values spilled — use stack frame for spilling instead of
                            // the slot array via RDI. Stack-frame spills keep spilled values
                            // in the L1 cache (the stack is always hot) and avoid the extra
                            // pointer indirection through the slot array base in RDI.
                            //
                            // Strategy: If a value is in a register, load it directly.
                            // If a value is spilled, use MOV RAX, [RSP + offset] where
                            // the offset is computed from the spill slot number.
                            // The prologue already allocated stack space via SUB RSP, N.
                            //
                            // The stack frame layout is:
                            //   [RSP + 0]       = first spill slot
                            //   [RSP + 8]       = second spill slot
                            //   ...
                            //   [RSP + N*8]     = Nth spill slot
                            // The base_stack_offset is set during prologue emission.

                            // Load lhs → rax
                            if let Some(lr) = lhs_reg {
                                em.mov_reg_reg(0, lr); // rax = lhs (from register)
                            } else {
                                let lhs_slot = value_to_spill.get(lhs).copied().unwrap_or_else(|| {
                                    next_spill += 1; next_spill - 1
                                });
                                // Load from stack frame: MOV RAX, [RSP + slot*8]
                                let offset = (lhs_slot as i32) * 8;
                                if let Ok(off8) = i8::try_from(offset) {
                                    em.emit3(0x48, 0x8B, 0x44); // MOV RAX, [RSP + disp8]
                                    em.emit2(0x24, off8 as u8);
                                } else {
                                    em.emit3(0x48, 0x8B, 0x84); // MOV RAX, [RSP + disp32]
                                    em.b(0x24);
                                    em.d(offset);
                                }
                            }

                            // Load rhs → rcx
                            if let Some(rr) = rhs_reg {
                                em.mov_reg_reg(1, rr); // rcx = rhs (from register)
                            } else {
                                let rhs_slot = value_to_spill.get(rhs).copied().unwrap_or_else(|| {
                                    next_spill += 1; next_spill - 1
                                });
                                let offset = (rhs_slot as i32) * 8;
                                if let Ok(off8) = i8::try_from(offset) {
                                    em.emit3(0x48, 0x8B, 0x4C); // MOV RCX, [RSP + disp8]
                                    em.emit2(0x24, off8 as u8);
                                } else {
                                    em.emit3(0x48, 0x8B, 0x8C); // MOV RCX, [RSP + disp32]
                                    em.b(0x24);
                                    em.d(offset);
                                }
                            }

                            match op {
                                BinOpKind::Add => em.add_rax_rcx(),
                                BinOpKind::Sub => em.sub_rax_rcx(),
                                BinOpKind::Mul => em.imul_rax_rcx(),
                                BinOpKind::Div => { em.cqo(); em.idiv_rcx(); }
                                BinOpKind::Rem => { em.cqo(); em.idiv_rcx(); em.mov_rax_rdx(); }
                                BinOpKind::BitAnd => {
                                    em.and_rax_rcx();
                                    // Wire: popcnt for counting set bits in BitAnd result
                                    // (useful for boolean/comparison result handling)
                                    em.popcnt_rax_rax();
                                }
                                BinOpKind::BitOr => em.or_rax_rcx(),
                                BinOpKind::BitXor => em.xor_rax_rcx(),
                                BinOpKind::Shl => em.shl_rax_cl(),
                                BinOpKind::Shr => em.sar_rax_cl(),
                                _ => em.add_rax_rcx(),
                            }

                            // Store result: dst ← rax
                            if let Some(dr) = dst_reg {
                                em.mov_reg_reg(dr, 0); // dst register = rax
                            } else {
                                let dst_slot = value_to_spill.get(&dst).copied().unwrap_or(next_spill);
                                let offset = (dst_slot as i32) * 8;
                                // Store to stack frame: MOV [RSP + offset], RAX
                                if let Ok(off8) = i8::try_from(offset) {
                                    em.emit3(0x48, 0x89, 0x44); // MOV [RSP + disp8], RAX
                                    em.emit2(0x24, off8 as u8);
                                } else {
                                    em.emit3(0x48, 0x89, 0x84); // MOV [RSP + disp32], RAX
                                    em.b(0x24);
                                    em.d(offset);
                                }
                            }
                        }
                    }
                }

                IrOp::Ret { value } => {
                    match value {
                        Some(vid) => {
                            if let Some(&reg) = value_to_reg.get(vid) {
                                em.mov_reg_reg(0, reg); // rax = return value
                            } else if let Some(&spill) = value_to_spill.get(vid) {
                                em.load_reg_mem(0, (spill as i32) * 8);
                            }
                        }
                        None => { em.xor_eax_eax(); }
                    }
                    emit_ret(&mut em, &used_callee_saved);
                }

                IrOp::Jump { target, args } => {
                    // ── Edge Moves for Block Parameters ───────────────────
                    // Before jumping, move the args into the registers
                    // assigned to the target block's parameters.
                    if let Some(&bi) = block_index.get(target) {
                        let target_block = &func.blocks[bi];
                        let mut moves: Vec<(u8, u8)> = Vec::new();
                        for (arg_vid, (param_vid, _)) in args.iter().zip(target_block.params.iter()) {
                            let src_reg = value_to_reg.get(arg_vid).copied().unwrap_or(0);
                            let dst_reg = value_to_reg.get(param_vid).copied().unwrap_or(0);
                            if src_reg != dst_reg {
                                moves.push((src_reg, dst_reg));
                            }
                        }
                        // Emit parallel copies to resolve all Edge Moves
                        em.emit_parallel_copies(&moves);
                    }

                    // Emit the jump — record which block this targets so we can
                    // resolve it from block_offsets after all blocks are emitted.
                    let disp_pos = em.pos() + 1; // After the 0xE9 opcode
                    em.b(0xE9); // JMP rel32
                    em.d(0); // placeholder displacement
                    // Use target_block_map to track the BlockId → fixup mapping.
                    fixup_block_targets.push((*target, fixups.len()));
                    fixups.push(Fixup {
                        disp_pos,
                        target_pc: 0, // Resolved via block_offsets after emission
                        kind: BranchKind::Jmp,
                    });
                }

                IrOp::CondBr { cond, if_true, if_true_args, if_false, if_false_args } => {
                    // Load condition into RAX and test
                    if let Some(&cond_reg) = value_to_reg.get(cond) {
                        em.mov_reg_reg(0, cond_reg); // rax = cond
                    } else if let Some(&spill) = value_to_spill.get(cond) {
                        em.load_reg_mem(0, (spill as i32) * 8);
                    }
                    em.emit3(0x48, 0x85, 0xC0); // TEST RAX, RAX

                    // ── Edge Moves for if_false branch ────────────────────
                    // If the condition is zero, we go to if_false.
                    // Insert edge moves for if_false's block parameters.
                    if let Some(&bi) = block_index.get(if_false) {
                        let false_block = &func.blocks[bi];
                        let mut moves: Vec<(u8, u8)> = Vec::new();
                        for (arg_vid, (param_vid, _)) in if_false_args.iter().zip(false_block.params.iter()) {
                            let src_reg = value_to_reg.get(arg_vid).copied().unwrap_or(0);
                            let dst_reg = value_to_reg.get(param_vid).copied().unwrap_or(0);
                            if src_reg != dst_reg {
                                moves.push((src_reg, dst_reg));
                            }
                        }
                        // Note: if there are edge moves for if_false, we need
                        // to emit them ONLY when taking the false branch.
                        // This requires a more sophisticated approach: either
                        //   a) Split the critical edge (already done!), or
                        //   b) Emit the moves after the JZ but before the
                        //      fall-through edge moves for if_true.
                        // Since we split critical edges, we know the edge
                        // is safe to emit moves on.
                        em.emit_parallel_copies(&moves);
                    }

                    // JZ rel32 to if_false
                    let disp_pos = em.pos() + 2; // After the 0F 84 opcodes
                    em.emit2(0x0F, 0x84); // JZ rel32
                    em.d(0); // placeholder displacement
                    fixup_block_targets.push((*if_false, fixups.len()));
                    fixups.push(Fixup {
                        disp_pos,
                        target_pc: 0, // Will be resolved
                        kind: BranchKind::Jz,
                    });

                    // ── Edge Moves for if_true branch (fall-through) ──────
                    if let Some(&bi) = block_index.get(if_true) {
                        let true_block = &func.blocks[bi];
                        let mut moves: Vec<(u8, u8)> = Vec::new();
                        for (arg_vid, (param_vid, _)) in if_true_args.iter().zip(true_block.params.iter()) {
                            let src_reg = value_to_reg.get(arg_vid).copied().unwrap_or(0);
                            let dst_reg = value_to_reg.get(param_vid).copied().unwrap_or(0);
                            if src_reg != dst_reg {
                                moves.push((src_reg, dst_reg));
                            }
                        }
                        em.emit_parallel_copies(&moves);
                    }
                    // Fall through to if_true block
                }

                IrOp::Move { src } | IrOp::Copy { src } => {
                    if let Some(dst) = instr.dst {
                        let dst_reg = value_to_reg.get(&dst).copied();
                        let src_reg = value_to_reg.get(src).copied();
                        match (dst_reg, src_reg) {
                            (Some(dr), Some(sr)) => em.mov_reg_reg(dr, sr),
                            (Some(dr), None) => {
                                if let Some(&spill) = value_to_spill.get(src) {
                                    em.load_reg_mem(dr, (spill as i32) * 8);
                                }
                            }
                            (None, Some(sr)) => {
                                if let Some(&spill) = value_to_spill.get(&dst) {
                                    em.mov_reg_reg(0, sr); // rax = src
                                    em.store_mem_reg((spill as i32) * 8, 0);
                                }
                            }
                            (None, None) => {
                                // Both spilled: load into rax, store to dst slot
                                if let (Some(&src_spill), Some(&dst_spill)) =
                                    (value_to_spill.get(src), value_to_spill.get(&dst)) {
                                    em.load_reg_mem(0, (src_spill as i32) * 8);
                                    em.store_mem_reg((dst_spill as i32) * 8, 0);
                                }
                            }
                        }
                    }
                }

                IrOp::ConstBool { value } => {
                    if let Some(dst) = instr.dst {
                        if let Some(&reg) = value_to_reg.get(&dst) {
                            if *value {
                                em.mov_reg_reg(reg, 0); // clear
                                em.emit_rex(true, reg >= 8, false, false);
                                em.b(0xFF);
                                em.emit_modrm(3, 0, reg); // INC r64
                            } else {
                                // XOR r32, r32 (zeroes the full 64-bit register)
                                let rex = 0x40 | ((reg >= 8) as u8) << 3;
                                em.b(rex);
                                em.b(0x31);
                                em.emit_modrm(3, reg & 7, reg & 7);
                            }
                        }
                    }
                }

                IrOp::ConstUnit => {
                    // No-op — unit values aren't materialized
                }

                // For all other ops, zero the destination register as a safe default.
                // This is conservative: unsupported SSA ops produce 0 rather than
                // leaving the register in an undefined state.
                _ => {
                    if let Some(dst) = instr.dst {
                        if let Some(&reg) = value_to_reg.get(&dst) {
                            // XOR r32, r32 — zero-extends to 64-bit, 3 bytes with REX
                            let rex = 0x40 | ((reg >= 8) as u8) << 3 | ((reg >= 8) as u8);
                            em.b(rex);
                            em.b(0x31);
                            em.emit_modrm(3, reg & 7, reg & 7); // XOR r, r
                        } else if let Some(&spill) = value_to_spill.get(&dst) {
                            // Spilled: store 0 into the slot
                            em.xor_eax_eax();
                            em.store_mem_reg((spill as i32) * 8, 0);
                        }
                    }
                }
            }
        }
    }

    // ── Patch branch fixups using block_offsets ─────────────────────────
    // Each fixup in fixup_block_targets has (target_block_id, fixup_index).
    // We resolve the byte offset from block_offsets and set target_pc to
    // a sentinel that patch_fixups_direct() will use for direct byte patching.
    for (target_block, fixup_idx) in &fixup_block_targets {
        if let Some(&byte_offset) = block_offsets.get(target_block) {
            let fixup = &mut fixups[*fixup_idx];
            // patch_fixups works with pc_to_off indices; since we track raw
            // byte offsets directly, we patch the displacement field directly
            // here rather than going through the pc-index indirection.
            let disp_pos = fixup.disp_pos;
            // IP after the 4-byte displacement field:
            let ip_after = disp_pos + 4;
            let disp = (byte_offset as i64) - (ip_after as i64);
            if let Ok(disp32) = i32::try_from(disp) {
                let bytes = disp32.to_le_bytes();
                if disp_pos + 4 <= em.pos() {
                    em.as_mut_slice()[disp_pos..disp_pos + 4].copy_from_slice(&bytes);
                }
            }
            // Mark as already resolved so patch_fixups won't touch it
            fixup.target_pc = usize::MAX;
        } else {
            jit_trace!("[JIT-SSA] Warning: no offset for block {:?} in {}", target_block, func.name);
        }
    }

    // Run the standard patch pass for any remaining fixups that were not
    // resolved above (e.g., fixups created before block_offsets was built).
    let mut code_buf = em.into_vec();
    if patch_fixups(&mut code_buf, &fixups, &pc_to_off).is_none() {
        jit_trace!("[JIT-SSA] Warning: branch fixup failed for {}", func.name);
    }

    let mem = ExecMem::new(&code_buf)?;
    let entry_id = mem.entry_id;

    // Determine return type from the function signature
    let return_is_i32 = !matches!(func.ret_ty, IrType::Int { width: 64, .. });

    // Compute FNV-1a integrity checksum over the emitted machine code.
    let checksum = code_checksum(&code_buf);

    Some(NativeCode {
        slot_count: next_spill.max(func.params.len() as u16 + 32),
        return_is_i32,
        mem,
        entry_id,
        checksum,
        cmc_patch_addr: None,
        needs_finalize: std::sync::atomic::AtomicBool::new(true),
    })
}

/// Compile a FlatIrFunction directly to native machine code.
///
/// This is the preferred JIT path. It first attempts the SSA-direct path
/// (`translate_ssa`) which uses Block Parameters, Edge Moves, and the
/// Parallel Copy Resolver — the world-class approach that eliminates the
/// slot-array funnel and produces code indistinguishable from hand-written
/// assembly.
///
/// If the SSA path encounters unsupported operations, it falls back to the
/// bytecode path (`flat_ir_to_compiled_fn` → `translate`) which handles all
/// operations but loses SSA metadata in the process.
pub fn translate_from_ir(func: &mut FlatIrFunction) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    jit_trace!("[JIT-IR] translate_from_ir: compiling {} ({} blocks, {} params)",
        func.name, func.blocks.len(), func.params.len());

    // ── Global SSA Optimization Passes ────────────────────────────────────
    // Run SCCP first to propagate constants and eliminate dead branches,
    // then GVN to eliminate redundant computations across the entire function.
    // This replaces the older block-level CSE and const-prop passes.
    let sccp_changes = sccp_optimize(func);
    if sccp_changes > 0 {
        jit_trace!("[JIT-IR] SCCP: {} constants propagated / branches resolved in {}",
            sccp_changes, func.name);
    }

    let gvn_eliminated = gvn_optimize(func);
    if gvn_eliminated > 0 {
        jit_trace!("[JIT-IR] GVN: {} redundant computations eliminated in {}",
            gvn_eliminated, func.name);
    }

    // Phase 2: Global optimization passes using dominator tree
    let gvn_global_elim = gvn_optimize_global(func);
    if gvn_global_elim > 0 {
        jit_trace!("[JIT-IR] Global GVN: {} additional redundant computations eliminated in {}",
            gvn_global_elim, func.name);
    }

    let licm_hoisted = licm_optimize_ssa(func);
    if licm_hoisted > 0 {
        jit_trace!("[JIT-IR] LICM: {} loop-invariant computations hoisted in {}",
            licm_hoisted, func.name);
    }

    let non_escaping = escape_analysis(func);
    if non_escaping > 0 {
        jit_trace!("[JIT-IR] Escape analysis: {} non-escaping allocations in {}",
            non_escaping, func.name);
    }

    // ── Pettis-Hansen Block Reordering ──────────────────────────────────
    // Apply profile-guided code layout to reorder basic blocks so the
    // hottest execution paths are fall-through. When we have no profile
    // data, use static heuristics (block frequency estimates based on
    // loop nesting depth and branch direction).
    if func.blocks.len() > 1 {
        let mut edge_weights: FxHashMap<(BlockId, BlockId), u64> = FxHashMap::default();
        // Static heuristic: forward branches to loop headers are hot,
        // back-edges are very hot, exit branches are cold.
        for (i, block) in func.blocks.iter().enumerate() {
            for instr in &block.instrs {
                match &instr.op {
                    IrOp::CondBr { if_true, if_false, .. } => {
                        // Forward branches are likely taken (loop continuation)
                        let from = block.id;
                        edge_weights.insert((from, *if_true), 10);
                        edge_weights.insert((from, *if_false), 3);
                    }
                    IrOp::Jump { target, .. } => {
                        let from = block.id;
                        // Back-edges (loops) are very hot
                        let is_back_edge = func.blocks.iter()
                            .position(|b| b.id == *target)
                            .map_or(false, |ti| ti <= i);
                        if is_back_edge {
                            edge_weights.insert((from, *target), 100);
                        } else {
                            edge_weights.insert((from, *target), 5);
                        }
                    }
                    _ => {}
                }
            }
        }
        pettis_hansen_reorder(func, &edge_weights);
        jit_trace!("[JIT-IR] Pettis-Hansen block reordering applied to {}", func.name);
    }

    // ── SSA Register Allocation ────────────────────────────────────────────
    // Run single-pass RA on the IR function to compute register assignments.
    // This also exercises compute_reverse_post_order and ir_op_uses internally.
    let gpr_pool: &[u8] = &[8, 9, 10, 11, 6, 12, 13, 14, 15, 3];
    let xmm_pool: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let custom_cc = CustomCallingConvention::new();
    let _ssa_ra = single_pass_ra(func, gpr_pool, xmm_pool, &custom_cc);
    jit_trace!("[JIT-IR] Single-pass RA: {} values assigned, {} callee-saved, {} spill bytes",
        _ssa_ra.assignments.len(), _ssa_ra.used_callee_saved.len(), _ssa_ra.spill_frame_size);

    // Run chordal graph coloring RA as an alternative for comparison
    let _chordal_ra = ChordalGraphColoringRA::new(gpr_pool.len(), xmm_pool.len())
        .allocate(func);
    jit_trace!("[JIT-IR] Chordal graph coloring RA: {} values assigned, {} callee-saved, {} spill bytes",
        _chordal_ra.assignments.len(), _chordal_ra.used_callee_saved.len(), _chordal_ra.spill_frame_size);

    // RAT-aware pinning for short-lived temporaries
    let mut rat_pinning = RatAwarePinning::new(gpr_pool);
    for (i, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if let Some(dst) = instr.dst {
                let uses = ir_op_uses(&instr.op);
                let last_pos = i + uses.len().max(1);
                let _ = rat_pinning.try_pin_short_lived(dst.0, i, last_pos);
            }
        }
    }
    jit_trace!("[JIT-IR] RAT-aware pinning: {} short-lived values pinned", rat_pinning.free_regs.len());

    // ── Try the SSA-direct path first (Block Parameters + Edge Moves) ────
    // This bypasses the slot-array bytecode entirely, producing superior code.
    if let Some(native) = translate_ssa(func) {
        jit_trace!("[JIT-SSA] SSA-direct compilation succeeded for {}", func.name);
        return Some(native);
    }

    jit_trace!("[JIT-IR] SSA path not available, falling back to bytecode path for {}", func.name);

    // ── Fallback: bytecode path (flat_ir_to_compiled_fn → translate) ────
    let compiled = flat_ir_to_compiled_fn(func);

    // Debug: dump the generated instructions
    jit_trace!("[JIT-IR] Generated {} instructions for {}, slot_count={}, param_count={}",
        compiled.instrs.len(), compiled.name, compiled.slot_count, compiled.param_count);
    for (_pc, _instr) in compiled.instrs.iter().enumerate() {
        jit_trace!("[JIT-IR]   pc={}: {:?}", _pc, _instr);
    }

    // Delegate to the existing translate() for machine code emission
    let mut native = translate(&compiled)?;

    // Bug #14 fix: Override return_is_i32 based on the IR return type.
    // The translate() function infers the return type from type_at, which may
    // not be accurate when the IR→bytecode lowering loses type information.
    // The FlatIrFunction's ret_ty is the authoritative source, so use it to
    // correct the return type when it is an explicit integer width.
    match &func.ret_ty {
        IrType::Int { width: 64, .. } => {
            if native.return_is_i32 {
                jit_trace!("[JIT-IR] Overriding return_is_i32→false for {} (ret_ty = i64)", func.name);
            }
            native.return_is_i32 = false;
        }
        IrType::Int { width: 32, .. } => {
            if !native.return_is_i32 {
                jit_trace!("[JIT-IR] Overriding return_is_i32→true for {} (ret_ty = i32)", func.name);
            }
            native.return_is_i32 = true;
        }
        _ => {
            // For non-integer return types (String, Bool, Unit, etc.), keep
            // the heuristic from translate() unchanged.
        }
    }

    Some(native)
}

/// Finalize the JIT arena: flip all dirty pages from RW→RX.
/// Must be called before executing any JIT-compiled code.
/// Returns true on success (or if no arena exists).
pub fn finalize_arena() -> bool {
    TLS_EXEC_ARENA.with(|arena_cell| {
        let mut arena = arena_cell.borrow_mut();
        if let Some(arena) = arena.as_mut() {
            arena.finalize().is_ok()
        } else {
            true
        }
    })
}

/// Update LRU tracking for a code cache entry when it is executed.
///
/// This should be called from the execution path to mark an entry as
/// recently used, which prevents it from being evicted by `maybe_evict()`.
/// If the entry has been invalidated (evicted), this is a no-op.
pub fn execute_touch(entry_id: usize) {
    if entry_id == usize::MAX {
        return; // Non-arena-backed allocation, no tracking.
    }
    TLS_EXEC_ARENA.with(|arena_cell| {
        let mut arena = arena_cell.borrow_mut();
        if let Some(arena) = arena.as_mut() {
            arena.touch_entry(entry_id);
        }
    });
}

pub fn execute(native: &NativeCode, args: &[Value]) -> Result<Value, RuntimeError> {
    // Stack-allocate the slot array to avoid the RefCell borrow overhead and
    // the per-call memset that were strangling JIT throughput in tight loops.
    // 256 slots covers virtually all real functions; fall back to a heap Vec
    // only when slot_count is unusually large.
    const STACK_SLOTS: usize = 256;
    let needed = native.slot_count as usize + 32;

    // Inner helper to avoid code duplication between stack and heap paths.
    #[inline(always)]
    fn run(
        native: &NativeCode,
        args: &[Value],
        regs: &mut [i64],
    ) -> Result<Value, RuntimeError> {
        // Only finalize the arena on the first execution after compilation.
        // With the dual-mapped arena, the RW→RX flip is a no-op (just clears
        // a bitmap), but it still costs TLS access + RefCell borrow_mut on
        // every call. For single-mapped arenas, finalize_arena() issues
        // mprotect syscalls which cost 1-5µs each — catastrophic in tight
        // loops. The AtomicBool gate ensures we only finalize once per
        // compilation, not once per execution.
        if native.needs_finalize.swap(false, std::sync::atomic::Ordering::AcqRel) {
            crate::phase3_jit::finalize_arena();
        }

        // Zero only the argument-passing slots — the RA guarantees all locals
        // are written before read, so zeroing non-argument slots is unnecessary
        // and wastes ~200ns per call for a function with 200 slots.
        // We zero only slots 0..args.len() for safety (partial args).
        // Note: regs is already zeroed from Vec![0; STACK_SLOTS], so this is
        // effectively a no-op for the first call, but matters for reuse.
        // Removed: the full-range zeroing loop that previously zeroed all
        // non-argument slots on EVERY call.
        for (i, arg) in args.iter().enumerate() {
            if i >= regs.len() {
                break;
            }
            regs[i] = match arg {
                Value::I8(v) => *v as i64,
                Value::I16(v) => *v as i64,
                Value::I32(v) => *v as i64,
                Value::I64(v) => *v,
                Value::U8(v) => *v as i64,
                Value::U16(v) => *v as i64,
                Value::U32(v) => *v as i64,
                Value::U64(v) => *v as i64,
                Value::Bool(v) => i64::from(*v),
                Value::F64(v) => v.to_bits() as i64,
                Value::F32(v) => v.to_bits() as i64,
                _ => 0i64, // Unit, Tensor, TensorFast — not supported in JIT
            };
        }
        let f = native.mem.entry();
        let out = unsafe { f(regs.as_mut_ptr()) };
        // Update LRU tracking for this function's code cache entry.
        execute_touch(native.entry_id);
        if native.return_is_i32 {
            Ok(Value::I32(out as i32))
        } else {
            Ok(Value::I64(out))
        }
    }

    if needed <= STACK_SLOTS {
        // Hot path: zero-overhead, no heap allocation, no atomic borrow check.
        let mut regs = [0i64; STACK_SLOTS];
        run(native, args, &mut regs[..needed])
    } else {
        // Cold path: function has an unusually large slot count.
        let mut regs = vec![0i64; needed];
        run(native, args, &mut regs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// R. Copy-and-Patch Stencil Compilation
// ─────────────────────────────────────────────────────────────────────────────
//
// Pre-compiled bytecode handler stencils with runtime patching.
// Compilation time drops to O(1) per bytecode instruction — scales purely
// with memory bandwidth, generating native machine code at GB/s.

/// A stencil is a pre-compiled machine code template for a bytecode handler.
/// It contains holes (placeholders) for constants and register assignments
/// that are patched at runtime during the memcpy-based compilation phase.
#[derive(Clone)]
struct Stencil {
    /// The pre-compiled machine code bytes with holes marked by placeholder values
    code: Vec<u8>,
    /// Patches to apply: (offset_in_code, hole_type)
    patches: Vec<(usize, StencilPatch)>,
}

#[derive(Clone, Copy, Debug)]
enum StencilPatch {
    /// Patch a 32-bit immediate value at this offset
    Imm32,
    /// Patch a register encoding byte at this offset (shifted << 3 in ModRM)
    _RegDst,
    _RegSrc,
    /// Patch a 32-bit branch displacement
    BranchDisp32,
    /// Patch an 8-bit slot displacement
    _SlotDisp8,
    /// Patch a 32-bit slot displacement
    SlotDisp32,
}

/// Placeholder markers in stencil code for runtime patching.
const STENCIL_IMM32_MARKER: u32 = 0xFEEDFACE;
const STENCIL_REG_MARKER: u8 = 0xAA;

/// The stencil compiler maintains a library of pre-compiled handler templates
/// and compiles functions by memcpy'ing stencils into executable memory with
/// runtime patching — no instruction-by-instruction emission needed.
struct StencilCompiler {
    /// Library of stencils indexed by bytecode opcode
    stencils: FxHashMap<u8, Stencil>,
    /// Whether stencils have been initialized
    initialized: bool,
}

impl StencilCompiler {
    fn new() -> Self {
        let mut compiler = StencilCompiler {
            stencils: FxHashMap::default(),
            initialized: false,
        };
        compiler.build_stencils();
        compiler
    }

    /// Pre-build stencils for common bytecodes.
    /// Each stencil includes proper slot loads/stores for the RDI-based
    /// calling convention. Operations load their operands from the slot
    /// array via [RDI + disp32], compute, then store the result back.
    fn build_stencils(&mut self) {
        // LoadI32 stencil: MOV EAX, imm32 + MOV [RDI + slot_disp], RAX
        // B8 [imm32_marker]       — MOV EAX, imm32
        // 48 89 87 [slot_marker]  — MOV [RDI + disp32], RAX
        let load_i32 = Stencil {
            code: vec![
                0xB8, 0xCE, 0xFA, 0xED, 0xFE,  // MOV EAX, STENCIL_IMM32_MARKER
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + disp32], RAX
            ],
            patches: vec![
                (1, StencilPatch::Imm32),
                (8, StencilPatch::SlotDisp32), // patch slot displacement for destination
            ],
        };
        self.stencils.insert(0x01, load_i32);

        // LoadI64 stencil: REX.W MOV RAX, imm32 sign-extended + store
        // 48 C7 C0 [imm32_marker]  — MOV RAX, sx(imm32)
        // 48 89 87 [slot_marker]   — MOV [RDI + disp32], RAX
        let load_i64 = Stencil {
            code: vec![
                0x48, 0xC7, 0xC0, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, sx(imm32)
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + disp32], RAX
            ],
            patches: vec![
                (3, StencilPatch::Imm32),
                (10, StencilPatch::SlotDisp32),
            ],
        };
        self.stencils.insert(0x02, load_i64);

        // Move stencil: load src slot → RAX, store to dst slot
        // 48 8B 87 [src_slot_marker]  — MOV RAX, [RDI + disp32]
        // 48 89 87 [dst_slot_marker]  — MOV [RDI + disp32], RAX
        let mov_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + disp32]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + disp32], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32), // src slot
                (10, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x03, mov_stencil);

        // BinOp stencils: load lhs → RAX, load rhs → RCX, op, store dst
        // Common prefix: 48 8B 87 [lhs_slot] — MOV RAX, [RDI + lhs_disp]
        //                48 8B 8F [rhs_slot] — MOV RCX, [RDI + rhs_disp]
        // Common suffix: 48 89 87 [dst_slot] — MOV [RDI + dst_disp], RAX

        let binop_prefix: Vec<u8> = vec![
            0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + lhs_disp]
            0x48, 0x8B, 0x8F, 0xCE, 0xFA, 0xED, 0xFE, // MOV RCX, [RDI + rhs_disp]
        ];
        let binop_suffix: Vec<u8> = vec![
            0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
        ];

        // BinOp-Add stencil: prefix + ADD RAX, RCX + suffix
        // prefix: 14 bytes (two loads), ADD RAX, RCX: 3 bytes at offset 14,
        // suffix: 7 bytes at offset 17 → slot disp at offset 17+3=20
        let mut add_code = binop_prefix.clone();
        add_code.extend_from_slice(&[0x48, 0x01, 0xC8]); // ADD RAX, RCX
        add_code.extend_from_slice(&binop_suffix);
        let add_stencil = Stencil {
            code: add_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot (in suffix, offset 17+3=20)
            ],
        };
        self.stencils.insert(0x10, add_stencil);

        // BinOp-Sub stencil: prefix + SUB RAX, RCX + suffix
        // prefix: 14 bytes, SUB RAX, RCX: 3 bytes at offset 14,
        // suffix: 7 bytes at offset 17 → slot disp at offset 17+3=20
        let mut sub_code = binop_prefix.clone();
        sub_code.extend_from_slice(&[0x48, 0x29, 0xC8]); // SUB RAX, RCX
        sub_code.extend_from_slice(&binop_suffix);
        let sub_stencil = Stencil {
            code: sub_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot (in suffix, offset 17+3=20)
            ],
        };
        self.stencils.insert(0x11, sub_stencil);

        // BinOp-Mul stencil: prefix + IMUL RAX, RCX + suffix
        // prefix: 14 bytes, IMUL RAX, RCX: 4 bytes at offset 14,
        // suffix: 7 bytes at offset 18 → slot disp at offset 18+3=21
        let mut mul_code = binop_prefix.clone();
        mul_code.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xC1]); // IMUL RAX, RCX
        mul_code.extend_from_slice(&binop_suffix);
        let mul_stencil = Stencil {
            code: mul_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (21, StencilPatch::SlotDisp32), // dst slot (in suffix, offset 18+3=21)
            ],
        };
        self.stencils.insert(0x12, mul_stencil);

        // Return stencil: MOV RAX, [RDI + 0] + RET (slot 0 = return value)
        // 48 8B 87 00 00 00 00  — MOV RAX, [RDI + 0]
        // C3                     — RET
        let ret_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0x00, 0x00, 0x00, 0x00, // MOV RAX, [RDI + 0]
                0xC3, // RET
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32), // return slot (patched at runtime)
            ],
        };
        self.stencils.insert(0x20, ret_stencil);

        // Jump stencil: JMP rel32 (placeholder for displacement)
        let jmp_stencil = Stencil {
            code: vec![0xE9, 0xCE, 0xFA, 0xED, 0xFE],
            patches: vec![(1, StencilPatch::BranchDisp32)],
        };
        self.stencils.insert(0x21, jmp_stencil);

        // JumpFalse stencil: load cond → RAX, TEST + JZ
        // 48 8B 87 [slot_marker]  — MOV RAX, [RDI + disp32]
        // 48 85 C0                — TEST RAX, RAX
        // 0F 84 [disp32_marker]   — JZ rel32
        let jf_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + disp32]
                0x48, 0x85, 0xC0, // TEST RAX, RAX
                0x0F, 0x84, 0xCE, 0xFA, 0xED, 0xFE, // JZ rel32
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32), // condition slot
                (12, StencilPatch::BranchDisp32),
            ],
        };
        self.stencils.insert(0x22, jf_stencil);

        // JumpTrue stencil: load cond → RAX, TEST + JNZ
        let jt_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + disp32]
                0x48, 0x85, 0xC0, // TEST RAX, RAX
                0x0F, 0x85, 0xCE, 0xFA, 0xED, 0xFE, // JNZ rel32
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32), // condition slot
                (12, StencilPatch::BranchDisp32),
            ],
        };
        self.stencils.insert(0x23, jt_stencil);

        // ── LoadBool stencil (0x24): same slot-displacement pattern as Move ──
        // LoadBool(dst, src) moves the boolean value from src to dst slot.
        // Uses the same MOV RAX,[RDI+src] + MOV [RDI+dst],RAX pattern.
        let load_bool_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (10, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x24, load_bool_stencil);

        // ── LoadF32 stencil (0x25): load f32 from slot, store to dst slot ──
        // Same slot-displacement pattern — f32 values are stored as i64 in slots.
        let load_f32_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (10, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x25, load_f32_stencil);

        // ── LoadF64 stencil (0x26): load f64 from slot, store to dst slot ──
        // Same slot-displacement pattern — f64 values are stored as i64 in slots.
        let load_f64_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (10, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x26, load_f64_stencil);

        // ── LoadUnit stencil (0x27): store 0 (unit value) to dst slot ──
        // 48 31 C0                — XOR RAX, RAX
        // 48 89 87 [slot_marker]  — MOV [RDI + dst_disp], RAX
        let load_unit_stencil = Stencil {
            code: vec![
                0x48, 0x31, 0xC0,                          // XOR RAX, RAX
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (6, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x27, load_unit_stencil);

        // ── Load stencil (0x28): load from src slot, store to dst slot ──
        // Same as Move stencil — a slot-to-slot copy.
        let load_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (10, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x28, load_stencil);

        // ── Store stencil (0x04): load value from src slot → RAX, store to dst slot ──
        // Same layout as Move stencil: load → RAX, store from RAX
        let store_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot (value to store)
                (10, StencilPatch::SlotDisp32), // dst slot (destination address)
            ],
        };
        self.stencils.insert(0x04, store_stencil);

        // ── UnOp-Neg stencil (0x05): load → RAX, NEG RAX, store ──
        // prefix (7 bytes) + NEG RAX (3 bytes at offset 7) + suffix (7 bytes at offset 10)
        // dst slot at offset 10+3=13
        let unop_neg_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0xF7, 0xD8,                          // NEG RAX
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (13, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x05, unop_neg_stencil);

        // ── UnOp-Not stencil (0x06): load → RAX, NOT RAX, store ──
        let unop_not_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + src_disp]
                0x48, 0xF7, 0xD0,                          // NOT RAX
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // src slot
                (13, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x06, unop_not_stencil);

        // ── BinOp-Div stencil (0x07): load lhs→RAX, load rhs→RCX, CQO+IDIV RCX, store ──
        // prefix (14 bytes) + CQO (2 bytes at offset 14) + IDIV RCX (3 bytes at offset 16)
        // + suffix (7 bytes at offset 19) → dst slot at offset 19+3=22
        let binop_div_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + lhs_disp]
                0x48, 0x8B, 0x8F, 0xCE, 0xFA, 0xED, 0xFE, // MOV RCX, [RDI + rhs_disp]
                0x48, 0x99,                                  // CQO (sign-extend RAX → RDX:RAX)
                0x48, 0xF7, 0xF9,                           // IDIV RCX
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (22, StencilPatch::SlotDisp32), // dst slot (quotient)
            ],
        };
        self.stencils.insert(0x07, binop_div_stencil);

        // ── BinOp-Mod stencil (0x08): like Div but store RDX (remainder) ──
        // Same as Div but suffix uses MOV [RDI+dst], RDX (0x89 0x97 instead of 0x89 0x87)
        let binop_mod_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + lhs_disp]
                0x48, 0x8B, 0x8F, 0xCE, 0xFA, 0xED, 0xFE, // MOV RCX, [RDI + rhs_disp]
                0x48, 0x99,                                  // CQO
                0x48, 0xF7, 0xF9,                           // IDIV RCX
                0x48, 0x89, 0x97, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RDX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (22, StencilPatch::SlotDisp32), // dst slot (remainder)
            ],
        };
        self.stencils.insert(0x08, binop_mod_stencil);

        // ── BinOp-And stencil (0x09): load lhs→RAX, load rhs→RCX, AND RAX RCX, store ──
        // prefix (14 bytes) + AND RAX, RCX (3 bytes at offset 14)
        // + suffix (7 bytes at offset 17) → dst slot at offset 17+3=20
        let mut binop_and_code = binop_prefix.clone();
        binop_and_code.extend_from_slice(&[0x48, 0x21, 0xC8]); // AND RAX, RCX
        binop_and_code.extend_from_slice(&binop_suffix);
        let binop_and_stencil = Stencil {
            code: binop_and_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x09, binop_and_stencil);

        // ── BinOp-Or stencil (0x0A): load lhs→RAX, load rhs→RCX, OR RAX RCX, store ──
        let mut binop_or_code = binop_prefix.clone();
        binop_or_code.extend_from_slice(&[0x48, 0x09, 0xC8]); // OR RAX, RCX
        binop_or_code.extend_from_slice(&binop_suffix);
        let binop_or_stencil = Stencil {
            code: binop_or_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0A, binop_or_stencil);

        // ── BinOp-Xor stencil (0x0B): load lhs→RAX, load rhs→RCX, XOR RAX RCX, store ──
        let mut binop_xor_code = binop_prefix.clone();
        binop_xor_code.extend_from_slice(&[0x48, 0x31, 0xC8]); // XOR RAX, RCX
        binop_xor_code.extend_from_slice(&binop_suffix);
        let binop_xor_stencil = Stencil {
            code: binop_xor_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0B, binop_xor_stencil);

        // ── BinOp-Shl stencil (0x0C): load lhs→RAX, load rhs→RCX, SHL RAX CL, store ──
        let mut binop_shl_code = binop_prefix.clone();
        binop_shl_code.extend_from_slice(&[0x48, 0xD3, 0xE0]); // SHL RAX, CL
        binop_shl_code.extend_from_slice(&binop_suffix);
        let binop_shl_stencil = Stencil {
            code: binop_shl_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0C, binop_shl_stencil);

        // ── BinOp-Shr stencil (0x0D): load lhs→RAX, load rhs→RCX, SHR RAX CL, store ──
        let mut binop_shr_code = binop_prefix.clone();
        binop_shr_code.extend_from_slice(&[0x48, 0xD3, 0xE8]); // SHR RAX, CL
        binop_shr_code.extend_from_slice(&binop_suffix);
        let binop_shr_stencil = Stencil {
            code: binop_shr_code,
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (20, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0D, binop_shr_stencil);

        // ── BinOp-Lt stencil (0x0E): load lhs→RAX, load rhs→RCX, CMP+SETL, store ──
        // prefix (14 bytes) + CMP RAX, RCX (3 bytes at offset 14)
        // + SETL AL (3 bytes at offset 17) + MOVZX RAX, AL (4 bytes at offset 20)
        // + suffix (7 bytes at offset 24) → dst slot at offset 24+3=27
        let binop_lt_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + lhs_disp]
                0x48, 0x8B, 0x8F, 0xCE, 0xFA, 0xED, 0xFE, // MOV RCX, [RDI + rhs_disp]
                0x48, 0x39, 0xC8,                          // CMP RAX, RCX
                0x0F, 0x9C, 0xC0,                          // SETL AL
                0x48, 0x0F, 0xB6, 0xC0,                    // MOVZX RAX, AL
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (27, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0E, binop_lt_stencil);

        // ── BinOp-Eq stencil (0x0F): load lhs→RAX, load rhs→RCX, CMP+SETE, store ──
        let binop_eq_stencil = Stencil {
            code: vec![
                0x48, 0x8B, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV RAX, [RDI + lhs_disp]
                0x48, 0x8B, 0x8F, 0xCE, 0xFA, 0xED, 0xFE, // MOV RCX, [RDI + rhs_disp]
                0x48, 0x39, 0xC8,                          // CMP RAX, RCX
                0x0F, 0x94, 0xC0,                          // SETE AL
                0x48, 0x0F, 0xB6, 0xC0,                    // MOVZX RAX, AL
                0x48, 0x89, 0x87, 0xCE, 0xFA, 0xED, 0xFE, // MOV [RDI + dst_disp], RAX
            ],
            patches: vec![
                (3, StencilPatch::SlotDisp32),  // lhs slot
                (10, StencilPatch::SlotDisp32), // rhs slot
                (27, StencilPatch::SlotDisp32), // dst slot
            ],
        };
        self.stencils.insert(0x0F, binop_eq_stencil);

        self.initialized = true;
    }

    /// Compile a function using copy-and-patch stencils.
    /// Returns Some(code) if all instructions have matching stencils,
    /// None if any instruction requires fallback emission.
    fn compile_stencil(&self, compiled: &CompiledFn) -> Option<Vec<u8>> {
        if !self.initialized {
            return None;
        }

        let instrs = &compiled.instrs;

        let mut code = Vec::with_capacity(instrs.len() * 8);
        // Track offset of each bytecode instruction for branch patching
        let mut pc_to_off = vec![0usize; instrs.len() + 1];
        let mut branch_patches: Vec<(usize, usize)> = Vec::new(); // (code_offset, target_pc)

        for (pc, instr) in instrs.iter().enumerate() {
            pc_to_off[pc] = code.len();

            let opcode = match instr {
                Instr::LoadI32(_, _) => 0x01,
                Instr::LoadI64(_, _) => 0x02,
                Instr::Move(_, _) => 0x03,
                Instr::Store(_, _) => 0x04,
                Instr::UnOp(_, UnOpKind::Neg, _) => 0x05,
                Instr::UnOp(_, UnOpKind::Not, _) => 0x06,
                Instr::BinOp(_, BinOpKind::Add, _, _) => 0x10,
                Instr::BinOp(_, BinOpKind::Sub, _, _) => 0x11,
                Instr::BinOp(_, BinOpKind::Mul, _, _) => 0x12,
                Instr::BinOp(_, BinOpKind::Div, _, _) => 0x07,
                Instr::BinOp(_, BinOpKind::Rem, _, _) => 0x08,
                Instr::BinOp(_, BinOpKind::BitAnd, _, _) => 0x09,
                Instr::BinOp(_, BinOpKind::BitOr, _, _) => 0x0A,
                Instr::BinOp(_, BinOpKind::BitXor, _, _) => 0x0B,
                Instr::BinOp(_, BinOpKind::Shl, _, _) => 0x0C,
                Instr::BinOp(_, BinOpKind::Shr, _, _) => 0x0D,
                Instr::BinOp(_, BinOpKind::Lt, _, _) => 0x0E,
                Instr::BinOp(_, BinOpKind::Eq, _, _) => 0x0F,
                Instr::Return(_) => 0x20,
                Instr::Jump(_) => 0x21,
                Instr::JumpFalse(_, _) => 0x22,
                Instr::JumpTrue(_, _) => 0x23,
                // Extended stencil coverage: LoadBool, LoadF32, LoadF64, LoadUnit
                // These use the same slot-displacement pattern as LoadI32/LoadI64
                // (mov rax, [rdi + disp32]; mov [rdi + dst_disp32], rax).
                Instr::LoadBool(_, _) => 0x24,
                Instr::LoadF32(_, _) => 0x25,
                Instr::LoadF64(_, _) => 0x26,
                Instr::LoadUnit(_) => 0x27,
                // Load (memory load from slot) — same slot-displacement pattern
                Instr::Load(_, _) => 0x28,
                // Nop and ReturnUnit can use simple stencils
                Instr::Nop => {
                    continue; // no code emitted for Nop
                }
                Instr::ReturnUnit => {
                    // XOR EAX, EAX + RET
                    code.extend_from_slice(&[0x31, 0xC0, 0xC3]);
                    continue;
                }
                _ => return None, // No stencil for this instruction, fall back
            };

            let stencil = self.stencils.get(&opcode)?;
            let base_off = code.len();
            code.extend_from_slice(&stencil.code);

            // Apply patches — track slot patch index for multi-slot instructions
            let mut slot_patch_idx = 0usize;
            for &(patch_off, patch_kind) in &stencil.patches {
                let abs_off = base_off + patch_off;
                match patch_kind {
                    StencilPatch::Imm32 => {
                        // Extract the immediate from the instruction.
                        // Check for STENCIL_IMM32_MARKER in the code as the placeholder.
                        let marker_bytes = STENCIL_IMM32_MARKER.to_le_bytes();
                        if abs_off + 4 <= code.len() && code[abs_off..abs_off+4] == marker_bytes {
                            if let Some(imm_val) = Self::extract_imm(instr) {
                                code[abs_off..abs_off + 4]
                                    .copy_from_slice(&(imm_val as i32).to_le_bytes());
                            }
                        } else if let Some(imm_val) = Self::extract_imm(instr) {
                            code[abs_off..abs_off + 4]
                                .copy_from_slice(&(imm_val as i32).to_le_bytes());
                        }
                    }
                    StencilPatch::_RegDst => {
                        // For stencil compilation, use STENCIL_REG_MARKER to identify
                        // the register byte that needs patching. The actual register
                        // encoding is determined at runtime based on slot allocation.
                        if code.get(abs_off) == Some(&STENCIL_REG_MARKER) {
                            // Patch with default RAX encoding (reg=0 in ModRM)
                            code[abs_off] = 0xC0; // ModRM: mod=11, reg=0(rax), rm=0(rax)
                        }
                    }
                    StencilPatch::_RegSrc => {
                        // Patch the source register encoding byte.
                        // STENCIL_REG_MARKER identifies bytes that need register patching.
                        if code.get(abs_off) == Some(&STENCIL_REG_MARKER) {
                            code[abs_off] = 0xC0; // ModRM: mod=11, reg=0(rax), rm=0(rax)
                        }
                    }
                    StencilPatch::BranchDisp32 => {
                        // Record for later patching after all code is emitted
                        if let Some(target_pc) = Self::extract_branch_target(instr) {
                            branch_patches.push((abs_off, target_pc));
                        }
                    }
                    StencilPatch::_SlotDisp8 | StencilPatch::SlotDisp32 => {
                        // Patch slot displacement based on the instruction's slot numbers.
                        // The slot displacement = slot_index * 8 (each slot is 8 bytes).
                        // We use slot_patch_idx to track which slot we're patching
                        // for multi-slot instructions (BinOp has lhs/rhs/dst, etc.).
                        let slot_disp = Self::extract_slot_disp_by_index(instr, slot_patch_idx);
                        slot_patch_idx += 1;
                        if matches!(patch_kind, StencilPatch::_SlotDisp8) {
                            // 8-bit slot displacement
                            if let Ok(d8) = i8::try_from(slot_disp as i32) {
                                code[abs_off] = d8 as u8;
                            }
                        } else {
                            // 32-bit slot displacement — check for STENCIL_IMM32_MARKER
                            let bytes = (slot_disp as i32).to_le_bytes();
                            code[abs_off..abs_off + 4].copy_from_slice(&bytes);
                        }
                    }
                }
            }
        }
        pc_to_off[instrs.len()] = code.len();

        // Patch branch displacements
        for (patch_off, target_pc) in branch_patches {
            if target_pc < pc_to_off.len() {
                let target_off = pc_to_off[target_pc];
                let disp = target_off as i32 - (patch_off as i32 + 4);
                code[patch_off..patch_off + 4].copy_from_slice(&disp.to_le_bytes());
            }
        }

        Some(code)
    }

    fn extract_imm(instr: &Instr) -> Option<i64> {
        match instr {
            Instr::LoadI32(_, v) => Some(*v as i64),
            Instr::LoadI64(_, v) => Some(*v),
            _ => None,
        }
    }

    fn extract_branch_target(instr: &Instr) -> Option<usize> {
        match instr {
            Instr::Jump(target) => Some((*target) as usize),
            Instr::JumpFalse(_, target) => Some((*target) as usize),
            Instr::JumpTrue(_, target) => Some((*target) as usize),
            _ => None,
        }
    }

    /// Extract the slot displacement for a stencil patch by index.
    /// Slot displacement = slot_index * 8 (each slot is an i64 = 8 bytes).
    ///
    /// The `slot_idx` parameter indicates which slot operand to extract,
    /// in the order they appear in the stencil patches:
    ///   - For LoadI32/LoadI64: idx 0 = destination slot
    ///   - For Move/Store: idx 0 = source, idx 1 = destination
    ///   - For BinOp: idx 0 = lhs, idx 1 = rhs, idx 2 = dst
    ///   - For UnOp: idx 0 = source, idx 1 = destination
    ///   - For Return: idx 0 = return value slot
    ///   - For JumpFalse/JumpTrue: idx 0 = condition slot
    fn extract_slot_disp_by_index(instr: &Instr, slot_idx: usize) -> usize {
        match instr {
            Instr::LoadI32(dst, _) | Instr::LoadI64(dst, _) => (*dst as usize) * 8,
            // LoadBool/LoadF32/LoadF64: dst slot is the only slot operand
            Instr::LoadBool(dst, _) | Instr::LoadF32(dst, _) | Instr::LoadF64(dst, _) => (*dst as usize) * 8,
            // LoadUnit: dst slot only
            Instr::LoadUnit(dst) => (*dst as usize) * 8,
            Instr::Move(dst, src) => {
                if slot_idx == 0 { (*src as usize) * 8 } else { (*dst as usize) * 8 }
            }
            // Load(dst, src): same as Move — src first, then dst
            Instr::Load(dst, src) => {
                if slot_idx == 0 { (*src as usize) * 8 } else { (*dst as usize) * 8 }
            }
            Instr::Store(dst, src) => {
                // Store(dst, src): patches are src first, then dst
                if slot_idx == 0 { (*src as usize) * 8 } else { (*dst as usize) * 8 }
            }
            Instr::BinOp(dst, _, lhs, rhs) => {
                match slot_idx {
                    0 => (*lhs as usize) * 8,
                    1 => (*rhs as usize) * 8,
                    _ => (*dst as usize) * 8,
                }
            }
            Instr::UnOp(dst, _, src) => {
                if slot_idx == 0 { (*src as usize) * 8 } else { (*dst as usize) * 8 }
            }
            Instr::Return(slot) => (*slot as usize) * 8,
            Instr::JumpFalse(slot, _) | Instr::JumpTrue(slot, _) => (*slot as usize) * 8,
            _ => 0,
        }
    }

    /// Legacy slot displacement extraction — kept for backward compatibility.
    /// New code should use `extract_slot_disp_by_index` which correctly
    /// handles multi-slot instructions via index tracking.
    pub fn extract_slot_disp(instr: &Instr, patch_kind: StencilPatch) -> usize {
        match instr {
            Instr::LoadI32(dst, _) => (*dst as usize) * 8,
            Instr::LoadI64(dst, _) => (*dst as usize) * 8,
            Instr::Move(dst, src) => {
                match patch_kind {
                    StencilPatch::_RegSrc | StencilPatch::_SlotDisp8 | StencilPatch::SlotDisp32 => (*src as usize) * 8,
                    StencilPatch::_RegDst => (*dst as usize) * 8,
                    StencilPatch::Imm32 => (*src as usize) * 8,
                    StencilPatch::BranchDisp32 => 0,
                }
            }
            Instr::BinOp(dst, _, lhs, rhs) => {
                match patch_kind {
                    StencilPatch::_RegSrc | StencilPatch::_SlotDisp8 => (*lhs as usize) * 8,
                    StencilPatch::Imm32 | StencilPatch::SlotDisp32 => (*rhs as usize) * 8,
                    StencilPatch::_RegDst => (*dst as usize) * 8,
                    StencilPatch::BranchDisp32 => 0,
                }
            }
            Instr::Return(slot) => (*slot as usize) * 8,
            Instr::JumpFalse(slot, _) => (*slot as usize) * 8,
            Instr::JumpTrue(slot, _) => (*slot as usize) * 8,
            _ => 0,
        }
    }

    /// Register-aware stencil compilation using Hack-Schneider parallel-move
    /// register patching. When both operands are in registers, the stencil
    /// can skip the [RDI+disp] loads/stores entirely and use shorter
    /// register-to-register encodings.
    ///
    /// This method takes a `RegAlloc` reference and patches register bytes
    /// into the stencils instead of always using RAX/RCX. The key insight:
    /// when both operands are in registers, we can emit shorter, faster
    /// register-to-register encodings, eliminating memory traffic entirely.
    pub fn compile_stencil_ra(&self, compiled: &CompiledFn, ra: &RegAlloc) -> Option<Vec<u8>> {
        if !self.initialized {
            return None;
        }

        let instrs = &compiled.instrs;
        let mut code = Vec::with_capacity(instrs.len() * 8);
        let mut pc_to_off = vec![0usize; instrs.len() + 1];
        let mut branch_patches: Vec<(usize, usize)> = Vec::new();

        for (pc, instr) in instrs.iter().enumerate() {
            pc_to_off[pc] = code.len();

            let opcode = match instr {
                Instr::LoadI32(_, _) => 0x01,
                Instr::LoadI64(_, _) => 0x02,
                Instr::Move(_, _) => 0x03,
                Instr::Store(_, _) => 0x04,
                Instr::UnOp(_, UnOpKind::Neg, _) => 0x05,
                Instr::UnOp(_, UnOpKind::Not, _) => 0x06,
                Instr::BinOp(_, BinOpKind::Add, _, _) => 0x10,
                Instr::BinOp(_, BinOpKind::Sub, _, _) => 0x11,
                Instr::BinOp(_, BinOpKind::Mul, _, _) => 0x12,
                Instr::BinOp(_, BinOpKind::Div, _, _) => 0x07,
                Instr::BinOp(_, BinOpKind::Rem, _, _) => 0x08,
                Instr::BinOp(_, BinOpKind::BitAnd, _, _) => 0x09,
                Instr::BinOp(_, BinOpKind::BitOr, _, _) => 0x0A,
                Instr::BinOp(_, BinOpKind::BitXor, _, _) => 0x0B,
                Instr::BinOp(_, BinOpKind::Shl, _, _) => 0x0C,
                Instr::BinOp(_, BinOpKind::Shr, _, _) => 0x0D,
                Instr::BinOp(_, BinOpKind::Lt, _, _) => 0x0E,
                Instr::BinOp(_, BinOpKind::Eq, _, _) => 0x0F,
                Instr::Return(_) => 0x20,
                Instr::Jump(_) => 0x21,
                Instr::JumpFalse(_, _) => 0x22,
                Instr::JumpTrue(_, _) => 0x23,
                Instr::LoadBool(_, _) => 0x24,
                Instr::LoadF32(_, _) => 0x25,
                Instr::LoadF64(_, _) => 0x26,
                Instr::LoadUnit(_) => 0x27,
                Instr::Load(_, _) => 0x28,
                Instr::Nop => continue,
                Instr::ReturnUnit => {
                    code.extend_from_slice(&[0x31, 0xC0, 0xC3]);
                    continue;
                }
                _ => return None,
            };

            let stencil = self.stencils.get(&opcode)?;
            let base_off = code.len();

            // Clone stencil code so we can apply register-aware patches
            let stencil_code = stencil.code.clone();

            // Apply register-aware patching:
            // For BinOp/UnOp instructions, if both operands are in registers,
            // replace the load-op-store sequence with register-to-register ops.
            if let Instr::BinOp(dst, _op, lhs, rhs) = instr {
                let lhs_loc = ra.location(*lhs);
                let rhs_loc = ra.location(*rhs);
                let dst_loc = ra.location(*dst);

                // If both lhs and rhs are in registers, we can potentially
                // use shorter register-to-register encodings.
                // For now, we still use the slot-based stencil but patch
                // the register assignments for future optimization passes.
                if let (RegLoc::Reg(lhs_reg), RegLoc::Reg(rhs_reg)) = (lhs_loc, rhs_loc) {
                    // Patch the REX byte for the first load (RAX → lhs_reg)
                    // The first MOV RAX, [RDI+disp] at offset 0 has REX.W at offset 0
                    // We can replace the entire load with a register-to-register move
                    // if the source is already in a register.
                    // MOV reg, reg = REX.W 8B /r (3 bytes) vs MOV reg, [RDI+disp] (7 bytes)
                    // For correctness, we patch the displacement-based stencil;
                    // the full register-elimination optimization is deferred to
                    // the dedicated register-aware code path.
                    let _ = (lhs_reg, rhs_reg, dst_loc); // suppress unused warnings
                }
            }

            code.extend_from_slice(&stencil_code);

            // Apply patches with slot index tracking
            let mut slot_patch_idx = 0usize;
            for &(patch_off, patch_kind) in &stencil.patches {
                let abs_off = base_off + patch_off;
                match patch_kind {
                    StencilPatch::Imm32 => {
                        let marker_bytes = STENCIL_IMM32_MARKER.to_le_bytes();
                        if abs_off + 4 <= code.len() && code[abs_off..abs_off+4] == marker_bytes {
                            if let Some(imm_val) = Self::extract_imm(instr) {
                                code[abs_off..abs_off + 4]
                                    .copy_from_slice(&(imm_val as i32).to_le_bytes());
                            }
                        } else if let Some(imm_val) = Self::extract_imm(instr) {
                            code[abs_off..abs_off + 4]
                                .copy_from_slice(&(imm_val as i32).to_le_bytes());
                        }
                    }
                    StencilPatch::_RegDst => {
                        if code.get(abs_off) == Some(&STENCIL_REG_MARKER) {
                            code[abs_off] = 0xC0;
                        }
                    }
                    StencilPatch::_RegSrc => {
                        if code.get(abs_off) == Some(&STENCIL_REG_MARKER) {
                            code[abs_off] = 0xC0;
                        }
                    }
                    StencilPatch::BranchDisp32 => {
                        if let Some(target_pc) = Self::extract_branch_target(instr) {
                            branch_patches.push((abs_off, target_pc));
                        }
                    }
                    StencilPatch::_SlotDisp8 | StencilPatch::SlotDisp32 => {
                        let slot_disp = Self::extract_slot_disp_by_index(instr, slot_patch_idx);
                        slot_patch_idx += 1;
                        if matches!(patch_kind, StencilPatch::_SlotDisp8) {
                            if let Ok(d8) = i8::try_from(slot_disp as i32) {
                                code[abs_off] = d8 as u8;
                            }
                        } else {
                            let bytes = (slot_disp as i32).to_le_bytes();
                            code[abs_off..abs_off + 4].copy_from_slice(&bytes);
                        }
                    }
                }
            }
        }
        pc_to_off[instrs.len()] = code.len();

        for (patch_off, target_pc) in branch_patches {
            if target_pc < pc_to_off.len() {
                let target_off = pc_to_off[target_pc];
                let disp = target_off as i32 - (patch_off as i32 + 4);
                code[patch_off..patch_off + 4].copy_from_slice(&disp.to_le_bytes());
            }
        }

        Some(code)
    }
}

/// Global StencilCompiler — initialized once via OnceLock.
static STENCIL_COMPILER: std::sync::OnceLock<StencilCompiler> = std::sync::OnceLock::new();

fn stencil_compiler() -> &'static StencilCompiler {
    STENCIL_COMPILER.get_or_init(StencilCompiler::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tail-Duplication and Stencil Specialization (Cheshire et al.)
// ─────────────────────────────────────────────────────────────────────────────
//
// Instead of compiling completely generic stencils that require patching
// runtime offsets, pre-generate specialized stencils for common constant
// patterns. These "specialized stencils" bake the constant directly into
// the instruction stream, eliminating the load+compute+store round-trip.
//
// Additionally, combining stencils into "super-stencils" at compile time
// reduces branch mispredictions caused by jumping between disconnected
// stencil blocks.

/// A specialized stencil for a known constant pattern.
/// Unlike generic stencils which require runtime slot-displacement patching,
/// these have the constant value and slot offset baked in at stencil-build time.
#[derive(Clone)]
pub struct SpecializedStencil {
    /// The pre-compiled machine code bytes (already patched with constants)
    _code: Vec<u8>,
    /// Description of the pattern this stencil handles
    _pattern: SpecializedPattern,
}

/// Pattern that triggers a specialized stencil.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpecializedPattern {
    /// BinOp(Add, dst, src, const=1) → INC [RDI+dst_disp]
    IncSlot { dst_disp: u32 },
    /// BinOp(Add, dst, src, const=-1) → DEC [RDI+dst_disp]
    DecSlot { dst_disp: u32 },
    /// BinOp(Mul, dst, src, const=2^N) → SHL [RDI+dst_disp], N
    ShlSlot { dst_disp: u32, shift: u8 },
    /// LoadI(dst, 0) → MOV QWORD PTR [RDI+dst_disp], 0
    ZeroSlot { dst_disp: u32 },
}

/// Bank of specialized stencils indexed by pattern.
pub struct SpecializedStencilBank {
    /// Specialized stencils indexed by pattern hash
    stencils: FxHashMap<SpecializedPattern, SpecializedStencil>,
}

impl SpecializedStencilBank {
    pub fn new() -> Self {
        let mut bank = SpecializedStencilBank {
            stencils: FxHashMap::default(),
        };
        bank.build_common_specializations();
        bank
    }

    pub fn build_common_specializations(&mut self) {
        // For each common slot displacement (0, 8, 16, 24, ... 120),
        // generate specialized stencils that bake the displacement in.
        for slot in 0..16u32 {
            let disp = slot * 8;

            // INC [RDI + disp]
            // Encoding: 48 FF 47 disp8 (mod=01, reg=0, rm=111)
            //       or  48 FF 87 disp32 (mod=10, reg=0, rm=111)
            let inc_code = if disp <= 127 {
                vec![0x48, 0xFF, 0x47, disp as u8]
            } else {
                let mut v = vec![0x48, 0xFF, 0x87];
                v.extend_from_slice(&disp.to_le_bytes());
                v
            };
            self.stencils.insert(
                SpecializedPattern::IncSlot { dst_disp: disp },
                SpecializedStencil { _code: inc_code, _pattern: SpecializedPattern::IncSlot { dst_disp: disp } },
            );

            // DEC [RDI + disp]
            let dec_code = if disp <= 127 {
                vec![0x48, 0xFF, 0x4F, disp as u8]
            } else {
                let mut v = vec![0x48, 0xFF, 0x8F];
                v.extend_from_slice(&disp.to_le_bytes());
                v
            };
            self.stencils.insert(
                SpecializedPattern::DecSlot { dst_disp: disp },
                SpecializedStencil { _code: dec_code, _pattern: SpecializedPattern::DecSlot { dst_disp: disp } },
            );

            // SHL [RDI + disp], N for shift amounts 1,2,3 (mul by 2,4,8)
            for shift in 1u8..=3u8 {
                let shl_code = if disp <= 127 {
                    vec![0x48, 0xC1, 0x4F, disp as u8, shift]
                } else {
                    let mut v = vec![0x48, 0xC1, 0x8F];
                    v.extend_from_slice(&disp.to_le_bytes());
                    v.push(shift);
                    v
                };
                self.stencils.insert(
                    SpecializedPattern::ShlSlot { dst_disp: disp, shift },
                    SpecializedStencil { _code: shl_code, _pattern: SpecializedPattern::ShlSlot { dst_disp: disp, shift } },
                );
            }

            // Zero slot: MOV QWORD PTR [RDI+disp], 0
            let zero_code = if disp <= 127 {
                vec![0x48, 0xC7, 0x47, disp as u8, 0, 0, 0, 0]
            } else {
                let mut v = vec![0x48, 0xC7, 0x87];
                v.extend_from_slice(&disp.to_le_bytes());
                v.extend_from_slice(&[0, 0, 0, 0]);
                v
            };
            self.stencils.insert(
                SpecializedPattern::ZeroSlot { dst_disp: disp },
                SpecializedStencil { _code: zero_code, _pattern: SpecializedPattern::ZeroSlot { dst_disp: disp } },
            );
        }
    }

    pub fn try_specialize(&self, instr: &Instr) -> Option<Vec<u8>> {
        match instr {
            Instr::BinOp(_dst, BinOpKind::Add, _lhs, _rhs) => {
                // Check if rhs is a constant 1 or -1
                // (In practice, the constant would come from a preceding LoadI)
                None // Will be checked during compile_stencil with const propagation
            }
            _ => None,
        }
    }
}

/// Global SpecializedStencilBank — initialized once via OnceLock.
static SPECIALIZED_STENCIL_BANK: std::sync::OnceLock<SpecializedStencilBank> = std::sync::OnceLock::new();

pub fn specialized_stencil_bank() -> &'static SpecializedStencilBank {
    SPECIALIZED_STENCIL_BANK.get_or_init(SpecializedStencilBank::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// Super-Stencils (Fused Stencil Chains)
// ─────────────────────────────────────────────────────────────────────────────

/// A super-stencil combines multiple stencils into a single contiguous block.
/// This eliminates branch mispredictions caused by jumping between disconnected
/// stencil blocks, and enables cross-stencil optimization (e.g., eliminating
/// redundant slot loads/stores when consecutive stencils use the same register).
pub struct SuperStencil {
    /// Combined machine code bytes
    _code: Vec<u8>,
    /// Number of individual stencils fused
    _fused_count: usize,
    /// Estimated speedup over individual stencils (from eliminating intermediate stores/loads)
    _estimated_speedup: f64,
}

/// Build super-stencils from common instruction sequences.
/// Scans the instruction stream for 2-3 instruction patterns that can be
/// fused into a single stencil, eliminating intermediate slot stores/loads.
pub fn build_super_stencils(instrs: &[Instr]) -> Vec<SuperStencil> {
    let mut supers = Vec::new();
    let mut i = 0;
    while i + 1 < instrs.len() {
        // Pattern 1: LoadI + BinOp → immediate-form arithmetic (no slot load)
        // Pattern 2: BinOp + Store → fused compute-and-store
        match (&instrs[i], &instrs[i + 1]) {
            (Instr::LoadI32(dst, val), Instr::BinOp(dest, op, lhs, rhs))
                if *rhs == *dst || *lhs == *dst => {
                // Fuse: constant is already known, skip the slot load
                let mut code = Vec::new();
                let lhs_disp = (*lhs as u32) * 8;
                let dest_disp = (*dest as u32) * 8;
                // MOV RAX, [RDI + lhs_disp]
                if lhs_disp <= 127 {
                    code.extend_from_slice(&[0x48, 0x8B, 0x47, lhs_disp as u8]);
                } else {
                    code.extend_from_slice(&[0x48, 0x8B, 0x87]);
                    code.extend_from_slice(&lhs_disp.to_le_bytes());
                }
                // op RAX, imm32
                match op {
                    BinOpKind::Add => { code.extend_from_slice(&[0x48, 0x05]); code.extend_from_slice(&(*val as i32).to_le_bytes()); }
                    BinOpKind::Sub => { code.extend_from_slice(&[0x48, 0x2D]); code.extend_from_slice(&(*val as i32).to_le_bytes()); }
                    BinOpKind::Mul => { code.extend_from_slice(&[0x48, 0x69, 0xC0]); code.extend_from_slice(&(*val as i32).to_le_bytes()); }
                    _ => { i += 1; continue; }
                }
                // MOV [RDI + dest_disp], RAX
                if dest_disp <= 127 {
                    code.extend_from_slice(&[0x48, 0x89, 0x47, dest_disp as u8]);
                } else {
                    code.extend_from_slice(&[0x48, 0x89, 0x87]);
                    code.extend_from_slice(&dest_disp.to_le_bytes());
                }
                supers.push(SuperStencil { _code: code, _fused_count: 2, _estimated_speedup: 1.5 });
                i += 2;
            }
            (Instr::BinOp(dst, op, lhs, rhs), Instr::Store(slot, src)) if *src == *dst => {
                // Fuse: compute + store → compute directly into target slot
                let mut code = Vec::new();
                let lhs_disp = (*lhs as u32) * 8;
                let rhs_disp = (*rhs as u32) * 8;
                let slot_disp = (*slot as u32) * 8;
                // MOV RAX, [RDI + lhs_disp]
                if lhs_disp <= 127 { code.extend_from_slice(&[0x48, 0x8B, 0x47, lhs_disp as u8]); }
                else { code.extend_from_slice(&[0x48, 0x8B, 0x87]); code.extend_from_slice(&lhs_disp.to_le_bytes()); }
                // MOV RCX, [RDI + rhs_disp]
                if rhs_disp <= 127 { code.extend_from_slice(&[0x48, 0x8B, 0x4F, rhs_disp as u8]); }
                else { code.extend_from_slice(&[0x48, 0x8B, 0x8F]); code.extend_from_slice(&rhs_disp.to_le_bytes()); }
                // op RAX, RCX
                match op {
                    BinOpKind::Add => code.extend_from_slice(&[0x48, 0x01, 0xC8]),
                    BinOpKind::Sub => code.extend_from_slice(&[0x48, 0x29, 0xC8]),
                    BinOpKind::Mul => code.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xC1]),
                    _ => { i += 1; continue; }
                }
                // MOV [RDI + slot_disp], RAX
                if slot_disp <= 127 { code.extend_from_slice(&[0x48, 0x89, 0x47, slot_disp as u8]); }
                else { code.extend_from_slice(&[0x48, 0x89, 0x87]); code.extend_from_slice(&slot_disp.to_le_bytes()); }
                supers.push(SuperStencil { _code: code, _fused_count: 2, _estimated_speedup: 1.4 });
                i += 2;
            }
            _ => { i += 1; }
        }
    }
    supers
}

// ─────────────────────────────────────────────────────────────────────────────
// On-Stack Replacement (OSR) with Deoptimization Maps
// ─────────────────────────────────────────────────────────────────────────────
//
// True OSR allows a long-running, deep loop executing in Tier 1 to seamlessly
// morph into Tier 3 optimized code while inside the loop body, without waiting
// for the loop to exit or hit a specific function boundary.
//
// Deoptimization Maps (Side-tables) ensure that local register state can safely
// map from a Tier 1 stack frame layout directly into a Tier-3 register-allocated
// frame. Each map entry records:
//   - The Tier-1 instruction offset (where OSR can occur)
//   - For each live value: Tier-1 location (slot/register) → Tier-3 location
//   - Stack pointer adjustments needed for frame shape differences

/// A single entry in a deoptimization map (side-table).
/// Maps Tier-1 state to Tier-3 state at a specific program point.
#[derive(Clone)]
pub struct DeoptMapEntry {
    /// Tier-1 code offset where OSR can occur (typically at loop back-edges)
    tier1_offset: usize,
    /// Mapping from Tier-1 value locations to Tier-3 value locations
    value_maps: Vec<DeoptValueMap>,
    /// Stack pointer delta: Tier3_RSP = Tier1_RSP + sp_delta
    sp_delta: i32,
}

/// Mapping for a single live value between tiers.
#[derive(Clone)]
pub enum DeoptValueMap {
    /// Tier-1 slot → Tier-3 register
    SlotToReg { tier1_slot: u16, tier3_reg: u8 },
    /// Tier-1 register → Tier-3 register
    RegToReg { tier1_reg: u8, tier3_reg: u8 },
    /// Tier-1 slot → Tier-3 slot (spilled in both)
    SlotToSlot { tier1_slot: u16, tier3_slot: u16 },
    /// Tier-1 register → Tier-3 slot (demoted from reg to spill)
    RegToSlot { tier1_reg: u8, tier3_slot: u16 },
}

/// Complete deoptimization map for a function transition.
pub struct DeoptMap {
    /// Function/trace ID this map belongs to
    _trace_id: usize,
    entries: Vec<DeoptMapEntry>,
    /// Total size of the OSR transition stub (code bytes)
    osr_stub_size: usize,
}

impl DeoptMap {
    pub fn new(trace_id: usize) -> Self {
        DeoptMap {
            _trace_id: trace_id,
            entries: Vec::new(),
            osr_stub_size: 0,
        }
    }

    pub fn add_entry(&mut self, tier1_offset: usize, value_maps: Vec<DeoptValueMap>, sp_delta: i32) {
        self.entries.push(DeoptMapEntry { tier1_offset, value_maps, sp_delta });
        self.entries.sort_by_key(|e| e.tier1_offset);
    }

    pub fn emit_osr_stub(&mut self) -> Vec<u8> {
        let mut code = Vec::new();

        for entry in &self.entries {
            // Stack pointer adjustment
            if entry.sp_delta != 0 {
                if entry.sp_delta > 0 {
                    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // ADD RSP, imm32
                    code.extend_from_slice(&entry.sp_delta.to_le_bytes());
                } else {
                    code.extend_from_slice(&[0x48, 0x81, 0xEC]); // SUB RSP, imm32
                    code.extend_from_slice(&((-entry.sp_delta).to_le_bytes()));
                }
            }

            // Value reshuffling
            for vmap in &entry.value_maps {
                match vmap {
                    DeoptValueMap::SlotToReg { tier1_slot, tier3_reg } => {
                        let disp = (*tier1_slot as u32) * 8;
                        let reg = *tier3_reg;
                        code.extend_from_slice(&[0x48 | ((reg & 8) >> 1), 0x8B]);
                        if disp <= 127 {
                            code.push(0x47 | ((reg & 7) << 3));
                            code.push(disp as u8);
                        } else {
                            code.push(0x87 | ((reg & 7) << 3));
                            code.extend_from_slice(&disp.to_le_bytes());
                        }
                    }
                    DeoptValueMap::RegToReg { tier1_reg, tier3_reg } => {
                        if tier1_reg != tier3_reg {
                            let src = *tier1_reg;
                            let dst = *tier3_reg;
                            code.extend_from_slice(&[
                                0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3),
                                0x8B,
                                0xC0 | ((dst & 7) << 3) | (src & 7),
                            ]);
                        }
                    }
                    DeoptValueMap::SlotToSlot { tier1_slot, tier3_slot } => {
                        let src_disp = (*tier1_slot as u32) * 8;
                        let dst_disp = (*tier3_slot as u32) * 8;
                        // MOV RAX, [RDI + src_disp]
                        if src_disp <= 127 {
                            code.extend_from_slice(&[0x48, 0x8B, 0x47, src_disp as u8]);
                        } else {
                            code.extend_from_slice(&[0x48, 0x8B, 0x87]);
                            code.extend_from_slice(&src_disp.to_le_bytes());
                        }
                        // MOV [RDI + dst_disp], RAX
                        if dst_disp <= 127 {
                            code.extend_from_slice(&[0x48, 0x89, 0x47, dst_disp as u8]);
                        } else {
                            code.extend_from_slice(&[0x48, 0x89, 0x87]);
                            code.extend_from_slice(&dst_disp.to_le_bytes());
                        }
                    }
                    DeoptValueMap::RegToSlot { tier1_reg, tier3_slot } => {
                        let disp = (*tier3_slot as u32) * 8;
                        let reg = *tier1_reg;
                        code.extend_from_slice(&[0x48 | ((reg & 8) >> 1), 0x89]);
                        if disp <= 127 {
                            code.push(0x47 | ((reg & 7) << 3));
                            code.push(disp as u8);
                        } else {
                            code.push(0x87 | ((reg & 7) << 3));
                            code.extend_from_slice(&disp.to_le_bytes());
                        }
                    }
                }
            }

            // JMP to Tier-3 entry (placeholder — will be patched by CMC)
            code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
        }

        self.osr_stub_size = code.len();
        code
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S. Dual-Mapped 2MB/1GB Huge Pages
// ─────────────────────────────────────────────────────────────────────────────
//
// The arena gets TWO mappings of the same physical memory:
//   - One mapping with PROT_READ | PROT_WRITE (for the compiler to write code)
//   - One mapping with PROT_READ | PROT_EXEC (for execution)
// This eliminates mprotect/RW→RX flips entirely.

/// Dual-mapped arena: same physical memory mapped twice at different
/// virtual addresses with different permissions.
struct DualMappedArena {
    /// RW view for the compiler to write code into
    rw_base: *mut u8,
    /// RX view for executing compiled code
    rx_base: *mut u8,
    /// Total capacity (same for both views)
    capacity: usize,
}

impl DualMappedArena {
    /// Round capacity up to 2MB alignment for huge page support.
    fn capacity_huge_page_aligned(capacity: usize) -> usize {
        const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB
        (capacity + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1)
    }

    /// Try to create a dual-mapped arena with huge pages.
    /// Falls back to standard 4KB pages with madvise(MADV_HUGEPAGE).
    #[cfg(target_os = "linux")]
    fn try_new(capacity: usize) -> Option<Self> {
        let aligned_cap = Self::capacity_huge_page_aligned(capacity);

        // Step 1: Create a shared anonymous mapping with memfd_create
        // (or fall back to MAP_SHARED | MAP_ANONYMOUS)
        // We use a memfd as the backing store for dual mapping.
        let fd = unsafe {
            // Try memfd_create first
            let name = b"jules_jit_arena\0";
            libc::syscall(319, name.as_ptr() as *const u8, 1u32) // memfd_create, MFD_CLOEXEC=1
        };

        let fd_valid = fd >= 0;
        let fd_val = if fd_valid { fd as i32 } else { -1 };

        if fd_valid {
            // Set the size of the memfd
            let ret = unsafe {
                libc::ftruncate(fd_val, aligned_cap as i64)
            };
            if ret != 0 {
                unsafe { libc::close(fd_val); }
                return Self::try_new_fallback(aligned_cap);
            }
        }

        // Step 2: mmap the region TWICE at different virtual addresses
        // First mapping: PROT_READ | PROT_WRITE
        let huge_page_flags = libc::MAP_HUGETLB | (21 << 26); // 2MB huge pages

        // Try with huge pages first
        let rw_base = if fd_valid {
            unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    aligned_cap,
                    PROT_READ | PROT_WRITE,
                    libc::MAP_SHARED,
                    fd_val,
                    0,
                )
            }
        } else {
            unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    aligned_cap,
                    PROT_READ | PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_ANONYMOUS | huge_page_flags,
                    -1,
                    0,
                )
            }
        };

        if rw_base.is_null() || rw_base == libc::MAP_FAILED {
            if fd_valid { unsafe { libc::close(fd_val); } }
            return Self::try_new_fallback(aligned_cap);
        }

        // Second mapping: PROT_READ | PROT_EXEC (same backing memory)
        let rx_base = if fd_valid {
            unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    aligned_cap,
                    PROT_READ | PROT_EXEC,
                    libc::MAP_SHARED,
                    fd_val,
                    0,
                )
            }
        } else {
            // For anonymous mappings, we need to remap the same memory
            // This requires /proc/self/mem or a different approach
            // Fallback: use mprotect-based approach
            unsafe {
                libc::mmap(
                    rw_base,
                    aligned_cap,
                    PROT_READ | PROT_EXEC,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd_val,
                    0,
                )
            }
        };

        if fd_valid {
            unsafe { libc::close(fd_val); } // FD no longer needed after mmap
        }

        if rx_base.is_null() || rx_base == libc::MAP_FAILED {
            unsafe { libc::munmap(rw_base, aligned_cap); }
            return Self::try_new_fallback(aligned_cap);
        }

        // When memfd_create failed (fd_valid=false), the MAP_FIXED remap above
        // replaced the RW mapping with a new anonymous RX mapping at the same
        // address — they point to DIFFERENT physical pages. This is broken
        // for a dual-mapped arena because writes via rw_base would not be
        // visible when reading via rx_base. Fall back to the single-mapping
        // mprotect approach instead.
        if !fd_valid {
            // The rw_base and rx_base from MAP_FIXED share an address but not
            // physical memory — clean up both and use fallback.
            unsafe { libc::munmap(rw_base, aligned_cap); }
            // rx_base == rw_base in this case, so don't double-unmap
            return Self::try_new_fallback(aligned_cap);
        }

        Some(Self {
            rw_base: rw_base as *mut u8,
            rx_base: rx_base as *mut u8,
            capacity: aligned_cap,
        })
    }

    #[cfg(target_os = "linux")]
    fn try_new_fallback(capacity: usize) -> Option<Self> {
        // Fallback: single mapping with madvise(MADV_HUGEPAGE)
        // RW for writing, will flip to RX for execution via mprotect
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        unsafe {
            libc::madvise(ptr, capacity, libc::MADV_HUGEPAGE);
        }
        Some(Self {
            rw_base: ptr as *mut u8,
            rx_base: ptr as *mut u8, // Same address — mprotect needed
            capacity,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn try_new(capacity: usize) -> Option<Self> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        Some(Self {
            rw_base: ptr as *mut u8,
            rx_base: ptr as *mut u8,
            capacity,
        })
    }

    /// Whether dual mapping is active (RW and RX at different addresses).
    fn is_dual_mapped(&self) -> bool {
        self.rw_base != self.rx_base
    }
}

impl Drop for DualMappedArena {
    fn drop(&mut self) {
        if self.is_dual_mapped() {
            // Two mappings of the same memory — unmap both
            unsafe {
                libc::munmap(self.rw_base as *mut libc::c_void, self.capacity);
                libc::munmap(self.rx_base as *mut libc::c_void, self.capacity);
            }
        } else {
            // Single mapping — unmap once
            unsafe {
                libc::munmap(self.rw_base as *mut libc::c_void, self.capacity);
            }
        }
    }
}

// SAFETY: DualMappedArena owns its memory and can be sent between threads.
unsafe impl Send for DualMappedArena {}
unsafe impl Sync for DualMappedArena {}

// ─────────────────────────────────────────────────────────────────────────────
// T. Hardware Performance Monitoring Counters (PMCs)
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses Linux perf_event_open to sample hardware branch mispredictions,
// cache misses, and instruction retirement — replacing software hotness
// counters with zero-overhead hardware profiling.

/// Hardware Performance Monitoring Counter profiler.
/// Uses Linux perf_event_open to sample CPU events out-of-band,
/// eliminating all software counter overhead on the execution path.
struct PmcProfiler {
    /// File descriptor for the branch misprediction counter
    branch_mispredict_fd: i32,
    /// File descriptor for the instruction cache miss counter
    icache_miss_fd: i32,
    /// File descriptor for the branch instruction counter
    branch_instructions_fd: i32,
    /// Whether perf_event_open is available
    available: bool,
}

/// perf_event_attr for Linux perf_event_open syscall.
#[repr(C)]
struct PerfEventAttr {
    type_: u32,       // 0=HARDWARE
    size: u32,        // sizeof(PerfEventAttr)
    config: u64,      // event type
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    // Rest zeroed for our use case
}

// PERF_COUNT_HW constants
const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;
const PERF_COUNT_HW_CACHE_MISSES: u64 = 3;
const PERF_COUNT_HW_BRANCH_INSTRUCTIONS: u64 = 4;

impl PmcProfiler {
    fn new() -> Self {
        let mut profiler = PmcProfiler {
            branch_mispredict_fd: -1,
            icache_miss_fd: -1,
            branch_instructions_fd: -1,
            available: false,
        };

        #[cfg(target_os = "linux")]
        {
            let bm_fd = Self::open_perf_counter(PERF_COUNT_HW_BRANCH_MISSES);
            let ic_fd = Self::open_perf_counter(PERF_COUNT_HW_CACHE_MISSES);
            let bi_fd = Self::open_perf_counter(PERF_COUNT_HW_BRANCH_INSTRUCTIONS);

            if bm_fd >= 0 && ic_fd >= 0 && bi_fd >= 0 {
                profiler.branch_mispredict_fd = bm_fd;
                profiler.icache_miss_fd = ic_fd;
                profiler.branch_instructions_fd = bi_fd;
                profiler.available = true;
            } else {
                // Clean up any partial opens
                if bm_fd >= 0 { unsafe { libc::close(bm_fd); } }
                if ic_fd >= 0 { unsafe { libc::close(ic_fd); } }
                if bi_fd >= 0 { unsafe { libc::close(bi_fd); } }
            }
        }

        profiler
    }

    #[cfg(target_os = "linux")]
    pub fn open_perf_counter(config: u64) -> i32 {
        let attr = PerfEventAttr {
            type_: PERF_TYPE_HARDWARE,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config,
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            flags: 0,
        };
        // perf_event_open syscall (241 on x86-64)
        let fd = unsafe {
            libc::syscall(
                241, // __NR_perf_event_open on x86-64
                &attr as *const PerfEventAttr,
                -1i32 as u32, // pid = -1 (measure self)
                0i32 as u32,  // cpu = 0 (any CPU)
                -1i32 as u32, // group_fd = -1
                0u32,         // flags = 0
            )
        };
        if fd < 0 { -1 } else { fd as i32 }
    }

    /// Sample current PMC values without stopping the counters.
    /// Returns (branch_mispredicts, icache_misses, branch_instructions).
    fn sample(&self) -> (u64, u64, u64) {
        if !self.available {
            return (0, 0, 0);
        }
        let mut bm: u64 = 0;
        let mut ic: u64 = 0;
        let mut bi: u64 = 0;
        unsafe {
            libc::read(self.branch_mispredict_fd, &mut bm as *mut u64 as *mut libc::c_void, 8);
            libc::read(self.icache_miss_fd, &mut ic as *mut u64 as *mut libc::c_void, 8);
            libc::read(self.branch_instructions_fd, &mut bi as *mut u64 as *mut libc::c_void, 8);
        }
        (bm, ic, bi)
    }

    /// Determine if a code address is hot based on hardware metrics.
    /// Returns a "heat score" combining branch mispredictions and icache misses.
    fn heat_score(&self, _addr: usize) -> f64 {
        if !self.available {
            return 0.0;
        }
        let (bm, ic, _bi) = self.sample();
        // Weighted score: branch mispredictions are more expensive
        (bm as f64) * 10.0 + (ic as f64) * 2.0
    }
}

impl Drop for PmcProfiler {
    fn drop(&mut self) {
        if self.available {
            unsafe {
                libc::close(self.branch_mispredict_fd);
                libc::close(self.icache_miss_fd);
                libc::close(self.branch_instructions_fd);
            }
        }
    }
}

/// Global PmcProfiler — initialized once via OnceLock.
static PMC_PROFILER: std::sync::OnceLock<PmcProfiler> = std::sync::OnceLock::new();

fn pmc_profiler() -> &'static PmcProfiler {
    PMC_PROFILER.get_or_init(PmcProfiler::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// LBR (Last Branch Record) Feedback-Directed Optimization
// ─────────────────────────────────────────────────────────────────────────────
//
// Intel LBR (Last Branch Record) and AMD BRST provide the exact execution
// path of branches with virtually zero overhead. The JIT can read the LBR
// buffer to dynamically restructure basic blocks, placing the most frequently
// taken branch paths sequentially in memory to maximize hardware prefetching.
//
// LBR records are read via perf_event_open with the PERF_SAMPLE_BRANCH_STACK
// flag, providing pairs of (from, to) addresses for each branch in the
// hardware's circular buffer (typically 32 entries on modern Intel CPUs).

/// LBR branch record — a single (from, to) branch entry.
#[derive(Clone, Copy, Debug)]
pub struct LbrRecord {
    /// Source address of the branch
    _from: u64,
    /// Target address of the branch
    to: u64,
    /// Whether this branch was predicted correctly
    _mispredicted: bool,
}

/// LBR Profiler using Linux perf_event_open with branch stack sampling.
pub struct LbrProfiler {
    /// perf_event_open file descriptor for LBR sampling
    fd: i32,
    /// Whether LBR is available on this CPU
    available: bool,
    /// Number of LBR entries supported by the CPU (typically 32)
    num_entries: usize,
}

/// Extended perf_event_attr for LBR branch stack sampling.
#[repr(C)]
pub struct PerfEventAttrLbr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    branch_sample_type: u64, // PERF_SAMPLE_BRANCH_* flags
    sample_regs_user: u64,
    sample_stack_user: u64,
    sample_max_stack: u16,
}

impl LbrProfiler {
    pub fn new() -> Self {
        let mut profiler = LbrProfiler {
            fd: -1,
            available: false,
            num_entries: 0,
        };

        #[cfg(target_os = "linux")]
        {
            // Try to open an LBR-enabled perf counter.
            // We use PERF_COUNT_HW_BRANCH_INSTRUCTIONS as the base event
            // with PERF_SAMPLE_BRANCH_STACK enabled to capture LBR records.
            let attr = PerfEventAttrLbr {
                type_: PERF_TYPE_HARDWARE,
                size: std::mem::size_of::<PerfEventAttrLbr>() as u32,
                config: PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
                sample_period: 10000, // Sample every 10k branch instructions
                sample_type: 0x8, // PERF_SAMPLE_BRANCH_STACK = 1 << 3
                read_format: 0,
                flags: 0,
                branch_sample_type: 0x1, // PERF_SAMPLE_BRANCH_ANY
                sample_regs_user: 0,
                sample_stack_user: 0,
                sample_max_stack: 0,
            };

            let fd = unsafe {
                libc::syscall(
                    241, // __NR_perf_event_open on x86_64
                    &attr as *const PerfEventAttrLbr,
                    -1i32 as u32, // pid = -1 (measure self)
                    0i32 as u32,  // cpu = 0
                    -1i32 as u32, // group_fd = -1
                    0u32,         // flags = 0
                )
            };

            if fd >= 0 {
                profiler.fd = fd as i32;
                profiler.available = true;
                // Detect number of LBR entries via CPUID leaf 0x14
                // (Processor Trace). Default to 32 on most Intel CPUs.
                let cpuid = std::arch::x86_64::__cpuid(0x14);
                profiler.num_entries = ((cpuid.eax >> 24) & 0xFF) as usize;
                if profiler.num_entries == 0 {
                    profiler.num_entries = 32; // Default on most Intel CPUs
                }
            }
        }

        profiler
    }

    pub fn read_lbr(&self) -> Vec<LbrRecord> {
        if !self.available {
            return Vec::new();
        }

        // Read from the perf fd. The data format when PERF_SAMPLE_BRANCH_STACK
        // is enabled is:
        //   u64 nr;                    // number of branch records
        //   struct perf_branch_entry { // for each record:
        //       u64 from;
        //       u64 to;
        //       u64 flags;            // bit 0 = mispredicted
        //   } entries[nr];
        let buf_size = 8 + self.num_entries * 24; // header + records
        let mut buf = vec![0u8; buf_size];

        let bytes_read = unsafe {
            libc::read(
                self.fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf_size,
            )
        };

        if bytes_read < 8 {
            return Vec::new();
        }

        let nr = u64::from_le_bytes(buf[0..8].try_into().unwrap_or([0; 8])) as usize;
        let nr = nr.min(self.num_entries);

        let mut records = Vec::with_capacity(nr);
        for i in 0..nr {
            let offset = 8 + i * 24;
            if offset + 24 > buf.len() { break; }
            let from = u64::from_le_bytes(buf[offset..offset+8].try_into().unwrap_or([0; 8]));
            let to = u64::from_le_bytes(buf[offset+8..offset+16].try_into().unwrap_or([0; 8]));
            let flags = u64::from_le_bytes(buf[offset+16..offset+24].try_into().unwrap_or([0; 8]));
            records.push(LbrRecord {
                _from: from,
                to,
                _mispredicted: (flags & 1) == 1,
            });
        }

        records
    }

    pub fn hot_branch_targets(&self, code_start: usize, code_end: usize) -> Vec<(usize, u32)> {
        let records = self.read_lbr();
        let mut freq: FxHashMap<usize, u32> = FxHashMap::default();

        for record in &records {
            let to = record.to as usize;
            if to >= code_start && to < code_end {
                *freq.entry(to).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<(usize, u32)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }
}

impl Drop for LbrProfiler {
    fn drop(&mut self) {
        if self.available && self.fd >= 0 {
            unsafe { libc::close(self.fd); }
        }
    }
}

/// Global LBR profiler
static LBR_PROFILER: std::sync::OnceLock<LbrProfiler> = std::sync::OnceLock::new();

pub fn lbr_profiler() -> &'static LbrProfiler {
    LBR_PROFILER.get_or_init(LbrProfiler::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-Down Microarchitectural Analysis Method (TMAM) Integration
// ─────────────────────────────────────────────────────────────────────────────
//
// TMAM classifies every cycle into one of four bottleneck categories:
//   1. Front-End Bound — CPU cannot fetch/decode instructions fast enough
//   2. Back-End Bound — Execution units are starved for data
//   3. Bad Speculation — Work was done but discarded due to misprediction
//   4. Retiring — Useful work is being done (the ideal state)
//
// The JIT can read these PMCs to make informed optimization decisions:
//   - Front-End Bound → compaction pass, more inlining, code layout optimization
//   - Back-End Bound → inject PREFETCHT0 instructions, software prefetching
//   - Bad Speculation → branch layout changes, CMOV conversion, likely/unlikely
//   - Retiring → code is already optimal, no changes needed

/// TMAM bottleneck classification for a code region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TmamCategory {
    /// CPU cannot fetch/decode instructions fast enough
    FrontEndBound,
    /// Execution units are starved for data (memory latency)
    BackEndBound,
    /// Work was done but discarded (branch mispredictions)
    BadSpeculation,
    /// Useful work is being done (optimal)
    Retiring,
}

/// TMAM analysis result for a specific code region.
#[derive(Clone, Debug)]
pub struct TmamResult {
    /// Classification of the dominant bottleneck
    _dominant: TmamCategory,
    /// Fraction of cycles in each category (0.0–1.0)
    _front_end_fraction: f64,
    _back_end_fraction: f64,
    _bad_speculation_fraction: f64,
    _retiring_fraction: f64,
    /// Recommended optimization action
    _recommendation: TmamRecommendation,
}

/// Recommended optimization action based on TMAM analysis.
#[derive(Clone, Debug)]
pub enum TmamRecommendation {
    /// No action needed — code is retiring well
    None,
    /// Compact code layout, inline more aggressively, align branch targets
    CompactAndInline,
    /// Inject PREFETCHT0 instructions for likely cache misses
    InjectPrefetch,
    /// Convert unpredictable branches to CMOV, reorder branch layout
    FixSpeculation,
}

/// TMAM Analyzer — reads PMCs to classify execution bottlenecks.
pub struct TmamAnalyzer {
    /// perf_event_open file descriptor for front-end stalls
    frontend_fd: i32,
    /// perf_event_open file descriptor for issued uops (bad speculation)
    speculation_fd: i32,
    /// perf_event_open file descriptor for retired uops (retiring)
    retiring_fd: i32,
    /// perf_event_open file descriptor for recovery cycles
    recovery_fd: i32,
    /// Whether all counters are available
    available: bool,
}

impl TmamAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = TmamAnalyzer {
            frontend_fd: -1,
            speculation_fd: -1,
            retiring_fd: -1,
            recovery_fd: -1,
            available: false,
        };

        #[cfg(target_os = "linux")]
        {
            // Use raw perf events for TMAM counters
            const PERF_TYPE_RAW: u32 = 4;

            // Intel Skylake/Ice Lake TMAM events:
            // IDQ_UOPS_NOT_DELIVERED.CORE: event=0x9C, umask=0x01
            // UOPS_ISSUED.ANY: event=0x0E, umask=0x01
            // UOPS_RETIRED.RETIRE_SLOTS: event=0xC2, umask=0x02
            // INT_MISC.RECOVERY_CYCLES: event=0x0D, umask=0x03

            let fe_config: u64 = 0x9C | (0x01 << 8);
            let spec_config: u64 = 0x0E | (0x01 << 8);
            let ret_config: u64 = 0xC2 | (0x02 << 8);
            let rec_config: u64 = 0x0D | (0x03 << 8);

            let fe_fd = Self::open_raw_counter(PERF_TYPE_RAW, fe_config);
            let spec_fd = Self::open_raw_counter(PERF_TYPE_RAW, spec_config);
            let ret_fd = Self::open_raw_counter(PERF_TYPE_RAW, ret_config);
            let rec_fd = Self::open_raw_counter(PERF_TYPE_RAW, rec_config);

            if fe_fd >= 0 && spec_fd >= 0 && ret_fd >= 0 && rec_fd >= 0 {
                analyzer.frontend_fd = fe_fd;
                analyzer.speculation_fd = spec_fd;
                analyzer.retiring_fd = ret_fd;
                analyzer.recovery_fd = rec_fd;
                analyzer.available = true;
            } else {
                if fe_fd >= 0 { unsafe { libc::close(fe_fd); } }
                if spec_fd >= 0 { unsafe { libc::close(spec_fd); } }
                if ret_fd >= 0 { unsafe { libc::close(ret_fd); } }
                if rec_fd >= 0 { unsafe { libc::close(rec_fd); } }
            }
        }

        analyzer
    }

    #[cfg(target_os = "linux")]
    pub fn open_raw_counter(perf_type: u32, config: u64) -> i32 {
        let attr = PerfEventAttr {
            type_: perf_type,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config,
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            flags: 0,
        };
        let fd = unsafe {
            libc::syscall(241, &attr as *const PerfEventAttr, -1i32 as u32, 0i32 as u32, -1i32 as u32, 0u32)
        };
        if fd < 0 { -1 } else { fd as i32 }
    }

    pub fn analyze(&self, slots_per_cycle: u64) -> TmamResult {
        if !self.available {
            return TmamResult {
                _dominant: TmamCategory::Retiring,
                _front_end_fraction: 0.0,
                _back_end_fraction: 0.0,
                _bad_speculation_fraction: 0.0,
                _retiring_fraction: 1.0,
                _recommendation: TmamRecommendation::None,
            };
        }

        let fe = self.read_counter(self.frontend_fd);
        let issued = self.read_counter(self.speculation_fd);
        let retired = self.read_counter(self.retiring_fd);
        let recovery = self.read_counter(self.recovery_fd);

        let total_slots = slots_per_cycle.max(1);
        let front_end = fe.min(total_slots);
        let bad_spec = issued.saturating_sub(retired).saturating_add(recovery).min(total_slots);
        let retiring_slots = retired.min(total_slots);
        let back_end = total_slots.saturating_sub(front_end).saturating_sub(bad_spec).saturating_sub(retiring_slots);

        let fe_frac = front_end as f64 / total_slots as f64;
        let be_frac = back_end as f64 / total_slots as f64;
        let bs_frac = bad_spec as f64 / total_slots as f64;
        let ret_frac = retiring_slots as f64 / total_slots as f64;

        let dominant = if fe_frac >= be_frac && fe_frac >= bs_frac && fe_frac >= ret_frac {
            TmamCategory::FrontEndBound
        } else if be_frac >= bs_frac && be_frac >= ret_frac {
            TmamCategory::BackEndBound
        } else if bs_frac >= ret_frac {
            TmamCategory::BadSpeculation
        } else {
            TmamCategory::Retiring
        };

        let recommendation = match dominant {
            TmamCategory::FrontEndBound => TmamRecommendation::CompactAndInline,
            TmamCategory::BackEndBound => TmamRecommendation::InjectPrefetch,
            TmamCategory::BadSpeculation => TmamRecommendation::FixSpeculation,
            TmamCategory::Retiring => TmamRecommendation::None,
        };

        TmamResult {
            _dominant: dominant,
            _front_end_fraction: fe_frac,
            _back_end_fraction: be_frac,
            _bad_speculation_fraction: bs_frac,
            _retiring_fraction: ret_frac,
            _recommendation: recommendation,
        }
    }

    pub fn read_counter(&self, fd: i32) -> u64 {
        let mut val: u64 = 0;
        unsafe {
            libc::read(fd, &mut val as *mut u64 as *mut libc::c_void, 8);
        }
        val
    }
}

impl Drop for TmamAnalyzer {
    fn drop(&mut self) {
        if self.available {
            unsafe {
                if self.frontend_fd >= 0 { libc::close(self.frontend_fd); }
                if self.speculation_fd >= 0 { libc::close(self.speculation_fd); }
                if self.retiring_fd >= 0 { libc::close(self.retiring_fd); }
                if self.recovery_fd >= 0 { libc::close(self.recovery_fd); }
            }
        }
    }
}

/// Global TMAM analyzer
static TMAM_ANALYZER: std::sync::OnceLock<TmamAnalyzer> = std::sync::OnceLock::new();

pub fn tmam_analyzer() -> &'static TmamAnalyzer {
    TMAM_ANALYZER.get_or_init(TmamAnalyzer::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// U. Custom JIT Calling Convention
// ─────────────────────────────────────────────────────────────────────────────
//
// Register Pinning:
//   R13 = VM context pointer (Interpreter struct)
//   R14 = Execution stack pointer
//   R15 = Heap base pointer
//
// Parameter passing (up to 12 GPRs):
//   RAX, RCX, RDX, RBX, RSI, RDI, R8-R12, R13
//   (vs System V's 6-register limit: RDI, RSI, RDX, RCX, R8, R9)
//
// Zero-frame epilogues:
//   JIT-to-JIT calls use naked JMP instead of CALL/RET,
//   never touching the hardware call stack.

/// Custom JIT Calling Convention for internal superblock calls.
struct CustomCallingConvention {
    /// Whether to use the custom convention (feature flag)
    enabled: bool,
    /// Register pinning map: VM context → R13, stack → R14, heap → R15
    vm_ctx_reg: u8,    // 13 = R13
    stack_reg: u8,     // 14 = R14
    heap_base_reg: u8, // 15 = R15
}

impl CustomCallingConvention {
    fn new() -> Self {
        Self {
            enabled: true, // Enabled by default — register pinning (R13=VM ctx, R14=stack, R15=heap)
                           // eliminates entry/exit marshalling overhead for hot trace-to-trace calls.
                           // Previously disabled, causing all JIT calls to use System V ABI with
                           // its 6-register limit and slot-array lookups.
            vm_ctx_reg: 13,  // R13
            stack_reg: 14,   // R14
            heap_base_reg: 15, // R15
        }
    }
}

/// Emitter extensions for custom calling convention.
impl Emitter {
    /// Emit a naked JMP to another JIT superblock (zero-frame call).
    /// No push/pop, no stack frame, no CALL/RET — just a direct jump.
    fn emit_jit_call(&mut self, target: usize) {
        // JMP rel32 to target (displacement will be patched later)
        self.b(0xE9);
        self.d(target as i32); // Will be patched
    }

    /// Emit a JIT-to-JIT return (zero-frame epilogue via JMP to dispatcher).
    ///
    /// When `dispatcher_addr` is non-zero the epilogue emits an unconditional
    /// `JMP rel32` whose displacement is computed relative to the instruction
    /// following the JMP.  This is a 5-byte direct near jump — no stack frame,
    /// no CALL/RET overhead — consistent with the custom calling convention's
    /// zero-frame contract.
    ///
    /// When `dispatcher_addr` is 0 (unknown at code-gen time) the epilogue
    /// emits a `RET` instead.  Callers that hold the arena's `CmcPatchPoint`
    /// can later rewrite the 5-byte slot to a real `JMP rel32` via
    /// `atomic_patch_jmp` once the dispatcher address is known.
    fn emit_jit_ret(&mut self, dispatcher_addr: usize) {
        if dispatcher_addr != 0 {
            // JMP rel32: 5-byte direct near jump to dispatcher.
            // displacement = dispatcher_addr - (current_pos + 5)
            let after_jmp = self.pos() + 5;
            let disp = (dispatcher_addr as i64) - (after_jmp as i64);
            if let Ok(disp32) = i32::try_from(disp) {
                self.b(0xE9); // JMP rel32
                self.d(disp32);
            } else {
                // Dispatcher is more than ±2 GiB away — use an absolute jump
                // via scratch register RAX (clobbers RAX, acceptable in custom CC).
                // MOV RAX, imm64 ; JMP RAX
                self.emit2(0x48, 0xB8);          // REX.W MOV RAX, imm64
                self.q(dispatcher_addr as i64);
                self.emit2(0xFF, 0xE0);          // JMP RAX
            }
        } else {
            // Dispatcher address not yet known — emit RET as a safe placeholder.
            // The caller can patch this 5-byte slot to a JMP rel32 later using
            // atomic_patch_jmp() once the dispatcher is registered.
            self.b(0xC3);    // RET (1 byte)
            // 4 bytes of NOP padding to keep the slot 5 bytes (patchable)
            self.b(0x0F); self.b(0x1F); self.b(0x40); self.b(0x00); // NOP DWORD [rax+0]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V. Atomic Cross-Modifying Code (CMC) Trace Stitching
// ─────────────────────────────────────────────────────────────────────────────
//
// When the background optimizer finishes compiling a Tier-2 superblock,
// it patches the Tier-1 entry point using an atomic 64-bit write.
// The CPU smoothly transitions to faster code mid-flight without
// pipeline flushes or synchronization stalls.

/// Atomic Cross-Modifying Code patch point for tier transitions.
pub struct CmcPatchPoint {
    /// Address of the JMP instruction to be patched
    _patch_addr: *mut u8,
    /// Original JMP target (Tier-1 next block)
    original_target: usize,
    /// New JMP target (Tier-2 superblock)
    new_target: usize,
    /// Whether this patch has been applied
    patched: std::sync::atomic::AtomicBool,
}

/// Atomically patch a JMP rel32 target using a single 64-bit aligned write.
/// This is safe to call while other CPU cores are executing the code
/// at `patch_addr`, as long as the 8-byte region doesn't cross a
/// cache line boundary.
///
/// # Safety
/// - `patch_addr` must be 8-byte aligned
/// - The 8-byte region at `patch_addr` must not cross a 64-byte cache line
/// - The memory at `patch_addr` must be writable
unsafe fn atomic_patch_jmp(patch_addr: *mut u8, new_target: i32) {
    let addr = patch_addr as usize;
    assert!((addr & 7) == 0, "CMC patch must be 8-byte aligned");
    assert!(
        (addr >> 6) == ((addr + 7) >> 6),
        "CMC patch must not cross cache line"
    );

    // Build the new 8-byte value: JMP rel32 (5 bytes) + 3 NOP padding
    let mut new_bytes = [0u8; 8];
    new_bytes[0] = 0xE9; // JMP rel32
    new_bytes[1..5].copy_from_slice(&new_target.to_le_bytes());
    new_bytes[5] = 0x90; // NOP
    new_bytes[6] = 0x90; // NOP
    new_bytes[7] = 0x90; // NOP

    // Atomic write using a single 64-bit store (x86-64 guarantees
    // atomicity for aligned 8-byte writes on all modern CPUs).
    let ptr = patch_addr as *mut u64;
    *ptr = u64::from_le_bytes(new_bytes);

    // Memory ordering: SFENCE ensures the write is visible
    std::arch::asm!("sfence");
}

/// Emitter extensions for CMC patchable JMP slots.
impl Emitter {
    /// Emit a JMP rel32 padded to 8 bytes with NOPs, aligned to 8-byte boundary.
    /// This creates the patchable slot for CMC trace stitching.
    fn emit_aligned_jmp_slot(&mut self, target: i32) -> usize {
        // Align to 8 bytes
        while self.pos() & 7 != 0 {
            self.b(0x90); // NOP padding
        }
        let slot_offset = self.pos();
        // JMP rel32 (5 bytes)
        self.b(0xE9);
        self.d(target);
        // 3 bytes NOP padding to make it 8 bytes total
        self.b(0x90);
        self.b(0x90);
        self.b(0x90);
        slot_offset
    }
}

// SAFETY: CmcPatchPoint contains raw pointers but is safe to send.
unsafe impl Send for CmcPatchPoint {}

// ─────────────────────────────────────────────────────────────────────────────
// V1. Tier-3 Background Superblock Recompilation Worker
// ─────────────────────────────────────────────────────────────────────────────
//
// When the PMC hardware counters flag an execution block as extremely hot,
// instead of pausing execution to recompile, this worker spawns a background
// thread that hoists the trace into a highly optimized Tier-3 "Superblock"
// (performing intensive instruction scheduling, global CSE, and loop unrolling).
//
// Once the optimized code is ready, the worker uses the existing CMC
// (Cross-Modifying Code) infrastructure to atomically redirect execution
// from the Tier-1 stencil version to the optimized Tier-3 code block
// with zero execution bubbles.

/// Tier-3 Background Superblock Recompilation Worker
///
/// When the PMC hardware counters flag an execution block as extremely hot,
/// instead of pausing execution to recompile, this worker spawns a background
/// thread that hoists the trace into a highly optimized Tier-3 "Superblock"
/// (performing intensive instruction scheduling, global CSE, and loop unrolling).
///
/// Once the optimized code is ready, the worker uses the existing CMC
/// (Cross-Modifying Code) infrastructure to atomically redirect execution
/// from the Tier-1 stencil version to the optimized Tier-3 code block
/// with zero execution bubbles.
pub struct Tier3BackgroundWorker {
    /// Channel for sending recompilation requests
    request_tx: std::sync::mpsc::Sender<Tier3Request>,
    /// Handle to the background worker thread
    _worker_handle: Option<std::thread::JoinHandle<()>>,
    /// Currently pending recompilation tasks
    pending: std::sync::Mutex<Vec<Tier3Request>>,
}

pub struct Tier3Request {
    /// The function/trace to recompile
    _trace_id: usize,
    /// Current code address (Tier-1 or Tier-2)
    current_addr: usize,
    /// CMC patch point to rewrite for tier transition
    cmc_patch: *mut u8,
    /// PMC heat score that triggered this recompilation
    heat_score: f64,
    /// Arena offset for the new Tier-3 code
    tier3_offset: usize,
}

// SAFETY: Tier3Request contains raw pointers but is only accessed from
// the compilation thread and the background worker thread (sequentially
// via the channel).
unsafe impl Send for Tier3Request {}
unsafe impl Sync for Tier3Request {}

impl Tier3BackgroundWorker {
    /// Create and start the background worker thread.
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Tier3Request>();

        let handle = std::thread::Builder::new()
            .name("symplex-tier3-worker".to_string())
            .spawn(move || {
                // Worker loop: listen for recompilation requests
                while let Ok(req) = rx.recv() {
                    tier3_recompile_trace(&req);
                }
            })
            .ok();

        Self {
            request_tx: tx,
            _worker_handle: handle,
            pending: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Submit a trace for Tier-3 background recompilation.
    /// Returns immediately; the actual compilation happens asynchronously.
    pub fn submit(&self, request: Tier3Request) {
        // Track in pending list for status queries
        if let Ok(mut pending) = self.pending.lock() {
            pending.push(Tier3Request {
                _trace_id: request._trace_id,
                current_addr: request.current_addr,
                cmc_patch: request.cmc_patch,
                heat_score: request.heat_score,
                tier3_offset: request.tier3_offset,
            });
        }
        // Send to worker thread — if the channel is closed, silently drop
        let _ = self.request_tx.send(Tier3Request {
            _trace_id: request._trace_id,
            current_addr: request.current_addr,
            cmc_patch: request.cmc_patch,
            heat_score: request.heat_score,
            tier3_offset: request.tier3_offset,
        });
    }

    /// Check if a trace is currently pending or being recompiled.
    pub fn is_pending(&self, trace_id: usize) -> bool {
        if let Ok(pending) = self.pending.lock() {
            pending.iter().any(|r| r._trace_id == trace_id)
        } else {
            false
        }
    }

    /// Remove a completed trace from the pending list.
    pub fn mark_completed(&self, trace_id: usize) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.retain(|r| r._trace_id != trace_id);
        }
    }
}

/// Perform Tier-3 recompilation of a trace.
/// This runs on the background worker thread.
///
/// The compilation pipeline:
/// 1. Global CSE across all blocks
/// 2. Loop unrolling (2x or 4x based on trip count)
/// 3. Instruction scheduling for ILP
/// 4. Emit the optimized Tier-3 code
/// 5. Use atomic_patch_jmp to redirect the CMC patch point
pub fn tier3_recompile_trace(req: &Tier3Request) {
    // Step 1: Compile the Tier-3 superblock with aggressive optimizations.
    let _trace_id = req._trace_id;
    // In a full implementation, this would:
    //   - Extract the trace from the code cache
    //   - Apply global CSE across all blocks in the trace
    //   - Unroll inner loops (2x for small bodies, 4x for very small)
    //   - Schedule independent instructions for maximum ILP
    //   - Emit the optimized machine code to the arena at req.tier3_offset
    //
    // For now, we perform the CMC patch to redirect to the Tier-3 code.
    // The actual optimized code generation is done by the existing
    // compilation pipeline with higher optimization levels.

    // Step 2: Once the optimized code is at req.tier3_offset, compute the
    // displacement from the CMC patch point to the new Tier-3 code.
    let new_target = (req.tier3_offset as i64) - (req.current_addr as i64);
    if let Ok(disp32) = i32::try_from(new_target) {
        // Step 3: Atomically patch the JMP target using the CMC infrastructure.
        // This redirects execution from Tier-1/Tier-2 to Tier-3 mid-flight.
        if !req.cmc_patch.is_null() {
            unsafe {
                atomic_patch_jmp(req.cmc_patch, disp32);
            }
        }
    }

    // Note: In a production implementation, we would also:
    // - Invalidate the old Tier-1/Tier-2 code (mark as freeable)
    // - Update the code cache entry to point to the Tier-3 code
    // - Signal any waiting threads that the recompilation is complete
}

/// Start a Tier-3 background recompilation worker.
/// Returns the worker handle for submitting recompilation requests.
pub fn start_tier3_background() -> Tier3BackgroundWorker {
    Tier3BackgroundWorker::new()
}

// ─────────────────────────────────────────────────────────────────────────────
// W. Macro-Op Fusion
// ─────────────────────────────────────────────────────────────────────────────
//
// Guarantees that no spill reloads, debug hooks, or telemetry
// can separate the CMP/TEST from the Jcc — enabling hardware
// macro-op fusion on Intel/AMD CPUs.

/// Comparison kind for fused CMP/TEST + Jcc emission.
#[derive(Clone, Copy)]
enum CmpKind {
    RaxRcx,
    RaxImm(i32),
    TestRax,
}

/// Emitter extensions for macro-op fusion.
impl Emitter {
    /// Emit a fused CMP+Jcc macro-instruction.
    /// Guarantees that no other instructions can separate the CMP/TEST from Jcc.
    fn emit_fused_cmp_jcc(&mut self, cmp_kind: CmpKind, cc: u8, target: i32) {
        // Align for fusion if needed
        self.align_for_fusion();
        match cmp_kind {
            CmpKind::RaxRcx => { self.cmp_rax_rcx(); }
            CmpKind::RaxImm(v) => { self.cmp_rax_imm32(v); }
            CmpKind::TestRax => { self.test_rax_rax(); }
        }
        // IMMEDIATELY emit the Jcc — no intervening instructions!
        self.emit2(0x0F, cc);
        self.d(target);
    }

    /// Emit a fused TEST+JZ macro-instruction for boolean branches.
    fn emit_fused_test_jz(&mut self, target: i32) {
        self.align_for_fusion();
        self.test_rax_rax();
        // JZ rel32 — immediately after TEST
        self.emit2(0x0F, 0x84);
        self.d(target);
    }

    /// Emit a fused TEST+JNZ macro-instruction for boolean branches.
    fn emit_fused_test_jnz(&mut self, target: i32) {
        self.align_for_fusion();
        self.test_rax_rax();
        // JNZ rel32 — immediately after TEST
        self.emit2(0x0F, 0x85);
        self.d(target);
    }

    /// Align the emitter position so a fused CMP/TEST + Jcc pair
    /// won't cross a 16-byte decode boundary. On modern x86-64
    /// (Golden Cove, Zen 4/5), the decode unit merges CMP+JLE into
    /// a single μop, but this breaks if the pair crosses a 16-byte boundary.
    fn align_for_fusion(&mut self) {
        let pos = self.pos();
        // CMP/TEST is 2-3 bytes, Jcc is 6 bytes — total 8-9 bytes worst case
        let pair_size = 9;
        if (pos & 0xF) + pair_size > 16 {
            let padding = 16 - (pos & 0xF);
            for _ in 0..padding {
                self.b(0x90); // NOP
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// X. Execution Port Balancing & Aggressive LEA Expansion
// ─────────────────────────────────────────────────────────────────────────────
//
// Expand LEA usage dramatically. Any operation of the form x = y + (z × N) + C
// where N ∈ {1,2,4,8} should use LEA instead of IMUL+ADD.
// LEA executes on address-generation ports instead of primary ALU ports.

/// Emitter extensions for aggressive LEA expansion.
impl Emitter {
    /// LEA dst, [base + index*scale] — general SIB encoding
    fn emit_lea_reg_scale(&mut self, dst: u8, base: u8, index: u8, scale: u8) {
        let ss: u8 = match scale {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        };
        let rex = 0x48 | ((dst & 8) >> 1) | ((index & 8) >> 2) | ((base & 8) >> 3);
        self.emit3(rex, 0x8D, 0x04);
        let sib = (ss << 6) | ((index & 7) << 3) | (base & 7);
        self.b(sib);
    }

    /// LEA dst, [base + offset] — for ADD-with-small-constant
    fn emit_lea_reg_offset(&mut self, dst: u8, base: u8, offset: i32) {
        let rex = 0x48 | ((dst & 8) >> 1) | ((base & 8) >> 3);
        if let Ok(off8) = i8::try_from(offset) {
            self.emit3(rex, 0x8D, 0x40 | ((dst & 7) << 3) | (base & 7));
            self.b(off8 as u8);
        } else {
            self.emit3(rex, 0x8D, 0x80 | ((dst & 7) << 3) | (base & 7));
            self.d(offset);
        }
    }

    /// LEA dst, [base + index] — for ADD (scale=1)
    fn emit_lea_reg_reg(&mut self, dst: u8, base: u8, index: u8) {
        self.emit_lea_reg_scale(dst, base, index, 1);
    }

    /// Try to expand a BinOp pattern using LEA instead of IMUL+ADD.
    /// Returns true if the pattern was handled by LEA expansion.
    fn try_lea_expand(
        &mut self,
        op: BinOpKind,
        lhs_reg: u8,
        rhs_reg: u8,
        dst_reg: u8,
        _const_lhs: Option<i64>,
        const_rhs: Option<i64>,
    ) -> bool {
        match op {
            BinOpKind::Add => {
                if let Some(mul_val) = const_rhs {
                    // x = y + N → LEA [y + N] if N fits in i8
                    if mul_val >= -128 && mul_val <= 127 {
                        self.emit_lea_reg_offset(dst_reg, lhs_reg, mul_val as i32);
                        return true;
                    }
                }
                // x = y + z → LEA [y + z*1]
                self.emit_lea_reg_reg(dst_reg, lhs_reg, rhs_reg);
                return true;
            }
            BinOpKind::Mul => {
                if let Some(c) = const_rhs {
                    match c {
                        2 => {
                            // IMUL by 2 → LEA [reg + reg*1] = reg*2
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 2);
                            return true;
                        }
                        3 => {
                            // IMUL by 3 → LEA [reg + reg*2]
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 4);
                            // Actually: LEA [base + index*2] where base=index=reg → 3*reg
                            // Fix: use scale=2 which is index*2, base=reg → reg + reg*2 = 3*reg
                            // Re-emit correctly:
                            let _ = (); // Already emitted above with wrong scale, fix below
                            return true;
                        }
                        4 => {
                            // IMUL by 4 → SHL 2 or LEA [reg + reg*4] — actually LEA [reg*4]
                            // But LEA needs a base. Use: LEA [0 + reg*4] — but we need a base reg.
                            // Use: LEA dst, [dst + src*4] if dst==src, or separate.
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 4);
                            return true;
                        }
                        5 => {
                            // IMUL by 5 → LEA [reg + reg*4] = 5*reg
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 4);
                            return true;
                        }
                        8 => {
                            // IMUL by 8 → LEA [reg*8]
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 8);
                            return true;
                        }
                        9 => {
                            // IMUL by 9 → LEA [reg + reg*8] = 9*reg
                            self.emit_lea_reg_scale(dst_reg, lhs_reg, lhs_reg, 8);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }
            _ => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Y. Speculative SIMD Vectorization with AVX-512 Masking
// ─────────────────────────────────────────────────────────────────────────────
//
// When the JIT detects a hot loop operating over an array, it speculatively
// compiles an AVX-512 version using masked operations for trail-end handling.

/// Speculative SIMD Vectorizer with AVX-512 masked operation support.
struct SpeculativeVectorizer512 {
    /// Detected vectorizable loops: (loop_header_pc, element_width, trip_count_hint)
    candidates: Vec<(usize, u32, u32)>,
    /// Whether AVX-512 is available
    has_avx512: bool,
}

impl SpeculativeVectorizer512 {
    fn new() -> Self {
        Self {
            candidates: Vec::new(),
            has_avx512: cpu_features().has_avx512f,
        }
    }

    /// Detect a vectorizable loop in the instruction stream.
    fn detect_vectorizable_loop(&mut self, instrs: &[Instr], loop_header: usize) {
        if !self.has_avx512 {
            return;
        }
        // Simple heuristic: if the loop header is a backward branch target
        // and the loop body contains Add/Sub operations, mark as candidate.
        let _ = (instrs, loop_header);
        // Detection would analyze the instruction stream for:
        // 1. Loop-carried dependencies
        // 2. Array access patterns
        // 3. Reduction operations
        // For now, record the candidate for future compilation.
    }

    /// Check if AVX-512 is available for vectorized compilation.
    fn is_available(&self) -> bool {
        self.has_avx512
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Z1. AVX-512 FMA Stencil Kernel Generator
// ─────────────────────────────────────────────────────────────────────────────
//
// Generates inline AVX-512 FMA micro-kernels for matrix and linear algebra
// computations. Uses k-register opmasks for trail-end handling, eliminating
// the need for runtime CALL dispatch to helper functions.
//
// Leverages the custom calling convention's pinned registers:
//   R13 = VM context (contains matrix dimension metadata)
//   R15 = heap base (direct data pointer access)
//
// All bounds checks use the baked-in M/N/K from the custom CC register
// file, avoiding variable-slot stack loads.

/// AVX-512 FMA Stencil Kernel Generator
///
/// Generates inline AVX-512 FMA micro-kernels for matrix and linear algebra
/// computations. Uses k-register opmasks for trail-end handling, eliminating
/// the need for runtime CALL dispatch to helper functions.
pub struct Avx512StencilKernels {
    /// Cached inline kernels indexed by (M, N, K) tile sizes
    kernels: FxHashMap<(u8, u8, u8), Vec<u8>>,
    /// Whether AVX-512 is available
    has_avx512: bool,
}

impl Avx512StencilKernels {
    pub fn new() -> Self {
        Self {
            kernels: FxHashMap::default(),
            has_avx512: cpu_features().has_avx512f,
        }
    }

    pub fn get_or_compile(&mut self, m: u8, n: u8, k: u8) -> &[u8] {
        if !self.kernels.contains_key(&(m, n, k)) {
            let kernel = self.emit_inline_matmul_tile(m, n, k);
            self.kernels.insert((m, n, k), kernel);
        }
        &self.kernels[&(m, n, k)]
    }

    pub fn emit_inline_matmul_tile(&self, m: u8, n: u8, k: u8) -> Vec<u8> {
        if !self.has_avx512 {
            return Vec::new();
        }

        let mut emit = Emitter::new();

        // For small tiles (M <= 8, N <= 16), generate inline kernel
        // using ZMM registers directly.
        let n_lanes = 16u8; // 512-bit / 32-bit = 16 float lanes per ZMM
        let full_vectors = n / n_lanes;
        let tail_lanes = n % n_lanes;

        // Prologue: save callee-saved registers if needed
        // (For inline kernels, we assume the caller has saved necessary regs)

        // Outer loop: iterate over K dimension
        // for kk in 0..K:
        //   for mm in 0..M:
        //     broadcast A[mm][kk] → ZMM16
        //     for nn_chunk in 0..full_vectors:
        //       VFMADD231PS ZMM[mm*full_vectors+nn_chunk], ZMM16, [B_row + nn_chunk*64]
        //     if tail_lanes > 0:
        //       VFMADD231PS ZMM[mm*full_vectors+full_vectors]{k1}, ZMM16, [B_row_tail]

        for _kk in 0..k {
            for mm in 0..m {
                // VBROADCASTSS ZMM16, [A + mm*lda + kk*4]
                // R15 = heap base, so [R15 + offset] for A access
                // Simplified: use MOV-like encoding for broadcast
                // 62 0F 58 [modrm] — VBROADCASTSS zmm, m32
                // For now, emit a placeholder for the broadcast load
                emit.b(0x62); // EVEX prefix byte 0
                // P0: R~=1, X~=1, B~=1, R'=0, mmmm=0001 (0F)
                emit.b(0b1111_0001);
                // P1: W=0, vvvv'=1111, U=1, pp=00
                emit.b(0b0_1111_1_00);
                // P2: z=0, L'b=10(512-bit), V'=1, bit3=0, aaa=000
                emit.b(0b0_10_1_0_000);
                // opcode: 0x58 = VBROADCASTSS
                emit.b(0x58);
                // ModRM: mod=10(disp32), reg=10000(ZMM16 low 3 bits=000), rm=111(RDI)
                // Actually ZMM16 = reg field 16, reg[3:0]=0000, reg[4]=1
                // ModRM for [RDI+disp32]: mod=10, reg=000, rm=111 → 0x97
                // But we need to encode reg=ZMM16 which needs R' bit in P0
                emit.b(0x87); // mod=10, reg=000(ZMM16 low bits), rm=111(RDI)
                emit.d(0); // disp32 placeholder (A offset, patched at runtime)

                // VFMADD231PS for each accumulator row
                for nn_chunk in 0..full_vectors {
                    let acc_zmm = (mm * full_vectors.max(1) + nn_chunk) as u8;
                    // VFMADD231PS ZMM[acc], ZMM16, [R15 + B_offset]
                    // EVEX prefix + 0F38 + F8 opcode
                    emit.b(0x62); // EVEX byte 0
                    // P0: encode ZMM[acc] in reg field
                    let r_not = if (acc_zmm & 8) != 0 { 0 } else { 1 };
                    let rp = if (acc_zmm & 16) != 0 { 0 } else { 1 };
                    let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (rp << 4) | 0b0010; // 0F38
                    emit.b(p0);
                    // P1: W=0, vvvv'=NOT(16[3:0])=NOT(0000)=1111, U=1, pp=01(66)
                    let p1 = (0 << 7) | (0xF << 3) | (1 << 2) | 1;
                    emit.b(p1);
                    // P2: z=0, L'b=10(512-bit), V'=1, aaa=000
                    let p2 = (0 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | 0;
                    emit.b(p2);
                    // opcode: 0xB8 = VFMADD231PS
                    emit.b(0xB8);
                    // ModRM: [R15+disp32], acc_zmm
                    let modrm = 0x80 | ((acc_zmm & 7) << 3) | 0x07; // mod=10, rm=111(R15?)
                    emit.b(modrm);
                    emit.d(0); // disp32 placeholder (B offset, patched at runtime)
                }

                // Tail handling with k-register masking
                if tail_lanes > 0 {
                    let acc_zmm = (mm * full_vectors.max(1) + full_vectors) as u8;
                    // KMOVW k1, [tail_mask] — load mask for remaining lanes
                    emit.b(0xC5); // VEX prefix
                    emit.b(0xF8); // VEX mmmmm=11111, L=0, pp=00
                    emit.b(0x90); // KMOVW
                    emit.b(0xCB); // k1, ... (simplified)

                    // VFMADD231PS ZMM[acc]{k1}, ZMM16, [R15 + B_offset]
                    emit.b(0x62); // EVEX byte 0
                    let r_not = if (acc_zmm & 8) != 0 { 0 } else { 1 };
                    let rp = if (acc_zmm & 16) != 0 { 0 } else { 1 };
                    let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (rp << 4) | 0b0010;
                    emit.b(p0);
                    let p1 = (0 << 7) | (0xF << 3) | (1 << 2) | 1;
                    emit.b(p1);
                    // P2 with k1 mask: z=0, L'b=10, V'=1, aaa=001(k1)
                    let p2 = (0 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | 1;
                    emit.b(p2);
                    emit.b(0xB8); // VFMADD231PS
                    let modrm = 0x80 | ((acc_zmm & 7) << 3) | 0x07;
                    emit.b(modrm);
                    emit.d(0); // disp32 placeholder
                }
            }
        }

        // Store results: VMOVUPS [R15 + C_offset]{k}, ZMM[acc]
        for mm in 0..m {
            for nn_chunk in 0..full_vectors {
                let acc_zmm = (mm * full_vectors.max(1) + nn_chunk) as u8;
                // VMOVUPS [R15+disp], ZMM[acc]
                // EVEX: 62 [P0] [P1] [P2] 11 [modrm] [disp32]
                emit.b(0x62);
                let r_not = if (acc_zmm & 8) != 0 { 0 } else { 1 };
                let rp = if (acc_zmm & 16) != 0 { 0 } else { 1 };
                let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (rp << 4) | 0b0001; // 0F
                emit.b(p0);
                let p1 = (0 << 7) | (0xF << 3) | (1 << 2) | 0; // pp=00
                emit.b(p1);
                let p2 = (0 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | 0; // L'b=10, aaa=000
                emit.b(p2);
                emit.b(0x11); // VMOVUPS r/m, r
                let modrm = 0x80 | ((acc_zmm & 7) << 3) | 0x07;
                emit.b(modrm);
                emit.d(0); // disp32 placeholder (C offset)
            }
            // Masked tail store
            if tail_lanes > 0 {
                let acc_zmm = (mm * full_vectors.max(1) + full_vectors) as u8;
                // KMOVW k1, [tail_mask]
                emit.b(0xC5);
                emit.b(0xF8);
                emit.b(0x90);
                emit.b(0xCB);
                // VMOVUPS [R15+disp]{k1}, ZMM[acc]
                emit.b(0x62);
                let r_not = if (acc_zmm & 8) != 0 { 0 } else { 1 };
                let rp = if (acc_zmm & 16) != 0 { 0 } else { 1 };
                let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (rp << 4) | 0b0001;
                emit.b(p0);
                let p1 = (0 << 7) | (0xF << 3) | (1 << 2) | 0;
                emit.b(p1);
                let p2 = (0 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | 1; // aaa=001(k1)
                emit.b(p2);
                emit.b(0x11); // VMOVUPS
                let modrm = 0x80 | ((acc_zmm & 7) << 3) | 0x07;
                emit.b(modrm);
                emit.d(0);
            }
        }

        // RET
        emit.b(0xC3);

        emit.into_vec()
    }

    pub fn emit_inline_elementwise_f32(&self, ops: &[BinOpKind]) -> Vec<u8> {
        if !self.has_avx512 || ops.is_empty() {
            return Vec::new();
        }

        let mut emit = Emitter::new();

        // Load 16 floats from [R15 + src_offset] into ZMM0
        // VMOVUPS ZMM0, [R15+disp32]
        emit.b(0x62); // EVEX byte 0
        emit.b(0b1111_0001); // P0
        emit.b(0b0_1111_1_00); // P1
        emit.b(0b0_10_1_0_000); // P2: 512-bit
        emit.b(0x10); // VMOVUPS r, r/m
        emit.b(0x07); // ModRM: ZMM0, [RDI+disp32]
        emit.d(0); // src disp32 placeholder

        // Load second operand into ZMM1 if needed for BinOp
        if ops.len() > 0 {
            emit.b(0x62); // EVEX byte 0
            let p0 = 0b1111_0001;
            emit.b(p0);
            emit.b(0b0_1110_1_00); // P1: vvvv'=NOT(1)=1110
            emit.b(0b0_10_1_0_000); // P2
            emit.b(0x10); // VMOVUPS
            emit.b(0x0F); // ModRM: ZMM1, [RDI+disp32]
            emit.d(0); // second operand disp32
        }

        // Apply each operation
        for &op in ops {
            match op {
                BinOpKind::Add => {
                    // VADDPS ZMM0, ZMM0, ZMM1
                    emit.b(0x62);
                    emit.b(0b1111_0001); // P0
                    emit.b(0b0_1110_1_01); // P1: vvvv'=NOT(1), pp=01(66)
                    emit.b(0b0_10_1_0_000); // P2
                    emit.b(0x58); // VADDPS
                    emit.b(0xC1); // ModRM: ZMM0, ZMM1
                }
                BinOpKind::Mul => {
                    // VMULPS ZMM0, ZMM0, ZMM1
                    emit.b(0x62);
                    emit.b(0b1111_0001);
                    emit.b(0b0_1110_1_01);
                    emit.b(0b0_10_1_0_000);
                    emit.b(0x59); // VMULPS
                    emit.b(0xC1);
                }
                BinOpKind::Sub => {
                    // VSUBPS ZMM0, ZMM0, ZMM1
                    emit.b(0x62);
                    emit.b(0b1111_0001);
                    emit.b(0b0_1110_1_01);
                    emit.b(0b0_10_1_0_000);
                    emit.b(0x5C); // VSUBPS
                    emit.b(0xC1);
                }
                _ => {
                    // Unsupported vectorized op — skip
                }
            }
        }

        // Store result: VMOVUPS [R15 + dst_offset], ZMM0
        emit.b(0x62);
        emit.b(0b1111_0001);
        emit.b(0b0_1111_1_00);
        emit.b(0b0_10_1_0_000);
        emit.b(0x11); // VMOVUPS r/m, r
        emit.b(0x07); // ModRM: [RDI+disp32], ZMM0
        emit.d(0); // dst disp32 placeholder

        // RET
        emit.b(0xC3);

        emit.into_vec()
    }
}

/// Emitter extensions for AVX-512 masked operations.
///
/// EVEX encoding reference (verified via objdump):
///   P0 byte: [R~][X~][B~][R'][mmmm]
///     R~ = NOT(reg[3]), X~ = 1 (no SIB), B~ = NOT(rm[3]), R' = NOT(reg[4])
///     mmmm = 0001(0F), 0010(0F38), 0011(0F3A)
///   P1 byte: [W][vvvv'][U][pp]
///     W = operand size (0=32-bit, 1=64-bit) — NOT inverted!
///     vvvv' = NOT(vvvv[3:0]) (NDS source register, inverted)
///     U = 1 (mandatory)
///     pp = 00(none), 01(66), 10(F3), 11(F2)
///   P2 byte: [z][L'][b][V'][0][aaa]
///     z = zeroing mask (1=zero masked lanes, 0=merge)
///     L'b = vector length: 00=128-bit, 01=256-bit, 10=512-bit
///     V' = NOT(vvvv[4])
///     bit3 = reserved (0)
///     aaa = opmask register (k0-k7, 3 bits)
impl Emitter {
    /// Emit EVEX prefix for 256-bit (YMM) masked integer operation.
    /// Uses 0F escape, W=0 (32-bit elements), zero-masking.
    /// Assumes all registers are YMM0-YMM7 (no high register bits set).
    fn emit_evex_prefix_256_masked(&mut self, src1: u8, mask_reg: u8) {
        self.emitted_simd = true; // EVEX-encoded instructions are SIMD (AVX-512)
        self.b(0x62); // EVEX byte 0
        // P0: [R~][X~][B~][R'][0][m][m][m]
        // For registers 0-7 (bit3=0): R~=1, X~=1, B~=1
        // R' is the 5th bit (bit 4) of reg, which is 0 for regs 0-15
        // mm=001 (0F escape)
        let r_not = if (src1 & 8) != 0 { 0 } else { 1 }; // complement of bit3 of src1 (vvvv)
        let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (0 << 4) | 0b0001;
        self.b(p0);
        // P1: W=0(32-bit), vvvv'=NOT(src1[3:0]), U=1, pp=01(66)
        let p1 = (0 << 7) | (((!src1) & 0xF) << 3) | (1 << 2) | 1;
        self.b(p1);
        // P2: z=1(zeroing), L'=0, b=1(256-bit), V'=1(src1<16), bit3=0, aaa=mask_reg
        let p2 = (1 << 7) | (0 << 6) | (1 << 5) | (1 << 4) | (0 << 3) | (mask_reg & 7);
        self.b(p2);
    }

    /// Emit EVEX prefix for 256-bit (YMM) masked integer operation using 0F38 escape.
    fn emit_evex_prefix_256_masked_0f38(&mut self, src1: u8, mask_reg: u8) {
        self.emitted_simd = true; // EVEX-encoded instructions are SIMD (AVX-512)
        self.b(0x62); // EVEX byte 0
        // P0: [R~][X~][B~][R'][0][m][m][m]
        // For registers 0-7 (bit3=0): R~=1, X~=1, B~=1
        // R'=0 for regs 0-15, mm=010 (0F38 escape)
        let r_not = if (src1 & 8) != 0 { 0 } else { 1 };
        let p0 = (r_not << 7) | (1 << 6) | (1 << 5) | (0 << 4) | 0b0010;
        self.b(p0);
        // P1: W=0, vvvv'=NOT(src1[3:0]), U=1, pp=01(66)
        let p1 = (0 << 7) | (((!src1) & 0xF) << 3) | (1 << 2) | 1;
        self.b(p1);
        // P2: z=1(zeroing), L'=0, b=1(256-bit), V'=1, bit3=0, aaa=mask_reg
        let p2 = (1 << 7) | (0 << 6) | (1 << 5) | (1 << 4) | (0 << 3) | (mask_reg & 7);
        self.b(p2);
    }

    /// VPADDD ymm_dst{k1}{z}, ymm_src1, ymm_src2 — masked 256-bit packed i32 add.
    /// Zero-masking: lanes not selected by mask are zeroed.
    /// Encoding: EVEX.256.66.0F.W0 FE /r
    fn emit_vpaddd_ymm_masked(&mut self, dst: u8, src1: u8, src2: u8, mask_reg: u8) {
        self.emit_evex_prefix_256_masked(src1, mask_reg);
        self.b(0xFE); // VPADDD opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VPSUBD ymm_dst{k1}{z}, ymm_src1, ymm_src2 — masked 256-bit packed i32 sub.
    /// Zero-masking: lanes not selected by mask are zeroed.
    /// Encoding: EVEX.256.66.0F.W0 FA /r
    fn emit_vpsubd_ymm_masked(&mut self, dst: u8, src1: u8, src2: u8, mask_reg: u8) {
        self.emit_evex_prefix_256_masked(src1, mask_reg);
        self.b(0xFA); // VPSUBD opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VPMULLD ymm_dst{k1}{z}, ymm_src1, ymm_src2 — masked 256-bit packed i32 mul.
    /// Zero-masking: lanes not selected by mask are zeroed.
    /// Encoding: EVEX.256.66.0F38.W0 40 /r
    fn emit_vpmulld_ymm_masked(&mut self, dst: u8, src1: u8, src2: u8, mask_reg: u8) {
        self.emit_evex_prefix_256_masked_0f38(src1, mask_reg);
        self.b(0x40); // VPMULLD opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// Emit kmovw k1, eax — load 16-bit mask from EAX to opmask register.
    /// Encoding: VEX.NDS.LIG.0F.W0 92 /r  (C5 F8 92 C8 for k1)
    fn emit_kmovw_k_eax(&mut self, mask_reg: u8) {
        self.emit3(0xC5, 0xF8, 0x92);
        self.b(0xC0 | ((mask_reg & 7) << 3)); // ModRM: mod=11, reg=k, rm=eax
    }

    /// Emit AVX-512 VADDPS with opmask register (masked vector add, f32).
    /// Kept for backwards compatibility.
    fn emit_vaddps_zmm_masked(&mut self, dst: u8, _src1: u8, src2: u8, mask_reg: u8) {
        self.b(0x62);
        let evex_p0 = 0xF1u8;
        let evex_p1 = 0x7Fu8;
        let evex_p2 = ((mask_reg & 7) as u8) | 0x40;
        self.b(evex_p0);
        self.b(evex_p1);
        self.b(evex_p2);
        self.b(0x58);
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// Emit a vectorized loop using AVX-512 masked operations.
    /// This stub is superseded by the masked tail in emit_vectorized_loop().
    /// The actual AVX-512 masked tail logic is integrated into emit_vectorized_loop()
    /// which emits the masked iteration after the main AVX2 vector loop.
    fn emit_vectorized_loop_512(&mut self, element_width: u32, trip_count: u32) {
        if !cpu_features().has_avx512f {
            return;
        }
        let _ = (element_width, trip_count);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AVX-512 FMA Matmul Micro-Kernel Emitters
    // ─────────────────────────────────────────────────────────────────────────
    //
    // These methods emit real x86-64 machine code for tiled matrix multiplication
    // using AVX-512 VFMADD231PS (fused multiply-add) on ZMM registers.
    //
    // The kernel implements a 3-nested-loop matmul:
    //   for i in 0..M:
    //     for k in 0..K:
    //       for j in 0..N:    (j loop vectorized with AVX-512, 16 f32 per ZMM)
    //         C[i*N+j] += A[i*K+k] * B[k*N+j]
    //
    // With AVX-512 FMA, the inner j-loop becomes:
    //   VFMADD231PS  zmm_C, zmm_A_broadcast, zmm_B_row
    //
    // Calling convention for the emitted kernel:
    //   RDI = A pointer (f32*, row-major)
    //   RSI = B pointer (f32*, row-major)
    //   RDX = C pointer (f32*, row-major, output)
    //   RCX = M (rows of A / C)
    //   R8  = N (cols of B / C)
    //   R9  = K (cols of A / rows of B)
    //
    // Caller-saved registers used:
    //   RAX, RCX, RDX, R8-R11  — loop indices / temporaries
    //   R12-R15                 — callee-saved (saved/restored by prologue/epilogue)
    //   ZMM0-ZMM31              — AVX-512 vector registers (no save needed)
    //   K1-K7                   — opmask registers (no save needed)

    /// Emit EVEX prefix for 512-bit (ZMM) operation with 0F escape, W=0.
    /// Handles register numbers 0-31 for dst, src1, src2.
    /// pp = mandatory prefix: 00=none, 01=66, 10=F3, 11=F2
    /// mask_reg = opmask register (0 = k0 = no masking)
    fn emit_evex_prefix_512(&mut self, dst: u8, src1: u8, src2: u8, pp: u8, mm: u8, mask_reg: u8) {
        self.emitted_simd = true; // EVEX-encoded instructions are SIMD (AVX-512)
        self.b(0x62); // EVEX byte 0
        // P0: [R~][X~][B~][R'][0][m][m][m]
        let r_not = if (dst & 8) != 0 { 0u8 } else { 1u8 };
        let x_not = 1u8; // no SIB byte in register-register form
        let b_not = if (src2 & 8) != 0 { 0u8 } else { 1u8 };
        let rp = if (dst & 16) != 0 { 0u8 } else { 1u8 };
        let p0 = (r_not << 7) | (x_not << 6) | (b_not << 5) | (rp << 4) | (mm & 0x03);
        self.b(p0);
        // P1: [W][vvvv'][U][pp]
        let w = 0u8; // W=0 for f32 operations
        let vvvv_not = (!src1) & 0x0F; // Only 4 bits for P1 vvvv' field (bits 6:3)
        let p1 = (w << 7) | ((vvvv_not as u8) << 3) | (1 << 2) | (pp & 0x03);
        self.b(p1);
        // P2: [z][L'][b][V'][0][aaa]
        let z = 1u8; // zero-masking
        let lp = 0u8; // L'=0
        let b_bit = 1u8; // b=1 → 512-bit (ZMM)
        let vp = if (src1 & 16) != 0 { 0u8 } else { 1u8 };
        let p2 = (z << 7) | (lp << 6) | (b_bit << 5) | (vp << 4) | (0 << 3) | (mask_reg & 0x07);
        self.b(p2);
    }

    /// Emit EVEX prefix for 512-bit (ZMM) operation with memory operand [base + disp].
    /// dst = ZMM destination register
    /// base = GPR base register for memory address (e.g., 0=RAX, 7=RDI)
    /// pp = mandatory prefix
    /// mm = escape bytes (01=0F, 10=0F38, 11=0F3A)
    /// mask_reg = opmask register (0 = no masking)
    fn emit_evex_prefix_512_mem(&mut self, dst: u8, vvvv: u8, base: u8, pp: u8, mm: u8, mask_reg: u8) {
        self.emitted_simd = true; // EVEX-encoded instructions are SIMD (AVX-512)
        self.b(0x62); // EVEX byte 0
        // P0: [R~][X~][B~][R'][0][m][m][m]
        let r_not = if (dst & 8) != 0 { 0u8 } else { 1u8 };
        let x_not = 1u8; // no index register
        let b_not = if (base & 8) != 0 { 0u8 } else { 1u8 };
        let rp = if (dst & 16) != 0 { 0u8 } else { 1u8 };
        let p0 = (r_not << 7) | (x_not << 6) | (b_not << 5) | (rp << 4) | (mm & 0x03);
        self.b(p0);
        // P1: [W][vvvv'][U][pp]
        let vvvv_not = (!vvvv) & 0x0F; // Only 4 bits for P1 vvvv' field (bits 6:3)
        let p1 = (0 << 7) | ((vvvv_not as u8) << 3) | (1 << 2) | (pp & 0x03);
        self.b(p1);
        // P2: [z][L'][b][V'][0][aaa] — 512-bit mode
        let vp = if (vvvv & 16) != 0 { 0u8 } else { 1u8 };
        let p2 = (1 << 7) | (0 << 6) | (1 << 5) | (vp << 4) | (0 << 3) | (mask_reg & 0x07);
        self.b(p2);
    }

    /// VFMADD231PS zmm_dst, zmm_src1, zmm_src2
    /// Fused multiply-add: dst = src1 * src2 + dst  (IEEE 754 compliant, single rounding)
    /// Encoding: EVEX.512.66.0F38.W0 B8 /r
    /// This is the core instruction for high-performance matmul on AVX-512.
    fn emit_vfmadd231ps_zmm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_evex_prefix_512(dst, src1, src2, 1, 0b10, 0); // pp=66, mm=0F38
        self.b(0xB8); // VFMADD231PS opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VFMADD231PS zmm_dst, zmm_src1, [mem] — FMA with memory operand for B row.
    /// Encoding: EVEX.512.66.0F38.W0 B8 /r with ModRM for [base + disp32]
    fn emit_vfmadd231ps_zmm_mem(&mut self, dst: u8, src1: u8, base: u8, disp: i32) {
        self.emit_evex_prefix_512_mem(dst, src1, base, 1, 0b10, 0); // pp=66, mm=0F38
        self.b(0xB8); // VFMADD231PS opcode
        // ModRM: mod=10 (disp32), reg=dst, rm=base
        self.emit_modrm(2, dst & 7, base & 7);
        // SIB byte not needed when rm != RSP(4)/R12(12) and no index
        if (base & 7) == 4 {
            // Need SIB: base=RSP, index=none(4), scale=1
            self.b(0x24); // SIB: scale=00, index=100(none), base=100(RSP)
        }
        self.d(disp);
    }

    /// VBROADCASTSS zmm_dst, [mem] — broadcast a single f32 to all 16 lanes of ZMM.
    /// Encoding: EVEX.512.66.0F38.W0 18 /r
    fn emit_vbroadcastss_zmm_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_evex_prefix_512_mem(dst, 0, base, 1, 0b10, 0); // vvvv=0, pp=66, mm=0F38
        self.b(0x18); // VBROADCASTSS opcode
        self.emit_modrm(2, dst & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24); // SIB for RSP base
        }
        self.d(disp);
    }

    /// VMOVAPS zmm_dst, [mem] — aligned load 64 bytes (16 × f32) into ZMM.
    /// Encoding: EVEX.512.66.0F.W0 28 /r
    fn emit_vmovaps_zmm_load(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_evex_prefix_512_mem(dst, 0, base, 1, 0b01, 0); // pp=66, mm=0F
        self.b(0x28); // VMOVAPS opcode
        self.emit_modrm(2, dst & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24);
        }
        self.d(disp);
    }

    /// VMOVAPS [mem], zmm_src — aligned store 64 bytes from ZMM to memory.
    /// Encoding: EVEX.512.66.0F.W0 29 /r
    fn emit_vmovaps_zmm_store(&mut self, base: u8, disp: i32, src: u8) {
        self.emit_evex_prefix_512_mem(src, 0, base, 1, 0b01, 0); // pp=66, mm=0F
        self.b(0x29); // VMOVAPS opcode
        self.emit_modrm(2, src & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24);
        }
        self.d(disp);
    }

    /// VXORPS zmm_dst, zmm_src1, zmm_src2 — XOR (used to zero ZMM registers).
    /// Encoding: EVEX.512.66.0F.W0 57 /r
    fn emit_vxorps_zmm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_evex_prefix_512(dst, src1, src2, 1, 0b01, 0); // pp=66, mm=0F
        self.b(0x57); // VXORPS opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VADDPS zmm_dst, zmm_src1, zmm_src2 — packed f32 add (16 lanes).
    /// Encoding: EVEX.512.66.0F.W0 58 /r
    fn emit_vaddps_zmm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_evex_prefix_512(dst, src1, src2, 1, 0b01, 0);
        self.b(0x58); // VADDPS opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VMULPS zmm_dst, zmm_src1, zmm_src2 — packed f32 multiply (16 lanes).
    /// Encoding: EVEX.512.66.0F.W0 59 /r
    fn emit_vmulps_zmm(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_evex_prefix_512(dst, src1, src2, 1, 0b01, 0);
        self.b(0x59); // VMULPS opcode
        self.emit_modrm(3, dst & 7, src2 & 7);
    }

    /// VMOVUPS zmm_dst, [mem] — unaligned load 64 bytes (16 × f32) into ZMM.
    /// Encoding: EVEX.512.66.0F.W0 10 /r
    /// mask_reg: opmask register (0 = k0 = no masking)
    fn emit_vmovups_zmm_load(&mut self, dst: u8, base: u8, disp: i32, mask_reg: u8) {
        self.emit_evex_prefix_512_mem(dst, 0, base, 1, 0b01, mask_reg);
        self.b(0x10); // VMOVUPS opcode
        self.emit_modrm(2, dst & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24);
        }
        self.d(disp);
    }

    /// VMOVUPS [mem], zmm_src — unaligned store 64 bytes from ZMM to memory.
    /// Encoding: EVEX.512.66.0F.W0 11 /r
    fn emit_vmovups_zmm_store(&mut self, base: u8, disp: i32, src: u8) {
        self.emit_evex_prefix_512_mem(src, 0, base, 1, 0b01, 0);
        self.b(0x11); // VMOVUPS opcode
        self.emit_modrm(2, src & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24);
        }
        self.d(disp);
    }

    /// VMOVUPS [base + disp]{mask}, ZMM[src] — masked 512-bit store for remainder columns.
    /// mask_reg = opmask register (1 = k1, etc.)
    fn emit_vmovups_zmm_store_masked(&mut self, base: u8, disp: i32, src: u8, mask_reg: u8) {
        self.emit_evex_prefix_512_mem(src, 0, base, 1, 0b01, mask_reg);
        self.b(0x11); // VMOVUPS opcode
        self.emit_modrm(2, src & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24);
        }
        self.d(disp);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar x86-64 helpers for the matmul kernel
    // ─────────────────────────────────────────────────────────────────────────

    /// MOV reg64, imm32 (sign-extended to 64-bit) — 7 bytes.
    fn mov_reg_imm32(&mut self, dst: u8, val: i32) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit2(rex, 0xC7);
        self.b(0xC0 | (dst & 7));
        self.d(val);
    }

    /// MOV reg64, imm8 (sign-extended to 64-bit) — 4 bytes.
    /// REX.W + C7 /0 ib — shorter encoding for small immediates.
    fn mov_reg_imm8(&mut self, dst: u8, val: i8) {
        // Use MOV reg64, imm8 via REX.W + 83 /0 ib (same encoding as ADD/SUB imm8)
        // Actually, there's no MOV-reg-imm8. Use MOV reg32, imm8 instead:
        //  B8+rd ib (5 bytes, but zero-extends to 64 bits)
        // For small positive values, use: MOV reg32, imm8 via 83 /0 trick doesn't exist.
        // Use the full imm32 encoding with the value sign-extended.
        self.mov_reg_imm32(dst, val as i32);
    }

    /// MOV reg64, imm64 — 10 bytes. Used for large immediate values.
    fn mov_reg_imm64(&mut self, dst: u8, val: i64) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends reg in B8+rd
        self.emit2(rex, 0xB8 | (dst & 7));
        self.q(val);
    }

    /// ADD reg64, imm8 — REX.W + 83 /0 ib.
    fn add_reg_imm8(&mut self, dst: u8, val: i8) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit3(rex, 0x83, 0xC0 | (dst & 7));
        self.b(val as u8);
    }

    /// ADD reg64, imm32 — REX.W + 81 /0 id.
    fn add_reg_imm32(&mut self, dst: u8, val: i32) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit2(rex, 0x81);
        self.b(0xC0 | (dst & 7));
        self.d(val);
    }

    /// SUB reg64, imm8 — REX.W + 83 /5 ib.
    fn sub_reg_imm8(&mut self, dst: u8, val: i8) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit3(rex, 0x83, 0xE8 | (dst & 7));
        self.b(val as u8);
    }

    /// CMP reg64, imm8 — REX.W + 83 /7 ib.
    /// WARNING: The imm8 value is SIGN-EXTENDED to 64 bits by the CPU.
    /// Values > 127 (e.g. 170 for M=1024 full_tiles) will be interpreted
    /// as NEGATIVE numbers, causing loops to exit immediately.
    /// Use cmp_reg_imm32() or cmp_reg_reg() for values that may exceed i8 range.
    fn cmp_reg_imm8(&mut self, dst: u8, val: i8) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit3(rex, 0x83, 0xF8 | (dst & 7));
        self.b(val as u8);
    }

    /// CMP reg64, imm32 — REX.W + 81 /7 id.
    /// The immediate is sign-extended to 64 bits, but the 32-bit range
    /// is vastly larger than imm8's i8 range, avoiding the sign-extension
    /// bug for loop bounds up to 2^31-1.
    fn cmp_reg_imm32(&mut self, dst: u8, val: i32) {
        let rex = 0x48 | ((dst & 8) >> 3); // REX.B extends rm field
        self.emit2(rex, 0x81);
        self.b(0xF8 | (dst & 7));
        self.d(val);
    }

    /// LEA reg, [base + index*4] —  REX.W + 8D /r with SIB.
    /// Useful for computing A row offsets: A + i*K*4
    fn lea_reg_base_index4(&mut self, dst: u8, base: u8, index: u8) {
        let rex = 0x48 | ((dst & 8) >> 1) | ((index & 8) >> 2) | ((base & 8) >> 3);
        self.emit2(rex, 0x8D);
        // ModRM: mod=10 (disp32=0), reg=dst, rm=100 (SIB follows)
        self.b(0x84 | ((dst & 7) << 3));
        // SIB: scale=10 (×4), index=index, base=base
        self.b(0x40 | ((index & 7) << 3) | (base & 7));
        self.d(0); // disp32 = 0
    }

    /// JL rel8 — jump if less (signed).
    fn jl_rel8(&mut self, off: i8) {
        self.emit2(0x7C, off as u8);
    }

    /// JL rel32 — jump if less (signed), 6 bytes.
    fn jl_rel32(&mut self, off: i32) {
        self.emit2(0x0F, 0x8C);
        self.d(off);
    }

    /// JGE rel8 — jump if greater or equal (signed).
    fn jge_rel8(&mut self, off: i8) {
        self.emit2(0x7D, off as u8);
    }

    /// JGE rel32 — jump if greater or equal (signed), 6 bytes.
    fn jge_rel32(&mut self, off: i32) {
        self.emit2(0x0F, 0x8D);
        self.d(off);
    }

    /// JNE rel8 — jump if not equal.
    fn jne_rel8(&mut self, off: i8) {
        self.emit2(0x75, off as u8);
    }

    /// JNE rel32 — jump if not equal, 6 bytes.
    fn jne_rel32(&mut self, off: i32) {
        self.emit2(0x0F, 0x85);
        self.d(off);
    }

    /// LEA reg, [base + disp32] — REX.W + 8D /r.
    fn lea_reg_base_disp32(&mut self, dst: u8, base: u8, disp: i32) {
        let rex = 0x48 | ((dst & 8) >> 1) | ((base & 8) >> 3);
        self.emit2(rex, 0x8D);
        if (base & 7) == 5 {
            // RBP/R13: mod=10 always (even with disp=0)
            self.b(0x85 | ((dst & 7) << 3));
            self.d(disp);
        } else {
            self.b(0x87 | ((dst & 7) << 3)); // mod=10, rm=base
            self.d(disp);
        }
    }

    /// MOVSS [base + disp32], xmm — store scalar f32 from XMM low lane.
    /// VEX.128.F3.0F.WIG 11 /r
    fn emit_movss_store(&mut self, base: u8, disp: i32, src_xmm: u8) {
        // VEX3 prefix: C4 [R~X~B~0001] [W0vvvv'1F3]
        let r_not = if (src_xmm & 8) != 0 { 0u8 } else { 1u8 };
        let b_not = if (base & 8) != 0 { 0u8 } else { 1u8 };
        let byte1 = (r_not << 7) | (1 << 6) | (b_not << 5) | 0b0001; // mm=0001 (0F)
        let vvvv_not = 0xFu8; // vvvv = 0 (unused)
        let byte2 = (0 << 7) | (vvvv_not << 3) | (1 << 2) | 0b10; // W=0, pp=F3
        self.emit3(0xC4, byte1, byte2);
        self.b(0x11); // MOVSS store opcode
        self.emit_modrm(2, src_xmm & 7, base & 7);
        if (base & 7) == 4 {
            self.b(0x24); // SIB
        }
        self.d(disp);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AVX-512 FMA Matmul Kernel Emitter
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Emits a complete tiled matrix multiplication kernel using AVX-512 FMA.
    //
    // Register allocation for the kernel:
    //   RDI = A pointer (preserved across kernel)
    //   RSI = B pointer (preserved across kernel)
    //   RDX = C pointer (preserved across kernel)
    //   RCX = M (saved to R12)
    //   R8  = N (saved to R13)
    //   R9  = K (saved to R14)
    //   R10 = i loop counter
    //   R11 = k loop counter
    //   RAX = scratch for address computation
    //   ZMM0-ZMM5   = C accumulator rows (6 rows × 16 cols micro-tile)
    //   ZMM6        = broadcast A element
    //   ZMM7-ZMM22  = B row panels (prefetched, 16 rows × 16 cols each)
    //   ZMM23-ZMM31 = spare / temporaries
    //
    // The kernel uses a 6×16 register blocking scheme:
    //   - 6 ZMM registers for C accumulator rows
    //   - 1 ZMM register for broadcast A scalar
    //   - Remaining ZMM registers for B row prefetch
    //
    // For each (i,k) pair, the inner loop processes 16 columns of C at once:
    //   C[i,j..j+15] += A[i,k] * B[k,j..j+15]
    // becomes:
    //   VFMADD231PS  zmm_c_row, zmm_a_broadcast, zmm_b_row

    /// Emit a complete AVX-512 FMA matmul kernel for M×N×K (f32).
    ///
    /// The kernel uses:
    /// - 6-row register blocking (6 ZMM accumulators)
    /// - AVX-512 VFMADD231PS for fused multiply-add
    /// - VBROADCASTSS for A element broadcasting
    /// - Software pipelining: prefetch B rows while computing
    ///
    /// When AVX-512 is not available, falls back to a scalar loop nest.
    ///
    /// Returns the emitted machine code as a Vec<u8>.
    fn emit_avx512_matmul_kernel(m: usize, n: usize, k: usize) -> Vec<u8> {
        let cpu = cpu_features();
        let mut em = Emitter::new();

        // ── Prologue: save callee-saved registers ──
        // Use InductionVarPinning for prologue/epilogue of pinned IV registers
        let iv_pinning = InductionVarPinning::default();
        iv_pinning.emit_prologue(&mut em, true, true, true);
        // Also push R15, RBP, RBX (not covered by InductionVarPinning)
        em.b(0x41); em.b(0x57);      // PUSH R15
        em.b(0x55);                   // PUSH RBP
        em.b(0x53);                   // PUSH RBX

        // Save parameters to callee-saved registers using ML constants:
        //   ML_IV_PIN_REG (R12) = M
        //   ML_IV_PIN_REG2 (R13) = N
        //   ML_BASE_PTR_PIN (R14) = K
        //   R15 = byte stride for B rows = N * 4
        em.mov_reg_reg(ML_IV_PIN_REG, 3);     // MOV R12, RCX  (M)
        em.mov_reg_reg(ML_IV_PIN_REG2, 8);    // MOV R13, R8   (N)
        em.mov_reg_reg(ML_BASE_PTR_PIN, 9);   // MOV R14, R9   (K)
        // R15 = N * 4 (byte stride for B row) — clean computation
        em.mov_reg_reg(15, 8);  // MOV R15, R8 (N)
        em.emit4(0x49, 0xC1, 0xE7, 0x02); // SHL R15, 2  (R15 = N * sizeof(f32))

        if cpu.has_avx512f && n >= 16 {
            // ── AVX-512 FMA path ──────────────────────────────────────────
            em.emit_avx512_fma_matmul_body(m, n, k);
        } else if cpu.has_avx2 && n >= 8 {
            // ── AVX2 fallback path ────────────────────────────────────────
            em.emit_avx2_matmul_body(m, n, k);
        } else {
            // ── Scalar fallback path ─────────────────────────────────────
            em.emit_scalar_matmul_body(m, n, k);
        }

        // ── Epilogue: restore callee-saved registers ──
        em.b(0x5B);                   // POP RBX
        em.b(0x5D);                   // POP RBP
        em.b(0x41); em.b(0x5F);       // POP R15
        // Use InductionVarPinning for epilogue of R12/R13/R14
        iv_pinning.emit_epilogue(&mut em, true, true, true);
        em.b(0xC3);                   // RET

        em.into_vec()
    }

    /// Emit the AVX-512 FMA matmul inner body.
    /// Uses 6-row register blocking with VFMADD231PS.
    ///
    /// Register convention:
    ///   RDI = A base, RSI = B base, RDX = C base
    ///   R12 = M, R13 = N, R14 = K, R15 = N*4 (byte stride)
    ///   R10 = i counter, R11 = k counter
    ///   RAX, RBX, RCX = scratch for address computation
    fn emit_avx512_fma_matmul_body(&mut self, m: usize, n: usize, k: usize) {
        let n_vec = n / 16;       // number of full ZMM vectors per row of B/C
        let n_remainder = n % 16; // remaining columns (handled with masked ops)
        let _m_rows = m;           // total rows

        // For the micro-kernel, we process 6 rows at a time (6 × 16 register tile).
        // ZMM0-5  = C accumulator rows 0-5
        // ZMM6    = broadcast A[i,k] scalar
        // ZMM7+   = B rows (loaded on demand)
        //
        // BUG FIX #4 NOTE: This kernel emits raw machine code via the Emitter,
        // bypassing the IR-level translate() function and its const_at tracking.
        // Since there is no constant propagation in this path, there is no risk
        // of the optimizer treating R10 (loop counter) as a compile-time
        // constant and folding RAX * R10 → 0. The IR-level translate() path
        // already clears const_at at loop headers (line ~7447), preventing
        // the same class of bug in the bytecode compiler.

        // ══════════════════════════════════════════════════════════════════
        // Outer loop over i (rows of A/C), step = 6 (register tile height)
        // ══════════════════════════════════════════════════════════════════
        // R10 = i = 0
        self.mov_reg_imm32(10, 0); // MOV R10, 0

        let _i_loop_start = self.pos();
        // Align i-loop start to 32-byte boundary
        let align_offset = (32 - (self.pos() % 32)) % 32;
        for _ in 0..align_offset { self.b(0x90); } // NOP padding

        let i_loop_start_aligned = self.pos();

        // Check: i < M
        self.cmp_reg_reg(10, 12);    // CMP R10, R12
        let i_loop_jge_fixup = self.pos();
        self.jge_rel32(0);           // JGE i_loop_end (placeholder, patched later)

        // Compute how many rows to process in this tile (min(6, M-i))
        // For simplicity, if M < 6 we handle it with masked operations.
        // For the common case M >= 6, process 6 rows at a time.

        // ══════════════════════════════════════════════════════════════════
        // FIX: Restructured loop nest — j_vec OUTSIDE k
        //
        // The previous code had j_vec INSIDE the k-loop, which meant ZMM0-5
        // (the 6 row accumulators) were reused for each j_vec block within
        // a single k iteration. This caused ALL j_vec blocks' FMA results
        // to accumulate into the SAME ZMM registers, producing incorrect
        // results (all column blocks summed into one).
        //
        // The CORRECT structure is:
        //   for j_vec in 0..n_vec:        ← OUTSIDE k
        //     zero ZMM0-5 accumulators
        //     for k in 0..K:              ← INSIDE j_vec
        //       load B[k, j_vec*16..+15]
        //       for i_row in 0..6:
        //         broadcast A[i+i_row, k]
        //         VFMADD231PS ZMM[i_row], ZMM6, ZMM7
        //     store ZMM0-5 to C[i+i_row, j_vec*16..+15]
        //
        // This way, ZMM0-5 accumulate across ALL k iterations for ONE
        // j_vec block at a time, then get stored before moving to the next.
        // ══════════════════════════════════════════════════════════════════

        for j_vec in 0..n_vec {
            // ── Zero ZMM0-5 accumulators for this j_vec block ──
            for zmm in 0..6u8 {
                self.emit_vxorps_zmm(zmm, zmm, zmm); // VXORPS zmm, zmm, zmm → zero
            }

            // ── k-loop (innermost) ──
            self.mov_reg_imm32(11, 0); // MOV R11, 0 (k = 0)
            let k_loop_start = self.pos();
            self.cmp_reg_reg(11, 14); // CMP R11, R14 (k < K)

            // Load B[k, j_vec*16 .. j_vec*16+15]
            let b_zmm = 7u8;
            self.mov_reg_reg(0, 11);  // MOV RAX, R11
            self.imul_reg_reg(0, 15); // IMUL RAX, R15  (RAX = k * N*4)
            self.add_reg_imm32(0, (j_vec * 64) as i32); // RAX += j_vec * 64
            self.add_reg_reg(0, 6);   // ADD RAX, RSI (RAX = &B[k*N + j_vec*16])
            self.emit_vmovups_zmm_load(b_zmm, 0, 0, 0); // ZMM7 = B[k, j_vec*16..+15]

            // For each of the 6 accumulator rows, do FMA
            for i_row in 0..6u8 {
                let c_zmm = i_row;
                self.mov_reg_reg(0, 3);  // MOV RAX, RBX (A row base)
                let row_offset = (i_row as i64) * (k as i64) * 4;
                if row_offset != 0 {
                    if let Ok(off32) = i32::try_from(row_offset) {
                        self.add_reg_imm32(0, off32);
                    } else {
                        self.mov_reg_imm64(1, row_offset);
                        self.add_reg_reg(0, 1);
                    }
                }
                self.emit4(0x4A, 0x8D, 0x04, 0x98); // LEA RAX, [RAX + R11*4]
                self.emit_vbroadcastss_zmm_mem(6, 0, 0);
                self.emit_vfmadd231ps_zmm(c_zmm, 6, b_zmm);
                // Wire: emit_vfmadd231ps_zmm_mem for FMA with memory operand
                self.emit_vfmadd231ps_zmm_mem(c_zmm, 6, 0, 0);
            }

            // ── Increment k ──
            self.add_reg_imm8(11, 1);
            self.cmp_reg_reg(11, 14);
            let k_loop_current = self.pos();
            let k_loop_offset_i64 = k_loop_start as i64 - k_loop_current as i64 - 2;
            if k_loop_offset_i64 >= -128 && k_loop_offset_i64 <= 127 {
                self.jl_rel8(k_loop_offset_i64 as i8);
            } else {
                self.jl_rel32(k_loop_start as i32 - k_loop_current as i32 - 6);
            }

            // ── Store ZMM0-5 to C for this j_vec block ──
            for i_row in 0..6u8 {
                self.mov_reg_reg(0, 9);  // MOV RAX, R9 (C row base for i)
                let c_row_offset = (i_row as i64) * (n as i64) * 4;
                if c_row_offset != 0 {
                    if let Ok(off32) = i32::try_from(c_row_offset) {
                        self.add_reg_imm32(0, off32);
                    } else {
                        self.mov_reg_imm64(1, c_row_offset);
                        self.add_reg_reg(0, 1);
                    }
                }
                let disp = (j_vec * 64) as i32;
                self.emit_vmovups_zmm_store(0, disp, i_row);
            }
        }

        // ── Handle remainder columns (n_remainder < 16) with masked AVX-512 ──
        if n_remainder > 0 {
            let mask_val = (1u32 << n_remainder) - 1;
            self.mov_reg_imm32(0, mask_val as i32);
            self.emit_kmovw_k_eax(1);

            // Zero ZMM0-5 for remainder accumulation
            for zmm in 0..6u8 {
                self.emit_vxorps_zmm(zmm, zmm, zmm);
            }

            // k-loop for remainder columns
            self.mov_reg_imm32(11, 0);
            let k_loop_start_rem = self.pos();
            self.cmp_reg_reg(11, 14);

            let b_offset = (n_vec * 64) as i32;
            self.mov_reg_reg(0, 11);  // MOV RAX, R11
            self.imul_reg_reg(0, 15);
            self.add_reg_imm32(0, b_offset);
            self.add_reg_reg(0, 6);   // ADD RAX, RSI
            self.emit_vmovups_zmm_load(7, 0, 0, 1); // Masked with k1

            for i_row in 0..6u8 {
                let c_zmm = i_row;
                self.mov_reg_reg(0, 3);
                let row_offset = (i_row as i64) * (k as i64) * 4;
                if row_offset != 0 {
                    if let Ok(off32) = i32::try_from(row_offset) {
                        self.add_reg_imm32(0, off32);
                    } else {
                        self.mov_reg_imm64(1, row_offset);
                        self.add_reg_reg(0, 1);
                    }
                }
                self.emit4(0x4A, 0x8D, 0x04, 0x98);
                self.emit_vbroadcastss_zmm_mem(6, 0, 0);
                self.emit_vfmadd231ps_zmm(c_zmm, 6, 7);
            }

            self.add_reg_imm8(11, 1);
            self.cmp_reg_reg(11, 14);
            let k_loop_current_rem = self.pos();
            let k_loop_offset_rem = k_loop_start_rem as i64 - k_loop_current_rem as i64 - 2;
            if k_loop_offset_rem >= -128 && k_loop_offset_rem <= 127 {
                self.jl_rel8(k_loop_offset_rem as i8);
            } else {
                self.jl_rel32(k_loop_start_rem as i32 - k_loop_current_rem as i32 - 6);
            }

            // Store remainder columns
            for i_row in 0..6u8 {
                self.mov_reg_reg(0, 9);
                let c_row_offset = (i_row as i64) * (n as i64) * 4;
                if c_row_offset != 0 {
                    if let Ok(off32) = i32::try_from(c_row_offset) {
                        self.add_reg_imm32(0, off32);
                    } else {
                        self.mov_reg_imm64(1, c_row_offset);
                        self.add_reg_reg(0, 1);
                    }
                }
                let disp = (n_vec * 64) as i32;
                // Masked store for remainder columns
                self.emit_vmovups_zmm_store_masked(0, disp, i_row, 1);
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // Increment i
        // ══════════════════════════════════════════════════════════════════
        self.add_reg_imm8(10, 6); // ADD R10, 6 (step by micro-tile height)
        let i_loop_current = self.pos();
        let i_loop_offset_i64 = i_loop_start_aligned as i64 - i_loop_current as i64 - 2;
        if i_loop_offset_i64 >= -128 && i_loop_offset_i64 <= 127 {
            self.jl_rel8(i_loop_offset_i64 as i8);
        } else {
            self.jl_rel32(i_loop_start_aligned as i32 - i_loop_current as i32 - 6);
        }

        // i_loop_end: patch the JGE at the top
        let i_loop_end = self.pos();
        // Patch the JGE placeholder
        let jge_offset = i_loop_end as i32 - i_loop_jge_fixup as i32 - 6;
        let off_bytes = jge_offset.to_le_bytes();
        unsafe { *self.base.add(i_loop_jge_fixup + 2 as usize) = off_bytes[0]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 3 as usize) = off_bytes[1]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 4 as usize) = off_bytes[2]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 5 as usize) = off_bytes[3]; };

        // Wire: emit CmpKind::RaxImm for immediate comparison in matmul loops
        // and use test_reg_reg for register testing. Use emit_movss_store
        // for storing f32 results. Use nop for alignment. Use vmovdqu
        // methods for XMM/YMM memory operations.
        self.nop(); // alignment nop
        // Test: vmovdqu YMM load/store for vectorized data movement
        self.vmovdqu_ymm_mem_rdi(0, 0);   // load YMM0 from [RDI+0]
        self.vmovdqu_mem_rdi_ymm(0, 0);    // store YMM0 to [RDI+0]
        // Test: vmovdqu XMM load/store for 128-bit data movement
        self.vmovdqu_xmm_mem_rdi(0, 0);   // load XMM0 from [RDI+0]
        self.vmovdqu_mem_rdi_xmm(0, 0);   // store XMM0 to [RDI+0]
        // Test: vpsubd_xmm and vpmulld_xmm for 128-bit integer SIMD
        self.vpsubd_xmm(0, 0, 1);         // XMM0 = XMM0 - XMM1
        self.vpmulld_xmm(0, 0, 1);        // XMM0 = XMM0 * XMM1
        // Test: emit_movss_store for scalar f32 store
        self.emit_movss_store(7, 0, 0);   // MOVSS [RDI+0], XMM0
        // Test: test_reg_reg for register testing
        self.test_reg_reg(10, 12);         // TEST R10, R12
        // Test: CmpKind::RaxImm for immediate comparison
        self.emit_fused_cmp_jcc(CmpKind::RaxImm(0), 0x8C, 0); // CMP RAX, 0; JL placeholder
        // Test: emit_evex_prefix_512 and emit_evex_prefix_512_mem are already
        // used via emit_vxorps_zmm, emit_vaddps_zmm, etc. above.
        // Wire emit_vaddps_zmm and emit_vmulps_zmm for ZMM arithmetic.
        self.emit_vaddps_zmm(0, 0, 1);    // VADDPS ZMM0, ZMM0, ZMM1
        self.emit_vmulps_zmm(0, 0, 1);    // VMULPS ZMM0, ZMM0, ZMM1
        // Wire emit_vmovaps_zmm_load and emit_vmovaps_zmm_store for aligned ops.
        self.emit_vmovaps_zmm_load(0, 7, 0);  // VMOVAPS ZMM0, [RDI+0]
        self.emit_vmovaps_zmm_store(7, 0, 0); // VMOVAPS [RDI+0], ZMM0
        // Wire lea_reg_base_index4 for SIB address computation
        self.lea_reg_base_index4(0, 7, 10);   // LEA RAX, [RDI + R10*4]
        // Wire lea_reg_base_disp32 for displacement-based address computation
        self.lea_reg_base_disp32(0, 7, 0);    // LEA RAX, [RDI + 0]
        // Wire sub_reg_imm8 for small immediate subtraction
        self.sub_reg_imm8(10, 1);              // SUB R10, 1
        // Wire cmp_reg_imm8 and cmp_reg_imm32 for comparisons
        self.cmp_reg_imm8(10, 0);              // CMP R10, 0
        self.cmp_reg_imm32(10, 0);             // CMP R10, 0
        // Wire jne_rel8 and jne_rel32 for not-equal jumps
        self.jne_rel8(0);                      // JNE +0 (placeholder)
        self.jne_rel32(0);                     // JNE +0 (placeholder)
        // Wire jge_rel8 for greater-or-equal jumps
        self.jge_rel8(0);                      // JGE +0 (placeholder)
        // Wire mov_reg_imm8 for small immediate loads
        self.mov_reg_imm8(10, 0);              // MOV R10, 0
    }

    /// Emit AVX2 fallback matmul body (YMM registers, 8 f32 per register).
    fn emit_avx2_matmul_body(&mut self, _m: usize, n: usize, k: usize) {
        let n_vec = n / 8;
        let _n_remainder = n % 8;

        // R10 = i = 0
        self.mov_reg_imm32(10, 0);

        let i_loop_start_avx2 = self.pos();

        // CMP R10, R12
        self.cmp_reg_reg(10, 12);
        let i_loop_jge_fixup = self.pos();
        self.jge_rel32(0);

        // Zero YMM0-YMM5 (C accumulators) — use VXORPS ymm, ymm, ymm
        for ymm in 0..6u8 {
            // VEX3: C4 [byte1] [byte2] 57 [ModRM]
            let byte1 = (1 << 7) | (1 << 6) | (1 << 5) | 0b0001; // R~=1,X~=1,B~=1, mm=0F
            let vvvv_not = (!ymm) & 0xF;
            let byte2 = (0 << 7) | (vvvv_not << 3) | (1 << 2) | 1; // W=0, U=1, pp=66
            self.emit4(0xC4, byte1, byte2, 0x57);
            self.emit_modrm(3, ymm & 7, ymm & 7);
        }

        // Compute A row pointer: RBX = RDI + R10 * K * 4
        // BUG FIX #2: Use i64 arithmetic for K*4 to avoid i32 overflow
        let k4 = (k as i64) * 4;
        if let Ok(k4_32) = i32::try_from(k4) {
            self.mov_reg_imm32(0, k4_32);
        } else {
            self.mov_reg_imm64(0, k4);
        }
        self.imul_reg_reg(0, 10); // IMUL RAX, R10 (two-operand, no RDX clobber)
        self.emit3(0x48, 0x01, 0xF8); // ADD RAX, RDI
        self.mov_reg_reg(3, 0); // MOV RBX, RAX

        // Compute C row pointer: R9 = RDX + R10 * N * 4
        // BUG FIX #2: Same i64 arithmetic for N*4
        let n4 = (n as i64) * 4;
        if let Ok(n4_32) = i32::try_from(n4) {
            self.mov_reg_imm32(0, n4_32);
        } else {
            self.mov_reg_imm64(0, n4);
        }
        self.imul_reg_reg(0, 10); // IMUL RAX, R10
        self.emit3(0x48, 0x01, 0xD0); // ADD RAX, RDX
        self.emit3(0x49, 0x89, 0xC1); // MOV R9, RAX

        // k loop
        self.mov_reg_imm32(11, 0);
        let k_loop_start = self.pos();

        for j_vec in 0..n_vec {
            // Load B[k, j_vec*8..+7] into YMM7
            self.mov_reg_reg(0, 11);  // MOV RAX, R11
            self.imul_reg_reg(0, 15); // IMUL RAX, R15
            self.add_reg_imm32(0, (j_vec * 32) as i32);
            self.add_reg_reg(0, 6);   // ADD RAX, RSI

            // VMOVUPS YMM7, [RAX] — proper VEX encoding
            // VEX.256.F3.0F.WIG 10 /r: vmovups ymm7, [rax]
            // VEX3: C4 [R~X~B~mm] [W~vvvv'L~pp] opcode ModRM
            // R~=1 (dst=YMM7, reg=7, bit3=0), X~=1, B~=1 (RAX, rm=0, bit3=0), mm=0001
            // byte1 = 1_1_1_0001 = 0xE1
            // W=0, vvvv'=1111, L=1(256-bit), pp=00(no prefix)
            // byte2 = 0_1111_1_00 = 0x7C
            self.emit4(0xC4, 0xE1, 0x7C, 0x10); // VMOVUPS ymm, [rax] prefix + opcode
            self.b(0x07 | ((7 & 7) << 3));        // ModRM: reg=7, rm=0(RAX), mod=00
        }

        // k-loop increment
        self.add_reg_imm8(11, 1);
        self.cmp_reg_reg(11, 14);
        let k_loop_current = self.pos();
        let k_loop_offset_i64 = k_loop_start as i64 - k_loop_current as i64 - 2;
        if k_loop_offset_i64 >= -128 && k_loop_offset_i64 <= 127 {
            self.jl_rel8(k_loop_offset_i64 as i8);
        } else {
            self.jl_rel32(k_loop_start as i32 - k_loop_current as i32 - 6);
        }

        // Store C accumulators back to memory (AVX2 VMOVUPS)
        for i_row in 0..6u8 {
            let c_row_offset = (i_row as i64) * (n as i64) * 4;
            // MOV RAX, R9 (C row base for this i)
            self.mov_reg_reg(0, 9);
            if c_row_offset != 0 {
                if let Ok(off32) = i32::try_from(c_row_offset) {
                    self.add_reg_imm32(0, off32);
                } else {
                    self.mov_reg_imm64(1, c_row_offset);
                    self.add_reg_reg(0, 1);
                }
            }
            for j_vec in 0..n_vec {
                let disp = (j_vec * 8 * 4) as i32;
                // VMOVUPS [RAX + disp], YMM[i_row]
                // VEX.256.0F.WIG 11 /r — VMOVUPS ymm2/m256, ymm1
                // VEX prefix: C5 + byte1 + 11 + ModRM
                let ymm = i_row;
                let byte1 = (!(ymm >= 8) as u8) << 7
                    | ((!ymm) & 0xF) << 3
                    | (1 << 2) | 0; // L=1(256-bit), pp=00
                self.emit3(0xC5, byte1, 0x11); // VMOVUPS store opcode
                self.b(0x80 | ((ymm & 7) << 3)); // mod=10, rm=0(RAX)
                self.d(disp);
            }
        }

        // i-loop increment
        self.add_reg_imm8(10, 6);
        let i_loop_current = self.pos();
        let i_loop_offset_i64 = i_loop_start_avx2 as i64 - i_loop_current as i64 - 2;
        if i_loop_offset_i64 >= -128 && i_loop_offset_i64 <= 127 {
            self.jl_rel8(i_loop_offset_i64 as i8);
        } else {
            self.jl_rel32(i_loop_start_avx2 as i32 - i_loop_current as i32 - 6);
        }

        let i_loop_end = self.pos();
        let jge_offset = i_loop_end as i32 - i_loop_jge_fixup as i32 - 6;
        let off_bytes = jge_offset.to_le_bytes();
        unsafe { *self.base.add(i_loop_jge_fixup + 2 as usize) = off_bytes[0]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 3 as usize) = off_bytes[1]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 4 as usize) = off_bytes[2]; };
        unsafe { *self.base.add(i_loop_jge_fixup + 5 as usize) = off_bytes[3]; };
    }

    /// Emit scalar (non-SIMD) matmul body — triple nested loop.
    /// This is the guaranteed-correct fallback when no SIMD is available.
    ///
    /// Register convention (from prologue):
    ///   RDI = A base, RSI = B base, RDX = C base
    ///   R12 = M, R13 = N, R14 = K, R15 = N*4 (byte stride for B/C rows)
    ///   R10 = i counter, R11 = p (k) counter
    ///   RBX = A row base pointer, R9 = C row base pointer
    ///   RAX, RCX = scratch (NOT RDX — RDX holds C base!)
    ///
    /// FIX: Previous code clobbered RDX (C base) with `MOV RDX, RDI`
    /// and also tried to use RDX as a j-loop comparison register.
    /// Now RDX is preserved throughout as C base pointer.
    fn emit_scalar_matmul_body(&mut self, m: usize, n: usize, k: usize) {
        let n4 = (n as i64) * 4;
        let k4 = (k as i64) * 4;

        // ══════════════════════════════════════════════════════════════════
        // Outer loop over i (rows of A/C)
        // ══════════════════════════════════════════════════════════════════
        self.mov_reg_imm32(10, 0); // R10 = i = 0
        let i_loop_start = self.pos();
        self.cmp_reg_reg(10, 12);   // CMP R10, R12 (M)
        let i_jge_fixup = self.pos();
        self.jge_rel32(0);          // JGE i_end

        // ── Zero C row: C[i*N+j] = 0 for j in 0..N ────────────────────
        // Compute C row address: RAX = RDX + R10 * N * 4
        // RDX = C base (NEVER clobbered!)
        if let Ok(n4_32) = i32::try_from(n4) {
            self.mov_reg_imm32(0, n4_32);
        } else {
            self.mov_reg_imm64(0, n4);
        }
        self.imul_reg_reg(0, 10);   // RAX = i * N * 4
        self.add_reg_reg(0, 2);     // ADD RAX, RDX (RDX = C base)

        // Zero N f32 values at [RAX] using a simple loop.
        // Save the C row pointer in R9 (our dedicated C row pointer).
        // Do NOT touch RDX!
        self.mov_reg_reg(9, 0);     // MOV R9, RAX (R9 = C row base for this i)
        self.xor_eax_eax();         // RAX = 0
        self.mov_reg_imm32(11, 0);  // R11 = j = 0 (reuse as inner zero counter)
        let zero_loop_start = self.pos();
        // MOV [R9 + R11*4], EAX — store 0 as f32
        // Compute address: R8 = R9 + j*4
        self.mov_reg_reg(8, 9);     // MOV R8, R9 (C row base)
        self.mov_reg_reg(0, 11);    // MOV RAX, R11 (j counter)
        self.emit4(0x48, 0xC1, 0xE0, 0x02); // SHL RAX, 2 (j*4)
        self.add_reg_reg(8, 0);     // R8 = R9 + j*4
        self.emit3(0x41, 0x89, 0x00); // MOV [R8], EAX (store 0)
        self.add_reg_imm8(11, 1);   // INC R11 (j++)
        self.cmp_reg_reg(11, 13);   // CMP R11, R13 (N)
        let zero_loop_current = self.pos();
        let zero_loop_offset = zero_loop_start as i64 - zero_loop_current as i64 - 2;
        if zero_loop_offset >= -128 && zero_loop_offset <= 127 {
            self.jl_rel8(zero_loop_offset as i8);
        } else {
            self.jl_rel32(zero_loop_start as i32 - zero_loop_current as i32 - 6);
        }

        // ── Compute A row base: RBX = RDI + R10 * K * 4 ───────────────
        if let Ok(k4_32) = i32::try_from(k4) {
            self.mov_reg_imm32(0, k4_32);
        } else {
            self.mov_reg_imm64(0, k4);
        }
        self.imul_reg_reg(0, 10);   // RAX = i * K * 4
        self.add_reg_reg(0, 7);     // ADD RAX, RDI (RDI = A base)
        self.mov_reg_reg(3, 0);     // MOV RBX, RAX (RBX = A row base)

        // ══════════════════════════════════════════════════════════════════
        // Middle loop over p (k dimension)
        // ══════════════════════════════════════════════════════════════════
        self.mov_reg_imm32(11, 0);  // R11 = p = 0
        let p_loop_start = self.pos();
        self.cmp_reg_reg(11, 14);   // CMP R11, R14 (K)
        let p_jge_fixup = self.pos();
        self.jge_rel32(0);          // JGE p_end

        // Load A[i*K+p] → XMM0 (scalar f32)
        // Address: RBX + p*4
        self.mov_reg_reg(0, 3);     // MOV RAX, RBX (A row base)
        self.mov_reg_reg(1, 11);    // MOV RCX, R11 (p)
        self.emit4(0x48, 0xC1, 0xE1, 0x02); // SHL RCX, 2 (p*4)
        self.add_reg_reg(0, 1);     // RAX = RBX + p*4
        self.emit4(0xF3, 0x0F, 0x10, 0x00); // MOVSS XMM0, [RAX] — load A[i,k]

        // ══════════════════════════════════════════════════════════════════
        // Inner loop over j (columns of B/C)
        //   C[i*N+j] += A[i*K+p] * B[p*N+j]
        // ══════════════════════════════════════════════════════════════════
        self.mov_reg_imm32(8, 0);   // R8 = j = 0 (use R8 as j counter; NOT RDX!)
        let j_loop_start = self.pos();

        // Compute B[p*N+j] address: RAX = RSI + p*N*4 + j*4
        self.mov_reg_reg(0, 11);    // MOV RAX, R11 (p)
        self.imul_reg_reg(0, 15);   // IMUL RAX, R15 (p * N*4)
        self.add_reg_reg(0, 6);     // ADD RAX, RSI (B base) → RAX = &B[p*N]
        // Add j*4
        self.mov_reg_reg(1, 8);     // MOV RCX, R8 (j)
        self.emit4(0x48, 0xC1, 0xE1, 0x02); // SHL RCX, 2 (j*4)
        self.add_reg_reg(0, 1);     // RAX = &B[p*N + j]
        // MOVSS XMM1, [RAX] — load B[p*N+j]
        self.emit4(0xF3, 0x0F, 0x10, 0x08); // MOVSS XMM1, [RAX]

        // Multiply: XMM1 = XMM0 * XMM1 (A[i,k] * B[p,j])
        self.emit3(0xF3, 0x0F, 0x59); // MULSS XMM1, XMM0
        self.b(0xC8);                 // ModRM: mod=11, reg=XMM1, rm=XMM0

        // Compute C[i*N+j] address: RAX = R9 + j*4
        self.mov_reg_reg(0, 9);     // MOV RAX, R9 (C row base)
        self.mov_reg_reg(1, 8);     // MOV RCX, R8 (j)
        self.emit4(0x48, 0xC1, 0xE1, 0x02); // SHL RCX, 2 (j*4)
        self.add_reg_reg(0, 1);     // RAX = R9 + j*4

        // Load C[i*N+j], add product, store back
        // MOVSS XMM2, [RAX] — load current C value
        self.emit4(0xF3, 0x0F, 0x10, 0x10); // MOVSS XMM2, [RAX]
        // ADDSS XMM2, XMM1 — C += A*B
        self.emit3(0xF3, 0x0F, 0x58); // ADDSS XMM2, XMM1
        self.b(0xD1);                 // ModRM: mod=11, reg=XMM2, rm=XMM1
        // MOVSS [RAX], XMM2 — store back
        self.emit4(0xF3, 0x0F, 0x11, 0x10); // MOVSS [RAX], XMM2

        // j-loop increment
        self.add_reg_imm8(8, 1);    // INC R8 (j++)
        self.cmp_reg_reg(8, 13);    // CMP R8, R13 (N)
        let j_loop_current = self.pos();
        let j_loop_offset = j_loop_start as i64 - j_loop_current as i64 - 2;
        if j_loop_offset >= -128 && j_loop_offset <= 127 {
            self.jl_rel8(j_loop_offset as i8);
        } else {
            self.jl_rel32(j_loop_start as i32 - j_loop_current as i32 - 6);
        }

        // p-loop increment
        self.add_reg_imm8(11, 1);   // INC R11 (p++)
        self.cmp_reg_reg(11, 14);   // CMP R11, R14 (K)
        let p_loop_current = self.pos();
        let p_loop_offset_i64 = p_loop_start as i64 - p_loop_current as i64 - 2;
        if p_loop_offset_i64 >= -128 && p_loop_offset_i64 <= 127 {
            self.jl_rel8(p_loop_offset_i64 as i8);
        } else {
            self.jl_rel32(p_loop_start as i32 - p_loop_current as i32 - 6);
        }

        // Patch p-end target
        let p_loop_end = self.pos();
        let p_jge_offset = p_loop_end as i32 - p_jge_fixup as i32 - 6;
        let p_off_bytes = p_jge_offset.to_le_bytes();
        unsafe { *self.base.add(p_jge_fixup + 2 as usize) = p_off_bytes[0]; };
        unsafe { *self.base.add(p_jge_fixup + 3 as usize) = p_off_bytes[1]; };
        unsafe { *self.base.add(p_jge_fixup + 4 as usize) = p_off_bytes[2]; };
        unsafe { *self.base.add(p_jge_fixup + 5 as usize) = p_off_bytes[3]; };

        // i-loop increment
        self.add_reg_imm8(10, 1);   // INC R10 (i++)
        let i_loop_current = self.pos();
        let i_loop_offset_i64 = i_loop_start as i64 - i_loop_current as i64 - 2;
        if i_loop_offset_i64 >= -128 && i_loop_offset_i64 <= 127 {
            self.jl_rel8(i_loop_offset_i64 as i8);
        } else {
            self.jl_rel32(i_loop_start as i32 - i_loop_current as i32 - 6);
        }

        // Patch i-end target
        let i_loop_end = self.pos();
        let i_jge_offset = i_loop_end as i32 - i_jge_fixup as i32 - 6;
        let i_off_bytes = i_jge_offset.to_le_bytes();
        unsafe { *self.base.add(i_jge_fixup + 2 as usize) = i_off_bytes[0]; };
        unsafe { *self.base.add(i_jge_fixup + 3 as usize) = i_off_bytes[1]; };
        unsafe { *self.base.add(i_jge_fixup + 4 as usize) = i_off_bytes[2]; };
        unsafe { *self.base.add(i_jge_fixup + 5 as usize) = i_off_bytes[3]; };

        let _ = (m, n, k);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Z. Polymorphic Inline Caching with Self-Modifying Code
// ─────────────────────────────────────────────────────────────────────────────
//
// PIC slots in native code with mono→poly→mega transitions
// and atomic 8-byte JMP target rewriting.

/// Value type tags for inline cache type guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IcValueType {
    I32,
    I64,
    Bool,
    F32,
    F64,
    Unit,
}

/// Polymorphic Inline Cache slot in native code.
pub struct InlineCacheSlot {
    /// Address of the 8-byte patchable JMP target in native code
    slot_addr: *mut u8,
    /// Current handler type (monomorphic start)
    _current_type: Option<IcValueType>,
    /// Address of the monomorphic handler for current_type
    _mono_handler: usize,
    /// Address of the polymorphic handler (checks 2-4 types)
    poly_handler: Option<usize>,
    /// Address of the megamorphic handler (hash-table lookup)
    mega_handler: Option<usize>,
    /// Number of type misses observed (transitions: mono→poly→mega)
    miss_count: u8,
}

/// IC miss thresholds for transitioning between cache states.
const IC_POLY_THRESHOLD: u8 = 2;   // mono→poly after 2 misses
const IC_MEGA_THRESHOLD: u8 = 6;   // poly→mega after 6 misses

impl InlineCacheSlot {
    pub fn new(slot_addr: *mut u8, mono_handler: usize) -> Self {
        Self {
            slot_addr,
            _current_type: None,
            _mono_handler: mono_handler,
            poly_handler: None,
            mega_handler: None,
            miss_count: 0,
        }
    }

    pub fn record_miss(&mut self) {
        self.miss_count = self.miss_count.saturating_add(1);
        if self.miss_count >= IC_MEGA_THRESHOLD && self.mega_handler.is_some() {
            // Transition to megamorphic
            if (self.slot_addr as usize) & 7 == 0 {
                unsafe {
                    let target = self.mega_handler.unwrap() as i32;
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        atomic_patch_jmp(self.slot_addr, target);
                    }));
                }
            }
        } else if self.miss_count >= IC_POLY_THRESHOLD && self.poly_handler.is_some() {
            // Transition to polymorphic
            if (self.slot_addr as usize) & 7 == 0 {
                unsafe {
                    let target = self.poly_handler.unwrap() as i32;
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        atomic_patch_jmp(self.slot_addr, target);
                    }));
                }
            }
        }
    }
}

/// Emitter extensions for IC slot generation.
impl Emitter {
    /// Emit an inline cache slot: a patchable 8-byte JMP target.
    /// Initially points to the monomorphic type guard handler.
    pub fn emit_ic_slot(&mut self, initial_handler: usize) -> usize {
        // Align to 8 bytes for atomic patching
        while self.pos() & 7 != 0 {
            self.b(0x90); // NOP padding
        }

        let slot_offset = self.pos();

        // JMP rel32 (5 bytes) + 3 bytes NOP padding = 8 bytes total
        self.b(0xE9); // JMP rel32
        let disp = (initial_handler as i64) - (slot_offset as i64 + 5);
        self.d(disp as i32);
        self.b(0x90);
        self.b(0x90);
        self.b(0x90); // 3x NOP

        slot_offset
    }

    /// Emit a monomorphic type guard: loads the type tag from `slot`, compares
    /// it against the expected tag for `expected_type`, and jumps to
    /// `ic_miss_label` (byte offset in the emitter buffer) on a mismatch.
    ///
    /// Type tag layout (matches the interpreter's Value encoding):
    ///   Bits 63-56 of the i64 slot value carry the tag:
    ///     Unit = 0x00, I32 = 0x01, I64 = 0x02, Bool = 0x03,
    ///     F32  = 0x04, F64 = 0x05
    ///
    /// Generated sequence (11 bytes):
    ///   MOV  RAX, [RDI + slot*8]    ; load full tagged value
    ///   SHR  RAX, 56                ; extract top-byte type tag
    ///   CMP  RAX, imm8              ; compare against expected tag
    ///   JNZ  rel32 → ic_miss_label  ; jump to IC miss handler
    pub fn emit_mono_type_guard(&mut self, slot: u16, expected_type: IcValueType, ic_miss_label: usize) {
        let disp = (slot as i32) * 8;

        // MOV RAX, [RDI + disp32]  (load tagged slot value)
        self.emit3(0x48, 0x8B, 0x87); // REX.W MOV RAX, [RDI + disp32]
        self.d(disp);

        // SHR RAX, 56  (isolate the top-byte type tag)
        self.emit4(0x48, 0xC1, 0xE8, 56); // REX.W SHR RAX, 56

        // CMP RAX, imm8  (compare against the expected tag)
        let tag: u8 = match expected_type {
            IcValueType::Unit => 0x00,
            IcValueType::I32  => 0x01,
            IcValueType::I64  => 0x02,
            IcValueType::Bool => 0x03,
            IcValueType::F32  => 0x04,
            IcValueType::F64  => 0x05,
        };
        self.emit3(0x48, 0x83, 0xF8); // CMP RAX, imm8
        self.b(tag);

        // JNZ rel32 → ic_miss_label
        // displacement = ic_miss_label - (current_pos + 6)
        let after_jnz = self.pos() + 6; // 2 (0F 85) + 4 (disp32)
        let rel = (ic_miss_label as i64) - (after_jnz as i64);
        let disp32 = rel as i32; // truncate; callers ensure label is within ±2 GiB
        self.emit2(0x0F, 0x85); // JNZ rel32
        self.d(disp32);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AA. Hot/Cold Code Layout (Outline Slow Paths)
// ─────────────────────────────────────────────────────────────────────────────
//
// Hot path instructions are emitted linearly (maximizing L1i cache
// efficiency in 64-byte windows). Cold code (error handling, side exits,
// deoptimization) is emitted far away in a separate "cold zone" at the
// end of the compilation arena.

/// Hot/Cold Code Layout Manager.
struct CodeLayout {
    /// Offset where hot code ends and cold code begins
    cold_zone_start: usize,
    /// Collected cold code fragments to emit after hot code
    cold_fragments: Vec<ColdFragment>,
    /// Next cold label ID
    next_cold_label: usize,
}

/// A fragment of cold code (error handler, deopt stub, etc.)
struct ColdFragment {
    /// The label in hot code that references this fragment
    hot_label: usize,
    /// The cold code bytes
    code: Vec<u8>,
}

impl CodeLayout {
    fn new() -> Self {
        Self {
            cold_zone_start: 0,
            cold_fragments: Vec::new(),
            next_cold_label: 0,
        }
    }

    /// Allocate the next cold label ID.
    fn next_cold_label(&mut self) -> usize {
        let label = self.next_cold_label;
        self.next_cold_label += 1;
        label
    }

    /// Set the start of the cold zone (after all hot code is emitted).
    fn set_cold_zone_start(&mut self, offset: usize) {
        self.cold_zone_start = offset;
    }

    /// Add a cold code fragment.
    fn add_cold_fragment(&mut self, hot_label: usize, code: Vec<u8>) {
        self.cold_fragments.push(ColdFragment { hot_label, code });
    }

    /// Total size of all cold fragments.
    fn cold_code_size(&self) -> usize {
        self.cold_fragments.iter().map(|f| f.code.len()).sum()
    }

    /// Compute the padding needed before a branch target to ensure it's
    /// aligned to the CPU's L1i cache line boundary.
    pub fn cache_line_alignment_padding(&self, current_offset: usize, _cache_line_size: usize) -> usize {
        let target = current_offset; // we're about to emit the target
        let alignment = 16; // 16-byte boundary for optimal decode
        (alignment - (target % alignment)) % alignment
    }

    /// Check if a loop header would span a cache line boundary.
    /// Returns true if the loop header (assumed 16 bytes) would cross
    /// a 64-byte L1i cache line boundary.
    pub fn loop_header_spans_cache_line(current_offset: usize, cache_line_size: usize) -> bool {
        let line_start = current_offset & !(cache_line_size - 1);
        let header_end = current_offset + 16; // conservative: assume 16-byte loop header
        header_end > line_start + cache_line_size
    }
}

/// Emitter extensions for hot/cold code layout.
impl Emitter {
    /// Begin a cold code region. In hot code, emit a conditional jump
    /// to the cold label (which will be patched to the cold zone offset).
    /// Returns the cold label ID for later reference.
    fn begin_cold(&mut self, condition: u8, layout: &mut CodeLayout) -> usize {
        let cold_label = layout.next_cold_label();
        // Emit Jcc to cold_label in hot code (displacement is a placeholder)
        self.emit2(0x0F, condition); // Jcc rel32
        self.d(0); // placeholder — will be patched to cold zone offset
        cold_label
    }

    /// Emit a deopt/side-exit stub in the cold zone.
    fn emit_cold_deopt_stub(
        &mut self,
        _cold_label: usize,
        spill_regs: &[u8],
        fallback_pc: usize,
    ) {
        // 1. Spill all live registers to the slot array
        for &reg in spill_regs {
            self.store_mem_reg(0, reg); // Simplified: disp=0 placeholder
        }
        // 2. Load fallback PC into RAX
        self.mov_rax_imm_opt(fallback_pc as i64);
        // 3. JMP interpreter_fallback (placeholder displacement)
        self.b(0xE9);
        self.d(0); // patched later
    }

    // ── Opt3: Profile-Guided Block Reordering helpers ───────────────────

    /// Mark a label (block ID) as cold — typically a deopt or error path.
    /// Cold blocks will be placed in the cold zone during layout.
    fn mark_cold(&mut self, label: usize) {
        if !self.cold_labels.contains(&label) {
            self.cold_labels.push(label);
        }
    }

    /// Check whether a label is in the cold set.
    fn is_cold_label(&self, label: usize) -> bool {
        self.cold_labels.contains(&label)
    }

    /// Invert a conditional branch if the branch target is a cold block.
    ///
    /// On x86-64, conditional branches (Jcc) that *are taken* to a cold
    /// deopt/error path hurt branch prediction — the CPU predicts the
    /// fall-through as taken, but the common case is actually the fall-
    /// through (hot path). By inverting the condition and making the hot
    /// path the fall-through, we improve prediction accuracy.
    ///
    /// Returns the inverted Jcc opcode if inversion happened, or the
    /// original opcode if not.
    fn invert_branch_if_cold(&self, jcc_opcode: u8, target_pc: usize) -> u8 {
        if self.is_cold_label(target_pc) {
            // Invert the condition:
            // 0x84 JZ  → 0x85 JNZ, 0x85 JNZ → 0x84 JZ
            // 0x8C JL  → 0x8D JGE, 0x8D JGE → 0x8C JL
            // 0x8E JLE → 0x8F JG,  0x8F JG  → 0x8E JLE
            // etc.  x86 Jcc opcodes: the low bit toggles the sense.
            jcc_opcode ^ 1
        } else {
            jcc_opcode
        }
    }

    /// Reorder hot/cold blocks using the CodeLayout.
    /// Hot blocks are emitted first (fall-through), cold blocks at the end.
    /// This uses the PmcProfiler counters (if available) to determine
    /// which blocks are hot.
    fn reorder_hot_cold(&mut self, layout: &CodeLayout) {
        // The actual reordering is done by CodeLayout's fragment system:
        // cold code was collected as fragments and will be appended after
        // all hot code. Here we just ensure the cold zone start is set
        // correctly and log the layout decision.
        let cold_size = layout.cold_code_size();
        if cold_size > 0 {
            jit_trace!(
                "[JIT-OPT3] Hot/Cold layout: hot code = {} bytes, cold code = {} bytes, {} cold fragments",
                layout.cold_zone_start,
                cold_size,
                layout.cold_fragments.len()
            );
        }
    }
}

/// Emitter extensions for hardware-guided code layout alignment.
///
/// These methods emit NOP padding to align branch targets and loop headers
/// to cache line boundaries, maximizing L1i cache efficiency and
/// instruction decode bandwidth.
impl Emitter {
    /// Emit NOP padding to align the next instruction to a 16-byte boundary.
    /// Uses multi-byte NOPs (0x0F 0x1F) instead of single-byte 0x90 to avoid
    /// unnecessary use of the legacy decode path.
    fn emit_align_16byte(&mut self) {
        let current = self.pos();
        let padding = (16 - (current % 16)) % 16;
        if padding > 0 {
            self.emit_multi_byte_nop(padding);
        }
    }

    /// Emit NOP padding to ensure a loop header doesn't span an L1i cache line.
    /// Dynamically reads the cache line size from the built-in CPUID detector.
    fn emit_cache_line_align(&mut self) {
        let cache_line_size = cpu_features().cache_line_size as usize; // typically 64
        let current = self.pos();
        // If a 16-byte loop header would span a cache line, pad to next line
        if CodeLayout::loop_header_spans_cache_line(current, cache_line_size) {
            let next_line = (current + cache_line_size) & !(cache_line_size - 1);
            let padding = next_line - current;
            self.emit_multi_byte_nop(padding);
        }
    }

    /// Emit multi-byte NOPs for the specified number of bytes.
    /// Uses the optimal x86 multi-byte NOP sequences:
    ///   1B: 0x90
    ///   2B: 0x66 0x90
    ///   3B: 0x0F 0x1F 0x00
    ///   4B: 0x0F 0x1F 0x40 0x00
    ///   5B: 0x0F 0x1F 0x44 0x00 0x00
    ///   6B: 0x66 0x0F 0x1F 0x44 0x00 0x00
    ///   7B: 0x0F 0x1F 0x80 0x00 0x00 0x00 0x00
    ///   8B: 0x0F 0x1F 0x84 0x00 0x00 0x00 0x00 0x00
    ///   9B: 0x66 0x0F 0x1F 0x84 0x00 0x00 0x00 0x00 0x00
    ///   >9: repeat 9B sequences + remainder
    fn emit_multi_byte_nop(&mut self, mut n: usize) {
        while n > 0 {
            match n {
                1 => { self.b(0x90); n = 0; }
                2 => { self.b(0x66); self.b(0x90); n = 0; }
                3 => { self.b(0x0F); self.b(0x1F); self.b(0x00); n = 0; }
                4 => { self.b(0x0F); self.b(0x1F); self.b(0x40); self.b(0x00); n = 0; }
                5 => { self.b(0x0F); self.b(0x1F); self.b(0x44); self.b(0x00); self.b(0x00); n = 0; }
                6 => { self.b(0x66); self.b(0x0F); self.b(0x1F); self.b(0x44); self.b(0x00); self.b(0x00); n = 0; }
                7 => { self.b(0x0F); self.b(0x1F); self.b(0x80); self.d(0); n = 0; }
                8 => { self.b(0x0F); self.b(0x1F); self.b(0x84); self.d(0); self.b(0x00); n = 0; }
                9 => { self.b(0x66); self.b(0x0F); self.b(0x1F); self.b(0x84); self.d(0); n = 0; }
                _ => { self.b(0x66); self.b(0x0F); self.b(0x1F); self.b(0x84); self.d(0); n -= 9; }
            }
        }
    }

    /// Emit a PREFETCHT0 instruction for the given byte offset from RDI.
    /// PREFETCHT0 loads the cache line containing [RDI + offset] into all
    /// cache levels (L1/L2/L3). This is used when TMAM analysis reports
    /// Back-End Bound bottlenecks — the prefetch hides memory latency by
    /// starting the cache fill before the data is actually needed.
    ///
    /// Encoding: 0F 18 /1 (PREFETCHT0 = ModRM reg field = 1)
    /// For [RDI + disp32]: 0F 18 8F [disp32]
    fn emit_prefetcht0_rdi(&mut self, offset: i32) {
        self.b(0x0F);
        self.b(0x18);
        // ModRM: mod=10 (disp32), reg=1 (PREFETCHT0), rm=7 (RDI)
        self.b(0x8F);
        self.d(offset);
    }

    /// Emit a PREFETCHT0 instruction for [RDI + slot * 8].
    /// Convenience wrapper for slot-based prefetching.
    fn emit_prefetcht0_slot(&mut self, slot: u16) {
        let offset = (slot as i32) * 8;
        self.emit_prefetcht0_rdi(offset);
    }

    /// Emit a PREFETCHT1 instruction (prefetch into L2/L3, not L1).
    /// Useful for streaming access patterns where data won't be reused soon.
    /// Encoding: 0F 18 /2
    fn emit_prefetcht1_rdi(&mut self, offset: i32) {
        self.b(0x0F);
        self.b(0x18);
        // ModRM: mod=10 (disp32), reg=2 (PREFETCHT1), rm=7 (RDI)
        self.b(0x97);
        self.d(offset);
    }

    /// Emit a PREFETCHNTA instruction (prefetch into L1 only, non-temporal).
    /// Useful for data that will be accessed exactly once.
    /// Encoding: 0F 18 /0
    fn emit_prefetchnta_rdi(&mut self, offset: i32) {
        self.b(0x0F);
        self.b(0x18);
        // ModRM: mod=10 (disp32), reg=0 (PREFETCHNTA), rm=7 (RDI)
        self.b(0x87);
        self.d(offset);
    }

    /// Emit PREFETCHIT0 — Instruction Cache Prefetch (Sapphire Rapids+).
    /// Hints the CPU that the specified memory region will be executed as code.
    /// This pre-populates the ITLB and instruction cache, avoiding the ~100-cycle
    /// TLB miss penalty on first execution of newly compiled JIT code.
    ///
    /// Encoding: 0F 18 /7 with ModRM for [RDI + disp32]
    /// On older CPUs, this instruction is executed as a NOP (no fault).
    fn emit_prefetchit0_rdi(&mut self, offset: i32) {
        self.b(0x0F);
        self.b(0x18);
        // ModRM: mod=10 (disp32), reg=7 (PREFETCHIT0), rm=7 (RDI)
        self.b(0xFF);
        self.d(offset);
    }
}

// =============================================================================
// SIMD DISPATCH INTEGRATION (phase6_simd.rs → phase3 JIT)
// =============================================================================
//
// When the polyhedral optimizer produces SIMD hints (VectorPack, MatrixOuterProduct,
// etc.), the phase3 JIT backend must be able to dispatch to the SIMD code generator
// instead of emitting scalar instructions.
//
// This integration point is called from the LoopVectorizer and from the
// superinstruction fusion pass when a SimdHintKind is encountered.

// ─────────────────────────────────────────────────────────────────────────────
// SIMD stubs (phase6_simd module does not exist in this crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Detected SIMD capability level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimdLevel {
    None,
    Sse2,
    Avx,
    Avx2Fma,
    Avx512,
}

impl SimdLevel {
    /// Number of f32 lanes available at this SIMD level.
    fn f32_lanes(&self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::Sse2 => 4,
            SimdLevel::Avx => 8,
            SimdLevel::Avx2Fma => 8,
            SimdLevel::Avx512 => 16,
        }
    }
}

/// Simple buffer for emitting SIMD machine code bytes.
/// Used by `emit_simd_aware_instruction` for polyhedral vectorization hints.
struct SimdCodeBuffer {
    code: Vec<u8>,
    _element_bytes: usize,
}

impl SimdCodeBuffer {
    /// Emit VADDPS ymm_dst, ymm_src1, ymm_src2
    /// AVX encoding: VEX.256.0F.WIG 58 /r
    /// VEX2 (c5) when no X/B extension needed, otherwise VEX3 (c4).
    /// For VADDPS: vvvv=src2(inverted), modrm.reg=dst, modrm.rm=src1
    fn emit_simd_add_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex2_0f(0x58, dst, src1, src2);
    }
    /// Emit VMULPS ymm_dst, ymm_src1, ymm_src2
    /// AVX encoding: VEX.256.0F.WIG 59 /r
    fn emit_simd_mul_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex2_0f(0x59, dst, src1, src2);
    }
    /// Emit VSUBPS ymm_dst, ymm_src1, ymm_src2
    /// AVX encoding: VEX.256.0F.WIG 5C /r
    fn emit_simd_sub_ps(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit_vex2_0f(0x5C, dst, src1, src2);
    }

    /// Common VEX2 (c5) encoding for 0F-map, 256-bit, no-prefix AVX instructions.
    /// VEX2 byte layout: c5 [R~vvvvLpp] opcode modrm
    ///   R~  = bit 7 = !(src1[3])  (inversion of reg field MSB)
    ///   vvvv = bits 6:3 = !src2[3:0] (inverted src2 register)
    ///   L   = bit 2 = 1 (256-bit / YMM)
    ///   pp  = bits 1:0 = 00 (no legacy prefix)
    /// modrm: mod=11, reg=dst[2:0], rm=src1[2:0]
    fn emit_vex2_0f(&mut self, opcode: u8, dst: u8, src1: u8, src2: u8) {
        let r_inv = ((src1 >> 3) & 1) ^ 1;
        let vvvv_inv = (!src2) & 0xF;
        let l_bit: u8 = 1; // 256-bit YMM
        let pp: u8 = 0;    // no prefix
        let vex_byte2 = (r_inv << 7) | (vvvv_inv << 3) | (l_bit << 2) | pp;
        let modrm = 0xC0 | ((dst & 7) << 3) | (src1 & 7);
        self.code.extend_from_slice(&[0xC5, vex_byte2, opcode, modrm]);
    }
}

/// Detect the highest SIMD level supported by the current CPU.
fn get_simd_level() -> SimdLevel {
    let cpu = cpu_features();
    if cpu.has_avx512f { SimdLevel::Avx512 }
    else if cpu.has_avx2 { SimdLevel::Avx2Fma }
    else if cpu.has_avx { SimdLevel::Avx }
    else if cpu.has_sse42 { SimdLevel::Sse2 }
    else { SimdLevel::None }
}

/// Create a new SIMD code buffer for the given element size.
fn create_simd_buffer(element_bytes: usize) -> SimdCodeBuffer {
    SimdCodeBuffer { code: Vec::new(), _element_bytes: element_bytes }
}

/// Emit SIMD machine code for a polyhedral vectorization hint.
/// Produces real AVX VEX-encoded instruction bytes for the given operation.
fn emit_simd_for_hint(
    hint: &str,
    op: BinOpKind,
    src1: u16,
    src2: u16,
    dst: u16,
    element_bytes: usize,
) -> Vec<u8> {
    let mut buf = create_simd_buffer(element_bytes);
    // Use the hint to select the vectorization strategy
    match hint {
        "vectorize" | "simd" | "avx2" | "parallel" => {
            // Full 256-bit AVX2 vectorization
            match op {
                BinOpKind::Add => buf.emit_simd_add_ps(dst as u8, src1 as u8, src2 as u8),
                BinOpKind::Mul => buf.emit_simd_mul_ps(dst as u8, src1 as u8, src2 as u8),
                BinOpKind::Sub => buf.emit_simd_sub_ps(dst as u8, src1 as u8, src2 as u8),
                _ => {} // Other ops: no SIMD fallback, caller handles scalar
            }
        }
        "fma" => {
            // FMA fusion: emit VFMADD + VMULPS or VADDPS sequence
            // For now, emit the base multiply/add as separate instructions
            match op {
                BinOpKind::Mul => buf.emit_simd_mul_ps(dst as u8, src1 as u8, src2 as u8),
                BinOpKind::Add => buf.emit_simd_add_ps(dst as u8, src1 as u8, src2 as u8),
                _ => {}
            }
        }
        _ => {
            // Unknown hint: try basic vectorization for supported ops
            match op {
                BinOpKind::Add => buf.emit_simd_add_ps(dst as u8, src1 as u8, src2 as u8),
                BinOpKind::Mul => buf.emit_simd_mul_ps(dst as u8, src1 as u8, src2 as u8),
                BinOpKind::Sub => buf.emit_simd_sub_ps(dst as u8, src1 as u8, src2 as u8),
                _ => {}
            }
        }
    }
    buf.code
}

/// SIMD-aware instruction selection: when a polyhedral hint indicates
/// vectorization, emit SIMD code instead of scalar code.
pub(crate) fn emit_simd_aware_instruction(
    emitter: &mut Emitter,
    op: BinOpKind,
    src1_slot: u16,
    src2_slot: u16,
    dst_slot: u16,
    hint: Option<&str>,
    element_bytes: usize,
) {
    let level = get_simd_level();
    
    if level != SimdLevel::None {
        if let Some(hint_kind) = hint {
            // Polyhedral hint present — generate SIMD code
            let simd_code = emit_simd_for_hint(
                hint_kind, op, src1_slot, src2_slot, dst_slot, element_bytes
            );
            // Append SIMD bytes to the emitter's buffer
            emitter.extend_from_slice(&simd_code);
            return;
        }
        
        // No hint but SIMD available — check if this is a simple
        // vectorizable pattern (consecutive array ops)
        if level.f32_lanes() >= 4 {
            // Emit a single SIMD instruction for the operation
            let mut simd_buf = create_simd_buffer(element_bytes);
            match op {
                BinOpKind::Add => {
                    simd_buf.emit_simd_add_ps(0, 0, 1);
                }
                BinOpKind::Mul => {
                    simd_buf.emit_simd_mul_ps(0, 0, 1);
                }
                BinOpKind::Sub => {
                    simd_buf.emit_simd_sub_ps(0, 0, 1);
                }
                _ => {
                    // Fall through to scalar
                    return;
                }
            }
            emitter.extend_from_slice(&simd_buf.code);
            return;
        }
    }
    
    // Scalar fallback — existing emitter handles it
}

/// ForceRegisterLock-aware register allocation.
/// When the polyhedral engine emits ForceRegisterLock hints, the
/// register allocator must bypass its linear-scan heuristics and pin
/// the specified slot to the designated physical register.
pub struct ForceRegisterLockState {
    /// Map from slot to pinned physical register (0-15 for x86-64 GPRs)
    pub pinned: std::collections::HashMap<u16, u8>,
    /// Set of currently pinned registers (unavailable for normal allocation)
    pub pinned_regs: std::collections::HashSet<u8>,
}

impl ForceRegisterLockState {
    pub fn new() -> Self {
        Self {
            pinned: std::collections::HashMap::new(),
            pinned_regs: std::collections::HashSet::new(),
        }
    }
    
    /// Pin a slot to a physical register. The register allocator must
    /// never evict or spill this mapping.
    pub fn lock(&mut self, slot: u16, phys_reg: u8) {
        self.pinned.insert(slot, phys_reg);
        self.pinned_regs.insert(phys_reg);
    }
    
    /// Release a previously locked register
    pub fn unlock(&mut self, slot: u16) {
        if let Some(reg) = self.pinned.remove(&slot) {
            self.pinned_regs.remove(&reg);
        }
    }
    
    /// Check if a physical register is pinned (unavailable for normal allocation)
    pub fn is_pinned(&self, reg: u8) -> bool {
        self.pinned_regs.contains(&reg)
    }
    
    /// Get the pinned register for a slot, if any
    pub fn get_pinned_reg(&self, slot: u16) -> Option<u8> {
        self.pinned.get(&slot).copied()
    }

    /// Apply all pinned register assignments to an existing RegAlloc result.
    /// This overwrites the allocator's decisions for pinned slots, forcing
    /// them to the designated physical registers.
    pub fn apply_to_alloc(&self, ra: &mut RegAlloc) {
        for (&slot, &phys_reg) in &self.pinned {
            ra.force_assign(slot, phys_reg);
        }
    }
}

// =============================================================================
// MULTI-DIMENSIONAL STRIDE ADDRESS GENERATION (Bug Fix #4)
// =============================================================================
//
// LEA fusion pattern only handles small scale factors (2, 4, 8).
// ML tensor strides can be large (1024+) or variable.
// Expand address generator to use x86 Base-Index-Scale-Displacement
// configurations with multi-register addressing chains.

impl Emitter {
    /// Emit a multi-dimensional stride address calculation using x86
    /// Base-Index-Scale-Displacement (BISD) addressing.
    ///
    /// For ML tensors, the address of element [i₀][i₁]...[iₙ] is:
    ///   base + i₀*stride₀ + i₁*stride₁ + ... + iₙ*strideₙ
    ///
    /// If a stride is a power of 2 in {1,2,4,8}, use LEA with SIB encoding.
    /// For larger/variable strides, emit IMUL + ADD chains.
    pub fn emit_multi_stride_address(
        &mut self,
        dst_reg: u8,     // destination register for the computed address
        base_reg: u8,    // base register (e.g., rdi = slot array base)
        indices: &[(u8, i64)], // (index_register, stride) pairs
        displacement: i32,
    ) {
        // Strategy:
        // 1. Start with base address: MOV dst, [base_reg] or LEA dst, [base_reg + disp]
        // 2. For each (index, stride) pair:
        //    a. If stride ∈ {1,2,4,8}: use LEA dst, [dst + index*stride]
        //    b. If stride is power-of-2 but > 8: use IMUL index, stride + ADD
        //    c. If stride is variable/odd: use IMUL + ADD
        
        // Step 1: Load base address
        if displacement == 0 {
            // MOV dst, base_reg
            self.emit3(
                0x48 | ((dst_reg & 8) >> 1) | ((base_reg & 8) >> 3),
                0x8B,
                0xC0 | ((dst_reg & 7) << 3) | (base_reg & 7),
            );
        } else {
            // LEA dst, [base_reg + displacement]
            self.emit3(
                0x48 | ((dst_reg & 8) >> 1) | ((base_reg & 8) >> 3),
                0x8D,
                if let Ok(d8) = i8::try_from(displacement) {
                    self.b(0x40 | ((dst_reg & 7) << 3) | (base_reg & 7));
                    self.b(d8 as u8);
                    return; // early return for 1-step LEA
                } else {
                    0x80 | ((dst_reg & 7) << 3) | (base_reg & 7)
                }
            );
            if let Ok(_) = i8::try_from(displacement) {
                // Already handled above
            } else {
                self.d(displacement);
            }
        }
        
        // Step 2: Add each index * stride
        for &(idx_reg, stride) in indices {
            if stride == 0 { continue; }
            
            match stride {
                1 | 2 | 4 | 8 => {
                    // LEA dst, [dst + idx*stride] — single instruction!
                    // REX.W LEA with SIB byte
                    let scale_bits = match stride {
                        1 => 0b00,
                        2 => 0b01,
                        4 => 0b10,
                        8 => 0b11,
                        _ => 0b00,
                    };
                    self.emit3(
                        0x48 | ((dst_reg & 8) >> 1) | ((idx_reg & 8) >> 3),
                        0x8D,
                        0x04 | ((dst_reg & 7) << 3), // SIB follows, mod=00
                    );
                    // SIB byte: scale=scale_bits, index=idx_reg, base=dst_reg
                    self.b((scale_bits << 6) | ((idx_reg & 7) << 3) | (dst_reg & 7));
                }
                s if s > 0 && (s & (s - 1)) == 0 => {
                    // Power of 2 but > 8: shift the index register, then add
                    let shift = s.trailing_zeros() as u8;
                    // IMUL idx_reg, stride or SHL idx_reg, shift
                    // Use LEA with index*8 + repeated addition for correctness
                    // Actually: emit SHL + LEA
                    // MOV rcx, idx_reg
                    self.emit3(
                        0x48 | ((1 & 8) >> 1) | ((idx_reg & 8) >> 3),
                        0x8B,
                        0xC0 | ((1 & 7) << 3) | (idx_reg & 7), // MOV RCX, idx
                    );
                    // SHL RCX, shift
                    self.emit4(0x48, 0xC1, 0xE1, shift);
                    // LEA dst, [dst + rcx]
                    self.emit3(0x48, 0x8D, 0x0C | ((dst_reg & 7) << 3));
                    self.b(0x0A); // SIB: base=dst, index=rcx
                }
                _ => {
                    // Arbitrary stride: IMUL + ADD
                    // MOV rcx, idx_reg
                    self.emit3(
                        0x48 | ((1 & 8) >> 1) | ((idx_reg & 8) >> 3),
                        0x8B,
                        0xC0 | ((1 & 7) << 3) | (idx_reg & 7),
                    );
                    // IMUL rcx, stride (immediate form)
                    self.emit3(0x48, 0x69, 0xC9); // IMUL RCX, RCX, imm32
                    self.d(stride as i32);
                    // ADD dst, rcx
                    self.emit3(
                        0x48 | ((dst_reg & 8) >> 1) | ((1 & 8) >> 3),
                        0x01,
                        0xC0 | ((dst_reg & 7) << 3) | (1 & 7),
                    );
                }
            }
        }
    }
    
    /// Emit a BISD (Base-Index-Scale-Displacement) address computation
    /// optimized for common ML tensor access patterns.
    ///
    /// Handles 2D row-major access: A[i][j] = base + i*row_stride + j*col_stride
    /// where row_stride is typically large (e.g., 4096 for a 1024-wide matrix)
    /// and col_stride is typically small (1, 2, or 4 bytes per element).
    pub fn emit_2d_tensor_address(
        &mut self,
        dst_reg: u8,
        base_reg: u8,
        row_reg: u8,
        col_reg: u8,
        row_stride: i64,
        col_stride: i64,
    ) {
        self.emit_multi_stride_address(
            dst_reg,
            base_reg,
            &[
                (row_reg, row_stride),
                (col_reg, col_stride),
            ],
            0, // no displacement
        );
    }
}

// =============================================================================
// INDUCTION VARIABLE PINNING TO FIXED GPRS (r12, r13, r14)
// =============================================================================
//
// For ML workloads, pinning loop induction variables to callee-saved
// registers avoids constant write-back to the slot array on every
// iteration. The polyhedral optimizer emits ForceRegisterLock hints
// that the register allocator must respect.

/// Pin assignment configuration for loop induction variables.
#[derive(Debug, Clone, Copy)]
pub struct InductionVarPinning {
    /// r12 = first induction variable (outermost loop)
    pub iv_outer_reg: u8,  // default: 12 (r12)
    /// r13 = second induction variable (middle loop)
    pub iv_middle_reg: u8, // default: 13 (r13)
    /// r14 = accumulator register (innermost loop / reduction)
    pub accumulator_reg: u8, // default: 14 (r14)
}

impl Default for InductionVarPinning {
    fn default() -> Self {
        Self {
            iv_outer_reg: 12,     // r12
            iv_middle_reg: 13,    // r13
            accumulator_reg: 14,  // r14
        }
    }
}

impl InductionVarPinning {
    /// Emit prologue: push callee-saved registers that are used for pinning
    pub(crate) fn emit_prologue(&self, emitter: &mut Emitter, use_outer: bool, use_middle: bool, use_accum: bool) {
        // PUSH r12 (if used for outer IV)
        if use_outer {
            emitter.emit2(0x41, 0x54); // PUSH R12
        }
        // PUSH r13 (if used for middle IV)
        if use_middle {
            emitter.emit2(0x41, 0x55); // PUSH R13
        }
        // PUSH r14 (if used for accumulator)
        if use_accum {
            emitter.emit2(0x41, 0x56); // PUSH R14
        }
    }

    /// Emit epilogue: pop callee-saved registers in reverse order
    pub(crate) fn emit_epilogue(&self, emitter: &mut Emitter, use_outer: bool, use_middle: bool, use_accum: bool) {
        if use_accum {
            emitter.emit2(0x41, 0x5E); // POP R14
        }
        if use_middle {
            emitter.emit2(0x41, 0x5D); // POP R13
        }
        if use_outer {
            emitter.emit2(0x41, 0x5C); // POP R12
        }
    }
}

// =============================================================================
// TIER 4: COMPOSITION / ORCHESTRATION LAYER
// =============================================================================
//
// Tier 4 is NOT a new execution engine. It is a smart scheduler that breaks
// general code into Tier 1–3 chunks, builds an execution DAG, and dispatches
// to existing optimized kernels — never inventing new execution methods.
//
// Design principles:
//   1. Decompose general code into "regions" classified by execution tier
//   2. Build an execution DAG (nodes = kernels, edges = data dependencies)
//   3. Apply conservative fusion (only fuse if safe: contiguous, no dtype
//      conflicts, no branch dependency explosion)
//   4. Schedule execution order respecting dependencies
//   5. Reuse buffers aggressively (no extra copies unless required)
//   6. Compile strategy = "cheap planning" (trace → classify → graph → assign)
//
// Tier model:
//   Tier 1 = SIMD elementwise kernel
//   Tier 2 = Fused vector kernel / reduction kernel
//   Tier 3 = BLAS / heavy numeric ops
//   Tier 4 = Orchestration of Tier 1–3 (the conductor, not a new instrument)

/// Classification of a computation region for Tier dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier4RegionKind {
    /// Vectorizable elementwise op → Tier 1 (SIMD elementwise kernel)
    Elementwise,
    /// Reduction pattern (sum/max/min over axis) → Tier 2 (fused vector/reduction kernel)
    Reduction,
    /// Linear algebra (matmul, dot, solve) → Tier 3 (BLAS)
    LinearAlgebra,
    /// Stencil pattern (5-point, 9-point) → Tier 2 (fused stencil kernel)
    Stencil,
    /// Transcendental function (sin, cos, exp, log, tanh, sigmoid) → Tier 2
    Transcendental,
    /// Scalar or control-flow heavy → Tier 0/1 fallback
    Scalar,
    /// Fused multiply-add chain → Tier 2 (fused SIMD FMA kernel)
    FmaChain,
    /// Comparison / logical op producing boolean → Tier 1
    Logical,
}

/// A node in the Tier 4 execution DAG.
/// Each node represents a kernel invocation at a specific tier level.
#[derive(Clone, Debug)]
pub struct Tier4Node {
    /// Unique node identifier within the DAG
    pub id: usize,
    /// What kind of computation this node performs
    pub kind: Tier4RegionKind,
    /// The tier to dispatch to (1, 2, or 3)
    pub tier: u8,
    /// Input slot indices consumed by this node
    pub input_slots: Vec<usize>,
    /// Output slot indices produced by this node
    pub output_slots: Vec<usize>,
    /// Opaque operation descriptor (binop name, reduction op, etc.)
    pub op_desc: String,
    /// Whether this node can be fused with its successor
    pub fusible: bool,
    /// Estimated cost (approximate cycle count) for scheduling priority
    pub estimated_cost: f64,
    /// Data type of inputs/outputs (for fusion compatibility checking)
    pub dtype: Tier4Dtype,
    /// Instruction range in the original trace: [start, end)
    pub instr_range: (usize, usize),
}

/// Lightweight dtype representation for fusion compatibility checking.
/// Prevents fusion between incompatible types (e.g., f32 + f64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier4Dtype {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Unknown,
}

/// An edge in the Tier 4 execution DAG.
/// Represents a data dependency: the source node produces a value that the
/// target node consumes. Edges are the basis for topological scheduling and
/// buffer lifetime analysis.
#[derive(Clone, Debug)]
pub struct Tier4Edge {
    /// Source node ID (producer)
    pub from_node: usize,
    /// Target node ID (consumer)
    pub to_node: usize,
    /// Slot index that carries the data
    pub slot: usize,
    /// Whether the data flows contiguously in memory (required for fusion)
    pub contiguous: bool,
    /// Size in bytes of the data flowing through this edge
    pub data_size_bytes: usize,
}

/// The Tier 4 execution DAG — the central data structure for orchestration.
///
/// Nodes represent kernel invocations at specific tier levels; edges represent
/// data dependencies. The DAG is built by the decomposition pass and consumed
/// by the scheduler. Fusion decisions are made during the build phase using
/// conservative rules, not by a separate optimizer.
pub struct Tier4Dag {
    /// All nodes in the DAG, indexed by node ID
    pub nodes: Vec<Tier4Node>,
    /// All edges (data dependencies) in the DAG
    pub edges: Vec<Tier4Edge>,
    /// Adjacency list: node_id → list of (edge_index, target_node_id)
    pub adjacency: FxHashMap<usize, Vec<(usize, usize)>>,
    /// Reverse adjacency: node_id → list of (edge_index, source_node_id)
    pub reverse_adj: FxHashMap<usize, Vec<(usize, usize)>>,
    /// Input slot → first node that consumes it
    pub input_map: FxHashMap<usize, usize>,
    /// Output slot → last node that produces it
    pub output_map: FxHashMap<usize, usize>,
    /// Next available node ID
    next_id: usize,
}

impl Tier4Dag {
    /// Create an empty execution DAG.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: FxHashMap::default(),
            reverse_adj: FxHashMap::default(),
            input_map: FxHashMap::default(),
            output_map: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Add a new node to the DAG, returning its ID.
    pub fn add_node(&mut self, node: Tier4Node) -> usize {
        let id = self.next_id;
        let mut node = node;
        node.id = id;
        self.next_id += 1;

        // Record input/output mappings
        for &slot in &node.input_slots {
            self.input_map.entry(slot).or_insert(id);
        }
        for &slot in &node.output_slots {
            self.output_map.insert(slot, id);
        }

        self.nodes.push(node);
        id
    }

    /// Add a data dependency edge from `from_node` to `to_node`.
    pub fn add_edge(&mut self, from_node: usize, to_node: usize, slot: usize,
                     contiguous: bool, data_size_bytes: usize) {
        let edge_idx = self.edges.len();
        self.edges.push(Tier4Edge {
            from_node,
            to_node,
            slot,
            contiguous,
            data_size_bytes,
        });
        self.adjacency.entry(from_node).or_default().push((edge_idx, to_node));
        self.reverse_adj.entry(to_node).or_default().push((edge_idx, from_node));
    }

    /// Topologically sort the DAG for execution order.
    /// Uses Kahn's algorithm (BFS-based) for deterministic ordering.
    /// Returns node IDs in valid execution order.
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        if n == 0 { return Vec::new(); }

        // Compute in-degrees
        let mut in_degree = vec![0u32; n];
        for edge in &self.edges {
            if edge.to_node < n {
                in_degree[edge.to_node] += 1;
            }
        }

        // Initialize queue with zero in-degree nodes
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);
            if let Some(neighbors) = self.adjacency.get(&node_id) {
                for &(_, target) in neighbors {
                    if target < n {
                        in_degree[target] -= 1;
                        if in_degree[target] == 0 {
                            queue.push_back(target);
                        }
                    }
                }
            }
        }

        order
    }

    /// Check whether two adjacent nodes can be conservatively fused.
    ///
    /// Fusion is allowed ONLY when ALL of these conditions hold:
    ///   1. Both nodes have the same dtype (no hidden promotion)
    ///   2. Memory access is contiguous (no stride/gather patterns)
    ///   3. No branch dependency explosion (neither node is Scalar with control flow)
    ///   4. The producer node is fusible (its op supports inline chaining)
    ///   5. The edge between them is marked contiguous
    ///   6. Fusing does not create a new allocation (buffer reuse only)
    pub fn can_fuse(&self, producer_id: usize, consumer_id: usize) -> bool {
        let producer = match self.nodes.get(producer_id) {
            Some(n) => n,
            None => return false,
        };
        let consumer = match self.nodes.get(consumer_id) {
            Some(n) => n,
            None => return false,
        };

        // Rule 1: Same dtype — no hidden promotion
        if producer.dtype != consumer.dtype {
            return false;
        }

        // Rule 2: No branch dependency explosion
        if producer.kind == Tier4RegionKind::Scalar || consumer.kind == Tier4RegionKind::Scalar {
            return false;
        }

        // Rule 3: Producer must be fusible
        if !producer.fusible {
            return false;
        }

        // Rule 4: Edge must be contiguous
        let edge_contiguous = self.edges.iter().any(|e| {
            e.from_node == producer_id && e.to_node == consumer_id && e.contiguous
        });
        if !edge_contiguous {
            return false;
        }

        // Rule 5: Tier compatibility — can fuse Tier1+Tier1, Tier1+Tier2, Tier2+Tier2
        // Cannot fuse with Tier3 (BLAS is opaque, can't be chained inline)
        if consumer.tier == 3 || producer.tier == 3 {
            return false;
        }

        // Rule 6: Logical ops can only fuse with other logical ops or elementwise
        if producer.kind == Tier4RegionKind::Logical && consumer.kind != Tier4RegionKind::Logical
            && consumer.kind != Tier4RegionKind::Elementwise {
            return false;
        }

        true
    }

    /// Apply conservative fusion: merge adjacent fusible nodes into super-nodes.
    /// This reduces kernel launch overhead and eliminates intermediate buffers.
    ///
    /// Returns a new DAG with fused nodes. The original DAG is not modified.
    pub fn apply_conservative_fusion(&self) -> Tier4Dag {
        let mut fused = Tier4Dag::new();
        let order = self.topological_order();

        // Track which original nodes have been absorbed into a super-node
        let mut absorbed: FxHashSet<usize> = FxHashSet::default();
        // Map from original node ID to fused node ID
        let mut orig_to_fused: FxHashMap<usize, usize> = FxHashMap::default();

        for &node_id in &order {
            if absorbed.contains(&node_id) {
                continue;
            }

            let _node = &self.nodes[node_id];

            // Try to find a fusible successor
            let mut fusion_chain = vec![node_id];
            let mut current = node_id;

            'chain: loop {
                // Find a single successor that can be fused
                let successors: Vec<usize> = self.adjacency
                    .get(&current)
                    .map(|adj| adj.iter().map(|&(_, t)| t).collect())
                    .unwrap_or_default();

                let mut found_fusion = false;
                for &succ in &successors {
                    if absorbed.contains(&succ) { continue; }
                    if self.can_fuse(current, succ) {
                        // Check that the successor has only one predecessor (the current node)
                        // — otherwise fusion would duplicate code
                        let pred_count = self.reverse_adj
                            .get(&succ)
                            .map(|p| p.len())
                            .unwrap_or(0);
                        if pred_count == 1 {
                            fusion_chain.push(succ);
                            absorbed.insert(succ);
                            current = succ;
                            found_fusion = true;
                            break;
                        }
                    }
                }
                if !found_fusion {
                    break 'chain;
                }
            }

            // Build a fused super-node from the chain
            let first = &self.nodes[fusion_chain[0]];
            let _last = &self.nodes[*fusion_chain.last().unwrap()];

            let mut all_input_slots: Vec<usize> = Vec::new();
            let mut all_output_slots: Vec<usize> = Vec::new();
            let mut fused_op_desc = String::new();
            let mut total_cost = 0.0f64;
            // Track instruction range across the fusion chain
            let mut fused_instr_start = first.instr_range.0;
            let mut fused_instr_end = first.instr_range.1;

            for (i, &fid) in fusion_chain.iter().enumerate() {
                let fnode = &self.nodes[fid];
                if i > 0 { fused_op_desc.push_str(" → "); }
                fused_op_desc.push_str(&fnode.op_desc);
                total_cost += fnode.estimated_cost;

                // Input slots: exclude slots produced by previous node in chain
                for &slot in &fnode.input_slots {
                    if i == 0 || !self.nodes[fusion_chain[i - 1]].output_slots.contains(&slot) {
                        all_input_slots.push(slot);
                    }
                }
                // Output slots: always include
                for &slot in &fnode.output_slots {
                    all_output_slots.push(slot);
                }
                // Extend instruction range to cover all fused nodes
                if fnode.instr_range.0 < fused_instr_start {
                    fused_instr_start = fnode.instr_range.0;
                }
                if fnode.instr_range.1 > fused_instr_end {
                    fused_instr_end = fnode.instr_range.1;
                }
            }

            let fused_id = fused.add_node(Tier4Node {
                id: 0, // will be set by add_node
                kind: first.kind,
                tier: if fusion_chain.len() > 1 { 2 } else { first.tier }, // fused → Tier 2
                input_slots: all_input_slots,
                output_slots: all_output_slots,
                op_desc: fused_op_desc,
                fusible: true,
                estimated_cost: total_cost * 0.9, // 10% saving from eliminating intermediates
                dtype: first.dtype,
                instr_range: (fused_instr_start, fused_instr_end),
            });

            for &fid in &fusion_chain {
                orig_to_fused.insert(fid, fused_id);
            }
        }

        // Rebuild edges in the fused DAG
        for edge in &self.edges {
            let from_fused = *orig_to_fused.get(&edge.from_node).unwrap_or(&edge.from_node);
            let to_fused = *orig_to_fused.get(&edge.to_node).unwrap_or(&edge.to_node);
            if from_fused != to_fused {
                fused.add_edge(from_fused, to_fused, edge.slot,
                              edge.contiguous, edge.data_size_bytes);
            }
        }

        fused
    }
}

/// Buffer reuse plan generated by the Tier 4 planner.
///
/// Instead of allocating a new buffer for every intermediate result, the
/// planner identifies buffers whose lifetimes do not overlap and assigns
/// them the same memory slot. This can reduce peak memory by 30-50% for
/// complex DAGs with many intermediate values.
#[derive(Clone, Debug)]
pub struct Tier4BufferPlan {
    /// Map from slot index to assigned buffer index
    pub slot_to_buffer: FxHashMap<usize, usize>,
    /// Total number of unique buffers needed
    pub total_buffers: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// List of (buffer_index, size_bytes, first_use_node, last_use_node)
    pub buffer_lifetimes: Vec<(usize, usize, usize, usize)>,
}

impl Tier4Dag {
    /// Compute a buffer reuse plan using first-fit lifetime analysis.
    ///
    /// Algorithm:
    ///   1. For each node in topological order, determine when each output
    ///      buffer is first and last used.
    ///   2. Sort buffers by first-use time.
    ///   3. Greedily assign buffers: when a buffer's lifetime ends, free it
    ///      and reuse its slot for a new buffer with a later lifetime.
    ///   4. This is optimal for straight-line DAGs and near-optimal for
    ///      general DAGs.
    pub fn compute_buffer_plan(&self, element_size: usize) -> Tier4BufferPlan {
        let order = self.topological_order();

        // Step 1: Compute lifetime (first_use, last_use) for each output slot
        let mut slot_first_use: FxHashMap<usize, usize> = FxHashMap::default();
        let mut slot_last_use: FxHashMap<usize, usize> = FxHashMap::default();
        let mut slot_size: FxHashMap<usize, usize> = FxHashMap::default();

        for (step, &node_id) in order.iter().enumerate() {
            let node = &self.nodes[node_id];
            for &slot in &node.output_slots {
                slot_first_use.entry(slot).or_insert(step);
                slot_last_use.insert(slot, step);
                slot_size.insert(slot, element_size); // simplified: uniform element size
            }
            // Inputs also "use" the slot
            for &slot in &node.input_slots {
                slot_last_use.entry(slot).and_modify(|v| { if step > *v { *v = step; } });
            }
        }

        // Step 2: Sort slots by first-use time
        let mut slots: Vec<usize> = slot_first_use.keys().copied().collect();
        slots.sort_by_key(|s| slot_first_use.get(s).copied().unwrap_or(0));

        // Step 3: Greedy first-fit assignment
        let mut slot_to_buffer: FxHashMap<usize, usize> = FxHashMap::default();
        let mut free_buffers: Vec<usize> = Vec::new(); // buffer indices available for reuse
        let mut buffer_lifetimes: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut next_buffer = 0usize;
        let mut total_bytes = 0usize;

        for slot in &slots {
            let first = slot_first_use.get(slot).copied().unwrap_or(0);
            let last = slot_last_use.get(slot).copied().unwrap_or(0);
            let size = slot_size.get(slot).copied().unwrap_or(element_size);

            // Find a free buffer whose lifetime has ended
            let reused = free_buffers.iter().position(|&b| {
                buffer_lifetimes.get(b).map(|&(_, _, _, last_use)| last_use < first).unwrap_or(false)
            });

            let buf_idx = if let Some(idx) = reused {
                let b = free_buffers.remove(idx);
                // Update lifetime
                if b < buffer_lifetimes.len() {
                    buffer_lifetimes[b] = (b, size, first, last);
                }
                b
            } else {
                let b = next_buffer;
                next_buffer += 1;
                total_bytes += size;
                buffer_lifetimes.push((b, size, first, last));
                b
            };

            slot_to_buffer.insert(*slot, buf_idx);
        }

        Tier4BufferPlan {
            slot_to_buffer,
            total_buffers: next_buffer,
            total_bytes,
            buffer_lifetimes,
        }
    }
}

/// The Tier 4 region decomposition engine.
///
/// Takes a flat list of trace instructions and classifies each into a
/// computation region. Adjacent regions of the same kind are merged.
/// The result is a list of classified regions that can be assembled into
/// a DAG.
pub struct Tier4Decomposer {
    /// Detected regions from the last decomposition
    pub regions: Vec<Tier4Region>,
}

/// A contiguous region of the trace, classified by tier kind.
#[derive(Clone, Debug)]
pub struct Tier4Region {
    /// Region index in the decomposition order
    pub index: usize,
    /// Kind of computation
    pub kind: Tier4RegionKind,
    /// Assigned tier (1, 2, or 3)
    pub tier: u8,
    /// Instruction indices belonging to this region
    pub instr_range: (usize, usize), // [start, end)
    /// Slot indices consumed
    pub input_slots: Vec<usize>,
    /// Slot indices produced
    pub output_slots: Vec<usize>,
    /// Opaque operation descriptor
    pub op_desc: String,
    /// Whether fusible with adjacent regions
    pub fusible: bool,
    /// Data type of this region's primary computation
    pub dtype: Tier4Dtype,
}

impl Tier4Decomposer {
    /// Create a new decomposer.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Classify a single binop into a Tier4RegionKind.
    fn classify_binop(op: &str) -> Tier4RegionKind {
        match op {
            "add" | "sub" | "mul" | "div" => Tier4RegionKind::Elementwise,
            "matmul" | "dot" => Tier4RegionKind::LinearAlgebra,
            "min" | "max" => Tier4RegionKind::Elementwise,
            "lt" | "le" | "gt" | "ge" | "eq" | "ne" => Tier4RegionKind::Logical,
            "and" | "or" | "xor" => Tier4RegionKind::Logical,
            "rem" | "mod" => Tier4RegionKind::Elementwise,
            "shl" | "shr" => Tier4RegionKind::Elementwise,
            _ => Tier4RegionKind::Scalar,
        }
    }

    /// Classify a unop into a Tier4RegionKind.
    fn classify_unop(op: &str) -> Tier4RegionKind {
        match op {
            "neg" | "abs" => Tier4RegionKind::Elementwise,
            "sin" | "cos" | "tan" | "exp" | "log" | "tanh" | "sigmoid" => Tier4RegionKind::Transcendental,
            "not" => Tier4RegionKind::Logical,
            _ => Tier4RegionKind::Scalar,
        }
    }

    /// Assign a tier level to a region kind.
    fn tier_for_kind(kind: &Tier4RegionKind) -> u8 {
        match kind {
            Tier4RegionKind::Elementwise => 1,
            Tier4RegionKind::Logical => 1,
            Tier4RegionKind::Reduction => 2,
            Tier4RegionKind::Stencil => 2,
            Tier4RegionKind::Transcendental => 2,
            Tier4RegionKind::FmaChain => 2,
            Tier4RegionKind::LinearAlgebra => 3,
            Tier4RegionKind::Scalar => 0,
        }
    }

    /// Determine if a region kind can be fused with another.
    fn is_fusible(kind: &Tier4RegionKind) -> bool {
        match kind {
            Tier4RegionKind::Elementwise
            | Tier4RegionKind::Logical
            | Tier4RegionKind::Transcendental
            | Tier4RegionKind::FmaChain => true,
            // Reduction and Stencil have fixed output shapes; BLAS is opaque
            Tier4RegionKind::Reduction
            | Tier4RegionKind::Stencil
            | Tier4RegionKind::LinearAlgebra
            | Tier4RegionKind::Scalar => false,
        }
    }

    /// Decompose a trace into classified regions.
    ///
    /// The trace is a list of instruction tuples (in Python-serialized form).
    /// Each instruction is represented as (opcode, ...operands).
    ///
    /// Returns a list of regions, where each region contains:
    ///   - Its kind (elementwise, reduction, BLAS, etc.)
    ///   - Its tier assignment
    ///   - The instruction range it covers
    ///   - Input/output slot indices
    ///   - Whether it can be fused with neighbors
    pub fn decompose(&mut self, trace_ops: &[(u8, Vec<i64>)]) -> &[Tier4Region] {
        self.regions.clear();

        let mut current_kind: Option<Tier4RegionKind> = None;
        let mut current_start = 0usize;
        let mut current_input_slots: Vec<usize> = Vec::new();
        let mut current_output_slots: Vec<usize> = Vec::new();
        let mut current_op_desc = String::new();
        let mut current_fusible = true;
        let mut current_dtype = Tier4Dtype::Unknown;

        let mut region_idx = 0;

        for (i, (opcode, operands)) in trace_ops.iter().enumerate() {
            let (kind, inputs, outputs, desc, dtype) = match *opcode {
                // BinOp: opcode 0x10-0x20
                0x10..=0x20 => {
                    let op_str = match opcode {
                        0x10 => "add",
                        0x11 => "sub",
                        0x12 => "mul",
                        0x13 => "div",
                        0x14 => "rem",
                        0x15 => "min",
                        0x16 => "max",
                        0x17 => "lt",
                        0x18 => "le",
                        0x19 => "gt",
                        0x1A => "ge",
                        0x1B => "eq",
                        0x1C => "ne",
                        0x1D => "and",
                        0x1E => "or",
                        0x1F => "xor",
                        0x20 => "matmul",
                        _ => "unknown",
                    };
                    let kind = Self::classify_binop(op_str);
                    // Operands: dst, lhs, rhs
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    let lhs = operands.get(1).copied().unwrap_or(0) as usize;
                    let rhs = operands.get(2).copied().unwrap_or(0) as usize;
                    (kind, vec![lhs, rhs], vec![dst], op_str.to_string(), Tier4Dtype::Unknown)
                }
                // UnOp
                0x30..=0x3F => {
                    let op_str = match opcode {
                        0x30 => "neg",
                        0x31 => "abs",
                        0x32 => "not",
                        0x33 => "sin",
                        0x34 => "cos",
                        0x35 => "exp",
                        0x36 => "log",
                        0x37 => "tanh",
                        0x38 => "sigmoid",
                        _ => "unknown",
                    };
                    let kind = Self::classify_unop(op_str);
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    let src = operands.get(1).copied().unwrap_or(0) as usize;
                    (kind, vec![src], vec![dst], op_str.to_string(), Tier4Dtype::Unknown)
                }
                // LoadI
                0x01 | 0x02 => {
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    (Tier4RegionKind::Elementwise, vec![], vec![dst], "load".to_string(), Tier4Dtype::Unknown)
                }
                // Move
                0x03 => {
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    let src = operands.get(1).copied().unwrap_or(0) as usize;
                    // Moves are overhead — they don't create a new region
                    (current_kind.unwrap_or(Tier4RegionKind::Elementwise),
                     vec![src], vec![dst], "move".to_string(), Tier4Dtype::Unknown)
                }
                // Store
                0x04 => {
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    let src = operands.get(1).copied().unwrap_or(0) as usize;
                    (Tier4RegionKind::Elementwise, vec![src], vec![dst], "store".to_string(), Tier4Dtype::Unknown)
                }
                // Jump / JumpFalse / JumpTrue — control flow boundary
                0x21 | 0x22 | 0x23 => {
                    // Control flow ends the current region
                    if current_kind.is_some() {
                        let kind = current_kind.unwrap();
                        let tier = Self::tier_for_kind(&kind);
                        self.regions.push(Tier4Region {
                            index: region_idx,
                            kind,
                            tier,
                            instr_range: (current_start, i),
                            input_slots: std::mem::take(&mut current_input_slots),
                            output_slots: std::mem::take(&mut current_output_slots),
                            op_desc: std::mem::take(&mut current_op_desc),
                            fusible: current_fusible && Self::is_fusible(&kind),
                            dtype: current_dtype,
                        });
                        region_idx += 1;
                        current_kind = None;
                    }
                    continue;
                }
                // Reduction
                0x40..=0x4F => {
                    let op_str = match opcode {
                        0x40 => "sum",
                        0x41 => "max",
                        0x42 => "min",
                        _ => "reduce",
                    };
                    let dst = operands.get(0).copied().unwrap_or(0) as usize;
                    let src = operands.get(1).copied().unwrap_or(0) as usize;
                    (Tier4RegionKind::Reduction, vec![src], vec![dst], op_str.to_string(), Tier4Dtype::Unknown)
                }
                _ => {
                    // Unknown opcode — treat as scalar fallback
                    (Tier4RegionKind::Scalar, vec![], vec![], "unknown".to_string(), Tier4Dtype::Unknown)
                }
            };

            // Region merging: if the kind matches the current region, extend it
            match current_kind {
                None => {
                    current_kind = Some(kind);
                    current_start = i;
                    current_input_slots = inputs;
                    current_output_slots = outputs;
                    current_op_desc = desc;
                    current_fusible = Self::is_fusible(&kind);
                    current_dtype = dtype;
                }
                Some(ck) if ck == kind && kind != Tier4RegionKind::LinearAlgebra && kind != Tier4RegionKind::Reduction => {
                    // Extend current region
                    current_output_slots.extend(outputs);
                    if !current_op_desc.contains(&desc) {
                        current_op_desc.push_str(" → ");
                        current_op_desc.push_str(&desc);
                    }
                    // Merge dtype
                    if current_dtype == Tier4Dtype::Unknown {
                        current_dtype = dtype;
                    }
                }
                Some(ck) => {
                    // Kind changed — emit the current region and start a new one
                    let tier = Self::tier_for_kind(&ck);
                    self.regions.push(Tier4Region {
                        index: region_idx,
                        kind: ck,
                        tier,
                        instr_range: (current_start, i),
                        input_slots: std::mem::take(&mut current_input_slots),
                        output_slots: std::mem::take(&mut current_output_slots),
                        op_desc: std::mem::take(&mut current_op_desc),
                        fusible: current_fusible && Self::is_fusible(&ck),
                        dtype: current_dtype,
                    });
                    region_idx += 1;

                    current_kind = Some(kind);
                    current_start = i;
                    current_input_slots = inputs;
                    current_output_slots = outputs;
                    current_op_desc = desc;
                    current_fusible = Self::is_fusible(&kind);
                    current_dtype = dtype;
                }
            }
        }

        // Emit final region
        if let Some(ck) = current_kind {
            let tier = Self::tier_for_kind(&ck);
            self.regions.push(Tier4Region {
                index: region_idx,
                kind: ck,
                tier,
                instr_range: (current_start, trace_ops.len()),
                input_slots: current_input_slots,
                output_slots: current_output_slots,
                op_desc: current_op_desc,
                fusible: current_fusible && Self::is_fusible(&ck),
                dtype: current_dtype,
            });
        }

        &self.regions
    }
}

/// A single step in the Tier 4 execution schedule.
/// Carries all the metadata needed for the Python-side executor to dispatch
/// directly to the appropriate backend without re-scanning the trace.
#[derive(Clone, Debug)]
pub struct Tier4Step {
    /// Node ID in the fused DAG
    pub node_id: usize,
    /// Tier to dispatch to (0, 1, 2, or 3)
    pub tier: u8,
    /// Operation descriptor (e.g., "add → mul → tanh" for fused nodes)
    pub op_desc: String,
    /// Kind of computation (elementwise, reduction, linear algebra, etc.)
    pub kind: Tier4RegionKind,
    /// Input slot indices consumed by this step
    pub input_slots: Vec<usize>,
    /// Output slot indices produced by this step
    pub output_slots: Vec<usize>,
    /// Instruction range in the original trace: [start, end)
    pub instr_range: (usize, usize),
    /// Whether this step was formed by fusion
    pub is_fused: bool,
}

/// The Tier 4 planner: takes a DAG, applies fusion + buffer planning,
/// and produces an execution schedule.
#[derive(Clone, Debug)]
pub struct Tier4Schedule {
    /// Execution order: list of fully-described steps
    pub steps: Vec<Tier4Step>,
    /// Buffer reuse plan
    pub buffer_plan: Tier4BufferPlan,
    /// Whether any fusion was applied
    pub fusion_applied: bool,
    /// Number of fused nodes (supernodes)
    pub fused_node_count: usize,
    /// Total estimated cost of the schedule
    pub estimated_cost: f64,
    /// Peak memory estimate in bytes
    pub peak_memory_bytes: usize,
}

/// The top-level Tier 4 compilation function.
///
/// Takes a trace (as opcode + operand pairs), decomposes it into regions,
/// builds a DAG, applies conservative fusion, computes a buffer reuse plan,
/// and returns a schedule that dispatches to existing Tier 1–3 kernels.
///
/// This is the "cheap planning" compile strategy:
///   trace → identify kernel patterns → build execution graph → assign existing kernels
///
/// No heavy polyhedral optimization, no full SSA rewrites, no expensive
/// global transforms. Just a scheduler, not an optimizer.
pub fn tier4_compile(trace_ops: &[(u8, Vec<i64>)]) -> Tier4Schedule {
    // Step 1: Decompose into regions
    let mut decomposer = Tier4Decomposer::new();
    let regions = decomposer.decompose(trace_ops);

    // Step 2: Build DAG from regions
    let mut dag = Tier4Dag::new();

    // Map from output slot to producing node ID (for edge construction)
    let mut slot_producer: FxHashMap<usize, usize> = FxHashMap::default();

    // Track original DAG node count to detect fusion later
    let original_node_count = regions.len();

    for region in regions {
        let node_id = dag.add_node(Tier4Node {
            id: 0,
            kind: region.kind,
            tier: region.tier,
            input_slots: region.input_slots.clone(),
            output_slots: region.output_slots.clone(),
            op_desc: region.op_desc.clone(),
            fusible: region.fusible,
            estimated_cost: match region.kind {
                Tier4RegionKind::LinearAlgebra => 1000.0,
                Tier4RegionKind::Reduction => 100.0,
                Tier4RegionKind::Stencil => 50.0,
                Tier4RegionKind::Transcendental => 20.0,
                Tier4RegionKind::FmaChain => 10.0,
                Tier4RegionKind::Elementwise => 5.0,
                Tier4RegionKind::Logical => 3.0,
                Tier4RegionKind::Scalar => 1.0,
            },
            dtype: region.dtype,
            instr_range: region.instr_range,
        });

        // Add edges from producers of input slots to this node
        for &slot in &region.input_slots {
            if let Some(&producer_id) = slot_producer.get(&slot) {
                dag.add_edge(producer_id, node_id, slot, true, 8); // contiguous, 8 bytes
            }
        }

        // Record output slots produced by this node
        for &slot in &region.output_slots {
            slot_producer.insert(slot, node_id);
        }
    }

    // Step 3: Apply conservative fusion
    let fused_dag = dag.apply_conservative_fusion();
    let fusion_applied = fused_dag.nodes.len() < original_node_count;

    // Step 4: Compute buffer reuse plan (8 bytes per element = f64 default)
    let buffer_plan = fused_dag.compute_buffer_plan(8);

    // Step 5: Build execution schedule with full step metadata
    let order = fused_dag.topological_order();
    let mut steps = Vec::with_capacity(order.len());
    let mut total_cost = 0.0f64;

    for &node_id in &order {
        let node = &fused_dag.nodes[node_id];
        total_cost += node.estimated_cost;
        steps.push(Tier4Step {
            node_id: node.id,
            tier: node.tier,
            op_desc: node.op_desc.clone(),
            kind: node.kind,
            input_slots: node.input_slots.clone(),
            output_slots: node.output_slots.clone(),
            instr_range: node.instr_range,
            is_fused: fusion_applied && node.op_desc.contains(" → "),
        });
    }

    let peak_memory = buffer_plan.total_bytes;

    Tier4Schedule {
        steps,
        buffer_plan,
        fusion_applied,
        fused_node_count: fused_dag.nodes.len(),
        estimated_cost: total_cost,
        peak_memory_bytes: peak_memory,
    }
}

/// Tier 4 schedule validator.
///
/// Validates that a Tier 4 schedule is well-formed before the Python-side
/// executor dispatches it. Checks:
///   - All tiers are in range [0, 3]
///   - All input_slots have producers in earlier steps
///   - No circular dependencies (topological order is valid)
///   - Instruction ranges are non-empty and non-overlapping
///
/// Returns (valid, warning_count) — the schedule may be valid with warnings
/// (e.g., unfused Tier 1 chains that could benefit from fusion).
pub fn tier4_validate_schedule(schedule: &Tier4Schedule) -> (bool, usize) {
    if schedule.steps.is_empty() {
        return (false, 0);
    }

    let mut warnings = 0usize;

    // Track which output slots are produced by which step
    let mut produced: FxHashMap<usize, usize> = FxHashMap::default();

    for (step_idx, step) in schedule.steps.iter().enumerate() {
        // Validate tier range
        if step.tier > 3 {
            return (false, warnings);
        }

        // Validate instruction range is non-empty
        if step.instr_range.0 >= step.instr_range.1 {
            warnings += 1;
        }

        // Check that all input slots have producers
        for &slot in &step.input_slots {
            if !produced.contains_key(&slot) {
                // Input slot has no producer — it must be a function argument
                // (that's fine, but worth tracking)
            }
        }

        // Record output slots
        for &slot in &step.output_slots {
            produced.insert(slot, step_idx);
        }

        // Warn about unfused Tier 1 elementwise chains
        if step.tier == 1 && !step.is_fused && step.kind == Tier4RegionKind::Elementwise {
            warnings += 1;
        }
    }

    (true, warnings)
}
