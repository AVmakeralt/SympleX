# Task: Add Fused Multi-Op SIMD Elementwise+Reduce Kernel

## Summary

Added fused multi-op SIMD kernels to the SympleX Rust engine that execute a chain of elementwise operations (e.g., `x * 2.0 + 1.0 → sum`) in a **single pass** over memory, keeping intermediate results in SIMD registers.

## Problem

The existing elementwise kernels do one pass per operation. For `x * 2.0 + 1.0`:
- Old: read x (800MB), mul, write temp (800MB), read temp (800MB), add, write temp2 (800MB), read temp2, reduce = ~4.8GB traffic
- New: read x (800MB), mul in registers, add in registers, accumulate = 800MB traffic (6x less!)

## Changes Made

### 1. `/home/z/my-project/rust-engine/src/x86_emitter.rs`

Added ~750 lines of code at the end of the file:

- **`FusedOpDesc` struct** - Operation descriptor with fields: `op` (add/sub/mul/div/min/max), `lhs_src` (input/constant/prev_result), `lhs_idx`, `rhs_src`, `rhs_idx`
- **`MAX_FUSED_OPS` const** - Maximum 8 ops in a fused chain (limited by YMM register count)
- **`fused_elem_f32_avx2_core`** - AVX2 kernel for f32:
  - Processes 8 f32 per YMM operation
  - 4x unrolled for reduce accumulation (32 elements per outer iteration)
  - Handles 8-element chunks and scalar tail
  - Horizontal reduction using permute+shuffle intrinsics
- **`fused_elem_f32_scalar`** - Scalar fallback for f32
- **`simd_fused_elementwise_f32`** - Public entry point, auto-selects AVX2 or scalar
- **`fused_elem_f64_avx2_core`** - AVX2 kernel for f64 (4-wide YMM, 16 elements per outer iteration)
- **`simd_fused_elementwise_f64`** - Public entry point for f64

### 2. `/home/z/my-project/rust-engine/src/lib.rs`

Added PyO3-exposed functions:
- **`simd_fused_elementwise_f32`** - Python-callable fused f32 kernel
- **`simd_fused_elementwise_f64`** - Python-callable fused f64 kernel
- Registered both in `#[pymodule]`

## API

```python
# Python calling convention
result = engine.simd_fused_elementwise_f32(
    ops=[(2, 0, 0, 1, 0), (0, 2, 0, 1, 1)],  # x*const[0] + const[1]
    input_ptrs=[x_ptr],     # raw pointer to input f32 array
    constants=[2.0, 1.0],   # f32 constants
    n=100000000,            # element count  
    reduce_op=0,            # 0=sum, 1=max, 2=min, 255=write to dst
    dst_ptr=0,              # output pointer (used when reduce_op=255)
)
```

## Compilation Status

Build succeeds with `cargo check --features pyo3/abi3-py38` - no errors, no warnings in the new code.
