#!/usr/bin/env python3
"""Direct test of SympleX JIT-compiled kernels via Python ctypes.

Tests the matmul and elementwise kernels from the compiled Rust engine.
This bypasses the full Python package and directly calls the native code.
"""

import numpy as np
import time
import sys

def test_native_kernels():
    """Test native JIT kernels via the compiled shared library."""
    print("Testing SympleX JIT-compiled native kernels")
    print("=" * 50)

    try:
        from symplex_engine import MatmulKernel, ElementwiseKernel, jit_info, has_avx2
        print(f"JIT Info: {jit_info()}")
        print(f"Has AVX2: {has_avx2()}")
    except ImportError:
        print("symplex_engine not available. Try: cd rust-engine && maturin develop --release")
        return

    # Test elementwise add
    print("\n--- Elementwise Add (N=1M) ---")
    N = 1_000_000
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    dst = np.zeros(N, dtype=np.float32)

    kernel = ElementwiseKernel("add")
    a_ptr = a.__array_interface__['data'][0]
    b_ptr = b.__array_interface__['data'][0]
    dst_ptr = dst.__array_interface__['data'][0]

    result = kernel.execute_raw(dst_ptr, a_ptr, b_ptr, N)
    print(f"  Kernel returned: {result}")

    # Verify correctness
    expected = a + b
    max_err = np.max(np.abs(dst - expected))
    print(f"  Max error: {max_err:.6e}")
    print(f"  Correct: {max_err < 1e-5}")

    # Benchmark
    def kernel_bench():
        kernel.execute_raw(dst_ptr, a_ptr, b_ptr, N)
    t_kernel = bench(kernel_bench)

    def numpy_bench():
        np.add(a, b, out=dst)
    t_numpy = bench(numpy_bench)

    print(f"  SympleX: {t_kernel*1e6:.1f} us")
    print(f"  NumPy:   {t_numpy*1e6:.1f} us")
    print(f"  Speedup: {t_numpy/t_kernel:.2f}x")

    # Test matmul
    print("\n--- Matmul 32x32 ---")
    M, K, N_size = 32, 32, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N_size).astype(np.float32)
    C = np.zeros((M, N_size), dtype=np.float32)

    mk = MatmulKernel()
    a_ptr = A.__array_interface__['data'][0]
    b_ptr = B.__array_interface__['data'][0]
    c_ptr = C.__array_interface__['data'][0]

    result = mk.execute_raw(a_ptr, b_ptr, c_ptr, M, N_size, K)
    print(f"  Kernel returned: {result}")

    # Verify
    expected = A @ B
    max_err = np.max(np.abs(C - expected))
    print(f"  Max error: {max_err:.6e}")
    print(f"  Correct: {max_err < 1e-2}")  # Scalar SSE is less precise

    # Benchmark
    def kernel_mm():
        mk.execute_raw(a_ptr, b_ptr, c_ptr, M, N_size, K)
    t_kernel = bench(kernel_mm)

    def numpy_mm():
        A @ B
    t_numpy = bench(numpy_mm)

    print(f"  SympleX: {t_kernel*1e6:.1f} us")
    print(f"  NumPy:   {t_numpy*1e6:.1f} us")
    print(f"  Speedup: {t_numpy/t_kernel:.2f}x" if t_kernel > 0 else "  Speedup: inf")


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


if __name__ == "__main__":
    test_native_kernels()
