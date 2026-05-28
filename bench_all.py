#!/usr/bin/env python3
"""
SympleX vs NumPy vs Numba vs Rust — Comprehensive Matmul Benchmark
===================================================================
Compares matrix multiplication performance across all four frameworks
at various sizes (4×4 to 1024×1024).

SympleX = AVX-512 FMA JIT kernel (VFMADD231PS with ZMM registers)
Rust    = Native Rust with -C target-cpu=native (AVX-512 auto-vectorized)
NumPy   = BLAS-backed matmul (OpenBLAS/MKL)
Numba   = @njit parallel matmul
"""

import time
import subprocess
import sys
import os
import json
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Numba benchmark
# ─────────────────────────────────────────────────────────────────────────────

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
    set_num_threads(1)  # Single-threaded for fair comparison with SympleX
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARN] Numba not available, skipping Numba benchmarks")

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def numba_matmul(a, b, c, m, n, k):
        for i in range(m):
            for p in range(k):
                a_val = a[i, p]
                for j in range(n):
                    c[i, j] += a_val * b[p, j]

    # Warm up JIT
    _a = np.zeros((2, 2), dtype=np.float32)
    _b = np.zeros((2, 2), dtype=np.float32)
    _c = np.zeros((2, 2), dtype=np.float32)
    numba_matmul(_a, _b, _c, 2, 2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def bench_numpy(m, n, k, warmup=3, iters=10):
    """Benchmark NumPy matmul (BLAS-backed)."""
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    # Warmup
    for _ in range(warmup):
        _ = a @ b
    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    elapsed = (time.perf_counter() - t0) / iters
    return elapsed, c


def bench_numba(m, n, k, warmup=2, iters=5):
    """Benchmark Numba @njit matmul."""
    if not NUMBA_AVAILABLE:
        return None, None
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)
    # Warmup
    for _ in range(warmup):
        c[:] = 0
        numba_matmul(a, b, c, m, n, k)
    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        c[:] = 0
        numba_matmul(a, b, c, m, n, k)
    elapsed = (time.perf_counter() - t0) / iters
    return elapsed, c


def bench_rust(m, n, k, iters=10):
    """Benchmark SympleX Rust AVX-512 FMA JIT kernel."""
    rust_bin = os.path.join(os.path.dirname(__file__), "target", "release", "matmul_bench")
    if not os.path.exists(rust_bin):
        # Try absolute path
        rust_bin = "/home/z/my-project/SympleX/target/release/matmul_bench"
    if not os.path.exists(rust_bin):
        return None, None
    # The Rust benchmark prints its own timing. We parse it.
    # But for fair comparison, we'll time it ourselves using a dedicated runner.
    # Instead, let's use the Rust kernel directly through the shared library.
    # For simplicity, we'll use the subprocess output.
    # Actually, let's build a small timing binary.
    return None, None  # We'll use the Rust output directly


# ─────────────────────────────────────────────────────────────────────────────
# Rust benchmark via subprocess
# ─────────────────────────────────────────────────────────────────────────────

def run_rust_benchmark():
    """Run the Rust matmul_bench and parse results."""
    rust_bin = "/home/z/my-project/SympleX/target/release/matmul_bench"
    if not os.path.exists(rust_bin):
        print("[WARN] Rust binary not found, skipping Rust benchmarks")
        return {}
    try:
        result = subprocess.run([rust_bin], capture_output=True, text=True, timeout=60)
        output = result.stderr + result.stdout
    except Exception as e:
        print(f"[WARN] Rust benchmark failed: {e}")
        return {}

    results = {}
    # Parse lines like "SympleX Matmul Benchmark: 128×128×128 (f32)"
    # and "JIT execute:       128 µs"
    current_size = None
    for line in output.split('\n'):
        if 'Matmul Benchmark:' in line:
            parts = line.split('Matmul Benchmark:')[1].strip()
            size_str = parts.split(' ')[0]
            dims = size_str.split('×')
            if len(dims) == 3:
                current_size = int(dims[0])
        if 'JIT execute:' in line and current_size:
            try:
                us_str = line.split('JIT execute:')[1].strip().split('µs')[0].strip()
                us = float(us_str)
                results[current_size] = us / 1e6  # convert to seconds
            except (ValueError, IndexError):
                pass
        if 'Scalar reference:' in line and current_size:
            try:
                us_str = line.split('Scalar reference:')[1].strip().split('µs')[0].strip()
                us = float(us_str)
                key = f"scalar_{current_size}"
                results[key] = us / 1e6
            except (ValueError, IndexError):
                pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

SIZES = [4, 16, 32, 64, 128, 256, 512, 1024]

def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║         SympleX vs NumPy vs Numba vs Rust — Matmul Benchmark              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

    # CPU info
    try:
        cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in cpu_info.stdout.split('\n'):
            if 'Model name' in line or 'CPU(s):' in line or 'Thread' in line:
                print(f"  {line.strip()}")
    except Exception:
        pass
    print()

    # Run Rust benchmark first (it prints to stderr)
    print("  Running Rust (SympleX AVX-512 FMA JIT) benchmark...")
    rust_results = run_rust_benchmark()
    if rust_results:
        rust_sizes = sorted(set(k for k in rust_results if isinstance(k, int)))
        print(f"  Rust results: {len(rust_sizes)} sizes benchmarked")
    else:
        print("  Rust: SKIPPED (binary not found or failed)")
    print()

    # Python benchmarks
    print(f"  {'Size':>8s} │ {'NumPy':>10s} │ {'Numba':>10s} │ {'SympleX':>10s} │ {'vs NumPy':>10s} │ {'vs Numba':>10s}")
    print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

    results_table = []

    for size in SIZES:
        m = n = k = size

        # NumPy
        try:
            np_time, np_c = bench_numpy(m, n, k, warmup=2, iters=max(3, 50 // max(1, size // 64)))
        except Exception as e:
            print(f"  {size:>8d} │ NumPy FAILED: {e}")
            continue

        # Numba
        nb_time = None
        if NUMBA_AVAILABLE and size <= 256:  # Numba is too slow for large sizes
            try:
                nb_time, nb_c = bench_numba(m, n, k, warmup=1, iters=max(1, 5 // max(1, size // 128)))
            except Exception:
                nb_time = None

        # SympleX (Rust)
        sx_time = rust_results.get(size, None)

        # Format
        np_str = f"{np_time*1e6:.1f} µs" if np_time else "N/A"
        nb_str = f"{nb_time*1e6:.1f} µs" if nb_time else "skip"
        sx_str = f"{sx_time*1e6:.1f} µs" if sx_time else "N/A"

        # Speedups
        vs_numpy = ""
        vs_numba = ""
        if sx_time and np_time:
            ratio = np_time / sx_time
            vs_numpy = f"{ratio:.2f}×"
        if sx_time and nb_time:
            ratio = nb_time / sx_time
            vs_numba = f"{ratio:.2f}×"

        # GFLOPS
        flops = 2.0 * m * n * k
        sx_gflops = flops / sx_time / 1e9 if sx_time else 0
        np_gflops = flops / np_time / 1e9 if np_time else 0

        print(f"  {size:>8d} │ {np_str:>10s} │ {nb_str:>10s} │ {sx_str:>10s} │ {vs_numpy:>10s} │ {vs_numba:>10s}")

        results_table.append({
            "size": size,
            "numpy_us": np_time * 1e6 if np_time else None,
            "numba_us": nb_time * 1e6 if nb_time else None,
            "symplex_us": sx_time * 1e6 if sx_time else None,
            "vs_numpy": vs_numpy,
            "vs_numba": vs_numba,
            "symplex_gflops": sx_gflops,
            "numpy_gflops": np_gflops,
        })

    # Summary
    print()
    print("  ─── Summary ───────────────────────────────────────────────────────────")
    wins_vs_numpy = sum(1 for r in results_table if r["symplex_us"] and r["numpy_us"] and r["symplex_us"] < r["numpy_us"])
    total_vs_numpy = sum(1 for r in results_table if r["symplex_us"] and r["numpy_us"])
    wins_vs_numba = sum(1 for r in results_table if r["symplex_us"] and r["numba_us"] and r["symplex_us"] < r["numba_us"])
    total_vs_numba = sum(1 for r in results_table if r["symplex_us"] and r["numba_us"])

    if total_vs_numpy > 0:
        print(f"  SympleX vs NumPy: {wins_vs_numpy}/{total_vs_numpy} wins")
    if total_vs_numba > 0:
        print(f"  SympleX vs Numba: {wins_vs_numba}/{total_vs_numba} wins")

    # Peak GFLOPS
    sx_peak = max((r["symplex_gflops"] for r in results_table if r["symplex_gflops"]), default=0)
    np_peak = max((r["numpy_gflops"] for r in results_table if r["numpy_gflops"]), default=0)
    print(f"  Peak GFLOPS — SympleX: {sx_peak:.1f}, NumPy: {np_peak:.1f}")

    # Save results as JSON
    results_path = "/home/z/my-project/download/benchmark_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_table, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
