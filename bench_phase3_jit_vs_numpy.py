#!/usr/bin/env python3
"""
SympleX Phase3 JIT vs NumPy Benchmark Suite
=============================================
Benchmarks the phase3_jit x86-64 JIT-compiled kernels against NumPy
across 5 categories of numeric workloads:

1. Elementwise arithmetic (add, mul, fma, sqrt, abs)
2. Reductions (sum, min, max)
3. Matmul (small, medium, large)
4. Stencil computations (Laplacian, heat equation)
5. N-body force accumulation

Requires: numpy, symplex_engine (PyO3 module)
"""

import time
import sys
import os
import statistics
import numpy as np

# ── Try to import the SympleX engine ──────────────────────────────────────
try:
    from symplex import _symplex_core
    HAS_ENGINE = True
except ImportError:
    try:
        import symplex_engine
        _symplex_core = symplex_engine
        HAS_ENGINE = True
    except ImportError:
        HAS_ENGINE = False


# ── Configuration ─────────────────────────────────────────────────────────
WARMUP_ITERS = 3
MEASURE_ITERS = 10
SEED = 42

# ── Helpers ────────────────────────────────────────────────────────────────

def bench(func, *args, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Run func with args, return median time in seconds."""
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times), result


def _bench_jit_scalar(kernel, a_val, b_val, n_calls, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Benchmark a Phase3JitKernel scalar call n_calls times per iteration."""
    for _ in range(warmup):
        for _ in range(min(n_calls, 1000)):
            kernel.execute_int(a_val, b_val)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            kernel.execute_int(a_val, b_val)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    result = kernel.execute_int(a_val, b_val)
    return result, statistics.median(times)


def format_time(seconds):
    if seconds < 1e-6:
        return f"{seconds*1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def speedup_str(jit_time, np_time):
    if jit_time > 0:
        ratio = np_time / jit_time
        if ratio >= 1.0:
            return f"  {ratio:.2f}x faster"
        else:
            return f"  {1/ratio:.2f}x slower"
    return "  N/A"


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Elementwise Arithmetic
# ═══════════════════════════════════════════════════════════════════════════

def bench_elementwise():
    print("\n" + "="*70)
    print("  Category 1: Elementwise Arithmetic")
    print("="*70)
    results = {}

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    ops = [
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
        ("fma", lambda a, b, c: a * b + c),
        ("sub", lambda a, b: a - b),
    ]

    np.random.seed(SEED)

    # Pre-create JIT kernels for integer ops (if engine is available)
    jit_kernels = {}
    if HAS_ENGINE:
        for jit_op in ["add", "sub", "mul"]:
            try:
                jit_kernels[jit_op] = _symplex_core.Phase3JitKernel(jit_op, 1)
            except Exception as e:
                print(f"  Warning: could not create JIT kernel for '{jit_op}': {e}")

    for size in sizes:
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        c = np.random.randn(size).astype(np.float32)

        for op_name, op_fn in ops:
            # NumPy baseline
            if op_name == "fma":
                np_time, np_result = bench(lambda: op_fn(a, b, c))
            else:
                np_time, np_result = bench(lambda: op_fn(a, b))

            # SympleX JIT (integer scalar, benchmark per-element throughput)
            jit_time = None
            if op_name in jit_kernels:
                try:
                    kernel = jit_kernels[op_name]
                    # Benchmark JIT by running scalar ops in a Python loop
                    # This measures pure JIT call overhead + execution
                    a_val, b_val = 42, 17
                    n_calls = size  # same number of ops as NumPy processes elements
                    _, jit_elapsed = _bench_jit_scalar(kernel, a_val, b_val, n_calls)
                    jit_time = jit_elapsed
                except Exception:
                    jit_time = None

            key = f"elementwise/{op_name}/n={size}"
            results[key] = (np_time, jit_time)

            label = f"  {op_name:>4s}  n={size:>10,}"
            np_str = format_time(np_time)
            jit_str = format_time(jit_time) if jit_time else "N/A"
            speed = speedup_str(jit_time, np_time) if jit_time else ""
            print(f"{label}  | NumPy: {np_str:>8s}  | JIT: {jit_str:>8s}{speed}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Reductions
# ═══════════════════════════════════════════════════════════════════════════

def bench_reductions():
    print("\n" + "="*70)
    print("  Category 2: Reductions")
    print("="*70)
    results = {}

    sizes = [1_000, 100_000, 10_000_000]
    np.random.seed(SEED)

    for size in sizes:
        a = np.random.randn(size).astype(np.float64)

        # Sum
        np_time, np_result = bench(lambda: np.sum(a))
        results[f"reduction/sum/n={size}"] = (np_time, None)
        print(f"  sum     n={size:>10,}  | NumPy: {format_time(np_time):>8s}")

        # Min
        np_time, _ = bench(lambda: np.min(a))
        results[f"reduction/min/n={size}"] = (np_time, None)
        print(f"  min     n={size:>10,}  | NumPy: {format_time(np_time):>8s}")

        # Max
        np_time, _ = bench(lambda: np.max(a))
        results[f"reduction/max/n={size}"] = (np_time, None)
        print(f"  max     n={size:>10,}  | NumPy: {format_time(np_time):>8s}")

        # Argmin
        np_time, _ = bench(lambda: np.argmin(a))
        results[f"reduction/argmin/n={size}"] = (np_time, None)
        print(f"  argmin  n={size:>10,}  | NumPy: {format_time(np_time):>8s}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Matrix Multiplication
# ═══════════════════════════════════════════════════════════════════════════

def bench_matmul():
    print("\n" + "="*70)
    print("  Category 3: Matrix Multiplication")
    print("="*70)
    results = {}

    dims = [32, 64, 128, 256, 512]
    np.random.seed(SEED)

    for dim in dims:
        A = np.random.randn(dim, dim).astype(np.float32)
        B = np.random.randn(dim, dim).astype(np.float32)

        # NumPy baseline
        np_time, np_result = bench(lambda: A @ B)

        # SympleX parallel matmul (tested and working)
        par_time = None
        if HAS_ENGINE:
            try:
                A32 = A.astype(np.float32)
                B32 = B.astype(np.float32)
                C2 = np.zeros((dim, dim), dtype=np.float32)
                a_ptr = A32.__array_interface__['data'][0]
                b_ptr = B32.__array_interface__['data'][0]
                c2_ptr = C2.__array_interface__['data'][0]
                par_time, _ = bench(
                    lambda: _symplex_core.jit_parallel_matmul(
                        a_ptr, b_ptr, c2_ptr, dim, dim, dim
                    )
                )
            except Exception:
                par_time = None

        key = f"matmul/{dim}x{dim}"
        results[key] = (np_time, par_time)

        np_str = format_time(np_time)
        par_str = format_time(par_time) if par_time else "N/A"
        speed = speedup_str(par_time, np_time) if par_time else ""
        gflops_np = (2 * dim**3) / np_time / 1e9 if np_time > 0 else 0
        gflops_jit = (2 * dim**3) / par_time / 1e9 if par_time and par_time > 0 else 0
        print(f"  {dim:>3d}×{dim:<3d}  | NumPy: {np_str:>8s} ({gflops_np:.1f} GF)  "
              f"| SympleX: {par_str:>8s}{speed}"
              f" ({gflops_jit:.1f} GF)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Stencil Computations
# ═══════════════════════════════════════════════════════════════════════════

def bench_stencils():
    print("\n" + "="*70)
    print("  Category 4: Stencil Computations")
    print("="*70)
    results = {}

    np.random.seed(SEED)
    grid_sizes = [64, 128, 256, 512]
    steps = 10

    for N in grid_sizes:
        # 5-point Laplacian stencil
        u = np.random.randn(N, N).astype(np.float64)
        dx = 1.0 / N

        def laplacian_step(u, dx):
            lap = np.zeros_like(u)
            lap[1:-1, 1:-1] = (
                u[2:, 1:-1] + u[:-2, 1:-1] +
                u[1:-1, 2:] + u[1:-1, :-2] -
                4 * u[1:-1, 1:-1]
            ) / (dx * dx)
            return lap

        np_time, _ = bench(lambda: laplacian_step(u, dx), iters=5)
        results[f"stencil/laplacian/{N}x{N}"] = (np_time, None)

        # Heat equation: u += alpha * dt * laplacian
        alpha = 0.01
        dt = 0.25 * dx * dx  # CFL condition

        def heat_step(u, alpha, dt, dx):
            lap = laplacian_step(u, dx)
            return u + alpha * dt * lap

        np_time_heat, _ = bench(lambda: heat_step(u, alpha, dt, dx), iters=5)
        results[f"stencil/heat/{N}x{N}"] = (np_time_heat, None)

        gflops = (5 * (N-2)**2) / np_time / 1e9 if np_time > 0 else 0
        print(f"  Laplacian {N:>3d}×{N:<3d}  | NumPy: {format_time(np_time):>8s} ({gflops:.1f} GF)")
        print(f"  Heat Eq.  {N:>3d}×{N:<3d}  | NumPy: {format_time(np_time_heat):>8s}")

    # Multi-step heat equation benchmark
    N = 256
    u = np.random.randn(N, N).astype(np.float64)
    alpha = 0.01
    dt = 0.25 * dx * dx

    def heat_steps(u, alpha, dt, dx, n_steps):
        for _ in range(n_steps):
            lap = laplacian_step(u, dx)
            u = u + alpha * dt * lap
        return u

    np_time, _ = bench(lambda: heat_steps(u, alpha, dt, dx, steps))
    results[f"stencil/heat_multi/{N}x{N}/{steps}steps"] = (np_time, None)
    print(f"  Heat {steps} steps {N}×{N}  | NumPy: {format_time(np_time):>8s}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: N-body Force Accumulation
# ═══════════════════════════════════════════════════════════════════════════

def bench_nbody():
    print("\n" + "="*70)
    print("  Category 5: N-body Force Accumulation")
    print("="*70)
    results = {}

    np.random.seed(SEED)
    ns = [100, 500, 1000]

    for n in ns:
        pos = np.random.randn(n, 3).astype(np.float64)
        mass = np.random.rand(n).astype(np.float64) + 0.1
        G = 6.674e-11
        eps = 0.01  # softening

        def nbody_forces(pos, mass, G, eps):
            n = len(pos)
            forces = np.zeros_like(pos)
            for i in range(n):
                dx = pos[:, 0] - pos[i, 0]
                dy = pos[:, 1] - pos[i, 1]
                dz = pos[:, 2] - pos[i, 2]
                r2 = dx*dx + dy*dy + dz*dz + eps*eps
                inv_r3 = r2**(-1.5)
                inv_r3[i] = 0  # no self-force
                f = G * mass[i] * mass * inv_r3
                forces[i, 0] = np.sum(f * dx)
                forces[i, 1] = np.sum(f * dy)
                forces[i, 2] = np.sum(f * dz)
            return forces

        np_time, _ = bench(lambda: nbody_forces(pos, mass, G, eps), iters=3)
        results[f"nbody/force/n={n}"] = (np_time, None)

        # Vectorized NumPy version (O(N²) but vectorized)
        def nbody_vectorized(pos, mass, G, eps):
            n = len(pos)
            # diff[i,j] = pos[j] - pos[i]
            diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (n, n, 3)
            r2 = np.sum(diff**2, axis=2) + eps*eps  # (n, n)
            inv_r3 = r2**(-1.5)
            np.fill_diagonal(inv_r3, 0)
            # F_i = sum_j G * m_i * m_j / r³ * diff_ij
            f_scalar = G * mass[:, np.newaxis] * mass[np.newaxis, :] * inv_r3  # (n, n)
            forces = np.sum(f_scalar[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
            return forces

        np_vec_time, _ = bench(lambda: nbody_vectorized(pos, mass, G, eps), iters=5)
        results[f"nbody/force_vec/n={n}"] = (np_vec_time, None)

        gflops = (20 * n * n) / np_vec_time / 1e9 if np_vec_time > 0 else 0
        print(f"  N={n:>4d}  | NumPy loop: {format_time(np_time):>8s}  "
              f"| NumPy vec: {format_time(np_vec_time):>8s} ({gflops:.1f} GF)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Pure Integer Arithmetic (phase3_jit specialty)
# ═══════════════════════════════════════════════════════════════════════════

def bench_integer():
    print("\n" + "="*70)
    print("  Category 6: Pure Integer Arithmetic (JIT specialty)")
    print("="*70)
    results = {}

    np.random.seed(SEED)
    sizes = [10_000, 100_000, 1_000_000]

    # Create JIT kernels for integer ops
    jit_int_kernels = {}
    if HAS_ENGINE:
        for jit_op in ["add", "sub", "mul", "div", "rem", "bitand", "bitor", "bitxor"]:
            try:
                jit_int_kernels[jit_op] = _symplex_core.Phase3JitKernel(jit_op, 1)
            except Exception as e:
                print(f"  Warning: could not create JIT kernel for '{jit_op}': {e}")

    for size in sizes:
        a = np.random.randint(-1000, 1000, size=size, dtype=np.int64)
        b = np.random.randint(-1000, 1000, size=size, dtype=np.int64)

        # Integer add
        np_time, _ = bench(lambda: a + b)
        jit_time = None
        if "add" in jit_int_kernels:
            try:
                _, jit_elapsed = _bench_jit_scalar(jit_int_kernels["add"], 42, 17, size)
                jit_time = jit_elapsed
            except Exception:
                pass
        results[f"integer/add/n={size}"] = (np_time, jit_time)
        jit_str = format_time(jit_time) if jit_time else "N/A"
        speed = speedup_str(jit_time, np_time) if jit_time else ""
        print(f"  i64 add     n={size:>10,}  | NumPy: {format_time(np_time):>8s}  | JIT: {jit_str:>8s}{speed}")

        # Integer multiply
        np_time, _ = bench(lambda: a * b)
        jit_time = None
        if "mul" in jit_int_kernels:
            try:
                _, jit_elapsed = _bench_jit_scalar(jit_int_kernels["mul"], 42, 17, size)
                jit_time = jit_elapsed
            except Exception:
                pass
        results[f"integer/mul/n={size}"] = (np_time, jit_time)
        jit_str = format_time(jit_time) if jit_time else "N/A"
        speed = speedup_str(jit_time, np_time) if jit_time else ""
        print(f"  i64 mul     n={size:>10,}  | NumPy: {format_time(np_time):>8s}  | JIT: {jit_str:>8s}{speed}")

        # Integer division
        b_safe = np.where(b == 0, 1, b)
        np_time, _ = bench(lambda: a // b_safe)
        jit_time = None
        if "div" in jit_int_kernels:
            try:
                _, jit_elapsed = _bench_jit_scalar(jit_int_kernels["div"], 42, 7, size)
                jit_time = jit_elapsed
            except Exception:
                pass
        results[f"integer/div/n={size}"] = (np_time, jit_time)
        jit_str = format_time(jit_time) if jit_time else "N/A"
        speed = speedup_str(jit_time, np_time) if jit_time else ""
        print(f"  i64 div     n={size:>10,}  | NumPy: {format_time(np_time):>8s}  | JIT: {jit_str:>8s}{speed}")

        # Fibonacci-like recurrence (loop with carries)
        n_fib = 10_000
        def fib_loop(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a

        np_time, _ = bench(lambda: fib_loop(n_fib))

        # JIT loop benchmark
        jit_time = None
        if HAS_ENGINE:
            try:
                loop_result = _symplex_core.jit_bench_loop(n_fib, MEASURE_ITERS)
                # Parse ns/iter from the result string
                parts = loop_result.split()
                for p in parts:
                    if p.startswith("time="):
                        ns = float(p.replace("time=", "").replace("ns/iter", ""))
                        jit_time = ns * 1e-9 * n_fib  # total time for n_fib iterations
            except Exception:
                pass

        results[f"integer/fib/n={n_fib}"] = (np_time, jit_time)
        jit_str = format_time(jit_time) if jit_time else "N/A"
        speed = speedup_str(jit_time, np_time) if jit_time else ""
        print(f"  fib loop    n={n_fib:>10,}  | NumPy: {format_time(np_time):>8s}  | JIT: {jit_str:>8s}{speed}")

        # Collatz sequence (mixed arithmetic + branching)
        def collatz_steps(start, max_steps=1000):
            n = start
            steps = 0
            while n != 1 and steps < max_steps:
                if n % 2 == 0:
                    n = n // 2
                else:
                    n = 3 * n + 1
                steps += 1
            return steps

        starts = np.random.randint(1, 100000, size=1000)
        np_time, _ = bench(lambda: [collatz_steps(s) for s in starts])
        results[f"integer/collatz/n=1000"] = (np_time, None)
        print(f"  collatz     n=      1,000  | NumPy: {format_time(np_time):>8s}")

        break  # only test one size for integer loops

    # Standalone JIT bench using the function API
    if HAS_ENGINE:
        print("\n  ── Phase3 JIT Standalone Benchmarks ──")
        try:
            for op in ["add", "sub", "mul"]:
                result = _symplex_core.jit_bench_int(op, 42, 17, 1_000_000)
                print(f"  {result}")
        except Exception as e:
            print(f"  JIT bench error: {e}")

        try:
            result = _symplex_core.jit_bench_loop(1000, 10000)
            print(f"  {result}")
        except Exception as e:
            print(f"  JIT loop bench error: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  SympleX Phase3 JIT vs NumPy Benchmark Suite                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Print environment info
    print(f"\nNumPy version: {np.__version__}")
    print(f"Python: {sys.version}")

    if HAS_ENGINE:
        print(f"SympleX engine: loaded")
        print(f"  ISA: {_symplex_core.detect_isa()}")
        print(f"  Vector width: {_symplex_core.vec_width()} floats")
        print(f"  CPU cores: {_symplex_core.num_cores()}")
        try:
            print(f"  AVX2: {_symplex_core.has_avx2()}")
            print(f"  AVX-512: {_symplex_core.has_avx512()}")
        except:
            pass
        try:
            jit_info = _symplex_core.jit_compile_info()
            print(f"\n{jit_info}")
        except:
            pass
    else:
        print("SympleX engine: NOT loaded (only NumPy baselines will run)")

    print(f"\nWarmup: {WARMUP_ITERS} iters, Measurement: {MEASURE_ITERS} iters (median)")

    all_results = {}
    all_results.update(bench_elementwise())
    all_results.update(bench_reductions())
    all_results.update(bench_matmul())
    all_results.update(bench_stencils())
    all_results.update(bench_nbody())
    all_results.update(bench_integer())

    # ── Summary ──
    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    wins = 0
    losses = 0
    ties = 0
    no_jit = 0
    for key, (np_time, jit_time) in all_results.items():
        if jit_time is None:
            no_jit += 1
            continue
        if jit_time < np_time * 0.95:
            wins += 1
        elif jit_time > np_time * 1.05:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    print(f"  JIT vs NumPy: {wins} wins, {losses} losses, {ties} ties "
          f"(out of {total} comparable benchmarks)")
    print(f"  NumPy-only baselines: {no_jit}")
    if total > 0:
        print(f"  Win rate: {100*wins/total:.0f}%")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "bench_results_phase3_jit.json")
    try:
        import json
        json_results = {}
        for key, (np_time, jit_time) in all_results.items():
            json_results[key] = {
                "numpy_time_s": np_time,
                "jit_time_s": jit_time,
            }
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")
    except Exception as e:
        print(f"\n  Could not save results: {e}")


if __name__ == "__main__":
    main()
