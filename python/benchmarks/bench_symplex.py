#!/usr/bin/env python3
"""SympleX Comprehensive Benchmark Suite

Benchmarks SympleX vs NumPy vs pure Python across multiple workloads:
  1. Element-wise arithmetic (add, mul, fused multiply-add)
  2. Matrix multiplication (various sizes)
  3. Reductions (sum, mean, max)
  4. Activation functions (relu, sigmoid, gelu, softmax)
  5. Chain of operations (multi-step compute graphs)
  6. Rust polyhedral optimizer throughput (serialize + optimize)
  7. Purity checker latency
  8. Grad (reverse-mode AD) performance
  9. JIT compilation overhead (first-call vs cached)

Usage:
    python benchmarks/bench_symplex.py
"""

import sys
import os
import time
import json
import statistics
import traceback
from typing import List, Tuple, Dict, Any, Callable

# Ensure symplex is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import symplex
from symplex import jit, grad, DeviceArray
from symplex._ast_checker import check_purity
from symplex._tracer import trace_function
from symplex._errors import ImpureFunctionError


# ── Benchmark harness ────────────────────────────────────────────────────────

WARMUP_ITERS = 3
BENCH_ITERS = 20
QUICK_MODE = os.environ.get("QUICK_BENCH", "0") == "1"

if QUICK_MODE:
    WARMUP_ITERS = 1
    BENCH_ITERS = 5


def bench(fn: Callable, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> Dict[str, Any]:
    """Benchmark a function, returning timing statistics."""
    # Warmup
    for _ in range(warmup):
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            return {"mean_ms": float("inf"), "error": True}

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ms

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "iters": iters,
        "error": False,
    }


def fmt_ms(ms: float) -> str:
    if ms < 0.001:
        return f"{ms*1e6:.0f}ns"
    if ms < 1.0:
        return f"{ms:.3f}ms"
    return f"{ms:.2f}ms"


def speedup_label(symplex_ms: float, baseline_ms: float) -> str:
    if symplex_ms == 0 or baseline_ms == 0:
        return "N/A"
    ratio = baseline_ms / symplex_ms
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"


# ── 1. Element-wise arithmetic ───────────────────────────────────────────────

def bench_elementwise():
    print("\n" + "="*70)
    print("  BENCHMARK 1: Element-wise Arithmetic")
    print("="*70)

    sizes = [100, 1000, 10000, 100000]
    ops = ["add", "mul", "fma"]  # fma = x*y+z

    results = {}
    for size in sizes:
        for op in ops:
            np_a = np.random.randn(size).astype(np.float64)
            np_b = np.random.randn(size).astype(np.float64)
            np_c = np.random.randn(size).astype(np.float64)
            sx_a = DeviceArray(np_a)
            sx_b = DeviceArray(np_b)
            sx_c = DeviceArray(np_c)

            # NumPy baseline
            if op == "add":
                def _np_add(a, b): return a + b
                np_result = bench(_np_add, np_a, np_b)
            elif op == "mul":
                def _np_mul(a, b): return a * b
                np_result = bench(_np_mul, np_a, np_b)
            else:  # fma
                def _np_fma(a, b, c): return a * b + c
                np_result = bench(_np_fma, np_a, np_b, np_c)

            # SympleX DeviceArray
            if op == "add":
                def _sx_add(a, b): return a + b
                sx_result = bench(_sx_add, sx_a, sx_b)
            elif op == "mul":
                def _sx_mul(a, b): return a * b
                sx_result = bench(_sx_mul, sx_a, sx_b)
            else:
                def _sx_fma(a, b, c): return a * b + c
                sx_result = bench(_sx_fma, sx_a, sx_b, sx_c)

            key = f"{op}_{size}"
            results[key] = {
                "numpy_ms": np_result["mean_ms"],
                "symplex_ms": sx_result["mean_ms"],
                "speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
            }
            print(f"  {op:>4} n={size:>7}:  NumPy {fmt_ms(np_result['mean_ms']):>10}  "
                  f"SympleX {fmt_ms(sx_result['mean_ms']):>10}  "
                  f"-> {results[key]['speedup']}")

    return results


# ── 2. Matrix multiplication ────────────────────────────────────────────────

def bench_matmul():
    print("\n" + "="*70)
    print("  BENCHMARK 2: Matrix Multiplication (DeviceArray vs NumPy)")
    print("="*70)

    sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128),
             (256, 64, 128), (128, 256, 64)]

    results = {}
    for M, K, N in sizes:
        np_a = np.random.randn(M, K).astype(np.float64)
        np_b = np.random.randn(K, N).astype(np.float64)
        sx_a = DeviceArray(np_a)
        sx_b = DeviceArray(np_b)

        def _np_matmul(a, b): return a @ b
        def _sx_matmul(a, b): return a @ b
        np_result = bench(_np_matmul, np_a, np_b)
        sx_result = bench(_sx_matmul, sx_a, sx_b)

        key = f"matmul_{M}x{K}x{N}"
        flops = 2 * M * K * N
        np_gflops = flops / (np_result["mean_ms"] * 1e6) if np_result["mean_ms"] > 0 else 0
        sx_gflops = flops / (sx_result["mean_ms"] * 1e6) if sx_result["mean_ms"] > 0 else 0

        results[key] = {
            "numpy_ms": np_result["mean_ms"],
            "symplex_ms": sx_result["mean_ms"],
            "numpy_gflops": np_gflops,
            "symplex_gflops": sx_gflops,
            "speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
        }
        print(f"  {M}x{K} @ {K}x{N}:  NumPy {fmt_ms(np_result['mean_ms']):>10} ({np_gflops:.2f} GF)  "
              f"SympleX {fmt_ms(sx_result['mean_ms']):>10} ({sx_gflops:.2f} GF)  "
              f"-> {results[key]['speedup']}")

    return results


# ── 3. JIT-compiled matmul with polyhedral optimization ─────────────────────

def bench_jit_matmul():
    print("\n" + "="*70)
    print("  BENCHMARK 3: JIT-compiled Matmul (with polyhedral optimization)")
    print("="*70)

    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]

    results = {}
    for M, K, N in sizes:
        np_a = np.random.randn(M, K).astype(np.float64)
        np_b = np.random.randn(K, N).astype(np.float64)

        # NumPy baseline
        def _np_mm(a, b): return a @ b
        np_result = bench(_np_mm, np_a, np_b)

        # SympleX JIT
        @jit
        def jitted_matmul(A, B):
            return A @ B

        # First call (includes compilation)
        t0 = time.perf_counter_ns()
        r1 = jitted_matmul(np_a, np_b)
        t1 = time.perf_counter_ns()
        first_call_ms = (t1 - t0) / 1e6

        # Subsequent calls (cached)
        sx_result = bench(lambda a, b: jitted_matmul(a, b), np_a, np_b)

        # Get optimization info
        info = jitted_matmul.optimize_info(np_a, np_b)

        flops = 2 * M * K * N
        np_gflops = flops / (np_result["mean_ms"] * 1e6) if np_result["mean_ms"] > 0 else 0
        sx_gflops = flops / (sx_result["mean_ms"] * 1e6) if sx_result["mean_ms"] > 0 else 0

        key = f"jit_matmul_{M}x{K}x{N}"
        results[key] = {
            "numpy_ms": np_result["mean_ms"],
            "symplex_jit_ms": sx_result["mean_ms"],
            "first_call_ms": first_call_ms,
            "numpy_gflops": np_gflops,
            "symplex_gflops": sx_gflops,
            "opt_instrs": info.get("instr_count", 0) if info else 0,
            "opt_hints": info.get("hint_count", 0) if info else 0,
            "estimated_gflops": info.get("estimated_gflops", 0) if info else 0,
            "speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
        }
        print(f"  {M}x{K} @ {K}x{N}:")
        print(f"    NumPy:       {fmt_ms(np_result['mean_ms']):>10} ({np_gflops:.2f} GF)")
        print(f"    SympleX JIT: {fmt_ms(sx_result['mean_ms']):>10} ({sx_gflops:.2f} GF)  "
              f"first_call={fmt_ms(first_call_ms)}")
        if info:
            print(f"    Opt: {info.get('instr_count',0)} instrs, {info.get('hint_count',0)} hints, "
                  f"est_gflops={info.get('estimated_gflops',0):.1f}")
        print(f"    -> {results[key]['speedup']}")

    return results


# ── 4. Activation functions ─────────────────────────────────────────────────

def bench_activations():
    print("\n" + "="*70)
    print("  BENCHMARK 4: Activation Functions")
    print("="*70)

    size = 100000
    np_x = np.random.randn(size).astype(np.float64)
    sx_x = DeviceArray(np_x)

    activations = {
        "relu": (
            lambda x: np.maximum(x, 0),
            lambda x: x.relu(),
        ),
        "sigmoid": (
            lambda x: 1 / (1 + np.exp(-x)),
            lambda x: x.sigmoid(),
        ),
        "gelu": (
            lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            lambda x: x.gelu(),
        ),
        "tanh": (
            lambda x: np.tanh(x),
            lambda x: x.tanh(),
        ),
        "softmax": (
            lambda x: (lambda e: e / e.sum())(np.exp(x - np.max(x))),
            lambda x: x.softmax(),
        ),
    }

    results = {}
    for name, (np_fn, sx_fn) in activations.items():
        np_result = bench(np_fn, np_x)
        sx_result = bench(sx_fn, sx_x)

        results[name] = {
            "numpy_ms": np_result["mean_ms"],
            "symplex_ms": sx_result["mean_ms"],
            "speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
        }
        print(f"  {name:>8}:  NumPy {fmt_ms(np_result['mean_ms']):>10}  "
              f"SympleX {fmt_ms(sx_result['mean_ms']):>10}  "
              f"-> {results[name]['speedup']}")

    return results


# ── 5. Reductions ───────────────────────────────────────────────────────────

def bench_reductions():
    print("\n" + "="*70)
    print("  BENCHMARK 5: Reductions")
    print("="*70)

    sizes = [1000, 10000, 100000]
    reductions = ["sum", "mean", "max"]

    results = {}
    for size in sizes:
        np_x = np.random.randn(size).astype(np.float64)
        sx_x = DeviceArray(np_x)

        for red in reductions:
            np_fn = getattr(np, red)

            # Use a wrapper for sx that works without axis arg
            if red == "sum":
                def _sx_sum(x): return x.sum()
                sx_fn = _sx_sum
            elif red == "mean":
                def _sx_mean(x): return x.mean()
                sx_fn = _sx_mean
            else:
                def _sx_max(x): return x.max()
                sx_fn = _sx_max

            np_result = bench(np_fn, np_x)
            sx_result = bench(sx_fn, sx_x)

            key = f"{red}_{size}"
            results[key] = {
                "numpy_ms": np_result["mean_ms"],
                "symplex_ms": sx_result["mean_ms"],
                "speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
            }
            print(f"  {red:>4} n={size:>7}:  NumPy {fmt_ms(np_result['mean_ms']):>10}  "
                  f"SympleX {fmt_ms(sx_result['mean_ms']):>10}  "
                  f"-> {results[key]['speedup']}")

    return results


# ── 6. Rust polyhedral optimizer throughput ──────────────────────────────────

def bench_polyhedral_optimizer():
    print("\n" + "="*70)
    print("  BENCHMARK 6: Rust Polyhedral Optimizer Throughput")
    print("="*70)

    if not symplex.is_rust_engine_available():
        print("  (Rust engine not available, skipping)")
        return {}

    from symplex._symplex_core import optimize_trace, optimize_specialized, serialize_instructions

    trace_sizes = [10, 50, 100, 500, 1000]
    results = {}

    for trace_size in trace_sizes:
        # Build a synthetic trace
        trace = []
        for i in range(trace_size):
            trace.append(("load_f64", i * 2, float(i)))
        for i in range(trace_size):
            dst = trace_size * 2 + i
            lhs = (i * 2) % (trace_size * 2)
            rhs = ((i * 2) + 1) % (trace_size * 2)
            op = ["add", "mul", "sub", "div"][i % 4]
            trace.append(("binop", dst, op, lhs, rhs))

        # Serialize
        t0 = time.perf_counter_ns()
        trace_bytes = serialize_instructions(trace)
        t1 = time.perf_counter_ns()
        serialize_ms = (t1 - t0) / 1e6

        # Standard optimization
        opt_result = bench(
            lambda tb: optimize_trace(tb, target="server", element_type="fp32"),
            trace_bytes,
        )

        # Specialized optimization
        spec_result = bench(
            lambda tb: optimize_specialized(tb, target="server", element_type="fp32"),
            trace_bytes,
        )

        key = f"poly_opt_{trace_size}"
        results[key] = {
            "trace_instrs": len(trace),
            "serialize_ms": serialize_ms,
            "optimize_ms": opt_result["mean_ms"],
            "specialize_ms": spec_result["mean_ms"],
        }
        print(f"  {trace_size:>5} instrs:  serialize {fmt_ms(serialize_ms):>10}  "
              f"optimize {fmt_ms(opt_result['mean_ms']):>10}  "
              f"specialize {fmt_ms(spec_result['mean_ms']):>10}")

    return results


# ── 7. Purity checker latency ───────────────────────────────────────────────

def bench_purity_checker():
    print("\n" + "="*70)
    print("  BENCHMARK 7: AST Purity Checker Latency")
    print("="*70)

    # Must use def functions (not lambdas) so inspect.getsource works
    def _pure_add(x, y): return x + y
    def _pure_loop(x):
        result = 0
        for i in range(100):
            result = result + x
        return result
    def _pure_if(x):
        if x > 0:
            return x
        elif x < -1:
            return -x
        else:
            return 0
    def _pure_complex(x, y, z):
        return (x * y + z) / (x - y + 0.001) + abs(x)

    test_cases = {
        "simple_add": _pure_add,
        "for_loop": _pure_loop,
        "nested_if": _pure_if,
        "complex_expr": _pure_complex,
    }

    results = {}
    for name, func in test_cases.items():
        result = bench(check_purity, func)
        results[name] = {"check_ms": result["mean_ms"]}
        print(f"  {name:>14}:  {fmt_ms(result['mean_ms']):>10}")

    # Impure function (should fail fast)
    def impure(x):
        print(x)
        return x

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter_ns()
        try:
            check_purity(impure)
        except ImpureFunctionError:
            pass
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    impure_ms = statistics.mean(times)
    results["impure_reject"] = {"check_ms": impure_ms}
    print(f"  {'impure_reject':>14}:  {fmt_ms(impure_ms):>10}  (rejected)")

    return results


# ── 8. Grad (reverse-mode AD) ───────────────────────────────────────────────

def bench_grad():
    print("\n" + "="*70)
    print("  BENCHMARK 8: Grad (Reverse-Mode AD)")
    print("="*70)

    sizes = [10, 50, 100, 500]

    results = {}
    for size in sizes:
        np_x = np.random.randn(size).astype(np.float64)

        # SympleX grad
        def f(x):
            return (x * x).sum()

        df = grad(f)

        grad_result = bench(df, DeviceArray(np_x))

        # Numerical reference (manual central differences)
        def numerical_grad(x):
            eps = 1e-4
            g = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy(); x_plus[i] += eps
                x_minus = x.copy(); x_minus[i] -= eps
                g[i] = (f(DeviceArray(x_plus)) - f(DeviceArray(x_minus))) / (2 * eps)
            return g

        num_result = bench(numerical_grad, np_x)

        key = f"grad_{size}"
        results[key] = {
            "symplex_grad_ms": grad_result["mean_ms"],
            "numerical_grad_ms": num_result["mean_ms"],
            "speedup": speedup_label(grad_result["mean_ms"], num_result["mean_ms"]),
        }
        print(f"  grad n={size:>5}:  SympleX {fmt_ms(grad_result['mean_ms']):>10}  "
              f"Numerical {fmt_ms(num_result['mean_ms']):>10}  "
              f"-> {results[key]['speedup']}")

    return results


# ── 9. JIT compilation overhead ─────────────────────────────────────────────

def bench_jit_overhead():
    print("\n" + "="*70)
    print("  BENCHMARK 9: JIT Compilation Overhead (First Call vs Cached)")
    print("="*70)

    # Must use def functions (not lambdas) for jit decorator
    def _fn_add(x, y): return x + y
    def _fn_fma(x, y, z): return x * y + z
    def _fn_mm_add(A, B, c): return A @ B + c

    test_functions = {
        "simple_add": _fn_add,
        "fma": _fn_fma,
        "matmul_add": _fn_mm_add,
    }

    np_data = {
        "x": np.random.randn(100).astype(np.float64),
        "y": np.random.randn(100).astype(np.float64),
        "z": np.random.randn(100).astype(np.float64),
        "A": np.random.randn(32, 32).astype(np.float64),
        "B": np.random.randn(32, 32).astype(np.float64),
        "c": np.random.randn(32, 32).astype(np.float64),
    }

    results = {}
    for name, func in test_functions.items():
        # Compile the function with jit
        jitted = jit(func)

        # Get args
        if name == "simple_add":
            args = (np_data["x"], np_data["y"])
        elif name == "fma":
            args = (np_data["x"], np_data["y"], np_data["z"])
        else:
            args = (np_data["A"], np_data["B"], np_data["c"])

        # First call (includes trace + compile + execute)
        t0 = time.perf_counter_ns()
        _ = jitted(*args)
        t1 = time.perf_counter_ns()
        first_call_ms = (t1 - t0) / 1e6

        # Cached calls
        cached_result = bench(jitted, *args)

        key = name
        results[key] = {
            "first_call_ms": first_call_ms,
            "cached_ms": cached_result["mean_ms"],
            "compile_overhead_ms": first_call_ms - cached_result["mean_ms"],
        }
        print(f"  {name:>14}:  first_call {fmt_ms(first_call_ms):>10}  "
              f"cached {fmt_ms(cached_result['mean_ms']):>10}  "
              f"overhead {fmt_ms(first_call_ms - cached_result['mean_ms']):>10}")

    return results


# ── 10. Tracing throughput ──────────────────────────────────────────────────

def bench_tracing():
    print("\n" + "="*70)
    print("  BENCHMARK 10: Tracing Throughput")
    print("="*70)

    shapes = [(100,), (1000,), (10, 10), (32, 32), (100, 100)]

    results = {}
    for shape in shapes:
        def trace_fn(x, y):
            z = x * y + x
            w = z * y
            return w + x

        n_args = 2
        arg_shapes = [shape] * n_args
        arg_dtypes = ["float64"] * n_args

        trace_result = bench(trace_function, trace_fn, arg_shapes, arg_dtypes)

        key = f"trace_{'x'.join(str(s) for s in shape)}"
        results[key] = {
            "trace_ms": trace_result["mean_ms"],
        }
        print(f"  shape {str(shape):>14}:  {fmt_ms(trace_result['mean_ms']):>10}")

    return results


# ── 11. End-to-end: Full pipeline ───────────────────────────────────────────

def bench_e2e():
    print("\n" + "="*70)
    print("  BENCHMARK 11: End-to-End Full Pipeline (JIT + Optimize + Execute)")
    print("="*70)

    def _e2e_add(x, y): return x + y
    def _e2e_fma(x, y, z): return x * y + z
    def _e2e_mm(A, B): return A @ B
    def _e2e_mm_relu(A, B): return symplex.relu(A @ B)

    workloads = {
        "vec_add_1k": (
            _e2e_add,
            (np.random.randn(1000).astype(np.float64), np.random.randn(1000).astype(np.float64)),
        ),
        "fma_1k": (
            _e2e_fma,
            (np.random.randn(1000).astype(np.float64),
             np.random.randn(1000).astype(np.float64),
             np.random.randn(1000).astype(np.float64)),
        ),
        "matmul_32": (
            _e2e_mm,
            (np.random.randn(32, 32).astype(np.float64),
             np.random.randn(32, 32).astype(np.float64)),
        ),
        "matmul_relu_32": (
            _e2e_mm_relu,
            (np.random.randn(32, 32).astype(np.float64),
             np.random.randn(32, 32).astype(np.float64)),
        ),
    }

    results = {}
    for name, (func, args) in workloads.items():
        # NumPy baseline
        np_result = bench(func, *args)

        # SympleX JIT
        jitted = jit(func)
        # Warmup (compile)
        _ = jitted(*args)

        # JIT execution
        sx_result = bench(jitted, *args)

        # JIT with specialization
        jitted_spec = jit(func, specialize=True)
        _ = jitted_spec(*args)
        spec_result = bench(jitted_spec, *args)

        key = name
        results[key] = {
            "numpy_ms": np_result["mean_ms"],
            "symplex_jit_ms": sx_result["mean_ms"],
            "symplex_spec_ms": spec_result["mean_ms"],
            "jit_speedup": speedup_label(sx_result["mean_ms"], np_result["mean_ms"]),
            "spec_speedup": speedup_label(spec_result["mean_ms"], np_result["mean_ms"]),
        }
        print(f"  {name:>18}:  NumPy {fmt_ms(np_result['mean_ms']):>10}  "
              f"JIT {fmt_ms(sx_result['mean_ms']):>10} ({results[key]['jit_speedup']})  "
              f"Spec {fmt_ms(spec_result['mean_ms']):>10} ({results[key]['spec_speedup']})")

    return results


# ── 12. Rust engine direct API benchmarks ────────────────────────────────────

def bench_rust_api():
    print("\n" + "="*70)
    print("  BENCHMARK 12: Rust Engine Direct API")
    print("="*70)

    if not symplex.is_rust_engine_available():
        print("  (Rust engine not available, skipping)")
        return {}

    from symplex._symplex_core import (
        optimize_trace, optimize_specialized,
        detect_hardware, micro_kernel_config,
        serialize_instructions,
    )

    results = {}

    # detect_hardware
    hw_result = bench(detect_hardware)
    results["detect_hardware"] = {"ms": hw_result["mean_ms"]}
    print(f"  detect_hardware: {fmt_ms(hw_result['mean_ms'])}")

    # micro_kernel_config
    mk_result = bench(micro_kernel_config, "server", "fp32")
    results["micro_kernel_config"] = {"ms": mk_result["mean_ms"]}
    print(f"  micro_kernel_config: {fmt_ms(mk_result['mean_ms'])}")

    # Optimize with different targets
    trace = [("load_f64", 0, 1.0), ("load_f64", 1, 2.0), ("binop", 2, "add", 0, 1)]
    trace_bytes = serialize_instructions(trace)

    for target in ["server", "edge", "tensor"]:
        opt = bench(optimize_trace, trace_bytes, target, "real", "fp32")
        results[f"optimize_{target}"] = {"ms": opt["mean_ms"]}
        print(f"  optimize (target={target}): {fmt_ms(opt['mean_ms'])}")

    # Optimize with different element types
    for etype in ["fp32", "fp64", "fp16", "bf16", "int8"]:
        opt = bench(optimize_trace, trace_bytes, "server", "real", etype)
        results[f"optimize_{etype}"] = {"ms": opt["mean_ms"]}
        print(f"  optimize (etype={etype}): {fmt_ms(opt['mean_ms'])}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("+" + "="*68 + "+")
    print("|          SympleX Polyhedral Tensor Superoptimizer Benchmarks        |")
    print("+" + "="*68 + "+")

    print(f"\n  Python:  {sys.version.split()[0]}")
    print(f"  NumPy:   {np.__version__}")
    print(f"  SympleX: {symplex.__version__()}")
    print(f"  Rust engine: {symplex.is_rust_engine_available()}")

    hw = symplex.hardware_info()
    print(f"  Hardware: target={hw['target']}, SIMD={hw['simd_level']}")
    print(f"  Peak GFLOPS: {hw['peak_gflops']}, BW: {hw['mem_bandwidth_gb_per_sec']} GB/s")

    mk = symplex.micro_kernel_config()
    print(f"  MicroKernel: {mk['tile_m']}x{mk['tile_n']}x{mk['tile_k']}, "
          f"acc_regs={mk['accumulator_registers']}, db={mk['double_buffer_count']}")

    if QUICK_MODE:
        print(f"\n  QUICK MODE (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    else:
        print(f"\n  Standard mode (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")

    all_results = {}

    try:
        all_results["elementwise"] = bench_elementwise()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["matmul"] = bench_matmul()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["jit_matmul"] = bench_jit_matmul()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["activations"] = bench_activations()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["reductions"] = bench_reductions()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["polyhedral"] = bench_polyhedral_optimizer()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["purity"] = bench_purity_checker()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["grad"] = bench_grad()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["jit_overhead"] = bench_jit_overhead()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["tracing"] = bench_tracing()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["e2e"] = bench_e2e()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    try:
        all_results["rust_api"] = bench_rust_api()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\n  Results saved to: {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    # Count wins/losses
    wins = 0
    losses = 0
    neutrals = 0

    for category, bench_data in all_results.items():
        for key, data in bench_data.items():
            if "speedup" in data:
                s = data["speedup"]
                if "faster" in s:
                    wins += 1
                elif "slower" in s:
                    losses += 1
                else:
                    neutrals += 1

    print(f"\n  SympleX vs NumPy: {wins} wins, {losses} losses, {neutrals} ties")
    print(f"\n  Note: SympleX's DeviceArray wraps NumPy internally, so element-wise")
    print(f"  ops have similar throughput. The advantage comes from:")
    print(f"    - Polyhedral optimization of tiled loop nests (matmul, etc.)")
    print(f"    - JIT compilation caching (avoid re-tracing)")
    print(f"    - Hardware-aware tiling (SIMD, cache hierarchy)")
    print(f"    - Purity enforcement (safe for aggressive optimization)")
    print(f"    - Specialized pipelines (FlashAttention, mixed-precision, AD)")

    print("\n" + "="*70)
    print("  Benchmarks complete!")
    print("="*70)


if __name__ == "__main__":
    main()
