#!/usr/bin/env python3
"""SympleX Physics Benchmark Suite — 5 Categories

Benchmarks SympleX JIT-compiled native kernels vs NumPy in:
  1. PDE Stencils (2D Laplacian)
  2. N-body Force Accumulation
  3. Linear Algebra Chains (FMA, chained matmul)
  4. Physics Integrators (Symplectic Euler)
  5. Tensor Field Operations (Divergence)

Each benchmark runs multiple iterations and reports median time.
"""

import time
import numpy as np
from typing import Callable, Tuple

# ── Timing utility ──────────────────────────────────────────────────────

def bench(fn: Callable, warmup: int = 3, iters: int = 20) -> float:
    """Run fn warmup+iters times, return median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def gflops(op_count: float, time_s: float) -> float:
    """Compute GFLOPS given operation count and time."""
    if time_s == 0:
        return 0.0
    return op_count / time_s / 1e9


# ── Category 1: PDE Stencils ───────────────────────────────────────────

def benchmark_stencil(rows: int = 512, cols: int = 512, scale: float = 0.25):
    """2D 5-point Laplacian stencil: out[i,j] = scale * (N+S+W+E - 4*center)."""
    print(f"\n{'='*60}")
    print(f"Category 1: PDE Stencil — {rows}x{cols} Laplacian (scale={scale})")
    print(f"{'='*60}")

    field = np.random.randn(rows, cols).astype(np.float64)
    out = np.zeros_like(field)

    # NumPy reference
    def numpy_stencil():
        result = np.zeros_like(field)
        result[1:-1, 1:-1] = scale * (
            field[:-2, 1:-1] +   # north
            field[2:, 1:-1] +    # south
            field[1:-1, :-2] +   # west
            field[1:-1, 2:] -    # east
            4.0 * field[1:-1, 1:-1]  # center
        )
        return result

    t_numpy = bench(numpy_stencil)
    flops = 5.0 * (rows - 2) * (cols - 2)  # 5 ops per interior point
    gf_numpy = gflops(flops, t_numpy)

    # SympleX JIT — try native kernel
    t_symplex = None
    gf_symplex = None
    try:
        from symplex._symplex_core import jit_compile, jit_execute

        shape_info = {"rows": rows, "cols": cols, "scale": scale}
        result = jit_compile("stencil", shape_info)
        if result.get("success", False):
            kid = result["kernel_id"]
            def symplex_stencil():
                return jit_execute(kid, [out, field])
            t_symplex = bench(symplex_stencil)
            gf_symplex = gflops(flops, t_symplex)
    except (ImportError, Exception) as e:
        print(f"  SympleX native: not available ({e})")

    print(f"  NumPy:   {t_numpy*1e6:10.1f} us  ({gf_numpy:.2f} GFLOPS)")
    if t_symplex is not None:
        speedup = t_numpy / t_symplex
        print(f"  SympleX: {t_symplex*1e6:10.1f} us  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    else:
        print(f"  SympleX: N/A (falling back to Python)")

    return {"numpy": t_numpy, "symplex": t_symplex}


# ── Category 2: N-body Force Accumulation ──────────────────────────────

def benchmark_nbody(n_bodies: int = 256, softening_sq: float = 0.01):
    """N-body gravitational force accumulation with softening."""
    print(f"\n{'='*60}")
    print(f"Category 2: N-body — {n_bodies} bodies (softening²={softening_sq})")
    print(f"{'='*60}")

    positions = np.random.randn(n_bodies * 3).astype(np.float64)
    masses = np.random.randn(n_bodies).astype(np.float64)
    forces = np.zeros(n_bodies * 3, dtype=np.float64)

    # NumPy reference: vectorized with broadcasting
    def numpy_nbody():
        pos = positions.reshape(n_bodies, 3)
        # Compute all pairwise displacements
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
        r2 = np.sum(diff**2, axis=2) + softening_sq  # (N, N)
        inv_r3 = r2**(-1.5)  # 1 / (r2 * sqrt(r2))
        np.fill_diagonal(inv_r3, 0.0)  # no self-force
        # Force on i = sum_j m_j * (x_j - x_i) / r^3
        # Note: diff[i,j] = pos[j] - pos[i], so force direction is correct
        fx = np.sum(masses[np.newaxis, :] * diff[:, :, 0] * inv_r3, axis=1)
        fy = np.sum(masses[np.newaxis, :] * diff[:, :, 1] * inv_r3, axis=1)
        fz = np.sum(masses[np.newaxis, :] * diff[:, :, 2] * inv_r3, axis=1)
        result = np.stack([fx, fy, fz], axis=1).flatten()
        return result

    t_numpy = bench(numpy_nbody, warmup=2, iters=10)
    flops = n_bodies * n_bodies * 20.0  # ~20 FLOPs per pair
    gf_numpy = gflops(flops, t_numpy)

    # SympleX JIT
    t_symplex = None
    gf_symplex = None
    try:
        from symplex._symplex_core import jit_compile, jit_execute

        shape_info = {"n_bodies": n_bodies, "softening_sq": softening_sq}
        result = jit_compile("nbody", shape_info)
        if result.get("success", False):
            kid = result["kernel_id"]
            def symplex_nbody():
                return jit_execute(kid, [forces, positions, masses])
            t_symplex = bench(symplex_nbody, warmup=2, iters=10)
            gf_symplex = gflops(flops, t_symplex)
    except (ImportError, Exception) as e:
        print(f"  SympleX native: not available ({e})")

    print(f"  NumPy:   {t_numpy*1e3:10.2f} ms  ({gf_numpy:.2f} GFLOPS)")
    if t_symplex is not None:
        speedup = t_numpy / t_symplex
        print(f"  SympleX: {t_symplex*1e3:10.2f} ms  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    else:
        print(f"  SympleX: N/A (falling back to Python)")

    return {"numpy": t_numpy, "symplex": t_symplex}


# ── Category 3: Linear Algebra Chains ──────────────────────────────────

def benchmark_la_chains():
    """Chained FMA operations: dst = a*b + c, repeated."""
    print(f"\n{'='*60}")
    print(f"Category 3: Linear Algebra Chains — FMA (N=1M)")
    print(f"{'='*60}")

    N = 1_000_000
    a = np.random.randn(N).astype(np.float64)
    b = np.random.randn(N).astype(np.float64)
    c = np.random.randn(N).astype(np.float64)
    dst = np.zeros(N, dtype=np.float64)

    def numpy_fma():
        return a * b + c

    t_numpy = bench(numpy_fma)
    flops = 2.0 * N  # 1 mul + 1 add per element
    gf_numpy = gflops(flops, t_numpy)

    # SympleX JIT
    t_symplex = None
    gf_symplex = None
    try:
        from symplex._symplex_core import jit_compile, jit_execute

        shape_info = {"n": N}
        result = jit_compile("fma", shape_info)
        if result.get("success", False):
            kid = result["kernel_id"]
            def symplex_fma():
                return jit_execute(kid, [dst, a, b, c])
            t_symplex = bench(symplex_fma)
            gf_symplex = gflops(flops, t_symplex)
    except (ImportError, Exception) as e:
        print(f"  SympleX native: not available ({e})")

    print(f"  NumPy:   {t_numpy*1e6:10.1f} us  ({gf_numpy:.2f} GFLOPS)")
    if t_symplex is not None:
        speedup = t_numpy / t_symplex
        print(f"  SympleX: {t_symplex*1e6:10.1f} us  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    else:
        print(f"  SympleX: N/A")

    # Also benchmark chained elementwise (A + B) * C - D
    d = np.random.randn(N).astype(np.float64)
    def numpy_chain():
        return (a + b) * c - d

    t_chain = bench(numpy_chain)
    flops_chain = 3.0 * N
    gf_chain = gflops(flops_chain, t_chain)
    print(f"\n  Chained (A+B)*C-D, NumPy: {t_chain*1e6:.1f} us  ({gf_chain:.2f} GFLOPS)")

    return {"numpy_fma": t_numpy, "symplex": t_symplex, "numpy_chain": t_chain}


# ── Category 4: Physics Integrators ────────────────────────────────────

def benchmark_integrator(n_particles: int = 500000, dt: float = 0.001):
    """Symplectic Euler integrator: q += dt*p, p -= dt*force."""
    print(f"\n{'='*60}")
    print(f"Category 4: Physics Integrator — {n_particles} particles (dt={dt})")
    print(f"{'='*60}")

    q = np.random.randn(n_particles).astype(np.float64)
    p = np.random.randn(n_particles).astype(np.float64)
    force = np.random.randn(n_particles).astype(np.float64)
    q_copy = q.copy()
    p_copy = p.copy()

    def numpy_integrator():
        q_new = q_copy + dt * p_copy
        p_new = p_copy - dt * force
        return q_new, p_new

    t_numpy = bench(numpy_integrator)
    flops = 4.0 * n_particles  # 2 mul + 2 add per particle
    gf_numpy = gflops(flops, t_numpy)

    # SympleX JIT
    t_symplex = None
    gf_symplex = None
    try:
        from symplex._symplex_core import jit_compile, jit_execute

        shape_info = {"n_particles": n_particles, "dt": dt}
        result = jit_compile("integrator", shape_info)
        if result.get("success", False):
            kid = result["kernel_id"]
            def symplex_integrator():
                return jit_execute(kid, [q_copy, p_copy, force])
            t_symplex = bench(symplex_integrator)
            gf_symplex = gflops(flops, t_symplex)
    except (ImportError, Exception) as e:
        print(f"  SympleX native: not available ({e})")

    print(f"  NumPy:   {t_numpy*1e6:10.1f} us  ({gf_numpy:.2f} GFLOPS)")
    if t_symplex is not None:
        speedup = t_numpy / t_symplex
        print(f"  SympleX: {t_symplex*1e6:10.1f} us  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    else:
        print(f"  SympleX: N/A")

    # Multi-step integrator benchmark (10 steps fused vs separate)
    n_steps = 10
    def numpy_multistep():
        q_local = q.copy()
        p_local = p.copy()
        for _ in range(n_steps):
            q_local = q_local + dt * p_local
            p_local = p_local - dt * force  # simplified: force doesn't change
        return q_local, p_local

    t_multi = bench(numpy_multistep)
    flops_multi = 4.0 * n_particles * n_steps
    gf_multi = gflops(flops_multi, t_multi)
    print(f"\n  10-step integrator, NumPy: {t_multi*1e3:.2f} ms  ({gf_multi:.2f} GFLOPS)")

    return {"numpy": t_numpy, "symplex": t_symplex, "numpy_multi": t_multi}


# ── Category 5: Tensor Field Operations ────────────────────────────────

def benchmark_tensor_field(rows: int = 512, cols: int = 512, dx: float = 1.0, dy: float = 1.0):
    """Divergence of vector field: div = dVx/dx + dVy/dy (central differences)."""
    print(f"\n{'='*60}")
    print(f"Category 5: Tensor Field — {rows}x{cols} divergence (dx={dx}, dy={dy})")
    print(f"{'='*60}")

    Vx = np.random.randn(rows, cols).astype(np.float64)
    Vy = np.random.randn(rows, cols).astype(np.float64)
    out = np.zeros((rows, cols), dtype=np.float64)

    def numpy_divergence():
        result = np.zeros_like(Vx)
        # Central differences for interior points
        result[1:-1, 1:-1] = (
            (Vx[1:-1, 2:] - Vx[1:-1, :-2]) / (2 * dx) +
            (Vy[2:, 1:-1] - Vy[:-2, 1:-1]) / (2 * dy)
        )
        return result

    t_numpy = bench(numpy_divergence)
    flops = 6.0 * (rows - 2) * (cols - 2)  # 2 subtractions + 2 divisions + 1 add per point
    gf_numpy = gflops(flops, t_numpy)

    # SympleX JIT
    t_symplex = None
    gf_symplex = None
    try:
        from symplex._symplex_core import jit_compile, jit_execute

        shape_info = {"rows": rows, "cols": cols, "dx": dx, "dy": dy}
        result = jit_compile("tensor_field", shape_info)
        if result.get("success", False):
            kid = result["kernel_id"]
            def symplex_divergence():
                return jit_execute(kid, [out, Vx, Vy])
            t_symplex = bench(symplex_divergence)
            gf_symplex = gflops(flops, t_symplex)
    except (ImportError, Exception) as e:
        print(f"  SympleX native: not available ({e})")

    print(f"  NumPy:   {t_numpy*1e6:10.1f} us  ({gf_numpy:.2f} GFLOPS)")
    if t_symplex is not None:
        speedup = t_numpy / t_symplex
        print(f"  SympleX: {t_symplex*1e6:10.1f} us  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    else:
        print(f"  SympleX: N/A")

    return {"numpy": t_numpy, "symplex": t_symplex}


# ── MCMC Benchmark ─────────────────────────────────────────────────────

def benchmark_mcmc(n_steps: int = 1000, n_dims: int = 100):
    """MCMC transition kernel: deterministic energy function + force computation."""
    print(f"\n{'='*60}")
    print(f"Category (MCMC): Transition Kernel — {n_steps} steps, {n_dims} dims")
    print(f"{'='*60}")

    q = np.random.randn(n_dims).astype(np.float64)
    p = np.random.randn(n_dims).astype(np.float64)
    dt = 0.001

    def numpy_mcmc_kernel():
        """Deterministic part of HMC transition kernel."""
        # Force = -gradient of potential (harmonic: V = 0.5*q^2, so F = -q)
        force = -q
        # Symplectic Euler step
        q_new = q + dt * p
        p_new = p - dt * force
        # Energy = 0.5 * (p^2 + q^2)
        kinetic = 0.5 * np.sum(p_new**2)
        potential = 0.5 * np.sum(q_new**2)
        return q_new, p_new, kinetic + potential

    t_numpy = bench(lambda: [numpy_mcmc_kernel() for _ in range(n_steps)], warmup=1, iters=5)
    flops = n_steps * n_dims * 10.0  # ~10 FLOPs per dim per step
    gf_numpy = gflops(flops, t_numpy)

    print(f"  NumPy ({n_steps} steps): {t_numpy*1e3:.2f} ms  ({gf_numpy:.2f} GFLOPS)")

    # SympleX @jit(mcmc=True) — try the decorator
    t_symplex = None
    try:
        import symplex

        @symplex.jit(mcmc=True)
        def hmc_kernel(q, p, dt=0.001):
            force = -q
            q_new = q + dt * p
            p_new = p - dt * force
            energy = 0.5 * (p_new * p_new + q_new * q_new).sum()
            return q_new, p_new, energy

        # Warmup
        _ = hmc_kernel(q, p)
        def symplex_mcmc():
            for _ in range(n_steps):
                _ = hmc_kernel(q, p)

        t_symplex = bench(symplex_mcmc, warmup=1, iters=5)
        gf_symplex = gflops(flops, t_symplex)
        speedup = t_numpy / t_symplex
        print(f"  SympleX @jit(mcmc=True) ({n_steps} steps): {t_symplex*1e3:.2f} ms  ({gf_symplex:.2f} GFLOPS)  [{speedup:.2f}x]")
    except (ImportError, Exception) as e:
        print(f"  SympleX @jit(mcmc=True): not available ({e})")

    return {"numpy": t_numpy, "symplex": t_symplex}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       SympleX Physics Benchmark Suite v2.0              ║")
    print("║       Polyhedral Tensor Superoptimizer                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Detect SIMD level
    try:
        from symplex._symplex_core import jit_info
        info = jit_info()
        print(f"\nJIT Engine Info:\n{info}")
    except ImportError:
        print("\nNote: SympleX native engine not available, using NumPy-only benchmarks")

    print(f"\nNumPy version: {np.__version__}")
    print(f"CPU cores: {np.show_config() if hasattr(np, 'show_config') else 'N/A'}")

    results = {}

    # Run all benchmarks
    results["stencil_512"] = benchmark_stencil(512, 512, 0.25)
    results["stencil_1024"] = benchmark_stencil(1024, 1024, 0.25)
    results["nbody_128"] = benchmark_nbody(128, 0.01)
    results["nbody_256"] = benchmark_nbody(256, 0.01)
    results["la_chains"] = benchmark_la_chains()
    results["integrator"] = benchmark_integrator(500000, 0.001)
    results["tensor_field"] = benchmark_tensor_field(512, 512, 1.0, 1.0)
    results["mcmc"] = benchmark_mcmc(1000, 100)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    wins = 0
    losses = 0
    ties = 0
    for name, r in results.items():
        np_t = r.get("numpy", r.get("numpy_fma"))
        sx_t = r.get("symplex")
        if np_t is not None and sx_t is not None:
            speedup = np_t / sx_t
            if speedup > 1.05:
                wins += 1
                marker = "WIN"
            elif speedup < 0.95:
                losses += 1
                marker = "LOSS"
            else:
                ties += 1
                marker = "TIE"
            print(f"  {name:25s}: {speedup:.2f}x  [{marker}]")
        else:
            print(f"  {name:25s}: N/A")

    total = wins + losses + ties
    if total > 0:
        print(f"\n  Total: {wins} wins, {losses} losses, {ties} ties out of {total} benchmarks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
