#!/usr/bin/env python3
"""
SympleX Physics Benchmark Suite — NumPy Reference Implementation

Runs the same 5 physics workload categories as the Rust benchmark
(physics_bench.rs) using pure NumPy for comparison.

Usage:
    python3 benchmarks/bench_physics_numpy.py

Categories:
  1. PDE Stencils (heat & wave equations)
  2. N-body simulation
  3. Linear algebra chains (fused matmul)
  4. Physics integrators (Euler, RK4)
  5. Tensor field operations (gradient, divergence, curl, laplacian)
"""

import numpy as np
import timeit
import sys

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def bench(fn, warmup=3, iters=15):
    """Run fn multiple times, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = timeit.timeit(fn, number=1)
        times.append(t * 1000.0)
    times.sort()
    return times[len(times) // 2]

def print_section(title):
    print()
    print("=" * 67)
    print(f"  {title}")
    print("=" * 67)

def print_result(name, time_ms, gflops):
    print(f"  {name:<24s}  {time_ms:>10.3f} ms  {gflops:>9.3f} GFLOPS")

# ══════════════════════════════════════════════════════════════════════════════
# 1. PDE Stencils
# ══════════════════════════════════════════════════════════════════════════════

def bench_pde_stencils():
    print_section("1. PDE Stencils — Heat & Wave Equations (256×256, 100 steps)")

    nx, ny = 256, 256
    steps = 100
    alpha = 0.01
    dt = 0.1
    c = 1.0
    coeff_heat = alpha * dt
    coeff_wave = c * c * dt * dt

    # Initialize with Gaussian bump
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_init = np.exp(-0.5 * ((X - nx/2)**2 + (Y - ny/2)**2) / 100.0)

    # ── Heat equation ────────────────────────────────────────────────────
    def heat_step():
        u = u_init.copy()
        for _ in range(steps):
            u_new = u.copy()
            u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + coeff_heat * (
                u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
            )
            u = u_new
        return u

    heat_time = bench(heat_step, warmup=2, iters=10)
    interior = (nx - 2) * (ny - 2)
    # 10 flops per interior point per step
    heat_flops = 10.0 * interior * steps
    heat_gflops = heat_flops / (heat_time * 1e-3) / 1e9

    print("\n  Heat Equation: u_new[i,j] = u[i,j] + α·dt·(stencil)")
    print_result("NumPy", heat_time, heat_gflops)

    # ── Wave equation ────────────────────────────────────────────────────
    def wave_step():
        u = u_init.copy()
        u_old = u.copy()
        for _ in range(steps):
            u_new = 2.0 * u - u_old + coeff_wave * (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4.0 * u
            )
            u_old = u
            u = u_new
        return u

    # More efficient wave with slicing
    def wave_step_sliced():
        u = u_init.copy()
        u_old = u.copy()
        for _ in range(steps):
            u_new = u.copy()
            u_new[1:-1, 1:-1] = 2.0 * u[1:-1, 1:-1] - u_old[1:-1, 1:-1] + coeff_wave * (
                u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
            )
            u_old = u
            u = u_new
        return u

    wave_time = bench(wave_step_sliced, warmup=2, iters=10)
    # ~12 flops per interior point per step
    wave_flops = 12.0 * interior * steps
    wave_gflops = wave_flops / (wave_time * 1e-3) / 1e9

    print("\n  Wave Equation: u_new = 2·u - u_old + c²·dt²·(stencil)")
    print_result("NumPy", wave_time, wave_gflops)

# ══════════════════════════════════════════════════════════════════════════════
# 2. N-body Simulation
# ══════════════════════════════════════════════════════════════════════════════

def bench_nbody():
    print_section("2. N-body Simulation (N=500, 10 steps, brute-force O(N²))")

    n = 500
    steps = 10
    G = 6.674e-11
    epsilon = 0.01
    dt = 0.001

    # Initialize particles
    angles = 2.0 * np.pi * np.arange(n) / n
    pos = np.zeros((n, 3))
    pos[:, 0] = 100.0 * np.cos(angles)
    pos[:, 1] = 100.0 * np.sin(angles)
    pos[:, 2] = np.sin(np.arange(n) * 0.1) * 10.0
    vel = np.zeros((n, 3))
    mass = 1e6 + np.arange(n, dtype=np.float64) * 100.0

    def nbody_step():
        p = pos.copy()
        v = vel.copy()
        for _ in range(steps):
            # Compute pairwise differences
            # diff[i,j] = pos[j] - pos[i], shape (N, N, 3)
            diff = p[np.newaxis, :, :] - p[:, np.newaxis, :]
            dist_sq = np.sum(diff**2, axis=2) + epsilon  # (N, N)
            inv_dist3 = dist_sq**(-1.5)
            np.fill_diagonal(inv_dist3, 0.0)

            # Force: F_ij = G * mi * mj / dist^3 * diff
            # Acceleration on i: a_i = sum_j G * mj / dist^3 * diff[i,j]
            f_over_r3 = G * mass[np.newaxis, :] * inv_dist3  # (N, N)
            a = np.sum(f_over_r3[:, :, np.newaxis] * diff, axis=1)  # (N, 3)

            v = v + a * dt
            p = p + v * dt
        return p

    nbody_time = bench(nbody_step, warmup=1, iters=5)
    # ~20 flops per pair per step
    nbody_flops = (n * (n - 1) / 2) * 20.0 * steps
    nbody_gflops = nbody_flops / (nbody_time * 1e-3) / 1e9

    print_result("NumPy (vectorized)", nbody_time, nbody_gflops)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Linear Algebra Chains
# ══════════════════════════════════════════════════════════════════════════════

def bench_linalg_chains():
    print_section("3. Linear Algebra Chains (128×128 matrices)")

    dim = 128
    n = dim * dim

    def make_mat():
        return np.sin(np.arange(n, dtype=np.float64) * 0.001).reshape(dim, dim)

    # ── Single matmul: C = A @ B ─────────────────────────────────────────
    print("\n  Benchmark: C = A @ B")

    A = make_mat()
    B = make_mat()

    def matmul_np():
        C = A @ B
        return C

    matmul_time = bench(matmul_np, warmup=5, iters=15)
    matmul_flops = 2.0 * dim**3
    matmul_gflops = matmul_flops / (matmul_time * 1e-3) / 1e9

    print_result("NumPy (@ / BLAS)", matmul_time, matmul_gflops)

    # ── Chain: result = A @ B + C @ D + E ────────────────────────────────
    print("\n  Benchmark: result = A @ B + C @ D + E (separate BLAS calls)")

    A_ = make_mat(); B_ = make_mat()
    C_ = make_mat(); D_ = make_mat()
    E_ = make_mat()

    def chain_separate():
        AB = A_ @ B_      # BLAS call 1
        CD = C_ @ D_      # BLAS call 2
        result = AB + CD + E_  # element-wise adds
        return result

    chain_time = bench(chain_separate, warmup=5, iters=15)
    chain_flops = 2.0 * 2.0 * dim**3 + 2.0 * n  # 2 matmuls + 2 adds
    chain_gflops = chain_flops / (chain_time * 1e-3) / 1e9

    print_result("NumPy (3 separate ops)", chain_time, chain_gflops)

    # ── Fused: (A @ B) * C + D ───────────────────────────────────────────
    print("\n  Benchmark: result = (A @ B) * C + D")

    A2 = make_mat(); B2 = make_mat()
    C2 = make_mat(); D2 = make_mat()

    def muladd_np():
        AB = A2 @ B2
        result = AB * C2 + D2
        return result

    muladd_time = bench(muladd_np, warmup=5, iters=15)
    muladd_flops = 2.0 * dim**3 + 2.0 * n
    muladd_gflops = muladd_flops / (muladd_time * 1e-3) / 1e9

    print_result("NumPy (2 separate ops)", muladd_time, muladd_gflops)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Physics Integrators (Euler, RK4)
# ══════════════════════════════════════════════════════════════════════════════

def bench_integrators():
    print_section("4. Physics Integrators — Euler & RK4 (Lorenz, 10000 steps)")

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dt = 0.01
    n_steps = 10000

    def lorenz(x, y, z):
        return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)

    # ── Euler ────────────────────────────────────────────────────────────
    def euler_lorenz():
        x, y, z = 1.0, 1.0, 1.0
        for _ in range(n_steps):
            dx, dy, dz = lorenz(x, y, z)
            x += dt * dx
            y += dt * dy
            z += dt * dz
        return x, y, z

    euler_time = bench(euler_lorenz, warmup=3, iters=15)
    euler_flops = 9.0 * n_steps
    euler_gflops = euler_flops / (euler_time * 1e-3) / 1e9

    print("\n  Lorenz System (3D chaotic ODE): σ=10, ρ=28, β=8/3")
    print_result("Euler (Python loop)", euler_time, euler_gflops)

    # ── RK4 ──────────────────────────────────────────────────────────────
    def rk4_lorenz():
        x, y, z = 1.0, 1.0, 1.0
        for _ in range(n_steps):
            k1x, k1y, k1z = lorenz(x, y, z)
            k2x, k2y, k2z = lorenz(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
            k3x, k3y, k3z = lorenz(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
            k4x, k4y, k4z = lorenz(x + dt*k3x, y + dt*k3y, z + dt*k3z)
            x += dt/6.0 * (k1x + 2.0*k2x + 2.0*k3x + k4x)
            y += dt/6.0 * (k1y + 2.0*k2y + 2.0*k3y + k4y)
            z += dt/6.0 * (k1z + 2.0*k2z + 2.0*k3z + k4z)
        return x, y, z

    rk4_time = bench(rk4_lorenz, warmup=3, iters=15)
    rk4_flops = 54.0 * n_steps
    rk4_gflops = rk4_flops / (rk4_time * 1e-3) / 1e9

    print_result("RK4 (Python loop)", rk4_time, rk4_gflops)

    # ── 2D Diffusion (explicit Euler on grid) ────────────────────────────
    print("\n  2D Diffusion: Explicit Euler on 256×256 grid, 100 steps")

    nx, ny = 256, 256
    alpha_d = 0.01
    dt_d = 0.1
    coeff_d = alpha_d * dt_d
    steps = 100

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    u_init = np.exp(-0.5 * ((X - nx/2)**2 + (Y - ny/2)**2) / 100.0)

    def diffusion_euler():
        u = u_init.copy()
        for _ in range(steps):
            u_new = u.copy()
            u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + coeff_d * (
                u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
            )
            u = u_new
        return u

    diff_time = bench(diffusion_euler, warmup=2, iters=10)
    interior = (nx - 2) * (ny - 2)
    diff_flops = 10.0 * interior * steps
    diff_gflops = diff_flops / (diff_time * 1e-3) / 1e9

    print_result("NumPy (vectorized)", diff_time, diff_gflops)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Tensor Field Operations
# ══════════════════════════════════════════════════════════════════════════════

def bench_tensor_fields():
    print_section("5. Tensor Field Operations (512×512 grid)")

    nx, ny = 512, 512
    n = nx * ny
    dx = 0.01

    # Initialize fields
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing='ij')
    f = np.sin(X) * np.cos(Y)
    u = np.cos(X * Y)
    v = np.sin(X + Y)

    # ── Gradient ─────────────────────────────────────────────────────────
    def gradient_np():
        grad_x = np.zeros_like(f)
        grad_y = np.zeros_like(f)
        grad_x[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2.0 * dx)
        grad_y[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2.0 * dx)
        return grad_x, grad_y

    grad_time = bench(gradient_np, warmup=3, iters=15)
    interior = (nx - 2) * (ny - 2)
    grad_flops = 4.0 * interior
    grad_gflops = grad_flops / (grad_time * 1e-3) / 1e9

    print("\n  grad_x = (f[i+1,j] - f[i-1,j]) / (2·dx), same for y")
    print_result("Gradient NumPy", grad_time, grad_gflops)

    # ── Divergence ───────────────────────────────────────────────────────
    def divergence_np():
        div = np.zeros_like(u)
        du_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
        dv_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx)
        div[1:-1, 1:-1] = du_dx + dv_dy
        return div

    div_time = bench(divergence_np, warmup=3, iters=15)
    div_flops = 7.0 * interior
    div_gflops = div_flops / (div_time * 1e-3) / 1e9

    print("\n  div = du/dx + dv/dy")
    print_result("Divergence NumPy", div_time, div_gflops)

    # ── Curl (2D scalar) ─────────────────────────────────────────────────
    def curl_np():
        curl = np.zeros_like(u)
        dv_dx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
        du_dy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
        curl[1:-1, 1:-1] = dv_dx - du_dy
        return curl

    curl_time = bench(curl_np, warmup=3, iters=15)
    curl_flops = 7.0 * interior
    curl_gflops = curl_flops / (curl_time * 1e-3) / 1e9

    print("\n  curl = dv/dx - du/dy")
    print_result("Curl NumPy", curl_time, curl_gflops)

    # ── Laplacian ────────────────────────────────────────────────────────
    def laplacian_np():
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            f[2:, 1:-1] + f[:-2, 1:-1] +
            f[1:-1, 2:] + f[1:-1, :-2] - 4.0 * f[1:-1, 1:-1]
        ) / (dx * dx)
        return lap

    lap_time = bench(laplacian_np, warmup=3, iters=15)
    lap_flops = 6.0 * interior
    lap_gflops = lap_flops / (lap_time * 1e-3) / 1e9

    print("\n  lap = (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4·f[i,j]) / dx²")
    print_result("Laplacian NumPy", lap_time, lap_gflops)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     SympleX Physics Benchmark — NumPy Reference                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  NumPy version: {np.__version__}")
    print(f"  BLAS config:   {np.show_config()}")
    print()

    bench_pde_stencils()
    bench_nbody()
    bench_linalg_chains()
    bench_integrators()
    bench_tensor_fields()

    print()
    print("=" * 67)
    print("  NumPy benchmark complete.")
    print("  Compare these numbers with the Rust benchmark output.")
    print("=" * 67)
