#!/usr/bin/env python3
"""
SympleX vs NumPy Benchmark Suite
==================================

Benchmarks across 5 physics categories:
1. PDE Stencils (2D Laplacian)
2. N-body (gravitational forces)
3. Linear Algebra chains (matmul at various sizes)
4. Physics Integrators (Euler, RK4, Verlet)
5. Tensor Field Operations (gradient magnitude, elementwise)

Usage:
    python benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctypes
import numpy as np
import time

# Load kernel library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "libsymplex_kernels.so")
if not os.path.exists(lib_path):
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsymplex_kernels.so")
lib = ctypes.CDLL(lib_path)

wins = 0
losses = 0

def benchmark(name, fn_symplex, fn_numpy, warmup=3, runs=10):
    global wins, losses
    for _ in range(warmup):
        fn_symplex()
        fn_numpy()
    
    t0 = time.perf_counter()
    for _ in range(runs):
        fn_symplex()
    t_sx = (time.perf_counter() - t0) / runs
    
    t0 = time.perf_counter()
    for _ in range(runs):
        fn_numpy()
    t_np = (time.perf_counter() - t0) / runs
    
    speedup = t_np / t_sx if t_sx > 0 else float('inf')
    verdict = "WIN" if speedup > 1.0 else "LOSS"
    if speedup > 1.0:
        wins += 1
    else:
        losses += 1
    print(f"  {name:40s}  SX={t_sx*1e6:10.1f}us  NP={t_np*1e6:10.1f}us  {speedup:6.2f}x  [{verdict}]")
    return speedup

print("=" * 90)
print("SympleX vs NumPy Benchmark Suite")
print("=" * 90)

# ═══════════════════════════════════════════════════════════════════════
# 1. PDE STENCILS
# ═══════════════════════════════════════════════════════════════════════
print("\n1. PDE Stencils (5-point 2D Laplacian)")

for N in [64, 128, 256]:
    in_g = np.random.randn(N, N).astype(np.float32)
    out_g = np.zeros((N, N), dtype=np.float32)
    dx = 0.01
    
    def sx_stencil(_in=in_g, _out=out_g, _N=N):
        _out[:] = 0
        lib.symplex_stencil_2d(_out.ctypes.data, _in.ctypes.data, ctypes.c_int64(_N), ctypes.c_float(dx))
    
    def np_stencil(_in=in_g, _N=N):
        lap = np.zeros_like(_in)
        lap[1:-1,1:-1] = (_in[:-2,1:-1] + _in[2:,1:-1] + _in[1:-1,:-2] + _in[1:-1,2:] - 4*_in[1:-1,1:-1])/(dx*dx)
    
    benchmark(f"Laplacian {N}x{N}", sx_stencil, np_stencil)

# ═══════════════════════════════════════════════════════════════════════
# 2. N-BODY
# ═══════════════════════════════════════════════════════════════════════
print("\n2. N-body Simulations (Gravitational Forces)")

for n in [100, 500, 1000]:
    px = np.random.randn(n).astype(np.float32)
    py = np.random.randn(n).astype(np.float32)
    mass = np.ones(n, dtype=np.float32)
    fx = np.zeros(n, dtype=np.float32)
    fy = np.zeros(n, dtype=np.float32)
    G = 6.674e-11; soft = 0.1
    
    def sx_nbody(_px=px, _py=py, _m=mass, _fx=fx, _fy=fy, _n=n):
        _fx[:] = 0; _fy[:] = 0
        lib.symplex_nbody_forces(_px.ctypes.data, _py.ctypes.data, _m.ctypes.data,
                                  _fx.ctypes.data, _fy.ctypes.data, ctypes.c_int64(_n),
                                  ctypes.c_float(G), ctypes.c_float(soft))
    
    def np_nbody(_px=px, _py=py, _m=mass, _n=n):
        _px2 = _px.reshape(-1,1); _py2 = _py.reshape(-1,1)
        dx_ = _px.reshape(1,-1) - _px2
        dy_ = _py.reshape(1,-1) - _py2
        r2 = dx_**2 + dy_**2 + soft**2
        np.fill_diagonal(r2, 1.0)
        f = G * _m.reshape(-1,1) * _m.reshape(1,-1) / r2
        np.fill_diagonal(f, 0.0)
        _ = np.sum(f * dx_, axis=1)
        _ = np.sum(f * dy_, axis=1)
    
    benchmark(f"N-body {n} particles", sx_nbody, np_nbody)

# ═══════════════════════════════════════════════════════════════════════
# 3. LINEAR ALGEBRA CHAINS
# ═══════════════════════════════════════════════════════════════════════
print("\n3. Linear Algebra (Matmul at various sizes)")
print("   Note: NumPy uses BLAS (OpenBLAS/MKL) for matmul — very hard to beat")

for m, n, k in [(8,8,8), (16,16,16), (32,32,32), (64,64,64)]:
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    C = np.zeros((m, n), dtype=np.float32)
    
    def sx_mm(_A=A, _B=B, _C=C, _m=m, _n=n, _k=k):
        _C[:] = 0
        lib.symplex_matmul_f32(_A.ctypes.data, _B.ctypes.data, _C.ctypes.data,
                                ctypes.c_int64(_m), ctypes.c_int64(_n), ctypes.c_int64(_k))
    
    def np_mm(_A=A, _B=B):
        _ = _A @ _B
    
    benchmark(f"Matmul {m}x{k}x{n}", sx_mm, np_mm)

# Elementwise chains (where SympleX should win)
print("\n3b. Elementwise LA Chains (add/mul chains)")

for n in [10000, 100000, 1000000]:
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    
    def sx_chain(_a=a, _b=b, _out=out, _n=n):
        lib.symplex_mul_f32(_out.ctypes.data, _a.ctypes.data, _b.ctypes.data, ctypes.c_int64(_n))
    
    def np_chain(_a=a, _b=b):
        _ = _a * _b
    
    benchmark(f"Multiply chain {n}", sx_chain, np_chain)

# ═══════════════════════════════════════════════════════════════════════
# 4. PHYSICS INTEGRATORS
# ═══════════════════════════════════════════════════════════════════════
print("\n4. Physics Integrators")

# Euler step
for n in [10000, 100000]:
    pos = np.random.randn(n).astype(np.float32)
    vel = np.random.randn(n).astype(np.float32)
    force = np.random.randn(n).astype(np.float32)
    
    def sx_euler(_p=pos.copy(), _v=vel.copy(), _f=force, _n=n):
        lib.symplex_euler_step(_p.ctypes.data, _v.ctypes.data, _f.ctypes.data,
                                ctypes.c_float(1.0), ctypes.c_float(0.001), ctypes.c_int64(_n))
    
    def np_euler(_p=pos.copy(), _v=vel.copy(), _f=force, _n=n):
        _v += (_f / 1.0) * 0.001
        _p += _v * 0.001
    
    benchmark(f"Euler step {n}", sx_euler, np_euler)

# RK4 step
for n in [10000, 100000]:
    x = np.random.randn(n).astype(np.float32)
    v = np.random.randn(n).astype(np.float32)
    
    def sx_rk4(_x=x.copy(), _v=v.copy(), _n=n):
        lib.symplex_rk4_step(_x.ctypes.data, _v.ctypes.data,
                              ctypes.c_float(10.0), ctypes.c_float(0.1), ctypes.c_float(1.0),
                              ctypes.c_float(0.001), ctypes.c_int64(_n))
    
    def np_rk4(_x=x.copy(), _v=v.copy(), _n=n):
        k1x = _v; k1v = (-10.0*_x - 0.1*_v)/1.0
        k2x = _v + 0.5*0.001*k1v; k2v = (-10.0*(_x+0.5*0.001*k1x) - 0.1*k2x)/1.0
        k3x = _v + 0.5*0.001*k2v; k3v = (-10.0*(_x+0.5*0.001*k2x) - 0.1*k3x)/1.0
        k4x = _v + 0.001*k3v; k4v = (-10.0*(_x+0.001*k3x) - 0.1*k4x)/1.0
        _x += (0.001/6)*(k1x + 2*k2x + 2*k3x + k4x)
        _v += (0.001/6)*(k1v + 2*k2v + 2*k3v + k4v)
    
    benchmark(f"RK4 step {n}", sx_rk4, np_rk4)

# ═══════════════════════════════════════════════════════════════════════
# 5. TENSOR FIELD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n5. Tensor Field Operations")

for n in [10000, 100000, 1000000]:
    gx = np.random.randn(n).astype(np.float32)
    gy = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    
    def sx_grad(_gx=gx, _gy=gy, _out=out, _n=n):
        lib.symplex_grad_magnitude(_out.ctypes.data, _gx.ctypes.data, _gy.ctypes.data, ctypes.c_int64(_n))
    
    def np_grad(_gx=gx, _gy=gy):
        _ = _gx**2 + _gy**2
    
    benchmark(f"Grad magnitude {n}", sx_grad, np_grad)

# Elementwise add on large arrays
for n in [100000, 1000000]:
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    
    def sx_add(_a=a, _b=b, _out=out, _n=n):
        lib.symplex_add_f32(_out.ctypes.data, _a.ctypes.data, _b.ctypes.data, ctypes.c_int64(_n))
    
    def np_add(_a=a, _b=b):
        _ = _a + _b
    
    benchmark(f"Elementwise add {n}", sx_add, np_add)

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"SUMMARY: {wins} wins / {losses} losses vs NumPy")
print("=" * 90)

if wins > losses:
    print("SympleX WINNER — optimized C kernels outperform NumPy in physics workloads!")
elif wins == losses:
    print("TIED — SympleX matches NumPy performance")
else:
    print("NumPy faster overall — SympleX needs SIMD vectorization to close the gap")
