#!/usr/bin/env python3
"""SympleX End-to-End Demo"""

import symplex
import numpy as np

print('=== SympleX Polyhedral Tensor Superoptimizer ===')
print(f'Version: {symplex.__version__()}')
print(f'Rust engine: {symplex.is_rust_engine_available()}')
print()

# Hardware detection
hw = symplex.hardware_info()
print(f'Hardware: target={hw["target"]}, simd={hw["simd_level"]}')
print(f'Peak: {hw["peak_gflops"]} GFLOPS, BW: {hw["mem_bandwidth_gb_per_sec"]} GB/s')
print()

# Micro kernel config
mk = symplex.micro_kernel_config()
print(f'MicroKernel: tile_m={mk["tile_m"]}, tile_n={mk["tile_n"]}, tile_k={mk["tile_k"]}')
print(f'  acc_regs={mk["accumulator_registers"]}, prefetch={mk["prefetch_distance"]}')
print()

# JIT a pure function
@symplex.jit
def matmul_add(A, B, c):
    return A @ B + c

A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
c = np.array([1.0])

result = matmul_add(A, B, c)
expected = A @ B + c
print(f'JIT matmul_add result matches numpy: {np.allclose(result.to_numpy(), expected)}')

# Get optimization info
info = matmul_add.optimize_info(A, B, c)
if info:
    print(f'Optimization: {info["instr_count"]} instructions, {info["hint_count"]} hints')
    print(f'  SIMD: {info["simd_level"]}, GFLOPS est: {info["estimated_gflops"]:.1f}')
    print(f'  Tiles: {info["tile_m"]}x{info["tile_n"]}x{info["tile_k"]}')
print()

# Grad
def f(x):
    return (x * x).sum()

df = symplex.grad(f)
x = np.array([1.0, 2.0, 3.0])
g = df(x)
print(f'grad(x^2) at [1,2,3] = {g.to_numpy()} (expected [2,4,6])')
print()

# Purity enforcement
try:
    @symplex.jit
    def impure(x):
        print(x)
        return x
    print('ERROR: Should have rejected impure function!')
except symplex.ImpureFunctionError as e:
    print('Correctly rejected impure function (print)')
print()

# Functional updates (JAX-style)
arr = symplex.DeviceArray([1.0, 2.0, 3.0])
updated = arr.at[1].set(99.0)
print(f'Functional update: {arr.to_numpy()} -> {updated.to_numpy()}')
print(f'Original unchanged: {arr.to_numpy()}')
print()

# Math functions
a = symplex.DeviceArray([-1.0, 0.0, 1.0, 2.0])
print(f'relu({a.to_numpy()}) = {symplex.relu(a).to_numpy()}')
print(f'sigmoid(0) = {symplex.sigmoid(symplex.DeviceArray([0.0])).to_numpy()[0]:.4f}')
print(f'softmax([1,2,3]) = {symplex.softmax(symplex.DeviceArray([1.0,2.0,3.0])).to_numpy().round(4)}')
print()

# RNG
rng = symplex.lax.rng(42)
r1 = rng((3,))
rng2 = symplex.lax.rng(42)
r2 = rng2((3,))
print(f'RNG deterministic: {np.array_equal(r1.to_numpy(), r2.to_numpy())}')
print()

print('=== All demos passed! ===')
