"""SympleX – Polyhedral Tensor Superoptimizer

A JAX-style Python library that enforces function purity and optimizes
numerical computations via polyhedral analysis, SIMD tiling, and
hardware-aware micro-kernel generation.

Key APIs:
  - ``symplex.jit``       — JIT-compile a pure function with polyhedral optimization
  - ``symplex.grad``      — Reverse-mode automatic differentiation
  - ``symplex.DeviceArray`` — Immutable array with functional updates (.at[].set())

Purity rules (like JAX):
  - No side effects (no print, IO, global mutation)
  - No in-place array mutation (use x = x.at[idx].set(val))
  - Only pure operations (arithmetic, numpy, symplex APIs)

Example::

    import symplex
    import numpy as np

    @symplex.jit
    def matmul_relu(A, B):
        C = A @ B
        return symplex.relu(C)

    A = np.random.randn(128, 64)
    B = np.random.randn(64, 256)
    result = matmul_relu(A, B)
"""

import numpy as np

from ._array import DeviceArray
from ._errors import (
    SympleXError,
    ImpureFunctionError,
    TracerError,
    CompilationError,
    ShapeError,
)
from ._jit import jit, grad, JitFunction
from . import linalg

# ── Type annotation helpers for @symplex.jit ──────────────────────────────
# These allow users to annotate function arguments with specific dtypes:
#   @symplex.jit
#   def compute(x: symplex.f32, y: symplex.f32) -> symplex.f32:
#       return x * 2.5 + y

class _DTypeAnnotation:
    """Base class for dtype annotations in function signatures."""
    def __init__(self, name, np_dtype):
        self.name = name
        self.np_dtype = np_dtype
    def __repr__(self):
        return f"symplex.{self.name}"
    def __call__(self, shape=None):
        """Allow f32['M,K'] syntax (future: symbolic shapes)."""
        return self

f32 = _DTypeAnnotation("f32", np.float32)
f64 = _DTypeAnnotation("f64", np.float64)
bf16 = _DTypeAnnotation("bf16", "bfloat16")
i8 = _DTypeAnnotation("i8", np.int8)
i16 = _DTypeAnnotation("i16", np.int16)
i32 = _DTypeAnnotation("i32", np.int32)
i64 = _DTypeAnnotation("i64", np.int64)
u8 = _DTypeAnnotation("u8", np.uint8)
u16 = _DTypeAnnotation("u16", np.uint16)
u32 = _DTypeAnnotation("u32", np.uint32)
u64 = _DTypeAnnotation("u64", np.uint64)

# Try to import the Rust engine
try:
    from ._symplex_core import (
        optimize_trace as _rust_optimize_trace,
        optimize_specialized as _rust_optimize_specialized,
        grad as _rust_grad,
        detect_hardware as _rust_detect_hardware,
        micro_kernel_config as _rust_micro_kernel_config,
        serialize_instructions as _rust_serialize_instructions,
    )
    _RUST_ENGINE_AVAILABLE = True
except ImportError:
    _RUST_ENGINE_AVAILABLE = False


# ── Functional math API (like jax.numpy / jax.lax) ──────────────────────────

# Import TracerVal for activation function dispatch (lazy to avoid circular import)
_TracrVal = None
def _get_tracer_val():
    global _TracrVal
    if _TracrVal is None:
        from ._tracer import TracerVal
        _TracrVal = TracerVal
    return _TracrVal


def relu(x):
    """ReLU activation: max(0, x)."""
    if isinstance(x, DeviceArray):
        return x.relu()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.relu()
    import numpy as np
    return np.maximum(x, 0)


def gelu(x):
    """GELU activation: x * Phi(x)."""
    if isinstance(x, DeviceArray):
        return x.gelu()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.gelu()
    import numpy as np
    x = np.asarray(x)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    if isinstance(x, DeviceArray):
        return x.sigmoid()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.sigmoid()
    import numpy as np
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Softmax: exp(x) / sum(exp(x)) along axis."""
    if isinstance(x, DeviceArray):
        return x.softmax(axis=axis)
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.softmax(axis=axis)
    import numpy as np
    x = np.asarray(x)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def exp(x):
    """Element-wise exponential."""
    if isinstance(x, DeviceArray):
        return x.exp()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.exp()
    import numpy as np
    return np.exp(x)


def log(x):
    """Element-wise natural logarithm."""
    if isinstance(x, DeviceArray):
        return x.log()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.log()
    import numpy as np
    return np.log(x)


def sqrt(x):
    """Element-wise square root."""
    if isinstance(x, DeviceArray):
        return x.sqrt()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.sqrt()
    import numpy as np
    return np.sqrt(x)


def sin(x):
    """Element-wise sine."""
    if isinstance(x, DeviceArray):
        return x.sin()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.sin()
    import numpy as np
    return np.sin(x)


def cos(x):
    """Element-wise cosine."""
    if isinstance(x, DeviceArray):
        return x.cos()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.cos()
    import numpy as np
    return np.cos(x)


def tanh(x):
    """Element-wise hyperbolic tangent."""
    if isinstance(x, DeviceArray):
        return x.tanh()
    TracerVal = _get_tracer_val()
    if isinstance(x, TracerVal):
        return x.tanh()
    import numpy as np
    return np.tanh(x)


def matmul(a, b):
    """Matrix multiplication using BLAS (via NumPy).

    NumPy's matmul is backed by optimized BLAS (OpenBLAS, MKL, or Apple
    Accelerate) which delivers near-peak performance on CPU. This is the
    recommended path for CPU matrix multiplications — SympleX's JIT-compiled
    matmul is used for fused kernels (e.g., matmul+bias+ReLU) where BLAS
    alone cannot eliminate the extra memory round-trips.

    For fused operations, use ``symplex.jit`` to compile the entire
    computation graph into a single kernel.

    Args:
        a: First matrix (M, K) or vector.
        b: Second matrix (K, N) or vector.

    Returns:
        DeviceArray with the result.
    """
    if isinstance(a, DeviceArray) and isinstance(b, DeviceArray):
        return DeviceArray._wrap(np.matmul(a._data, b._data))
    if isinstance(a, DeviceArray):
        return DeviceArray._wrap(np.matmul(a._data, b))
    if isinstance(b, DeviceArray):
        return DeviceArray._wrap(np.matmul(a, b._data))
    return DeviceArray._wrap(np.matmul(a, b))


def dot(a, b):
    """Dot product of two arrays."""
    if isinstance(a, DeviceArray) and isinstance(b, DeviceArray):
        return DeviceArray(a.to_numpy().dot(b.to_numpy()))
    import numpy as np
    return np.dot(a, b)


def sum(x, axis=None, keepdims=False):
    """Sum of array elements."""
    if isinstance(x, DeviceArray):
        return x.sum(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    """Mean of array elements."""
    if isinstance(x, DeviceArray):
        return x.mean(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    """Maximum of array elements."""
    if isinstance(x, DeviceArray):
        return x.max(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """Minimum of array elements."""
    if isinstance(x, DeviceArray):
        return x.min(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.min(x, axis=axis, keepdims=keepdims)


def clip(x, a_min, a_max):
    """Clip array values."""
    import numpy as np
    if isinstance(x, DeviceArray):
        return DeviceArray(np.clip(x.to_numpy(), a_min, a_max))
    return np.clip(x, a_min, a_max)


def where(condition, x, y):
    """Where with condition."""
    import numpy as np
    if isinstance(condition, DeviceArray):
        condition = condition.to_numpy()
    if isinstance(x, DeviceArray):
        x = x.to_numpy()
    if isinstance(y, DeviceArray):
        y = y.to_numpy()
    return DeviceArray(np.where(condition, x, y))


def concatenate(arrays, axis=0):
    """Concatenate arrays."""
    import numpy as np
    np_arrays = [a.to_numpy() if isinstance(a, DeviceArray) else a for a in arrays]
    return DeviceArray(np.concatenate(np_arrays, axis=axis))


def stack(arrays, axis=0):
    """Stack arrays."""
    import numpy as np
    np_arrays = [a.to_numpy() if isinstance(a, DeviceArray) else a for a in arrays]
    return DeviceArray(np.stack(np_arrays, axis=axis))


def reshape(x, shape):
    """Reshape array."""
    if isinstance(x, DeviceArray):
        return x.reshape(*shape)
    import numpy as np
    return np.reshape(x, shape)


def transpose(x, axes=None):
    """Transpose array."""
    if isinstance(x, DeviceArray):
        return x.transpose() if axes is None else x.transpose(*axes)
    import numpy as np
    return np.transpose(x, axes)


def zeros(shape, dtype=np.float32):
    """Create array of zeros."""
    return DeviceArray(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32):
    """Create array of ones."""
    return DeviceArray(np.ones(shape, dtype=dtype))


def array(data, dtype=np.float32):
    """Create a DeviceArray from data."""
    return DeviceArray(np.array(data, dtype=dtype))


def arange(start, stop=None, step=1, dtype=np.float32):
    """Create an array with evenly spaced values."""
    return DeviceArray(np.arange(start, stop, step, dtype=dtype))


def linspace(start, stop, num=50, dtype=np.float32):
    """Create an array with evenly spaced values over interval."""
    return DeviceArray(np.linspace(start, stop, num, dtype=dtype))


def eye(n, m=None, k=0, dtype=np.float32):
    """Create an identity matrix."""
    return DeviceArray(np.eye(n, m, k, dtype=dtype))


# ── lax module (like jax.lax) ───────────────────────────────────────────────

class lax:
    """Low-level array operations (like jax.lax).

    Provides control flow primitives that work with traced values.
    """

    @staticmethod
    def cond(pred, true_fun, false_fun, *operands):
        """Conditional execution (like jax.lax.cond)."""
        if isinstance(pred, DeviceArray):
            pred = pred.to_numpy()
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)

    @staticmethod
    def while_loop(cond_fun, body_fun, init_val):
        """While loop (like jax.lax.while_loop)."""
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    @staticmethod
    def fori_loop(lower, upper, body_fun, init_val):
        """For loop with integer range (like jax.lax.fori_loop)."""
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    @staticmethod
    def scan(f, init, xs, length=None, reverse=False):
        """Scan (sequential loop with carry, like jax.lax.scan)."""
        if isinstance(xs, DeviceArray):
            xs = xs.to_numpy()
        if not isinstance(xs, (list, tuple)):
            xs = [xs]

        carry = init
        ys = []
        n = length if length is not None else len(xs[0]) if hasattr(xs[0], '__len__') else 1

        indices = range(n - 1, -1, -1) if reverse else range(n)
        for i in indices:
            x_slice = [x[i] if hasattr(x, '__getitem__') else x for x in xs]
            carry, y = f(carry, *x_slice)
            ys.append(y)

        if reverse:
            ys = ys[::-1]

        return carry, ys

    @staticmethod
    def rng(seed: int):
        """Create a deterministic random number generator (pure).

        Unlike Python's random module, this is pure and reproducible.
        Uses a simple LCG (Linear Congruential Generator) for portability.

        Args:
            seed: Integer seed for reproducibility.

        Returns:
            A function that takes a shape and returns a DeviceArray of random values.
        """
        import numpy as np

        class _RNG:
            def __init__(self, state):
                self.state = state

            def __call__(self, shape, dtype=np.float32):
                """Generate random uniform values in [0, 1)."""
                n = 1
                for s in shape:
                    n *= s
                result = np.empty(n, dtype=dtype)
                for i in range(n):
                    self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
                    result[i] = dtype(self.state / 0x7FFFFFFF)
                return DeviceArray(result.reshape(shape))

            def normal(self, shape, dtype=np.float32):
                """Generate random normal values (Box-Muller)."""
                u1 = self(shape, dtype).to_numpy()
                u2 = self(shape, dtype).to_numpy()
                u1 = np.clip(u1, 1e-10, 1.0)  # Avoid log(0)
                z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
                return DeviceArray(z)

        return _RNG(seed)


# ── Module-level info ────────────────────────────────────────────────────────

__version__ = "1.5.2"


def is_rust_engine_available():
    """Check if the Rust polyhedral engine is available."""
    return _RUST_ENGINE_AVAILABLE


def hardware_info():
    """Detect and return hardware capabilities."""
    if _RUST_ENGINE_AVAILABLE:
        return _rust_detect_hardware()
    return {
        "target": "unknown",
        "simd_level": "unknown",
        "peak_gflops": 0,
        "mem_bandwidth_gb_per_sec": 0,
        "l1_cache_bytes": 0,
        "l2_cache_bytes": 0,
    }


def micro_kernel_config(target="server", element_type="fp32"):
    """Get micro-kernel tile configuration for the given target."""
    if _RUST_ENGINE_AVAILABLE:
        return _rust_micro_kernel_config(target, element_type)
    return {
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "accumulator_registers": 32,
        "prefetch_distance": 2,
        "double_buffer_count": 2,
    }


__all__ = [
    # Core APIs
    "jit",
    "grad",
    "DeviceArray",
    # Math functions
    "relu", "gelu", "sigmoid", "softmax",
    "exp", "log", "sqrt", "sin", "cos", "tanh",
    "matmul", "dot",
    "sum", "mean", "max", "min",
    "clip", "where",
    "concatenate", "stack", "reshape", "transpose",
    # Array creation
    "zeros", "ones", "array", "arange", "linspace", "eye",
    # Control flow
    "lax",
    # Linear algebra (BLAS-backed)
    "linalg",
    # Errors
    "SympleXError", "ImpureFunctionError", "TracerError", "CompilationError", "ShapeError",
    # Info
    "is_rust_engine_available", "hardware_info", "micro_kernel_config",
    # Type annotations
    "f32", "f64", "bf16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
]


# ── C Kernel Library (ctypes fallback for physics kernels) ──────────────────────
# These functions use the compiled C kernel library for physics workloads
# (PDE stencils, N-body, integrators). They operate on the same DeviceArray
# type imported from _array.py above.
# If the C library is not available, they fall back to NumPy implementations.

import ctypes
import os
import numpy as np

# Try to load the compiled kernel library (optional, not required for Colab)
_lib = None
try:
    _KERNEL_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "libsymplex_kernels.so")
    if os.path.exists(_KERNEL_LIB_PATH):
        _lib = ctypes.CDLL(_KERNEL_LIB_PATH)
except (OSError, Exception):
    _lib = None


# ── BLAS-backed matmul (default path) ──

def _blas_matmul(A, B):
    """Matrix multiply using NumPy's BLAS-backed matmul.

    This is the recommended path for CPU matmul — NumPy uses optimized
    BLAS (OpenBLAS, MKL, or Apple Accelerate) which delivers near-peak
    GEMM performance. SympleX's JIT matmul is kept for fused kernels
    (e.g., matmul+bias+ReLU) where BLAS alone cannot help.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    return np.matmul(A, B)


def matmul_jit(A, B):
    """Matrix multiply using SympleX's JIT-compiled kernel.

    Uses the JIT-compiled AVX-512/AVX2/SSE2 kernel instead of BLAS.
    This is useful for benchmarking or when you need fused operations.

    For production CPU matmul, use ``symplex.matmul`` which delegates
    to BLAS for maximum performance.

    Args:
        A: (M, K) array-like, float32
        B: (K, N) array-like, float32

    Returns:
        DeviceArray of shape (M, N)
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_jit requires 2D arrays")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")

    M, K = A.shape
    _, N = B.shape

    # Try the Rust JIT kernel
    if _RUST_ENGINE_AVAILABLE:
        try:
            from ._symplex_core import jit_parallel_matmul
            C = np.zeros((M, N), dtype=np.float32)
            jit_parallel_matmul(A.ctypes.data, B.ctypes.data, C.ctypes.data,
                               M, N, K)
            return DeviceArray(C)
        except Exception:
            pass

    # Fallback to C library if available
    if _lib is not None:
        try:
            C = np.zeros((M, N), dtype=np.float32)
            _lib.symplex_matmul_f32(A.ctypes.data, B.ctypes.data, C.ctypes.data,
                                     ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K))
            return DeviceArray(C)
        except (OSError, AttributeError):
            pass

    # Final fallback: use NumPy BLAS
    return DeviceArray(np.matmul(A, B))


def matmul_tiled(A, B):
    """Tiled matmul with L1 cache blocking (64x64x64 tiles).

    Falls back to BLAS if the C kernel library is not available.
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    M, K = A.shape
    _, N = B.shape

    if _lib is not None:
        try:
            C = np.zeros((M, N), dtype=np.float32)
            _lib.symplex_matmul_tiled_f32(A.ctypes.data, B.ctypes.data, C.ctypes.data,
                                           ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K))
            return DeviceArray(C)
        except (OSError, AttributeError):
            pass

    # Fallback to BLAS
    return DeviceArray(np.matmul(A, B))


# ── Elementwise ──

def add(a, b):
    """Elementwise add: out[i] = a[i] + b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    b = np.ascontiguousarray(b, dtype=np.float32).ravel()
    n = len(a)
    if len(b) != n:
        raise ValueError("Array length mismatch")

    if _lib is not None:
        try:
            out = np.zeros(n, dtype=np.float32)
            _lib.symplex_add_f32(out.ctypes.data, a.ctypes.data, b.ctypes.data, ctypes.c_int64(n))
            return DeviceArray(out)
        except (OSError, AttributeError):
            pass

    # Fallback to NumPy
    return DeviceArray(a + b)


def mul(a, b):
    """Elementwise multiply: out[i] = a[i] * b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    b = np.ascontiguousarray(b, dtype=np.float32).ravel()
    n = len(a)
    if len(b) != n:
        raise ValueError("Array length mismatch")

    if _lib is not None:
        try:
            out = np.zeros(n, dtype=np.float32)
            _lib.symplex_mul_f32(out.ctypes.data, a.ctypes.data, b.ctypes.data, ctypes.c_int64(n))
            return DeviceArray(out)
        except (OSError, AttributeError):
            pass

    return DeviceArray(a * b)


def sub(a, b):
    """Elementwise subtract: out[i] = a[i] - b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    b = np.ascontiguousarray(b, dtype=np.float32).ravel()
    n = len(a)
    if len(b) != n:
        raise ValueError("Array length mismatch")

    if _lib is not None:
        try:
            out = np.zeros(n, dtype=np.float32)
            _lib.symplex_sub_f32(out.ctypes.data, a.ctypes.data, b.ctypes.data, ctypes.c_int64(n))
            return DeviceArray(out)
        except (OSError, AttributeError):
            pass

    return DeviceArray(a - b)


# ── PDE Stencil ──

def stencil_laplacian(field, dx=0.01):
    """5-point 2D Laplacian stencil: (top + bottom + left + right - 4*center) / dx^2

    Args:
        field: (N, N) array-like, float32
        dx: grid spacing

    Returns:
        DeviceArray of shape (N, N) with Laplacian
    """
    field = np.ascontiguousarray(field, dtype=np.float32)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("stencil_laplacian requires square 2D array")

    if _lib is not None:
        try:
            N = field.shape[0]
            out = np.zeros_like(field)
            _lib.symplex_stencil_2d(out.ctypes.data, field.ctypes.data,
                                     ctypes.c_int64(N), ctypes.c_float(dx))
            return DeviceArray(out)
        except (OSError, AttributeError):
            pass

    # NumPy fallback for 5-point Laplacian
    N = field.shape[0]
    out = np.zeros_like(field)
    out[1:-1, 1:-1] = (
        field[2:, 1:-1] + field[:-2, 1:-1] +
        field[1:-1, 2:] + field[1:-1, :-2] -
        4 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return DeviceArray(out)


# ── N-body ──

def nbody_forces(pos_x, pos_y, mass, G=6.674e-11, softening=0.1):
    """Compute gravitational forces for N-body simulation.

    Falls back to a NumPy implementation if the C kernel library is not available.
    """
    pos_x = np.ascontiguousarray(pos_x, dtype=np.float32)
    pos_y = np.ascontiguousarray(pos_y, dtype=np.float32)
    mass = np.ascontiguousarray(mass, dtype=np.float32)
    n = len(pos_x)

    if _lib is not None:
        try:
            force_x = np.zeros(n, dtype=np.float32)
            force_y = np.zeros(n, dtype=np.float32)
            _lib.symplex_nbody_forces(pos_x.ctypes.data, pos_y.ctypes.data, mass.ctypes.data,
                                       force_x.ctypes.data, force_y.ctypes.data,
                                       ctypes.c_int64(n), ctypes.c_float(G), ctypes.c_float(softening))
            return DeviceArray(force_x), DeviceArray(force_y)
        except (OSError, AttributeError):
            pass

    # NumPy fallback: O(n^2) direct summation
    dx = pos_x[:, None] - pos_x[None, :]
    dy = pos_y[:, None] - pos_y[None, :]
    r2 = dx**2 + dy**2 + softening**2
    inv_r3 = r2 ** (-1.5)
    force_x = (G * mass[None, :] * dx * inv_r3).sum(axis=1).astype(np.float32)
    force_y = (G * mass[None, :] * dy * inv_r3).sum(axis=1).astype(np.float32)
    return DeviceArray(force_x), DeviceArray(force_y)


# ── Integrators ──

def euler_step(pos, vel, force, mass=1.0, dt=0.001):
    """Euler integration step: vel += (force/mass)*dt, pos += vel*dt"""
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    vel = np.ascontiguousarray(vel, dtype=np.float32)
    force = np.ascontiguousarray(force, dtype=np.float32)

    if _lib is not None:
        try:
            n = len(pos)
            _lib.symplex_euler_step(pos.ctypes.data, vel.ctypes.data, force.ctypes.data,
                                     ctypes.c_float(mass), ctypes.c_float(dt), ctypes.c_int64(n))
            return DeviceArray(pos), DeviceArray(vel)
        except (OSError, AttributeError):
            pass

    # NumPy fallback
    vel = vel + (force / mass) * dt
    pos = pos + vel * dt
    return DeviceArray(pos), DeviceArray(vel)


def rk4_step(x, v, k=10.0, c=0.1, m=1.0, dt=0.001):
    """RK4 integration step for spring-mass-damper: a = (-k*x - c*v)/m"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)

    if _lib is not None:
        try:
            n = len(x)
            _lib.symplex_rk4_step(x.ctypes.data, v.ctypes.data,
                                   ctypes.c_float(k), ctypes.c_float(c), ctypes.c_float(m),
                                   ctypes.c_float(dt), ctypes.c_int64(n))
            return DeviceArray(x), DeviceArray(v)
        except (OSError, AttributeError):
            pass

    # NumPy fallback
    def accel(xx, vv):
        return (-k * xx - c * vv) / m
    k1v = accel(x, v) * dt
    k1x = v * dt
    k2v = accel(x + 0.5 * k1x, v + 0.5 * k1v) * dt
    k2x = (v + 0.5 * k1v) * dt
    k3v = accel(x + 0.5 * k2x, v + 0.5 * k2v) * dt
    k3x = (v + 0.5 * k2v) * dt
    k4v = accel(x + k3x, v + k3v) * dt
    k4x = (v + k3v) * dt
    x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    v = v + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return DeviceArray(x), DeviceArray(v)


# ── Tensor field ops ──

def gradient_magnitude(gx, gy):
    """Compute gradient magnitude squared: out[i] = gx[i]^2 + gy[i]^2"""
    gx = np.ascontiguousarray(gx, dtype=np.float32)
    gy = np.ascontiguousarray(gy, dtype=np.float32)
    n = len(gx)
    if len(gy) != n:
        raise ValueError("Array length mismatch")

    if _lib is not None:
        try:
            out = np.zeros(n, dtype=np.float32)
            _lib.symplex_grad_magnitude(out.ctypes.data, gx.ctypes.data, gy.ctypes.data, ctypes.c_int64(n))
            return DeviceArray(out)
        except (OSError, AttributeError):
            pass

    return DeviceArray(gx ** 2 + gy ** 2)


# ── MCMC support ──
# The jit decorator from _jit.py already supports mcmc=True parameter.
