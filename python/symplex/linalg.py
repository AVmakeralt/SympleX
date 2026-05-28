"""SympleX — BLAS-backed Linear Algebra Module

Provides optimized linear algebra operations backed by scipy.linalg
(when available) or numpy.linalg as fallback. These operations use
optimized BLAS/LAPACK implementations (OpenBLAS, MKL, Apple Accelerate)
for near-peak CPU performance.

Functions:
  matmul  — Matrix multiplication (BLAS-backed)
  dot     — Dot product / generalized matrix multiply
  solve   — Solve linear system Ax = b
  inv     — Matrix inverse
  det     — Determinant
  norm    — Matrix/vector norm
  svd     — Singular value decomposition
  eig     — Eigenvalue decomposition
  cholesky— Cholesky decomposition
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from ._array import DeviceArray

# ── Try scipy.linalg first (more routines, better LAPACK coverage) ──────────

try:
    import scipy.linalg as _scipy_linalg
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _to_ndarray(x) -> np.ndarray:
    """Unwrap DeviceArray to raw ndarray."""
    if isinstance(x, DeviceArray):
        return x._data
    return np.asarray(x)


def _wrap(x) -> Union[DeviceArray, np.ndarray, float]:
    """Wrap result back to DeviceArray if it's an ndarray."""
    if isinstance(x, np.ndarray):
        return DeviceArray._wrap(x)
    # Scalars (float, complex, int) pass through unchanged
    return x


# ── Matrix multiplication ────────────────────────────────────────────────────

def matmul(a, b):
    """Matrix multiplication using BLAS (via NumPy).

    NumPy's matmul delegates to optimized BLAS (OpenBLAS, MKL, or
    Apple Accelerate) for near-peak GEMM performance.

    Args:
        a: First matrix (M, K) or vector.
        b: Second matrix (K, N) or vector.

    Returns:
        DeviceArray with the result.
    """
    a = _to_ndarray(a)
    b = _to_ndarray(b)
    return DeviceArray._wrap(np.matmul(a, b))


# ── Dot product ──────────────────────────────────────────────────────────────

def dot(a, b):
    """Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for
    1-D arrays to inner product of vectors. Uses NumPy's BLAS-backed
    dot implementation.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Result as DeviceArray or scalar.
    """
    a = _to_ndarray(a)
    b = _to_ndarray(b)
    result = np.dot(a, b)
    return _wrap(result)


# ── Solve linear system ─────────────────────────────────────────────────────

def solve(a, b):
    """Solve the linear system Ax = b for x.

    Uses scipy.linalg.solve when available (which calls LAPACK's
    xGESV), falling back to numpy.linalg.solve.

    Args:
        a: Coefficient matrix (N, N).
        b: Right-hand side (N,) or (N, K).

    Returns:
        DeviceArray with the solution x.
    """
    a = _to_ndarray(a)
    b = _to_ndarray(b)
    if _HAS_SCIPY:
        return DeviceArray._wrap(_scipy_linalg.solve(a, b))
    return DeviceArray._wrap(np.linalg.solve(a, b))


# ── Matrix inverse ──────────────────────────────────────────────────────────

def inv(a):
    """Compute the inverse of a square matrix.

    Uses scipy.linalg.inv when available (LAPACK's xGETRI), falling
    back to numpy.linalg.inv.

    Args:
        a: Square matrix (N, N).

    Returns:
        DeviceArray with the inverse matrix.
    """
    a = _to_ndarray(a)
    if _HAS_SCIPY:
        return DeviceArray._wrap(_scipy_linalg.inv(a))
    return DeviceArray._wrap(np.linalg.inv(a))


# ── Determinant ────────────────────────────────────────────────────────────

def det(a):
    """Compute the determinant of a square matrix.

    Args:
        a: Square matrix (N, N).

    Returns:
        Determinant as a float.
    """
    a = _to_ndarray(a)
    return float(np.linalg.det(a))


# ── Norm ────────────────────────────────────────────────────────────────────

def norm(x, ord=None, axis=None, keepdims=False):
    """Compute matrix or vector norm.

    Uses scipy.linalg.norm when available, falling back to numpy.linalg.norm.

    Args:
        x: Input array.
        ord: Order of the norm (default: Frobenius for matrices, 2-norm for vectors).
        axis: Axis along which to compute the norm.
        keepdims: Whether to keep the reduced dimension.

    Returns:
        Norm as DeviceArray or scalar.
    """
    x = _to_ndarray(x)
    if _HAS_SCIPY and axis is None:
        result = _scipy_linalg.norm(x, ord=ord)
    else:
        result = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return _wrap(result)


# ── Singular Value Decomposition ───────────────────────────────────────────

def svd(a, full_matrices=True, compute_uv=True):
    """Singular Value Decomposition.

    Factorizes the matrix A as U @ diag(s) @ Vh, where U and Vh are
    unitary and s is a 1-D array of singular values.

    Uses scipy.linalg.svd when available (LAPACK's xGESDD), falling
    back to numpy.linalg.svd.

    Args:
        a: Matrix (M, N) to decompose.
        full_matrices: If True, U and Vh are (M, M) and (N, N).
        compute_uv: If False, only compute singular values.

    Returns:
        Tuple (U, s, Vh) as DeviceArrays, or just s if compute_uv=False.
    """
    a = _to_ndarray(a)
    if _HAS_SCIPY:
        result = _scipy_linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    else:
        result = np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    if compute_uv:
        U, s, Vh = result
        return DeviceArray._wrap(U), DeviceArray._wrap(s), DeviceArray._wrap(Vh)
    else:
        return DeviceArray._wrap(result)


# ── Eigenvalue Decomposition ───────────────────────────────────────────────

def eig(a):
    """Compute the eigenvalues and right eigenvectors of a square matrix.

    Uses scipy.linalg.eig when available (LAPACK's xGEEV), falling
    back to numpy.linalg.eig.

    Args:
        a: Square matrix (N, N).

    Returns:
        Tuple (eigenvalues, eigenvectors) as DeviceArrays.
    """
    a = _to_ndarray(a)
    if _HAS_SCIPY:
        w, v = _scipy_linalg.eig(a)
    else:
        w, v = np.linalg.eig(a)
    return DeviceArray._wrap(w), DeviceArray._wrap(v)


# ── Cholesky Decomposition ─────────────────────────────────────────────────

def cholesky(a, lower=False):
    """Compute the Cholesky decomposition of a positive-definite matrix.

    Returns L such that A = L @ L.T (if lower=True) or
    A = L.T @ L (if lower=False, the default).

    Uses scipy.linalg.cholesky when available (LAPACK's xPOTRF),
    falling back to numpy.linalg.cholesky.

    Args:
        a: Positive-definite matrix (N, N).
        lower: If True, compute lower triangular factor.

    Returns:
        DeviceArray with the Cholesky factor.
    """
    a = _to_ndarray(a)
    if _HAS_SCIPY:
        return DeviceArray._wrap(_scipy_linalg.cholesky(a, lower=lower))
    # numpy.linalg.cholesky only returns lower triangular
    result = np.linalg.cholesky(a)
    if not lower:
        result = result.T
    return DeviceArray._wrap(result)


__all__ = [
    "matmul", "dot", "solve", "inv", "det", "norm",
    "svd", "eig", "cholesky",
]
