"""SympleX — DeviceArray: NumPy-backed array with polyhedral optimization.

DeviceArray is the core data structure returned by JIT-compiled functions.
It wraps a NumPy ndarray and provides JAX-style functional update semantics
(.at[].set(), .at[].add(), etc.) that enforce immutability.

Performance notes:
  - All arithmetic ops unwrap to raw ndarrays before calling NumPy,
    eliminating the "trampoline tax" of per-call wrapper overhead.
  - __array_ufunc__ and __array_function__ ensure NumPy never tries
    to interpret a DeviceArray as an integer/shape parameter.
  - Reductions always unwrap nested DeviceArrays to avoid TypeError
    when NumPy's C-ext internals try to inspect the wrapper object.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple, Union
import numpy as np


def _raw(x):
    """Unwrap any DeviceArray to its underlying ndarray (fast path)."""
    return x._data if isinstance(x, DeviceArray) else x


def _wrap_result(x):
    """Wrap a result back into DeviceArray only if it's an ndarray."""
    if isinstance(x, np.ndarray):
        return DeviceArray._wrap(x)
    return x


class DeviceArray:
    """An array backed by NumPy with functional update semantics.

    DeviceArray objects are immutable — all operations return new arrays.
    This enforces the purity discipline required by the polyhedral optimizer.

    Attributes:
        shape: Tuple of ints describing array dimensions.
        dtype: NumPy dtype of the array elements.
    """

    __slots__ = ("_data",)

    # Tell NumPy this type is array-like (prevents TypeError when NumPy
    # tries to interpret a DeviceArray as an integer/shape parameter)
    __array_priority__ = 20.0

    def __init__(self, data, _internal=False):
        if _internal:
            # Skip copy — caller guarantees data is fresh and won't be shared
            self._data = data
        elif isinstance(data, DeviceArray):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            # Preserve input dtype — don't force f64.
            # np.array infers dtype from the input, so passing [1.0, 2.0] gives
            # float64 (Python float is f64), but np.float32([1.0, 2.0]) stays f32.
            self._data = np.array(data)

    @classmethod
    def _wrap(cls, data: np.ndarray) -> 'DeviceArray':
        """Wrap an ndarray without copying (internal use only)."""
        obj = cls.__new__(cls)
        obj._data = data
        return obj

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def T(self) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.T)

    def to_numpy(self) -> np.ndarray:
        """Convert to a plain NumPy array (copies data)."""
        return self._data.copy()

    def __array__(self, dtype=None):
        """Allow NumPy to convert DeviceArray to ndarray."""
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support NumPy ufuncs (np.add, np.multiply, etc.) on DeviceArray.

        This prevents TypeError when NumPy tries to use DeviceArray as an
        integer/shape — all DeviceArray inputs are unwrapped to raw ndarrays
        before the ufunc is invoked, and the result is re-wrapped.
        """
        if method != '__call__':
            return NotImplemented

        unwrapped = []
        for inp in inputs:
            if isinstance(inp, DeviceArray):
                unwrapped.append(inp._data)
            elif isinstance(inp, np.ndarray):
                unwrapped.append(inp)
            else:
                unwrapped.append(inp)

        result = ufunc(*unwrapped, **kwargs)
        if isinstance(result, np.ndarray):
            return DeviceArray._wrap(result)
        if isinstance(result, tuple):
            # Some ufuncs return tuples (e.g., np.frexp)
            return tuple(DeviceArray._wrap(r) if isinstance(r, np.ndarray) else r for r in result)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """Support NumPy array functions (np.concatenate, np.stack, etc.).

        This ensures that NumPy never tries to inspect a DeviceArray object
        as a parameter — all DeviceArray arguments are unwrapped first.
        """
        # Unwrap all DeviceArray arguments
        new_args = []
        for arg in args:
            if isinstance(arg, DeviceArray):
                new_args.append(arg._data)
            elif isinstance(arg, (list, tuple)):
                unwrapped_items = []
                for item in arg:
                    if isinstance(item, DeviceArray):
                        unwrapped_items.append(item._data)
                    else:
                        unwrapped_items.append(item)
                new_args.append(type(arg)(unwrapped_items))
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)
        if isinstance(result, np.ndarray):
            return DeviceArray._wrap(result)
        return result

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        result = self._data[idx]
        if isinstance(result, np.ndarray):
            return DeviceArray._wrap(result)
        return result

    # ── Functional updates (JAX-style .at[]) ─────────────────────────────

    @property
    def at(self) -> '_AtIndexer':
        """Return an indexer for functional updates.

        Example::

            x = DeviceArray([1.0, 2.0, 3.0])
            y = x.at[1].set(5.0)      # y = [1.0, 5.0, 3.0], x unchanged
            z = x.at[1].add(10.0)     # z = [1.0, 12.0, 3.0], x unchanged
        """
        return _AtIndexer(self)

    # ── Arithmetic operators ─────────────────────────────────────────────
    # All operators unwrap DeviceArray inputs to raw ndarrays first,
    # then call the NumPy operation directly — zero trampoline tax.

    def __add__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data + _raw(other))

    def __radd__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other + self._data)

    def __sub__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data - _raw(other))

    def __rsub__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other - self._data)

    def __mul__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data * _raw(other))

    def __rmul__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other * self._data)

    def __truediv__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data / _raw(other))

    def __rtruediv__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other / self._data)

    def __floordiv__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data // _raw(other))

    def __neg__(self) -> 'DeviceArray':
        return DeviceArray._wrap(-self._data)

    def __abs__(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.abs(self._data))

    def __pow__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data ** _raw(other))

    def __rpow__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other ** self._data)

    def __matmul__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(self._data @ _raw(other))

    def __rmatmul__(self, other) -> 'DeviceArray':
        return DeviceArray._wrap(other @ self._data)

    # ── Comparison operators ─────────────────────────────────────────────

    def __lt__(self, other): return DeviceArray._wrap(self._data < _raw(other))
    def __le__(self, other): return DeviceArray._wrap(self._data <= _raw(other))
    def __gt__(self, other): return DeviceArray._wrap(self._data > _raw(other))
    def __ge__(self, other): return DeviceArray._wrap(self._data >= _raw(other))
    def __eq__(self, other): return DeviceArray._wrap(self._data == _raw(other))
    def __ne__(self, other): return DeviceArray._wrap(self._data != _raw(other))

    # ── Reduction operations ─────────────────────────────────────────────
    # CRITICAL: Always unwrap self._data (which is always an ndarray),
    # call the ndarray method, then wrap the result. This prevents TypeError
    # when NumPy's C internals try to inspect DeviceArray as an integer.

    def sum(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        # Defensive: unwrap any nested DeviceArray (shouldn't happen but safe)
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.sum(axis=axis, keepdims=keepdims, **kwargs)
        # Ensure result is always ndarray (np.asarray handles Python scalars)
        return DeviceArray._wrap(np.asarray(r))

    def mean(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.mean(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def max(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.max(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def min(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.min(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def var(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.var(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def std(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.std(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def prod(self, axis=None, keepdims=False, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.prod(axis=axis, keepdims=keepdims, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def argmax(self, axis=None, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.argmax(axis=axis, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def argmin(self, axis=None, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.argmin(axis=axis, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    def cumsum(self, axis=None, **kwargs) -> 'DeviceArray':
        raw = self._data
        while isinstance(raw, DeviceArray):
            raw = raw._data
        r = raw.cumsum(axis=axis, **kwargs)
        return DeviceArray._wrap(np.asarray(r))

    # ── Shape manipulation ───────────────────────────────────────────────

    def reshape(self, *shape) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.reshape(*shape))

    def transpose(self, *axes) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.transpose(*axes))

    def squeeze(self, axis=None) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.squeeze(axis=axis))

    def expand_dims(self, axis=None) -> 'DeviceArray':
        return DeviceArray._wrap(np.expand_dims(self._data, axis=axis))

    def flatten(self) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.flatten())

    def ravel(self) -> 'DeviceArray':
        return DeviceArray._wrap(self._data.ravel())

    def clip(self, a_min=None, a_max=None) -> 'DeviceArray':
        return DeviceArray._wrap(np.clip(self._data, a_min, a_max))

    # ── Math functions ───────────────────────────────────────────────────

    def sqrt(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.sqrt(self._data))

    def exp(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.exp(self._data))

    def log(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.log(self._data))

    def sin(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.sin(self._data))

    def cos(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.cos(self._data))

    def tanh(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.tanh(self._data))

    def relu(self) -> 'DeviceArray':
        return DeviceArray._wrap(np.maximum(self._data, 0))

    def gelu(self) -> 'DeviceArray':
        return DeviceArray._wrap(0.5 * self._data * (1 + np.tanh(np.sqrt(2 / np.pi) * (self._data + 0.044715 * self._data ** 3))))

    def sigmoid(self) -> 'DeviceArray':
        return DeviceArray._wrap(1 / (1 + np.exp(-self._data)))

    def softmax(self, axis=-1) -> 'DeviceArray':
        e = np.exp(self._data - np.max(self._data, axis=axis, keepdims=True))
        return DeviceArray._wrap(e / np.sum(e, axis=axis, keepdims=True))

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return str(self._data)

    def __float__(self):
        return float(self._data)

    def __int__(self):
        return int(self._data)

    def __bool__(self):
        return bool(self._data.all())

    def __hash__(self):
        return id(self)


class _AtIndexer:
    """Functional array update indexer (JAX-style)."""

    def __init__(self, arr: DeviceArray):
        self._arr = arr

    def set(self, val) -> DeviceArray:
        """Return a new array with val at the indexed position."""
        result = self._arr._data.copy()
        result[self._idx] = _raw(val) if isinstance(val, DeviceArray) else val
        return DeviceArray(result)

    def add(self, val) -> DeviceArray:
        """Return a new array with val added at the indexed position."""
        result = self._arr._data.copy()
        result[self._idx] += _raw(val) if isinstance(val, DeviceArray) else val
        return DeviceArray(result)

    def mul(self, val) -> DeviceArray:
        """Return a new array with val multiplied at the indexed position."""
        result = self._arr._data.copy()
        result[self._idx] *= _raw(val) if isinstance(val, DeviceArray) else val
        return DeviceArray(result)

    def sub(self, val) -> DeviceArray:
        """Return a new array with val subtracted at the indexed position."""
        result = self._arr._data.copy()
        result[self._idx] -= _raw(val) if isinstance(val, DeviceArray) else val
        return DeviceArray(result)

    def __getitem__(self, idx):
        self._idx = idx
        return self
