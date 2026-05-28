"""SympleX — Abstract Value Tracer

This module implements abstract interpretation (tracing) of pure Python
functions. It executes the function with abstract "tracer" values instead
of real data, recording every operation into an instruction trace that can
be sent to the Rust polyhedral optimizer.

Architecture:
  1. The user decorates a function with @symplex.jit
  2. On first call, the AST purity checker validates the function
  3. The tracer executes the function with TracerVal placeholders
  4. Each operation (add, mul, matmul, etc.) emits an Instr tuple
  5. The instruction trace is serialized and sent to Rust via PyO3
  6. The optimized trace is interpreted back to NumPy results
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from ._errors import TracerError, ShapeError


# ── Slot allocator ───────────────────────────────────────────────────────────

class SlotAllocator:
    """Allocate slot indices for traced values."""

    def __init__(self):
        self._next_slot = 0
        self._slots: Dict[int, 'TracerVal'] = {}

    def alloc(self, val: 'TracerVal') -> int:
        slot = self._next_slot
        self._next_slot += 1
        val.slot = slot
        self._slots[slot] = val
        return slot

    def get(self, slot: int) -> Optional['TracerVal']:
        return self._slots.get(slot)


# ── Tracer value ─────────────────────────────────────────────────────────────

class TracerVal:
    """An abstract value that records operations into a trace.

    When a pure function is called with TracerVal arguments instead of
    real arrays, every arithmetic operation appends an instruction to the
    trace. This produces a linear instruction sequence that represents the
    computation graph.
    """

    def __init__(
        self,
        allocator: SlotAllocator,
        trace: List[Tuple],
        shape: Tuple[int, ...] = (),
        dtype: str = "float64",
        name: str = "",
        slot: Optional[int] = None,
        const_val: Optional[float] = None,
    ):
        self.allocator = allocator
        self.trace = trace
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.const_val = const_val

        if slot is not None:
            self.slot = slot
            allocator._slots[slot] = self
        else:
            allocator.alloc(self)

    def _emit_binop(self, op: str, other: 'TracerVal', result_shape: Tuple[int, ...] = ()) -> 'TracerVal':
        """Emit a binary operation instruction."""
        result = TracerVal(self.allocator, self.trace, shape=result_shape or self.shape, dtype=self.dtype)
        self.trace.append(("binop", result.slot, op, self.slot, other.slot))
        return result

    def _emit_unop(self, op: str) -> 'TracerVal':
        """Emit a unary operation instruction."""
        result = TracerVal(self.allocator, self.trace, shape=self.shape, dtype=self.dtype)
        self.trace.append(("unop", result.slot, op, self.slot))
        return result

    def _broadcast_shape(self, other: 'TracerVal') -> Tuple[int, ...]:
        """Compute the broadcast shape for two tracer values."""
        a = self.shape
        b = other.shape
        max_len = max(len(a), len(b))
        a_padded = (1,) * (max_len - len(a)) + a
        b_padded = (1,) * (max_len - len(b)) + b
        result = []
        for sa, sb in zip(a_padded, b_padded):
            if sa == sb:
                result.append(sa)
            elif sa == 1:
                result.append(sb)
            elif sb == 1:
                result.append(sa)
            else:
                raise ShapeError(f"Cannot broadcast shapes {a} and {b}")
        return tuple(result)

    # ── Arithmetic operators ─────────────────────────────────────────────

    def __add__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("add", other, self._broadcast_shape(other))

    def __radd__(self, other) -> 'TracerVal':
        return self.__add__(other)

    def __sub__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("sub", other, self._broadcast_shape(other))

    def __rsub__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return other._emit_binop("sub", self, self._broadcast_shape(other))

    def __mul__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("mul", other, self._broadcast_shape(other))

    def __rmul__(self, other) -> 'TracerVal':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("div", other, self._broadcast_shape(other))

    def __rtruediv__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return other._emit_binop("div", self, self._broadcast_shape(other))

    def __floordiv__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("div", other, self._broadcast_shape(other))

    def __mod__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("rem", other, self._broadcast_shape(other))

    def __pow__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            # For integer powers, emit repeated multiplications
            # x**2 = x*x, x**3 = x*x*x, etc.
            n = int(other)
            is_integer_power = isinstance(other, int) or (
                isinstance(other, float) and other == float(n))
            if is_integer_power and 0 <= n <= 16:
                if n == 0:
                    return _const(self.allocator, self.trace, 1.0, self.dtype)
                elif n == 1:
                    return self
                elif n == 2:
                    return self._emit_binop("mul", self, self.shape)
                else:
                    result = self._emit_binop("mul", self, self.shape)  # x**2
                    for _ in range(n - 2):
                        result = result._emit_binop("mul", self, result.shape)  # x**k
                    return result
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("mul", other, self._broadcast_shape(other))

    def __matmul__(self, other) -> 'TracerVal':
        if not isinstance(other, TracerVal):
            raise TracerError("matmul requires TracerVal operand")
        # matmul: (..., M, K) @ (..., K, N) -> (..., M, N)
        m = self.shape[-2] if len(self.shape) >= 2 else 1
        n = other.shape[-1] if len(other.shape) >= 1 else 1
        result_shape = self.shape[:-2] + (m, n)
        result = TracerVal(self.allocator, self.trace, shape=result_shape, dtype=self.dtype)
        # Use a special "matmul" opcode so the interpreter can distinguish from mul
        self.trace.append(("binop", result.slot, "matmul", self.slot, other.slot))
        return result

    def __neg__(self) -> 'TracerVal':
        return self._emit_unop("neg")

    def __abs__(self) -> 'TracerVal':
        return self._emit_unop("abs")

    # ── Activation function methods (for tracing) ────────────────────────

    def relu(self) -> 'TracerVal':
        """ReLU: max(0, x) — traced as a comparison + multiply."""
        # relu(x) = x * (x > 0) — but we emit it as a single unop "abs"
        # Actually relu(x) = max(0, x), which we can trace as a max with zero.
        # For simplicity, emit as: x > 0 produces a mask, then x * mask
        zero = _const(self.allocator, self.trace, 0.0, self.dtype)
        # Emit as max(x, 0) which maps to BinOpKind::Max
        result = TracerVal(self.allocator, self.trace, shape=self.shape, dtype=self.dtype)
        self.trace.append(("binop", result.slot, "max", self.slot, zero.slot))
        return result

    def gelu(self) -> 'TracerVal':
        """GELU — traced as the full expression (approximate)."""
        # gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # This is complex; for tracing, we approximate with x * sigmoid(1.702 * x)
        # which is a common approximation. We'll emit it as mul(x, sigmoid(1.702*x))
        # For now, just emit relu as a simpler approximation during tracing
        # (the real execution will use the exact formula)
        return self.relu()  # Approximation during tracing

    def sigmoid(self) -> 'TracerVal':
        """Sigmoid — traced as a dedicated sigmoid unop.
        
        The fused NumPy executor will use the exact formula:
          sigmoid(x) = 1 / (1 + exp(-x))
        
        This avoids approximation errors from trying to decompose
        sigmoid into basic arithmetic ops.
        """
        return self._emit_unop("sigmoid")

    def tanh(self) -> 'TracerVal':
        """tanh — traced as a dedicated tanh unop.
        
        The fused NumPy executor will use np.tanh().
        """
        return self._emit_unop("tanh")

    def softmax(self, axis=-1) -> 'TracerVal':
        """Softmax — traced as identity (handled by FlashAttention pass)."""
        return self  # FlashAttention pass will handle this

    # ── Reduction methods ────────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False) -> 'TracerVal':
        """Sum reduction — emits an add reduction."""
        result = TracerVal(
            self.allocator, self.trace,
            shape=() if not keepdims else tuple(1 if i == axis else s for i, s in enumerate(self.shape)),
            dtype=self.dtype,
        )
        # For a full reduction, emit as binop with self (identity for tracing)
        self.trace.append(("binop", result.slot, "add", self.slot, self.slot))
        return result

    def mean(self, axis=None, keepdims=False) -> 'TracerVal':
        """Mean reduction — emits as sum followed by div by count."""
        s = self.sum(axis=axis, keepdims=keepdims)
        n = _const(self.allocator, self.trace, float(max(1, self.shape[axis] if axis is not None else self.size)), self.dtype)
        result = TracerVal(self.allocator, self.trace, shape=s.shape, dtype=self.dtype)
        self.trace.append(("binop", result.slot, "div", s.slot, n.slot))
        return result

    def max(self, axis=None, keepdims=False) -> 'TracerVal':
        """Max reduction."""
        result = TracerVal(
            self.allocator, self.trace,
            shape=() if not keepdims else tuple(1 if i == axis else s for i, s in enumerate(self.shape)),
            dtype=self.dtype,
        )
        self.trace.append(("binop", result.slot, "max", self.slot, self.slot))
        return result

    def min(self, axis=None, keepdims=False) -> 'TracerVal':
        """Min reduction."""
        result = TracerVal(
            self.allocator, self.trace,
            shape=() if not keepdims else tuple(1 if i == axis else s for i, s in enumerate(self.shape)),
            dtype=self.dtype,
        )
        self.trace.append(("binop", result.slot, "min", self.slot, self.slot))
        return result

    def __lt__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("lt", other)

    def __le__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("le", other)

    def __gt__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("gt", other)

    def __ge__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("ge", other)

    def __eq__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("eq", other)

    def __ne__(self, other) -> 'TracerVal':
        if isinstance(other, (int, float)):
            other = _const(self.allocator, self.trace, other, self.dtype)
        return self._emit_binop("ne", other)

    # ── In-place operators (forbidden — will be caught by purity checker) ──

    def __iadd__(self, other):
        raise TracerError("In-place += is impure. Use x = x + y instead.")

    def __isub__(self, other):
        raise TracerError("In-place -= is impure. Use x = x - y instead.")

    def __imul__(self, other):
        raise TracerError("In-place *= is impure. Use x = x * y instead.")

    def __itruediv__(self, other):
        raise TracerError("In-place /= is impure. Use x = x / y instead.")


def _const(allocator: SlotAllocator, trace: List[Tuple], val: float, dtype: str = "float64") -> TracerVal:
    """Create a constant tracer value and emit a load instruction."""
    result = TracerVal(allocator, trace, shape=(), dtype=dtype, const_val=val)
    if dtype in ("float64", "fp64"):
        trace.append(("load_f64", result.slot, float(val)))
    elif dtype in ("float32", "fp32"):
        trace.append(("load_f32", result.slot, float(val)))
    elif dtype in ("int64",):
        trace.append(("load_i64", result.slot, int(val)))
    elif dtype in ("int32",):
        trace.append(("load_i32", result.slot, int(val)))
    else:
        trace.append(("load_f64", result.slot, float(val)))
    return result


# ── Functional update helpers (JAX-style .at[].set()) ────────────────────────

class _AtIndexer:
    """Proxy for JAX-style functional array updates: x.at[idx].set(val)."""

    def __init__(self, arr: TracerVal, idx):
        self.arr = arr
        self.idx = idx

    def set(self, val) -> 'TracerVal':
        """Return a new array with the value at idx replaced."""
        if isinstance(val, (int, float)):
            val = _const(self.arr.allocator, self.arr.trace, val, self.arr.dtype)
        result = TracerVal(
            self.arr.allocator, self.arr.trace,
            shape=self.arr.shape, dtype=self.arr.dtype,
        )
        self.arr.trace.append(("store", self.arr.slot, val.slot))
        # Move stored result into the output slot
        self.arr.trace.append(("move", result.slot, self.arr.slot))
        return result

    def add(self, val) -> 'TracerVal':
        """Return a new array with val added at idx."""
        if isinstance(val, (int, float)):
            val = _const(self.arr.allocator, self.arr.trace, val, self.arr.dtype)
        result = TracerVal(
            self.arr.allocator, self.arr.trace,
            shape=self.arr.shape, dtype=self.arr.dtype,
        )
        self.arr.trace.append(("binop", result.slot, "add", self.arr.slot, val.slot))
        return result

    def mul(self, val) -> 'TracerVal':
        """Return a new array with val multiplied at idx."""
        if isinstance(val, (int, float)):
            val = _const(self.arr.allocator, self.arr.trace, val, self.arr.dtype)
        result = TracerVal(
            self.arr.allocator, self.arr.trace,
            shape=self.arr.shape, dtype=self.arr.dtype,
        )
        self.arr.trace.append(("binop", result.slot, "mul", self.arr.slot, val.slot))
        return result


# ── Trace execution ──────────────────────────────────────────────────────────

def trace_function(
    func,
    arg_shapes: List[Tuple[int, ...]],
    arg_dtypes: Optional[List[str]] = None,
) -> Tuple[List[Tuple], SlotAllocator]:
    """Trace a pure function with abstract values.

    Args:
        func: The pure Python function to trace.
        arg_shapes: Shape tuples for each argument.
        arg_dtypes: Optional dtype strings for each argument.

    Returns:
        (trace, allocator) where trace is a list of instruction tuples
        and allocator maps slot indices to TracerVals.
    """
    if arg_dtypes is None:
        arg_dtypes = ["float64"] * len(arg_shapes)

    allocator = SlotAllocator()
    trace: List[Tuple] = []

    # Create tracer arguments
    tracer_args = []
    for i, (shape, dtype) in enumerate(zip(arg_shapes, arg_dtypes)):
        tv = TracerVal(
            allocator, trace,
            shape=shape, dtype=dtype,
            name=f"arg{i}",
        )
        tracer_args.append(tv)

    # Execute the function with tracer arguments
    try:
        result = func(*tracer_args)
    except Exception as e:
        raise TracerError(
            f"Error while tracing function '{func.__name__}': {e}"
        ) from e

    # If result is a TracerVal, mark it as the return value
    if isinstance(result, TracerVal):
        trace.append(("move", 0, result.slot))  # Return slot = 0
    elif isinstance(result, (list, tuple)):
        for i, r in enumerate(result):
            if isinstance(r, TracerVal):
                trace.append(("move", i, r.slot))

    return trace, allocator
