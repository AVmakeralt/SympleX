"""SympleX — Error types for purity enforcement and compilation."""


class SympleXError(Exception):
    """Base exception for all SympleX errors."""
    pass


class ImpureFunctionError(SympleXError):
    """Raised when a function violates SympleX's purity rules.

    SympleX only compiles pure functions — functions with no side effects,
    no mutation of external state, and no IO. This is the same discipline
    that JAX enforces to enable safe program transformations like
    automatic differentiation, vectorization, and polyhedral optimization.

    Common causes:
      - Assignment to a variable that is not local to the function
      - In-place mutation of arrays (e.g., ``x[0] = 1.0``)
      - Calls to impure builtins (``print``, ``input``, ``open``, etc.)
      - Access to global mutable state
      - Use of Python ``random`` module (use ``symplex.lax.rng`` instead)
    """
    pass


class TracerError(SympleXError):
    """Raised when the abstract tracer encounters an unsupported operation."""
    pass


class CompilationError(SympleXError):
    """Raised when the polyhedral optimizer fails to produce a valid result."""
    pass


class ShapeError(SympleXError):
    """Raised when tensor shapes are incompatible for an operation."""
    pass
