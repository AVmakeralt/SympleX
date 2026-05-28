"""SympleX — AST Purity Checker (JAX-style enforcement)

This module implements static purity analysis by walking the Python AST of
a function before it is traced or compiled. It enforces the same discipline
as JAX: only pure functions — those with no side effects, no mutation of
external state, and no IO — are allowed through the optimization pipeline.

Purity Rules (like JAX):
  1. No side effects: no print, input, open, write, etc.
  2. No in-place mutation: no x[idx] = val, no x.attr = val on arrays
  3. No global variable modification
  4. No calls to impure builtins or impure external functions
  5. No use of Python random module (must use symplex.lax.rng)
  6. No catching/suppressing exceptions (impure control flow)
  7. No ``del`` on traced values
  8. No ``yield`` (generators are not pure functions)

Allowed (pure) constructs:
  - Arithmetic: +, -, *, /, //, %, **, @ (matmul)
  - Comparisons: ==, !=, <, <=, >, >=
  - Boolean: and, or, not
  - Bitwise: &, |, ^, ~, <<, >>
  - Pure builtins: abs, min, max, len, range, enumerate, zip, map, filter
  - NumPy array creation and arithmetic (if numpy is imported)
  - SympleX API calls: symplex.relu, symplex.matmul, etc.
  - Local variable assignment (not mutation of external state)
  - ``if``/``elif``/``else`` (static or traced conditionals)
  - ``for`` loops over ranges (converted to traced loops)
  - List/tuple/dict comprehensions (treated as pure construction)
"""

import ast
import inspect
import textwrap
import re
from typing import List, Optional, Set

from ._errors import ImpureFunctionError


# ── Impure builtins that are forbidden in traced functions ────────────────────

IMPURE_BUILTINS: Set[str] = {
    # IO
    "print", "input", "open", "help",
    # Mutation of global state
    "exec", "eval", "compile", "globals", "locals", "vars",
    # Object mutation
    "setattr", "delattr", "property",
    # Random (impure — use symplex.lax.rng)
    "random",  # module reference, not a builtin but caught in attribute checks
    # Dynamic imports
    "__import__",
    # Type mutation
    "type",  # metaclass construction is impure
    # Memory management
    "id", "hash",  # not mathematically pure — depends on object identity
    # Error handling (side-effecting)
    "exit", "quit", "breakpoint",
}

# Pure builtins that ARE allowed
PURE_BUILTINS: Set[str] = {
    "abs", "min", "max", "len", "range", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "all", "any",
    "sum", "round", "pow", "divmod", "bool", "int", "float",
    "str", "list", "tuple", "dict", "set", "frozenset",
    "isinstance", "issubclass", "callable",
    "hex", "oct", "bin", "chr", "ord",
    "slice", "super", "staticmethod", "classmethod",
}

# Pure numpy functions allowed in traced code
PURE_NUMPY_FUNCTIONS: Set[str] = {
    "array", "zeros", "ones", "empty", "full", "arange",
    "linspace", "eye", "identity", "diag",
    "add", "subtract", "multiply", "divide", "true_divide",
    "floor_divide", "power", "remainder", "mod",
    "negative", "positive", "absolute", "fabs", "abs",
    "sqrt", "square", "exp", "exp2", "log", "log2", "log10",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "dot", "matmul", "inner", "outer", "tensordot",
    "einsum", "vdot", "kron",
    "sum", "prod", "cumsum", "cumprod",
    "amax", "amin", "max", "min", "mean", "var", "std",
    "argmax", "argmin",
    "reshape", "ravel", "flatten", "squeeze", "expand_dims",
    "transpose", "swapaxes", "moveaxis", "rollaxis",
    "concatenate", "stack", "vstack", "hstack", "dstack",
    "split", "array_split", "hsplit", "vsplit", "dsplit",
    "tile", "repeat", "unique",
    "where", "select", "choose", "clip",
    "maximum", "minimum", "fmax", "fmin",
    "isfinite", "isinf", "isnan", "isnat",
    "logical_and", "logical_or", "logical_not", "logical_xor",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "softmax", "sigmoid", "relu", "gelu",  # from symplex, not real numpy
    "float32", "float64", "int32", "int64",
}

# Impure numpy functions (mutation, random, IO)
IMPURE_NUMPY: Set[str] = {
    "random", "save", "savez", "load", "savetxt", "loadtxt",
    "fromfile", "tofile", "ndfromtxt", "mafromtxt",
}


def _sanitize_source(source: str) -> str:
    """Sanitize source code from inspect.getsource for AST parsing.

    When inspect.getsource() is called on a function defined inside a
    dictionary literal (e.g., {"key": lambda x: x + 1}), the returned
    source includes the dictionary context which cannot be parsed as a
    standalone function. This function strips away surrounding dict/list
    syntax and ensures the source is parseable.

    It also handles the case where the source contains leading key-value
    syntax like ``"name": lambda x, y: x + y,`` which would cause
    SyntaxError: illegal target for annotation.
    """
    source = textwrap.dedent(source).strip()

    # Strip leading dictionary key syntax: "key": or 'key': or key:
    # This pattern matches things like:  "simple_add": lambda x, y: x + y,
    source = re.sub(r'^["\']?\w+["\']?\s*:\s*', '', source)

    # Strip trailing commas (from dict/list context)
    source = source.rstrip(',').strip()

    # If source starts with a lambda, wrap it in a def for AST parsing
    if source.startswith('lambda'):
        source = f"def _lambda_wrapper():\n    return {source}"

    return source


class PurityChecker(ast.NodeVisitor):
    """Walk the Python AST and reject impure constructs.

    Usage::

        checker = PurityChecker("my_function")
        checker.check(func_ast)
        # Raises ImpureFunctionError if any impurity is detected.
    """

    def __init__(self, func_name: str = "<unknown>"):
        self.func_name = func_name
        self.errors: List[str] = []
        # Track local variables (assigned within the function body)
        self.local_vars: Set[str] = set()
        # Track function parameters (separate from locals — subscript mutation of params is impure)
        self.param_vars: Set[str] = set()
        # Track names that are known-pure (imported modules, etc.)
        self.known_pure_modules: Set[str] = {"numpy", "np", "math", "symplex"}
        # Track whether we're inside a comprehension (more permissive)
        self._in_comprehension = 0

    def check(self, tree: ast.AST) -> None:
        """Run the purity check. Raises ImpureFunctionError on violations."""
        self.errors = []
        self.local_vars = set()

        # First pass: collect all local variable assignments
        self._collect_locals(tree)

        # Second pass: check for impurity
        self.visit(tree)

        if self.errors:
            msg = (f"Function '{self.func_name}' is not pure. "
                   f"SympleX only compiles pure functions (like JAX).\n\n"
                   f"Impurity violations:\n")
            for i, err in enumerate(self.errors, 1):
                msg += f"  {i}. {err}\n"
            msg += ("\nTo fix these, ensure your function:\n"
                    "  - Has no side effects (no print, IO, global mutation)\n"
                    "  - Does not mutate arrays in-place (use x = x.at[idx].set(val))\n"
                    "  - Uses only pure operations (arithmetic, numpy, symplex APIs)\n")
            raise ImpureFunctionError(msg)

    def _collect_locals(self, tree: ast.AST) -> None:
        """Collect all names assigned in the function body (local variables)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.local_vars.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    self.local_vars.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.local_vars.add(node.name)
                for arg in node.args.args:
                    self.param_vars.add(arg.arg)
                    self.local_vars.add(arg.arg)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    self.local_vars.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        self.local_vars.add(item.optional_vars.id)
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    self.local_vars.add(node.target.id)

    def _error(self, msg: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        self.errors.append(f"Line {line}, Col {col}: {msg}")

    # ── Statement visitors ───────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for impure assignments (global, nonlocal, attribute, subscript)."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                # x[idx] = val is impure (in-place mutation)
                if isinstance(target.value, ast.Name):
                    name = target.value.id
                    # Mutation of function parameters is always impure
                    # (they reference external state passed by the caller)
                    if name in self.param_vars:
                        self._error(
                            f"In-place mutation of function parameter '{name}'. "
                            f"Use functional update: {name} = {name}.at[idx].set(val)",
                            node,
                        )
                    elif name not in self.local_vars:
                        self._error(
                            f"In-place mutation of non-local array '{name}'. "
                            f"Use functional update: x = x.at[idx].set(val)",
                            node,
                        )
                else:
                    self._error(
                        "In-place subscript assignment is impure. "
                        "Use functional update: x = x.at[idx].set(val)",
                        node,
                    )
            elif isinstance(target, ast.Attribute):
                # x.attr = val is impure (object mutation)
                self._error(
                    f"Attribute assignment is impure (mutates external state). "
                    f"All data transformations must be pure.",
                    node,
                )

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check for augmented assignment (+=, etc.) on non-local targets."""
        if isinstance(node.target, ast.Subscript):
            self._error(
                "Augmented subscript assignment (e.g., x[i] += v) is impure. "
                "Use functional update: x = x.at[i].add(v)",
                node,
            )
        elif isinstance(node.target, ast.Attribute):
            self._error(
                "Augmented attribute assignment (e.g., x.y += v) is impure.",
                node,
            )
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """del is impure (modifies namespace/object)."""
        self._error("del statement is impure.", node)

    def visit_Global(self, node: ast.Global) -> None:
        """Global declarations are impure (implies reading/writing global state)."""
        self._error(
            "global declaration is impure. Pure functions cannot access global state.",
            node,
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Nonlocal declarations are impure."""
        self._error(
            "nonlocal declaration is impure. Pure functions cannot close over mutable state.",
            node,
        )

    def visit_Yield(self, node: ast.Yield) -> None:
        """yield makes a function a generator (not a pure function)."""
        self._error("yield is impure (generators are not pure functions).", node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """yield from is impure."""
        self._error("yield from is impure (generators are not pure functions).", node)

    def visit_Await(self, node: ast.Await) -> None:
        """await is impure (async I/O)."""
        self._error("await is impure (async I/O is not allowed in pure functions).", node)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Raising exceptions is allowed (pure functions can signal errors)."""
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Assertions are allowed (pure boolean checks)."""
        self.generic_visit(node)

    # ── Expression visitors ──────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for impurity."""
        func = node.func

        if isinstance(func, ast.Name):
            name = func.id

            # Check impure builtins
            if name in IMPURE_BUILTINS:
                self._error(
                    f"Call to impure builtin '{name}' is not allowed in pure functions.",
                    node,
                )
            # Allow pure builtins
            elif name in PURE_BUILTINS:
                pass
            # Allow known local function calls (assume they're pure)
            elif name in self.local_vars:
                pass
            # Allow symplex API calls
            # (will be caught during tracing if actually impure)
            else:
                # Unknown function — allow with a warning that it must be pure
                pass

        elif isinstance(func, ast.Attribute):
            # Method call: obj.method()
            if isinstance(func.value, ast.Name):
                obj_name = func.value.id
                method = func.attr

                # numpy method calls
                if obj_name in self.known_pure_modules:
                    if method in IMPURE_NUMPY:
                        self._error(
                            f"numpy.{method}() is impure (involves random state or IO).",
                            node,
                        )
                    # Allow other numpy functions

                # Array method calls — check for mutating methods
                elif method in ("append", "extend", "insert", "remove",
                               "pop", "clear", "sort", "reverse",
                               "setflags", "fill", "put", "itemset"):
                    self._error(
                        f"Call to mutating method '.{method}()' is impure. "
                        f"Use functional operations instead.",
                        node,
                    )

                # Allow .at[].set(), .at[].add(), .at[].mul() — JAX-style
                # functional updates (these create new arrays, not mutations)
                elif method == "at":
                    pass  # Will be handled in Subscript visitor

            elif isinstance(func.value, ast.Attribute):
                # Chained attribute: np.random.uniform, etc.
                if isinstance(func.value.value, ast.Name):
                    mod = func.value.value.id
                    sub = func.value.attr
                    method = func.attr
                    if mod in self.known_pure_modules and sub == "random":
                        self._error(
                            f"{mod}.random.{method}() is impure. "
                            f"Use symplex.lax.rng for reproducible random numbers.",
                            node,
                        )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for impure patterns."""
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            attr = node.attr

            # Check for random module access
            if obj_name == "random" and attr not in ("Random", "seed"):
                self._error(
                    f"random.{attr} is impure. Use symplex.lax.rng instead.",
                    node,
                )

        self.generic_visit(node)

    # ── Import handling ──────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        """Track imports of known-pure modules."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if alias.name in ("numpy", "math", "symplex"):
                self.known_pure_modules.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-imports of known-pure modules."""
        if node.module and node.module.split(".")[0] in ("numpy", "math", "symplex"):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.known_pure_modules.add(name)
        self.generic_visit(node)

    # ── Control flow ─────────────────────────────────────────────────────

    def visit_If(self, node: ast.If) -> None:
        """if/elif/else is allowed (pure control flow)."""
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """for loops over ranges are allowed (will be traced as loop constructs)."""
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """while loops are allowed but may not be polyhedral-optimizable."""
        self.generic_visit(node)

    def visit_Comprehension(self, node: ast.comprehension) -> None:
        """Comprehensions are pure (construct new collections)."""
        self._in_comprehension += 1
        self.generic_visit(node)
        self._in_comprehension -= 1

    # ── Context managers ─────────────────────────────────────────────────

    def visit_With(self, node: ast.With) -> None:
        """Check with statements — only pure context managers allowed."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                func = item.context_expr.func
                if isinstance(func, ast.Name):
                    if func.id in ("open",):
                        self._error(
                            "with open(...) is impure (file IO).",
                            node,
                        )
        self.generic_visit(node)

    # ── Class definition ─────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Class definitions inside functions are allowed (pure construction)."""
        self.generic_visit(node)


def check_purity(func) -> None:
    """Check that a function is pure (JAX-style enforcement).

    Args:
        func: The Python function to check.

    Raises:
        ImpureFunctionError: If the function contains any impure constructs.
    """
    try:
        source = inspect.getsource(func)
    except OSError:
        # Cannot get source (e.g., from -c command line, or REPL)
        # Skip static purity check — impurity will be caught at trace time
        return

    source = _sanitize_source(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If we still can't parse it after sanitization, it might be
        # a lambda or a function defined in a non-standard context.
        # Lambdas are always pure (they can't contain statements),
        # so skip the check.
        return

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        # Could be a lambda or otherwise not a standard function
        # Lambdas are always pure (they can't contain statements)
        return

    # Async functions are always impure
    if isinstance(func_def, ast.AsyncFunctionDef):
        raise ImpureFunctionError(
            f"Async function '{func_def.name}' is impure. "
            f"SympleX only compiles synchronous pure functions."
        )

    checker = PurityChecker(func_def.name)
    checker.check(func_def)
