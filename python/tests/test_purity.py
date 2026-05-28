"""Tests for the SympleX AST purity checker."""

import pytest
import ast
from symplex._errors import ImpureFunctionError
from symplex._ast_checker import check_purity, PurityChecker


# ── Pure functions (should pass) ─────────────────────────────────────────────

def test_pure_arithmetic():
    """Pure arithmetic should pass."""
    def f(x, y):
        return x + y * 2

    check_purity(f)  # Should not raise


def test_pure_numpy():
    """Pure numpy operations should pass."""
    def f(x):
        import numpy as np
        return np.sqrt(x) + np.exp(x)

    check_purity(f)


def test_pure_local_assignment():
    """Local variable assignment is pure."""
    def f(x):
        y = x * 2
        z = y + 1
        return z

    check_purity(f)


def test_pure_if_else():
    """If/else control flow is pure."""
    def f(x):
        if x > 0:
            return x
        else:
            return -x

    check_purity(f)


def test_pure_for_loop():
    """For loops over ranges are pure."""
    def f(x):
        result = 0
        for i in range(10):
            result = result + x
        return result

    check_purity(f)


def test_pure_comprehension():
    """List comprehensions are pure."""
    def f(x):
        return [i * x for i in range(10)]

    check_purity(f)


def test_pure_nested_function():
    """Nested function definitions are pure."""
    def f(x):
        def g(y):
            return y * 2
        return g(x)

    check_purity(f)


def test_pure_multiple_returns():
    """Multiple return paths are pure."""
    def f(x):
        if x > 0:
            return x
        return 0

    check_purity(f)


def test_pure_builtin_abs():
    """abs() is pure."""
    def f(x):
        return abs(x)

    check_purity(f)


def test_pure_builtin_min_max():
    """min() and max() are pure."""
    def f(x, y):
        return min(x, y) + max(x, y)

    check_purity(f)


# ── Impure functions (should fail) ───────────────────────────────────────────

def test_impure_print():
    """print() is impure."""
    def f(x):
        print(x)
        return x

    with pytest.raises(ImpureFunctionError, match="print"):
        check_purity(f)


def test_impure_input():
    """input() is impure."""
    def f(x):
        y = input("enter: ")
        return x

    with pytest.raises(ImpureFunctionError, match="input"):
        check_purity(f)


def test_impure_open():
    """open() is impure."""
    def f(x):
        with open("file.txt") as f:
            return x

    with pytest.raises(ImpureFunctionError, match="open"):
        check_purity(f)


def test_impure_global():
    """global declarations are impure."""
    counter = 0

    def f(x):
        global counter
        counter = counter + 1
        return x

    with pytest.raises(ImpureFunctionError, match="global"):
        check_purity(f)


def test_impure_nonlocal():
    """nonlocal declarations are impure."""
    def make_f():
        count = 0

        def f(x):
            nonlocal count
            count += 1
            return x

        return f

    with pytest.raises(ImpureFunctionError, match="nonlocal"):
        check_purity(make_f())


def test_impure_subscript_assignment():
    """In-place array mutation is impure."""
    def f(x):
        x[0] = 1.0  # Impure: mutating non-local array
        return x

    with pytest.raises(ImpureFunctionError, match="In-place mutation"):
        check_purity(f)


def test_impure_attribute_assignment():
    """Attribute assignment is impure."""
    class Obj:
        val = 0

    def f(x, obj):
        obj.val = x  # Impure: mutating external object
        return x

    with pytest.raises(ImpureFunctionError, match="Attribute assignment"):
        check_purity(f)


def test_impure_yield():
    """yield (generators) is impure."""
    def f(x):
        yield x

    with pytest.raises(ImpureFunctionError, match="yield"):
        check_purity(f)


def test_impure_await():
    """await is impure."""
    async def f(x):
        await x
        return x

    with pytest.raises(ImpureFunctionError, match="Async"):
        check_purity(f)


def test_impure_del():
    """del is impure."""
    def f(x):
        del x
        return 0

    with pytest.raises(ImpureFunctionError, match="del"):
        check_purity(f)


def test_impure_augmented_subscript():
    """Augmented subscript assignment (x[i] += v) is impure."""
    def f(x):
        x[0] += 1  # Impure: in-place mutation
        return x

    with pytest.raises(ImpureFunctionError, match="Augmented subscript"):
        check_purity(f)


def test_impure_numpy_random():
    """numpy.random is impure."""
    def f(x):
        import numpy as np
        return x + np.random.uniform(0, 1)

    with pytest.raises(ImpureFunctionError, match="random"):
        check_purity(f)


def test_impure_mutating_method():
    """Mutating list methods are impure."""
    def f(x):
        x.append(1)  # This would be caught if x were a list
        return x

    # The checker catches .append() as a mutating method
    with pytest.raises(ImpureFunctionError, match="mutating method"):
        check_purity(f)


def test_impure_exec():
    """exec() is impure."""
    def f(x):
        exec("pass")
        return x

    with pytest.raises(ImpureFunctionError, match="exec"):
        check_purity(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
