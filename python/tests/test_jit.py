"""Tests for the SympleX JIT compiler and DeviceArray."""

import pytest
import numpy as np
from symplex import jit, grad, DeviceArray
from symplex._errors import ImpureFunctionError


# ── DeviceArray tests ────────────────────────────────────────────────────────

class TestDeviceArray:
    def test_creation(self):
        a = DeviceArray([1.0, 2.0, 3.0])
        assert a.shape == (3,)
        np.testing.assert_array_equal(a.to_numpy(), [1.0, 2.0, 3.0])

    def test_arithmetic(self):
        a = DeviceArray([1.0, 2.0])
        b = DeviceArray([3.0, 4.0])

        np.testing.assert_array_equal((a + b).to_numpy(), [4.0, 6.0])
        np.testing.assert_array_equal((a - b).to_numpy(), [-2.0, -2.0])
        np.testing.assert_array_equal((a * b).to_numpy(), [3.0, 8.0])
        np.testing.assert_array_equal((a / b).to_numpy(), [1/3, 0.5])

    def test_scalar_arithmetic(self):
        a = DeviceArray([1.0, 2.0, 3.0])

        np.testing.assert_array_equal((a + 1).to_numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal((1 + a).to_numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal((a * 2).to_numpy(), [2.0, 4.0, 6.0])
        np.testing.assert_array_equal((2 * a).to_numpy(), [2.0, 4.0, 6.0])

    def test_negation(self):
        a = DeviceArray([1.0, -2.0, 3.0])
        np.testing.assert_array_equal((-a).to_numpy(), [-1.0, 2.0, -3.0])

    def test_abs(self):
        a = DeviceArray([1.0, -2.0, 3.0])
        np.testing.assert_array_equal(abs(a).to_numpy(), [1.0, 2.0, 3.0])

    def test_functional_update_set(self):
        """JAX-style .at[].set() creates new array without mutation."""
        a = DeviceArray([1.0, 2.0, 3.0])
        b = a.at[1].set(5.0)

        # Original is unchanged
        np.testing.assert_array_equal(a.to_numpy(), [1.0, 2.0, 3.0])
        # New array has the updated value
        np.testing.assert_array_equal(b.to_numpy(), [1.0, 5.0, 3.0])

    def test_functional_update_add(self):
        """JAX-style .at[].add() creates new array without mutation."""
        a = DeviceArray([1.0, 2.0, 3.0])
        b = a.at[1].add(10.0)

        np.testing.assert_array_equal(a.to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(b.to_numpy(), [1.0, 12.0, 3.0])

    def test_matmul(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        b = DeviceArray([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.to_numpy(), expected)

    def test_reductions(self):
        a = DeviceArray([1.0, 2.0, 3.0])
        assert float(a.sum()) == 6.0
        assert float(a.mean()) == 2.0
        assert float(a.max()) == 3.0
        assert float(a.min()) == 1.0

    def test_reshape(self):
        a = DeviceArray([1.0, 2.0, 3.0, 4.0])
        b = a.reshape(2, 2)
        assert b.shape == (2, 2)

    def test_transpose(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        b = a.T
        np.testing.assert_array_equal(b.to_numpy(), [[1.0, 3.0], [2.0, 4.0]])

    def test_relu(self):
        a = DeviceArray([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(a.relu().to_numpy(), [0.0, 0.0, 1.0, 2.0])

    def test_sigmoid(self):
        a = DeviceArray([0.0])
        result = a.sigmoid().to_numpy()
        np.testing.assert_almost_equal(result[0], 0.5)

    def test_softmax(self):
        a = DeviceArray([1.0, 2.0, 3.0])
        result = a.softmax().to_numpy()
        np.testing.assert_almost_equal(result.sum(), 1.0)
        assert result[2] > result[1] > result[0]


# ── JIT compiler tests ───────────────────────────────────────────────────────

class TestJit:
    def test_jit_pure_function(self):
        """JIT should compile and execute a pure function."""
        @jit
        def add(x, y):
            return x + y

        a = DeviceArray([1.0, 2.0, 3.0])
        b = DeviceArray([4.0, 5.0, 6.0])
        result = add(a, b)

        np.testing.assert_array_equal(result.to_numpy(), [5.0, 7.0, 9.0])

    def test_jit_scalar_arithmetic(self):
        """JIT should handle scalar operations."""
        @jit
        def f(x, y):
            return x * y + x

        a = DeviceArray([1.0, 2.0])
        b = DeviceArray([3.0, 4.0])
        result = f(a, b)

        # x * y + x = [1*3+1, 2*4+2] = [4, 10]
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 10.0])

    def test_jit_rejects_impure(self):
        """JIT should reject impure functions at decoration time."""
        with pytest.raises(ImpureFunctionError):
            @jit
            def f(x):
                print(x)
                return x

    def test_jit_numpy_inputs(self):
        """JIT should accept plain NumPy arrays."""
        @jit
        def f(x, y):
            return x + y

        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = f(a, b)

        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])

    def test_jit_with_config(self):
        """JIT should accept configuration parameters."""
        @jit(target="server", element_type="fp32", enable_flash_attention=True)
        def f(x, y):
            return x * y

        a = DeviceArray([2.0, 3.0])
        b = DeviceArray([4.0, 5.0])
        result = f(a, b)

        np.testing.assert_array_equal(result.to_numpy(), [8.0, 15.0])

    def test_jit_caching(self):
        """JIT should cache compilation results for same shapes."""
        @jit
        def f(x):
            return x * 2

        a = DeviceArray([1.0, 2.0])
        result1 = f(a)
        result2 = f(a)  # Should use cached trace

        np.testing.assert_array_equal(result1.to_numpy(), [2.0, 4.0])
        np.testing.assert_array_equal(result2.to_numpy(), [2.0, 4.0])


# ── Grad tests ───────────────────────────────────────────────────────────────

class TestGrad:
    def test_grad_simple(self):
        """Gradient of x^2 should be 2x."""
        def f(x):
            return (x * x).sum()

        df = grad(f)
        x = DeviceArray([1.0, 2.0, 3.0])
        g = df(x)

        np.testing.assert_array_almost_equal(g.to_numpy(), [2.0, 4.0, 6.0])

    def test_grad_rejects_impure(self):
        """grad() should reject impure functions."""
        with pytest.raises(ImpureFunctionError):
            def f(x):
                print(x)
                return x

            grad(f)

    def test_grad_sum(self):
        """Gradient of sum(x) should be all ones."""
        def f(x):
            return x.sum()

        df = grad(f)
        x = DeviceArray([1.0, 2.0, 3.0])
        g = df(x)

        np.testing.assert_array_almost_equal(g.to_numpy(), [1.0, 1.0, 1.0])

    def test_grad_linear(self):
        """Gradient of 3*x should be 3."""
        def f(x):
            return (x * 3.0).sum()

        df = grad(f)
        x = DeviceArray([1.0, 2.0, 3.0])
        g = df(x)

        np.testing.assert_array_almost_equal(g.to_numpy(), [3.0, 3.0, 3.0])


# ── Module-level API tests ──────────────────────────────────────────────────

class TestModuleAPI:
    def test_relu(self):
        import symplex
        a = DeviceArray([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(symplex.relu(a).to_numpy(), [0.0, 0.0, 1.0])

    def test_exp_log(self):
        import symplex
        a = DeviceArray([1.0, 2.0, 3.0])
        result = symplex.log(symplex.exp(a))
        np.testing.assert_array_almost_equal(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_zeros_ones(self):
        import symplex
        z = symplex.zeros(3)
        assert z.shape == (3,)
        np.testing.assert_array_equal(z.to_numpy(), [0.0, 0.0, 0.0])

        o = symplex.ones(3)
        np.testing.assert_array_equal(o.to_numpy(), [1.0, 1.0, 1.0])

    def test_array_creation(self):
        import symplex
        a = symplex.array([1.0, 2.0, 3.0])
        assert a.shape == (3,)

    def test_lax_rng(self):
        """lax.rng should be deterministic and pure."""
        import symplex
        rng1 = symplex.lax.rng(42)
        rng2 = symplex.lax.rng(42)

        a = rng1((3,))
        b = rng2((3,))

        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_lax_cond(self):
        import symplex
        result = symplex.lax.cond(
            True,
            lambda x: x * 2,
            lambda x: x * 3,
            DeviceArray([1.0])
        )
        np.testing.assert_array_equal(result.to_numpy(), [2.0])

    def test_hardware_info(self):
        import symplex
        info = symplex.hardware_info()
        assert "target" in info
        assert "simd_level" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
