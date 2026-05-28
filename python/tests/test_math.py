"""Tests for SympleX math functions and advanced features."""

import pytest
import numpy as np
from symplex import DeviceArray
import symplex


class TestActivations:
    def test_relu(self):
        a = DeviceArray([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(symplex.relu(a).to_numpy(), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_sigmoid(self):
        # sigmoid(0) = 0.5
        a = DeviceArray([0.0])
        np.testing.assert_almost_equal(symplex.sigmoid(a).to_numpy()[0], 0.5)

        # sigmoid(large positive) ~ 1
        b = DeviceArray([100.0])
        np.testing.assert_almost_equal(symplex.sigmoid(b).to_numpy()[0], 1.0)

        # sigmoid(large negative) ~ 0
        c = DeviceArray([-100.0])
        np.testing.assert_almost_equal(symplex.sigmoid(c).to_numpy()[0], 0.0)

    def test_gelu(self):
        # GELU(0) should be 0
        a = DeviceArray([0.0])
        np.testing.assert_almost_equal(symplex.gelu(a).to_numpy()[0], 0.0)

    def test_softmax(self):
        a = DeviceArray([1.0, 2.0, 3.0])
        result = symplex.softmax(a).to_numpy()
        np.testing.assert_almost_equal(result.sum(), 1.0)
        # softmax should preserve ordering
        assert result[2] > result[1] > result[0]

    def test_softmax_2d(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        result = symplex.softmax(a, axis=-1).to_numpy()
        # Each row should sum to 1
        np.testing.assert_almost_equal(result[0].sum(), 1.0)
        np.testing.assert_almost_equal(result[1].sum(), 1.0)


class TestMathFunctions:
    def test_exp(self):
        a = DeviceArray([0.0, 1.0])
        result = symplex.exp(a).to_numpy()
        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[1], np.e)

    def test_log(self):
        a = DeviceArray([1.0, np.e])
        result = symplex.log(a).to_numpy()
        np.testing.assert_almost_equal(result[0], 0.0)
        np.testing.assert_almost_equal(result[1], 1.0)

    def test_sqrt(self):
        a = DeviceArray([4.0, 9.0, 16.0])
        result = symplex.sqrt(a).to_numpy()
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_sin_cos(self):
        a = DeviceArray([0.0, np.pi / 2])
        np.testing.assert_almost_equal(symplex.sin(a).to_numpy()[0], 0.0)
        np.testing.assert_almost_equal(symplex.sin(a).to_numpy()[1], 1.0)
        np.testing.assert_almost_equal(symplex.cos(a).to_numpy()[0], 1.0)

    def test_tanh(self):
        a = DeviceArray([0.0])
        np.testing.assert_almost_equal(symplex.tanh(a).to_numpy()[0], 0.0)


class TestArrayOps:
    def test_matmul(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        b = DeviceArray([[5.0, 6.0], [7.0, 8.0]])
        c = symplex.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.to_numpy(), expected)

    def test_dot(self):
        a = DeviceArray([1.0, 2.0, 3.0])
        b = DeviceArray([4.0, 5.0, 6.0])
        result = symplex.dot(a, b)
        np.testing.assert_almost_equal(float(result), 32.0)

    def test_sum(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_almost_equal(float(symplex.sum(a)), 10.0)

    def test_mean(self):
        a = DeviceArray([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(float(symplex.mean(a)), 2.5)

    def test_clip(self):
        a = DeviceArray([-1.0, 0.5, 2.0])
        result = symplex.clip(a, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result.to_numpy(), [0.0, 0.5, 1.0])

    def test_where(self):
        cond = DeviceArray([True, False, True])
        x = DeviceArray([1.0, 2.0, 3.0])
        y = DeviceArray([4.0, 5.0, 6.0])
        result = symplex.where(cond, x, y)
        np.testing.assert_array_almost_equal(result.to_numpy(), [1.0, 5.0, 3.0])

    def test_concatenate(self):
        a = DeviceArray([1.0, 2.0])
        b = DeviceArray([3.0, 4.0])
        result = symplex.concatenate([a, b])
        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_stack(self):
        a = DeviceArray([1.0, 2.0])
        b = DeviceArray([3.0, 4.0])
        result = symplex.stack([a, b])
        assert result.shape == (2, 2)

    def test_reshape(self):
        a = DeviceArray([1.0, 2.0, 3.0, 4.0])
        result = symplex.reshape(a, (2, 2))
        assert result.shape == (2, 2)

    def test_transpose(self):
        a = DeviceArray([[1.0, 2.0], [3.0, 4.0]])
        result = symplex.transpose(a)
        np.testing.assert_array_equal(result.to_numpy(), [[1.0, 3.0], [2.0, 4.0]])


class TestArrayCreation:
    def test_zeros(self):
        a = symplex.zeros(5)
        assert a.shape == (5,)
        np.testing.assert_array_equal(a.to_numpy(), np.zeros(5))

    def test_ones(self):
        a = symplex.ones(3)
        np.testing.assert_array_equal(a.to_numpy(), np.ones(3))

    def test_arange(self):
        a = symplex.arange(5)
        np.testing.assert_array_almost_equal(a.to_numpy(), np.arange(5, dtype=np.float64))

    def test_linspace(self):
        a = symplex.linspace(0, 1, 5)
        assert a.shape == (5,)

    def test_eye(self):
        a = symplex.eye(3)
        np.testing.assert_array_almost_equal(a.to_numpy(), np.eye(3))


class TestLax:
    def test_cond_true(self):
        result = symplex.lax.cond(
            True,
            lambda x: x * 2,
            lambda x: x * 3,
            DeviceArray([5.0])
        )
        np.testing.assert_array_equal(result.to_numpy(), [10.0])

    def test_cond_false(self):
        result = symplex.lax.cond(
            False,
            lambda x: x * 2,
            lambda x: x * 3,
            DeviceArray([5.0])
        )
        np.testing.assert_array_equal(result.to_numpy(), [15.0])

    def test_fori_loop(self):
        result = symplex.lax.fori_loop(0, 5, lambda i, x: x + 1, 0)
        assert result == 5

    def test_rng_deterministic(self):
        """RNG should be deterministic for same seed."""
        rng1 = symplex.lax.rng(42)
        rng2 = symplex.lax.rng(42)

        a = rng1((10,))
        b = rng2((10,))

        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_rng_different_seeds(self):
        """RNG should produce different values for different seeds."""
        rng1 = symplex.lax.rng(42)
        rng2 = symplex.lax.rng(99)

        a = rng1((10,))
        b = rng2((10,))

        assert not np.array_equal(a.to_numpy(), b.to_numpy())


class TestImmutability:
    def test_device_array_immutable(self):
        """DeviceArray operations should not modify the original."""
        a = DeviceArray([1.0, 2.0, 3.0])
        b = a + DeviceArray([10.0, 20.0, 30.0])

        # Original is unchanged
        np.testing.assert_array_equal(a.to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(b.to_numpy(), [11.0, 22.0, 33.0])

    def test_functional_update_immutable(self):
        """JAX-style .at[].set() should not modify the original."""
        a = DeviceArray([1.0, 2.0, 3.0])
        b = a.at[0].set(99.0)

        np.testing.assert_array_equal(a.to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(b.to_numpy(), [99.0, 2.0, 3.0])

    def test_relu_immutable(self):
        """ReLU should not modify the original."""
        a = DeviceArray([-1.0, 0.0, 1.0])
        b = a.relu()

        np.testing.assert_array_equal(a.to_numpy(), [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(b.to_numpy(), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
