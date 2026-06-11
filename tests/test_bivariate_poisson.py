"""Tests for the BivariatePoisson distribution (Karlis-Ntzoufras)."""

from math import comb, exp, factorial, log

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import pytest

from dynestyx import BivariatePoisson

jax.config.update("jax_enable_x64", True)

L1, L2, L3 = 1.3, 0.9, 0.2


def _direct_pmf(y1, y2, l1, l2, l3):
    """Bivariate Poisson PMF in the direct (Karlis-Ntzoufras) form."""
    s = sum(
        comb(y1, k) * comb(y2, k) * factorial(k) * (l3 / (l1 * l2)) ** k
        for k in range(min(y1, y2) + 1)
    )
    return (
        exp(-(l1 + l2 + l3)) * (l1**y1 / factorial(y1)) * (l2**y2 / factorial(y2)) * s
    )


@pytest.mark.parametrize("y", [(0, 0), (2, 1), (3, 3), (1, 4), (5, 2), (0, 6)])
def test_log_prob_matches_direct_pmf(y):
    bp = BivariatePoisson(L1, L2, L3, max_goals=20)
    got = float(bp.log_prob(jnp.array(y)))
    expected = log(_direct_pmf(y[0], y[1], L1, L2, L3))
    assert abs(got - expected) < 1e-9


def test_normalizes_over_grid():
    bp = BivariatePoisson(L1, L2, L3, max_goals=30)
    total = sum(
        float(jnp.exp(bp.log_prob(jnp.array([a, b]))))
        for a in range(40)
        for b in range(40)
    )
    assert abs(total - 1.0) < 1e-6


def test_mean_and_covariance():
    bp = BivariatePoisson(L1, L2, L3, max_goals=20)
    assert jnp.allclose(bp.mean, jnp.array([L1 + L3, L2 + L3]))
    # Exact closed-form moments (used by the moments-linearized EKF).
    expected_cov = jnp.array([[L1 + L3, L3], [L3, L2 + L3]])
    assert jnp.allclose(bp.covariance_matrix, expected_cov)
    assert jnp.allclose(bp.variance, jnp.diagonal(expected_cov))
    s = bp.sample(jax.random.PRNGKey(0), (100_000,)).astype(float)
    assert jnp.allclose(s.mean(0), jnp.array([L1 + L3, L2 + L3]), atol=0.03)
    cov = jnp.cov(s.T)
    assert jnp.allclose(cov, expected_cov, atol=0.05)  # MC agreement


def test_covariance_matrix_batched():
    bp = BivariatePoisson(
        jnp.array([1.0, 2.0]),
        jnp.array([1.0, 0.5]),
        jnp.array([0.1, 0.3]),
        max_goals=15,
    )
    cov = bp.covariance_matrix
    assert cov.shape == (2, 2, 2)
    assert jnp.allclose(cov[1], jnp.array([[2.3, 0.3], [0.3, 0.8]]))


def test_reduces_to_independent_poissons_as_lam3_to_zero():
    bp = BivariatePoisson(L1, L2, 1e-9, max_goals=20)
    for y in [(0, 0), (2, 1), (3, 0), (0, 4)]:
        indep = float(jstats.poisson.logpmf(y[0], L1) + jstats.poisson.logpmf(y[1], L2))
        assert abs(float(bp.log_prob(jnp.array(y))) - indep) < 1e-6


def test_batched_rates():
    bp = BivariatePoisson(
        jnp.array([1.0, 2.0]),
        jnp.array([1.0, 0.5]),
        jnp.array([0.1, 0.3]),
        max_goals=15,
    )
    assert bp.batch_shape == (2,)
    lp = bp.log_prob(jnp.array([[1, 1], [2, 0]]))
    assert lp.shape == (2,)
    assert jnp.all(jnp.isfinite(lp))


def test_log_prob_differentiable_in_log_rates():
    def loss(theta):
        alpha, beta = theta
        bp = BivariatePoisson(
            jnp.exp(alpha), jnp.exp(alpha - 0.2), jnp.exp(beta), max_goals=20
        )
        return bp.log_prob(jnp.array([2, 1]))

    g = jax.grad(loss)(jnp.array([0.3, -1.5]))
    assert jnp.all(jnp.isfinite(g))
