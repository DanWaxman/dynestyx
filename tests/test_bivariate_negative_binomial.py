"""Unit tests for the BivariateNegativeBinomial distribution (overdispersed scoreline).

Independent NB margins with shared dispersion r: mean lambda_j, variance lambda_j +
lambda_j^2/r, zero correlation, and the Poisson product as r -> infinity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from dynestyx import (
    BivariateNegativeBinomial,
    BivariateNegativeBinomialScoreObservation,
)

jax.config.update("jax_enable_x64", True)

LAM1, LAM2 = 1.4, 1.1


def test_log_prob_normalizes_over_grid():
    """The pmf sums to ~1 over a large enough count grid."""
    nb = BivariateNegativeBinomial(LAM1, LAM2, 5.0)
    g = 60
    aa, bb = jnp.meshgrid(jnp.arange(g + 1), jnp.arange(g + 1), indexing="ij")
    grid = jnp.stack([aa.ravel(), bb.ravel()], axis=-1).astype(float)
    total = jnp.exp(nb.log_prob(grid)).sum()
    assert jnp.allclose(total, 1.0, atol=1e-5)


def test_poisson_limit():
    """As dispersion -> infinity the NB log_prob matches independent Poissons."""
    nb_big = BivariateNegativeBinomial(LAM1, LAM2, 1e8)
    y = jnp.array([[2.0, 1.0], [0.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    pois = dist.Poisson(LAM1).log_prob(y[:, 0]) + dist.Poisson(LAM2).log_prob(y[:, 1])
    assert jnp.allclose(nb_big.log_prob(y), pois, atol=1e-4)


def test_matches_numpyro_negative_binomial2_margins():
    """log_prob equals the sum of two numpyro NegativeBinomial2 margin log-probs."""
    r = 5.0
    nb = BivariateNegativeBinomial(LAM1, LAM2, r)
    y = jnp.array([[2.0, 1.0], [0.0, 0.0], [4.0, 3.0]])
    ref = dist.NegativeBinomial2(jnp.asarray(LAM1), jnp.asarray(r)).log_prob(
        y[:, 0]
    ) + dist.NegativeBinomial2(jnp.asarray(LAM2), jnp.asarray(r)).log_prob(y[:, 1])
    assert jnp.allclose(nb.log_prob(y), ref, atol=1e-10)


def test_overdispersion_and_zero_correlation():
    """Empirical moments: variance > mean (overdispersed), correlation ~ 0."""
    r = 5.0
    nb = BivariateNegativeBinomial(LAM1, LAM2, r)
    s = np.asarray(nb.sample(jax.random.PRNGKey(0), (200_000,)))
    assert np.allclose(s.mean(0), [LAM1, LAM2], atol=0.02)
    # variance = lambda + lambda^2/r, strictly above the mean.
    assert np.allclose(s.var(0), [LAM1 + LAM1**2 / r, LAM2 + LAM2**2 / r], rtol=0.03)
    assert np.all(s.var(0) > s.mean(0))
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.01


def test_mean_property():
    nb = BivariateNegativeBinomial(LAM1, LAM2, 3.0)
    assert jnp.allclose(nb.mean, jnp.array([LAM1, LAM2]))


def test_variance_and_covariance_matrix():
    r = 3.0
    nb = BivariateNegativeBinomial(LAM1, LAM2, r)
    expected_var = jnp.array([LAM1 + LAM1**2 / r, LAM2 + LAM2**2 / r])
    assert jnp.allclose(nb.variance, expected_var)
    # Independent margins: diagonal covariance (used by the moments-linearized EKF).
    assert jnp.allclose(nb.covariance_matrix, jnp.diag(expected_var))


def test_log_prob_differentiable():
    """log_prob and its gradient w.r.t. lambda1, lambda2, log r are finite (PF/VI)."""

    def ll(l1, l2, log_r):
        return BivariateNegativeBinomial(l1, l2, jnp.exp(log_r)).log_prob(
            jnp.array([2.0, 1.0])
        )

    val = ll(jnp.array(LAM1), jnp.array(LAM2), jnp.log(jnp.array(5.0)))
    g = jax.grad(ll, argnums=(0, 1, 2))(
        jnp.array(LAM1), jnp.array(LAM2), jnp.log(jnp.array(5.0))
    )
    assert jnp.isfinite(val)
    assert all(jnp.isfinite(gi) for gi in g)


def test_batched_log_prob_shapes():
    """Broadcasting: batched rates with a batch of scorelines."""
    nb = BivariateNegativeBinomial(
        jnp.array([1.2, 0.8, 1.5]), jnp.array([1.0, 1.3, 0.9]), jnp.array(4.0)
    )
    assert nb.batch_shape == (3,)
    assert nb.event_shape == (2,)
    y = jnp.array([[2.0, 1.0], [0.0, 0.0], [3.0, 2.0]])
    assert nb.log_prob(y).shape == (3,)


def test_observation_model_means_and_dispersion():
    """The score observation returns NB rates from the skill contrasts; r = exp(log_disp)."""
    obs = BivariateNegativeBinomialScoreObservation(
        alpha=0.0, home_advantage=0.5, log_dispersion=float(jnp.log(7.0))
    )
    x = jnp.array([0.4, 0.1, 0.2, 0.3])  # [att_i, def_i, att_j, def_j]
    d = obs(x, None, 0.0)
    assert isinstance(d, BivariateNegativeBinomial)
    assert d.event_shape == (2,)
    # lam1 = exp(alpha + home_adv + att_i - def_j); lam2 = exp(alpha + att_j - def_i)
    assert jnp.allclose(d.lam1, jnp.exp(0.5 + 0.4 - 0.3))
    assert jnp.allclose(d.lam2, jnp.exp(0.2 - 0.1))
    assert jnp.allclose(d.dispersion, 7.0)
    # neutral venue removes the home advantage.
    d_neutral = obs(x, jnp.array([1.0, 0.0]), 0.0)
    assert jnp.allclose(d_neutral.lam1, jnp.exp(0.4 - 0.3))
