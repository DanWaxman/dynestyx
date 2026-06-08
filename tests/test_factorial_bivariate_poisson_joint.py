"""Validate the bivariate-Poisson (score) factorial marginal likelihood vs an exact joint.

The score observation is non-Gaussian, so there is no closed-form joint marginal
likelihood. We build deterministic **grid Bayes filters** on the full F=2, d=2 (4-D)
latent state:

* ``_grid_joint_factored`` keeps the full 4-D posterior -> the **exact** marginal
  likelihood; and (same code) projects to a product of per-factor marginals after each
  pairwise update -> the strict mean-field **factored** approximation.

Repeated ``(0, 1)`` matches form a cyclic schedule (the worst case for the factored
approximation). This pins, for the *actual* model the World Cup notebook forecasts with:

1. the factored projection genuinely loses likelihood on a cyclic schedule (and the
   grid reference is converged); and
2. the dynestyx **EKF** marginal likelihood **overestimates** the exact joint value on
   these rank-deficient *contrast* observations -- so parameter inference should use
   ``FactorialPFConfig`` (validated against a joint particle filter to MC noise in the
   companion analysis), not ``FactorialEKFConfig``, for the likelihood value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from dynestyx import (
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    RandomWalkEvolution,
)
from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    compute_factorial_filter,
)
from dynestyx.models.distributions import BivariatePoisson

jax.config.update("jax_enable_x64", True)

_ALPHA, _BETA, _MAXG = 0.2, -1.5, 8
_MU0 = np.array([0.0, 0.0])
_SIG0 = np.diag([0.6**2, 0.5**2])
_TAU = np.array([0.32, 0.28])  # per-dim tau (sqrt rate)
_TAU2 = _TAU**2


def _bp_logpmf(yh, ya, att_i, def_i, att_j, def_j):
    lam1 = jnp.exp(_ALPHA + att_i - def_j)
    lam2 = jnp.exp(_ALPHA + att_j - def_i)
    lam3 = jnp.broadcast_to(jnp.exp(jnp.asarray(_BETA, float)), lam1.shape)
    d = BivariatePoisson(lam1, lam2, lam3, max_goals=_MAXG)
    val = jnp.stack(
        [
            jnp.broadcast_to(jnp.asarray(yh, float), lam1.shape),
            jnp.broadcast_to(jnp.asarray(ya, float), lam1.shape),
        ],
        axis=-1,
    )
    return d.log_prob(val)


def _grid_joint_factored(times, y, G=21, R=2.6):
    """Exact joint and strict-mean-field factored marginal loglik (F=2 repeated (0,1))."""
    g = jnp.linspace(-R, R, G)
    pa = jnp.exp(-0.5 * (g - _MU0[0]) ** 2 / _SIG0[0, 0])
    pd = jnp.exp(-0.5 * (g - _MU0[1]) ** 2 / _SIG0[1, 1])
    pa, pd = pa / pa.sum(), pd / pd.sum()
    A, B, C, D = jnp.meshgrid(g, g, g, g, indexing="ij")  # att0, def0, att1, def1

    def kernel(tau2_m, dt):
        var = tau2_m * dt + 1e-12
        diff = g[None, :] - g[:, None]
        K = jnp.exp(-0.5 * diff**2 / var)
        return K / K.sum(axis=1, keepdims=True)

    p = (
        pa[:, None, None, None]
        * pd[None, :, None, None]
        * pa[None, None, :, None]
        * pd[None, None, None, :]
    )
    q0 = pa[:, None] * pd[None, :]
    q1 = pa[:, None] * pd[None, :]

    def prop_joint(p, dt):
        ka, kd = kernel(_TAU2[0], dt), kernel(_TAU2[1], dt)
        p = jnp.einsum("abcd,aA->Abcd", p, ka)
        p = jnp.einsum("abcd,bB->aBcd", p, kd)
        p = jnp.einsum("abcd,cC->abCd", p, ka)
        return jnp.einsum("abcd,dD->abcD", p, kd)

    def prop_factor(q, dt):
        ka, kd = kernel(_TAU2[0], dt), kernel(_TAU2[1], dt)
        return jnp.einsum("ab,bB->aB", jnp.einsum("ab,aA->Ab", q, ka), kd)

    ll_j, ll_f, t_prev = 0.0, 0.0, 0.0
    for k in range(len(times)):
        dt = float(times[k]) - t_prev
        t_prev = float(times[k])
        lc = jnp.exp(_bp_logpmf(int(y[k][0]), int(y[k][1]), A, B, C, D))

        p = prop_joint(p, dt)
        Zj = (p * lc).sum()
        ll_j += float(jnp.log(Zj))
        p = p * lc / Zj

        q0, q1 = prop_factor(q0, dt), prop_factor(q1, dt)
        unf = (q0[:, :, None, None] * q1[None, None, :, :]) * lc
        Zf = unf.sum()
        ll_f += float(jnp.log(Zf))
        post = unf / Zf
        q0, q1 = post.sum(axis=(2, 3)), post.sum(axis=(0, 1))
    return ll_j, ll_f


def _dsx_ekf(times, idx, y):
    model = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.asarray(_MU0), jnp.asarray(_SIG0)
        ),
        state_evolution=RandomWalkEvolution(tau=jnp.asarray(_TAU), factor_state_dim=2),
        observation_model=BivariatePoissonScoreObservation(
            alpha=_ALPHA, beta=_BETA, factor_state_dim=2, max_goals=_MAXG
        ),
        num_factors=2,
        num_local_factors=2,
        t0=0.0,
    )
    ll, *_ = compute_factorial_filter(
        model,
        FactorialEKFConfig(),
        None,
        obs_times=jnp.asarray(times, float),
        obs_values=jnp.asarray(y, jnp.int32),
        obs_factor_indices=jnp.asarray(idx, jnp.int32),
    )
    return float(ll)


_TIMES = 0.7 * np.arange(1, 4)  # K=3
_IDX = np.array([[0, 1]] * 3)
_Y = np.array([[3, 1], [0, 1], [2, 2]])


def test_grid_reference_converged_and_factored_loses_likelihood_on_cycle():
    """The exact-joint grid is converged, and the factored projection loses likelihood."""
    ll_j, ll_f = _grid_joint_factored(_TIMES, _Y, G=21)
    ll_j2, ll_f2 = _grid_joint_factored(_TIMES, _Y, G=27)
    # Grid Bayes filter is converged in resolution.
    assert abs(ll_j - ll_j2) < 1e-2, f"joint grid not converged: {ll_j} vs {ll_j2}"
    assert abs(ll_f - ll_f2) < 1e-2, f"factored grid not converged: {ll_f} vs {ll_f2}"
    # On a cyclic (repeated-match) schedule the product-of-marginals projection is lossy:
    # it underestimates the exact joint marginal likelihood.
    assert ll_f < ll_j - 1e-3, (
        f"expected projection loss, got joint {ll_j}, factored {ll_f}"
    )
    assert ll_j - ll_f < 1.0, "projection error unexpectedly large for this small cycle"


def test_ekf_marginal_loglik_overestimates_exact_joint_on_contrast_obs():
    """The EKF marginal likelihood is biased *upward* vs the exact joint (rank-deficient obs)."""
    ll_j, _ = _grid_joint_factored(_TIMES, _Y, G=27)
    ekf = _dsx_ekf(_TIMES, _IDX, _Y)
    assert np.isfinite(ekf)
    # The contrast (att_i - def_j, att_j - def_i) leaves the additive direction
    # unobserved -> the Taylor/ignore_nan_dims path *overestimates* log p(y_{1:K}).
    assert ekf > ll_j + 0.1, (
        f"expected EKF to overestimate the exact joint {ll_j}, got {ekf}"
    )
    assert ekf - ll_j < 12.0, f"EKF bias {ekf - ll_j:.2f} outside the expected band"
