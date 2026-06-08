"""Validation tests for the factorial marginal log-likelihood accumulation.

The core correctness claim (Duffield et al. §3.4 + the cuthbert factorial scan) is
that ``log p(y_{1:K}) = sum_k log p(y_k | y_{1:k-1})`` is accumulated correctly
across the extract -> filter -> marginalize -> insert loop. We check this two ways:

1. **No-leakage / accumulation**: on a schedule where every factor appears in
   exactly one match, the matches are independent, so the total marginal
   log-likelihood must equal the sum of the per-match marginal log-likelihoods
   (each computed as a stand-alone two-factor problem). This is exact for the
   deterministic EKF backend.
2. **Per-match correctness**: the factorial particle-filter marginal
   log-likelihood matches a brute-force Monte-Carlo estimate of the per-match
   predictive outcome probabilities.

Plus a differentiability check (the PF marginal log-likelihood has finite
gradients in the dynamics parameters).
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx import (
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialPFConfig,
    MatchOutcomeObservation,
    RandomWalkEvolution,
)
from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    compute_factorial_filter,
)

jax.config.update("jax_enable_x64", True)

D = 1
INIT_VAR = 0.5
TAU = 0.2
EPS = 0.3
T0 = 0.0
# No-repeat schedule: 6 teams, 3 matches, each team plays exactly once.
PAIRS = jnp.array([[0, 1], [2, 3], [4, 5]])
TIMES = jnp.array([1.0, 2.0, 3.0])
YS = jnp.array([1, 2, 0])


def _model(num_factors, tau=TAU, eps=EPS):
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), INIT_VAR * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=tau, factor_state_dim=D),
        observation_model=MatchOutcomeObservation(draw_margin=eps, factor_state_dim=D),
        num_factors=num_factors,
        num_local_factors=2,
        t0=T0,
    )


def test_ekf_loglik_decomposes_over_independent_matches():
    """No-repeat schedule => total loglik == sum of per-match logliks (EKF, exact)."""
    fm_full = _model(num_factors=6)
    total_ll, *_ = compute_factorial_filter(
        fm_full,
        FactorialEKFConfig(),
        None,
        obs_times=TIMES,
        obs_values=YS,
        obs_factor_indices=PAIRS,
    )

    per_match = 0.0
    fm2 = _model(num_factors=2)
    for k in range(PAIRS.shape[0]):
        ll_k, *_ = compute_factorial_filter(
            fm2,
            FactorialEKFConfig(),
            None,
            obs_times=TIMES[k : k + 1],
            obs_values=YS[k : k + 1],
            obs_factor_indices=jnp.array([[0, 1]]),
        )
        per_match = per_match + ll_k

    assert jnp.allclose(total_ll, per_match, atol=1e-6), (total_ll, per_match)


def _mc_reference_loglik(key, n_samples=40_000):
    """Brute-force MC estimate of the no-repeat-schedule marginal log-likelihood."""
    obs_model = MatchOutcomeObservation(draw_margin=EPS, factor_state_dim=D)
    total = 0.0
    for k in range(PAIRS.shape[0]):
        key, ka, kb = jax.random.split(key, 3)
        var = INIT_VAR + TAU**2 * (TIMES[k] - T0)
        std = jnp.sqrt(var)
        xa = std * jax.random.normal(ka, (n_samples,))
        xb = std * jax.random.normal(kb, (n_samples,))
        joint = jnp.stack([xa, xb], axis=1)  # (n, 2)
        probs = jax.vmap(lambda xj: obs_model(xj, None, 0.0).probs)(joint)
        py = probs[:, int(YS[k])]
        total = total + jnp.log(jnp.mean(py))
    return total


def test_pf_loglik_matches_monte_carlo_reference():
    """Factorial PF marginal loglik matches a brute-force MC estimate (no-repeat)."""
    fm = _model(num_factors=6)
    pf_ll, *_ = compute_factorial_filter(
        fm,
        FactorialPFConfig(n_particles=6000),
        jax.random.PRNGKey(0),
        obs_times=TIMES,
        obs_values=YS,
        obs_factor_indices=PAIRS,
    )
    mc_ll = _mc_reference_loglik(jax.random.PRNGKey(1))
    assert jnp.allclose(pf_ll, mc_ll, atol=0.25), (pf_ll, mc_ll)


def test_pf_loglik_is_differentiable_in_tau():
    fm_idx = PAIRS
    times = TIMES
    ys = YS

    def loss(tau):
        fm = _model(num_factors=6, tau=tau)
        ll, *_ = compute_factorial_filter(
            fm,
            FactorialPFConfig(n_particles=2000),
            jax.random.PRNGKey(0),
            obs_times=times,
            obs_values=ys,
            obs_factor_indices=fm_idx,
        )
        return ll

    g = jax.grad(loss)(0.2)
    assert jnp.isfinite(g)
