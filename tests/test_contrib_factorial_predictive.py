"""Tests for the contrib held-out predictive helper (dynestyx.contrib.factorial_predictive)."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from dynestyx import (
    BivariateNegativeBinomialScoreObservation,
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    MatchOutcomeObservation,
    RandomWalkEvolution,
)
from dynestyx.contrib import (
    one_step_ahead_local_states,
    outcome_predictive_probs,
    score_predictive,
    split_nll,
)

jax.config.update("jax_enable_x64", True)

F = 5
OBS_TIMES = jnp.array([1.0, 2.0, 3.0, 4.0])  # days
# Loop-free hub schedule: player 0 vs fresh opponents.
IDX = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4]])
RESULTS = jnp.array([1, 0, 2, 1])  # [home, draw, away, home]


def _outcome_model(tau=0.3, init_var=0.5, eps=0.2):
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), init_var * jnp.eye(1)),
        state_evolution=RandomWalkEvolution(tau=tau, factor_state_dim=1),
        observation_model=MatchOutcomeObservation(
            draw_margin=eps, scale=1.0, factor_state_dim=1
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )


def test_outcome_predictive_shapes_and_simplex():
    pp = outcome_predictive_probs(
        _outcome_model(),
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=RESULTS,
        obs_factor_indices=IDX,
    )
    assert pp.shape == (4, 3)
    assert jnp.all(jnp.isfinite(pp))
    assert jnp.allclose(pp.sum(-1), 1.0, atol=1e-5)
    assert jnp.all(pp > 0.0)


def test_first_appearance_predicted_marginal_is_prior_propagated():
    """The core reconstruction: a player's first-match predictive is the prior propagated."""
    init_var, tau = 0.5, 0.3
    pm, pc = one_step_ahead_local_states(
        _outcome_model(tau=tau, init_var=init_var),
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=RESULTS,
        obs_factor_indices=IDX,
    )
    # Match 0 = (0, 1): both first appearances at t=1, prior at t0=0.
    assert jnp.allclose(pm[0], 0.0, atol=1e-6)
    expected_var = init_var + tau**2 * (float(OBS_TIMES[0]) - 0.0)
    assert jnp.allclose(pc[0, 0, 0, 0] ** 2, expected_var, atol=1e-6)
    assert jnp.allclose(pc[0, 1, 0, 0] ** 2, expected_var, atol=1e-6)


def test_split_nll_arithmetic():
    pp = outcome_predictive_probs(
        _outcome_model(),
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=RESULTS,
        obs_factor_indices=IDX,
    )
    r = split_nll(pp, RESULTS, split_index=2)
    lp = jnp.log(pp[jnp.arange(4), RESULTS])
    assert np.isclose(r["nll"], float(-lp.mean()), atol=1e-6)
    assert np.isclose(r["train_nll"], float(-lp[:2].mean()), atol=1e-6)
    assert np.isclose(r["test_nll"], float(-lp[2:].mean()), atol=1e-6)


def test_ghq_matches_mc_outcome():
    """The deterministic GHQ predictive agrees with the Monte-Carlo predictive."""
    m = _outcome_model()
    pp_ghq = outcome_predictive_probs(
        m,
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=RESULTS,
        obs_factor_indices=IDX,
        method="ghq",
    )
    pp_mc = outcome_predictive_probs(
        m,
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=RESULTS,
        obs_factor_indices=IDX,
        method="mc",
        n_samples=40000,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(pp_ghq, pp_mc, atol=1.5e-2)


def test_score_predictive():
    model = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), 0.4 * jnp.eye(2)),
        state_evolution=RandomWalkEvolution(
            tau=jnp.array([0.3, 0.3]), factor_state_dim=2
        ),
        observation_model=BivariatePoissonScoreObservation(
            alpha=0.2, beta=-1.5, factor_state_dim=2, max_goals=9
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )
    scores = jnp.array([[2, 1], [0, 0], [3, 1], [1, 2]])
    out = score_predictive(
        model,
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=scores,
        obs_factor_indices=IDX,
        key=jax.random.PRNGKey(0),
        n_samples=400,
        max_goals_grid=8,
    )
    assert out["wdl"].shape == (4, 3)
    assert jnp.allclose(out["wdl"].sum(-1), 1.0, atol=1e-5)
    assert jnp.all(jnp.isfinite(out["score_logprob"]))
    # scores -> outcomes: [2,1]->home(1), [0,0]->draw(0), [3,1]->home(1), [1,2]->away(2)
    assert list(np.asarray(out["outcomes"])) == [1, 0, 1, 2]
    assert split_nll(out["wdl"], out["outcomes"], split_index=2)["test_nll"] > 0.0


def test_score_predictive_negative_binomial():
    """The overdispersed NB scoreline observation works in the same predictive path."""
    model = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), 0.4 * jnp.eye(2)),
        state_evolution=RandomWalkEvolution(
            tau=jnp.array([0.3, 0.3]), factor_state_dim=2
        ),
        observation_model=BivariateNegativeBinomialScoreObservation(
            alpha=0.2, log_dispersion=float(jnp.log(6.0)), factor_state_dim=2
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )
    scores = jnp.array([[2, 1], [0, 0], [3, 1], [1, 2]])
    out = score_predictive(
        model,
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=scores,
        obs_factor_indices=IDX,
        key=jax.random.PRNGKey(0),
        n_samples=400,
        max_goals_grid=8,
    )
    assert out["wdl"].shape == (4, 3)
    assert jnp.allclose(out["wdl"].sum(-1), 1.0, atol=1e-5)
    assert jnp.all(jnp.isfinite(out["score_logprob"]))
    assert list(np.asarray(out["outcomes"])) == [1, 0, 1, 2]
    assert split_nll(out["wdl"], out["outcomes"], split_index=2)["test_nll"] > 0.0
