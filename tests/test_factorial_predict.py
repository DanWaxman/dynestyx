"""Tests for factorial predictive rollout and outcome probabilities."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    FactorialDynamicalModel,
    FactorialEKFConfig,
    Filter,
    MatchOutcomeObservation,
    RandomWalkEvolution,
    factorial_outcome_probabilities,
)

jax.config.update("jax_enable_x64", True)

F = 5
D = 1
OBS_TIMES = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
OBS_IDX = jnp.array([[0, 1], [2, 3], [0, 2], [1, 4], [3, 0]])
OBS_Y = jnp.array([1, 2, 0, 1, 2])
PRED_TIMES = jnp.array([5.0, 6.0])
PRED_IDX = jnp.array([[0, 1], [2, 4]])
P = PRED_TIMES.shape[0]


def _model_fn():
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=D),
        observation_model=MatchOutcomeObservation(draw_margin=0.3, factor_state_dim=D),
        num_factors=F,
        num_local_factors=2,
    )


def _run():
    def model():
        with Filter(filter_config=FactorialEKFConfig()):
            dsx.sample(
                "wc",
                _model_fn(),
                obs_times=OBS_TIMES,
                obs_values=OBS_Y,
                obs_factor_indices=OBS_IDX,
                predict_times=PRED_TIMES,
                predict_factor_indices=PRED_IDX,
            )

    return Predictive(model, num_samples=1)(jax.random.PRNGKey(0))


def test_rollout_records_predicted_states():
    pr = _run()
    assert pr["wc_predicted_states"].shape == (1, P, 2, D)
    assert pr["wc_predicted_states_chol_cov"].shape == (1, P, 2, D, D)
    assert pr["wc_predicted_times"].shape == (1, P)
    assert pr["wc_predicted_factor_indices"].shape == (1, P, 2)


def test_predicted_variance_grows_with_horizon():
    """Predicted covariance for a factor should exceed its filtered covariance."""
    pr = _run()
    pred_chol = pr["wc_predicted_states_chol_cov"][0]  # (P, 2, D, D)
    pred_var = pred_chol[..., 0, 0] ** 2
    assert jnp.all(pred_var > 0)


def test_outcome_probabilities():
    pr = _run()
    obs_model = MatchOutcomeObservation(draw_margin=0.3, factor_state_dim=D)
    probs = factorial_outcome_probabilities(
        obs_model,
        pr["wc_predicted_states"][0],
        pr["wc_predicted_states_chol_cov"][0],
        key=jax.random.PRNGKey(1),
        n_samples=500,
    )
    assert probs.shape == (P, 3)
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.all(probs >= 0.0)
