"""Smoke/shape tests for the attack/defense bivariate-Poisson factorial model."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialEKFSmootherConfig,
    FactorialPFConfig,
    Filter,
    RandomWalkEvolution,
    Smoother,
    factorial_score_probabilities,
)

from .fixtures import _n_particles

jax.config.update("jax_enable_x64", True)

F = 5
D = 2  # (attack, defense)
OBS_TIMES = jnp.arange(8.0)
OBS_IDX = jnp.array([[0, 1], [2, 3], [0, 2], [1, 4], [3, 0], [2, 1], [0, 3], [1, 2]])
SCORES = jnp.array([[2, 1], [0, 0], [3, 1], [1, 2], [2, 2], [0, 3], [1, 1], [4, 0]])
PRED_TIMES = jnp.array([8.0, 9.0])
PRED_IDX = jnp.array([[0, 1], [2, 4]])
K = OBS_TIMES.shape[0]
P = PRED_TIMES.shape[0]


def _model():
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.3 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(
            tau=jnp.array([0.2, 0.2]), factor_state_dim=D
        ),
        observation_model=BivariatePoissonScoreObservation(
            alpha=0.2, beta=-1.5, factor_state_dim=D, max_goals=12
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: FactorialEKFConfig(record_filtered_states_mean=True),
        lambda: FactorialPFConfig(
            n_particles=_n_particles(1500), record_filtered_states_mean=True
        ),
    ],
    ids=["ekf", "pf"],
)
def test_score_model_filter_shapes(config_factory):
    def model():
        with Filter(filter_config=config_factory()):
            dsx.sample(
                "sc",
                _model(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["sc_marginal_loglik"][0])
    assert pr["sc_filtered_states_mean"].shape == (1, F, D)
    assert pr["sc_filtered_states_chol_cov"].shape == (1, F, D, D)


def test_score_model_rejects_scalar_state():
    with pytest.raises(ValueError, match="factor_state_dim=2"):
        BivariatePoissonScoreObservation(alpha=0.2, beta=-1.5, factor_state_dim=1)


def test_score_model_smoother_shapes():
    def model():
        with Smoother(
            smoother_config=FactorialEKFSmootherConfig(record_smoothed_states_mean=True)
        ):
            dsx.sample(
                "sc",
                _model(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert pr["sc_smoothed_local_states_mean"].shape == (1, K, 2, D)


def test_score_prediction_grids_and_wdl():
    def model():
        with Filter(filter_config=FactorialEKFConfig()):
            dsx.sample(
                "sc",
                _model(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
                predict_times=PRED_TIMES,
                predict_factor_indices=PRED_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert pr["sc_predicted_states"].shape == (1, P, 2, D)

    grids, wdl = factorial_score_probabilities(
        BivariatePoissonScoreObservation(
            alpha=0.2, beta=-1.5, factor_state_dim=D, max_goals=12
        ),
        pr["sc_predicted_states"][0],
        pr["sc_predicted_states_chol_cov"][0],
        key=jax.random.PRNGKey(1),
        n_samples=400,
        max_goals_grid=8,
    )
    assert grids.shape == (P, 9, 9)
    assert jnp.all(grids >= 0.0)
    # Grids sum to <= 1 and close to 1 (truncated at the grid cap).
    grid_mass = grids.sum(axis=(-2, -1))
    assert jnp.all(grid_mass <= 1.0 + 1e-6)
    assert jnp.all(grid_mass > 0.9)
    assert wdl.shape == (P, 3)
    assert jnp.allclose(wdl.sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.all(wdl >= 0.0)
