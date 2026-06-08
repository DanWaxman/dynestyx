"""Smoke tests for factorial state-space model filtering."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialKFConfig,
    FactorialPFConfig,
    Filter,
    GaussianObservation,
    MatchOutcomeObservation,
    RandomWalkEvolution,
)

from .fixtures import _n_particles

jax.config.update("jax_enable_x64", True)

F = 4
D = 1
N_LOCAL = 2
OBS_TIMES = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
OBS_IDX = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3], [3, 0], [2, 1]])
OBS_Y = jnp.array([1, 2, 0, 1, 2, 0])
SCORE_Y = jnp.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.4])[:, None]
K = OBS_TIMES.shape[0]


def _outcome_model(tau=0.1, eps=0.3):
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=tau, factor_state_dim=D),
        observation_model=MatchOutcomeObservation(draw_margin=eps, factor_state_dim=D),
        num_factors=F,
        num_local_factors=N_LOCAL,
    )


def _score_model(tau=0.1):
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=tau, factor_state_dim=D),
        observation_model=GaussianObservation(
            h=lambda x, u, t: jnp.atleast_1d(x[0] - x[1]), R=jnp.eye(1)
        ),
        num_factors=F,
        num_local_factors=N_LOCAL,
    )


def _run_filter(model_fn, config, obs_values, *, record=True):
    def model():
        cfg = config
        if record:
            cfg = config
        with Filter(filter_config=cfg):
            dsx.sample(
                "wc",
                model_fn(),
                obs_times=OBS_TIMES,
                obs_values=obs_values,
                obs_factor_indices=OBS_IDX,
            )

    return Predictive(model, num_samples=1)(jax.random.PRNGKey(0))


@pytest.mark.parametrize(
    "config_factory,model_fn,obs",
    [
        (
            lambda: FactorialEKFConfig(record_filtered_states_mean=True),
            _outcome_model,
            OBS_Y,
        ),
        (
            lambda: FactorialPFConfig(
                n_particles=_n_particles(2000), record_filtered_states_mean=True
            ),
            _outcome_model,
            OBS_Y,
        ),
        (
            lambda: FactorialKFConfig(record_filtered_states_mean=True),
            _score_model,
            SCORE_Y,
        ),
    ],
    ids=["ekf", "pf", "kf"],
)
def test_factorial_filter_shapes(config_factory, model_fn, obs):
    pr = _run_filter(model_fn, config_factory(), obs)
    loglik = pr["wc_marginal_loglik"][0]
    assert jnp.isfinite(loglik)
    assert pr["wc_filtered_states_mean"].shape == (1, F, D)
    assert pr["wc_filtered_states_chol_cov"].shape == (1, F, D, D)


def test_factorial_default_config_is_ekf():
    """No filter_config -> FactorialEKFConfig default for a factorial model."""

    def model():
        with Filter():
            dsx.sample(
                "wc",
                _outcome_model(),
                obs_times=OBS_TIMES,
                obs_values=OBS_Y,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["wc_marginal_loglik"][0])
