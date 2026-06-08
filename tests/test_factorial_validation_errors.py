"""Validation-error tests for the factorial sample() interface."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    DynamicalModel,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    Filter,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    MatchOutcomeObservation,
    RandomWalkEvolution,
)

jax.config.update("jax_enable_x64", True)

F = 4
D = 1
OT = jnp.array([0.0, 1.0, 2.0])
OIDX = jnp.array([[0, 1], [2, 3], [0, 2]])
OY = jnp.array([1, 2, 0])


def _factorial_model():
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=D),
        observation_model=MatchOutcomeObservation(draw_margin=0.3, factor_state_dim=D),
        num_factors=F,
        num_local_factors=2,
    )


def test_missing_obs_factor_indices():
    with pytest.raises(ValueError, match="obs_factor_indices"):
        dsx.sample("f", _factorial_model(), obs_times=OT, obs_values=OY)


def test_wrong_shape_obs_factor_indices():
    with pytest.raises(ValueError, match="shape"):
        dsx.sample(
            "f",
            _factorial_model(),
            obs_times=OT,
            obs_values=OY,
            obs_factor_indices=jnp.array([0, 1, 2]),
        )


def test_out_of_range_indices():
    with pytest.raises(ValueError, match="outside"):
        dsx.sample(
            "f",
            _factorial_model(),
            obs_times=OT,
            obs_values=OY,
            obs_factor_indices=jnp.array([[0, 1], [2, 9], [0, 2]]),
        )


def test_indices_on_non_factorial_model():
    dm = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(2), cov=jnp.eye(2)),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=jnp.eye(2)),
    )
    with pytest.raises(ValueError, match="only valid for a FactorialDynamicalModel"):
        dsx.sample(
            "f",
            dm,
            obs_times=OT,
            obs_values=jnp.zeros((3, 2)),
            obs_factor_indices=OIDX,
        )


def test_continuous_state_evolution_rejected():
    from dynestyx import ContinuousTimeStateEvolution

    with pytest.raises(ValueError, match="discrete-time only"):
        FactorialDynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(D), jnp.eye(D)),
            state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: -x),
            observation_model=MatchOutcomeObservation(
                draw_margin=0.3, factor_state_dim=D
            ),
            num_factors=F,
            num_local_factors=2,
        )


def test_scalar_event_initial_condition_rejected():
    with pytest.raises(ValueError, match="event_shape"):
        FactorialDynamicalModel(
            initial_condition=dist.Normal(0.0, 1.0),  # event_shape == ()
            state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=D),
            observation_model=MatchOutcomeObservation(
                draw_margin=0.3, factor_state_dim=D
            ),
            num_factors=F,
            num_local_factors=2,
        )


def test_factorial_under_plate_rejected():
    def model():
        with dsx.plate("p", 2):
            dsx.sample(
                "f",
                _factorial_model(),
                obs_times=OT,
                obs_values=OY,
                obs_factor_indices=OIDX,
            )

    def filtered():
        with Filter(filter_config=FactorialEKFConfig()):
            model()

    with pytest.raises(ValueError, match="does not support dsx.plate"):
        Predictive(filtered, num_samples=1)(jax.random.PRNGKey(0))
