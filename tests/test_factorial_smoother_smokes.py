"""Smoke tests for factorial state-space model smoothing."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    FactorialDynamicalModel,
    FactorialEKFSmootherConfig,
    FactorialKFSmootherConfig,
    FactorialPFSmootherConfig,
    MatchOutcomeObservation,
    RandomWalkEvolution,
    Smoother,
)

jax.config.update("jax_enable_x64", True)

F = 4
D = 1
OBS_TIMES = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
OBS_IDX = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3], [3, 0], [2, 1]])
OBS_Y = jnp.array([1, 2, 0, 1, 2, 0])
K = OBS_TIMES.shape[0]


def _model():
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=D),
        observation_model=MatchOutcomeObservation(draw_margin=0.3, factor_state_dim=D),
        num_factors=F,
        num_local_factors=2,
    )


def test_factorial_ekf_smoother_shapes():
    def model():
        with Smoother(
            smoother_config=FactorialEKFSmootherConfig(record_smoothed_states_mean=True)
        ):
            dsx.sample(
                "wc",
                _model(),
                obs_times=OBS_TIMES,
                obs_values=OBS_Y,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["wc_marginal_loglik"][0])
    assert pr["wc_smoothed_local_states_mean"].shape == (1, K, 2, D)
    assert pr["wc_smoothed_local_states_chol_cov"].shape == (1, K, 2, D, D)


def test_factorial_kf_smoother_runs():
    """KF smoothing requires a linear-Gaussian observation (here a score margin)."""
    from dynestyx import GaussianObservation

    score_y = jnp.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.4])[:, None]

    def score_model():
        return FactorialDynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
            state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=D),
            observation_model=GaussianObservation(
                h=lambda x, u, t: jnp.atleast_1d(x[0] - x[1]), R=jnp.eye(1)
            ),
            num_factors=F,
            num_local_factors=2,
        )

    def model():
        with Smoother(smoother_config=FactorialKFSmootherConfig()):
            dsx.sample(
                "wc",
                score_model(),
                obs_times=OBS_TIMES,
                obs_values=score_y,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["wc_marginal_loglik"][0])


def test_factorial_pf_smoother_not_implemented():
    def model():
        with Smoother(smoother_config=FactorialPFSmootherConfig()):
            dsx.sample(
                "wc",
                _model(),
                obs_times=OBS_TIMES,
                obs_values=OBS_Y,
                obs_factor_indices=OBS_IDX,
            )

    with pytest.raises(NotImplementedError):
        Predictive(model, num_samples=1)(jax.random.PRNGKey(0))


def test_smoothed_equals_filtered_at_last_match():
    """For a factor's final match, the smoothed estimate equals the filtered one."""
    from dynestyx.inference.integrations.cuthbert.factorial_filter import (
        compute_factorial_filter,
    )
    from dynestyx.inference.integrations.cuthbert.factorial_smoother import (
        compute_factorial_smoother,
    )

    fm = _model()
    _, final, _local, _lt = compute_factorial_filter(
        fm,
        dsx.FactorialEKFConfig(),
        None,
        obs_times=OBS_TIMES,
        obs_values=OBS_Y,
        obs_factor_indices=OBS_IDX,
    )
    _, sm_means, _ = compute_factorial_smoother(
        fm,
        FactorialEKFSmootherConfig(),
        None,
        obs_times=OBS_TIMES,
        obs_values=OBS_Y,
        obs_factor_indices=OBS_IDX,
    )
    # Factor 3's last match is match index 4 (position 0). Smoothed there == filtered.
    final_mean_factor3 = final.elem.b[3]
    smoothed_local_match4_pos0 = sm_means[4, 0]
    assert jnp.allclose(final_mean_factor3, smoothed_local_match4_pos0, atol=1e-6)
