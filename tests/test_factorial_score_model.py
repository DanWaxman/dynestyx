"""Smoke/shape tests for the attack/defense bivariate-Poisson factorial model."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    BivariateNegativeBinomialScoreObservation,
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialEKFSmootherConfig,
    FactorialPFConfig,
    Filter,
    OrnsteinUhlenbeckEvolution,
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
        # The default ("ekf") now exercises the moments path via auto-selection;
        # keep the Taylor flavour pinned explicitly.
        lambda: FactorialEKFConfig(use_taylor=True, record_filtered_states_mean=True),
        lambda: FactorialPFConfig(
            n_particles=_n_particles(1500), record_filtered_states_mean=True
        ),
    ],
    ids=["ekf", "taylor-ekf", "pf"],
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
    # Per-match local mean + chol-cov recorded by both backends (PF computes the
    # particle-weighted local covariance, matching the EKF) -- used for uncertainty bands.
    assert pr["sc_filtered_local_states_mean"].shape == (1, K, 2, D)
    assert pr["sc_filtered_local_states_chol_cov"].shape == (1, K, 2, D, D)
    assert jnp.all(jnp.isfinite(pr["sc_filtered_local_states_chol_cov"][0]))


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
        max_goals_grid=12,
    )
    assert grids.shape == (P, 13, 13)
    assert jnp.all(grids >= 0.0)
    # Grids sum to <= 1 and close to 1 (truncated at the grid cap).
    grid_mass = grids.sum(axis=(-2, -1))
    assert jnp.all(grid_mass <= 1.0 + 1e-6)
    assert jnp.all(grid_mass > 0.9)
    assert wdl.shape == (P, 3)
    assert jnp.allclose(wdl.sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.all(wdl >= 0.0)


def test_score_observation_per_match_controls():
    """u=[neutral, friendly] scales home advantage and offsets the baseline rate."""
    obs = BivariatePoissonScoreObservation(
        alpha=0.0,
        beta=-2.0,
        home_advantage=0.5,
        friendly_offset=0.3,
        factor_state_dim=2,
    )
    x = jnp.array([0.4, 0.1, 0.2, 0.3])  # [att_i, def_i, att_j, def_j]
    d_none = obs(x, None, 0.0)
    d_neutral = obs(x, jnp.array([1.0, 0.0]), 0.0)
    d_friendly = obs(x, jnp.array([0.0, 1.0]), 0.0)
    # lam1 = exp(alpha + friendly_off*friendly + home_adv*(1-neutral) + att_i - def_j)
    assert jnp.allclose(d_none.lam1, jnp.exp(0.5 + 0.4 - 0.3))
    assert jnp.allclose(
        d_neutral.lam1, jnp.exp(0.0 + 0.4 - 0.3)
    )  # home advantage removed
    assert jnp.allclose(
        d_friendly.lam1, jnp.exp(0.3 + 0.5 + 0.4 - 0.3)
    )  # baseline shifted


def test_obs_controls_change_filter_loglik():
    """Supplying obs_controls (neutral flags) changes the filter marginal likelihood."""
    from dynestyx.inference.integrations.cuthbert.factorial_filter import (
        compute_factorial_filter,
    )

    fm = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.4 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(
            tau=jnp.array([0.3, 0.3]), factor_state_dim=D
        ),
        observation_model=BivariatePoissonScoreObservation(
            alpha=0.2, beta=-1.5, home_advantage=0.4, factor_state_dim=D, max_goals=12
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )
    controls = jnp.zeros((K, 2)).at[:, 0].set(1.0)  # all neutral -> no home advantage
    ll_home, *_ = compute_factorial_filter(
        fm,
        FactorialEKFConfig(),
        None,
        obs_times=OBS_TIMES,
        obs_values=SCORES,
        obs_factor_indices=OBS_IDX,
    )
    ll_neutral, *_ = compute_factorial_filter(
        fm,
        FactorialEKFConfig(),
        None,
        obs_times=OBS_TIMES,
        obs_values=SCORES,
        obs_factor_indices=OBS_IDX,
        obs_controls=controls,
    )
    assert jnp.isfinite(ll_home) and jnp.isfinite(ll_neutral)
    assert not jnp.allclose(ll_home, ll_neutral)


def _model_nb():
    """Same score model with the overdispersed bivariate-negative-binomial likelihood."""
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.3 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(
            tau=jnp.array([0.2, 0.2]), factor_state_dim=D
        ),
        observation_model=BivariateNegativeBinomialScoreObservation(
            alpha=0.2, log_dispersion=float(jnp.log(8.0)), factor_state_dim=D
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: FactorialEKFConfig(record_filtered_states_mean=True),
        # The default ("ekf") now exercises the moments path via auto-selection;
        # keep the Taylor flavour pinned explicitly.
        lambda: FactorialEKFConfig(use_taylor=True, record_filtered_states_mean=True),
        lambda: FactorialPFConfig(
            n_particles=_n_particles(1500), record_filtered_states_mean=True
        ),
    ],
    ids=["ekf", "taylor-ekf", "pf"],
)
def test_nb_score_model_filter_shapes(config_factory):
    def model():
        with Filter(filter_config=config_factory()):
            dsx.sample(
                "sc",
                _model_nb(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["sc_marginal_loglik"][0])
    assert pr["sc_filtered_states_mean"].shape == (1, F, D)
    assert pr["sc_filtered_states_chol_cov"].shape == (1, F, D, D)
    # Per-match local mean + chol-cov recorded by both backends (PF computes the
    # particle-weighted local covariance, matching the EKF) -- used for uncertainty bands.
    assert pr["sc_filtered_local_states_mean"].shape == (1, K, 2, D)
    assert pr["sc_filtered_local_states_chol_cov"].shape == (1, K, 2, D, D)
    assert jnp.all(jnp.isfinite(pr["sc_filtered_local_states_chol_cov"][0]))


def test_nb_score_model_rejects_scalar_state():
    with pytest.raises(ValueError, match="factor_state_dim=2"):
        BivariateNegativeBinomialScoreObservation(alpha=0.2, factor_state_dim=1)


def test_nb_score_prediction_grids_and_wdl():
    def model():
        with Filter(filter_config=FactorialEKFConfig()):
            dsx.sample(
                "sc",
                _model_nb(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
                predict_times=PRED_TIMES,
                predict_factor_indices=PRED_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    grids, wdl = factorial_score_probabilities(
        BivariateNegativeBinomialScoreObservation(
            alpha=0.2, log_dispersion=float(jnp.log(8.0)), factor_state_dim=D
        ),
        pr["sc_predicted_states"][0],
        pr["sc_predicted_states_chol_cov"][0],
        key=jax.random.PRNGKey(1),
        n_samples=400,
        max_goals_grid=12,
    )
    assert grids.shape == (P, 13, 13)
    assert jnp.all(grids >= 0.0)
    grid_mass = grids.sum(axis=(-2, -1))
    assert jnp.all(grid_mass <= 1.0 + 1e-6)
    assert jnp.all(grid_mass > 0.9)
    assert wdl.shape == (P, 3)
    assert jnp.allclose(wdl.sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.all(wdl >= 0.0)


def test_ou_evolution_transition_and_limits():
    """OU transition matches the analytic mean/variance and its stationary/BM limits."""
    r, s0, mu = 0.8, 0.6, 0.3
    ou = OrnsteinUhlenbeckEvolution(
        reversion_rate=r, equilibrium_scale=s0, long_run_mean=mu, factor_state_dim=1
    )
    x = jnp.array([1.0])
    for dt in (0.5, 2.0):
        d = ou(x, None, 0.0, dt)
        decay = jnp.exp(-r * dt)
        assert jnp.allclose(d.mean[0], decay + mu * (1.0 - decay))
        assert jnp.allclose(
            d.covariance_matrix[0, 0], s0**2 * (1.0 - jnp.exp(-2.0 * r * dt)), atol=1e-9
        )
    # dt -> infinity: stationary distribution N(mu, s0^2).
    d_inf = ou(x, None, 0.0, 1e6)
    assert jnp.allclose(d_inf.mean[0], mu, atol=1e-4)
    assert jnp.allclose(d_inf.covariance_matrix[0, 0], s0**2, atol=1e-4)
    # dt = 0: identity transition with (near-)zero variance.
    d0 = ou(x, None, 5.0, 5.0)
    assert jnp.allclose(d0.mean[0], 1.0)
    assert float(d0.covariance_matrix[0, 0]) < 1e-9


def test_ou_evolution_differentiable_vector():
    """OU log_prob is finite and differentiable in (reversion_rate, equilibrium_scale)."""

    def loss(p):
        ou = OrnsteinUhlenbeckEvolution(
            reversion_rate=p[0],
            equilibrium_scale=p[1],
            long_run_mean=0.0,
            factor_state_dim=2,
        )
        return ou(jnp.array([0.5, -0.2]), None, 0.0, 1.5).log_prob(
            jnp.array([0.3, 0.1])
        )

    g = jax.grad(loss)(jnp.array([0.5, 0.7]))
    assert jnp.all(jnp.isfinite(g))


def test_ou_score_model_filter_smoke():
    """A d=2 score model with OU skill dynamics filters to finite loglik + right shapes."""
    fm = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.5 * jnp.eye(D)),
        state_evolution=OrnsteinUhlenbeckEvolution(
            reversion_rate=jnp.array([0.5, 0.5]),
            equilibrium_scale=jnp.array([0.5, 0.5]),
            long_run_mean=jnp.zeros(D),
            factor_state_dim=D,
        ),
        observation_model=BivariatePoissonScoreObservation(
            alpha=0.2, beta=-1.5, factor_state_dim=D, max_goals=12
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )

    def model():
        with Filter(filter_config=FactorialEKFConfig(record_filtered_states_mean=True)):
            dsx.sample(
                "sc",
                fm,
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(model, num_samples=1)(jax.random.PRNGKey(0))
    assert jnp.isfinite(pr["sc_marginal_loglik"][0])
    assert pr["sc_filtered_states_mean"].shape == (1, F, D)
