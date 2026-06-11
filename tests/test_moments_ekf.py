"""Tests for the moments-linearized EKF path (``EKFConfig.use_taylor``).

The cuthbert EKF integration auto-selects between exact-conditional-moments
linearization (``cuthbert.gaussian.moments``: Jacobian of the mean, no Hessian)
and Taylor linearization of the log-density. Exact numeric agreement is asserted
only where it is mathematically exact (linear-Gaussian models, loglik
accumulation over independent matches); elsewhere we assert finiteness,
fallback identity, and the headline capability — a *differentiable* factorial
EKF marginal log-likelihood for the bivariate-Poisson score model, where the
Taylor flavour's gradient is NaN.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

import dynestyx as dsx
from dynestyx import (
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialEKFSmootherConfig,
    FactorialKFConfig,
    FactorialPFConfig,
    Filter,
    GaussianObservation,
    MatchOutcomeObservation,
    RandomWalkEvolution,
    Smoother,
)
from dynestyx.inference.filter_configs import EKFConfig, KFConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    compute_cuthbert_smoother,
)
from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    compute_factorial_filter,
)
from dynestyx.inference.smoother_configs import EKFSmootherConfig, KFSmootherConfig

from .fixtures import _n_particles

jax.config.update("jax_enable_x64", True)


def _cov(chol):
    return chol @ jnp.swapaxes(chol, -1, -2)


# ---------------------------------------------------------------------------
# Non-factorial (discrete_filter / discrete_smoother)
# ---------------------------------------------------------------------------
def _lti_dynamics():
    return dsx.LTI_discrete(
        A=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
        Q=0.1 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.25]]),
    )


LTI_TIMES = jnp.arange(8.0)
LTI_YS = jr.normal(jr.PRNGKey(0), (8, 1))


def test_discrete_moments_ekf_matches_taylor_and_kf_on_lti():
    """KF, Taylor-EKF, and moments-EKF are mathematically exact on linear-Gaussian."""
    dynamics = _lti_dynamics()
    results = {}
    for name, cfg in [
        ("kf", KFConfig(filter_source="cuthbert")),
        ("taylor", EKFConfig(use_taylor=True)),
        ("moments", EKFConfig(use_taylor=False)),
        ("auto", EKFConfig()),
    ]:
        ll, states = compute_cuthbert_filter(
            dynamics, cfg, obs_times=LTI_TIMES, obs_values=LTI_YS
        )
        results[name] = (ll, states.mean, _cov(states.chol_cov))

    ll_kf, mean_kf, cov_kf = results["kf"]
    for name in ("taylor", "moments", "auto"):
        ll, mean, cov = results[name]
        assert jnp.allclose(ll, ll_kf, rtol=1e-6, atol=1e-6), (name, ll, ll_kf)
        assert jnp.allclose(mean, mean_kf, rtol=1e-6, atol=1e-6), name
        assert jnp.allclose(cov, cov_kf, rtol=1e-6, atol=1e-6), name

    # The LTI model exposes moments everywhere, so auto takes the moments path.
    assert jnp.array_equal(results["auto"][0], results["moments"][0])


def _poisson_obs_dynamics():
    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.5 * jnp.eye(1)),
        state_evolution=dsx.models.LinearGaussianStateEvolution(
            A=jnp.array([[0.9]]), cov=0.1 * jnp.eye(1)
        ),
        observation_model=lambda x, u, t: dist.Poisson(rate=jnp.exp(x[0])),
    )


def test_discrete_moments_auto_falls_back_without_moments():
    """Poisson obs exposes no covariance: auto == Taylor; use_taylor=False raises."""
    dynamics = _poisson_obs_dynamics()
    counts = jnp.array([[1.0], [0.0], [2.0], [1.0]])
    times = jnp.arange(4.0)

    ll_auto, _ = compute_cuthbert_filter(
        dynamics, EKFConfig(), obs_times=times, obs_values=counts
    )
    ll_taylor, _ = compute_cuthbert_filter(
        dynamics, EKFConfig(use_taylor=True), obs_times=times, obs_values=counts
    )
    assert jnp.array_equal(ll_auto, ll_taylor)  # identical code path

    with pytest.raises(TypeError, match="covariance"):
        compute_cuthbert_filter(
            dynamics,
            EKFConfig(use_taylor=False),
            obs_times=times,
            obs_values=counts,
        )


def test_discrete_moments_smoother_matches_kf_smoother_on_lti():
    """Moments-EKF smoothing is exact on linear-Gaussian: matches the RTS smoother."""
    dynamics = _lti_dynamics()
    ll_kf, st_kf = compute_cuthbert_smoother(
        dynamics,
        KFSmootherConfig(filter_source="cuthbert"),
        obs_times=LTI_TIMES,
        obs_values=LTI_YS,
    )
    ll_m, st_m = compute_cuthbert_smoother(
        dynamics,
        EKFSmootherConfig(use_taylor=False),
        obs_times=LTI_TIMES,
        obs_values=LTI_YS,
    )
    assert jnp.allclose(ll_m, ll_kf, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(st_m.mean, st_kf.mean, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(_cov(st_m.chol_cov), _cov(st_kf.chol_cov), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Factorial (factorial_filter)
# ---------------------------------------------------------------------------
F = 5
D = 2  # (attack, defense)
OBS_TIMES = jnp.arange(8.0)
OBS_IDX = jnp.array([[0, 1], [2, 3], [0, 2], [1, 4], [3, 0], [2, 1], [0, 3], [1, 2]])
SCORES = jnp.array([[2, 1], [0, 0], [3, 1], [1, 2], [2, 2], [0, 3], [1, 1], [4, 0]])
K = OBS_TIMES.shape[0]


def _bp_model(alpha=0.2, beta=-1.5, tau=None, t0=0.0):
    tau = jnp.array([0.2, 0.2]) if tau is None else tau
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.3 * jnp.eye(D)),
        state_evolution=RandomWalkEvolution(tau=tau, factor_state_dim=D),
        observation_model=BivariatePoissonScoreObservation(
            alpha=alpha, beta=beta, factor_state_dim=D, max_goals=12
        ),
        num_factors=F,
        num_local_factors=2,
        t0=t0,
    )


def _gaussian_contrast_model():
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.5 * jnp.eye(1)),
        state_evolution=RandomWalkEvolution(tau=0.1, factor_state_dim=1),
        observation_model=GaussianObservation(
            h=lambda x, u, t: jnp.atleast_1d(x[0] - x[1]), R=jnp.eye(1)
        ),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )


def test_factorial_moments_matches_kalman_on_linear_gaussian_obs():
    """Moments-EKF is exact for a linear-Gaussian local observation: matches the KF."""
    score_y = jnp.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.4, 0.0, 0.3])[:, None]
    model = _gaussian_contrast_model()
    ll_kf, *_ = compute_factorial_filter(
        model,
        FactorialKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=score_y,
        obs_factor_indices=OBS_IDX,
    )
    ll_m, *_ = compute_factorial_filter(
        model,
        FactorialEKFConfig(use_taylor=False),
        obs_times=OBS_TIMES,
        obs_values=score_y,
        obs_factor_indices=OBS_IDX,
    )
    assert jnp.allclose(ll_m, ll_kf, rtol=1e-6, atol=1e-6), (ll_m, ll_kf)


def test_factorial_moments_loglik_decomposes_over_independent_matches():
    """No-repeat schedule => total loglik == sum of per-match logliks (exact)."""
    pairs = jnp.array([[0, 1], [2, 3]])
    times = jnp.array([1.0, 2.0])
    ys = SCORES[:2]

    def _model(num_factors):
        return FactorialDynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(D), 0.3 * jnp.eye(D)),
            state_evolution=RandomWalkEvolution(
                tau=jnp.array([0.2, 0.2]), factor_state_dim=D
            ),
            observation_model=BivariatePoissonScoreObservation(
                alpha=0.2, beta=-1.5, factor_state_dim=D, max_goals=12
            ),
            num_factors=num_factors,
            num_local_factors=2,
            t0=0.0,
        )

    total_ll, *_ = compute_factorial_filter(
        _model(4),
        FactorialEKFConfig(use_taylor=False),
        obs_times=times,
        obs_values=ys,
        obs_factor_indices=pairs,
    )
    per_match = 0.0
    for k in range(2):
        ll_k, *_ = compute_factorial_filter(
            _model(2),
            FactorialEKFConfig(use_taylor=False),
            obs_times=times[k : k + 1],
            obs_values=ys[k : k + 1],
            obs_factor_indices=jnp.array([[0, 1]]),
        )
        per_match = per_match + ll_k
    assert jnp.allclose(total_ll, per_match, atol=1e-6), (total_ll, per_match)


def test_factorial_moments_bp_loglik_close_to_pf():
    """Sanity: the moments-EKF marginal loglik is near the (consistent) PF estimate."""
    model = _bp_model()
    ll_m, *_ = compute_factorial_filter(
        model,
        FactorialEKFConfig(use_taylor=False),
        obs_times=OBS_TIMES,
        obs_values=SCORES,
        obs_factor_indices=OBS_IDX,
    )
    ll_pf, *_ = compute_factorial_filter(
        model,
        FactorialPFConfig(n_particles=_n_particles(4000)),
        jr.PRNGKey(0),
        obs_times=OBS_TIMES,
        obs_values=SCORES,
        obs_factor_indices=OBS_IDX,
    )
    # Generous tolerance: the Gaussian moment-matched update approximates the
    # discrete-count likelihood; we only pin "same ballpark" (the Taylor EKF is
    # off by >1 nat/match on this fixture).
    assert jnp.isfinite(ll_m)
    assert jnp.abs(ll_m - ll_pf) < 2.5, (ll_m, ll_pf)


def test_factorial_moments_grad_finite_taylor_grad_nan():
    """The headline capability: differentiable marginal loglik under moments.

    The Taylor flavour's observation Hessian is rank-deficient for the
    contrast-only score likelihood, so its gradient contains NaNs; if cuthbert
    ever fixes that, this test will flag that the docs caveat can be dropped.
    """

    def nll(theta, use_taylor):
        model = _bp_model(
            alpha=theta[0], beta=theta[1], tau=jnp.exp(theta[2]) * jnp.ones(2)
        )
        ll, *_ = compute_factorial_filter(
            model,
            FactorialEKFConfig(use_taylor=use_taylor),
            obs_times=OBS_TIMES,
            obs_values=SCORES,
            obs_factor_indices=OBS_IDX,
        )
        return -ll

    theta0 = jnp.array([0.2, -1.5, jnp.log(0.2)])
    grad_moments = jax.grad(lambda th: nll(th, False))(theta0)
    grad_taylor = jax.grad(lambda th: nll(th, True))(theta0)
    assert jnp.all(jnp.isfinite(grad_moments)), grad_moments
    assert not jnp.all(jnp.isfinite(grad_taylor)), grad_taylor


def test_factorial_use_taylor_false_requires_moments():
    """Categorical outcome model has no moments: TypeError; auto == Taylor."""
    model = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.5 * jnp.eye(1)),
        state_evolution=RandomWalkEvolution(tau=0.2, factor_state_dim=1),
        observation_model=MatchOutcomeObservation(draw_margin=0.3, factor_state_dim=1),
        num_factors=F,
        num_local_factors=2,
        t0=0.0,
    )
    outcomes = jnp.array([1, 2, 0, 1, 2, 0, 1, 1])

    with pytest.raises(TypeError, match="covariance"):
        compute_factorial_filter(
            model,
            FactorialEKFConfig(use_taylor=False),
            obs_times=OBS_TIMES,
            obs_values=outcomes,
            obs_factor_indices=OBS_IDX,
        )

    ll_auto, *_ = compute_factorial_filter(
        model,
        FactorialEKFConfig(),
        obs_times=OBS_TIMES,
        obs_values=outcomes,
        obs_factor_indices=OBS_IDX,
    )
    ll_taylor, *_ = compute_factorial_filter(
        model,
        FactorialEKFConfig(use_taylor=True),
        obs_times=OBS_TIMES,
        obs_values=outcomes,
        obs_factor_indices=OBS_IDX,
    )
    assert jnp.array_equal(ll_auto, ll_taylor)


def test_factorial_moments_svi_smoke():
    """SVI/MAP runs directly through the (deterministic) moments-EKF likelihood."""

    def svi_model():
        p = {
            "alpha": numpyro.sample("alpha", dist.Normal(0.2, 0.5)),
            "beta": numpyro.sample("beta", dist.Normal(-1.5, 0.5)),
            "log_tau": numpyro.sample("log_tau", dist.Normal(jnp.log(0.2), 0.5)),
        }
        # t0=None is required for dsx.sample under SVI/NUTS tracing.
        model = _bp_model(
            alpha=p["alpha"],
            beta=p["beta"],
            tau=jnp.exp(p["log_tau"]) * jnp.ones(2),
            t0=None,
        )
        dsx.sample(
            "sc",
            model,
            obs_times=OBS_TIMES,
            obs_values=SCORES,
            obs_factor_indices=OBS_IDX,
        )

    with Filter(filter_config=FactorialEKFConfig()):
        guide = AutoDelta(svi_model)
        svi = SVI(svi_model, guide, optax.adam(0.05), loss=Trace_ELBO())
        result = svi.run(jr.PRNGKey(0), 40, progress_bar=False)

    losses = jnp.asarray(result.losses)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.mean(losses[-10:]) < jnp.mean(losses[:10])


def test_factorial_moments_sites_and_smoother_smoke():
    """Site parity with the Taylor/PF branches, plus a smoother pass."""

    def filter_model():
        with Filter(
            filter_config=FactorialEKFConfig(
                use_taylor=False, record_filtered_states_mean=True
            )
        ):
            dsx.sample(
                "sc",
                _bp_model(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    pr = Predictive(filter_model, num_samples=1)(jr.PRNGKey(0))
    assert jnp.isfinite(pr["sc_marginal_loglik"][0])
    assert pr["sc_filtered_states_mean"].shape == (1, F, D)
    assert pr["sc_filtered_states_chol_cov"].shape == (1, F, D, D)
    assert pr["sc_filtered_local_states_mean"].shape == (1, K, 2, D)
    assert pr["sc_filtered_local_states_chol_cov"].shape == (1, K, 2, D, D)
    assert jnp.all(jnp.isfinite(pr["sc_filtered_local_states_chol_cov"][0]))

    def smoother_model():
        with Smoother(
            smoother_config=FactorialEKFSmootherConfig(record_smoothed_states_mean=True)
        ):
            dsx.sample(
                "sc",
                _bp_model(),
                obs_times=OBS_TIMES,
                obs_values=SCORES,
                obs_factor_indices=OBS_IDX,
            )

    spr = Predictive(smoother_model, num_samples=1)(jr.PRNGKey(0))
    assert jnp.isfinite(spr["sc_marginal_loglik"][0])
    assert spr["sc_smoothed_local_states_mean"].shape == (1, K, 2, D)
    assert jnp.all(jnp.isfinite(spr["sc_smoothed_local_states_mean"][0]))
