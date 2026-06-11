"""Cuthbert factorial filtering integration.

Implements filtering for :class:`~dynestyx.models.FactorialDynamicalModel` by
wrapping the ``cuthbert.factorial`` API. The state is partitioned into ``F``
conditionally independent factors; each observation (match) is *local*, touching
only ``n_local`` factors (e.g. the home and away teams). Inference uses cuthbert's
factorial extract/join/marginalize/insert machinery, but we run our **own scan**
(a thin variant of ``cuthbert.factorial.filter``) so that we can accumulate the
marginal log-likelihood :math:`\\log p(y_{1:K}) = \\sum_k \\log p(y_k \\mid y_{1:k-1})`
as a differentiable scalar for the NumPyro factor.

Marginal-likelihood strategy (verified against cuthbert source):
``filter_combine`` returns ``prior_nc + Δ_k`` where ``Δ_k`` is the per-match
predictive log-density. The factorializers leave the per-state normalizing-constant
scalar untouched, so after ``extract_and_join`` we reset the joined-local prior to a
clean unconditional prior (zeroing the associative-scan bookkeeping ``A``/``eta``/``Z``
and the normalizing constant ``ell``), run ``filter_combine``, and read ``Δ_k``
directly from the combined state.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cuthbert.factorial import gaussian as factorial_gaussian
from cuthbert.factorial import smc as factorial_smc
from cuthbert.gaussian import kalman, moments, taylor
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.smc import particle_filter
from cuthbertlib.kalman.filtering import FilterScanElement
from cuthbertlib.resampling import (
    multinomial,
    no_resampling,
    stop_gradient_decorator,
    systematic,
)

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    FactorialEKFConfig,
    FactorialKFConfig,
    FactorialPFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    _distribution_has_moments,
    _gaussian_moments_chol,
    _resolve_use_moments,
)
from dynestyx.inference.integrations.utils import covariance_from_cholesky
from dynestyx.models.factorial import FactorialDynamicalModel
from dynestyx.utils import _should_record_field


class FactorialCuthbertInputs(NamedTuple):
    """Per-match model inputs for the factorial filter (leading dim ``K``).

    Unlike the dense cuthbert adapter, there is no dummy initial step: the initial
    factorial state is constructed manually and the scan runs over all ``K`` real
    matches.
    """

    factor_indices: jax.Array  # (K, n_local) int — factors involved per match
    time: jax.Array  # (K,) — match time
    time_prev: jax.Array  # (K, n_local) — per-factor previous-match time
    y: jax.Array  # (K, ...) — match outcome
    u: jax.Array  # (K, control_dim) — per-match covariates passed to the observation


def _get_factorial_indices(mi: FactorialCuthbertInputs) -> jax.Array:
    return mi.factor_indices


# ---------------------------------------------------------------------------
# Schedule preprocessing
# ---------------------------------------------------------------------------
def _compute_time_prev_and_last(
    obs_times: jax.Array,
    obs_factor_indices: jax.Array,
    t0: jax.Array,
    num_factors: int,
):
    """Per-factor previous-match times (Duffield et al. §3.4.3 pairwise updates).

    For each match, each involved factor's previous time is the time it last
    played (or ``t0`` on first appearance). Implemented as a ``lax.scan`` so it is
    jit-safe; the gather happens before the scatter so two factors in the same
    match both read their pre-match times.

    Returns:
        ``(time_prev, last_time)`` of shapes ``(K, n_local)`` and ``(F,)``.
    """
    K = obs_factor_indices.shape[0]

    def body(last_time, k):
        inds = obs_factor_indices[k]
        tp = last_time[inds]
        last_time = last_time.at[inds].set(obs_times[k])
        return last_time, tp

    init_last = jnp.full((num_factors,), t0, dtype=obs_times.dtype)
    last_time, time_prev = jax.lax.scan(body, init_last, jnp.arange(K))
    return time_prev, last_time


def _build_factorial_inputs(
    dynamics: FactorialDynamicalModel,
    obs_times: jax.Array,
    obs_values: jax.Array,
    obs_factor_indices: jax.Array,
    obs_controls: jax.Array | None = None,
):
    obs_times = jnp.asarray(obs_times)
    obs_factor_indices = jnp.asarray(obs_factor_indices).astype(jnp.int32)
    K = obs_times.shape[0]
    if obs_controls is None:
        u = jnp.zeros((K, 0), dtype=obs_times.dtype)
    else:
        u = jnp.asarray(obs_controls, dtype=obs_times.dtype).reshape(K, -1)
    t0 = dynamics.t0 if dynamics.t0 is not None else obs_times[0]
    time_prev, last_time = _compute_time_prev_and_last(
        obs_times,
        obs_factor_indices,
        jnp.asarray(t0, dtype=obs_times.dtype),
        dynamics.num_factors,
    )
    mi = FactorialCuthbertInputs(
        factor_indices=obs_factor_indices,
        time=obs_times,
        time_prev=time_prev,
        y=jnp.asarray(obs_values),
        u=u,
    )
    return mi, last_time


# ---------------------------------------------------------------------------
# Initial factorial state construction
# ---------------------------------------------------------------------------
def _diag_chol(scale: jax.Array) -> jax.Array:
    """Build diagonal Cholesky factors from per-dimension scales.

    ``scale`` of shape ``(..., d)`` -> ``(..., d, d)``.
    """
    d = scale.shape[-1]
    eye = jnp.eye(d, dtype=scale.dtype)
    return scale[..., None] * eye


def _init_mean_chol(ic: dist.Distribution, num_factors: int, d: int):
    """Per-factor initial means ``(F, d)`` and Cholesky covariances ``(F, d, d)``."""
    mean = jnp.asarray(ic.mean)
    if mean.ndim == 1:
        mean = jnp.broadcast_to(mean, (num_factors, d))

    if isinstance(ic, dist.MultivariateNormal):
        chol = jnp.asarray(ic.scale_tril)
    elif isinstance(ic, dist.Independent) and isinstance(ic.base_dist, dist.Normal):
        chol = _diag_chol(jnp.asarray(ic.base_dist.scale))
    else:
        cov = jnp.asarray(ic.covariance_matrix)
        chol = jnp.linalg.cholesky(cov)

    if chol.ndim == 2:
        chol = jnp.broadcast_to(chol, (num_factors, d, d))
    return mean, chol


def _make_gaussian_init_state(
    means: jax.Array,
    chols: jax.Array,
    init_mi,
    *,
    linearized: bool,
):
    """Construct an init factorial Gaussian state with a scalar normalizing const.

    ``means`` ``(F, d)``, ``chols`` ``(F, d, d)``. The associative bookkeeping
    (``A``/``eta``/``Z``) is zeroed (an unconditional prior); ``ell`` is a scalar.
    """
    elem = FilterScanElement(
        A=jnp.zeros_like(chols),
        b=means,
        U=chols,
        eta=jnp.zeros_like(means),
        Z=jnp.zeros_like(chols),
        ell=jnp.asarray(0.0),
    )
    if linearized:
        return LinearizedKalmanFilterState(
            elem=elem, model_inputs=init_mi, mean_prev=means
        )
    return kalman.KalmanFilterState(elem=elem, model_inputs=init_mi)


def _reset_gaussian_prior(state):
    """Reset a joined-local Gaussian state to a clean unconditional prior.

    Keeps the prior mean (``elem.b``) and Cholesky covariance (``elem.U``); zeroes
    the associative-scan bookkeeping and the normalizing constant so the next
    ``filter_combine`` returns exactly ``Δ_k = log p(y_k | y_{1:k-1})``.
    """
    elem = state.elem
    new_elem = elem._replace(
        A=jnp.zeros_like(elem.A),
        eta=jnp.zeros_like(elem.eta),
        Z=jnp.zeros_like(elem.Z),
        ell=jnp.zeros_like(elem.ell),
    )
    return state._replace(elem=new_elem)


# ---------------------------------------------------------------------------
# Per-config cuthbert filter object builders (joint-local parameter functions)
# ---------------------------------------------------------------------------
def _factorial_taylor_filter(dynamics: FactorialDynamicalModel, filter_kwargs: dict):
    """Linearized (EKF) cuthbert filter acting on the joint-local state."""
    rtol = filter_kwargs.get("rtol", None)
    n_local = dynamics.num_local_factors
    d = dynamics.factor_state_dim

    def get_init_log_density(mi):
        # Unused by our scan (we build the init state manually) but required to
        # construct the cuthbert filter object.
        joint = n_local * d
        return (lambda x: jnp.asarray(0.0), jnp.zeros((joint,)))

    def get_dynamics_log_density(state: LinearizedKalmanFilterState, mi):
        time = mi.time
        time_prev = mi.time_prev  # (n_local,)

        def dynamics_log_density(x_prev, x):
            xp = jnp.reshape(x_prev, (n_local, d))
            xc = jnp.reshape(x, (n_local, d))

            def per_factor(j):
                td = dynamics.state_evolution(xp[j], None, time_prev[j], time)
                return jnp.asarray(td.log_prob(xc[j])).sum()

            return jnp.sum(jnp.stack([per_factor(j) for j in range(n_local)]))

        x_prev_lin = jnp.atleast_1d(jnp.asarray(state.mean))
        xp = jnp.reshape(x_prev_lin, (n_local, d))
        x_lin = jnp.concatenate(
            [
                jnp.atleast_1d(
                    jnp.asarray(
                        dynamics.state_evolution(xp[j], None, time_prev[j], time).mean
                    )
                )
                for j in range(n_local)
            ]
        )
        return dynamics_log_density, x_prev_lin, x_lin

    def get_observation_func(state: LinearizedKalmanFilterState, mi):
        u = mi.u if mi.u.shape[-1] > 0 else None

        def log_potential(x):
            edist = dynamics.observation_model(x, u, mi.time)
            return jnp.asarray(edist.log_prob(mi.y)).sum()

        return log_potential, jnp.atleast_1d(jnp.asarray(state.mean))

    return taylor.build_filter(
        get_init_log_density,  # type: ignore[arg-type]
        get_dynamics_log_density,  # type: ignore[arg-type]
        get_observation_func,  # type: ignore[arg-type]
        associative=False,
        rtol=rtol,
        # Local pairwise observations (e.g. a match outcome that depends only on
        # the skill *difference*) yield a rank-deficient observation Hessian; the
        # NaN-dim handling treats the unconstrained direction as unobserved.
        ignore_nan_dims=filter_kwargs.get("ignore_nan_dims", True),
    )


def _probe_factorial_moments(
    dynamics: FactorialDynamicalModel, model_inputs: FactorialCuthbertInputs
) -> tuple[bool, str]:
    """Structurally check whether the factorial conditionals expose exact moments."""
    d = dynamics.factor_state_dim
    joint = dynamics.num_local_factors * d
    u = model_inputs.u[0] if model_inputs.u.shape[-1] > 0 else None
    t0 = jnp.zeros(())
    try:
        d_dyn = dynamics.state_evolution(jnp.zeros((d,)), None, t0, t0 + 1.0)
        d_obs = dynamics.observation_model(jnp.zeros((joint,)), u, t0)
    except Exception:
        return False, "probing state_evolution/observation_model raised an exception"
    if not _distribution_has_moments(d_dyn):
        return False, (
            f"state_evolution returned {type(d_dyn).__name__} without a covariance "
            "('covariance_matrix' or 'scale_tril')"
        )
    if not _distribution_has_moments(d_obs):
        return False, (
            f"observation_model returned {type(d_obs).__name__} without a covariance "
            "('covariance_matrix' or 'scale_tril')"
        )
    return True, ""


def _factorial_moments_filter(dynamics: FactorialDynamicalModel, filter_kwargs: dict):
    """Moments-linearized (EKF) cuthbert filter on the joint-local state.

    Uses the conditional distributions' exact means and covariances
    (``cuthbert.gaussian.moments``); only the Jacobian of the mean is taken, so —
    unlike the Taylor flavour — the marginal log-likelihood stays differentiable
    for contrast-only local observations (e.g. the bivariate-Poisson scoreline).
    """
    n_local = dynamics.num_local_factors
    d = dynamics.factor_state_dim
    joint = n_local * d

    def get_init_params(mi):
        # Unused by our scan (we build the init state manually) but required to
        # construct the cuthbert filter object.
        return jnp.zeros((joint,)), jnp.eye(joint)

    def get_dynamics_params(state: LinearizedKalmanFilterState, mi):
        time = mi.time
        time_prev = mi.time_prev  # (n_local,)

        def mean_and_chol_cov(x):
            xp = jnp.reshape(x, (n_local, d))
            dists = [
                dynamics.state_evolution(xp[j], None, time_prev[j], time)
                for j in range(n_local)
            ]
            mean = jnp.concatenate(
                [jnp.atleast_1d(jnp.asarray(td.mean)) for td in dists]
            )
            chol = jax.scipy.linalg.block_diag(
                *[_gaussian_moments_chol(td) for td in dists]
            )
            return mean, chol

        return mean_and_chol_cov, jnp.atleast_1d(jnp.asarray(state.mean))

    def get_observation_params(state: LinearizedKalmanFilterState, mi):
        u = mi.u if mi.u.shape[-1] > 0 else None

        def mean_and_chol_cov(x):
            edist = dynamics.observation_model(x, u, mi.time)
            return (
                jnp.atleast_1d(jnp.asarray(edist.mean)),
                _gaussian_moments_chol(edist),
            )

        y = jnp.atleast_1d(jnp.asarray(mi.y, dtype=float))
        return mean_and_chol_cov, jnp.atleast_1d(jnp.asarray(state.mean)), y

    return moments.build_filter(
        get_init_params,  # type: ignore[arg-type]
        get_dynamics_params,  # type: ignore[arg-type]
        get_observation_params,  # type: ignore[arg-type]
        associative=False,
    )


def _factorial_kalman_filter(dynamics: FactorialDynamicalModel, filter_kwargs: dict):
    """Exact Kalman cuthbert filter on the joint-local state (linear-Gaussian only).

    Per-factor dynamics and the local observation are linearized via autodiff,
    which is *exact* for genuinely linear-Gaussian components and raises a clear
    error for non-Gaussian observations (which lack a covariance).
    """
    n_local = dynamics.num_local_factors
    d = dynamics.factor_state_dim
    joint = n_local * d

    def _block_diag(mats):
        return jax.scipy.linalg.block_diag(*mats)

    def get_init_params(mi):
        return jnp.zeros((joint,)), jnp.eye(joint)

    def get_dynamics_params(mi):
        time = mi.time
        time_prev = mi.time_prev
        Fs, cs, Qs = [], [], []
        x0 = jnp.zeros((d,))
        for j in range(n_local):

            def mean_fn(x, j=j):
                return dynamics.state_evolution(x, None, time_prev[j], time).mean

            td = dynamics.state_evolution(x0, None, time_prev[j], time)
            Fj = jax.jacobian(mean_fn)(x0)
            cj = jnp.asarray(td.mean) - Fj @ x0
            cholQj = jnp.linalg.cholesky(
                jnp.atleast_2d(jnp.asarray(td.covariance_matrix))
            )
            Fs.append(Fj)
            cs.append(cj)
            Qs.append(cholQj)
        return _block_diag(Fs), jnp.concatenate(cs), _block_diag(Qs)

    def get_observation_params(mi):
        u = mi.u if mi.u.shape[-1] > 0 else None
        x0 = jnp.zeros((joint,))
        od = dynamics.observation_model(x0, u, mi.time)
        if not hasattr(od, "covariance_matrix"):
            raise TypeError(
                "FactorialKFConfig requires a linear-Gaussian observation model "
                "(with a covariance), e.g. LinearGaussianObservation. For "
                "categorical match outcomes use FactorialEKFConfig or "
                "FactorialPFConfig."
            )

        def mean_fn(x):
            return dynamics.observation_model(x, u, mi.time).mean

        H = jax.jacobian(mean_fn)(x0)
        dvec = jnp.asarray(od.mean) - H @ x0
        chol_R = jnp.linalg.cholesky(jnp.atleast_2d(jnp.asarray(od.covariance_matrix)))
        y = jnp.atleast_1d(jnp.asarray(mi.y))
        return H, dvec, chol_R, y

    return kalman.build_filter(
        get_init_params,  # type: ignore[arg-type]
        get_dynamics_params,  # type: ignore[arg-type]
        get_observation_params,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Gaussian factorial scan (shared by EKF / KF)
# ---------------------------------------------------------------------------
def _run_gaussian_factorial_scan(filter_obj, factorializer, init_state, model_inputs):
    """Sequential factorial filter scan accumulating the marginal log-likelihood.

    Returns ``(marginal_loglik, final_factorial_state, local_filtered_seq)`` where
    ``local_filtered_seq`` stacks the per-match marginalized local factorial
    states (leading axes ``(K, n_local, ...)``).
    """

    def body(carry, mi):
        factorial_state, loglik = carry
        inds = jnp.asarray(_get_factorial_indices(mi))
        n_local = inds.shape[0]

        local_prior = factorializer.extract_and_join(factorial_state, mi)
        local_prior = _reset_gaussian_prior(local_prior)

        prep = filter_obj.filter_prepare(mi)
        filtered_joint = filter_obj.filter_combine(local_prior, prep)
        delta = filtered_joint.log_normalizing_constant

        local_filtered = factorializer.marginalize(filtered_joint, n_local)
        factorial_state = factorializer.insert(local_filtered, factorial_state, inds)
        return (factorial_state, loglik + delta), local_filtered

    (final_state, loglik), local_seq = jax.lax.scan(
        body, (init_state, jnp.asarray(0.0)), model_inputs
    )
    return loglik, final_state, local_seq


# ---------------------------------------------------------------------------
# Particle-filter factorial path
# ---------------------------------------------------------------------------
def _factorial_pf_objects(dynamics: FactorialDynamicalModel, filter_kwargs: dict):
    """Build the cuthbert factorial SMC filter object and factorializer."""
    n_factors = dynamics.num_factors

    def init_sample(key, mi):
        # Sample the full factorial initial state (F, d) for one particle.
        return jnp.asarray(dynamics.initial_condition.expand((n_factors,)).sample(key))

    def propagate_sample(key, x_prev, mi):
        # x_prev is the joint-local state (n_local * d,) for one particle.
        n_local = dynamics.num_local_factors
        d = dynamics.factor_state_dim
        xp = jnp.reshape(x_prev, (n_local, d))
        keys = jax.random.split(key, n_local)

        def per_factor(j):
            td = dynamics.state_evolution(xp[j], None, mi.time_prev[j], mi.time)
            return jnp.atleast_1d(jnp.asarray(td.sample(keys[j])))

        return jnp.concatenate([per_factor(j) for j in range(n_local)])

    def log_potential(x_prev, x, mi):
        u = mi.u if mi.u.shape[-1] > 0 else None
        edist = dynamics.observation_model(x, u, mi.time)
        return jnp.asarray(edist.log_prob(mi.y)).sum()

    base_method = filter_kwargs.get("resampling_base_method", "systematic")
    if base_method == "systematic":
        join_resampling = systematic.resampling
    elif base_method == "multinomial":
        join_resampling = multinomial.resampling
    else:
        raise ValueError(
            f"Unsupported factorial PF base resampling method: {base_method!r}."
        )
    if filter_kwargs.get("resampling_differential_method", "stop_gradient") == (
        "stop_gradient"
    ):
        join_resampling = stop_gradient_decorator(join_resampling)

    filter_obj = particle_filter.build_filter(
        init_sample=init_sample,  # type: ignore[arg-type]
        propagate_sample=propagate_sample,  # type: ignore[arg-type]
        log_potential=log_potential,  # type: ignore[arg-type]
        n_filter_particles=int(filter_kwargs.get("n_filter_particles", 1_000)),
        resampling_fn=no_resampling.resampling,  # resampling handled in join
    )
    factorializer = factorial_smc.build_factorializer(
        _get_factorial_indices, resampling_fn=join_resampling
    )
    return filter_obj, factorializer


def _run_pf_factorial_scan(filter_obj, factorializer, model_inputs, key):
    """Factorial particle-filter scan accumulating the marginal log-likelihood."""
    init_mi = jax.tree.map(lambda x: x[0], model_inputs)
    init_state = filter_obj.init_prepare(init_mi, key=key)
    init_state = factorializer.factorialize_init_state(init_state, init_mi)

    K = jax.tree.leaves(model_inputs)[0].shape[0]
    scan_keys = jax.random.split(key, K)

    def body(carry, mi_and_key):
        factorial_state, loglik = carry
        mi, k = mi_and_key
        inds = jnp.asarray(_get_factorial_indices(mi))
        n_local = inds.shape[0]

        factorial_state = factorial_state._replace(key=k)
        local_prior = factorializer.extract_and_join(factorial_state, mi)
        local_prior = local_prior._replace(
            log_normalizing_constant=jnp.zeros_like(
                local_prior.log_normalizing_constant
            )
        )
        prep = filter_obj.filter_prepare(mi, key=k)
        filtered_joint = filter_obj.filter_combine(local_prior, prep)
        delta = filtered_joint.log_normalizing_constant

        local_filtered = factorializer.marginalize(filtered_joint, n_local)
        factorial_state = factorializer.insert(local_filtered, factorial_state, inds)
        return (factorial_state, loglik + delta), local_filtered

    (final_state, loglik), local_seq = jax.lax.scan(
        body, (init_state, jnp.asarray(0.0)), (model_inputs, scan_keys)
    )
    return loglik, final_state, local_seq


# ---------------------------------------------------------------------------
# Public compute / run entry points
# ---------------------------------------------------------------------------
def compute_factorial_filter(
    dynamics: FactorialDynamicalModel,
    config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    obs_factor_indices: jax.Array,
    obs_controls: jax.Array | None = None,
):
    """Pure-JAX factorial filter (no NumPyro side-effects).

    Returns:
        ``(marginal_loglik, final_factorial_state, local_filtered_seq, last_time)``.
        ``last_time`` is each factor's most-recent observation time ``(F,)``.
    """
    filter_kwargs = dict(config.extra_filter_kwargs)
    if isinstance(config, FactorialPFConfig):
        filter_kwargs["n_filter_particles"] = config.n_particles
        filter_kwargs["resampling_base_method"] = config.resampling_method.base_method
        filter_kwargs["resampling_differential_method"] = (
            config.resampling_method.differential_method
        )

    model_inputs, last_time = _build_factorial_inputs(
        dynamics, obs_times, obs_values, obs_factor_indices, obs_controls
    )
    init_mi = jax.tree.map(lambda x: x[0], model_inputs)

    if isinstance(config, FactorialPFConfig):
        if key is None:
            raise ValueError(
                "Factorial particle filter requires a PRNG key: set 'crn_seed' in "
                "the config, or run inside a NumPyro seeded context."
            )
        filter_obj, factorializer = _factorial_pf_objects(dynamics, filter_kwargs)
        loglik, final_state, local_seq = _run_pf_factorial_scan(
            filter_obj, factorializer, model_inputs, key
        )
        return loglik, final_state, local_seq, last_time

    d = dynamics.factor_state_dim
    means, chols = _init_mean_chol(dynamics.initial_condition, dynamics.num_factors, d)
    factorializer = factorial_gaussian.build_factorializer(_get_factorial_indices)

    if isinstance(config, FactorialEKFConfig):
        available, why = _probe_factorial_moments(dynamics, model_inputs)
        if _resolve_use_moments(config.use_taylor, available, why=why):
            filter_obj = _factorial_moments_filter(dynamics, filter_kwargs)
        else:
            filter_obj = _factorial_taylor_filter(dynamics, filter_kwargs)
        init_state = _make_gaussian_init_state(means, chols, init_mi, linearized=True)
    elif isinstance(config, FactorialKFConfig):
        filter_obj = _factorial_kalman_filter(dynamics, filter_kwargs)
        init_state = _make_gaussian_init_state(means, chols, init_mi, linearized=False)
    else:
        raise ValueError(
            f"Unsupported factorial filter config: {type(config).__name__}. "
            "Expected FactorialEKFConfig, FactorialKFConfig, or FactorialPFConfig."
        )

    loglik, final_state, local_seq = _run_gaussian_factorial_scan(
        filter_obj, factorializer, init_state, model_inputs
    )
    return loglik, final_state, local_seq, last_time


def _factorial_rollout(
    name: str,
    dynamics: FactorialDynamicalModel,
    final_means: jax.Array,
    final_covs: jax.Array,
    last_time: jax.Array,
    predict_times: jax.Array,
    predict_factor_indices: jax.Array,
):
    """Predict per-future-match joint-local skill distributions and record sites.

    Each involved factor's final filtered distribution is propagated (EKF predict)
    from its last observation time to the future match time.
    """
    predict_times = jnp.asarray(predict_times)
    predict_factor_indices = jnp.asarray(predict_factor_indices).astype(jnp.int32)

    def predict_factor(mean, cov, t_prev, t_now):
        def mean_fn(x):
            return dynamics.state_evolution(x, None, t_prev, t_now).mean

        pred_mean = jnp.asarray(
            dynamics.state_evolution(mean, None, t_prev, t_now).mean
        )
        Jf = jax.jacobian(mean_fn)(mean)
        Q = jnp.atleast_2d(
            jnp.asarray(
                dynamics.state_evolution(mean, None, t_prev, t_now).covariance_matrix
            )
        )
        pred_cov = Jf @ cov @ Jf.T + Q
        return pred_mean, pred_cov

    def per_match(tp, inds):
        means = final_means[inds]  # (n_local, d)
        covs = final_covs[inds]  # (n_local, d, d)
        tprev = last_time[inds]  # (n_local,)
        pm, pc = jax.vmap(predict_factor, in_axes=(0, 0, 0, None))(
            means, covs, tprev, tp
        )
        return pm, pc

    pred_means, pred_covs = jax.vmap(per_match)(predict_times, predict_factor_indices)
    pred_chols = jnp.linalg.cholesky(pred_covs)

    numpyro.deterministic(f"{name}_predicted_times", predict_times)
    numpyro.deterministic(f"{name}_predicted_factor_indices", predict_factor_indices)
    numpyro.deterministic(f"{name}_predicted_states", pred_means)
    numpyro.deterministic(f"{name}_predicted_states_chol_cov", pred_chols)


def _add_sites_factorial(name: str, final_state, local_seq, record_kwargs: dict):
    """Record per-factor final filtered means/chols and per-match local states."""
    max_elems = record_kwargs["record_max_elems"]

    if isinstance(final_state, (kalman.KalmanFilterState, LinearizedKalmanFilterState)):
        final_means = final_state.elem.b  # (F, d)
        final_chols = final_state.elem.U  # (F, d, d)
        local_means = local_seq.elem.b  # (K, n_local, d)
        local_chols = local_seq.elem.U  # (K, n_local, d, d)
    else:  # particle filter state
        # Weighted per-factor means: final_state.particles (F, N, d)
        w = jax.nn.softmax(final_state.log_weights, axis=-1)  # (F, N)
        final_means = jnp.einsum("fn,fnd->fd", w, final_state.particles)
        centered = final_state.particles - final_means[:, None, :]
        cov = jnp.einsum("fn,fni,fnj->fij", w, centered, centered)
        final_chols = jnp.linalg.cholesky(cov + 1e-9 * jnp.eye(cov.shape[-1]))
        lw = jax.nn.softmax(local_seq.log_weights, axis=-1)  # (K, n_local, N)
        local_means = jnp.einsum("knp,knpd->knd", lw, local_seq.particles)
        # Particle-weighted local covariance, so the PF records the same local
        # chol-cov site as the EKF (used for per-match uncertainty bands / parity).
        local_centered = local_seq.particles - local_means[:, :, None, :]
        local_covs = jnp.einsum(
            "knp,knpi,knpj->knij", lw, local_centered, local_centered
        )
        d_local = local_covs.shape[-1]
        local_chols = jnp.linalg.cholesky(local_covs + 1e-9 * jnp.eye(d_local))

    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], final_means.shape, max_elems
    )
    add_chol = _should_record_field(
        record_kwargs["record_filtered_states_chol_cov"], final_chols.shape, max_elems
    )
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", final_means)
    if add_chol:
        numpyro.deterministic(f"{name}_filtered_states_chol_cov", final_chols)

    add_local = _should_record_field(
        record_kwargs["record_filtered_states_mean"], local_means.shape, max_elems
    )
    if add_local:
        numpyro.deterministic(f"{name}_filtered_local_states_mean", local_means)
        if local_chols is not None:
            numpyro.deterministic(f"{name}_filtered_local_states_chol_cov", local_chols)


def run_factorial_filter(
    name: str,
    dynamics: FactorialDynamicalModel,
    config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    obs_factor_indices: jax.Array,
    obs_controls: jax.Array | None = None,
    predict_times: jax.Array | None = None,
    predict_factor_indices: jax.Array | None = None,
    **kwargs,
) -> None:
    """Run the factorial filter and add NumPyro factor/deterministic sites."""
    if obs_factor_indices is None:
        raise ValueError(
            "FactorialDynamicalModel filtering requires 'obs_factor_indices' "
            "(an int array of shape (K, num_local_factors))."
        )

    loglik, final_state, local_seq, last_time = compute_factorial_filter(
        dynamics,
        config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_factor_indices=obs_factor_indices,
        obs_controls=obs_controls,
    )

    numpyro.factor(f"{name}_marginal_log_likelihood", loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", loglik)

    record_kwargs = _config_to_record_kwargs(config)
    _add_sites_factorial(name, final_state, local_seq, record_kwargs)

    if predict_times is not None:
        if predict_factor_indices is None:
            raise ValueError(
                "predict_times for a FactorialDynamicalModel requires "
                "'predict_factor_indices' of shape (P, num_local_factors)."
            )
        if isinstance(
            final_state, (kalman.KalmanFilterState, LinearizedKalmanFilterState)
        ):
            final_means = final_state.elem.b
            final_covs = covariance_from_cholesky(final_state.elem.U)
        else:
            w = jax.nn.softmax(final_state.log_weights, axis=-1)
            final_means = jnp.einsum("fn,fnd->fd", w, final_state.particles)
            centered = final_state.particles - final_means[:, None, :]
            final_covs = jnp.einsum("fn,fni,fnj->fij", w, centered, centered)
        _factorial_rollout(
            name,
            dynamics,
            final_means,
            final_covs,
            last_time,
            predict_times,
            predict_factor_indices,
        )

    return None
