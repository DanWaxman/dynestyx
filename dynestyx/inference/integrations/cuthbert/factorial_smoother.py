"""Cuthbert factorial smoothing integration.

Smoothing in a factorial state-space model is *embarrassingly parallel across
factors*: the local observations are fully absorbed during filtering, so each
factor's skill trajectory is smoothed independently with a single-factor backward
pass over that factor's own filtered sequence (Duffield et al. 2024, §3.3, §3.4).

We run a per-factor Rauch--Tung--Striebel (RTS / extended-RTS) recursion over each
factor's filtered Gaussians. The transition Jacobian is obtained by autodiff, so
this is exact for the (linear) random-walk dynamics and a standard EKS otherwise.
Unlike the EKF *filter*, the RTS backward pass never linearizes the local
observation potential, so it is well-behaved and differentiable.

The marginal log-likelihood reported as a NumPyro factor comes from the forward
filtering pass (smoothing does not re-touch the likelihood). The per-factor index
bookkeeping uses Python loops over concrete factor indices, so this runs as a
post-hoc pass (e.g. under ``numpyro.infer.Predictive`` on concrete data), not
inside a jitted parameter-inference loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    _init_mean_chol,
    compute_factorial_filter,
)
from dynestyx.inference.integrations.utils import covariance_from_cholesky
from dynestyx.inference.smoother_configs import (
    FactorialEKFSmootherConfig,
    FactorialKFSmootherConfig,
    FactorialPFSmootherConfig,
    _config_to_smoother_record_kwargs,
)
from dynestyx.models.factorial import FactorialDynamicalModel
from dynestyx.utils import _should_record_field


def _per_factor_rts(state_evolution, means, covs, times):
    """Per-factor extended RTS smoother over one factor's filtered Gaussians.

    Args:
        state_evolution: The per-factor :class:`DiscreteTimeStateEvolution`.
        means: Filtered means ``(L, d)``; index 0 is the prior at ``times[0]``.
        covs: Filtered covariances ``(L, d, d)``.
        times: The time of each filtered state ``(L,)``.

    Returns:
        Smoothed means ``(L, d)`` and covariances ``(L, d, d)``.
    """
    L = means.shape[0]

    def transition(m, P, t0, t1):
        def mean_fn(x):
            return state_evolution(x, None, t0, t1).mean

        td = state_evolution(m, None, t0, t1)
        F = jax.jacobian(mean_fn)(m)
        Q = jnp.atleast_2d(jnp.asarray(td.covariance_matrix))
        m_pred = jnp.asarray(td.mean)
        P_pred = F @ P @ F.T + Q
        return m_pred, P_pred, F

    def body(carry, k):
        ms_next, Ps_next = carry  # smoothed state at k+1
        m_k, P_k = means[k], covs[k]
        m_pred, P_pred, F = transition(m_k, P_k, times[k], times[k + 1])
        # Smoothing gain G = P_k F^T P_pred^{-1}; solve for numerical stability.
        gain = jnp.linalg.solve(P_pred, F @ P_k).T
        m_s = m_k + gain @ (ms_next - m_pred)
        P_s = P_k + gain @ (Ps_next - P_pred) @ gain.T
        return (m_s, P_s), (m_s, P_s)

    if L == 1:
        return means, covs

    ks = jnp.arange(L - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(body, (means[L - 1], covs[L - 1]), ks)
    smoothed_means = jnp.concatenate([ms_rev[::-1], means[L - 1][None]], axis=0)
    smoothed_covs = jnp.concatenate([Ps_rev[::-1], covs[L - 1][None]], axis=0)
    return smoothed_means, smoothed_covs


def compute_factorial_smoother(
    dynamics: FactorialDynamicalModel,
    config,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    obs_factor_indices: jax.Array,
):
    """Pure-JAX factorial smoother (no NumPyro side-effects).

    Returns ``(marginal_loglik, smoothed_local_means, smoothed_local_chols)`` where
    the smoothed local states have shape ``(K, n_local, d)`` / ``(K, n_local, d, d)``,
    aligned with the matches (column ``j`` is the ``j``-th factor of match ``k``).
    """
    if isinstance(config, FactorialPFSmootherConfig):
        raise NotImplementedError(
            "Factorial particle smoothing is not implemented. Use "
            "FactorialEKFSmootherConfig or FactorialKFSmootherConfig (per-factor "
            "RTS smoothing), or run the particle filter without smoothing."
        )
    if not isinstance(config, (FactorialEKFSmootherConfig, FactorialKFSmootherConfig)):
        raise ValueError(
            f"Unsupported factorial smoother config: {type(config).__name__}."
        )

    loglik, _final_state, local_seq, _last_time = compute_factorial_filter(
        dynamics,
        config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_factor_indices=obs_factor_indices,
    )

    d = dynamics.factor_state_dim
    means_local = local_seq.elem.b  # (K, n_local, d)
    covs_local = covariance_from_cholesky(local_seq.elem.U)  # (K, n_local, d, d)

    init_means, init_chols = _init_mean_chol(
        dynamics.initial_condition, dynamics.num_factors, d
    )
    init_covs = covariance_from_cholesky(init_chols)

    obs_times = jnp.asarray(obs_times)
    t0 = dynamics.t0 if dynamics.t0 is not None else obs_times[0]
    t0 = jnp.asarray(t0, dtype=obs_times.dtype)

    idx_np = np.asarray(obs_factor_indices)
    K, n_local = idx_np.shape

    # Precompute, per factor, the (match, position) cells it occupies in time order.
    positions_by_factor: list[list[tuple[int, int]]] = [
        [] for _ in range(dynamics.num_factors)
    ]
    for k in range(K):
        for j in range(n_local):
            positions_by_factor[int(idx_np[k, j])].append((k, j))

    smoothed_local_means = jnp.zeros((K, n_local, d))
    smoothed_local_covs = jnp.zeros((K, n_local, d, d))

    for i, positions in enumerate(positions_by_factor):
        if not positions:
            continue
        ks = jnp.asarray([k for (k, _) in positions])
        js = jnp.asarray([j for (_, j) in positions])

        m_seq = jnp.concatenate([init_means[i][None], means_local[ks, js]], axis=0)
        P_seq = jnp.concatenate([init_covs[i][None], covs_local[ks, js]], axis=0)
        t_seq = jnp.concatenate([t0[None], obs_times[ks]], axis=0)

        sm, sP = _per_factor_rts(dynamics.state_evolution, m_seq, P_seq, t_seq)
        # sm[0] is the smoothed prior; sm[1:] are smoothed at the factor's matches.
        # Scatter all of this factor's matches at once (one update per factor).
        smoothed_local_means = smoothed_local_means.at[ks, js].set(sm[1:])
        smoothed_local_covs = smoothed_local_covs.at[ks, js].set(sP[1:])

    smoothed_local_chols = jnp.linalg.cholesky(smoothed_local_covs + 1e-12 * jnp.eye(d))
    return loglik, smoothed_local_means, smoothed_local_chols


def run_factorial_smoother(
    name: str,
    dynamics: FactorialDynamicalModel,
    config,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    obs_factor_indices: jax.Array,
    **kwargs,
) -> None:
    """Run the factorial smoother and add NumPyro factor/deterministic sites."""
    if obs_factor_indices is None:
        raise ValueError(
            "FactorialDynamicalModel smoothing requires 'obs_factor_indices'."
        )

    loglik, smoothed_means, smoothed_chols = compute_factorial_smoother(
        dynamics,
        config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_factor_indices=obs_factor_indices,
    )

    numpyro.factor(f"{name}_marginal_log_likelihood", loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", loglik)

    record_kwargs = _config_to_smoother_record_kwargs(config)
    max_elems = record_kwargs["record_max_elems"]
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], smoothed_means.shape, max_elems
    )
    add_chol = _should_record_field(
        record_kwargs["record_smoothed_states_chol_cov"],
        smoothed_chols.shape,
        max_elems,
    )
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_local_states_mean", smoothed_means)
    if add_chol:
        numpyro.deterministic(f"{name}_smoothed_local_states_chol_cov", smoothed_chols)

    return None


__all__ = ["compute_factorial_smoother", "run_factorial_smoother"]
