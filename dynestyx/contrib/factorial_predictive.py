"""Held-out one-step-ahead predictive evaluation for factorial models.

The standard benchmark metric for online skill rating (Duffield et al., 2024) is the
average *one-step-ahead* predictive negative log-likelihood (NLL) of match results on a
held-out period: for each match :math:`k`, predict :math:`p(y_k \\mid y_{1:k-1})` from the
two involved factors' filtered marginals *as of their previous appearance* (propagated to
the match time), then score the observed result.

dynestyx's factorial filter computes this predictive internally (it is exactly the
per-match marginal-likelihood increment), but does not currently surface it per match. This
module reconstructs it from the public outputs of
:func:`~dynestyx.inference.integrations.cuthbert.factorial_filter.compute_factorial_filter`:
the per-match marginalized local states. The reconstruction reproduces the filter's own
one-step-ahead predictive; only the final integration over the propagated 2-factor Gaussian
differs (Monte Carlo here, via :func:`factorial_outcome_probabilities` /
:func:`factorial_score_probabilities`, vs the filter's internal Taylor expansion), so
results match published benchmarks closely rather than bit-for-bit.

This lives in :mod:`dynestyx.contrib` (reusable but not yet part of the stable core API).
"""

import jax
import jax.numpy as jnp
import numpy as np

from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    _init_mean_chol,
    compute_factorial_filter,
)
from dynestyx.models.factorial import (
    FactorialDynamicalModel,
    factorial_outcome_probabilities,
    factorial_score_probabilities,
)


def _local_means_covs(local_seq):
    """Per-match marginalized local means/covs from a factorial filter's local_seq.

    Returns ``(means, covs)`` of shapes ``(K, n_local, d)`` and ``(K, n_local, d, d)``,
    handling both the Gaussian (EKF/KF) and particle-filter local-state containers.
    """
    if hasattr(local_seq, "elem"):  # Gaussian (Linearized)KalmanFilterState
        means = jnp.asarray(local_seq.elem.b)
        chols = jnp.asarray(local_seq.elem.U)
        covs = jnp.einsum("knij,knlj->knil", chols, chols)
        return means, covs
    if hasattr(local_seq, "particles"):  # particle filter
        w = jax.nn.softmax(local_seq.log_weights, axis=-1)  # (K, n_local, N)
        particles = local_seq.particles  # (K, n_local, N, d)
        means = jnp.einsum("knp,knpd->knd", w, particles)
        centered = particles - means[:, :, None, :]
        covs = jnp.einsum("knp,knpi,knpj->knij", w, centered, centered)
        d = covs.shape[-1]
        covs = covs + 1e-9 * jnp.eye(d)
        return means, covs
    raise TypeError(f"Unrecognized local-state container: {type(local_seq).__name__}")


def one_step_ahead_local_states(
    model: FactorialDynamicalModel,
    config,
    *,
    obs_times,
    obs_values,
    obs_factor_indices,
    key=None,
):
    """One-step-ahead predicted joint-local distributions for every match.

    Runs the factorial filter once, then for each match reconstructs the predictive
    (pre-update) distribution of the two involved factors: each factor's filtered marginal
    *as of its previous match* (the prior for its first appearance), propagated to the
    current match time by the model's own dynamics (EKF predict). This is the distribution
    the filter integrates to form ``p(y_k | y_{1:k-1})``.

    Args:
        model: The factorial model.
        config: A factorial filter config (``FactorialEKFConfig`` / ``FactorialKFConfig`` /
            ``FactorialPFConfig``).
        obs_times: Match times ``(K,)`` (in the model's time units).
        obs_values: Match observations ``(K, ...)``.
        obs_factor_indices: Involved factor indices per match ``(K, n_local)``.
        key: PRNG key (required for a particle-filter config).

    Returns:
        ``(pred_means, pred_chol_covs)`` of shapes ``(K, n_local, d)`` and
        ``(K, n_local, d, d)`` -- the per-match predicted joint-local means and Cholesky
        covariances, ready for :func:`factorial_outcome_probabilities` /
        :func:`factorial_score_probabilities`.
    """
    obs_times = jnp.asarray(obs_times)
    obs_factor_indices = jnp.asarray(obs_factor_indices).astype(int)
    _, _, local_seq, _ = compute_factorial_filter(
        model,
        config,
        key,
        obs_times=obs_times,
        obs_values=jnp.asarray(obs_values),
        obs_factor_indices=obs_factor_indices,
    )
    local_means, local_covs = _local_means_covs(local_seq)

    F = int(model.num_factors)
    d = int(model.factor_state_dim)
    n_local = int(model.num_local_factors)
    K = int(obs_times.shape[0])

    prior_means, prior_chols = _init_mean_chol(model.initial_condition, F, d)
    prior_means = np.asarray(prior_means)
    prior_covs = np.asarray(jnp.einsum("fij,fkj->fik", prior_chols, prior_chols))
    t0 = float(model.t0) if model.t0 is not None else float(obs_times[0])

    lm = np.asarray(local_means)
    lc = np.asarray(local_covs)
    idx = np.asarray(obs_factor_indices)
    times = np.asarray(obs_times, dtype=float)

    # Walk matches in order, tracking each factor's most-recent filtered state/time.
    last_mean = prior_means.copy()  # (F, d)
    last_cov = prior_covs.copy()  # (F, d, d)
    last_time = np.full(F, t0)
    prev_means = np.zeros((K, n_local, d))
    prev_covs = np.zeros((K, n_local, d, d))
    prev_times = np.zeros((K, n_local))
    for k in range(K):
        for ll in range(n_local):
            p = int(idx[k, ll])
            prev_means[k, ll] = last_mean[p]
            prev_covs[k, ll] = last_cov[p]
            prev_times[k, ll] = last_time[p]
        for ll in range(n_local):
            p = int(idx[k, ll])
            last_mean[p] = lm[k, ll]
            last_cov[p] = lc[k, ll]
            last_time[p] = times[k]

    # EKF-predict each previous filtered state to its match time (vectorized).
    def predict_one(mean, cov, t_prev, t_now):
        def mean_fn(x):
            return jnp.asarray(model.state_evolution(x, None, t_prev, t_now).mean)

        pred_mean = mean_fn(mean)
        jac = jax.jacobian(mean_fn)(mean)
        q = jnp.atleast_2d(
            jnp.asarray(
                model.state_evolution(mean, None, t_prev, t_now).covariance_matrix
            )
        )
        pred_cov = jac @ cov @ jac.T + q
        return pred_mean, pred_cov

    flat_predict = jax.vmap(predict_one)
    target_times = jnp.broadcast_to(jnp.asarray(times)[:, None], (K, n_local)).reshape(
        -1
    )
    pm, pc = flat_predict(
        jnp.asarray(prev_means).reshape(K * n_local, d),
        jnp.asarray(prev_covs).reshape(K * n_local, d, d),
        jnp.asarray(prev_times).reshape(-1),
        target_times,
    )
    pred_means = pm.reshape(K, n_local, d)
    pred_covs = pc.reshape(K, n_local, d, d)
    pred_chols = jnp.linalg.cholesky(pred_covs + 1e-9 * jnp.eye(d))
    return pred_means, pred_chols


def _outcome_predictive_ghq(
    observation_model, pred_means, pred_chols, n_local, d, degree=24
):
    """Exact (deterministic) outcome predictive via 1-D Gauss-Hermite quadrature.

    For a scalar (``d == 1``) pairwise categorical model whose outcome probabilities
    depend on the skill difference ``δ = x_home[0] - x_away[0]``, the one-step-ahead
    predictive ``E_{δ~N(m, v)}[probs(δ)]`` is a 1-D integral, computed exactly (to
    quadrature) -- matching the deterministic predictive used in online-rating benchmarks
    and avoiding Monte-Carlo noise. ``m = m_h - m_a``, ``v = v_h + v_a``.
    """
    m = pred_means[:, 0, 0] - pred_means[:, 1, 0]
    v = pred_chols[:, 0, 0, 0] ** 2 + pred_chols[:, 1, 0, 0] ** 2
    nodes, weights = np.polynomial.hermite.hermgauss(int(degree))
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights) / jnp.sqrt(jnp.pi)
    deltas = m[:, None] + jnp.sqrt(2.0 * v)[:, None] * nodes[None, :]  # (K, degree)

    def probs_at(delta):
        x = jnp.zeros((n_local * d,)).at[0].set(delta)
        return jnp.asarray(observation_model(x, None, jnp.asarray(0.0)).probs)

    probs = jax.vmap(jax.vmap(probs_at))(deltas)  # (K, degree, n_out)
    return jnp.sum(weights[None, :, None] * probs, axis=1)


def outcome_predictive_probs(
    model: FactorialDynamicalModel,
    config,
    *,
    obs_times,
    obs_values,
    obs_factor_indices,
    key=None,
    method: str = "ghq",
    n_samples: int = 2000,
    ghq_degree: int = 24,
):
    """Per-match one-step-ahead predictive outcome probabilities ``(K, n_outcomes)``.

    For a categorical (e.g. win/draw/loss) observation model. Columns follow the
    observation model's own ``.probs`` order (for
    :class:`~dynestyx.models.MatchOutcomeObservation` that is ``[draw, home, away]``).

    ``method="ghq"`` (default) integrates the predictive deterministically via Gauss-Hermite
    quadrature over the skill difference (exact for a scalar ``d == 1`` model); ``method="mc"``
    Monte-Carlo samples the joint skills via :func:`factorial_outcome_probabilities` (works
    for any dimension). ``ghq`` falls back to ``mc`` if ``d != 1``.
    """
    pred_means, pred_chols = one_step_ahead_local_states(
        model,
        config,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_factor_indices=obs_factor_indices,
        key=key,
    )
    d = int(model.factor_state_dim)
    n_local = int(model.num_local_factors)
    if method == "ghq" and d == 1:
        return _outcome_predictive_ghq(
            model.observation_model, pred_means, pred_chols, n_local, d, ghq_degree
        )
    mc_key = jax.random.PRNGKey(0) if key is None else key
    return factorial_outcome_probabilities(
        model.observation_model, pred_means, pred_chols, key=mc_key, n_samples=n_samples
    )


def score_predictive(
    model: FactorialDynamicalModel,
    config,
    *,
    obs_times,
    obs_values,
    obs_factor_indices,
    key=None,
    n_samples: int = 2000,
    max_goals_grid: int = 9,
):
    """One-step-ahead scoreline predictive for a bivariate-Poisson model.

    Returns a dict with ``score_grids`` ``(K, G+1, G+1)``, ``wdl`` ``(K, 3)`` reordered to
    ``[draw, home, away]`` (to match outcome labels), ``score_logprob`` ``(K,)`` (log
    predictive probability of the observed scoreline), and ``outcomes`` ``(K,)`` (the
    observed win/draw/loss label derived from the score, ``0=draw, 1=home, 2=away``).
    """
    pred_means, pred_chols = one_step_ahead_local_states(
        model,
        config,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_factor_indices=obs_factor_indices,
        key=key,
    )
    mc_key = jax.random.PRNGKey(0) if key is None else key
    grids, wdl_hda = factorial_score_probabilities(
        model.observation_model,
        pred_means,
        pred_chols,
        key=mc_key,
        n_samples=n_samples,
        max_goals_grid=max_goals_grid,
    )
    # factorial_score_probabilities returns [home, draw, away]; reorder to [draw, home, away].
    wdl = jnp.stack([wdl_hda[:, 1], wdl_hda[:, 0], wdl_hda[:, 2]], axis=-1)

    scores = np.asarray(obs_values).astype(int)
    home_g = np.clip(scores[:, 0], 0, max_goals_grid)
    away_g = np.clip(scores[:, 1], 0, max_goals_grid)
    grids_np = np.asarray(grids)
    score_prob = grids_np[np.arange(len(scores)), home_g, away_g]
    score_logprob = jnp.log(jnp.clip(jnp.asarray(score_prob), 1e-12, None))
    outcomes = jnp.asarray(
        np.where(
            scores[:, 0] > scores[:, 1], 1, np.where(scores[:, 0] < scores[:, 1], 2, 0)
        )
    )
    return {
        "score_grids": grids,
        "wdl": wdl,
        "score_logprob": score_logprob,
        "outcomes": outcomes,
    }


def split_nll(predict_probs, results, split_index=None):
    """Average NLL ``-mean log p(result)`` overall and (optionally) by train/test split.

    Args:
        predict_probs: Per-match predictive probabilities ``(K, n_outcomes)``.
        results: Observed result labels ``(K,)`` indexing into ``predict_probs``.
        split_index: If given, matches ``[:split_index]`` are train and ``[split_index:]``
            are test.

    Returns:
        Dict with ``"nll"`` and, if ``split_index`` is given, ``"train_nll"`` / ``"test_nll"``.
    """
    predict_probs = jnp.asarray(predict_probs)
    results = jnp.asarray(results).astype(int)
    lp = jnp.log(
        jnp.clip(predict_probs[jnp.arange(results.shape[0]), results], 1e-12, None)
    )
    out = {"nll": float(-jnp.mean(lp))}
    if split_index is not None:
        s = int(split_index)
        out["train_nll"] = float(-jnp.mean(lp[:s]))
        out["test_nll"] = float(-jnp.mean(lp[s:]))
    return out
