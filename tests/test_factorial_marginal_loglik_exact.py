"""Validate the factor-marginalized marginal likelihood against an *exact joint* filter.

The factorial filter computes an **approximate** marginal likelihood: after each
pairwise update it projects the joint belief back onto a product of per-factor
marginals, discarding the induced cross-covariances. For a **linear-Gaussian**
model with pairwise *contrast* observations (``y = z_i - z_j + noise``) this
projection is provably:

* **exact** when the match graph is *loop-free* (a forest) -- no observation ever
  couples two factors that are already correlated through a shared history; and
* an **approximation** when the match graph has *cycles* (including a repeated
  pairing, which is a 2-cycle).

We pin both regimes by comparing the dynestyx factor-marginalized marginal
log-likelihood against two **independently-computed exact joint** marginal
log-likelihoods (a sequential full-joint Kalman filter and a one-shot joint
Gaussian log-density), on fixed data so the test is deterministic.

The pairwise contrast observation has a rank-deficient observation Hessian (only
the *difference* is observed), so the EKF/Taylor marginal likelihood is *biased*
relative to the exact value; this is characterized and pinned here too (and is
why parameter inference should use ``FactorialKFConfig`` / ``FactorialPFConfig``,
not ``FactorialEKFConfig``, for the likelihood value).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jaxtyping import Array, Float

from dynestyx import (
    FactorialDynamicalModel,
    FactorialEKFConfig,
    FactorialKFConfig,
    RandomWalkEvolution,
)
from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    compute_factorial_filter,
)
from dynestyx.models.core import ObservationModel

jax.config.update("jax_enable_x64", True)


class _LinearGaussianLocalObs(ObservationModel):
    """y = H_local @ x_joint + eps, eps ~ N(0, R). x_joint = [factor_i ; factor_j]."""

    H_local: Float[Array, "obs joint"]
    R: Float[Array, "obs obs"]
    num_local_factors: int = eqx.field(static=True)
    factor_state_dim: int = eqx.field(static=True)

    def __init__(self, H_local, R, factor_state_dim):
        self.H_local = jnp.asarray(H_local, dtype=float)
        self.R = jnp.asarray(R, dtype=float)
        self.num_local_factors = 2
        self.factor_state_dim = int(factor_state_dim)

    def __call__(self, x, u, t):
        return dist.MultivariateNormal(
            loc=self.H_local @ jnp.asarray(x), covariance_matrix=self.R
        )


# ---------------------------------------------------------------------------
# Independent exact ground truths (plain numpy, no dynestyx / cuthbert)
# ---------------------------------------------------------------------------
def _seq_joint_kf_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y):
    """Exact marginal loglik via a sequential *full-joint* Kalman filter."""
    n = F * d
    obs_dim = Hl.shape[0]
    m = np.tile(mu0, F)
    P = np.zeros((n, n))
    for f in range(F):
        P[f * d : (f + 1) * d, f * d : (f + 1) * d] = S0
    qdiag = np.tile(tau2, F)
    ll, t_prev = 0.0, float(t0)
    for k in range(len(times)):
        P = P + np.diag(qdiag * (float(times[k]) - t_prev))
        t_prev = float(times[k])
        i, j = int(idx[k][0]), int(idx[k][1])
        Hk = np.zeros((obs_dim, n))
        Hk[:, i * d : (i + 1) * d] = Hl[:, 0:d]
        Hk[:, j * d : (j + 1) * d] = Hl[:, d : 2 * d]
        S = Hk @ P @ Hk.T + R
        innov = np.asarray(y[k], float) - Hk @ m
        _, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (
            obs_dim * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov)
        )
        K = P @ Hk.T @ np.linalg.inv(S)
        m = m + K @ innov
        P = P - K @ Hk @ P
    return float(ll)


def _oneshot_joint_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y):
    """Exact marginal loglik via a one-shot joint Gaussian over all observations."""
    n = F * d
    obs_dim = Hl.shape[0]
    K = len(times)
    Hg = np.zeros((K, obs_dim, n))
    for k in range(K):
        i, j = int(idx[k][0]), int(idx[k][1])
        Hg[k, :, i * d : (i + 1) * d] = Hl[:, 0:d]
        Hg[k, :, j * d : (j + 1) * d] = Hl[:, d : 2 * d]
    full_mu = np.tile(mu0, F)
    mY = np.concatenate([Hg[k] @ full_mu for k in range(K)])
    SigY = np.zeros((K * obs_dim, K * obs_dim))
    for k in range(K):
        for kp in range(K):
            s = min(float(times[k]), float(times[kp]))
            blk = S0 + np.diag(tau2 * (s - float(t0)))  # per-factor block
            Cz = np.kron(np.eye(F), blk)
            block = Hg[k] @ Cz @ Hg[kp].T + (R if k == kp else 0.0)
            SigY[k * obs_dim : (k + 1) * obs_dim, kp * obs_dim : (kp + 1) * obs_dim] = (
                block
            )
    diff = np.asarray(y, float).reshape(K * obs_dim) - mY
    _, logdet = np.linalg.slogdet(SigY)
    return float(
        -0.5
        * (
            K * obs_dim * np.log(2 * np.pi)
            + logdet
            + diff @ np.linalg.solve(SigY, diff)
        )
    )


def _factorial_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y, config):
    model = FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.asarray(mu0, float), jnp.asarray(S0, float)
        ),
        state_evolution=RandomWalkEvolution(
            tau=jnp.sqrt(jnp.asarray(tau2, float)), factor_state_dim=d
        ),
        observation_model=_LinearGaussianLocalObs(Hl, R, factor_state_dim=d),
        num_factors=F,
        num_local_factors=2,
        t0=float(t0),
    )
    ll, *_ = compute_factorial_filter(
        model,
        config,
        None,
        obs_times=jnp.asarray(times, float),
        obs_values=jnp.asarray(y, float),
        obs_factor_indices=jnp.asarray(idx, jnp.int32),
    )
    return float(ll)


# Scalar (d=1) contrast model: y = z_i - z_j + noise.
_MU0_1, _S0_1, _TAU2_1, _R1 = (
    np.array([0.0]),
    np.array([[0.4]]),
    np.array([0.25]),
    np.array([[0.6]]),
)
_H1 = np.array([[1.0, -1.0]])

# Vector (d=2, attack/defense) contrast model: y = [att_i - def_j, att_j - def_i].
_MU0_2, _S0_2 = np.array([0.0, 0.0]), np.array([[0.4, 0.05], [0.05, 0.3]])
_TAU2_2, _R2 = np.array([0.25, 0.16]), np.array([[0.6, 0.0], [0.0, 0.5]])
_H2 = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, -1.0, 1.0, 0.0]])


def _times(K, t0=0.0, step=0.7):
    return t0 + step * np.arange(1, K + 1)


# Loop-free (forest) match graphs -> factor-marginalized filter is EXACT.
_EXACT_CASES = {
    "matching_d1": (4, 1, [[0, 1], [2, 3]], [[1.5], [-0.5]]),
    "hub_fresh_d1": (
        5,
        1,
        [[0, 1], [0, 2], [0, 3], [0, 4]],
        [[1.0], [-1.0], [0.5], [2.0]],
    ),
    "chain_d1": (4, 1, [[0, 1], [1, 2], [2, 3]], [[0.7], [-1.3], [0.4]]),
    "hub_fresh_d2": (
        4,
        2,
        [[0, 1], [0, 2], [0, 3]],
        [[1.0, -0.5], [0.3, 1.2], [-0.8, 0.6]],
    ),
}

# Cyclic match graphs -> factor-marginalized filter is an APPROXIMATION.
_CYCLIC_CASES = {
    "triangle_d1": (3, 1, [[0, 1], [1, 2], [0, 2]], [[1.0], [-0.5], [0.8]]),
    "round_robin4_d1": (
        4,
        1,
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        [[1.0], [-0.5], [0.8], [0.2], [-1.1], [0.6]],
    ),
}


def _params(d):
    if d == 1:
        return _MU0_1, _S0_1, _TAU2_1, _H1, _R1
    return _MU0_2, _S0_2, _TAU2_2, _H2, _R2


@pytest.mark.parametrize("case", list(_EXACT_CASES))
def test_factorial_kf_marginal_loglik_exact_on_loop_free_graphs(case):
    """On loop-free graphs the factor-marginalized KF loglik == the exact joint."""
    F, d, idx, y = _EXACT_CASES[case]
    mu0, S0, tau2, Hl, R = _params(d)
    t0, times = 0.0, _times(len(idx))

    r2 = _seq_joint_kf_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y)
    r3 = _oneshot_joint_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y)
    r1 = _factorial_loglik(
        F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y, FactorialKFConfig()
    )

    # The two independent exact ground truths must agree...
    assert abs(r2 - r3) < 1e-8, f"ground-truth mismatch: {r2} vs {r3}"
    # ...and the factor-marginalized KF must match them to machine precision.
    assert abs(r1 - r3) < 1e-7, f"{case}: factorial {r1} != exact joint {r3}"


@pytest.mark.parametrize("case", list(_CYCLIC_CASES))
def test_factorial_kf_marginal_loglik_approximate_on_cyclic_graphs(case):
    """On cyclic graphs the factored loglik is a (close, bounded) approximation."""
    F, d, idx, y = _CYCLIC_CASES[case]
    mu0, S0, tau2, Hl, R = _params(d)
    t0, times = 0.0, _times(len(idx))

    r2 = _seq_joint_kf_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y)
    r3 = _oneshot_joint_loglik(F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y)
    r1 = _factorial_loglik(
        F, d, mu0, S0, tau2, t0, times, idx, Hl, R, y, FactorialKFConfig()
    )

    assert abs(r2 - r3) < 1e-8, f"ground-truth mismatch: {r2} vs {r3}"
    gap = abs(r1 - r3)
    # It is genuinely an approximation here (not exact)...
    assert gap > 1e-4, f"{case}: expected an approximation gap, got {gap:.2e}"
    # ...but a tight one (mean-field projection error stays well below 1 nat here).
    assert gap < 1.0, f"{case}: approximation gap {gap:.4f} unexpectedly large"


def test_factorial_kf_equals_oneshot_loglik_chain():
    """Spot-check an exact value so a sign/scale regression is caught numerically."""
    F, d, idx, y = _EXACT_CASES["chain_d1"]
    mu0, S0, tau2, Hl, R = _params(d)
    times = _times(len(idx))
    r1 = _factorial_loglik(
        F, d, mu0, S0, tau2, 0.0, times, idx, Hl, R, y, FactorialKFConfig()
    )
    r3 = _oneshot_joint_loglik(F, d, mu0, S0, tau2, 0.0, times, idx, Hl, R, y)
    assert np.isfinite(r1) and r1 < 0.0
    assert abs(r1 - r3) < 1e-7


def test_factorial_ekf_loglik_biased_on_contrast_but_exact_on_full_rank_obs():
    """The EKF (Taylor) marginal loglik is biased on rank-deficient contrast obs.

    The contrast ``y = z_i - z_j`` leaves the sum direction unobserved (rank-
    deficient observation Hessian), so the Taylor filter's ``ignore_nan_dims``
    path yields a *biased* marginal likelihood. A *full-rank* observation (both
    factors observed) removes the deficiency and the EKF recovers the exact KF.
    """
    F, d, idx, y = _EXACT_CASES["chain_d1"]
    mu0, S0, tau2, _, _ = _params(d)
    times = _times(len(idx))

    # Contrast observation: EKF disagrees with the exact KF/joint value.
    Hl_c, R_c = _H1, _R1
    kf = _factorial_loglik(
        F, d, mu0, S0, tau2, 0.0, times, idx, Hl_c, R_c, y, FactorialKFConfig()
    )
    ekf = _factorial_loglik(
        F, d, mu0, S0, tau2, 0.0, times, idx, Hl_c, R_c, y, FactorialEKFConfig()
    )
    assert abs(ekf - kf) > 1e-2, "expected the EKF to be biased on contrast obs"

    # Full-rank observation: y = [z_i, z_j] directly -> EKF == KF (no deficiency).
    Hl_full = np.array([[1.0, 0.0], [0.0, 1.0]])
    R_full = np.array([[0.6, 0.0], [0.0, 0.6]])
    y_full = [[1.0, 0.3], [-0.4, 0.9], [0.2, -0.7]]
    kf_f = _factorial_loglik(
        F,
        d,
        mu0,
        S0,
        tau2,
        0.0,
        times,
        idx,
        Hl_full,
        R_full,
        y_full,
        FactorialKFConfig(),
    )
    ekf_f = _factorial_loglik(
        F,
        d,
        mu0,
        S0,
        tau2,
        0.0,
        times,
        idx,
        Hl_full,
        R_full,
        y_full,
        FactorialEKFConfig(),
    )
    assert abs(ekf_f - kf_f) < 1e-6, "EKF should match KF on a full-rank observation"
