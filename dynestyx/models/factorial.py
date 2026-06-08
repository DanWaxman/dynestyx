"""Factorial dynamical models.

A *factorial* state-space model (fSSM) partitions the latent state into ``F``
conditionally independent **factors** (e.g. players or teams), each evolving with
its own Markov dynamics, where every observation is *local* -- it depends only on
the small subset of factors involved in that observation (e.g. the home and away
teams in a match):

$$
p(x_t \\mid x_{t-1}) = \\prod_{f=1}^{F} p(x_t^f \\mid x_{t-1}^f),
\\qquad p(y_k \\mid x_k) = p(y_k \\mid x_k^{S_k}).
$$

This is the model class of Duffield et al. (2024), *"A state-space perspective on
modelling and inference for online skill rating"* (JRSS-C). Inference uses a
*factored approximation* of the filtering/smoothing distributions
($p(x_t \\mid y_{1:t}) \\approx \\prod_f p(x_t^f \\mid y_{1:t})$), giving cost
$O(N + K)$ in the number of factors $N$ and observations $K$ via match-sparsity
and pairwise updates.

The :class:`FactorialDynamicalModel` reinterprets the standard
:class:`~dynestyx.models.core.DynamicalModel` fields to describe a *single factor*
plus the *local* observation:

- ``initial_condition`` -- the per-factor prior $p(x_0^f)$, a vector distribution
  with ``event_shape == (d,)`` where ``d`` is the per-factor state dimension.
- ``state_evolution`` -- the per-factor Markov transition, a
  :class:`~dynestyx.models.core.DiscreteTimeStateEvolution` whose ``(t_now,
  t_next)`` arguments carry that factor's own previous/current observation times
  (so the time-gap is per-factor; see the pairwise updates of Duffield et al.
  §3.4.3).
- ``observation_model`` -- the *local/joint* observation acting on the
  concatenation of the involved factors' states, of shape ``(n_local * d,)``.

Inference for factorial models is dispatched by model type (an early branch on
``isinstance(dynamics, FactorialDynamicalModel)`` in the ``Filter``/``Smoother``
handlers) and is implemented by the cuthbert factorial integration. Per-match
factor indices are supplied as a new ``obs_factor_indices`` keyword argument to
``dsx.sample``.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real
from numpyro.distributions import Distribution

from dynestyx.models.checkers import _infer_vector_dim_from_distribution
from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    ObservationModel,
    ObservationModelLike,
    StateEvolutionLike,
)
from dynestyx.types import as_scalar_time_array


class RandomWalkEvolution(DiscreteTimeStateEvolution):
    r"""Per-factor random-walk skill evolution with time-gap-dependent variance.

    The skill of a single factor evolves as a Brownian random walk observed at
    irregular times:

    $$
    x_{t_{k+1}}^f \\sim \\mathcal{N}\\!\\left(x_{t_k}^f,\\ \\tau^2 (t_{k+1} - t_k)\\, I_d\\right).
    $$

    Because the variance grows linearly with the elapsed time, propagating by a
    gap ``dt1`` then ``dt2`` is equivalent to propagating by ``dt1 + dt2`` -- the
    property that makes the match-sparse factorial representation exact for the
    dynamics. The ``(t_now, t_next)`` passed by the factorial backend are that
    factor's own previous and current observation times.

    Attributes:
        tau (jax.Array): Skill-evolution rate $\\tau$. A scalar (isotropic) or a
            vector of length ``factor_state_dim`` (per-dimension rates).
        factor_state_dim (int): Per-factor state dimension ``d``.
    """

    tau: Float[Array, "*tau_shape"]
    factor_state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        tau: float | int | Float[Array, "*tau_shape"],
        factor_state_dim: int = 1,
    ):
        """
        Args:
            tau: Skill-evolution rate (scalar or length-``factor_state_dim``).
            factor_state_dim: Per-factor state dimension ``d``. Defaults to ``1``.
        """
        self.tau = jnp.asarray(tau, dtype=float)
        self.factor_state_dim = int(factor_state_dim)

    def __call__(self, x, u, t_now, t_next):
        dt = jnp.asarray(t_next, dtype=float) - jnp.asarray(t_now, dtype=float)
        var = jnp.square(self.tau) * dt
        var = jnp.broadcast_to(jnp.atleast_1d(var), (self.factor_state_dim,))
        # Build the covariance directly (with a positive floor) rather than via a
        # standard deviation: ``sqrt(var)`` has an infinite gradient at ``var == 0``
        # (which happens at the first observation, where ``dt == 0``), so flooring
        # the variance keeps the marginal log-likelihood differentiable in ``tau``.
        cov_diag = var + 1e-12
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.diag(cov_diag))


class MatchOutcomeObservation(ObservationModel):
    r"""Pairwise match-outcome observation acting on a joint-local state.

    Given the joint-local state of the two involved factors (home then away,
    concatenated to shape ``(2 * d,)``), this returns a 3-way
    ``Categorical`` over the outcomes ``{0: draw, 1: home win, 2: away win}``
    following the sigmoidal model of Duffield et al. (2024), eq. (5). With skill
    difference $\\delta = x^{\\text{home}} - x^{\\text{away}} + h$ (home advantage
    $h$), scale $s$, and draw margin $\\varepsilon$:

    $$
    \\begin{aligned}
    p(\\text{home}) &= \\sigma\\!\\left(\\tfrac{\\delta - \\varepsilon}{s}\\right), \\\\
    p(\\text{away}) &= 1 - \\sigma\\!\\left(\\tfrac{\\delta + \\varepsilon}{s}\\right), \\\\
    p(\\text{draw}) &= \\sigma\\!\\left(\\tfrac{\\delta + \\varepsilon}{s}\\right)
                      - \\sigma\\!\\left(\\tfrac{\\delta - \\varepsilon}{s}\\right).
    \\end{aligned}
    $$

    For a vector per-factor state (``d > 1``) the first component is used as the
    scalar strength.

    Attributes:
        draw_margin (jax.Array): Draw-propensity parameter $\\varepsilon \\ge 0$.
            Larger values make draws more likely. Set to ``0`` for sports
            without draws.
        home_advantage (jax.Array): Additive shift $h$ applied to the home
            factor's strength. Defaults to ``0``.
        scale (jax.Array): Logistic scale $s > 0$. Defaults to ``1``.
        num_local_factors (int): Number of factors per observation. Must be ``2``
            (pairwise). Defaults to ``2``.
        factor_state_dim (int): Per-factor state dimension ``d``. Defaults to ``1``.
    """

    draw_margin: Float[Array, ""]
    home_advantage: Float[Array, ""]
    scale: Float[Array, ""]
    num_local_factors: int = eqx.field(static=True)
    factor_state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        draw_margin: float | int | Float[Array, ""] = 0.0,
        home_advantage: float | int | Float[Array, ""] = 0.0,
        scale: float | int | Float[Array, ""] = 1.0,
        *,
        num_local_factors: int = 2,
        factor_state_dim: int = 1,
    ):
        """
        Args:
            draw_margin: Draw-propensity parameter $\\varepsilon \\ge 0$.
            home_advantage: Additive home-team strength shift $h$.
            scale: Logistic scale $s > 0$.
            num_local_factors: Factors per observation (must be ``2``).
            factor_state_dim: Per-factor state dimension ``d``.
        """
        if int(num_local_factors) != 2:
            raise ValueError(
                "MatchOutcomeObservation currently supports pairwise comparisons "
                f"only (num_local_factors=2); got {num_local_factors}."
            )
        self.draw_margin = jnp.asarray(draw_margin, dtype=float)
        self.home_advantage = jnp.asarray(home_advantage, dtype=float)
        self.scale = jnp.asarray(scale, dtype=float)
        self.num_local_factors = int(num_local_factors)
        self.factor_state_dim = int(factor_state_dim)

    def __call__(self, x, u, t):
        x = jnp.reshape(x, (self.num_local_factors, self.factor_state_dim))
        x_home = x[0, 0]
        x_away = x[1, 0]
        delta = x_home - x_away + self.home_advantage
        eps = self.draw_margin
        s = self.scale
        sig_plus = jax.nn.sigmoid((delta + eps) / s)
        sig_minus = jax.nn.sigmoid((delta - eps) / s)
        p_home = sig_minus
        p_away = 1.0 - sig_plus
        p_draw = sig_plus - sig_minus
        probs = jnp.stack([p_draw, p_home, p_away])
        # Floor to keep ``log_prob`` finite for the EKF linearization, then
        # renormalize (the unclipped probabilities already sum to 1).
        probs = jnp.clip(probs, 1e-7, 1.0)
        probs = probs / jnp.sum(probs)
        return dist.Categorical(probs=probs)


class FactorialDynamicalModel(DynamicalModel):
    r"""A factorial state-space model with conditionally independent factors.

    See the module docstring for the modelling assumptions. This subclass of
    :class:`~dynestyx.models.core.DynamicalModel` reinterprets the inherited
    fields to describe one factor's sub-model plus the local observation, and
    adds the factorial structure (number of factors ``F`` and factors-per-
    observation ``n_local``). The inherited ``state_dim`` equals the per-factor
    dimension ``d`` (also exposed as :attr:`factor_state_dim`).

    Factorial models are **discrete-time only** (the per-factor transition is a
    :class:`~dynestyx.models.core.DiscreteTimeStateEvolution` whose ``(t_now,
    t_next)`` carry irregular per-factor observation times); passing a
    continuous-time state evolution raises ``ValueError``.

    Usage::

        fmodel = FactorialDynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.5 * jnp.eye(1)),
            state_evolution=RandomWalkEvolution(tau=0.05, factor_state_dim=1),
            observation_model=MatchOutcomeObservation(draw_margin=0.3),
            num_factors=num_teams,
            num_local_factors=2,
        )
        with Filter(filter_config=FactorialEKFConfig()):
            dsx.sample("skills", fmodel, obs_times=t, obs_values=y,
                       obs_factor_indices=home_away)

    Attributes:
        num_factors (int): Total number of factors ``F`` (e.g. teams).
        num_local_factors (int): Number of factors involved per observation
            ``n_local`` (e.g. ``2`` for pairwise matches).
        factor_state_dim (int): Per-factor state dimension ``d`` (alias of
            :attr:`state_dim`).
    """

    num_factors: int = eqx.field(static=True)
    num_local_factors: int = eqx.field(static=True)

    def __init__(
        self,
        initial_condition: Distribution,
        state_evolution: StateEvolutionLike,
        observation_model: ObservationModelLike,
        control_dim: int | None = None,
        control_model: Any = None,
        *,
        num_factors: int,
        num_local_factors: int = 2,
        t0: float | int | Array | None = None,
        state_dim: int | None = None,
        observation_dim: int | None = None,
        categorical_state: bool | None = None,
        continuous_time: bool | None = None,
    ):
        """
        Args:
            initial_condition: Per-factor prior $p(x_0^f)$. A NumPyro distribution
                with ``event_shape == (d,)`` (use ``dist.Normal(...).to_event(1)``
                or ``dist.MultivariateNormal`` for scalar skills, ``d == 1``). May
                be shared (``batch_shape == ()``) or per-factor
                (``batch_shape == (F,)``).
            state_evolution: Per-factor :class:`DiscreteTimeStateEvolution`.
            observation_model: Local observation acting on the joint-local state
                of shape ``(num_local_factors * d,)``.
            control_dim: Control dimension (defaults to ``0``; controls are not
                yet used by factorial backends).
            control_model: Reserved; not used.
            num_factors: Total number of factors ``F``.
            num_local_factors: Factors per observation ``n_local`` (default ``2``).
            t0: Optional declared start time.
            state_dim, observation_dim, categorical_state, continuous_time:
                Inferred automatically; accepted only so the model round-trips
                cleanly through effectful handlers. Do not set them by hand.
        """
        if isinstance(state_evolution, ContinuousTimeStateEvolution):
            raise ValueError(
                "FactorialDynamicalModel is discrete-time only; got a "
                "ContinuousTimeStateEvolution. Use a DiscreteTimeStateEvolution "
                "(e.g. RandomWalkEvolution) whose (t_now, t_next) carry the "
                "per-factor observation times."
            )

        event_shape = tuple(
            int(s) for s in getattr(initial_condition, "event_shape", ())
        )
        if len(event_shape) != 1:
            raise ValueError(
                "FactorialDynamicalModel.initial_condition must describe a single "
                "factor as a vector distribution with event_shape (d,). For scalar "
                "skills use dist.Normal(...).to_event(1) or dist.MultivariateNormal. "
                f"Got event_shape {event_shape}."
            )
        inferred_state_dim = _infer_vector_dim_from_distribution(
            initial_condition, "initial_condition", allow_batch_shape=True
        )
        if state_dim is not None and int(state_dim) != int(inferred_state_dim):
            raise ValueError(
                "state_dim does not match inferred per-factor state dimension. "
                f"Got state_dim={state_dim}, inferred={inferred_state_dim}."
            )

        if int(num_local_factors) < 1:
            raise ValueError(
                f"num_local_factors must be >= 1; got {num_local_factors}."
            )
        if int(num_factors) < int(num_local_factors):
            raise ValueError(
                f"num_factors ({num_factors}) must be >= num_local_factors "
                f"({num_local_factors})."
            )

        if control_dim is None:
            control_dim = 0

        # Probe the local observation model on a joint-local zero state to infer
        # the observation dimension.
        joint_local_dim = int(num_local_factors) * int(inferred_state_dim)
        t_probe = jnp.array(0.0) if t0 is None else as_scalar_time_array(t0, name="t0")
        u_probe = None if control_dim == 0 else jnp.zeros((control_dim,))
        if observation_dim is not None:
            inferred_obs_dim = int(observation_dim)
        else:
            try:
                obs_dist = observation_model(
                    jnp.zeros((joint_local_dim,)), u_probe, t_probe
                )
                inferred_obs_dim = int(
                    _infer_vector_dim_from_distribution(
                        obs_dist, "observation_model(joint_local_x, u, t)"
                    )
                )
            except Exception:
                inferred_obs_dim = 1

        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model
        self.t0 = None if t0 is None else as_scalar_time_array(t0, name="t0")
        self.state_dim = int(inferred_state_dim)
        self.observation_dim = int(inferred_obs_dim)
        self.control_dim = int(control_dim)
        self.categorical_state = False
        self.continuous_time = False
        self.num_factors = int(num_factors)
        self.num_local_factors = int(num_local_factors)

    @property
    def factor_state_dim(self) -> int:
        """Per-factor state dimension ``d`` (alias of :attr:`state_dim`)."""
        return self.state_dim


def factorial_outcome_probabilities(
    observation_model: ObservationModelLike,
    predicted_means: Float[Array, "predict n_local factor_state_dim"],
    predicted_chol_covs: Float[
        Array, "predict n_local factor_state_dim factor_state_dim"
    ],
    *,
    key: jax.Array,
    n_samples: int = 1000,
    t: float | int | Real[Array, ""] = 0.0,
) -> Float[Array, "predict n_outcomes"]:
    r"""Monte-Carlo predicted outcome probabilities for future matches.

    Given the predicted *joint-local* skill distribution for each future match
    (the propagated per-factor filtered/smoothed distributions, as recorded by
    the factorial rollout under the ``{name}_predicted_*`` sites), this integrates
    the (categorical) observation model over those skills by sampling and
    averaging:

    $$
    \\hat p(y \\mid \\text{match}) \\approx \\frac{1}{S} \\sum_{s=1}^{S}
        p\\!\\left(y \\mid x^{(s)}\\right),\\quad x^{(s)} \\sim
        \\mathcal{N}(m, L L^\\top).
    $$

    Args:
        observation_model: The local match-outcome observation model. Must return
            a distribution exposing ``.probs`` (e.g. ``MatchOutcomeObservation``).
        predicted_means: Predicted per-factor means, shape
            ``(P, n_local, d)``.
        predicted_chol_covs: Predicted per-factor covariance Cholesky factors,
            shape ``(P, n_local, d, d)``.
        key: PRNG key.
        n_samples: Number of Monte-Carlo samples per match.
        t: Time passed to the observation model (unused by outcome-only models).

    Returns:
        Predicted outcome probabilities, shape ``(P, n_outcomes)``; rows sum to 1.
    """
    P, n_local, d = predicted_means.shape
    t_arr = jnp.asarray(t, dtype=float)

    def per_match(key_p, mean_p, chol_p):
        # mean_p: (n_local, d); chol_p: (n_local, d, d)
        eps = jax.random.normal(key_p, (n_samples, n_local, d))
        samples = mean_p[None] + jnp.einsum("lij,nlj->nli", chol_p, eps)
        joint = samples.reshape(n_samples, n_local * d)
        probs = jax.vmap(lambda xj: observation_model(xj, None, t_arr).probs)(joint)
        return jnp.mean(probs, axis=0)

    keys = jax.random.split(key, P)
    return jax.vmap(per_match)(keys, predicted_means, predicted_chol_covs)
