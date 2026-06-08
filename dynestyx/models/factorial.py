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
from dynestyx.models.distributions import BivariateNegativeBinomial, BivariatePoisson
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


class OrnsteinUhlenbeckEvolution(DiscreteTimeStateEvolution):
    r"""Per-factor mean-reverting (Ornstein--Uhlenbeck) skill evolution.

    A stationary alternative to :class:`RandomWalkEvolution`: instead of diffusing
    without bound, each factor's skill reverts toward a long-run mean $\mu_0$ at rate
    $r$, with a *stationary* equilibrium variance $\sigma_0^2$. The exact discrete-time
    OU transition over a gap $\Delta t = t_{k+1} - t_k$ (Duffield et al. 2024, §4.6,
    "Stationary dynamics") is

    $$
    x_{t_{k+1}}^f \mid x_{t_k}^f \sim \mathcal N\!\Big(
        x_{t_k}^f e^{-r\,\Delta t} + \mu_0\,(1 - e^{-r\,\Delta t}),\;
        \sigma_0^2\,(1 - e^{-2 r\,\Delta t})\,I_d \Big).
    $$

    Unlike Brownian motion, whose variance grows linearly with elapsed time (so a
    long gap implies an arbitrarily diffuse prior), the OU variance saturates at
    $\sigma_0^2$ regardless of the gap, so $\sigma_0$ directly and stably controls the
    spread of skills. As $r \to 0$ with $2 r \sigma_0^2$ held fixed it recovers the
    Brownian random walk (rate $\tau^2 = 2 r \sigma_0^2$). The stationary distribution
    is $\mathcal N(\mu_0, \sigma_0^2 I_d)$, which is the natural initial condition.

    Attributes:
        reversion_rate (jax.Array): Mean-reversion rate $r > 0$ (scalar or length
            ``factor_state_dim``). Larger $r$ = faster forgetting of past form.
        equilibrium_scale (jax.Array): Stationary standard deviation $\sigma_0$
            (scalar or length ``factor_state_dim``).
        long_run_mean (jax.Array): Long-run mean $\mu_0$ (scalar or length
            ``factor_state_dim``); defaults to ``0``.
        factor_state_dim (int): Per-factor state dimension ``d``.
    """

    reversion_rate: Float[Array, "*r_shape"]
    equilibrium_scale: Float[Array, "*scale_shape"]
    long_run_mean: Float[Array, "*mean_shape"]
    factor_state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        reversion_rate: float | int | Float[Array, "*r_shape"],
        equilibrium_scale: float | int | Float[Array, "*scale_shape"],
        long_run_mean: float | int | Float[Array, "*mean_shape"] = 0.0,
        factor_state_dim: int = 1,
    ):
        """
        Args:
            reversion_rate: Mean-reversion rate $r > 0$.
            equilibrium_scale: Stationary standard deviation $\\sigma_0$.
            long_run_mean: Long-run mean $\\mu_0$ (default ``0``).
            factor_state_dim: Per-factor state dimension ``d``. Defaults to ``1``.
        """
        self.reversion_rate = jnp.asarray(reversion_rate, dtype=float)
        self.equilibrium_scale = jnp.asarray(equilibrium_scale, dtype=float)
        self.long_run_mean = jnp.asarray(long_run_mean, dtype=float)
        self.factor_state_dim = int(factor_state_dim)

    def __call__(self, x, u, t_now, t_next):
        d = self.factor_state_dim
        dt = jnp.asarray(t_next, dtype=float) - jnp.asarray(t_now, dtype=float)
        r = jnp.broadcast_to(jnp.atleast_1d(self.reversion_rate), (d,))
        sigma2 = jnp.broadcast_to(
            jnp.atleast_1d(jnp.square(self.equilibrium_scale)), (d,)
        )
        mu = jnp.broadcast_to(jnp.atleast_1d(self.long_run_mean), (d,))
        decay = jnp.exp(-r * dt)
        loc = x * decay + mu * (1.0 - decay)
        # Variance saturates at sigma0^2; build the covariance directly (with a
        # positive floor) so it stays differentiable at dt == 0 (where var == 0).
        var = sigma2 * (1.0 - jnp.exp(-2.0 * r * dt)) + 1e-12
        return dist.MultivariateNormal(loc=loc, covariance_matrix=jnp.diag(var))


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


class BivariatePoissonScoreObservation(ObservationModel):
    r"""Bivariate-Poisson scoreline observation from attack/defense skills.

    Acts on the joint-local state of the two involved factors, a length-``4`` vector
    ``[att_i, def_i, att_j, def_j]`` (home ``i`` then away ``j``, each with a 2-D
    attack/defense state). It returns a :class:`~dynestyx.models.distributions.BivariatePoisson`
    over the scoreline ``(home_goals, away_goals)`` with rates (Duffield et al. §4.6,
    Karlis & Ntzoufras):

    $$
    \lambda_1 = \exp(\alpha + h + x^{\mathrm{att},i} - x^{\mathrm{def},j}),\quad
    \lambda_2 = \exp(\alpha + x^{\mathrm{att},j} - x^{\mathrm{def},i}),\quad
    \lambda_3 = \exp(\beta),
    $$

    where $\alpha$ is the baseline log scoring rate, $\beta$ the log shared-goals
    correlation rate, and $h$ an optional additive home advantage on the home team's
    scoring rate.

    Per-match covariates may be supplied through the observation control ``u`` (e.g.
    via ``obs_controls`` / ``predict_controls``), interpreted as
    ``u = [neutral, friendly]``: the home advantage is scaled to $h\,(1-\text{neutral})$
    (so neutral-venue matches get no home edge) and the baseline rate is shifted by
    ``friendly_offset * friendly`` (friendlies often have a different scoring level).
    With ``u=None`` the model reduces to a constant home advantage and baseline.

    Note:
        As with :class:`MatchOutcomeObservation`, the EKF (Taylor) factorial filter
        is **forward-only** for this observation: the score potential depends on the
        4-D state only through the two contrasts ``att_i - def_j`` and
        ``att_j - def_i``, so its linearized Hessian is rank-deficient and the
        marginal log-likelihood is not differentiable through the EKF. Use the EKF for
        filtering/smoothing/prediction and ``FactorialPFConfig`` (particle filter) for
        gradient-based parameter inference.

    Attributes:
        alpha (jax.Array): Baseline log scoring rate $\alpha$.
        beta (jax.Array): Log shared-goals correlation rate $\beta$ ($\lambda_3 = e^\beta$).
        home_advantage (jax.Array): Additive home-team log-rate shift $h$ (default 0).
        num_local_factors (int): Factors per observation; must be ``2`` (pairwise).
        factor_state_dim (int): Per-factor state dimension; must be ``2`` (attack, defense).
        max_goals (int): Convolution-sum cap of the bivariate Poisson (default ``12``).
    """

    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    home_advantage: Float[Array, ""]
    friendly_offset: Float[Array, ""]
    num_local_factors: int = eqx.field(static=True)
    factor_state_dim: int = eqx.field(static=True)
    max_goals: int = eqx.field(static=True)

    def __init__(
        self,
        alpha: float | int | Float[Array, ""] = 0.0,
        beta: float | int | Float[Array, ""] = -1.5,
        home_advantage: float | int | Float[Array, ""] = 0.0,
        friendly_offset: float | int | Float[Array, ""] = 0.0,
        *,
        num_local_factors: int = 2,
        factor_state_dim: int = 2,
        max_goals: int = 12,
    ):
        """
        Args:
            alpha: Baseline log scoring rate $\\alpha$.
            beta: Log shared-goals correlation rate $\\beta$.
            home_advantage: Additive home-team log-rate shift $h$ (scaled by
                ``1 - neutral`` when a per-match ``u = [neutral, friendly]`` is given).
            friendly_offset: Additive baseline-rate shift applied when ``friendly``
                (the second control) is set.
            num_local_factors: Factors per observation (must be ``2``).
            factor_state_dim: Per-factor state dimension (must be ``2``).
            max_goals: Convolution-sum cap of the bivariate Poisson.
        """
        if int(num_local_factors) != 2:
            raise ValueError(
                "BivariatePoissonScoreObservation supports pairwise comparisons only "
                f"(num_local_factors=2); got {num_local_factors}."
            )
        if int(factor_state_dim) != 2:
            raise ValueError(
                "BivariatePoissonScoreObservation requires a 2-D attack/defense state "
                f"(factor_state_dim=2); got {factor_state_dim}."
            )
        self.alpha = jnp.asarray(alpha, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)
        self.home_advantage = jnp.asarray(home_advantage, dtype=float)
        self.friendly_offset = jnp.asarray(friendly_offset, dtype=float)
        self.num_local_factors = int(num_local_factors)
        self.factor_state_dim = int(factor_state_dim)
        self.max_goals = int(max_goals)

    def __call__(self, x, u, t):
        x = jnp.reshape(x, (self.num_local_factors, self.factor_state_dim))
        att_i, def_i = x[0, 0], x[0, 1]
        att_j, def_j = x[1, 0], x[1, 1]
        neutral, friendly = 0.0, 0.0
        if u is not None:
            u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
            if u.shape[0] >= 1:
                neutral = u[0]
            if u.shape[0] >= 2:
                friendly = u[1]
        home = self.home_advantage * (1.0 - neutral)
        alpha_eff = self.alpha + self.friendly_offset * friendly
        lam1 = jnp.exp(alpha_eff + home + att_i - def_j)
        lam2 = jnp.exp(alpha_eff + att_j - def_i)
        lam3 = jnp.exp(self.beta)
        return BivariatePoisson(lam1, lam2, lam3, max_goals=self.max_goals)


class BivariateNegativeBinomialScoreObservation(ObservationModel):
    r"""Overdispersed scoreline observation from attack/defense skills.

    A drop-in, overdispersed alternative to :class:`BivariatePoissonScoreObservation`.
    Acts on the same length-``4`` joint-local state ``[att_i, def_i, att_j, def_j]`` and
    returns a :class:`~dynestyx.models.distributions.BivariateNegativeBinomial` over the
    scoreline ``(home_goals, away_goals)`` with the **same skill-contrast means** as the
    bivariate Poisson,

    $$
    \lambda_1 = \exp(\alpha + h + x^{\mathrm{att},i} - x^{\mathrm{def},j}),\quad
    \lambda_2 = \exp(\alpha + x^{\mathrm{att},j} - x^{\mathrm{def},i}),
    $$

    but with each margin negative-binomial (mean $\lambda_j$, variance
    $\lambda_j + \lambda_j^2 / r$) rather than Poisson (variance $= \lambda_j$), governed
    by a learnable dispersion $r = \exp(\texttt{log\_dispersion})$. As $r \to \infty$ this
    reduces to a product of independent Poissons (the bivariate Poisson with
    $\lambda_3 = 0$), so it strictly generalizes the Poisson scoreline likelihood with one
    extra parameter.

    Unlike the bivariate Poisson there is **no $\beta$/$\lambda_3$ shared-goals term**: the
    margins are modelled as independent (football scores are empirically
    near-uncorrelated, with the bivariate Poisson's $\lambda_3 = e^\beta \approx 0.01$ in
    fits), so $r$ captures overdispersion without imposing a spurious correlation. Per-match
    covariates ``u = [neutral, friendly]`` are handled exactly as in
    :class:`BivariatePoissonScoreObservation`.

    Note:
        As with :class:`BivariatePoissonScoreObservation`, the EKF (Taylor) factorial
        filter is **forward-only** for this observation (the score potential depends on the
        4-D state only through the two contrasts ``att_i - def_j`` and ``att_j - def_i``, so
        its linearized Hessian is rank-deficient). Use the EKF for
        filtering/smoothing/prediction and ``FactorialPFConfig`` (particle filter) for
        gradient-based / variational parameter inference.

    Attributes:
        alpha (jax.Array): Baseline log scoring rate $\alpha$.
        home_advantage (jax.Array): Additive home-team log-rate shift $h$ (default 0).
        friendly_offset (jax.Array): Baseline-rate shift applied to friendlies.
        log_dispersion (jax.Array): Log NB dispersion $\log r$; larger $\to$ Poisson.
        num_local_factors (int): Factors per observation; must be ``2`` (pairwise).
        factor_state_dim (int): Per-factor state dimension; must be ``2`` (attack, defense).
    """

    alpha: Float[Array, ""]
    home_advantage: Float[Array, ""]
    friendly_offset: Float[Array, ""]
    log_dispersion: Float[Array, ""]
    num_local_factors: int = eqx.field(static=True)
    factor_state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        alpha: float | int | Float[Array, ""] = 0.0,
        home_advantage: float | int | Float[Array, ""] = 0.0,
        friendly_offset: float | int | Float[Array, ""] = 0.0,
        log_dispersion: float | int | Float[Array, ""] = float(jnp.log(10.0)),
        *,
        num_local_factors: int = 2,
        factor_state_dim: int = 2,
    ):
        """
        Args:
            alpha: Baseline log scoring rate $\\alpha$.
            home_advantage: Additive home-team log-rate shift $h$ (scaled by
                ``1 - neutral`` when a per-match ``u = [neutral, friendly]`` is given).
            friendly_offset: Additive baseline-rate shift applied when ``friendly``
                (the second control) is set.
            log_dispersion: Log of the shared NB dispersion $r$ (default $\\log 10$, mildly
                overdispersed / near-Poisson); larger values approach the Poisson.
            num_local_factors: Factors per observation (must be ``2``).
            factor_state_dim: Per-factor state dimension (must be ``2``).
        """
        if int(num_local_factors) != 2:
            raise ValueError(
                "BivariateNegativeBinomialScoreObservation supports pairwise comparisons "
                f"only (num_local_factors=2); got {num_local_factors}."
            )
        if int(factor_state_dim) != 2:
            raise ValueError(
                "BivariateNegativeBinomialScoreObservation requires a 2-D attack/defense "
                f"state (factor_state_dim=2); got {factor_state_dim}."
            )
        self.alpha = jnp.asarray(alpha, dtype=float)
        self.home_advantage = jnp.asarray(home_advantage, dtype=float)
        self.friendly_offset = jnp.asarray(friendly_offset, dtype=float)
        self.log_dispersion = jnp.asarray(log_dispersion, dtype=float)
        self.num_local_factors = int(num_local_factors)
        self.factor_state_dim = int(factor_state_dim)

    def __call__(self, x, u, t):
        x = jnp.reshape(x, (self.num_local_factors, self.factor_state_dim))
        att_i, def_i = x[0, 0], x[0, 1]
        att_j, def_j = x[1, 0], x[1, 1]
        neutral, friendly = 0.0, 0.0
        if u is not None:
            u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
            if u.shape[0] >= 1:
                neutral = u[0]
            if u.shape[0] >= 2:
                friendly = u[1]
        home = self.home_advantage * (1.0 - neutral)
        alpha_eff = self.alpha + self.friendly_offset * friendly
        lam1 = jnp.exp(alpha_eff + home + att_i - def_j)
        lam2 = jnp.exp(alpha_eff + att_j - def_i)
        return BivariateNegativeBinomial(lam1, lam2, jnp.exp(self.log_dispersion))


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


def factorial_score_probabilities(
    observation_model: ObservationModelLike,
    predicted_means: Float[Array, "predict n_local factor_state_dim"],
    predicted_chol_covs: Float[
        Array, "predict n_local factor_state_dim factor_state_dim"
    ],
    *,
    key: jax.Array,
    n_samples: int = 1000,
    max_goals_grid: int = 6,
    t: float | int | Real[Array, ""] = 0.0,
    controls: Float[Array, "predict control_dim"] | None = None,
) -> tuple[Float[Array, "predict goals goals"], Float[Array, "predict three"]]:
    r"""Monte-Carlo predicted scoreline grids and win/draw/loss for future matches.

    For each future match, Monte-Carlo samples the predicted joint-local attack/defense
    skills $\sim \mathcal N(m, LL^\top)$, evaluates the bivariate-Poisson scoreline PMF
    over the $(0..G)\times(0..G)$ grid, and averages over the skill samples. The grid
    is then reduced to win/draw/loss probabilities (home goals on the first axis):

    $$
    \hat p(\text{score}) = \tfrac{1}{S}\sum_s p_{\mathrm{BP}}(\text{score} \mid x^{(s)}),
    \quad p(\text{home win}) = \sum_{a>b} \hat p(a, b),\ \text{etc.}
    $$

    Args:
        observation_model: A scoreline observation (e.g.
            :class:`BivariatePoissonScoreObservation`) returning a distribution over the
            ``(2,)`` scoreline. Its internal ``max_goals`` must be ``>= max_goals_grid``.
        predicted_means: Predicted per-factor means, shape ``(P, n_local, d)``.
        predicted_chol_covs: Predicted per-factor covariance Cholesky factors,
            shape ``(P, n_local, d, d)``.
        key: PRNG key.
        n_samples: Number of Monte-Carlo skill samples per match.
        max_goals_grid: Maximum goals ``G`` shown per team in the score grid.
        t: Time passed to the observation model (unused by scoreline models).
        controls: Optional per-match covariates ``(P, control_dim)`` passed to the
            observation as ``u`` (e.g. ``[neutral, friendly]`` flags).

    Returns:
        ``(score_grids, win_draw_loss)`` where ``score_grids`` has shape
        ``(P, G+1, G+1)`` (cell ``[a, b]`` is P(home scores ``a``, away scores ``b``);
        sums to ~1 up to the ``G`` truncation) and ``win_draw_loss`` has shape ``(P, 3)``
        as ``[home win, draw, away win]`` rows that sum to 1.
    """
    P, n_local, d = predicted_means.shape
    g = int(max_goals_grid)
    t_arr = jnp.asarray(t, dtype=float)
    aa, bb = jnp.meshgrid(jnp.arange(g + 1), jnp.arange(g + 1), indexing="ij")
    grid_values = jnp.stack([aa.ravel(), bb.ravel()], axis=-1).astype(float)
    if controls is None:
        controls_arr = jnp.zeros((P, 0))
    else:
        controls_arr = jnp.asarray(controls, dtype=float).reshape(P, -1)

    def per_match(key_p, mean_p, chol_p, u_p):
        u = u_p if u_p.shape[-1] > 0 else None
        eps = jax.random.normal(key_p, (n_samples, n_local, d))
        samples = mean_p[None] + jnp.einsum("lij,nlj->nli", chol_p, eps)
        joint = samples.reshape(n_samples, n_local * d)

        def grid_for_sample(xj):
            edist = observation_model(xj, u, t_arr)
            return jnp.exp(edist.log_prob(grid_values))

        grids = jax.vmap(grid_for_sample)(joint)  # (n_samples, (g+1)^2)
        return jnp.mean(grids, axis=0).reshape(g + 1, g + 1)

    keys = jax.random.split(key, P)
    score_grids = jax.vmap(per_match)(
        keys, predicted_means, predicted_chol_covs, controls_arr
    )

    home_win = jnp.sum(jnp.tril(score_grids, k=-1), axis=(-2, -1))
    away_win = jnp.sum(jnp.triu(score_grids, k=1), axis=(-2, -1))
    draw = jnp.sum(jnp.diagonal(score_grids, axis1=-2, axis2=-1), axis=-1)
    win_draw_loss = jnp.stack([home_win, draw, away_win], axis=-1)
    win_draw_loss = win_draw_loss / jnp.sum(win_draw_loss, axis=-1, keepdims=True)
    return score_grids, win_draw_loss
