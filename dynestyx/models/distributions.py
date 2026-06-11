"""Custom probability distributions for dynamical models.

Currently provides :class:`BivariatePoisson`, the Karlis--Ntzoufras bivariate Poisson
used by the attack/defense football score model (Duffield et al. 2024, §4.6).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Float, Int, Real
from numpyro.distributions import Distribution, constraints


class BivariatePoisson(Distribution):
    r"""Bivariate Poisson distribution (Karlis & Ntzoufras, 2003).

    A distribution over a pair of non-negative integer counts $(Y_1, Y_2)$ with a
    shared component inducing positive correlation. Constructively,
    $Y_1 = Z_1 + Z_3$, $Y_2 = Z_2 + Z_3$ with independent
    $Z_m \sim \mathrm{Poisson}(\lambda_m)$, giving

    $$
    p(y_1, y_2) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)}
        \frac{\lambda_1^{y_1}}{y_1!} \frac{\lambda_2^{y_2}}{y_2!}
        \sum_{k=0}^{\min(y_1, y_2)} \binom{y_1}{k}\binom{y_2}{k} k!
        \left(\frac{\lambda_3}{\lambda_1 \lambda_2}\right)^k,
    $$

    with marginals $\mathrm{Poisson}(\lambda_1 + \lambda_3)$ and
    $\mathrm{Poisson}(\lambda_2 + \lambda_3)$ and $\mathrm{cov}(Y_1, Y_2) = \lambda_3$.
    As $\lambda_3 \to 0$ it reduces to a product of two independent Poissons.

    In the football score model (Duffield et al. §4.6) this is the likelihood of a
    scoreline given two teams' attack/defense skills, with
    $\lambda_1 = \exp(\alpha + x^{\mathrm{att},i} - x^{\mathrm{def},j})$,
    $\lambda_2 = \exp(\alpha + x^{\mathrm{att},j} - x^{\mathrm{def},i})$, and
    $\lambda_3 = \exp(\beta)$.

    The log-density is evaluated in the numerically stable convolution form
    $\log p(y) = \log \sum_{k} \mathrm{Pois}(y_1 - k; \lambda_1)\,
    \mathrm{Pois}(y_2 - k; \lambda_2)\,\mathrm{Pois}(k; \lambda_3)$, truncated at
    ``max_goals`` (which must be at least $\min(y_1, y_2)$ for every evaluated point).

    Exact first and second moments are exposed via :attr:`mean`, :attr:`variance`, and
    :attr:`covariance_matrix`
    ($[[\lambda_1 + \lambda_3, \lambda_3], [\lambda_3, \lambda_2 + \lambda_3]]$, always
    positive-definite), which the moments-linearized EKF uses in place of Taylor
    linearization of the log-density.

    Attributes:
        lam1, lam2, lam3 (jax.Array): The component rates $\lambda_1, \lambda_2,
            \lambda_3 > 0$ (broadcast to the batch shape).
        max_goals (int): Upper limit of the convolution sum. Must be $\ge \min(y_1,
            y_2)$ for all evaluated/sampled points (defaults to ``12``).
    """

    arg_constraints = {
        "lam1": constraints.positive,
        "lam2": constraints.positive,
        "lam3": constraints.positive,
    }
    support = constraints.independent(constraints.nonnegative_integer, 1)
    pytree_data_fields = ("lam1", "lam2", "lam3")
    pytree_aux_fields = ("max_goals",)

    def __init__(
        self,
        lam1: Float[Array, "..."] | float,
        lam2: Float[Array, "..."] | float,
        lam3: Float[Array, "..."] | float,
        max_goals: int = 12,
        *,
        validate_args=None,
    ):
        self.lam1 = jnp.asarray(lam1, dtype=float)
        self.lam2 = jnp.asarray(lam2, dtype=float)
        self.lam3 = jnp.asarray(lam3, dtype=float)
        self.max_goals = int(max_goals)
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.lam1), jnp.shape(self.lam2), jnp.shape(self.lam3)
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    def log_prob(self, value: Real[Array, "*batch two"]) -> Float[Array, "*batch"]:
        value = jnp.asarray(value, dtype=float)
        y1 = value[..., 0, None]  # (..., 1)
        y2 = value[..., 1, None]
        lam1 = self.lam1[..., None]
        lam2 = self.lam2[..., None]
        lam3 = self.lam3[..., None]

        k = jnp.arange(self.max_goals + 1)  # (M+1,)
        n1 = y1 - k
        n2 = y2 - k
        valid = (n1 >= 0) & (n2 >= 0)
        # Clamp the count arguments before ``gammaln`` so invalid terms stay finite
        # (then masked to -inf), keeping the result twice-differentiable in the
        # rates for the EKF Hessian and PF gradients.
        safe_n1 = jnp.where(valid, n1, 0.0)
        safe_n2 = jnp.where(valid, n2, 0.0)

        def log_pois(n, lam):
            return n * jnp.log(lam) - lam - gammaln(n + 1.0)

        terms = (
            log_pois(safe_n1, lam1)
            + log_pois(safe_n2, lam2)
            + log_pois(k.astype(float), lam3)
        )
        terms = jnp.where(valid, terms, -jnp.inf)
        return logsumexp(terms, axis=-1)

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> Int[Array, "*sample_and_batch two"]:
        shape = tuple(sample_shape) + self.batch_shape
        k1, k2, k3 = jax.random.split(key, 3)
        z1 = jax.random.poisson(k1, jnp.broadcast_to(self.lam1, shape), shape)
        z2 = jax.random.poisson(k2, jnp.broadcast_to(self.lam2, shape), shape)
        z3 = jax.random.poisson(k3, jnp.broadcast_to(self.lam3, shape), shape)
        return jnp.stack([z1 + z3, z2 + z3], axis=-1)

    @property
    def mean(self) -> Float[Array, "*batch two"]:
        return jnp.stack([self.lam1 + self.lam3, self.lam2 + self.lam3], axis=-1)

    @property
    def variance(self) -> Float[Array, "*batch two"]:
        return jnp.stack([self.lam1 + self.lam3, self.lam2 + self.lam3], axis=-1)

    @property
    def covariance_matrix(self) -> Float[Array, "*batch two two"]:
        lam1 = jnp.broadcast_to(self.lam1, self.batch_shape)
        lam2 = jnp.broadcast_to(self.lam2, self.batch_shape)
        lam3 = jnp.broadcast_to(self.lam3, self.batch_shape)
        row1 = jnp.stack([lam1 + lam3, lam3], axis=-1)
        row2 = jnp.stack([lam3, lam2 + lam3], axis=-1)
        return jnp.stack([row1, row2], axis=-2)


class BivariateNegativeBinomial(Distribution):
    r"""Bivariate negative-binomial distribution (independent overdispersed margins).

    An overdispersed alternative to :class:`BivariatePoisson` for a pair of counts
    $(Y_1, Y_2)$. Each margin is an independent negative binomial with mean $\lambda_j$
    and a shared dispersion $r > 0$:

    $$
    Y_j \sim \mathrm{NB}\!\left(\text{mean}=\lambda_j,\ \text{dispersion}=r\right),
    \qquad j \in \{1, 2\}\ \text{(independent)},
    $$

    so $\mathbb{E}[Y_j] = \lambda_j$ and $\mathrm{Var}[Y_j] = \lambda_j + \lambda_j^2 / r$
    (variance $>$ mean: overdispersed). The two scores are **independent**
    ($\mathrm{cov}(Y_1, Y_2) = 0$); as $r \to \infty$ the variance collapses to the mean
    and the distribution reduces to a product of two independent
    $\mathrm{Poisson}(\lambda_j)$.

    The log-density is the sum of the two NB log-pmfs in the mean/dispersion
    parameterization,

    $$
    \log p(y_1, y_2) = \sum_{j=1,2}
        \log\Gamma(y_j + r) - \log\Gamma(r) - \log\Gamma(y_j + 1)
        + r\,\log\!\frac{r}{r + \lambda_j} + y_j\,\log\!\frac{\lambda_j}{r + \lambda_j},
    $$

    closed-form and twice-differentiable in $\lambda_1, \lambda_2, r$ (no convolution
    sum), so it slots into the factorial EKF/PF filters and VI in place of
    :class:`BivariatePoisson`. Exact moments are exposed via :attr:`mean`,
    :attr:`variance`, and the (diagonal) :attr:`covariance_matrix` for the
    moments-linearized EKF.

    In the football score model this is the likelihood of a scoreline given two teams'
    attack/defense skills, with
    $\lambda_1 = \exp(\alpha + h + x^{\mathrm{att},i} - x^{\mathrm{def},j})$ and
    $\lambda_2 = \exp(\alpha + x^{\mathrm{att},j} - x^{\mathrm{def},i})$ -- the same mean
    structure as the bivariate Poisson, with overdispersion added through $r$. (There is
    no shared-goals $\lambda_3$ term: football scores are empirically near-uncorrelated,
    so the margins are modelled independently and $r$ captures overdispersion alone.)

    Attributes:
        lam1, lam2 (jax.Array): The margin means $\lambda_1, \lambda_2 > 0$ (broadcast to
            the batch shape).
        dispersion (jax.Array): The shared NB dispersion $r > 0$; smaller is more
            overdispersed, $r \to \infty$ recovers the Poisson.
    """

    arg_constraints = {
        "lam1": constraints.positive,
        "lam2": constraints.positive,
        "dispersion": constraints.positive,
    }
    support = constraints.independent(constraints.nonnegative_integer, 1)
    pytree_data_fields = ("lam1", "lam2", "dispersion")

    def __init__(
        self,
        lam1: Float[Array, "..."] | float,
        lam2: Float[Array, "..."] | float,
        dispersion: Float[Array, "..."] | float,
        *,
        validate_args=None,
    ):
        self.lam1 = jnp.asarray(lam1, dtype=float)
        self.lam2 = jnp.asarray(lam2, dtype=float)
        self.dispersion = jnp.asarray(dispersion, dtype=float)
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.lam1), jnp.shape(self.lam2), jnp.shape(self.dispersion)
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    def _nb2_log_prob(self, y, lam):
        # No jaxtyping annotations: ``y`` (e.g. a score grid) and ``lam`` (the batch rate)
        # broadcast to different shapes, which a shared ``*batch`` axis would reject.
        r = self.dispersion
        # NB in (mean=lam, dispersion=r) form; log r/(r+lam) and log lam/(r+lam) written
        # as differences to stay twice-differentiable in lam and r (for the EKF Hessian
        # and PF/VI gradients).
        log_r = jnp.log(r)
        log_lam = jnp.log(lam)
        log_rl = jnp.log(r + lam)
        return (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * (log_r - log_rl)
            + y * (log_lam - log_rl)
        )

    def log_prob(self, value: Real[Array, "*batch two"]) -> Float[Array, "*batch"]:
        value = jnp.asarray(value, dtype=float)
        y1 = value[..., 0]
        y2 = value[..., 1]
        return self._nb2_log_prob(y1, self.lam1) + self._nb2_log_prob(y2, self.lam2)

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> Int[Array, "*sample_and_batch two"]:
        shape = tuple(sample_shape) + self.batch_shape
        kg1, kp1, kg2, kp2 = jax.random.split(key, 4)
        r = jnp.broadcast_to(self.dispersion, shape)

        def nb_sample(kg, kp, lam):
            lam = jnp.broadcast_to(lam, shape)
            # NB == Gamma(shape=r, rate=r/lam)-mixed Poisson: draw the Poisson rate then count.
            rate = jax.random.gamma(kg, r, shape) * lam / r
            return jax.random.poisson(kp, rate, shape)

        y1 = nb_sample(kg1, kp1, self.lam1)
        y2 = nb_sample(kg2, kp2, self.lam2)
        return jnp.stack([y1, y2], axis=-1)

    @property
    def mean(self) -> Float[Array, "*batch two"]:
        return jnp.stack([self.lam1, self.lam2], axis=-1)

    @property
    def variance(self) -> Float[Array, "*batch two"]:
        r = self.dispersion
        return jnp.stack(
            [self.lam1 + self.lam1**2 / r, self.lam2 + self.lam2**2 / r], axis=-1
        )

    @property
    def covariance_matrix(self) -> Float[Array, "*batch two two"]:
        return self.variance[..., None] * jnp.eye(2)
