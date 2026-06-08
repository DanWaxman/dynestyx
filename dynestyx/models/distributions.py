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
