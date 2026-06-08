"""Smoke tests for variational (SVI) training of a factorial model via dsx.sample.

Mirrors ``tests/test_hierarchical_svi_smokes.py`` but for a factorial outcome model:
priors on the hyperparameters → the idiomatic ``with Filter(FactorialPFConfig): dsx.sample``
→ ``numpyro.SVI`` with an ``AutoMultivariateNormal`` guide. The model is built with
``t0=None`` (an explicit scalar ``t0`` triggers a ``.item()`` ``ConcretizationTypeError`` in
t0 validation under JIT tracing); the PF makes the marginal likelihood differentiable, and a
fixed ``crn_seed`` (common random numbers) gives a deterministic likelihood / smooth ELBO.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)

import dynestyx as dsx
from dynestyx import (
    FactorialDynamicalModel,
    FactorialPFConfig,
    Filter,
    MatchOutcomeObservation,
    RandomWalkEvolution,
)

from .fixtures import _n_particles

jax.config.update("jax_enable_x64", True)

F = 4
OBS_TIMES = jnp.arange(4.0)  # starts at 0
OBS_IDX = jnp.array([[0, 1], [0, 2], [0, 3], [1, 2]])
RESULTS = jnp.array([1, 0, 2, 1])  # 0=draw, 1=home, 2=away


def _make_model(p):
    # t0=None is required for dsx.sample under SVI/NUTS tracing.
    return FactorialDynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.zeros(1), jnp.exp(p["log_init_var"]) * jnp.eye(1)
        ),
        state_evolution=RandomWalkEvolution(
            tau=jnp.exp(p["log_tau"]), factor_state_dim=1
        ),
        observation_model=MatchOutcomeObservation(
            draw_margin=jnp.exp(p["log_eps"]), scale=1.0, factor_state_dim=1
        ),
        num_factors=F,
        num_local_factors=2,
        t0=None,
    )


def _svi_model():
    p = {
        "log_init_var": numpyro.sample("log_init_var", dist.Normal(jnp.log(0.5), 0.5)),
        "log_tau": numpyro.sample("log_tau", dist.Normal(jnp.log(0.3), 0.5)),
        "log_eps": numpyro.sample("log_eps", dist.Normal(jnp.log(0.2), 0.5)),
    }
    with Filter(
        filter_config=FactorialPFConfig(
            n_particles=_n_particles(300), crn_seed=jr.PRNGKey(0)
        )
    ):
        dsx.sample(
            "skill",
            _make_model(p),
            obs_times=OBS_TIMES,
            obs_values=RESULTS,
            obs_factor_indices=OBS_IDX,
        )


def test_factorial_svi_mvn_smoke():
    """AutoMultivariateNormal SVI trains the factorial model and improves the ELBO."""
    guide = AutoMultivariateNormal(_svi_model)
    svi = SVI(_svi_model, guide, optax.adam(0.05), loss=Trace_ELBO())
    result = svi.run(jr.PRNGKey(0), 40, progress_bar=False)
    losses = result.losses
    assert jnp.all(jnp.isfinite(losses))
    # Deterministic (CRN) likelihood -> the loss should decrease.
    assert float(jnp.mean(losses[-10:])) < float(jnp.mean(losses[:10]))
    post = guide.sample_posterior(jr.PRNGKey(1), result.params, sample_shape=(16,))
    for k in ("log_init_var", "log_tau", "log_eps"):
        assert post[k].shape == (16,)
        assert jnp.all(jnp.isfinite(post[k]))


def test_factorial_svi_lowrank_smoke():
    """The AutoLowRankMultivariateNormal guide path also runs without error."""
    guide = AutoLowRankMultivariateNormal(_svi_model, rank=2)
    svi = SVI(_svi_model, guide, optax.adam(0.05), loss=Trace_ELBO())
    result = svi.run(jr.PRNGKey(0), 15, progress_bar=False)
    assert jnp.all(jnp.isfinite(result.losses))
