"""Dynamical models: core interfaces, state evolution, and observations.

Structure anticipates future extension to LTI factories, Neural SDEs, etc.
"""

from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    Drift,
    DynamicalModel,
    ObservationModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.diffusions import (
    DiagonalDiffusion,
    Diffusion,
    FullDiffusion,
    ScalarDiffusion,
)
from dynestyx.models.distributions import BivariatePoisson
from dynestyx.models.factorial import (
    BivariatePoissonScoreObservation,
    FactorialDynamicalModel,
    MatchOutcomeObservation,
    OrnsteinUhlenbeckEvolution,
    RandomWalkEvolution,
    factorial_outcome_probabilities,
    factorial_score_probabilities,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete
from dynestyx.models.observations import (
    DiracIdentityObservation,
    GaussianObservation,
    LinearGaussianObservation,
)
from dynestyx.models.state_evolution import (
    AffineDrift,
    GaussianStateEvolution,
    LinearGaussianStateEvolution,
)

__all__ = [
    "ContinuousTimeStateEvolution",
    "DeterministicContinuousTimeStateEvolution",
    "AffineDrift",
    "DiracIdentityObservation",
    "Diffusion",
    "DiscreteTimeStateEvolution",
    "DiagonalDiffusion",
    "BivariatePoisson",
    "BivariatePoissonScoreObservation",
    "DynamicalModel",
    "Drift",
    "FactorialDynamicalModel",
    "FullDiffusion",
    "GaussianObservation",
    "GaussianStateEvolution",
    "LinearGaussianObservation",
    "LinearGaussianStateEvolution",
    "MatchOutcomeObservation",
    "ObservationModel",
    "OrnsteinUhlenbeckEvolution",
    "RandomWalkEvolution",
    "factorial_outcome_probabilities",
    "factorial_score_probabilities",
    "StochasticContinuousTimeStateEvolution",
    "LTI_continuous",
    "LTI_discrete",
    "ScalarDiffusion",
]
