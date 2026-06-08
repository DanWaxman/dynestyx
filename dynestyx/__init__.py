"""Dynestyx package."""

from importlib.metadata import version

__version__ = version("dynestyx")

from dynestyx.discretizers import Discretizer, euler_maruyama
from dynestyx.handlers import plate, sample
from dynestyx.inference.filter_configs import (
    FactorialEKFConfig,
    FactorialKFConfig,
    FactorialPFConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.inference.smoother_configs import (
    FactorialEKFSmootherConfig,
    FactorialKFSmootherConfig,
    FactorialPFSmootherConfig,
)
from dynestyx.inference.smoothers import Smoother
from dynestyx.models import (
    BivariatePoisson,
    BivariatePoissonScoreObservation,
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DiagonalDiffusion,
    Diffusion,
    DiracIdentityObservation,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    FactorialDynamicalModel,
    FullDiffusion,
    GaussianObservation,
    GaussianStateEvolution,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    MatchOutcomeObservation,
    ObservationModel,
    OrnsteinUhlenbeckEvolution,
    RandomWalkEvolution,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
    factorial_outcome_probabilities,
    factorial_score_probabilities,
)
from dynestyx.simulators import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    Simulator,
)
from dynestyx.utils import flatten_draws

__all__ = [
    "__version__",
    "ContinuousTimeStateEvolution",
    "DeterministicContinuousTimeStateEvolution",
    "Diffusion",
    "FullDiffusion",
    "DiagonalDiffusion",
    "ScalarDiffusion",
    "StochasticContinuousTimeStateEvolution",
    "DiscreteTimeStateEvolution",
    "DynamicalModel",
    "FactorialDynamicalModel",
    "RandomWalkEvolution",
    "OrnsteinUhlenbeckEvolution",
    "MatchOutcomeObservation",
    "BivariatePoisson",
    "BivariatePoissonScoreObservation",
    "factorial_outcome_probabilities",
    "factorial_score_probabilities",
    "FactorialEKFConfig",
    "FactorialKFConfig",
    "FactorialPFConfig",
    "FactorialEKFSmootherConfig",
    "FactorialKFSmootherConfig",
    "FactorialPFSmootherConfig",
    "AffineDrift",
    "LTI_continuous",
    "LTI_discrete",
    "LinearGaussianStateEvolution",
    "GaussianStateEvolution",
    "Discretizer",
    "ObservationModel",
    "Filter",
    "Smoother",
    "flatten_draws",
    "plate",
    "sample",
    "DiracIdentityObservation",
    "LinearGaussianObservation",
    "GaussianObservation",
    "DiscreteTimeSimulator",
    "ODESimulator",
    "SDESimulator",
    "Simulator",
    "euler_maruyama",
]
