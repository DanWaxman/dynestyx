![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/BasisResearch/dynestyx/test.yml) ![GitHub License](https://img.shields.io/github/license/BasisResearch/dynestyx)

# Welcome to Dynestyx

![dynestyx logo](docs/logo/dynestyx.gif)

`dynestyx` is a library designed for Bayesian modeling and inference of dynamical systems. It is an extension of [NumPyro](https://num.pyro.ai/en/stable/), and incorporates a wide variety of state-of-the-art inference methods for state space models.

To get started, you can [read the documentation](https://basisresearch.github.io/dynestyx) (version menu: **stable** = latest release, **latest** = `main`) or go straight to the [quickstart](https://basisresearch.github.io/dynestyx/stable/tutorials/quickstart/).

> ## 🤖 AI-coded factorial-SSM extension
>
> This fork is an **AI-coded extension of [Dynestyx](https://github.com/BasisResearch/dynestyx)** that adds first-class support for **factorial state-space models** (fSSMs), designed as a `dynestyx`-based counterpart to [`cuthberto-carlos`](https://github.com/state-space-models/cuthberto-carlos) — the football demo built on [`cuthbert`](https://github.com/state-space-models/cuthbert).
>
> A factorial SSM treats a large collection of conditionally-independent *factors* (e.g. players or teams), each with its own Markov skill dynamics, observed only through *local* — typically pairwise — comparisons (e.g. one match between two teams). This extension makes that a first-class [`FactorialDynamicalModel`](dynestyx/models/factorial.py) and wraps `cuthbert`'s factor-marginalized filtering/smoothing backends (EKF, KF, and particle filter) behind dynestyx's unified `DynamicalModel` + `dsx.sample()` API, so the same model code runs under any inference method.
>
> It also adds a **bivariate-Poisson scoreline observation** (attack/defense skills, à la Karlis–Ntzoufras / Dixon–Coles), a [`dynestyx.contrib`](dynestyx/contrib/) helper for held-out one-step-ahead predictive evaluation, and deep-dive notebooks that forecast the **2026 World Cup** and **reproduce the online skill-rating results of [Duffield et al. (2024)](https://doi.org/10.1093/jrsssc/qlae035)** (their package [`abile`](https://github.com/SamDuffield/abile)) on tennis, chess, and football data.
>
> **Deep dives:** [World Cup](docs/deep_dives/factorial_world_cup.ipynb) · [Duffield — tennis](docs/deep_dives/duffield_tennis_wta.ipynb) · [Duffield — chess](docs/deep_dives/duffield_chess.ipynb) · [Duffield — football](docs/deep_dives/duffield_football_epl.ipynb)

## Goals of `dynestyx`

The goal of `dynestyx` is to decouple model code and inference code for dynamical systems, a common theme in *probabilistic programming languages* like [NumPyro](https://num.pyro.ai/en/stable/). The benefits of this are two-fold: modellers get an interface that is simple to use, with access to advanced inference methods for free. Methods researchers simultaneously get a platform where their methodologies can be immediately used, with a natural testbed of problems to evaluate performance on.

### Relation to Existing Libraries

While many probabilistic programming languages now exist (e.g., [Pyro](https://pyro.ai/), [NumPyro](https://num.pyro.ai/en/stable/), and [Stan](https://mc-stan.org/)), these solutions do not offer support of structured inference methods specifically designed for the dynamical systems setting, leading to subpar inference and ad-hoc code that may be difficult to write for practitioners. In `dynestyx`, we treat dynamical systems as first-class objects, with direct interfacing to methods like pseudo-marginal MCMC and stochastic variational inference for parameter inference.

Simultaneously, many strong solutions exist for inference in dynamical systems; modern examples include [dynamax](https://github.com/probml/dynamax) for discrete-time dynamical systems, [cd-dynamax](https://github.com/hd-UQ/cd_dynamax) for continuous-time dynamical systems, and [PFJax](https://pfjax.readthedocs.io/en/latest/) for nonlinear and non-Gaussian discrete-time dynamical systems. While featureful, one drawback of this suite of methods is a varied set of APIs, with model code that is tightly coupled with the resulting inference method. In `dynestyx`, we offer a large variety of different inference methods under the same roof in a unified, abstract API. Iterating and selecting the appropriate inference methods is thus a significantly simpler process. Using tools from PPLs, we are also able to introspectively analyze a given model, and select appropriate inference methods which take advantage of model structure (e.g., linearity or Gaussianity).

## Installation

For installation, we recommend [`uv`](https://docs.astral.sh/uv/):
```bash
uv pip install dynestyx
```

But `pip` works as well:
```bash
pip install dynestyx
```

> **Developers**: See [Contributing Guidelines](CONTRIBUTING.md) for the development setup using `uv sync`.

## Quickstart

We provide a more mathematical introduction in the [Introduction](docs/math_intro.md) section. For a hands-on tutorial with code examples, check out the [Quickstart Tutorial](docs/tutorials/quickstart.ipynb).

## Contributing

Contributions are welcome. See [Contributing Guidelines](CONTRIBUTING.md) for development setup, testing expectations, and the pull request workflow.
