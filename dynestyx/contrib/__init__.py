"""Experimental, reusable utilities built on top of dynestyx.

Code here is supported but not part of the stable core API; modules may move or
change. Currently provides held-out one-step-ahead predictive evaluation for
factorial state-space models (:mod:`dynestyx.contrib.factorial_predictive`), the
standard metric for online skill-rating benchmarks (Duffield et al., 2024).
"""

from dynestyx.contrib.factorial_predictive import (
    one_step_ahead_local_states,
    outcome_predictive_probs,
    score_predictive,
    split_nll,
)

__all__ = [
    "one_step_ahead_local_states",
    "outcome_predictive_probs",
    "score_predictive",
    "split_nll",
]
