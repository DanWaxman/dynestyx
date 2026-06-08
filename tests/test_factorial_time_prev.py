"""Test the per-factor previous-time bookkeeping (Duffield et al. §3.4.3)."""

import jax
import jax.numpy as jnp

from dynestyx.inference.integrations.cuthbert.factorial_filter import (
    _compute_time_prev_and_last,
)

jax.config.update("jax_enable_x64", True)


def test_time_prev_hand_checked():
    # 4 factors; matches: [0,1]@1, [0,2]@2, [1,2]@3; t0 = 0.
    obs_times = jnp.array([1.0, 2.0, 3.0])
    obs_idx = jnp.array([[0, 1], [0, 2], [1, 2]])
    time_prev, last_time = _compute_time_prev_and_last(
        obs_times, obs_idx, jnp.array(0.0), num_factors=4
    )
    expected_time_prev = jnp.array(
        [
            [0.0, 0.0],  # match 0: both teams first appearance -> t0
            [1.0, 0.0],  # match 1: team 0 last @1, team 2 first -> t0
            [1.0, 2.0],  # match 2: team 1 last @1, team 2 last @2
        ]
    )
    assert jnp.allclose(time_prev, expected_time_prev)
    # last_time: team0 @2, team1 @3, team2 @3, team3 never -> t0=0
    assert jnp.allclose(last_time, jnp.array([2.0, 3.0, 3.0, 0.0]))


def test_time_prev_same_match_reads_pre_match_times():
    """Two factors meeting again read each other's pre-match times, not the current."""
    obs_times = jnp.array([1.0, 5.0])
    obs_idx = jnp.array([[0, 1], [0, 1]])
    time_prev, _ = _compute_time_prev_and_last(
        obs_times, obs_idx, jnp.array(0.0), num_factors=2
    )
    assert jnp.allclose(time_prev, jnp.array([[0.0, 0.0], [1.0, 1.0]]))
