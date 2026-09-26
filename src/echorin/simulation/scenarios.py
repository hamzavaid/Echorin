"""Repeatable scenario presets."""

from __future__ import annotations

from math import cos, sin

import numpy as np

from echorin.simulation.target import Target
from echorin.simulation.world import World


def single_stationary_target(
    range_m: float = 1_500.0, bearing_rad: float = 0.0
) -> World:
    """Create one stationary point target at known polar coordinates."""
    if range_m < 0.0:
        raise ValueError("range_m must be nonnegative")
    return World(
        targets=[
            Target(
                "target-1",
                x_m=range_m * cos(bearing_rad),
                y_m=range_m * sin(bearing_rad),
            )
        ]
    )


def crossing_targets(seed: int = 7) -> World:
    """Create two repeatable crossings within the forward array field of view.

    A linear receiver array cannot distinguish a target behind the platform
    from its front/back mirror. Keep both preset trajectories in the visible
    half-plane during their encounter so this demo exercises tracking, not
    an unobservable array ambiguity.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-100.0, 100.0, size=(2, 2))
    return World(
        targets=[
            Target(
                "crossing-a",
                x_m=1_500.0 + float(jitter[0, 0]),
                y_m=-600.0 + float(jitter[0, 1]),
                vx_mps=100.0,
                vy_mps=120.0,
                reflectivity=1.0,
            ),
            Target(
                "crossing-b",
                x_m=2_300.0 + float(jitter[1, 0]),
                y_m=600.0 + float(jitter[1, 1]),
                vx_mps=80.0,
                vy_mps=0.0,
                reflectivity=0.8,
            ),
        ]
    )
