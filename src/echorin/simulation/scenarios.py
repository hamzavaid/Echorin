"""Repeatable scenario presets."""

from __future__ import annotations

from math import cos, sin

import numpy as np

from echorin.config import SensorConfig
from echorin.models.platform import PlatformState
from echorin.sensors.components import Emitter, Receiver, SensorPlatform
from echorin.simulation.target import Target
from echorin.simulation.trajectories import PlatformTrajectory, TrajectoryKind
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


def bistatic_scenario(config: SensorConfig, *, multistatic: bool = False) -> World:
    """Moving synchronized TX/RX reference, scaled to the selected medium."""
    scale = config.max_range_m / 10
    speed = config.propagation_speed_mps / config.carrier_frequency_hz * config.prf_hz
    speed = min(speed, 7500.0)  # Cap default Radar target motion at 300 m/s.
    trajectory = PlatformTrajectory(TrajectoryKind.CONSTANT_VELOCITY)

    def state(x, y, vx, vy):
        return PlatformState(np.array([x, y]), np.array([vx, vy]), np.zeros(2), 0, 0, 0)

    platforms = [
        SensorPlatform(
            "transmitter",
            state(-scale, 0, 0.01 * speed, 0),
            (Emitter("tx-1", config),),
            trajectory=trajectory,
        ),
        SensorPlatform(
            "receiver",
            state(0, 0, 0, 0.005 * speed),
            receivers=(Receiver("rx-1", config),),
            trajectory=trajectory,
        ),
    ]
    if multistatic:
        platforms.extend(
            [
                SensorPlatform(
                    "transmitter-2",
                    state(-scale, -scale, 0.008 * speed, 0),
                    (Emitter("tx-2", config),),
                    trajectory=trajectory,
                ),
                SensorPlatform(
                    "receiver-2",
                    state(0, -scale, 0.004 * speed, 0),
                    receivers=(Receiver("rx-2", config),),
                    trajectory=trajectory,
                ),
            ]
        )
    return World(
        targets=(Target("reference", 3 * scale, scale, vx_mps=0.04 * speed),),
        sensor_platforms=platforms,
    )
