"""Persistent world-coordinate scatterer field (synthesis boundary only)."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from echorin.environment.clutter import scatterers
from echorin.environment.config import EnvironmentConfig
from echorin.models.geometry import SensorPose


@dataclass(frozen=True, slots=True)
class FieldReturn:
    """Internal reflectivity source without any target identity."""

    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    reflectivity: float
    phase_rad: float

    def at(self, elapsed_s: float) -> FieldReturn:
        return FieldReturn(
            self.x_m + elapsed_s * self.vx_mps,
            self.y_m + elapsed_s * self.vy_mps,
            self.vx_mps,
            self.vy_mps,
            self.reflectivity,
            self.phase_rad,
        )


def build_field(
    config: EnvironmentConfig,
    max_range_m: float,
    pose: SensorPose,
    rng: np.random.Generator,
) -> tuple[FieldReturn, ...]:
    """Anchor seeded scatterers to the initial receiver position and heading."""
    result = []
    for model in config.clutter:
        for point in scatterers(model, max_range_m, rng):
            angle = point.bearing_rad + pose.heading_rad
            result.append(
                FieldReturn(
                    pose.x_m + point.range_m * cos(angle),
                    pose.y_m + point.range_m * sin(angle),
                    point.radial_velocity_mps * cos(angle),
                    point.radial_velocity_mps * sin(angle),
                    point.reflectivity,
                    point.phase_rad,
                )
            )
    return tuple(result)
