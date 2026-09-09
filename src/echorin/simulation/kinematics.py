"""Analytical target and sensor-relative kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from typing import Protocol

from echorin.models import SensorPose


class CartesianTarget(Protocol):
    """Minimum target state required for sensor-relative geometry."""

    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float


@dataclass(frozen=True, slots=True)
class TargetGeometry:
    """Exact simulation geometry relative to a stationary sensor."""

    range_m: float
    bearing_rad: float
    radial_velocity_mps: float


def relative_geometry(sensor: SensorPose, target: CartesianTarget) -> TargetGeometry:
    """Calculate range, absolute bearing, and radial velocity.

    This function belongs to the ground-truth/synthesis boundary. Detectors and
    later trackers must consume observations rather than this object.
    """
    dx = target.x_m - sensor.x_m
    dy = target.y_m - sensor.y_m
    range_m = hypot(dx, dy)
    radial_velocity = (
        (dx * target.vx_mps + dy * target.vy_mps) / range_m if range_m > 0.0 else 0.0
    )
    return TargetGeometry(
        range_m=range_m,
        bearing_rad=atan2(dy, dx),
        radial_velocity_mps=radial_velocity,
    )
