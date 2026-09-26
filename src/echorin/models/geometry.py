"""Geometry models with no simulation or GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensorPose:
    """Timestamped receiver pose and velocity in world coordinates."""

    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    timestamp_s: float = 0.0
