"""Geometry models with no simulation or GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensorPose:
    """Cartesian sensor position and boresight heading."""

    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0
