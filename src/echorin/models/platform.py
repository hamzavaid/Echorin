"""Timestamped world-frame platform state and body-frame sensor mounts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin

import numpy as np
from numpy.typing import NDArray

from echorin.models.geometry import SensorPose


def _vector2(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 2D vector")
    return vector.copy()


@dataclass(frozen=True, slots=True)
class MountTransform:
    """Fixed position and orientation of a receiver in platform body axes."""

    position_m: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    heading_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_m", _vector2(self.position_m, "mount position")
        )
        if not np.isfinite(self.heading_rad):
            raise ValueError("mount heading must be finite")


@dataclass(frozen=True, slots=True)
class PlatformState:
    """Public moving sensor platform state in east/north world coordinates."""

    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    acceleration_mps2: NDArray[np.float64]
    heading_rad: float
    angular_velocity_rad_s: float
    timestamp_s: float

    def __post_init__(self) -> None:
        for name in ("position_m", "velocity_mps", "acceleration_mps2"):
            object.__setattr__(self, name, _vector2(getattr(self, name), name))
        if not all(
            np.isfinite(value)
            for value in (
                self.heading_rad,
                self.angular_velocity_rad_s,
                self.timestamp_s,
            )
        ):
            raise ValueError("platform orientation, turn rate, and time must be finite")

    def sensor_pose(self, mount: MountTransform | None = None) -> SensorPose:
        """Rotate/translate a body mount, including omega cross offset velocity."""
        mount = mount or MountTransform()
        c, s = cos(self.heading_rad), sin(self.heading_rad)
        rotation = np.array([[c, -s], [s, c]])
        offset = rotation @ mount.position_m
        position = self.position_m + offset
        velocity = self.velocity_mps + self.angular_velocity_rad_s * np.array(
            [-offset[1], offset[0]]
        )
        return SensorPose(
            float(position[0]),
            float(position[1]),
            self.heading_rad + mount.heading_rad,
            float(velocity[0]),
            float(velocity[1]),
            self.timestamp_s,
        )
