"""Sensor-derived detection model."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from echorin.models.geometry import SensorPose


@dataclass(frozen=True, slots=True)
class Detection:
    """A target-like measurement derived entirely from processed sensor data."""

    timestamp_s: float
    range_m: float
    bearing_rad: float
    radial_velocity_mps: float | None
    amplitude: float
    snr_db: float
    confidence: float
    source_bin: int
    source_angle_bin: int | None = None
    receiver_id: str = "receiver-0"
    sensor_pose: SensorPose | None = None

    def __post_init__(self) -> None:
        if self.range_m < 0.0:
            raise ValueError("range_m must be nonnegative")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be nonnegative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.source_bin < 0:
            raise ValueError("source_bin must be nonnegative")

    @property
    def world_position_m(self) -> tuple[float, float]:
        """Transform receiver-relative range and bearing into world coordinates."""
        pose = self.sensor_pose or SensorPose()
        angle = self.bearing_rad + pose.heading_rad
        return (
            pose.x_m + self.range_m * cos(angle),
            pose.y_m + self.range_m * sin(angle),
        )
