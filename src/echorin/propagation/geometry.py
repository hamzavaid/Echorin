"""Centralized narrowband TX-target-RX geometry (positive increasing path)."""

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np
from numpy.typing import ArrayLike

from echorin.models.geometry import SensorPose


@dataclass(frozen=True, slots=True)
class PathGeometry:
    """Synthesis-only kinematics; never a detector input."""

    transmitter_range_m: float
    receiver_range_m: float
    path_length_m: float
    delay_s: float
    path_rate_mps: float
    doppler_hz: float
    transmitter_bearing_rad: float
    receiver_bearing_rad: float


def path_geometry(
    transmitter: SensorPose,
    receiver: SensorPose,
    target_position_m: ArrayLike,
    target_velocity_mps: ArrayLike,
    propagation_speed_mps: float,
    carrier_frequency_hz: float,
) -> PathGeometry:
    """Evaluate both moving legs, held fixed geometrically during a CPI.

    Positive Doppler means increasing total path length, matching legacy
    Echorin. No absolute carrier phase is modeled (legacy coherent reference).
    """
    p, v = np.asarray(target_position_m, float), np.asarray(target_velocity_mps, float)
    scalars = (propagation_speed_mps, carrier_frequency_hz)
    if (
        p.shape != (2,)
        or v.shape != (2,)
        or not np.all(np.isfinite([p, v]))
        or not np.all(np.isfinite(scalars))
        or min(scalars) <= 0
    ):
        raise ValueError("finite 2D states and positive speed/carrier required")
    legs, rates, bearings = [], [], []
    for pose in (transmitter, receiver):
        r = p - [pose.x_m, pose.y_m]
        distance = float(np.linalg.norm(r))
        if distance <= 0 or not np.isfinite(distance):
            raise ValueError("target must not coincide with transmitter or receiver")
        legs.append(distance)
        rates.append(float((v - [pose.vx_mps, pose.vy_mps]) @ (r / distance)))
        bearings.append(atan2(r[1], r[0]))
    length, rate = sum(legs), sum(rates)
    return PathGeometry(
        *legs,
        length,
        length / propagation_speed_mps,
        rate,
        rate * carrier_frequency_hz / propagation_speed_mps,
        *bearings,
    )


def receiver_range_from_path(
    path_length_m: float,
    receiver_bearing_rad: float,
    transmitter: SensorPose,
    receiver: SensorPose,
) -> float:
    """Intersect a measured arrival ray with a measured bistatic ellipse.

    Bearing is world-frame. Only known device poses and measured path/bearing
    are used. Baseline/forward-scatter degeneracy and impossible paths are
    rejected rather than reporting a fabricated Cartesian measurement.
    """
    length = path_length_m
    if not np.all(np.isfinite([length, receiver_bearing_rad])) or length <= 0:
        raise ValueError("positive finite path and finite bearing required")
    baseline = np.array(
        [transmitter.x_m - receiver.x_m, transmitter.y_m - receiver.y_m]
    )
    distance = float(np.linalg.norm(baseline))
    direction = np.array([cos(receiver_bearing_rad), sin(receiver_bearing_rad)])
    denominator = 2 * (length - float(baseline @ direction))
    if length <= distance or denominator <= 1e-12 * max(length, 1):
        raise ValueError("impossible or degenerate bistatic path measurement")
    return float((length**2 - distance**2) / denominator)
