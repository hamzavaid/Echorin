"""Deterministic 2D sensor-platform motion models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import atan2, cos, sin

import numpy as np
from numpy.typing import NDArray

from echorin.models.platform import PlatformState


class TrajectoryKind(StrEnum):
    """Supported v1.3 platform motion laws."""

    STATIONARY = "stationary"
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    COORDINATED_TURN = "coordinated_turn"
    WAYPOINT = "waypoint"


@dataclass(frozen=True, slots=True)
class Waypoint:
    """World position reached at an absolute scenario time."""

    timestamp_s: float
    position_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if (
            not np.isfinite(self.timestamp_s)
            or position.shape != (2,)
            or not np.all(np.isfinite(position))
        ):
            raise ValueError("waypoint time and position must be finite 2D values")
        object.__setattr__(self, "position_m", position.copy())


@dataclass(frozen=True, slots=True)
class PlatformTrajectory:
    """A validated motion law evaluated from the current platform state."""

    kind: TrajectoryKind = TrajectoryKind.STATIONARY
    waypoints: tuple[Waypoint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TrajectoryKind(self.kind))
        if self.kind is TrajectoryKind.WAYPOINT:
            if len(self.waypoints) < 2:
                raise ValueError("waypoint trajectory needs at least two points")
            times = [point.timestamp_s for point in self.waypoints]
            if any(
                right <= left for left, right in zip(times, times[1:], strict=False)
            ):
                raise ValueError("waypoint timestamps must strictly increase")
        elif self.waypoints:
            raise ValueError("waypoints are only valid for waypoint trajectories")

    def advance(self, state: PlatformState, dt_s: float) -> PlatformState:
        """Advance exactly by positive simulation time under the selected law."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        time_s = state.timestamp_s + dt_s
        if self.kind is TrajectoryKind.STATIONARY:
            return replace(
                state,
                velocity_mps=np.zeros(2),
                acceleration_mps2=np.zeros(2),
                angular_velocity_rad_s=0.0,
                timestamp_s=time_s,
            )
        if self.kind is TrajectoryKind.WAYPOINT:
            return self._at_waypoint_time(state, time_s)
        heading = state.heading_rad + state.angular_velocity_rad_s * dt_s
        if self.kind is TrajectoryKind.CONSTANT_VELOCITY:
            return replace(
                state,
                position_m=state.position_m + state.velocity_mps * dt_s,
                heading_rad=heading,
                timestamp_s=time_s,
            )
        if self.kind is TrajectoryKind.CONSTANT_ACCELERATION:
            return replace(
                state,
                position_m=state.position_m
                + state.velocity_mps * dt_s
                + 0.5 * state.acceleration_mps2 * dt_s**2,
                velocity_mps=state.velocity_mps + state.acceleration_mps2 * dt_s,
                heading_rad=heading,
                timestamp_s=time_s,
            )
        omega = state.angular_velocity_rad_s
        if abs(omega) < 1e-12:
            displacement = state.velocity_mps * dt_s
            velocity = state.velocity_mps
        else:
            turn = omega * dt_s
            rotation = np.array([[cos(turn), -sin(turn)], [sin(turn), cos(turn)]])
            displacement = np.array(
                [
                    (
                        sin(turn) * state.velocity_mps[0]
                        - (1 - cos(turn)) * state.velocity_mps[1]
                    )
                    / omega,
                    (
                        (1 - cos(turn)) * state.velocity_mps[0]
                        + sin(turn) * state.velocity_mps[1]
                    )
                    / omega,
                ]
            )
            velocity = rotation @ state.velocity_mps
        return replace(
            state,
            position_m=state.position_m + displacement,
            velocity_mps=velocity,
            heading_rad=heading,
            timestamp_s=time_s,
        )

    def _at_waypoint_time(self, state: PlatformState, time_s: float) -> PlatformState:
        first, last = self.waypoints[0], self.waypoints[-1]
        if time_s <= first.timestamp_s:
            return replace(
                state,
                position_m=first.position_m,
                velocity_mps=np.zeros(2),
                timestamp_s=time_s,
            )
        if time_s >= last.timestamp_s:
            return replace(
                state,
                position_m=last.position_m,
                velocity_mps=np.zeros(2),
                timestamp_s=time_s,
            )
        for left, right in zip(self.waypoints, self.waypoints[1:], strict=False):
            if left.timestamp_s <= time_s < right.timestamp_s:
                duration = right.timestamp_s - left.timestamp_s
                fraction = (time_s - left.timestamp_s) / duration
                velocity = (right.position_m - left.position_m) / duration
                return replace(
                    state,
                    position_m=left.position_m
                    + fraction * (right.position_m - left.position_m),
                    velocity_mps=velocity,
                    acceleration_mps2=np.zeros(2),
                    heading_rad=atan2(velocity[1], velocity[0]),
                    angular_velocity_rad_s=0.0,
                    timestamp_s=time_s,
                )
        raise RuntimeError("waypoint segment could not be located")
