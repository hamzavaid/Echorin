"""Target collection, clock, and reset semantics for a scenario."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from math import degrees

import numpy as np

from echorin.config import ArrayConfig
from echorin.models import SensorPose
from echorin.models.platform import MountTransform, PlatformState
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.target import Target
from echorin.simulation.trajectories import PlatformTrajectory


class World:
    """Own mutable ground-truth state for one repeatable scenario."""

    def __init__(
        self,
        targets: Iterable[Target] = (),
        sensor_pose: SensorPose | None = None,
        platform_state: PlatformState | None = None,
        platform_trajectory: PlatformTrajectory | None = None,
        sensor_mount: MountTransform | None = None,
        array_config: ArrayConfig | None = None,
    ) -> None:
        if sensor_pose is not None and platform_state is not None:
            raise ValueError("provide either sensor_pose or platform_state")
        pose = sensor_pose or SensorPose()
        self.platform_state = platform_state or PlatformState(
            np.array([pose.x_m, pose.y_m]),
            np.array([pose.vx_mps, pose.vy_mps]),
            np.zeros(2),
            pose.heading_rad,
            0.0,
            pose.timestamp_s,
        )
        self.platform_trajectory = platform_trajectory or PlatformTrajectory()
        self.sensor_mount = sensor_mount or MountTransform()
        self.array_config = array_config or ArrayConfig()
        self._targets: dict[str, Target] = {}
        self.time_s = self.platform_state.timestamp_s
        self._platform_history: list[tuple[float, float]] = [
            tuple(float(x) for x in self.platform_state.position_m)
        ]
        for target in targets:
            self.add_target(target)
        self._initial_targets = deepcopy(self._targets)
        self._initial_platform_state = deepcopy(self.platform_state)

    @property
    def sensor_pose(self) -> SensorPose:
        """Current receiver pose with body mount and rotational velocity."""
        return self.platform_state.sensor_pose(self.sensor_mount)

    @property
    def platform_history(self) -> tuple[tuple[float, float], ...]:
        """Bounded world positions for the optional platform PPI trail."""
        return tuple(self._platform_history)

    def set_platform(
        self,
        state: PlatformState,
        trajectory: PlatformTrajectory | None = None,
        mount: MountTransform | None = None,
        array_config: ArrayConfig | None = None,
    ) -> None:
        """Edit the scenario platform at the current simulation timestamp."""
        if state.timestamp_s != self.time_s:
            raise ValueError("platform timestamp must match world time")
        self.platform_state = state
        if trajectory is not None:
            self.platform_trajectory = trajectory
        if mount is not None:
            self.sensor_mount = mount
        if array_config is not None:
            self.array_config = array_config
        self._platform_history[-1] = tuple(float(x) for x in state.position_m)

    @property
    def targets(self) -> tuple[Target, ...]:
        """Targets in stable insertion order, exposed without the backing map."""
        return tuple(self._targets.values())

    def add_target(self, target: Target) -> None:
        """Add a uniquely identified target."""
        if target.target_id in self._targets:
            raise ValueError(f"duplicate target_id: {target.target_id}")
        self._targets[target.target_id] = target

    def get_target(self, target_id: str) -> Target:
        """Return a target by simulation identity."""
        return self._targets[target_id]

    def remove_target(self, target_id: str) -> Target:
        """Remove and return a target by simulation identity."""
        return self._targets.pop(target_id)

    def replace_target(self, target_id: str, replacement: Target) -> None:
        """Replace one target while enforcing identity uniqueness."""
        if target_id not in self._targets:
            raise KeyError(target_id)
        if (
            replacement.target_id != target_id
            and replacement.target_id in self._targets
        ):
            raise ValueError(f"duplicate target_id: {replacement.target_id}")
        items = list(self._targets.items())
        self._targets = {
            (replacement.target_id if key == target_id else key): (
                replacement if key == target_id else value
            )
            for key, value in items
        }

    def advance(self, dt_s: float) -> None:
        """Advance every target and the world clock atomically."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        next_platform = self.platform_trajectory.advance(self.platform_state, dt_s)
        for target in self._targets.values():
            target.advance(dt_s)
        self.platform_state = next_platform
        self.time_s += dt_s
        self._platform_history.append(
            tuple(float(x) for x in self.platform_state.position_m)
        )
        if len(self._platform_history) > 2000:
            self._platform_history.pop(0)

    def checkpoint_reset_state(self) -> None:
        """Make the current edited scenario the new reset baseline."""
        self._initial_targets = deepcopy(self._targets)
        self.platform_state = replace(self.platform_state, timestamp_s=0.0)
        self._initial_platform_state = deepcopy(self.platform_state)
        self.time_s = 0.0
        self._platform_history = [
            tuple(float(x) for x in self.platform_state.position_m)
        ]

    def reset(self) -> None:
        """Restore a deep copy of the scenario's reset baseline."""
        self._targets = deepcopy(self._initial_targets)
        self.platform_state = deepcopy(self._initial_platform_state)
        self.time_s = self.platform_state.timestamp_s
        self._platform_history = [
            tuple(float(x) for x in self.platform_state.position_m)
        ]

    def console_frames(self, steps: int, dt_s: float) -> list[str]:
        """Advance and format deterministic geometry frames for console use."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        lines: list[str] = []
        for _ in range(steps):
            self.advance(dt_s)
            parts = [f"t={self.time_s:.3f}s"]
            for target in self.targets:
                geometry = relative_geometry(self.sensor_pose, target)
                parts.append(
                    f"{target.target_id}: range={geometry.range_m:.3f}m "
                    f"bearing={degrees(geometry.bearing_rad):.3f}deg"
                )
            lines.append(" | ".join(parts))
        return lines
