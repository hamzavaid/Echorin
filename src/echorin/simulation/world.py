"""Target collection, clock, and reset semantics for a scenario."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from math import degrees

from echorin.models import SensorPose
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.target import Target


class World:
    """Own mutable ground-truth state for one repeatable scenario."""

    def __init__(
        self,
        targets: Iterable[Target] = (),
        sensor_pose: SensorPose | None = None,
    ) -> None:
        self.sensor_pose = sensor_pose or SensorPose()
        self._targets: dict[str, Target] = {}
        self.time_s = 0.0
        for target in targets:
            self.add_target(target)
        self._initial_targets = deepcopy(self._targets)

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
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        for target in self._targets.values():
            target.advance(dt_s)
        self.time_s += dt_s

    def checkpoint_reset_state(self) -> None:
        """Make the current edited scenario the new reset baseline."""
        self._initial_targets = deepcopy(self._targets)
        self.time_s = 0.0

    def reset(self) -> None:
        """Restore a deep copy of the scenario's reset baseline."""
        self._targets = deepcopy(self._initial_targets)
        self.time_s = 0.0

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
