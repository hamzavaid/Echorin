"""Stateless simulation-clock scan scheduling."""

from dataclasses import dataclass
from math import isfinite, pi
from typing import Literal


@dataclass(frozen=True, slots=True)
class ScanConfig:
    kind: Literal["fixed", "rotation", "sector", "step"] = "fixed"
    boresight_rad: float = 0.0
    rate_rad_s: float = pi / 6
    start_rad: float = -pi / 3
    stop_rad: float = pi / 3
    ping_pong: bool = True
    dwell_s: float = 0.5
    step_angles_rad: tuple[float, ...] = (-pi / 3, 0.0, pi / 3)

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "rotation", "sector", "step"):
            raise ValueError("unknown scan kind")
        for value in (
            self.boresight_rad,
            self.rate_rad_s,
            self.start_rad,
            self.stop_rad,
            self.dwell_s,
            *self.step_angles_rad,
        ):
            if not isfinite(value):
                raise ValueError("scan parameters must be finite")
        if self.stop_rad <= self.start_rad or self.stop_rad - self.start_rad > 2 * pi:
            raise ValueError("scan sector must increase and span no more than 2pi")
        if self.dwell_s <= 0 or not self.step_angles_rad or self.rate_rad_s == 0:
            raise ValueError("scan requires positive dwell, steps and nonzero rate")
        object.__setattr__(self, "step_angles_rad", tuple(self.step_angles_rad))


@dataclass(frozen=True, slots=True)
class ScanScheduler:
    config: ScanConfig = ScanConfig()

    def boresight(self, timestamp_s: float) -> float:
        """Compute receiver-local boresight (CPI uses one dwell snapshot)."""
        if not isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("scan time must be finite and nonnegative")
        c = self.config
        if c.kind == "fixed":
            return c.boresight_rad
        if c.kind == "rotation":
            angle = c.boresight_rad + c.rate_rad_s * timestamp_s
            return (angle + pi) % (2 * pi) - pi
        if c.kind == "step":
            return c.step_angles_rad[
                int(timestamp_s / c.dwell_s) % len(c.step_angles_rad)
            ]
        span = c.stop_rad - c.start_rad
        progress = abs(c.rate_rad_s) * timestamp_s
        if c.ping_pong:
            position = progress % (2 * span)
            position = position if position <= span else 2 * span - position
        else:
            position = progress % span
        return c.start_rad + position if c.rate_rad_s > 0 else c.stop_rad - position
