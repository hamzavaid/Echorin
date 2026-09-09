"""Sensor abstraction that publishes observations without truth metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ReflectiveTarget(Protocol):
    """Ground-truth properties required only for echo synthesis."""

    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    reflectivity: float


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Raw signals published from the sensing boundary."""

    timestamp_s: float
    transmitted_signal: NDArray[np.float64]
    received_signal: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DirectionalSensorFrame(SensorFrame):
    """Raw signal from one measured angular channel, without truth metadata."""

    bearing_rad: float


class Sensor(ABC):
    """Abstract sensor-to-signal interface."""

    @abstractmethod
    def acquire(
        self, targets: Iterable[ReflectiveTarget], timestamp_s: float
    ) -> SensorFrame:
        """Synthesize one observation frame from ground-truth targets."""
