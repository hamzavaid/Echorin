"""Persistent target-track state and lifecycle models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class TrackStatus(StrEnum):
    """Lifecycle state of a sensor-derived track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass(slots=True)
class Track:
    """Persistent Cartesian state estimate with no ground-truth identity."""

    track_id: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 1
    last_timestamp_s: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float64).copy()
        self.covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if self.state.shape != (4,):
            raise ValueError("track state must have shape (4,)")
        if self.covariance.shape != (4, 4):
            raise ValueError("track covariance must have shape (4, 4)")
        if self.track_id < 1:
            raise ValueError("track_id must be positive")
        if not self.history:
            self.history.append((self.x_m, self.y_m))

    @classmethod
    def initial(
        cls,
        track_id: int,
        position_m: NDArray[np.float64],
        timestamp_s: float,
        position_std_m: float = 10.0,
        velocity_std_mps: float = 100.0,
    ) -> Track:
        """Initialize a tentative track from one Cartesian measurement."""
        position = np.asarray(position_m, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError("position_m must have shape (2,)")
        state = np.array([position[0], position[1], 0.0, 0.0], dtype=np.float64)
        covariance = np.diag(
            [
                position_std_m**2,
                position_std_m**2,
                velocity_std_mps**2,
                velocity_std_mps**2,
            ]
        )
        return cls(
            track_id=track_id,
            state=state,
            covariance=covariance,
            last_timestamp_s=timestamp_s,
        )

    @property
    def x_m(self) -> float:
        return float(self.state[0])

    @property
    def y_m(self) -> float:
        return float(self.state[1])

    @property
    def vx_mps(self) -> float:
        return float(self.state[2])

    @property
    def vy_mps(self) -> float:
        return float(self.state[3])

    def append_history(self, maximum_length: int) -> None:
        """Append the current position while bounding display memory."""
        self.history.append((self.x_m, self.y_m))
        if len(self.history) > maximum_length:
            del self.history[: len(self.history) - maximum_length]
