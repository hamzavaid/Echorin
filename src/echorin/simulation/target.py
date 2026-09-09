"""Ground-truth target state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Target:
    """A point target following a constant Cartesian velocity model."""

    target_id: str
    x_m: float
    y_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    reflectivity: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must be nonempty")
        if self.reflectivity < 0.0:
            raise ValueError("reflectivity must be nonnegative")

    @property
    def state(self) -> NDArray[np.float64]:
        """Return ``[x, y, vx, vy]`` as a detached numerical state vector."""
        return np.array(
            [self.x_m, self.y_m, self.vx_mps, self.vy_mps], dtype=np.float64
        )

    def advance(self, dt_s: float) -> None:
        """Advance position by one exact constant-velocity step."""
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.x_m += self.vx_mps * dt_s
        self.y_m += self.vy_mps * dt_s
