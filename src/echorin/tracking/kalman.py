"""Linear constant-velocity Kalman filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConstantVelocityKalmanFilter:
    """Four-state ``[x, y, vx, vy]`` linear Kalman filter."""

    def __init__(
        self,
        state: ArrayLike,
        covariance: ArrayLike,
        process_acceleration_std_mps2: float = 2.0,
        measurement_std_m: float = 20.0,
    ) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.covariance = np.asarray(covariance, dtype=np.float64).copy()
        if self.state.shape != (4,):
            raise ValueError("state must have shape (4,)")
        if self.covariance.shape != (4, 4):
            raise ValueError("covariance must have shape (4, 4)")
        if process_acceleration_std_mps2 < 0.0:
            raise ValueError(
                "process acceleration standard deviation must be nonnegative"
            )
        if measurement_std_m <= 0.0:
            raise ValueError("measurement_std_m must be positive")
        self.process_acceleration_std_mps2 = process_acceleration_std_mps2
        self.measurement_std_m = measurement_std_m

    @classmethod
    def from_position(
        cls,
        position_m: ArrayLike,
        position_std_m: float,
        velocity_std_mps: float,
        process_acceleration_std_mps2: float = 2.0,
        measurement_std_m: float = 20.0,
    ) -> ConstantVelocityKalmanFilter:
        """Initialize position from a measurement and velocity at zero."""
        position = np.asarray(position_m, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError("position_m must have shape (2,)")
        if position_std_m <= 0.0 or velocity_std_mps <= 0.0:
            raise ValueError("initial standard deviations must be positive")
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
            state,
            covariance,
            process_acceleration_std_mps2,
            measurement_std_m,
        )

    @staticmethod
    def transition_matrix(dt_s: float) -> NDArray[np.float64]:
        """Return the exact constant-velocity state transition matrix."""
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        return np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def process_covariance(self, dt_s: float) -> NDArray[np.float64]:
        """Discretized white-acceleration process covariance."""
        variance = self.process_acceleration_std_mps2**2
        dt2 = dt_s**2
        dt3 = dt_s**3
        dt4 = dt_s**4
        return variance * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    @property
    def measurement_covariance(self) -> NDArray[np.float64]:
        return np.eye(2, dtype=np.float64) * self.measurement_std_m**2

    def predict(self, dt_s: float) -> NDArray[np.float64]:
        """Advance state and covariance without a measurement."""
        transition = self.transition_matrix(dt_s)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + self.process_covariance(dt_s)
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return self.state

    def innovation(
        self, measurement_m: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return position innovation and its covariance."""
        measurement = np.asarray(measurement_m, dtype=np.float64)
        if measurement.shape != (2,):
            raise ValueError("measurement_m must have shape (2,)")
        residual = measurement - self.state[:2]
        innovation_covariance = self.covariance[:2, :2] + self.measurement_covariance
        return residual, innovation_covariance

    def update(self, measurement_m: ArrayLike) -> NDArray[np.float64]:
        """Correct state using a Cartesian position measurement."""
        residual, innovation_covariance = self.innovation(measurement_m)
        measurement_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        gain = np.linalg.solve(
            innovation_covariance, measurement_matrix @ self.covariance
        ).T
        self.state = self.state + gain @ residual
        identity = np.eye(4, dtype=np.float64)
        residual_transform = identity - gain @ measurement_matrix
        self.covariance = (
            residual_transform @ self.covariance @ residual_transform.T
            + gain @ self.measurement_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return self.state
