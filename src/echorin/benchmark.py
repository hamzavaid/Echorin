"""Deterministic numerical benchmark scenarios for release validation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from echorin.tracking.kalman import ConstantVelocityKalmanFilter


@dataclass(frozen=True, slots=True)
class TrackingBenchmarkResult:
    """Truth, measurement, estimate, error, and timing benchmark outputs."""

    truth_positions_m: NDArray[np.float64]
    measured_positions_m: NDArray[np.float64]
    filtered_positions_m: NDArray[np.float64]
    raw_position_rmse_m: float
    filtered_position_rmse_m: float
    average_processing_time_ms: float

    def summary(self) -> dict[str, float | int]:
        """Return JSON-ready scalar benchmark metrics."""
        return {
            "sample_count": int(self.truth_positions_m.shape[0]),
            "raw_position_rmse_m": self.raw_position_rmse_m,
            "filtered_position_rmse_m": self.filtered_position_rmse_m,
            "average_processing_time_ms": self.average_processing_time_ms,
        }


def run_tracking_benchmark(
    seed: int = 7,
    sample_count: int = 200,
    dt_s: float = 0.1,
    measurement_std_m: float = 20.0,
) -> TrackingBenchmarkResult:
    """Compare noisy measurements and Kalman estimates to hidden truth."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    if dt_s <= 0.0 or measurement_std_m <= 0.0:
        raise ValueError("dt_s and measurement_std_m must be positive")
    rng = np.random.default_rng(seed)
    times_s = np.arange(sample_count, dtype=np.float64) * dt_s
    truth = np.column_stack((500.0 + 18.0 * times_s, -250.0 + 7.5 * times_s))
    measurements = truth + rng.normal(0.0, measurement_std_m, size=truth.shape)
    filter_ = ConstantVelocityKalmanFilter.from_position(
        measurements[0],
        position_std_m=measurement_std_m,
        velocity_std_mps=50.0,
        process_acceleration_std_mps2=1.0,
        measurement_std_m=measurement_std_m,
    )
    estimates = [filter_.state[:2].copy()]
    started = perf_counter()
    for measurement in measurements[1:]:
        filter_.predict(dt_s)
        filter_.update(measurement)
        estimates.append(filter_.state[:2].copy())
    elapsed_s = perf_counter() - started
    filtered = np.asarray(estimates, dtype=np.float64)

    def position_rmse(values: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(np.sum((values - truth) ** 2, axis=1))))

    return TrackingBenchmarkResult(
        truth_positions_m=truth,
        measured_positions_m=measurements,
        filtered_positions_m=filtered,
        raw_position_rmse_m=position_rmse(measurements),
        filtered_position_rmse_m=position_rmse(filtered),
        average_processing_time_ms=elapsed_s * 1_000.0 / (sample_count - 1),
    )
