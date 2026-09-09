"""Simplified monostatic propagation and echo placement."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from echorin.config import SensorConfig


def round_trip_delay_s(range_m: float, propagation_speed_mps: float) -> float:
    """Return monostatic outbound-and-return propagation time."""
    if range_m < 0.0:
        raise ValueError("range_m must be nonnegative")
    if propagation_speed_mps <= 0.0:
        raise ValueError("propagation_speed_mps must be positive")
    return 2.0 * range_m / propagation_speed_mps


def sample_delay(range_m: float, config: SensorConfig) -> int:
    """Quantize round-trip delay to the nearest acquisition sample."""
    return int(
        round(
            round_trip_delay_s(range_m, config.propagation_speed_mps)
            * config.sample_rate_hz
        )
    )


def amplitude_at_range(
    range_m: float,
    reflectivity: float = 1.0,
    reference_range_m: float = 1_000.0,
    exponent: float = 2.0,
) -> float:
    """Apply a documented simplified inverse-power amplitude model.

    Amplitude is capped inside the reference range, avoiding a singularity at
    the sensor. This is deliberately simpler than the radar equation.
    """
    if range_m < 0.0:
        raise ValueError("range_m must be nonnegative")
    if reflectivity < 0.0:
        raise ValueError("reflectivity must be nonnegative")
    if reference_range_m <= 0.0:
        raise ValueError("reference_range_m must be positive")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    effective_range = max(range_m, reference_range_m)
    return reflectivity * (reference_range_m / effective_range) ** exponent


def delayed_echo(
    transmitted_signal: ArrayLike,
    range_m: float,
    config: SensorConfig,
    reflectivity: float = 1.0,
    attenuation_exponent: float = 2.0,
) -> NDArray[np.float64]:
    """Place one attenuated echo into a zero-filled acquisition window."""
    if range_m > config.max_range_m:
        raise ValueError("range_m exceeds configured max_range_m")
    transmitted = np.asarray(transmitted_signal, dtype=np.float64)
    if transmitted.ndim != 1:
        raise ValueError("transmitted_signal must be one-dimensional")
    delay = sample_delay(range_m, config)
    end = delay + transmitted.size
    if end > config.acquisition_samples:
        raise ValueError("echo does not fit inside acquisition window")
    received = np.zeros(config.acquisition_samples, dtype=np.float64)
    amplitude = amplitude_at_range(
        range_m,
        reflectivity=reflectivity,
        exponent=attenuation_exponent,
    )
    received[delay:end] = transmitted * amplitude
    return received
