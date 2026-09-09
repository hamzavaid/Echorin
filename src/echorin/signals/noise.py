"""Seedable additive white Gaussian noise."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from echorin.config import NoiseConfig


def noise_standard_deviation(signal: ArrayLike, config: NoiseConfig) -> float:
    """Resolve noise sigma from explicit standard deviation or requested SNR."""
    values = np.asarray(signal, dtype=np.float64)
    if config.snr_db is None:
        return config.standard_deviation
    signal_power = float(np.mean(np.square(values))) if values.size else 0.0
    if signal_power == 0.0:
        return config.standard_deviation
    noise_power = signal_power / (10.0 ** (config.snr_db / 10.0))
    return float(np.sqrt(noise_power))


def add_awgn(
    signal: ArrayLike,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return a copy of ``signal`` with independent Gaussian samples added."""
    values = np.asarray(signal, dtype=np.float64)
    sigma = noise_standard_deviation(values, config)
    return values + rng.normal(0.0, sigma, size=values.shape)
