"""Receiver noise models operating along the final (fast-time) axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

from echorin.config import NoiseConfig
from echorin.signals.noise import add_awgn, noise_standard_deviation

Samples = NDArray[np.float64] | NDArray[np.complex128]


class NoiseModel(Protocol):
    """A receiver disturbance driven exclusively by the supplied RNG."""

    def add(self, signal: ArrayLike, rng: np.random.Generator) -> Samples: ...


@dataclass(frozen=True, slots=True)
class AwgnNoise:
    """Adapter preserving the v1.0 real/complex AWGN and SNR contract."""

    config: NoiseConfig = field(default_factory=NoiseConfig)

    def add(self, signal: ArrayLike, rng: np.random.Generator) -> Samples:
        return add_awgn(signal, self.config, rng)


@dataclass(frozen=True, slots=True)
class ColoredNoise:
    """Stationary AR(1): PSD proportional to |1-a exp(-j omega)|^-2.

    Innovation variance is sigma²(1-a²); independently sampled stationary
    initial states avoid cold-start bias. Filtering never crosses channels.
    Negative correlation produces a high-pass spectral emphasis.
    """

    config: NoiseConfig = field(default_factory=NoiseConfig)
    correlation: float = 0.85

    def __post_init__(self) -> None:
        if not isfinite(self.correlation) or abs(self.correlation) >= 1:
            raise ValueError("correlation must be finite and strictly between -1 and 1")

    def add(self, signal: ArrayLike, rng: np.random.Generator) -> Samples:
        values = np.asarray(signal)
        if values.ndim == 0:
            raise ValueError("noise input must have a fast-time axis")
        if not values.size:
            return values.copy()
        sigma = noise_standard_deviation(values, self.config)
        white = add_awgn(np.zeros_like(values), NoiseConfig(sigma), rng)
        initial = (
            add_awgn(
                np.zeros((*values.shape[:-1], 1), dtype=values.dtype),
                NoiseConfig(sigma),
                rng,
            )
            * self.correlation
        )
        colored, _ = lfilter(
            [np.sqrt(1 - self.correlation**2)],
            [1, -self.correlation],
            white,
            axis=-1,
            zi=initial,
        )
        return np.asarray(values + colored)


@dataclass(frozen=True, slots=True)
class ImpulsiveNoise:
    """Independent Bernoulli impulses with signed/complex unit-phase amplitude."""

    config: NoiseConfig = field(default_factory=NoiseConfig)
    event_probability: float = 0.001
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.event_probability) or not 0 <= self.event_probability <= 1:
            raise ValueError("event_probability must be in [0, 1]")
        if not isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("impulse amplitude must be finite and nonnegative")

    def add(self, signal: ArrayLike, rng: np.random.Generator) -> Samples:
        result = add_awgn(signal, self.config, rng)
        events = rng.random(result.shape) < self.event_probability
        phase = (
            np.exp(2j * np.pi * rng.random(result.shape))
            if np.iscomplexobj(result)
            else rng.choice([-1.0, 1.0], result.shape)
        )
        return np.asarray(result + events * self.amplitude * phase)
