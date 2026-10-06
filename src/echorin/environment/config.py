"""Immutable environment settings composed separately from physical sampling."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from echorin.config import NoiseConfig
from echorin.environment.clutter import ClutterConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.environment.noise import (
    AwgnNoise,
    ColoredNoise,
    CorrelatedArrayNoise,
    ImpulsiveNoise,
    NoiseModel,
)
from echorin.propagation.multipath import MultipathComponent


@dataclass(frozen=True, slots=True)
class ReceiverNoiseConfig:
    """Noise kind; scale/SNR remains the backwards-compatible NoiseConfig."""

    kind: Literal["awgn", "colored", "impulsive", "correlated"] = "awgn"
    correlation: float = 0.85
    event_probability: float = 0.001
    impulse_amplitude: float = 1.0
    array_correlation: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in ("awgn", "colored", "impulsive", "correlated"):
            raise ValueError("unknown receiver noise kind")
        ColoredNoise(correlation=self.correlation)
        ImpulsiveNoise(
            event_probability=self.event_probability, amplitude=self.impulse_amplitude
        )
        CorrelatedArrayNoise(correlation=self.array_correlation)

    def model(self, scale: NoiseConfig) -> NoiseModel:
        """Construct a numerical model without owning an implicit RNG."""
        if self.kind == "correlated":
            return CorrelatedArrayNoise(scale, self.array_correlation)
        if self.kind == "colored":
            return ColoredNoise(scale, self.correlation)
        if self.kind == "impulsive":
            return ImpulsiveNoise(scale, self.event_probability, self.impulse_amplitude)
        return AwgnNoise(scale)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Optional raw-signal effects. Empty defaults reproduce v1.3 exactly.

    absorption_db_per_m is an extra amplitude loss per total path metre;
    inverse-power spreading remains the existing selectable baseline.
    """

    receiver_noise: ReceiverNoiseConfig = ReceiverNoiseConfig()
    interference: tuple[NarrowbandInterference, ...] = ()
    clutter: tuple[ClutterConfig, ...] = ()
    multipath: tuple[MultipathComponent, ...] = ()
    attenuation_exponent: float = 2.0
    absorption_db_per_m: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.attenuation_exponent) or self.attenuation_exponent <= 0:
            raise ValueError("attenuation exponent must be finite and positive")
        if not isfinite(self.absorption_db_per_m) or self.absorption_db_per_m < 0:
            raise ValueError("absorption must be finite and nonnegative")
        for name in ("interference", "clutter", "multipath"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
