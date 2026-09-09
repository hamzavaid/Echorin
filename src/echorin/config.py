"""Validated configuration shared by Echorin's numerical layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil
from typing import Any

SPEED_OF_LIGHT_MPS = 299_792_458.0
NOMINAL_WATER_SOUND_SPEED_MPS = 1_500.0


class SensorMode(StrEnum):
    """Configured sensing medium.

    Radar is the implemented mode through Milestone 5. Sonar is represented in
    the stable configuration interface but its processing mode is Milestone 8.
    """

    RADAR = "radar"
    SONAR = "sonar"


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Additive Gaussian noise configuration.

    When ``snr_db`` is supplied, the noise generator derives its standard
    deviation from the active signal power. Otherwise ``standard_deviation`` is
    used directly.
    """

    standard_deviation: float = 0.02
    snr_db: float | None = None

    def __post_init__(self) -> None:
        if self.standard_deviation < 0.0:
            raise ValueError("standard_deviation must be nonnegative")


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Physical and sampled-data parameters for a monostatic sensor."""

    mode: SensorMode = SensorMode.RADAR
    sample_rate_hz: float = 10_000_000.0
    carrier_frequency_hz: float = 1_000_000.0
    bandwidth_hz: float = 1_000_000.0
    pulse_width_s: float = 10e-6
    prf_hz: float = 1_000.0
    max_range_m: float = 10_000.0
    propagation_speed_mps: float = SPEED_OF_LIGHT_MPS
    noise_model: NoiseConfig = field(default_factory=NoiseConfig)

    @classmethod
    def radar(cls, **overrides: Any) -> SensorConfig:
        """Create a radar-mode configuration with RF simulation defaults."""
        values: dict[str, Any] = {"mode": SensorMode.RADAR}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def sonar(cls, **overrides: Any) -> SensorConfig:
        """Create a sonar-mode configuration with water-acoustic defaults."""
        values: dict[str, Any] = {
            "mode": SensorMode.SONAR,
            "sample_rate_hz": 96_000.0,
            "carrier_frequency_hz": 20_000.0,
            "bandwidth_hz": 8_000.0,
            "pulse_width_s": 5e-3,
            "prf_hz": 3.0,
            "max_range_m": 200.0,
            "propagation_speed_mps": NOMINAL_WATER_SOUND_SPEED_MPS,
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        positive_fields = (
            "sample_rate_hz",
            "carrier_frequency_hz",
            "bandwidth_hz",
            "pulse_width_s",
            "prf_hz",
            "max_range_m",
            "propagation_speed_mps",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.mode is SensorMode.SONAR and self.propagation_speed_mps > 10_000.0:
            raise ValueError(
                "sonar propagation speed must be explicitly configured for acoustics"
            )
        if self.sample_rate_hz < 2.0 * self.bandwidth_hz:
            raise ValueError("sample_rate_hz must be at least twice bandwidth_hz")
        highest_waveform_frequency = self.carrier_frequency_hz + self.bandwidth_hz / 2.0
        if self.sample_rate_hz / 2.0 < highest_waveform_frequency:
            raise ValueError(
                "sample_rate_hz violates Nyquist for the real passband waveform"
            )
        if self.max_range_m > self.unambiguous_range_m:
            raise ValueError("max_range_m exceeds the unambiguous range for prf_hz")
        if self.pulse_width_s >= self.pri_s:
            raise ValueError("pulse_width_s must be shorter than the pulse interval")

    @property
    def pri_s(self) -> float:
        """Pulse-repetition interval in seconds."""
        return 1.0 / self.prf_hz

    @property
    def pulse_samples(self) -> int:
        """Number of samples in one transmitted pulse."""
        return max(1, int(round(self.pulse_width_s * self.sample_rate_hz)))

    @property
    def acquisition_samples(self) -> int:
        """Samples needed to retain the full echo at maximum range."""
        delay = ceil(
            2.0 * self.max_range_m / self.propagation_speed_mps * self.sample_rate_hz
        )
        return delay + self.pulse_samples

    @property
    def unambiguous_range_m(self) -> float:
        """Maximum round-trip range contained in one pulse interval."""
        return self.propagation_speed_mps * self.pri_s / 2.0


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """World stepping and repeatability parameters."""

    dt_s: float = 0.05
    random_seed: int = 7
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.duration_s is not None and self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive when supplied")
