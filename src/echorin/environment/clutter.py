"""Seeded distributed scatterers for range clutter and acoustic reverberation."""

from dataclasses import dataclass
from math import isfinite, pi
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class ClutterConfig:
    """Poisson density per metre, Rayleigh clutter or decaying exponential reverb.

    Scatterers persist in world coordinates for the life of the sensor. Density
    is integrated over range (not an area density). Speed is bounded and radial
    at initialization. This is a phenomenological model, not an ocean solver.
    """

    kind: Literal["clutter", "reverberation"] = "clutter"
    density_per_m: float = 0.01
    reflectivity_scale: float = 0.05
    minimum_range_m: float = 0.0
    maximum_speed_mps: float = 0.0
    decay_range_m: float = 100.0
    start_bearing_rad: float = -pi / 2
    stop_bearing_rad: float = pi / 2
    maximum_scatterers: int = 2048

    def __post_init__(self) -> None:
        if self.kind not in ("clutter", "reverberation"):
            raise ValueError("unknown clutter kind")
        for value in (
            self.density_per_m,
            self.reflectivity_scale,
            self.minimum_range_m,
            self.maximum_speed_mps,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError("clutter scales must be finite and nonnegative")
        if not isfinite(self.decay_range_m) or self.decay_range_m <= 0:
            raise ValueError("decay_range_m must be finite and positive")
        if (
            not isfinite(self.start_bearing_rad)
            or not isfinite(self.stop_bearing_rad)
            or self.stop_bearing_rad <= self.start_bearing_rad
        ):
            raise ValueError("clutter bearing interval must be finite and increasing")
        if self.maximum_scatterers < 1:
            raise ValueError("maximum_scatterers must be positive")


@dataclass(frozen=True, slots=True)
class Scatterer:
    """Internal synthesis geometry; never published as a measurement."""

    range_m: float
    bearing_rad: float
    reflectivity: float
    radial_velocity_mps: float
    phase_rad: float


def scatterers(
    config: ClutterConfig, max_range_m: float, rng: np.random.Generator
) -> tuple[Scatterer, ...]:
    """Sample a persistent scattering field; reject unsafe work, never truncate."""
    span = max_range_m - config.minimum_range_m
    if not isfinite(max_range_m) or span <= 0:
        raise ValueError("clutter minimum range must be below sensor maximum range")
    expected = config.density_per_m * span
    if expected > config.maximum_scatterers:
        raise ValueError("clutter density exceeds configured scatterer work limit")
    count = int(rng.poisson(expected))
    if count > config.maximum_scatterers:
        raise ValueError("sampled clutter exceeds configured scatterer work limit")
    ranges = rng.uniform(config.minimum_range_m, max_range_m, count)
    bearings = rng.uniform(config.start_bearing_rad, config.stop_bearing_rad, count)
    amplitudes = (
        rng.rayleigh(config.reflectivity_scale, count)
        if config.kind == "clutter"
        else rng.exponential(config.reflectivity_scale, count)
        * np.exp(-ranges / config.decay_range_m)
    )
    speeds = rng.uniform(-config.maximum_speed_mps, config.maximum_speed_mps, count)
    phases = rng.uniform(-pi, pi, count)
    return tuple(
        Scatterer(*map(float, row))
        for row in zip(ranges, bearings, amplitudes, speeds, phases, strict=True)
    )
