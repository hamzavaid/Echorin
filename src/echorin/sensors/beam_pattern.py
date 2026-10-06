"""Normalized voltage beam gains; angle offsets are radians."""

from dataclasses import dataclass
from math import cos, exp, isfinite, log, pi, sin
from typing import Literal, Protocol

import numpy as np

from echorin.sensors.scan import ScanConfig


def wrapped(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def _width(width: float) -> None:
    if not isfinite(width) or not 0 < width <= 2 * pi:
        raise ValueError("beam width must be finite and in (0, 2pi]")


class BeamPattern(Protocol):
    """One-way amplitude gain; monostatic sensor applies transmit and receive."""

    def gain(
        self, angle_offset_rad: float, frequency_hz: float | None = None
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class IsotropicBeam:
    def gain(self, angle_offset_rad: float, frequency_hz: float | None = None) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class SectorBeam:
    width_rad: float = pi / 3
    sidelobe_gain: float = 0.0

    def __post_init__(self) -> None:
        _width(self.width_rad)
        if not isfinite(self.sidelobe_gain) or not 0 <= self.sidelobe_gain <= 1:
            raise ValueError("sidelobe_gain must be in [0,1]")

    def gain(self, angle_offset_rad: float, frequency_hz: float | None = None) -> float:
        return (
            1.0
            if abs(wrapped(angle_offset_rad)) <= self.width_rad / 2
            else self.sidelobe_gain
        )


@dataclass(frozen=True, slots=True)
class GaussianBeam:
    """width_rad is full one-way half-power width, not amplitude FWHM."""

    width_rad: float = pi / 3

    def __post_init__(self) -> None:
        _width(self.width_rad)

    def gain(self, angle_offset_rad: float, frequency_hz: float | None = None) -> float:
        return exp(-2 * log(2) * (wrapped(angle_offset_rad) / self.width_rad) ** 2)


@dataclass(frozen=True, slots=True)
class ApertureSincBeam:
    """Uniform broadside aperture; rear hemisphere suppressed explicitly.

    Aperture in nominal carrier wavelengths, narrowband (frequency argument
    reserved for future dispersive patterns). Magnitude retains sidelobes.
    """

    aperture_wavelengths: float = 4.0

    def __post_init__(self) -> None:
        if not isfinite(self.aperture_wavelengths) or self.aperture_wavelengths <= 0:
            raise ValueError("aperture must be finite and positive")

    def gain(self, angle_offset_rad: float, frequency_hz: float | None = None) -> float:
        angle = wrapped(angle_offset_rad)
        return (
            float(abs(np.sinc(self.aperture_wavelengths * sin(angle))))
            if cos(angle) >= 0
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class BeamConfig:
    """Monostatic co-located transmit/receive beam and local scan settings."""

    kind: Literal["isotropic", "sector", "gaussian", "sinc"] = "isotropic"
    width_rad: float = pi / 3
    sidelobe_gain: float = 0.0
    aperture_wavelengths: float = 4.0
    scan: ScanConfig = ScanConfig()

    def __post_init__(self) -> None:
        if self.kind not in ("isotropic", "sector", "gaussian", "sinc"):
            raise ValueError("unknown beam kind")
        SectorBeam(self.width_rad, self.sidelobe_gain)
        ApertureSincBeam(self.aperture_wavelengths)
        if not isinstance(self.scan, ScanConfig):
            raise ValueError("beam scan must be ScanConfig")

    def pattern(self) -> BeamPattern:
        if self.kind == "sector":
            return SectorBeam(self.width_rad, self.sidelobe_gain)
        if self.kind == "gaussian":
            return GaussianBeam(self.width_rad)
        if self.kind == "sinc":
            return ApertureSincBeam(self.aperture_wavelengths)
        return IsotropicBeam()

    @property
    def display_width_rad(self) -> float:
        if self.kind == "isotropic":
            return 2 * pi
        if self.kind == "sinc":
            return 2 * float(np.arcsin(min(1, 0.443 / self.aperture_wavelengths)))
        return self.width_rad
