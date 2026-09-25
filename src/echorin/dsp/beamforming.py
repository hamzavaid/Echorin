"""Narrowband steering and conventional Bartlett direction finding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from echorin.sensors.array import ArrayGeometry


@dataclass(frozen=True, slots=True)
class BartlettSpectrum:
    """Beam power indexed by the supplied receiver-relative angle grid."""

    bearings_rad: NDArray[np.float64]
    power: NDArray[np.float64]


def steering_vectors(
    array: ArrayGeometry,
    wavelength_m: float,
    bearings_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Return [element, angle] plane-wave phases for +x broadside."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    bearings = np.asarray(bearings_rad, dtype=np.float64)
    if bearings.ndim != 1 or not len(bearings) or not np.all(np.isfinite(bearings)):
        raise ValueError("bearings_rad must be a finite nonempty axis")
    relative = (
        array.element_positions_m - array.element_positions_m[array.reference_element]
    )
    local = bearings - array.orientation_rad
    directions = np.vstack((np.cos(local), np.sin(local)))
    return np.asarray(
        np.exp(2j * np.pi * (relative @ directions) / wavelength_m), dtype=np.complex128
    )


def bartlett_spectrum(
    samples: ArrayLike,
    array: ArrayGeometry,
    wavelength_m: float,
    bearings_rad: ArrayLike,
) -> BartlettSpectrum:
    """Compute a^H R a / (a^H a) from composite element snapshots."""
    data = np.asarray(samples, dtype=np.complex128)
    if data.ndim != 2 or data.shape[0] != array.element_count or not data.shape[1]:
        raise ValueError("samples must have shape (elements, snapshots)")
    angles = np.asarray(bearings_rad, dtype=np.float64)
    steering = steering_vectors(array, wavelength_m, angles)
    projections = steering.conj().T @ data
    power = np.mean(np.abs(projections) ** 2, axis=1) / array.element_count
    return BartlettSpectrum(angles.copy(), np.asarray(power, dtype=np.float64))
