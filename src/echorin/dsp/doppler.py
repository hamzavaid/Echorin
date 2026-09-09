"""Coherent slow-time Doppler processing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from echorin.config import SensorConfig
from echorin.models.detection import Detection


@dataclass(frozen=True, slots=True)
class DopplerProduct:
    """FFT-shifted Doppler spectrum and physical velocity coordinates."""

    doppler_frequency_hz: NDArray[np.float64]
    radial_velocity_mps: NDArray[np.float64]
    spectrum: NDArray[np.complex128]

    @property
    def magnitude(self) -> NDArray[np.float64]:
        return np.asarray(np.abs(self.spectrum), dtype=np.float64)


def doppler_axis(pulse_count: int, prf_hz: float) -> NDArray[np.float64]:
    """Return FFT-shifted slow-time frequency bins."""
    if pulse_count < 2:
        raise ValueError("pulse_count must be at least two")
    if prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive")
    return np.fft.fftshift(np.fft.fftfreq(pulse_count, d=1.0 / prf_hz))


def radial_velocity_to_doppler_hz(
    radial_velocity_mps: float | NDArray[np.float64], config: SensorConfig
) -> float | NDArray[np.float64]:
    """Map monostatic radial velocity to Doppler frequency."""
    return (
        2.0
        * np.asarray(radial_velocity_mps)
        * config.carrier_frequency_hz
        / config.propagation_speed_mps
    )


def doppler_spectrum(
    pulse_matrix: ArrayLike,
    config: SensorConfig,
    *,
    apply_window: bool = True,
) -> DopplerProduct:
    """FFT each range bin across coherent pulses (slow time)."""
    pulses = np.asarray(pulse_matrix, dtype=np.complex128)
    if pulses.ndim != 2:
        raise ValueError("pulse_matrix must have shape (pulses, range_bins)")
    if pulses.shape[0] < 2 or pulses.shape[1] < 1:
        raise ValueError("pulse_matrix must contain at least two pulses and one bin")
    if apply_window:
        window = np.hanning(pulses.shape[0])
        coherent_gain = float(np.sum(window))
        weighted = pulses * window[:, None]
    else:
        coherent_gain = float(pulses.shape[0])
        weighted = pulses
    spectrum = np.fft.fftshift(np.fft.fft(weighted, axis=0), axes=0)
    if coherent_gain > 0.0:
        spectrum /= coherent_gain
    frequencies = doppler_axis(pulses.shape[0], config.prf_hz)
    wavelength_m = config.propagation_speed_mps / config.carrier_frequency_hz
    velocities = frequencies * wavelength_m / 2.0
    return DopplerProduct(frequencies, velocities, spectrum)


def enrich_detections_with_velocity(
    detections: Sequence[Detection], product: DopplerProduct
) -> tuple[Detection, ...]:
    """Estimate each detection's velocity at its sensor-derived range bin."""
    enriched: list[Detection] = []
    range_bin_count = product.spectrum.shape[1]
    for detection in detections:
        if detection.source_bin >= range_bin_count:
            raise ValueError("detection source_bin exceeds Doppler product")
        velocity_index = int(np.argmax(product.magnitude[:, detection.source_bin]))
        enriched.append(
            replace(
                detection,
                radial_velocity_mps=float(product.radial_velocity_mps[velocity_index]),
            )
        )
    return tuple(enriched)
