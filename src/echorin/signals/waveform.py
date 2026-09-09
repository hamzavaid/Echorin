"""Sampled transmit-waveform generation."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from echorin.config import SensorConfig


class WaveformKind(StrEnum):
    """Waveforms supported by the Milestone 3 sensor."""

    RECTANGULAR = "rectangular"
    LFM = "lfm"


def _time_axis(config: SensorConfig) -> NDArray[np.float64]:
    return np.arange(config.pulse_samples, dtype=np.float64) / config.sample_rate_hz


def rectangular_pulse(
    config: SensorConfig, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """Generate a rectangular-envelope cosine pulse at the carrier frequency."""
    if amplitude < 0.0:
        raise ValueError("amplitude must be nonnegative")
    time_s = _time_axis(config)
    return amplitude * np.cos(2.0 * np.pi * config.carrier_frequency_hz * time_s)


def lfm_chirp(config: SensorConfig, amplitude: float = 1.0) -> NDArray[np.float64]:
    """Generate a cosine LFM chirp centered on the carrier frequency."""
    if amplitude < 0.0:
        raise ValueError("amplitude must be nonnegative")
    time_s = _time_axis(config)
    start_frequency_hz = config.carrier_frequency_hz - config.bandwidth_hz / 2.0
    slope_hz_s = config.bandwidth_hz / config.pulse_width_s
    phase_cycles = start_frequency_hz * time_s + 0.5 * slope_hz_s * time_s**2
    return amplitude * np.cos(2.0 * np.pi * phase_cycles)


def generate_waveform(
    config: SensorConfig,
    kind: WaveformKind = WaveformKind.LFM,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Dispatch a validated transmit-waveform generator."""
    if kind is WaveformKind.RECTANGULAR:
        return rectangular_pulse(config, amplitude)
    if kind is WaveformKind.LFM:
        return lfm_chirp(config, amplitude)
    raise ValueError(f"unsupported waveform kind: {kind}")
