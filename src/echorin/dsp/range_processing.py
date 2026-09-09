"""Physical range axes, profiles, and fixed-threshold peak detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from echorin.config import SensorConfig
from echorin.dsp.matched_filter import matched_filter
from echorin.models.detection import Detection


def range_axis(
    sample_count: int,
    sample_rate_hz: float,
    propagation_speed_mps: float,
) -> NDArray[np.float64]:
    """Map round-trip delay bins to monostatic range in metres."""
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if propagation_speed_mps <= 0.0:
        raise ValueError("propagation_speed_mps must be positive")
    delay_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    return propagation_speed_mps * delay_s / 2.0


@dataclass(frozen=True, slots=True)
class RangeProfile:
    """Matched-filter response and its physical range coordinates."""

    ranges_m: NDArray[np.float64]
    response: NDArray[np.floating] | NDArray[np.complexfloating]

    def __post_init__(self) -> None:
        if self.ranges_m.ndim != 1 or self.response.ndim != 1:
            raise ValueError("range-profile arrays must be one-dimensional")
        if self.ranges_m.shape != self.response.shape:
            raise ValueError("ranges_m and response must have equal shape")

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """Absolute matched-filter response used by amplitude detectors."""
        return np.asarray(np.abs(self.response), dtype=np.float64)


class SignalProcessor:
    """Stateless range-processing facade bound to sensor configuration."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config

    def matched_filter(
        self, received_signal: ArrayLike, transmitted_signal: ArrayLike
    ) -> NDArray[np.floating] | NDArray[np.complexfloating]:
        """Apply the configured sensor's matched-filter stage."""
        return matched_filter(received_signal, transmitted_signal)

    def range_axis(self, sample_count: int) -> NDArray[np.float64]:
        """Build the configured physical range axis."""
        return range_axis(
            sample_count,
            self.config.sample_rate_hz,
            self.config.propagation_speed_mps,
        )

    def range_profile(
        self, received_signal: ArrayLike, transmitted_signal: ArrayLike
    ) -> RangeProfile:
        """Produce one lag-aligned matched-filter range profile."""
        response = self.matched_filter(received_signal, transmitted_signal)
        return RangeProfile(self.range_axis(response.size), response)

    def pulse_matrix_range_responses(
        self, received_pulses: ArrayLike, transmitted_signal: ArrayLike
    ) -> NDArray[np.complex128]:
        """Matched-filter every pulse while retaining coherent phase."""
        pulses = np.asarray(received_pulses)
        if pulses.ndim != 2:
            raise ValueError("received_pulses must have shape (pulses, samples)")
        return np.asarray(
            [self.matched_filter(pulse, transmitted_signal) for pulse in pulses],
            dtype=np.complex128,
        )


class FixedThresholdDetector:
    """Reference local-maximum detector with a constant amplitude threshold."""

    def __init__(self, threshold: float, minimum_separation_bins: int = 1) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be nonnegative")
        if minimum_separation_bins < 1:
            raise ValueError("minimum_separation_bins must be at least one")
        self.threshold = threshold
        self.minimum_separation_bins = minimum_separation_bins

    def detect(
        self,
        profile: RangeProfile,
        timestamp_s: float,
        bearing_rad: float,
    ) -> list[Detection]:
        """Convert above-threshold local maxima into sensor measurements."""
        magnitude = profile.magnitude
        peak_bins, properties = find_peaks(
            magnitude,
            height=self.threshold,
            distance=self.minimum_separation_bins,
        )
        noise_floor = float(np.median(magnitude))
        detections: list[Detection] = []
        for source_bin, amplitude in zip(
            peak_bins, properties["peak_heights"], strict=True
        ):
            amplitude_value = float(amplitude)
            snr_db = (
                float(20.0 * np.log10(amplitude_value / noise_floor))
                if noise_floor > 0.0 and amplitude_value > 0.0
                else float("inf")
            )
            confidence = (
                min(1.0, max(0.0, 1.0 - self.threshold / amplitude_value))
                if amplitude_value > 0.0
                else 0.0
            )
            detections.append(
                Detection(
                    timestamp_s=timestamp_s,
                    range_m=float(profile.ranges_m[source_bin]),
                    bearing_rad=bearing_rad,
                    radial_velocity_mps=None,
                    amplitude=amplitude_value,
                    snr_db=snr_db,
                    confidence=confidence,
                    source_bin=int(source_bin),
                )
            )
        return detections
