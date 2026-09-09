"""Cell-averaging constant-false-alarm-rate detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from echorin.dsp.range_processing import RangeProfile
from echorin.models.detection import Detection


@dataclass(frozen=True, slots=True)
class CfarConfig:
    """Symmetric CA-CFAR window and false-alarm parameters."""

    training_cells: int = 16
    guard_cells: int = 4
    false_alarm_probability: float = 1e-3
    minimum_separation_bins: int = 1


@dataclass(frozen=True, slots=True)
class CfarResult:
    """Adaptive threshold product and resulting local-peak detections."""

    threshold: NDArray[np.float64]
    noise_power: NDArray[np.float64]
    detections: tuple[Detection, ...]


class CaCfarDetector:
    """Square-law CA-CFAR detector for complex or real range profiles."""

    def __init__(self, config: CfarConfig | None = None) -> None:
        self.config = config or CfarConfig()
        if self.config.training_cells < 2:
            raise ValueError("training_cells must be at least two per side")
        if self.config.guard_cells < 0:
            raise ValueError("guard_cells must be nonnegative")
        if not 0.0 < self.config.false_alarm_probability < 1.0:
            raise ValueError("false_alarm_probability must lie between zero and one")
        if self.config.minimum_separation_bins < 1:
            raise ValueError("minimum_separation_bins must be at least one")

    @property
    def training_count(self) -> int:
        """Total training cells on both sides of each CUT."""
        return 2 * self.config.training_cells

    @property
    def threshold_scale(self) -> float:
        """CA-CFAR power multiplier for exponentially distributed noise."""
        count = self.training_count
        return count * (self.config.false_alarm_probability ** (-1.0 / count) - 1.0)

    def detect(
        self,
        profile: RangeProfile,
        timestamp_s: float,
        bearing_rad: float,
    ) -> CfarResult:
        """Estimate local noise and detect threshold-crossing local maxima."""
        magnitude = profile.magnitude
        power = np.square(magnitude)
        training = self.config.training_cells
        guard = self.config.guard_cells
        edge = training + guard

        kernel = np.ones(2 * edge + 1, dtype=np.float64)
        kernel[training : training + 2 * guard + 1] = 0.0
        training_sum = np.convolve(power, kernel, mode="same")
        noise_power = training_sum / self.training_count
        threshold = np.sqrt(noise_power * self.threshold_scale)

        if edge:
            noise_power[:edge] = np.nan
            noise_power[-edge:] = np.nan
            threshold[:edge] = np.nan
            threshold[-edge:] = np.nan

        peak_bins, properties = find_peaks(
            magnitude,
            height=threshold,
            distance=self.config.minimum_separation_bins,
        )
        detections: list[Detection] = []
        for source_bin, amplitude in zip(
            peak_bins, properties["peak_heights"], strict=True
        ):
            local_noise_power = float(noise_power[source_bin])
            signal_power = float(amplitude**2)
            snr_db = (
                float(10.0 * np.log10(signal_power / local_noise_power))
                if local_noise_power > 0.0 and signal_power > 0.0
                else float("inf")
            )
            local_threshold = float(threshold[source_bin])
            confidence = min(1.0, max(0.0, 1.0 - local_threshold / float(amplitude)))
            detections.append(
                Detection(
                    timestamp_s=timestamp_s,
                    range_m=float(profile.ranges_m[source_bin]),
                    bearing_rad=bearing_rad,
                    radial_velocity_mps=None,
                    amplitude=float(amplitude),
                    snr_db=snr_db,
                    confidence=confidence,
                    source_bin=int(source_bin),
                )
            )
        return CfarResult(threshold, noise_power, tuple(detections))
