"""Signal-derived range-angle products and angular peak extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from echorin.config import SensorConfig
from echorin.dsp.beamforming import steering_vectors
from echorin.models.detection import Detection
from echorin.models.geometry import SensorPose
from echorin.sensors.array import ArrayGeometry


@dataclass(frozen=True, slots=True)
class RangeAngleProduct:
    """Bartlett power with axis order [bearing, range]."""

    ranges_m: NDArray[np.float64]
    bearings_rad: NDArray[np.float64]
    power: NDArray[np.float64]
    timestamp_s: float
    receiver_id: str = "receiver-0"
    receiver_pose: SensorPose | None = None

    def __post_init__(self) -> None:
        if self.power.shape != (len(self.bearings_rad), len(self.ranges_m)):
            raise ValueError("range-angle power must have shape (angle, range)")


def range_angle_product(
    range_responses: ArrayLike,
    ranges_m: NDArray[np.float64],
    array: ArrayGeometry,
    config: SensorConfig,
    bearings_rad: ArrayLike,
    timestamp_s: float,
    receiver_pose: SensorPose | None = None,
) -> RangeAngleProduct:
    """Beamform composite matched-filter responses over pulse snapshots."""
    data = np.asarray(range_responses, dtype=np.complex128)
    if (
        data.ndim != 3
        or data.shape[0] != array.element_count
        or data.shape[2] != len(ranges_m)
    ):
        raise ValueError("range_responses must have shape (element, pulse, range)")
    angles = np.asarray(bearings_rad, dtype=np.float64)
    steering = steering_vectors(
        array, config.propagation_speed_mps / config.carrier_frequency_hz, angles
    )
    projections = np.einsum("ea,epr->apr", steering.conj(), data, optimize=True)
    power = np.mean(abs(projections) ** 2, axis=1) / array.element_count
    return RangeAngleProduct(
        np.asarray(ranges_m, dtype=np.float64),
        angles,
        np.asarray(power, dtype=np.float64),
        timestamp_s,
        receiver_pose=receiver_pose,
    )


def angle_detections(
    range_detections: tuple[Detection, ...],
    product: RangeAngleProduct,
    *,
    relative_peak_height: float = 0.5,
) -> tuple[Detection, ...]:
    """Locate separate angular maxima at CFAR accepted range bins."""
    if not 0 < relative_peak_height <= 1:
        raise ValueError("relative_peak_height must lie in (0, 1]")
    output: list[Detection] = []
    for detection in range_detections:
        row = product.power[:, detection.source_bin]
        if not np.any(row > 0):
            continue
        peaks, _ = find_peaks(row, height=float(row.max()) * relative_peak_height)
        if row[0] >= row[1] and row[0] >= row.max() * relative_peak_height:
            peaks = np.r_[0, peaks]
        if row[-1] >= row[-2] and row[-1] >= row.max() * relative_peak_height:
            peaks = np.r_[peaks, len(row) - 1]
        for angle_bin in peaks:
            output.append(
                replace(
                    detection,
                    bearing_rad=float(product.bearings_rad[angle_bin]),
                    source_angle_bin=int(angle_bin),
                    receiver_id=product.receiver_id,
                    sensor_pose=product.receiver_pose,
                )
            )
    return tuple(output)
