"""Truth-free numerical adapters for engineering plots and selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from echorin.dsp.doppler import DopplerProduct


def to_db(
    magnitude: NDArray[np.float64], floor_db: float = -100.0
) -> NDArray[np.float64]:
    """Convert unit-reference magnitude to finite amplitude dB with a floor."""
    values = np.asarray(magnitude, dtype=np.float64)
    if np.any(values[np.isfinite(values)] < 0.0):
        raise ValueError("magnitude must be nonnegative")
    if not np.isfinite(floor_db) or floor_db >= 0.0:
        raise ValueError("floor_db must be finite and negative")
    safe = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    with np.errstate(divide="ignore"):
        result = 20.0 * np.log10(safe)
    return np.maximum(np.where(np.isfinite(result), result, floor_db), floor_db)


@dataclass(frozen=True, slots=True)
class RangeDopplerCell:
    """Selected coordinates and signal level from one processed cell."""

    velocity_bin: int
    range_bin: int
    radial_velocity_mps: float
    range_m: float
    magnitude: float
    level_db: float


@dataclass(frozen=True, slots=True)
class RangeDopplerImage:
    """2D image with row-major [velocity, range] cell-center coordinates."""

    ranges_m: NDArray[np.float64]
    velocities_mps: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    level_db: NDArray[np.float64]

    def cell_at(self, range_m: float, velocity_mps: float) -> tuple[int, int] | None:
        """Return nearest cell if the point is within the image cell edges."""
        x = self.ranges_m
        y = self.velocities_mps
        dx = float(x[1] - x[0]) if x.size > 1 else 1.0
        dy = float(y[1] - y[0]) if y.size > 1 else 1.0
        if not x[0] - dx / 2 <= range_m < x[-1] + dx / 2:
            return None
        if not y[0] - dy / 2 <= velocity_mps < y[-1] + dy / 2:
            return None
        return (
            int(np.argmin(np.abs(y - velocity_mps))),
            int(np.argmin(np.abs(x - range_m))),
        )

    def selected_cell(self, velocity_bin: int, range_bin: int) -> RangeDopplerCell:
        """Describe one bin without consulting simulation truth."""
        return RangeDopplerCell(
            velocity_bin=velocity_bin,
            range_bin=range_bin,
            radial_velocity_mps=float(self.velocities_mps[velocity_bin]),
            range_m=float(self.ranges_m[range_bin]),
            magnitude=float(self.magnitude[velocity_bin, range_bin]),
            level_db=float(self.level_db[velocity_bin, range_bin]),
        )


def range_doppler_image(product: DopplerProduct) -> RangeDopplerImage:
    """Map the existing coherent FFT product to a physically labeled image."""
    if product.ranges_m is None:
        raise ValueError("Doppler product needs physical range coordinates")
    ranges = np.asarray(product.ranges_m, dtype=np.float64)
    velocities = np.asarray(product.radial_velocity_mps, dtype=np.float64)
    for axis in (ranges, velocities):
        if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
            raise ValueError("heatmap axes must contain at least two finite cells")
        steps = np.diff(axis)
        if np.any(steps <= 0.0) or not np.allclose(steps, steps[0]):
            raise ValueError("heatmap axes must be uniformly increasing")
    magnitude = product.magnitude
    return RangeDopplerImage(ranges, velocities, magnitude, to_db(magnitude))
