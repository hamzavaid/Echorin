"""Physical receiver element geometry for narrowband planar arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Element offsets in receiver coordinates; +x is array broadside."""

    element_positions_m: NDArray[np.float64]
    reference_element: int = 0
    orientation_rad: float = 0.0
    carrier_frequency_hz: float | None = None
    propagation_speed_mps: float | None = None
    allow_ambiguous_spacing: bool = False

    def __post_init__(self) -> None:
        positions = np.asarray(self.element_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
            raise ValueError("array requires at least two 2D element positions")
        if not np.all(np.isfinite(positions)) or not np.isfinite(self.orientation_rad):
            raise ValueError("array geometry must be finite")
        if not 0 <= self.reference_element < len(positions):
            raise ValueError("reference_element is outside the array")
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        if np.any((distances + np.eye(len(positions))) == 0):
            raise ValueError("array elements must occupy distinct positions")
        if (self.carrier_frequency_hz is None) != (self.propagation_speed_mps is None):
            raise ValueError("carrier frequency and propagation speed must be paired")
        if self.carrier_frequency_hz is not None:
            if self.carrier_frequency_hz <= 0 or self.propagation_speed_mps <= 0:
                raise ValueError(
                    "carrier frequency and propagation speed must be positive"
                )
            nearest = np.min(np.where(distances > 0, distances, np.inf), axis=1)
            wavelength = self.propagation_speed_mps / self.carrier_frequency_hz
            if not self.allow_ambiguous_spacing and np.any(
                nearest > wavelength / 2 + 1e-12
            ):
                raise ValueError("element spacing exceeds half wavelength")
        object.__setattr__(self, "element_positions_m", positions.copy())

    @property
    def element_count(self) -> int:
        """Number of receiver channels."""
        return self.element_positions_m.shape[0]


def uniform_linear_array(
    n_elements: int,
    spacing_m: float,
    *,
    orientation_rad: float = 0.0,
    carrier_frequency_hz: float | None = None,
    propagation_speed_mps: float | None = None,
    allow_ambiguous_spacing: bool = False,
) -> ArrayGeometry:
    """Place ULA elements along local +y, with element zero at the origin."""
    if n_elements < 2 or spacing_m <= 0 or not np.isfinite(spacing_m):
        raise ValueError("ULA needs at least two elements and positive spacing")
    positions = np.column_stack(
        (np.zeros(n_elements), np.arange(n_elements) * spacing_m)
    )
    return ArrayGeometry(
        positions,
        orientation_rad=orientation_rad,
        carrier_frequency_hz=carrier_frequency_hz,
        propagation_speed_mps=propagation_speed_mps,
        allow_ambiguous_spacing=allow_ambiguous_spacing,
    )
