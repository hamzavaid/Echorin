"""Coherent narrowband receiver interference at absolute simulation time."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class NarrowbandInterference:
    """Tone/chirp; optional angle is local to receiver-array broadside.

    With no angle the disturbance is common-mode across array channels.
    Amplitude is a voltage/sample amplitude, not power or reflectivity.
    """

    frequency_hz: float
    amplitude: float = 0.1
    phase_rad: float = 0.0
    drift_hz_s: float = 0.0
    arrival_angle_rad: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.frequency_hz,
            self.amplitude,
            self.phase_rad,
            self.drift_hz_s,
            self.arrival_angle_rad,
        ):
            if value is not None and not isfinite(value):
                raise ValueError("interference parameters must be finite")
        if self.amplitude < 0:
            raise ValueError("interference amplitude must be nonnegative")

    def samples(
        self, times_s: ArrayLike, *, complex_output: bool = True
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        times = np.asarray(times_s, dtype=float)
        phase = self.phase_rad + 2 * np.pi * (
            self.frequency_hz * times + 0.5 * self.drift_hz_s * times**2
        )
        values = self.amplitude * np.exp(1j * phase)
        return values if complex_output else values.real
