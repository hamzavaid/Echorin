"""Explicit generalized path expansion with a compatible monostatic wrapper."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class MultipathComponent:
    """Extra total (outbound plus return) distance relative to the direct path."""

    extra_path_length_m: float
    attenuation_scale: float = 0.3
    phase_offset_rad: float = 0.0
    angle_offset_rad: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isfinite(v)
            for v in (
                self.extra_path_length_m,
                self.attenuation_scale,
                self.phase_offset_rad,
                self.angle_offset_rad,
            )
        ):
            raise ValueError("multipath parameters must be finite")
        if self.extra_path_length_m < 0 or not 0 <= self.attenuation_scale <= 1:
            raise ValueError(
                "path distance must be nonnegative and attenuation in [0,1]"
            )


@dataclass(frozen=True, slots=True)
class PropagationPath:
    """Synthesis-only path with no identity or hidden target state."""

    path_length_m: float
    delay_s: float
    amplitude_scale: float = 1.0
    phase_rad: float = 0.0
    angle_offset_rad: float = 0.0
    path_kind: str = "direct"
    transmitter_id: str | None = None
    receiver_id: str | None = None


def expand_paths(
    range_m: float,
    speed_mps: float,
    components: tuple[MultipathComponent, ...] = (),
) -> tuple[PropagationPath, ...]:
    """Return direct plus secondary paths; windows are clipped by the sensor."""
    if (
        not isfinite(range_m)
        or range_m < 0
        or not isfinite(speed_mps)
        or speed_mps <= 0
    ):
        raise ValueError("range must be nonnegative and propagation speed positive")
    length = 2 * range_m
    return (PropagationPath(length, length / speed_mps),) + tuple(
        PropagationPath(
            length + p.extra_path_length_m,
            (length + p.extra_path_length_m) / speed_mps,
            p.attenuation_scale,
            p.phase_offset_rad,
            p.angle_offset_rad,
            "multipath",
        )
        for p in components
    )
