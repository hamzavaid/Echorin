"""Published processing-frame model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FrameResult:
    """Outputs published by one complete simulation/sensor processing cycle."""

    timestamp_s: float
    transmitted_signal: Any | None = None
    received_signal: Any | None = None
    range_profile: Any | None = None
    detections: list[Any] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    ground_truth: list[Any] | None = None
    timing_metrics_s: dict[str, float] = field(default_factory=dict)
