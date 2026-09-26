"""Selection and readout models sourced only from detections and tracks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from echorin.models.detection import Detection
from echorin.models.track import Track


@dataclass(frozen=True, slots=True)
class InspectionDetails:
    """Human-readable fields for one published sensor estimate."""

    title: str
    fields: dict[str, str]


class InspectionModel:
    """Keep a stable track ID or frame-local detection index selected."""

    def __init__(self) -> None:
        self.detections: tuple[Detection, ...] = ()
        self.tracks: tuple[Track, ...] = ()
        self.selected_kind: str | None = None
        self.selected_id: int | None = None
        self.current: InspectionDetails | None = None

    def update(
        self, detections: Sequence[Detection], tracks: Sequence[Track]
    ) -> InspectionDetails | None:
        """Replace frame data and follow a selected track across frames."""
        self.detections = tuple(detections)
        self.tracks = tuple(tracks)
        if self.selected_kind == "detection" and self.selected_id is not None:
            return self.select_detection(self.selected_id)
        if self.selected_kind == "track" and self.selected_id is not None:
            return self.select_track(self.selected_id)
        return self.current

    def clear(self) -> None:
        self.detections = ()
        self.tracks = ()
        self.selected_kind = None
        self.selected_id = None
        self.current = None

    def select_detection(self, index: int) -> InspectionDetails | None:
        if not 0 <= index < len(self.detections):
            self.selected_kind = None
            self.selected_id = None
            self.current = None
            return None
        detection = self.detections[index]
        bearing = detection.bearing_rad
        finite_bearing = bool(np.isfinite(bearing))
        fields = {
            "Timestamp": f"{detection.timestamp_s:.3f} s",
            "Range": f"{detection.range_m:.3f} m",
            "Bearing": (
                f"{np.degrees(bearing):.3f}°" if finite_bearing else "Unavailable"
            ),
            "Radial velocity": (
                f"{detection.radial_velocity_mps:.3f} m/s"
                if detection.radial_velocity_mps is not None
                else "Unavailable"
            ),
            "Cartesian X": (
                f"{detection.world_position_m[0]:.3f} m"
                if finite_bearing
                else "Unavailable"
            ),
            "Cartesian Y": (
                f"{detection.world_position_m[1]:.3f} m"
                if finite_bearing
                else "Unavailable"
            ),
            "Amplitude": f"{detection.amplitude:.4g}",
            "SNR": f"{detection.snr_db:.2f} dB",
            "Confidence": f"{detection.confidence:.3f}",
            "Source bin": str(detection.source_bin),
            "Angle bin": (
                str(detection.source_angle_bin)
                if detection.source_angle_bin is not None
                else "Unavailable"
            ),
            "Receiver": detection.receiver_id,
        }
        self.selected_kind = "detection"
        self.selected_id = index
        self.current = InspectionDetails(f"Detection {index + 1}", fields)
        return self.current

    def select_track(self, track_id: int) -> InspectionDetails | None:
        track = next((t for t in self.tracks if t.track_id == track_id), None)
        if track is None:
            self.selected_kind = None
            self.selected_id = None
            self.current = None
            return None
        speed = float(np.hypot(track.vx_mps, track.vy_mps))
        fields = {
            "Track ID": str(track.track_id),
            "Status": track.status.value,
            "Timestamp": f"{track.last_timestamp_s:.3f} s",
            "Cartesian X": f"{track.x_m:.3f} m",
            "Cartesian Y": f"{track.y_m:.3f} m",
            "Velocity X": f"{track.vx_mps:.3f} m/s",
            "Velocity Y": f"{track.vy_mps:.3f} m/s",
            "Speed": f"{speed:.3f} m/s",
            "Heading": f"{np.degrees(np.arctan2(track.vy_mps, track.vx_mps)):.2f}°",
            "Position covariance": np.array2string(
                track.covariance[:2, :2], precision=2, suppress_small=True
            )
            + " m²",
            "Velocity covariance": np.array2string(
                track.covariance[2:, 2:], precision=2, suppress_small=True
            )
            + " (m/s)²",
            "Age / hits / misses": f"{track.age} / {track.hits} / {track.misses}",
        }
        self.selected_kind = "track"
        self.selected_id = track_id
        self.current = InspectionDetails(f"Track {track_id}", fields)
        return self.current
