"""Multi-target tracking and lifecycle management."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from echorin.models.detection import Detection
from echorin.models.track import Track, TrackStatus
from echorin.tracking.association import associate_nearest_neighbor
from echorin.tracking.kalman import ConstantVelocityKalmanFilter


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Kalman, gating, lifecycle, and history parameters."""

    confirm_hits: int = 3
    max_missed_updates: int = 3
    measurement_std_m: float = 20.0
    initial_velocity_std_mps: float = 100.0
    process_acceleration_std_mps2: float = 2.0
    gate_threshold: float = 9.21
    maximum_history: int = 100

    def __post_init__(self) -> None:
        if self.confirm_hits < 1:
            raise ValueError("confirm_hits must be positive")
        if self.max_missed_updates < 0:
            raise ValueError("max_missed_updates must be nonnegative")
        if self.measurement_std_m <= 0.0:
            raise ValueError("measurement_std_m must be positive")
        if self.initial_velocity_std_mps <= 0.0:
            raise ValueError("initial_velocity_std_mps must be positive")
        if self.process_acceleration_std_mps2 < 0.0:
            raise ValueError(
                "process acceleration standard deviation must be nonnegative"
            )
        if self.gate_threshold <= 0.0:
            raise ValueError("gate_threshold must be positive")
        if self.maximum_history < 1:
            raise ValueError("maximum_history must be positive")


def detection_to_cartesian(detection: Detection) -> NDArray[np.float64]:
    """Convert one finite sensor polar measurement into Cartesian coordinates."""
    return np.array(
        [
            detection.range_m * np.cos(detection.bearing_rad),
            detection.range_m * np.sin(detection.bearing_rad),
        ],
        dtype=np.float64,
    )


class MultiTargetTracker:
    """Nearest-neighbor constant-velocity tracker with persistent local IDs."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[Track] = []
        self.deleted_tracks: list[Track] = []
        self._next_track_id = 1
        self._last_timestamp_s: float | None = None

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def reset(self) -> None:
        """Clear tracks and restart deterministic local ID allocation."""
        self._tracks.clear()
        self.deleted_tracks.clear()
        self._next_track_id = 1
        self._last_timestamp_s = None

    def update(
        self, detections: Sequence[Detection], timestamp_s: float
    ) -> tuple[Track, ...]:
        """Predict, associate, correct, and manage lifecycle for one frame."""
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("timestamp_s must increase between tracker updates")
        valid_detections = [
            detection
            for detection in detections
            if np.isfinite(detection.range_m) and np.isfinite(detection.bearing_rad)
        ]
        measurements = np.asarray(
            [detection_to_cartesian(detection) for detection in valid_detections],
            dtype=np.float64,
        ).reshape((-1, 2))

        if self._last_timestamp_s is not None:
            dt_s = timestamp_s - self._last_timestamp_s
            for track in self._tracks:
                filter_ = self._filter_for(track)
                filter_.predict(dt_s)
                track.state = filter_.state
                track.covariance = filter_.covariance
                track.age += 1
                track.last_timestamp_s = timestamp_s

        associations = associate_nearest_neighbor(
            self._tracks,
            measurements,
            measurement_variance_m2=self.config.measurement_std_m**2,
            gate_threshold=self.config.gate_threshold,
        )
        for track_index, measurement_index in associations.matches:
            track = self._tracks[track_index]
            filter_ = self._filter_for(track)
            filter_.update(measurements[measurement_index])
            track.state = filter_.state
            track.covariance = filter_.covariance
            track.hits += 1
            track.misses = 0
            track.status = (
                TrackStatus.CONFIRMED
                if track.hits >= self.config.confirm_hits
                else TrackStatus.TENTATIVE
            )

        for track_index in associations.unassigned_track_indices:
            track = self._tracks[track_index]
            track.misses += 1
            if track.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING):
                track.status = TrackStatus.COASTING

        for measurement_index in associations.unassigned_measurement_indices:
            track = Track.initial(
                self._next_track_id,
                measurements[measurement_index],
                timestamp_s,
                position_std_m=self.config.measurement_std_m,
                velocity_std_mps=self.config.initial_velocity_std_mps,
            )
            if self.config.confirm_hits == 1:
                track.status = TrackStatus.CONFIRMED
            self._next_track_id += 1
            self._tracks.append(track)

        survivors: list[Track] = []
        for track in self._tracks:
            if track.misses > self.config.max_missed_updates:
                track.status = TrackStatus.DELETED
                self.deleted_tracks.append(track)
                continue
            if track.last_timestamp_s != timestamp_s:
                track.last_timestamp_s = timestamp_s
            if not (
                track.age == 1
                and len(track.history) == 1
                and track.last_timestamp_s == timestamp_s
            ):
                track.append_history(self.config.maximum_history)
            survivors.append(track)
        self._tracks = survivors
        self._last_timestamp_s = timestamp_s
        return self.tracks

    def _filter_for(self, track: Track) -> ConstantVelocityKalmanFilter:
        return ConstantVelocityKalmanFilter(
            track.state,
            track.covariance,
            process_acceleration_std_mps2=self.config.process_acceleration_std_mps2,
            measurement_std_m=self.config.measurement_std_m,
        )
