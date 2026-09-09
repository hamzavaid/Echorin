"""One-to-one nearest-neighbor data association."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from echorin.models.track import Track


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Assigned and unassigned indices for one tracker update."""

    matches: tuple[tuple[int, int], ...]
    unassigned_track_indices: tuple[int, ...]
    unassigned_measurement_indices: tuple[int, ...]


def associate_nearest_neighbor(
    tracks: Sequence[Track],
    measurements_m: ArrayLike,
    measurement_variance_m2: float,
    gate_threshold: float,
) -> AssociationResult:
    """Greedily assign globally sorted Mahalanobis distances within a gate."""
    measurements = np.asarray(measurements_m, dtype=np.float64)
    if measurements.size == 0:
        measurements = np.empty((0, 2), dtype=np.float64)
    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements_m must have shape (n, 2)")
    if measurement_variance_m2 <= 0.0:
        raise ValueError("measurement_variance_m2 must be positive")
    if gate_threshold <= 0.0:
        raise ValueError("gate_threshold must be positive")

    candidates: list[tuple[float, int, int]] = []
    measurement_covariance = np.eye(2) * measurement_variance_m2
    for track_index, track in enumerate(tracks):
        innovation_covariance = track.covariance[:2, :2] + measurement_covariance
        for measurement_index, measurement in enumerate(measurements):
            residual = measurement - track.state[:2]
            distance_squared = float(
                residual @ np.linalg.solve(innovation_covariance, residual)
            )
            if distance_squared <= gate_threshold:
                candidates.append((distance_squared, track_index, measurement_index))

    assigned_tracks: set[int] = set()
    assigned_measurements: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_index, measurement_index in sorted(candidates):
        if track_index in assigned_tracks or measurement_index in assigned_measurements:
            continue
        assigned_tracks.add(track_index)
        assigned_measurements.add(measurement_index)
        matches.append((track_index, measurement_index))

    return AssociationResult(
        matches=tuple(sorted(matches)),
        unassigned_track_indices=tuple(
            index for index in range(len(tracks)) if index not in assigned_tracks
        ),
        unassigned_measurement_indices=tuple(
            index
            for index in range(len(measurements))
            if index not in assigned_measurements
        ),
    )
