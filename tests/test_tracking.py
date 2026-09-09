"""Milestone 6 numerical and integration tests for multi-target tracking."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import NoiseConfig, SensorConfig, SimulationConfig
from echorin.gui.main_window import MainWindow
from echorin.models.detection import Detection
from echorin.models.track import Track, TrackStatus
from echorin.sensors.radar import RadarSensor
from echorin.simulation.scenarios import single_stationary_target
from echorin.simulation.target import Target
from echorin.tracking.association import associate_nearest_neighbor
from echorin.tracking.kalman import ConstantVelocityKalmanFilter
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig


def detection(
    x_m: float,
    y_m: float,
    timestamp_s: float,
    amplitude: float = 10.0,
) -> Detection:
    return Detection(
        timestamp_s=timestamp_s,
        range_m=float(np.hypot(x_m, y_m)),
        bearing_rad=float(np.arctan2(y_m, x_m)),
        radial_velocity_mps=None,
        amplitude=amplitude,
        snr_db=20.0,
        confidence=0.9,
        source_bin=0,
    )


def test_kalman_predict_uses_constant_velocity_transition() -> None:
    state = np.array([10.0, -5.0, 3.0, -2.0])
    covariance = np.eye(4)
    filter_ = ConstantVelocityKalmanFilter(
        state, covariance, process_acceleration_std_mps2=0.0, measurement_std_m=2.0
    )

    filter_.predict(dt_s=0.5)

    np.testing.assert_allclose(filter_.state, [11.5, -6.0, 3.0, -2.0])
    expected_f = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        filter_.covariance, expected_f @ covariance @ expected_f.T
    )


def test_kalman_filter_reduces_noisy_position_rmse() -> None:
    rng = np.random.default_rng(12)
    dt_s = 0.2
    times = np.arange(120) * dt_s
    truth = np.column_stack((100.0 + 8.0 * times, -40.0 + 3.0 * times))
    measurements = truth + rng.normal(0.0, 20.0, size=truth.shape)
    filter_ = ConstantVelocityKalmanFilter.from_position(
        measurements[0],
        position_std_m=20.0,
        velocity_std_mps=50.0,
        process_acceleration_std_mps2=1.0,
        measurement_std_m=20.0,
    )
    estimates = [filter_.state[:2].copy()]
    for measurement in measurements[1:]:
        filter_.predict(dt_s)
        filter_.update(measurement)
        estimates.append(filter_.state[:2].copy())

    raw_rmse = float(np.sqrt(np.mean(np.sum((measurements - truth) ** 2, axis=1))))
    filtered_rmse = float(
        np.sqrt(np.mean(np.sum((np.asarray(estimates) - truth) ** 2, axis=1)))
    )
    assert filtered_rmse < raw_rmse * 0.55


def test_nearest_neighbor_association_is_one_to_one_and_gated() -> None:
    tracks = (
        Track.initial(1, np.array([0.0, 0.0]), timestamp_s=0.0),
        Track.initial(2, np.array([100.0, 0.0]), timestamp_s=0.0),
    )
    measurements = np.array([[3.0, 1.0], [96.0, -2.0], [5_000.0, 5_000.0]])

    result = associate_nearest_neighbor(
        tracks, measurements, measurement_variance_m2=25.0, gate_threshold=9.21
    )

    assert result.matches == ((0, 0), (1, 1))
    assert result.unassigned_track_indices == ()
    assert result.unassigned_measurement_indices == (2,)


def test_tracker_lifecycle_confirms_coasts_and_deletes_without_id_change() -> None:
    tracker = MultiTargetTracker(
        TrackerConfig(confirm_hits=2, max_missed_updates=1, measurement_std_m=5.0)
    )

    first = tracker.update([detection(100.0, 50.0, 0.0)], timestamp_s=0.0)
    track_id = first[0].track_id
    assert first[0].status is TrackStatus.TENTATIVE

    second = tracker.update([detection(102.0, 50.0, 1.0)], timestamp_s=1.0)
    assert second[0].track_id == track_id
    assert second[0].status is TrackStatus.CONFIRMED

    coasted = tracker.update([], timestamp_s=2.0)
    assert coasted[0].track_id == track_id
    assert coasted[0].status is TrackStatus.COASTING
    assert coasted[0].misses == 1

    assert tracker.update([], timestamp_s=3.0) == ()
    assert tracker.deleted_tracks[-1].track_id == track_id
    assert tracker.deleted_tracks[-1].status is TrackStatus.DELETED


def test_two_targets_keep_stable_ids_and_position_histories() -> None:
    tracker = MultiTargetTracker(
        TrackerConfig(confirm_hits=2, max_missed_updates=2, measurement_std_m=3.0)
    )
    ids_by_side: list[tuple[int, int]] = []
    for frame in range(8):
        timestamp = float(frame)
        tracks = tracker.update(
            [
                detection(-500.0 + 15.0 * frame, -100.0, timestamp),
                detection(500.0 - 12.0 * frame, 100.0, timestamp),
            ],
            timestamp_s=timestamp,
        )
        ordered = sorted(tracks, key=lambda track: track.x_m)
        ids_by_side.append((ordered[0].track_id, ordered[1].track_id))

    assert len(set(ids_by_side)) == 1
    assert all(track.status is TrackStatus.CONFIRMED for track in tracks)
    assert all(len(track.history) == 8 for track in tracks)


def test_tracker_api_and_track_model_have_no_truth_identity_input() -> None:
    parameters = inspect.signature(MultiTargetTracker.update).parameters
    fields = Track.__dataclass_fields__

    assert "targets" not in parameters
    assert "ground_truth" not in parameters
    assert "target_id" not in fields
    assert "truth_id" not in fields


def test_directional_sensor_frames_publish_measured_bearing_without_target_ids() -> (
    None
):
    config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.0))
    sensor = RadarSensor(config, random_seed=8, bearing_noise_std_rad=0.0)
    targets = (Target("secret-a", 1_000.0, 0.0), Target("secret-b", 0.0, 2_000.0))

    frames = sensor.acquire_directional(targets, timestamp_s=1.5)

    assert len(frames) == 2
    assert [frame.bearing_rad for frame in frames] == pytest.approx([0.0, np.pi / 2.0])
    assert all(not hasattr(frame, "target_id") for frame in frames)
    assert all(not hasattr(frame, "target_state") for frame in frames)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_forms_confirmed_track_and_draws_history(app: QApplication) -> None:
    sensor_config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.001))
    window = MainWindow(
        world=single_stationary_target(range_m=1_500.0, bearing_rad=0.4),
        simulation_config=SimulationConfig(dt_s=0.1, random_seed=3),
        sensor_config=sensor_config,
    )

    for _ in range(3):
        window.step_once()

    confirmed = [
        track for track in window.last_tracks if track.status is TrackStatus.CONFIRMED
    ]
    assert confirmed
    assert len(window.ppi_view.track_history_items) >= 1
    x_data, y_data = window.ppi_view.track_item.getData()
    assert len(x_data) == len(y_data) >= 1
    assert window.track_table.table.rowCount() >= 1
    window.close()
