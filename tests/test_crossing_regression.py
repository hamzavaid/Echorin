"""Regression coverage for observable crossing targets and angular false alarms."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig, CfarResult
from echorin.dsp.range_angle import RangeAngleProduct, angle_detections
from echorin.dsp.range_processing import RangeProfile, SignalProcessor
from echorin.dsp.sidelobes import suppress_matched_filter_sidelobes
from echorin.models.detection import Detection
from echorin.models.geometry import SensorPose
from echorin.models.track import Track
from echorin.sensors.factory import create_sensor
from echorin.simulation.scenarios import crossing_targets, single_stationary_target
from echorin.simulation.target import Target
from echorin.tracking.association import associate_nearest_neighbor
from echorin.tracking.kalman import ConstantVelocityKalmanFilter
from echorin.tracking.tracker import (
    MultiTargetTracker,
    TrackerConfig,
    measurement_covariance_for_detection,
)


def _production_frames(world, count: int):
    config = SensorConfig()
    sensor = create_sensor(config, world.sensor_pose, random_seed=7)
    processor = SignalProcessor(config)
    cfar = CaCfarDetector(
        CfarConfig(
            training_cells=16,
            guard_cells=4,
            false_alarm_probability=1e-3,
            minimum_separation_bins=max(1, config.pulse_samples // 8),
        )
    )
    tracker = MultiTargetTracker(
        TrackerConfig(
            measurement_std_m=max(
                5.0, config.propagation_speed_mps / (2 * config.sample_rate_hz)
            ),
            bearing_std_rad=world.array_config.nominal_bearing_std_rad,
        )
    )
    results = []
    for _ in range(count):
        world.advance(0.05)
        sensor.pose = world.sensor_pose
        results.append(
            process_frame(
                world.targets,
                world.time_s,
                sensor,
                processor,
                cfar,
                tracker,
                32,
            )
        )
    return results


def test_crossing_preset_starts_with_two_observable_targets() -> None:
    world = crossing_targets(seed=7)
    first_target, second_target = world.targets
    first_crossing_time = (second_target.y_m - first_target.y_m) / first_target.vy_mps
    crossing_x = first_target.x_m + first_crossing_time * first_target.vx_mps
    second_crossing_time = (crossing_x - second_target.x_m) / second_target.vx_mps
    assert first_crossing_time > 0.0
    assert second_crossing_time > 0.0
    assert first_crossing_time - second_crossing_time > 3.0
    initial_positions = [
        np.array([target.x_m + 0.05 * target.vx_mps, target.y_m + 0.05 * target.vy_mps])
        for target in world.targets
    ]
    results = _production_frames(world, 160)
    first = results[0]
    assert len(first.detections) == 2
    assert len({round(d.range_m) for d in first.detections}) == 2
    for position in initial_positions:
        error = min(
            np.linalg.norm(d.world_position_m - position) for d in first.detections
        )
        assert error < 100.0
    confirmed = [
        track for track in results[-1].tracks if track.status.value == "confirmed"
    ]
    assert len(confirmed) == 2
    assert all(len(frame.detections) == 2 for frame in results)
    assert all(
        {track.track_id for track in frame.tracks if track.status.value == "confirmed"}
        == {1, 2}
        for frame in results[3:]
    )


def test_noise_only_angular_peaks_do_not_spawn_multiple_detections() -> None:
    rng = np.random.default_rng(18)
    power = rng.exponential(1.0, (181, 128))
    power[100, 20] = 1_000.0
    product = RangeAngleProduct(
        np.arange(128, dtype=float),
        np.linspace(-np.pi / 2, np.pi / 2, 181),
        power,
        0.0,
    )
    candidates = (
        Detection(0.0, 20.0, float("nan"), None, 1.0, 20.0, 0.8, 20),
        Detection(0.0, 80.0, float("nan"), None, 1.0, 4.0, 0.2, 80),
    )
    detections = angle_detections(candidates, product)
    assert len(detections) == 1
    assert detections[0].source_bin == 20


def test_one_isolated_target_does_not_become_multiple_confirmed_tracks() -> None:
    results = _production_frames(single_stationary_target(2_000.0, 0.2), 12)
    assert len([t for t in results[-1].tracks if t.status.value == "confirmed"]) == 1
    assert all(len(frame.detections) <= 1 for frame in results)


def test_sidelobe_filter_keeps_a_second_physical_echo() -> None:
    waveform = np.exp(1j * np.linspace(0, 5 * np.pi, 100) ** 2)
    correlation = fftconvolve(waveform, waveform[::-1].conj(), mode="full")
    template = correlation / correlation[99]
    response = np.zeros(400, dtype=np.complex128)
    response[1:200] += 10.0 * template
    response[71:270] += 7.0 * template
    candidates = (100, 170, 195)
    cfar = CfarResult(
        threshold=np.full(400, 0.5),
        noise_power=np.full(400, 0.1),
        detections=tuple(
            Detection(
                0.0, float(index), 0.0, None, abs(response[index]), 20, 0.9, index
            )
            for index in candidates
        ),
    )
    filtered = suppress_matched_filter_sidelobes(
        cfar, RangeProfile(np.arange(400, dtype=float), response), waveform
    )
    assert {d.source_bin for d in filtered.detections} == {100, 170}


def test_real_targets_inside_pulse_extent_are_not_suppressed() -> None:
    config = SensorConfig.radar(
        max_range_m=4_000.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    result = process_frame(
        (Target("private-a", 1_800.0, 0.0), Target("private-b", 2_200.0, 0.0)),
        0.05,
        create_sensor(config, random_seed=7),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        32,
    )
    assert len(result.detections) == 2
    assert [d.range_m for d in result.detections] == sorted(
        d.range_m for d in result.detections
    )
    assert abs(result.detections[0].range_m - 1_800) < 15
    assert abs(result.detections[1].range_m - 2_200) < 15


def test_bearing_uncertainty_grows_with_range_and_rotates_with_receiver() -> None:
    near = Detection(0.0, 100.0, 0.0, None, 1.0, 20.0, 0.8, 1)
    far = Detection(0.0, 3_000.0, 0.0, None, 1.0, 20.0, 0.8, 1)
    near_covariance = measurement_covariance_for_detection(near, 15.0, 0.04)
    far_covariance = measurement_covariance_for_detection(far, 15.0, 0.04)
    np.testing.assert_allclose(near_covariance, np.diag([225.0, 225.0]))
    np.testing.assert_allclose(far_covariance, np.diag([225.0, 14_400.0]))
    rotated = Detection(
        0.0,
        3_000.0,
        0.0,
        None,
        1.0,
        20.0,
        0.8,
        1,
        sensor_pose=SensorPose(heading_rad=np.pi / 2),
    )
    np.testing.assert_allclose(
        measurement_covariance_for_detection(rotated, 15.0, 0.04),
        np.diag([14_400.0, 225.0]),
        atol=1e-10,
    )


def test_crossing_seed_variation_preserves_two_confirmed_track_ids() -> None:
    results = _production_frames(crossing_targets(seed=19), 120)
    assert all(len(frame.detections) == 2 for frame in results)
    assert all(
        {track.track_id for track in frame.tracks if track.status.value == "confirmed"}
        == {1, 2}
        for frame in results[3:]
    )


def test_association_and_kalman_use_per_detection_covariance() -> None:
    track = Track.initial(1, np.zeros(2), 0.0, position_std_m=5.0)
    measurement = np.array([[0.0, 100.0]])
    isotropic = associate_nearest_neighbor([track], measurement, 15.0**2, 9.21)
    anisotropic = associate_nearest_neighbor(
        [track],
        measurement,
        15.0**2,
        9.21,
        measurement_covariances_m2=np.array([np.diag([225.0, 14_400.0])]),
    )
    assert isotropic.matches == ()
    assert anisotropic.matches == ((0, 0),)
    filter_ = ConstantVelocityKalmanFilter(
        np.zeros(4), np.eye(4) * 400.0, measurement_std_m=15.0
    )
    filter_.update(np.array([100.0, 100.0]), np.diag([225.0, 14_400.0]))
    assert filter_.state[0] > 50.0
    assert filter_.state[1] < 10.0
