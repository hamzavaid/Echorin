"""Milestone 4 tests for matched filtering and independent range estimates."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.matched_filter import matched_filter
from echorin.dsp.range_processing import (
    FixedThresholdDetector,
    RangeProfile,
    SignalProcessor,
    range_axis,
)
from echorin.models.detection import Detection
from echorin.sensors.radar import RadarSensor
from echorin.signals.waveform import lfm_chirp
from echorin.simulation.target import Target


def test_matched_filter_peak_occurs_at_injected_delay() -> None:
    config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.0))
    transmitted = lfm_chirp(config)
    injected_delay = 137
    received = np.zeros(500)
    received[injected_delay : injected_delay + transmitted.size] = transmitted

    response = matched_filter(received, transmitted)

    assert response.shape == received.shape
    assert int(np.argmax(np.abs(response))) == injected_delay


def test_range_axis_maps_sample_delay_to_two_way_distance() -> None:
    axis = range_axis(4, sample_rate_hz=10e6, propagation_speed_mps=300e6)

    np.testing.assert_allclose(axis, [0.0, 15.0, 30.0, 45.0])


def test_signal_processor_builds_aligned_range_profile() -> None:
    config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.0))
    transmitted = lfm_chirp(config)
    received = np.zeros(config.acquisition_samples)
    received[25 : 25 + transmitted.size] = 0.5 * transmitted
    processor = SignalProcessor(config)

    profile = processor.range_profile(received, transmitted)

    assert isinstance(profile, RangeProfile)
    assert profile.response.shape == received.shape
    assert profile.magnitude.shape == received.shape
    assert profile.ranges_m[25] == pytest.approx(
        25 * config.propagation_speed_mps / (2.0 * config.sample_rate_hz)
    )
    assert int(np.argmax(profile.magnitude)) == 25


def test_fixed_threshold_detector_returns_local_peaks_with_measurement_fields() -> None:
    response = np.zeros(30, dtype=np.complex128)
    response[5] = 3.0
    response[6] = 2.0  # Above threshold but not a separate local maximum.
    response[20] = 5.0
    profile = RangeProfile(np.arange(30, dtype=float) * 10.0, response)
    detector = FixedThresholdDetector(threshold=1.0, minimum_separation_bins=2)

    detections = detector.detect(profile, timestamp_s=4.0, bearing_rad=0.25)

    assert [d.source_bin for d in detections] == [5, 20]
    assert all(isinstance(d, Detection) for d in detections)
    assert detections[0].range_m == 50.0
    assert detections[0].bearing_rad == 0.25
    assert detections[0].radial_velocity_mps is None
    assert detections[0].amplitude == 3.0


def test_high_snr_sensor_range_estimate_is_within_one_range_bin() -> None:
    true_range_m = 2_500.0
    config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.001))
    sensor = RadarSensor(config, random_seed=5)
    frame = sensor.acquire(
        [Target("hidden-id", true_range_m, 0.0, reflectivity=1.0)],
        timestamp_s=1.0,
    )
    profile = SignalProcessor(config).range_profile(
        frame.received_signal, frame.transmitted_signal
    )
    detections = FixedThresholdDetector(
        threshold=0.5, minimum_separation_bins=config.pulse_samples // 2
    ).detect(profile, timestamp_s=frame.timestamp_s, bearing_rad=0.0)
    strongest = max(detections, key=lambda detection: detection.amplitude)
    range_bin_m = config.propagation_speed_mps / (2.0 * config.sample_rate_hz)

    assert abs(strongest.range_m - true_range_m) <= range_bin_m


def test_detection_boundary_contains_no_ground_truth_identity_or_target_input() -> None:
    fields = Detection.__dataclass_fields__
    parameters = inspect.signature(FixedThresholdDetector.detect).parameters

    assert "target_id" not in fields
    assert "ground_truth" not in fields
    assert "targets" not in parameters
    assert set(fields) == {
        "timestamp_s",
        "range_m",
        "bearing_rad",
        "radial_velocity_mps",
        "amplitude",
        "snr_db",
        "confidence",
        "source_bin",
    }
