"""Milestone 8 tests for the shared acoustic sensing pipeline."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import (
    NOMINAL_WATER_SOUND_SPEED_MPS,
    NoiseConfig,
    SensorConfig,
    SensorMode,
)
from echorin.dsp.doppler import doppler_spectrum
from echorin.dsp.range_processing import SignalProcessor
from echorin.gui.main_window import MainWindow
from echorin.sensors import RadarSensor, Sensor, SonarSensor, create_sensor
from echorin.signals.propagation import sample_delay
from echorin.simulation.scenarios import single_stationary_target
from echorin.simulation.target import Target


def test_sonar_default_configuration_is_physically_valid() -> None:
    config = SensorConfig.sonar()

    assert config.mode is SensorMode.SONAR
    assert config.propagation_speed_mps == NOMINAL_WATER_SOUND_SPEED_MPS
    assert config.carrier_frequency_hz < config.sample_rate_hz / 2.0
    assert config.max_range_m <= config.unambiguous_range_m
    assert config.acquisition_samples > config.pulse_samples


def test_mode_specific_config_requires_correct_propagation_speed() -> None:
    with pytest.raises(ValueError, match="sonar propagation"):
        SensorConfig(mode=SensorMode.SONAR)


def test_factory_returns_common_sensor_abstraction_for_both_modes() -> None:
    radar = create_sensor(SensorConfig.radar(), random_seed=1)
    sonar = create_sensor(SensorConfig.sonar(), random_seed=1)

    assert isinstance(radar, Sensor)
    assert isinstance(sonar, Sensor)
    assert isinstance(radar, RadarSensor)
    assert isinstance(sonar, SonarSensor)


def test_radar_and_sonar_recover_same_geometry_with_mode_specific_delays() -> None:
    true_range_m = 100.0
    target = Target("private", true_range_m, 0.0)
    radar_config = SensorConfig.radar(
        max_range_m=500.0, noise_model=NoiseConfig(standard_deviation=0.0)
    )
    sonar_config = SensorConfig.sonar(
        max_range_m=200.0, noise_model=NoiseConfig(standard_deviation=0.0)
    )
    radar = create_sensor(radar_config, random_seed=2)
    sonar = create_sensor(sonar_config, random_seed=2)

    radar_frame = radar.acquire([target], timestamp_s=0.0)
    sonar_frame = sonar.acquire([target], timestamp_s=0.0)
    radar_profile = SignalProcessor(radar_config).range_profile(
        radar_frame.received_signal, radar_frame.transmitted_signal
    )
    sonar_profile = SignalProcessor(sonar_config).range_profile(
        sonar_frame.received_signal, sonar_frame.transmitted_signal
    )
    radar_estimate = radar_profile.ranges_m[int(np.argmax(radar_profile.magnitude))]
    sonar_estimate = sonar_profile.ranges_m[int(np.argmax(sonar_profile.magnitude))]

    assert sample_delay(true_range_m, sonar_config) > sample_delay(
        true_range_m, radar_config
    )
    assert abs(radar_estimate - true_range_m) <= (
        radar_config.propagation_speed_mps / (2.0 * radar_config.sample_rate_hz)
    )
    assert abs(sonar_estimate - true_range_m) <= (
        sonar_config.propagation_speed_mps / (2.0 * sonar_config.sample_rate_hz)
    )


def test_sonar_uses_shared_coherent_doppler_pipeline() -> None:
    config = SensorConfig.sonar(
        prf_hz=8.0,
        max_range_m=80.0,
        noise_model=NoiseConfig(standard_deviation=0.0),
    )
    pulse_count = 32
    velocity_resolution = (
        config.prf_hz
        / pulse_count
        * config.propagation_speed_mps
        / config.carrier_frequency_hz
        / 2.0
    )
    true_velocity = 3.0 * velocity_resolution
    sensor = SonarSensor(config, random_seed=3)
    frame = sensor.acquire_pulse_train(
        [Target("private", 50.0, 0.0, vx_mps=true_velocity)],
        timestamp_s=0.0,
        pulse_count=pulse_count,
    )
    responses = SignalProcessor(config).pulse_matrix_range_responses(
        frame.received_pulses, frame.transmitted_signal
    )
    product = doppler_spectrum(responses, config, apply_window=False)
    range_bin = sample_delay(50.0, config)
    estimate = product.radial_velocity_mps[
        int(np.argmax(product.magnitude[:, range_bin]))
    ]

    assert abs(estimate - true_velocity) <= velocity_resolution / 2.0


def test_seeded_sonar_frames_are_repeatable() -> None:
    config = SensorConfig.sonar(noise_model=NoiseConfig(standard_deviation=0.01))
    target = [Target("private", 100.0, 0.0)]
    first = SonarSensor(config, random_seed=55).acquire(target, 1.0)
    second = SonarSensor(config, random_seed=55).acquire(target, 1.0)

    np.testing.assert_array_equal(first.received_signal, second.received_signal)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_gui_switches_mode_and_runs_shared_sonar_pipeline(app: QApplication) -> None:
    window = MainWindow(world=single_stationary_target(range_m=100.0))

    window.controls.mode_combo.setCurrentText("Sonar")
    window.step_once()

    assert window.sensor_config.mode is SensorMode.SONAR
    assert isinstance(window.sensor, SonarSensor)
    assert window.last_range_profile is not None
    assert window.last_doppler_product is not None
    assert window.ppi_view.max_range_m == window.sensor_config.max_range_m
    window.controls.mode_combo.setCurrentText("Radar")
    assert isinstance(window.sensor, RadarSensor)
    window.close()
