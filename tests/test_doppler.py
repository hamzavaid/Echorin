"""Milestone 7 tests for coherent pulse trains and Doppler estimation."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import NoiseConfig, SensorConfig, SimulationConfig
from echorin.dsp.doppler import (
    DopplerProduct,
    doppler_axis,
    doppler_spectrum,
    enrich_detections_with_velocity,
)
from echorin.dsp.matched_filter import matched_filter
from echorin.gui.main_window import MainWindow
from echorin.gui.signal_plots import SignalPlots
from echorin.models.detection import Detection
from echorin.sensors.radar import RadarSensor
from echorin.signals.propagation import sample_delay
from echorin.simulation.scenarios import single_stationary_target
from echorin.simulation.target import Target


def doppler_config(**overrides: object) -> SensorConfig:
    values: dict[str, object] = {
        "sample_rate_hz": 250e6,
        "carrier_frequency_hz": 100e6,
        "bandwidth_hz": 10e6,
        "pulse_width_s": 1e-6,
        "prf_hz": 1_000.0,
        "max_range_m": 1_000.0,
        "noise_model": NoiseConfig(standard_deviation=0.0),
    }
    values.update(overrides)
    return SensorConfig(**values)


def test_doppler_axis_is_fft_shifted_and_has_prf_resolution() -> None:
    axis = doppler_axis(pulse_count=8, prf_hz=800.0)

    np.testing.assert_allclose(
        axis, [-400.0, -300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0]
    )


def test_slow_time_fft_recovers_an_exact_doppler_bin() -> None:
    config = doppler_config(prf_hz=1_600.0)
    pulse_count = 64
    expected_bin_offset = 7
    expected_hz = expected_bin_offset * config.prf_hz / pulse_count
    slow_time = np.arange(pulse_count) / config.prf_hz
    pulse_matrix = np.exp(1j * 2.0 * np.pi * expected_hz * slow_time)[:, None]

    product = doppler_spectrum(pulse_matrix, config, apply_window=False)
    peak_index = int(np.argmax(product.magnitude[:, 0]))

    assert isinstance(product, DopplerProduct)
    assert product.doppler_frequency_hz[peak_index] == pytest.approx(expected_hz)
    expected_velocity = (
        expected_hz * config.propagation_speed_mps / config.carrier_frequency_hz / 2.0
    )
    assert product.radial_velocity_mps[peak_index] == pytest.approx(expected_velocity)


def test_radar_pulse_train_estimates_known_radial_velocity_within_fft_bin() -> None:
    config = doppler_config()
    pulse_count = 64
    velocity_resolution = (
        config.prf_hz
        / pulse_count
        * config.propagation_speed_mps
        / config.carrier_frequency_hz
        / 2.0
    )
    true_velocity_mps = 2.0 * velocity_resolution
    target = Target("not-published", 500.0, 0.0, vx_mps=true_velocity_mps)
    sensor = RadarSensor(config, random_seed=4)

    frame = sensor.acquire_pulse_train(
        [target], timestamp_s=0.0, pulse_count=pulse_count
    )
    range_responses = np.stack(
        [
            matched_filter(pulse, frame.transmitted_signal)
            for pulse in frame.received_pulses
        ]
    )
    product = doppler_spectrum(range_responses, config, apply_window=False)
    range_bin = sample_delay(500.0, config)
    estimated_velocity = product.radial_velocity_mps[
        int(np.argmax(product.magnitude[:, range_bin]))
    ]

    assert frame.received_pulses.shape == (pulse_count, config.acquisition_samples)
    assert abs(estimated_velocity - true_velocity_mps) <= velocity_resolution / 2.0
    assert not hasattr(frame, "target_ids")


def test_pulse_train_rejects_doppler_above_nyquist() -> None:
    config = doppler_config(prf_hz=100.0)
    nyquist_velocity = (
        config.prf_hz
        / 2.0
        * config.propagation_speed_mps
        / config.carrier_frequency_hz
        / 2.0
    )
    sensor = RadarSensor(config)

    with pytest.raises(ValueError, match="unambiguous Doppler"):
        sensor.acquire_pulse_train(
            [Target("fast", 500.0, 0.0, vx_mps=nyquist_velocity * 1.01)],
            timestamp_s=0.0,
            pulse_count=32,
        )


def test_doppler_product_enriches_detection_from_its_range_bin() -> None:
    config = doppler_config(prf_hz=800.0)
    pulse_count = 32
    responses = np.zeros((pulse_count, 20), dtype=np.complex128)
    frequency_hz = 3.0 * config.prf_hz / pulse_count
    responses[:, 7] = np.exp(
        1j * 2.0 * np.pi * frequency_hz * np.arange(pulse_count) / config.prf_hz
    )
    product = doppler_spectrum(responses, config, apply_window=False)
    original = Detection(0.0, 100.0, 0.2, None, 4.0, 10.0, 0.8, 7)

    enriched = enrich_detections_with_velocity([original], product)

    expected_velocity = (
        frequency_hz * config.propagation_speed_mps / config.carrier_frequency_hz / 2.0
    )
    assert enriched[0].radial_velocity_mps == pytest.approx(expected_velocity)
    assert original.radial_velocity_mps is None


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_signal_plot_displays_selected_range_doppler_spectrum(
    app: QApplication,
) -> None:
    config = doppler_config()
    matrix = np.ones((16, 5), dtype=np.complex128)
    product = doppler_spectrum(matrix, config)
    plot = SignalPlots()

    plot.set_doppler_product(product, selected_range_bin=2)

    x_data, y_data = plot.doppler_curve.getData()
    np.testing.assert_array_equal(x_data, product.radial_velocity_mps)
    np.testing.assert_array_equal(y_data, product.magnitude[:, 2])
    plot.close()


def test_main_window_publishes_live_doppler_product(app: QApplication) -> None:
    window = MainWindow(
        world=single_stationary_target(range_m=1_500.0),
        simulation_config=SimulationConfig(dt_s=0.1, random_seed=2),
    )

    window.step_once()

    assert window.last_doppler_product is not None
    assert window.last_doppler_product.spectrum.shape[0] == window.doppler_pulse_count
    x_data, _ = window.signal_plots.doppler_curve.getData()
    assert len(x_data) == window.doppler_pulse_count
    window.close()
