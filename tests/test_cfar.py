"""Milestone 5 analytical, statistical, and UI tests for CA-CFAR."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import SimulationConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import RangeProfile
from echorin.gui.main_window import MainWindow
from echorin.gui.signal_plots import SignalPlots
from echorin.simulation.scenarios import single_stationary_target


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_cfar_threshold_matches_ca_cfar_closed_form_for_uniform_power() -> None:
    config = CfarConfig(training_cells=8, guard_cells=2, false_alarm_probability=1e-3)
    profile = RangeProfile(np.arange(100, dtype=float), np.full(100, 2.0))

    result = CaCfarDetector(config).detect(profile, timestamp_s=0.0, bearing_rad=0.0)

    training_count = 2 * config.training_cells
    alpha = training_count * (
        config.false_alarm_probability ** (-1.0 / training_count) - 1.0
    )
    expected_amplitude_threshold = 2.0 * np.sqrt(alpha)
    cut = 50
    assert result.noise_power[cut] == pytest.approx(4.0)
    assert result.threshold[cut] == pytest.approx(expected_amplitude_threshold)
    edge = config.training_cells + config.guard_cells
    assert np.isnan(result.threshold[:edge]).all()
    assert np.isnan(result.threshold[-edge:]).all()


def test_guard_cells_keep_strong_target_out_of_its_noise_estimate() -> None:
    response = np.ones(101, dtype=np.complex128)
    response[48:53] = [5.0, 10.0, 50.0, 10.0, 5.0]
    profile = RangeProfile(np.arange(101, dtype=float) * 10.0, response)
    detector = CaCfarDetector(
        CfarConfig(training_cells=10, guard_cells=2, false_alarm_probability=1e-3)
    )

    result = detector.detect(profile, timestamp_s=3.0, bearing_rad=0.4)

    assert result.noise_power[50] == pytest.approx(1.0)
    assert [d.source_bin for d in result.detections] == [50]
    assert result.detections[0].range_m == 500.0
    assert result.detections[0].bearing_rad == 0.4


def test_noise_only_complex_gaussian_false_alarm_rate_is_bounded() -> None:
    rng = np.random.default_rng(2026)
    sample_count = 200_000
    response = (
        rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)
    ) / np.sqrt(2.0)
    profile = RangeProfile(np.arange(sample_count, dtype=float), response)
    config = CfarConfig(training_cells=20, guard_cells=4, false_alarm_probability=1e-2)

    result = CaCfarDetector(config).detect(profile, timestamp_s=0.0, bearing_rad=0.0)
    valid_count = sample_count - 2 * (config.training_cells + config.guard_cells)
    observed_rate = len(result.detections) / valid_count

    assert observed_rate <= config.false_alarm_probability * 1.1
    assert observed_rate > 0.0


@pytest.mark.parametrize(
    "config",
    (
        CfarConfig(training_cells=1),
        CfarConfig(guard_cells=-1),
        CfarConfig(false_alarm_probability=0.0),
        CfarConfig(false_alarm_probability=1.0),
        CfarConfig(minimum_separation_bins=0),
    ),
)
def test_cfar_configuration_rejects_invalid_parameters(config: CfarConfig) -> None:
    with pytest.raises(ValueError):
        CaCfarDetector(config)


def test_signal_plot_displays_profile_threshold_and_detection_markers(
    app: QApplication,
) -> None:
    plot = SignalPlots()
    profile = RangeProfile(
        np.array([0.0, 10.0, 20.0, 30.0]), np.array([0.1, 1.0, 0.2, 2.0])
    )
    threshold = np.array([np.nan, 0.8, 0.8, np.nan])

    plot.set_range_product(profile, threshold, detection_bins=[1, 3])

    px, py = plot.profile_curve.getData()
    tx, ty = plot.threshold_curve.getData()
    dx, dy = plot.detection_item.getData()
    np.testing.assert_array_equal(px, profile.ranges_m)
    np.testing.assert_array_equal(py, profile.magnitude)
    np.testing.assert_array_equal(tx, profile.ranges_m)
    np.testing.assert_array_equal(ty, threshold)
    np.testing.assert_array_equal(dx, [10.0, 30.0])
    np.testing.assert_array_equal(dy, [1.0, 2.0])
    plot.close()


def test_main_window_runs_live_signal_chain_and_publishes_cfar_product(
    app: QApplication,
) -> None:
    window = MainWindow(
        world=single_stationary_target(range_m=1_500.0),
        simulation_config=SimulationConfig(dt_s=0.1, random_seed=99),
    )

    window.step_once()

    assert window.last_sensor_frame is not None
    assert window.last_range_profile is not None
    assert window.last_cfar_result is not None
    assert window.last_sensor_frame.timestamp_s == pytest.approx(0.1)
    assert window.last_cfar_result.threshold.shape == (
        window.sensor_config.acquisition_samples,
    )
    x_data, _ = window.signal_plots.threshold_curve.getData()
    assert len(x_data) == window.sensor_config.acquisition_samples
    window.close()
