"""Near-range Single preset and edge-aware CA-CFAR regressions."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import RangeProfile
from echorin.gui.main_window import MainWindow
from echorin.simulation.scenarios import single_stationary_target


def test_adaptive_cfar_uses_available_training_cells_at_near_edge() -> None:
    response = np.ones(80, dtype=np.complex128)
    response[1] = 20.0
    config = CfarConfig(
        training_cells=8,
        guard_cells=2,
        false_alarm_probability=1e-3,
        edge_mode="adaptive",
    )
    result = CaCfarDetector(config).detect(
        RangeProfile(np.arange(80, dtype=float), response), 0.0, 0.0
    )
    alpha = 8 * (config.false_alarm_probability ** (-1 / 8) - 1)
    assert result.noise_power[1] == pytest.approx(1.0)
    assert result.threshold[1] == pytest.approx(np.sqrt(alpha))
    assert [d.source_bin for d in result.detections] == [1]


@pytest.mark.parametrize("mode", ("Radar", "Sonar"))
def test_single_preset_detects_close_target_and_confirms_one_track(
    tmp_path, mode: str
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        settings=QSettings(
            str(tmp_path / f"single-{mode}.ini"), QSettings.Format.IniFormat
        )
    )
    window.controls.mode_combo.setCurrentText(mode)
    window.controls.preset_combo.setCurrentText("Single target")
    expected_range_m = 1_500.0 if mode == "Radar" else 100.0
    assert window.world.targets[0].x_m == expected_range_m
    for _ in range(3):
        window.step_once()
    assert len(window.last_detections) == 1
    detection = window.last_detections[0]
    range_bin_m = window.sensor_config.propagation_speed_mps / (
        2 * window.sensor_config.sample_rate_hz
    )
    assert abs(detection.range_m - expected_range_m) <= range_bin_m
    assert (
        len(
            [track for track in window.last_tracks if track.status.value == "confirmed"]
        )
        == 1
    )
    window.close()
    app.processEvents()


def test_radar_still_detects_a_genuinely_close_target(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        world=single_stationary_target(range_m=100.0),
        settings=QSettings(str(tmp_path / "close.ini"), QSettings.Format.IniFormat),
    )
    window.step_once()
    assert len(window.last_detections) == 1
    assert abs(window.last_detections[0].range_m - 100.0) <= 15.0
    window.close()
    app.processEvents()
