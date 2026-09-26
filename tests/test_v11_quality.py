"""Diagnostics, theme persistence, and live Radar/Sonar smoke tests."""

from __future__ import annotations

import os
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QDockWidget

from echorin.config import SensorMode
from echorin.gui.main_window import MainWindow
from echorin.simulation.scenarios import single_stationary_target


def test_diagnostics_and_theme_are_live_and_persistent(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "quality.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        world=single_stationary_target(range_m=100.0), settings=settings
    )
    window.step_once()

    assert "Sensing" in window.diagnostics.text()
    assert "DSP" in window.diagnostics.text()
    assert "GUI" in window.diagnostics.text()
    assert "Radar" in window.statusBar().currentMessage()
    window.show()
    app.processEvents()
    prior_state = window.saveState()
    window.set_ppi_focus(True)
    assert all(not dock.isVisible() for dock in window.findChildren(QDockWidget))
    window.set_ppi_focus(False)
    assert not window.ppi_focus
    assert window.saveState() == prior_state
    window.apply_theme("light")
    assert window.theme == "light"
    window.close()

    reopened = MainWindow(settings=settings)
    assert reopened.theme == "light"
    reopened.close()
    app.processEvents()


def test_dark_theme_styles_native_headers_and_dock_tabs(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "labels.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.apply_theme("dark")

    stylesheet = window.styleSheet()
    assert "QHeaderView::section" in stylesheet
    assert "QTableCornerButton::section" in stylesheet
    assert "QTabBar::tab" in stylesheet
    assert "QTabBar::tab:selected" in stylesheet
    assert "selection-color" in stylesheet
    assert stylesheet.count("{") == stylesheet.count("}")
    window.close()
    app.processEvents()


def test_both_modes_refresh_full_product(tmp_path) -> None:
    """Manual steps publish complete signal products in both sensor modes."""
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "modes.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        world=single_stationary_target(range_m=100.0), settings=settings
    )
    for mode in ("Radar", "Sonar"):
        window.controls.mode_combo.setCurrentText(mode)
        window.step_once()
        app.processEvents()
        assert window.sensor_config.mode is SensorMode(mode.lower())
        assert window.last_doppler_product is not None
        assert window.range_doppler_view.image_item.image.shape == (
            window.doppler_pulse_count,
            window.sensor_config.acquisition_samples,
        )
        assert window.last_frame_result is not None
        if mode == "Sonar":
            assert window.timer.interval() >= 400
    window.reset()
    assert window.range_doppler_view.product is None
    assert window.range_doppler_view.image_item.image is None
    window.close()


@pytest.mark.parametrize("mode", ("Radar", "Sonar"))
def test_live_processing_keeps_qt_event_loop_responsive(tmp_path, mode: str) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "worker.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        world=single_stationary_target(range_m=100.0), settings=settings
    )
    window.controls.mode_combo.setCurrentText(mode)
    original = window.sensor.acquire_array_pulse_train

    def delayed_acquisition(*args, **kwargs):
        sleep(0.15)
        return original(*args, **kwargs)

    window.sensor.acquire_array_pulse_train = delayed_acquisition
    heartbeat: list[bool] = []
    QTimer.singleShot(30, lambda: heartbeat.append(window.last_frame_result is None))
    window.timer.setInterval(1)
    window.timer.start()
    deadline = perf_counter() + 10.0
    while not heartbeat and perf_counter() < deadline:
        app.processEvents()
        sleep(0.005)
    assert heartbeat == [True]
    while window.last_frame_result is None and perf_counter() < deadline:
        app.processEvents()
        sleep(0.005)
    assert window.last_frame_result is not None
    assert window.sensor_config.mode is SensorMode(mode.lower())
    assert window.last_doppler_product is not None
    window.timer.stop()
    window.close()


def test_reset_discards_an_inflight_frame(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "reset.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        world=single_stationary_target(range_m=100.0), settings=settings
    )
    original = window.sensor.acquire_array_pulse_train

    def delayed_acquisition(*args, **kwargs):
        sleep(0.12)
        return original(*args, **kwargs)

    window.sensor.acquire_array_pulse_train = delayed_acquisition
    window.timer.setInterval(1)
    window.timer.start()
    deadline = perf_counter() + 2.0
    while not window._inflight and perf_counter() < deadline:
        app.processEvents()
        sleep(0.005)
    assert window._inflight
    window.reset()
    assert window.world.time_s == 0.0
    window._executor.shutdown(wait=True)
    app.processEvents()
    assert window.last_frame_result is None
    assert window.range_doppler_view.product is None
    window.close()
