"""Diagnostics, theme persistence, and live Radar/Sonar smoke tests."""

from __future__ import annotations

import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
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


def test_both_modes_refresh_full_product_without_ui_stall(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "modes.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        world=single_stationary_target(range_m=100.0), settings=settings
    )
    for mode in ("Radar", "Sonar"):
        window.controls.mode_combo.setCurrentText(mode)
        started = perf_counter()
        window.step_once()
        app.processEvents()
        assert perf_counter() - started < 2.0
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
