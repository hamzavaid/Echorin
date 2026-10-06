"""Scenario migration and engineering controls for environmental realism."""

import json
import os
from threading import Event
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from echorin.config import SensorConfig, SimulationConfig
from echorin.environment.clutter import ClutterConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.gui.main_window import MainWindow
from echorin.propagation.multipath import MultipathComponent
from echorin.recording import load_scenario_json, save_scenario_json
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.scan import ScanConfig
from echorin.simulation.scenarios import single_stationary_target


def test_environment_and_scan_roundtrip_and_legacy_schema_migration(tmp_path):
    world = single_stationary_target()
    world.environment_config = EnvironmentConfig(
        receiver_noise=ReceiverNoiseConfig(kind="impulsive", event_probability=0.02),
        interference=(NarrowbandInterference(100, 0.01, drift_hz_s=1),),
        clutter=(ClutterConfig(density_per_m=0.001),),
        multipath=(MultipathComponent(300, 0.4, 0.5, 0.1),),
        absorption_db_per_m=0.0001,
    )
    world.beam_config = BeamConfig(kind="gaussian", scan=ScanConfig(kind="step"))
    path = tmp_path / "environment.json"
    save_scenario_json(path, world, SimulationConfig(), SensorConfig())
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 3
    restored, _, _ = load_scenario_json(path)
    assert restored.environment_config == world.environment_config
    assert restored.beam_config == world.beam_config
    payload["schema_version"] = 2
    payload.pop("environment")
    payload.pop("beam")
    path.write_text(json.dumps(payload))
    legacy, _, _ = load_scenario_json(path)
    assert legacy.environment_config == EnvironmentConfig()
    assert legacy.beam_config == BeamConfig()


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_gui_environment_apply_scan_overlay_and_reset(mode, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = getattr(SensorConfig, mode)()
    window = MainWindow(
        world=single_stationary_target(range_m=config.max_range_m / 3),
        sensor_config=config,
        settings=QSettings(
            str(tmp_path / "environment.ini"), QSettings.Format.IniFormat
        ),
    )
    controls = window.environment_controls
    controls.noise_combo.setCurrentText("colored")
    controls.beam_combo.setCurrentText("gaussian")
    controls.scan_combo.setCurrentText("sector")
    controls.apply_button.click()
    assert not controls.error_label.text()
    assert window.sensor.environment.receiver_noise.kind == "colored"
    assert window.sensor.beam.kind == "gaussian"
    window.step_once()
    assert window.last_frame_result is not None
    assert len(window.ppi_view.beam_items[0].xData) > 1
    assert "colored" in window.diagnostics.text()
    assert "Beam" in window.diagnostics.text()
    first = window.last_sensor_frame.received_signal.copy()
    window.reset()
    window.step_once()
    np.testing.assert_array_equal(first, window.last_sensor_frame.received_signal)
    window.close()
    app.processEvents()


def test_invalid_gui_environment_edit_preserves_running_configuration(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        settings=QSettings(str(tmp_path / "invalid.ini"), QSettings.Format.IniFormat)
    )
    old = window.sensor
    window.environment_controls.effects_edit.setPlainText("{invalid json")
    window.environment_controls.apply_button.click()
    assert window.environment_controls.error_label.text()
    assert window.sensor is old
    window.environment_controls.effects_edit.setPlainText(
        '{"interference": [{"frequency_hz": 1000000000}]}'
    )
    window.environment_controls.apply_button.click()
    assert "Nyquist" in window.environment_controls.error_label.text()
    assert window.sensor is old
    window.close()
    app.processEvents()


def test_mode_switch_rejects_incompatible_environment_without_partial_mutation(
    tmp_path,
):
    app = QApplication.instance() or QApplication([])
    world = single_stationary_target()
    world.environment_config = EnvironmentConfig(
        interference=(NarrowbandInterference(100_000),)
    )
    window = MainWindow(
        world=world,
        settings=QSettings(str(tmp_path / "mode.ini"), QSettings.Format.IniFormat),
    )
    old = window.sensor
    window.controls.mode_combo.setCurrentText("Sonar")
    assert window.sensor is old
    assert window.sensor_config.mode.value == "radar"
    assert window.controls.mode_combo.currentText() == "Radar"
    assert "Nyquist" in window.environment_controls.error_label.text()
    window.close()
    app.processEvents()


def test_beam_overlay_toggle_and_dark_effect_editor(tmp_path):
    app = QApplication.instance() or QApplication([])
    world = single_stationary_target()
    world.beam_config = BeamConfig(kind="sector")
    window = MainWindow(
        world=world,
        settings=QSettings(str(tmp_path / "theme.ini"), QSettings.Format.IniFormat),
    )
    assert window.ppi_view.beam_items[0].isVisible()
    window.environment_controls.beam_checkbox.setChecked(False)
    window.step_once()
    assert not window.ppi_view.beam_items[0].isVisible()
    assert "QPlainTextEdit" in window.styleSheet()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_environment_presets_use_medium_scaled_configurations(mode, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = getattr(SensorConfig, mode)()
    window = MainWindow(
        sensor_config=config,
        settings=QSettings(str(tmp_path / "presets.ini"), QSettings.Format.IniFormat),
    )
    controls = window.environment_controls
    controls.preset_combo.setCurrentText("Clutter / Reverberation")
    controls.apply_button.click()
    field = window.world.environment_config.clutter[0]
    assert field.density_per_m * config.max_range_m == pytest.approx(20)
    assert field.kind == ("clutter" if mode == "radar" else "reverberation")
    controls.preset_combo.setCurrentText("Multipath")
    controls.apply_button.click()
    assert (
        window.world.environment_config.multipath[0].extra_path_length_m
        < config.max_range_m
    )
    window.close()
    app.processEvents()


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_environment_live_worker_leaves_event_loop_free(mode, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = getattr(SensorConfig, mode)()
    world = single_stationary_target(range_m=config.max_range_m / 3)
    world.environment_config = EnvironmentConfig(
        receiver_noise=ReceiverNoiseConfig(kind="colored")
    )
    window = MainWindow(
        world=world,
        sensor_config=config,
        settings=QSettings(str(tmp_path / "live.ini"), QSettings.Format.IniFormat),
    )
    entered, release = Event(), Event()
    original = window.sensor.acquire_array_pulse_train

    def gated(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test failed to release acquisition gate")
        return original(*args, **kwargs)

    window.sensor.acquire_array_pulse_train = gated
    heartbeat = []
    try:
        window._request_live_step()
        assert entered.wait(5)
        QTimer.singleShot(0, lambda: heartbeat.append(window.last_frame_result is None))
        deadline = perf_counter() + 5
        while not heartbeat and perf_counter() < deadline:
            app.processEvents()
        assert heartbeat == [True]
        release.set()
        while window.last_frame_result is None and perf_counter() < deadline:
            app.processEvents()
            sleep(0.005)
        assert window.last_frame_result is not None
    finally:
        release.set()
        window._executor.shutdown(wait=True)
        window.close()
        app.processEvents()
