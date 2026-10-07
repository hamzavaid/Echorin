"""Configuration edits retain reset semantics and reject invalid updates."""

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from echorin.config import SensorConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.gui.main_window import MainWindow
from echorin.models.geometry import SensorPose
from echorin.sensors.components import Emitter, Receiver
from echorin.sensors.factory import create_link_sensor
from echorin.signals.waveform import WaveformKind
from echorin.simulation.scenarios import bistatic_scenario


def test_receiver_noise_override_is_independent_of_transmitter():
    config = SensorConfig()
    receiver = Receiver("rx", config, noise_model=ReceiverNoiseConfig(kind="colored"))
    sensor = create_link_sensor(
        Emitter("tx", config), receiver, SensorPose(), SensorPose()
    )
    assert sensor.environment.receiver_noise.kind == "colored"


def test_network_noise_waveform_and_mode_changes_survive_reset(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = SensorConfig(max_range_m=3000)
    window = MainWindow(
        world=bistatic_scenario(config),
        sensor_config=config,
        settings=QSettings(str(tmp_path / "changes.ini"), QSettings.Format.IniFormat),
    )
    try:
        window.controls.noise_spin.setValue(0.003)
        window.controls.waveform_combo.setCurrentText("Rectangular")
        window.reset()
        assert window.sensor.config.noise_model.standard_deviation == 0.003
        assert window.sensor.waveform_kind == WaveformKind.RECTANGULAR
        window.controls.mode_combo.setCurrentText("Sonar")
        window.reset()
        assert window.sensor.config.mode.value == "sonar"
        window.network_controls.preset_combo.setCurrentText("Bistatic")
        window.network_controls.preset_button.click()
        window.step_once()
        assert window.last_detections
        assert window.last_range_angle_product.is_bistatic
    finally:
        window.close()
        app.processEvents()


def test_environment_rejects_aliasing_for_nondefault_network_sample_config(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = SensorConfig(sample_rate_hz=4_000_000)
    window = MainWindow(
        world=bistatic_scenario(config),
        settings=QSettings(str(tmp_path / "alias.ini"), QSettings.Format.IniFormat),
    )
    try:
        old = window.sensor
        env = EnvironmentConfig(interference=(NarrowbandInterference(3_000_000),))
        window._set_environment(env, window.world.beam_config)
        assert "Nyquist" in window.environment_controls.error_label.text()
        assert window.sensor is old
        assert window.world.environment_config == EnvironmentConfig()
    finally:
        window.close()
        app.processEvents()


def test_duplicate_or_nonfinite_device_configuration_rejected():
    config = SensorConfig()
    for name in ("sample_rate_hz", "carrier_frequency_hz"):
        with pytest.raises(ValueError, match="finite"):
            Emitter("bad", replace(config, **{name: float("nan")}))
