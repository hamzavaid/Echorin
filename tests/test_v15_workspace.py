"""Engineering source selection must not disguise bistatic measurement units."""

import json
import os
from threading import Event
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from echorin.config import NoiseConfig, SensorConfig
from echorin.gui.main_window import MainWindow
from echorin.recording import FrameRecorder
from echorin.simulation.scenarios import bistatic_scenario


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_source_dock_selects_full_products_and_preserves_replay_metadata(
    mode, tmp_path
):
    app = QApplication.instance() or QApplication([])
    config = getattr(SensorConfig, mode)(
        max_range_m=3000 if mode == "radar" else 60, noise_model=NoiseConfig(0.0001)
    )
    window = MainWindow(
        world=bistatic_scenario(config, multistatic=True),
        sensor_config=config,
        settings=QSettings(str(tmp_path / "network.ini"), QSettings.Format.IniFormat),
    )
    try:
        window.step_once()
        assert len(window.last_frame_result.source_frames) == 4
        assert window.network_controls.source_combo.count() == 4
        assert "Half-path" in window.range_doppler_view.plot.getAxis("bottom").labelText
        assert (
            "Half-path rate" in window.range_doppler_view.plot.getAxis("left").labelText
        )
        first = window.last_sensor_frame.received_signal.copy()
        window.network_controls.source_combo.setCurrentIndex(3)
        assert window.last_range_angle_product.emitter_id == "tx-2"
        assert window.last_range_angle_product.receiver_id == "rx-2"
        detection = window.last_detections[0]
        window.inspector.select_detection(0)
        fields = window.inspector.visible_fields()
        assert fields["Emitter"] == "tx-2"
        assert "Total path length" in fields and "Path rate" in fields
        assert fields["Radial velocity"] == "Unavailable"
        np.testing.assert_allclose(
            window.range_angle_view.detection_item.getData()[0],
            [d.display_range_m for d in window.last_detections],
        )
        assert len(window.ppi_view.device_label_items) == 4
        recorder = FrameRecorder()
        recorder.capture(window.last_frame_result)
        path = tmp_path / "frames.json"
        recorder.save_json(path)
        loaded = FrameRecorder.load_json(path)
        assert len(loaded.records[0]["source_frames"]) == 4
        assert loaded.records[0]["detections"][0]["emitter_id"] == detection.emitter_id
        recorder.export_csv(tmp_path / "frames.csv")
        assert "emitter_id" in (tmp_path / "frames.csv").read_text()
        window.reset()
        window.network_controls.source_combo.setCurrentIndex(0)
        window.step_once()
        np.testing.assert_array_equal(first, window.last_sensor_frame.received_signal)
    finally:
        window.close()
        app.processEvents()


def test_invalid_network_editor_is_atomic_and_presets_are_available(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        settings=QSettings(str(tmp_path / "edit.ini"), QSettings.Format.IniFormat)
    )
    try:
        window.network_controls.preset_combo.setCurrentText("Bistatic")
        window.network_controls.preset_button.click()
        assert len(window.world.sensor_platforms) == 2
        old = window.sensor
        data = json.loads(window.network_controls.config_edit.toPlainText())
        data.append(data[0])
        window.network_controls.config_edit.setPlainText(json.dumps(data))
        window.network_controls.apply_button.click()
        assert "unique" in window.network_controls.error_label.text()
        assert window.sensor is old
        window.network_controls.preset_combo.setCurrentText("Monostatic")
        window.network_controls.preset_button.click()
        assert window.world.sensor_platforms == ()
        window.step_once()
        assert window.last_detections
    finally:
        window.close()
        app.processEvents()


def test_multistatic_live_worker_is_responsive_and_reset_discards_stale_frame(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = SensorConfig(max_range_m=3000)
    window = MainWindow(
        world=bistatic_scenario(config, multistatic=True),
        sensor_config=config,
        settings=QSettings(str(tmp_path / "live.ini"), QSettings.Format.IniFormat),
    )
    entered, release = Event(), Event()
    original = window.sensor.acquire_array_pulse_train

    def gated(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test gate not released")
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
        window.reset()
        release.set()
        window._executor.shutdown(wait=True)
        for _ in range(3):
            app.processEvents()
            sleep(0.005)
        assert window.last_frame_result is None
    finally:
        release.set()
        window._executor.shutdown(wait=True)
        window.close()
        app.processEvents()
