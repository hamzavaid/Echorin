"""Sensor-only detection and track inspection acceptance tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from echorin.gui.inspection_data import InspectionModel
from echorin.gui.main_window import MainWindow
from echorin.models.detection import Detection
from echorin.models.track import Track, TrackStatus
from echorin.simulation.scenarios import single_stationary_target


def test_inspection_model_selects_sensor_measurements_without_truth() -> None:
    detection = Detection(1.0, 100.0, np.pi / 2, -2.0, 4.5, 18.0, 0.8, 11)
    track = Track.initial(7, np.array([10.0, 20.0]), 1.0)
    track.state[2:] = [3.0, -4.0]
    track.status = TrackStatus.COASTING
    model = InspectionModel()
    model.update((detection,), (track,))

    measurement = model.select_detection(0)
    assert measurement is not None
    assert measurement.fields["Range"] == "100.000 m"
    assert measurement.fields["Cartesian X"] == "0.000 m"
    assert measurement.fields["Radial velocity"] == "-2.000 m/s"
    assert measurement.fields["Source bin"] == "11"
    assert "target_id" not in repr(measurement)

    estimate = model.select_track(7)
    assert estimate is not None
    assert estimate.fields["Track ID"] == "7"
    assert estimate.fields["Status"] == "coasting"
    assert estimate.fields["Speed"] == "5.000 m/s"
    assert "Position covariance" in estimate.fields
    model.update((detection,), (track,))
    assert model.current is not None
    assert model.current.fields["Track ID"] == "7"
    model.update((), ())
    assert model.current is None


def test_inspection_handles_unavailable_bearing_and_velocity() -> None:
    model = InspectionModel()
    model.update((Detection(0.0, 50.0, float("nan"), None,
                            1.0, 3.0, 0.2, 4),), ())
    details = model.select_detection(0)
    assert details is not None
    assert details.fields["Bearing"] == "Unavailable"
    assert details.fields["Cartesian X"] == "Unavailable"
    assert details.fields["Radial velocity"] == "Unavailable"


def test_gui_inspector_tracks_live_selection_and_reset() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(world=single_stationary_target(range_m=1_000.0))
    window.step_once()
    assert window.last_detections
    window.inspector.select_detection(0)
    assert "Range" in window.inspector.visible_fields()
    window.step_once()
    assert window.inspector.model.current is not None
    window.reset()
    assert window.inspector.model.current is None
    window.close()
    app.processEvents()
