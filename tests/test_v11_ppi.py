"""Track overlay geometry and visibility tests for v1.1."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.gui.ppi_view import PpiView
from echorin.gui.visualization_data import covariance_ellipse, track_overlay
from echorin.models.track import Track, TrackStatus


def test_covariance_ellipse_matches_eigen_axes_and_rotation() -> None:
    covariance = np.array([[13.0, 12.0], [12.0, 13.0]])
    ellipse = covariance_ellipse(np.array([10.0, -5.0]), covariance)
    scale = np.sqrt(-2.0 * np.log(0.05))

    assert ellipse.major_radius_m == pytest.approx(5.0 * scale)
    assert ellipse.minor_radius_m == pytest.approx(scale)
    assert ellipse.angle_rad == pytest.approx(np.pi / 4)
    assert ellipse.x_m.shape == ellipse.y_m.shape
    assert ellipse.x_m.size >= 60
    assert ellipse.x_m[0] == pytest.approx(ellipse.x_m[-1])
    with pytest.raises(ValueError, match="positive semidefinite"):
        covariance_ellipse(np.zeros(2), np.diag([1.0, -2.0]))


def test_track_overlay_uses_estimate_only_and_velocity_has_physical_scale() -> None:
    track = Track.initial(4, np.array([200.0, -100.0]), 0.0)
    track.state[2:] = [3.0, -4.0]
    track.status = TrackStatus.CONFIRMED
    overlay = track_overlay(track, vector_horizon_s=2.0)

    assert overlay.track_id == 4
    assert overlay.status is TrackStatus.CONFIRMED
    assert overlay.velocity_end_m == pytest.approx((206.0, -108.0))
    assert overlay.speed_mps == pytest.approx(5.0)
    assert overlay.ellipse.major_radius_m > 0
    assert not hasattr(overlay, "target_id")


def test_ppi_draws_labels_vectors_uncertainty_and_independent_toggles() -> None:
    app = QApplication.instance() or QApplication([])
    view = PpiView(max_range_m=1000.0)
    tentative = Track.initial(1, np.array([100.0, 30.0]), 0.0)
    confirmed = Track.initial(2, np.array([200.0, -40.0]), 0.0)
    confirmed.status = TrackStatus.CONFIRMED
    confirmed.state[2:] = [5.0, 2.0]
    coasting = Track.initial(3, np.array([300.0, 50.0]), 0.0)
    coasting.status = TrackStatus.COASTING

    view.set_tracks([tentative, confirmed, coasting])

    assert set(view.track_label_items) == {1, 2, 3}
    assert set(view.velocity_items) == {1, 2, 3}
    assert set(view.uncertainty_items) == {1, 2, 3}
    assert "2" in view.track_label_items[2].toPlainText()
    assert "confirmed" in view.track_label_items[2].toPlainText()
    view.set_uncertainty_visible(False)
    assert not view.uncertainty_items[2].isVisible()
    assert view.velocity_items[2].isVisible()
    view.set_vectors_visible(False)
    assert not view.velocity_items[2].isVisible()
    view.set_labels_visible(False)
    assert not view.track_label_items[2].isVisible()
    view.set_tracks([])
    assert view.track_label_items == {}
    view.close()
    app.processEvents()
