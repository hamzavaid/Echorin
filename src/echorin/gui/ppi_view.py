"""Plan-position indicator widget."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from echorin.gui.visualization_data import track_overlay
from echorin.models.detection import Detection
from echorin.models.track import Track, TrackStatus
from echorin.simulation.target import Target


class PpiView(QWidget):
    """Cartesian plan-position view with polar range references."""

    track_selected = Signal(int)
    detection_selected = Signal(int)

    STATUS_COLORS = {
        TrackStatus.TENTATIVE: "#f9b04f",
        TrackStatus.CONFIRMED: "#50e0a1",
        TrackStatus.COASTING: "#68baff",
    }

    def __init__(
        self,
        max_range_m: float = 10_000.0,
        ring_count: int = 4,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive")
        if ring_count <= 0:
            raise ValueError("ring_count must be positive")
        self.max_range_m = max_range_m
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="#07151d")
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "East", units="m")
        self.plot.setLabel("left", "North", units="m")
        self.plot.setXRange(-max_range_m, max_range_m, padding=0.02)
        self.plot.setYRange(-max_range_m, max_range_m, padding=0.02)
        self.plot.addLegend(offset=(12, 12))
        for title, color, symbol in (
            ("Detection", "#ff7979", "x"),
            ("Tentative", self.STATUS_COLORS[TrackStatus.TENTATIVE], "o"),
            ("Confirmed", self.STATUS_COLORS[TrackStatus.CONFIRMED], "o"),
            ("Coasting", self.STATUS_COLORS[TrackStatus.COASTING], "o"),
            ("Ground truth", "#ffc547", "o"),
        ):
            self.plot.plot([], [], pen=None, symbol=symbol,
                           symbolBrush=color, name=title)
        layout.addWidget(self.plot)

        self.range_rings: list[pg.PlotDataItem] = []
        self._ring_count = ring_count
        self._draw_range_rings()
        self.sensor_item = pg.ScatterPlotItem(
            x=[0.0], y=[0.0], symbol="+", size=16, pen=pg.mkPen("c", width=2)
        )
        self.target_item = pg.ScatterPlotItem(
            symbol="o",
            size=10,
            pen=pg.mkPen("y", width=1.5),
            brush=pg.mkBrush(255, 190, 0, 140),
        )
        self.plot.addItem(self.sensor_item)
        self.plot.addItem(self.target_item)
        self.detection_item = pg.ScatterPlotItem(
            symbol="x", size=11, pen=pg.mkPen("r", width=2)
        )
        self.track_item = pg.ScatterPlotItem(
            symbol="o",
            size=12,
            pen=pg.mkPen("g", width=2),
            brush=pg.mkBrush(0, 220, 100, 90),
        )
        self.track_history_items: dict[int, pg.PlotDataItem] = {}
        self.track_label_items: dict[int, pg.TextItem] = {}
        self.velocity_items: dict[int, pg.PlotDataItem] = {}
        self.velocity_arrow_items: dict[int, pg.ArrowItem] = {}
        self.uncertainty_items: dict[int, pg.PlotDataItem] = {}
        self.plot.addItem(self.detection_item)
        self.plot.addItem(self.track_item)
        self._trails_visible = True
        self._tracks_visible = True
        self._labels_visible = True
        self._vectors_visible = True
        self._uncertainty_visible = True
        self.track_item.sigClicked.connect(self._track_clicked)
        self.detection_item.sigClicked.connect(self._detection_clicked)

    def _track_clicked(self, _item: object, points: list[object]) -> None:
        if points:
            self.track_selected.emit(int(points[0].data()))

    def _detection_clicked(self, _item: object, points: list[object]) -> None:
        if points:
            self.detection_selected.emit(int(points[0].data()))

    def set_targets(self, targets: Iterable[Target]) -> None:
        """Redraw the optional ground-truth target overlay."""
        target_list = list(targets)
        self.target_item.setData(
            x=[target.x_m for target in target_list],
            y=[target.y_m for target in target_list],
            data=[target.target_id for target in target_list],
        )

    def set_ground_truth_visible(self, visible: bool) -> None:
        """Show or hide the ground-truth-only target layer."""
        self.target_item.setVisible(visible)

    def set_max_range(self, max_range_m: float) -> None:
        """Rescale the PPI and rebuild physical range rings."""
        if max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive")
        self.max_range_m = max_range_m
        self.plot.setXRange(-max_range_m, max_range_m, padding=0.02)
        self.plot.setYRange(-max_range_m, max_range_m, padding=0.02)
        for ring in self.range_rings:
            self.plot.removeItem(ring)
        self.range_rings.clear()
        self._draw_range_rings()

    def _draw_range_rings(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        ring_pen = pg.mkPen((65, 120, 145, 150), width=1)
        for radius in np.linspace(
            self.max_range_m / self._ring_count,
            self.max_range_m,
            self._ring_count,
        ):
            ring = self.plot.plot(
                radius * np.cos(angles), radius * np.sin(angles), pen=ring_pen
            )
            self.range_rings.append(ring)

    def set_detections(self, detections: Iterable[Detection]) -> None:
        """Draw finite polar detections converted to Cartesian coordinates."""
        finite = [
            (index, detection)
            for index, detection in enumerate(detections)
            if np.isfinite(detection.range_m) and np.isfinite(detection.bearing_rad)
        ]
        self.detection_item.setData(
            x=[d.range_m * np.cos(d.bearing_rad) for _, d in finite],
            y=[d.range_m * np.sin(d.bearing_rad) for _, d in finite],
            data=[index for index, _ in finite],
        )

    def set_tracks(self, tracks: Iterable[Track]) -> None:
        """Draw active track states and bounded histories."""
        active = list(tracks)
        self.track_item.setData(spots=[
            {
                "pos": (track.x_m, track.y_m),
                "data": track.track_id,
                "symbol": "o" if track.status is not TrackStatus.COASTING else "s",
                "brush": pg.mkBrush(self.STATUS_COLORS.get(track.status, "#999999")),
                "pen": pg.mkPen("w", width=1),
                "size": 12,
            }
            for track in active
        ])
        active_ids = {track.track_id for track in active}
        for stale_id in set(self.track_history_items) - active_ids:
            self.plot.removeItem(self.track_history_items.pop(stale_id))
        for collection in (
            self.track_label_items, self.velocity_items,
            self.velocity_arrow_items, self.uncertainty_items,
        ):
            for stale_id in set(collection) - active_ids:
                self.plot.removeItem(collection.pop(stale_id))
        for track in active:
            overlay = track_overlay(track)
            color = self.STATUS_COLORS.get(track.status, "#999999")
            history = np.asarray(track.history, dtype=np.float64)
            item = self.track_history_items.get(track.track_id)
            if item is None:
                item = self.plot.plot(
                    pen=pg.mkPen(pg.intColor(track.track_id), width=1.5)
                )
                self.track_history_items[track.track_id] = item
            item.setData(history[:, 0], history[:, 1])
            item.setVisible(self._trails_visible)
            label = self.track_label_items.get(track.track_id)
            if label is None:
                label = pg.TextItem(anchor=(0, 1))
                self.track_label_items[track.track_id] = label
                self.plot.addItem(label)
            label.setText(f"{track.track_id} · {track.status.value}", color=color)
            label.setPos(track.x_m, track.y_m)
            label.setVisible(self._tracks_visible and self._labels_visible)
            velocity = self.velocity_items.get(track.track_id)
            if velocity is None:
                velocity = self.plot.plot(pen=pg.mkPen(color, width=2))
                self.velocity_items[track.track_id] = velocity
            velocity.setPen(pg.mkPen(color, width=2))
            velocity.setData(
                [track.x_m, overlay.velocity_end_m[0]],
                [track.y_m, overlay.velocity_end_m[1]],
            )
            velocity.setVisible(self._tracks_visible and self._vectors_visible)
            arrow = self.velocity_arrow_items.get(track.track_id)
            if arrow is None:
                arrow = pg.ArrowItem(headLen=10, tipAngle=28)
                self.plot.addItem(arrow)
                self.velocity_arrow_items[track.track_id] = arrow
            arrow.setStyle(
                angle=180.0 - float(np.degrees(np.arctan2(track.vy_mps,
                                                           track.vx_mps))),
                brush=pg.mkBrush(color), pen=pg.mkPen(color),
            )
            arrow.setPos(*overlay.velocity_end_m)
            arrow.setVisible(
                self._tracks_visible and self._vectors_visible
                and overlay.speed_mps > 0.0
            )
            ellipse_item = self.uncertainty_items.get(track.track_id)
            if ellipse_item is None:
                ellipse_item = self.plot.plot(pen=pg.mkPen(color, width=1))
                self.uncertainty_items[track.track_id] = ellipse_item
            ellipse_item.setPen(pg.mkPen(color, width=1))
            ellipse_item.setData(overlay.ellipse.x_m, overlay.ellipse.y_m)
            ellipse_item.setVisible(
                self._tracks_visible and self._uncertainty_visible
            )

    def set_detections_visible(self, visible: bool) -> None:
        self.detection_item.setVisible(visible)

    def set_tracks_visible(self, visible: bool) -> None:
        self._tracks_visible = visible
        self.track_item.setVisible(visible)
        for collection, flag in (
            (self.track_label_items, self._labels_visible),
            (self.velocity_items, self._vectors_visible),
            (self.velocity_arrow_items, self._vectors_visible),
            (self.uncertainty_items, self._uncertainty_visible),
        ):
            for item in collection.values():
                item.setVisible(visible and flag)

    def set_labels_visible(self, visible: bool) -> None:
        self._labels_visible = visible
        for item in self.track_label_items.values():
            item.setVisible(visible and self._tracks_visible)

    def set_vectors_visible(self, visible: bool) -> None:
        self._vectors_visible = visible
        vector_items = (*self.velocity_items.values(),
                        *self.velocity_arrow_items.values())
        for item in vector_items:
            item.setVisible(visible and self._tracks_visible)

    def set_uncertainty_visible(self, visible: bool) -> None:
        self._uncertainty_visible = visible
        for item in self.uncertainty_items.values():
            item.setVisible(visible and self._tracks_visible)

    def set_trails_visible(self, visible: bool) -> None:
        self._trails_visible = visible
        for item in self.track_history_items.values():
            item.setVisible(visible)
