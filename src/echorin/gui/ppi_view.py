"""Plan-position indicator widget."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from echorin.models.detection import Detection
from echorin.models.track import Track
from echorin.simulation.target import Target


class PpiView(QWidget):
    """Cartesian plan-position view with polar range references."""

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
        self.plot.addItem(self.detection_item)
        self.plot.addItem(self.track_item)
        self._trails_visible = True

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
            detection
            for detection in detections
            if np.isfinite(detection.range_m) and np.isfinite(detection.bearing_rad)
        ]
        self.detection_item.setData(
            x=[d.range_m * np.cos(d.bearing_rad) for d in finite],
            y=[d.range_m * np.sin(d.bearing_rad) for d in finite],
        )

    def set_tracks(self, tracks: Iterable[Track]) -> None:
        """Draw active track states and bounded histories."""
        active = list(tracks)
        self.track_item.setData(
            x=[track.x_m for track in active],
            y=[track.y_m for track in active],
            data=[track.track_id for track in active],
        )
        active_ids = {track.track_id for track in active}
        for stale_id in set(self.track_history_items) - active_ids:
            self.plot.removeItem(self.track_history_items.pop(stale_id))
        for track in active:
            history = np.asarray(track.history, dtype=np.float64)
            item = self.track_history_items.get(track.track_id)
            if item is None:
                item = self.plot.plot(
                    pen=pg.mkPen(pg.intColor(track.track_id), width=1.5)
                )
                self.track_history_items[track.track_id] = item
            item.setData(history[:, 0], history[:, 1])
            item.setVisible(self._trails_visible)

    def set_detections_visible(self, visible: bool) -> None:
        self.detection_item.setVisible(visible)

    def set_tracks_visible(self, visible: bool) -> None:
        self.track_item.setVisible(visible)

    def set_trails_visible(self, visible: bool) -> None:
        self._trails_visible = visible
        for item in self.track_history_items.values():
            item.setVisible(visible)
