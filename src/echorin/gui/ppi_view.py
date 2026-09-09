"""Plan-position indicator widget."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

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

        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        ring_pen = pg.mkPen((65, 120, 145, 150), width=1)
        self.range_rings: list[pg.PlotDataItem] = []
        for radius in np.linspace(max_range_m / ring_count, max_range_m, ring_count):
            ring = self.plot.plot(
                radius * np.cos(angles), radius * np.sin(angles), pen=ring_pen
            )
            self.range_rings.append(ring)
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
