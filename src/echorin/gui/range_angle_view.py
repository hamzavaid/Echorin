"""Interactive signal-derived range-angle map."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from echorin.dsp.range_angle import RangeAngleProduct
from echorin.gui.visualization_data import to_db
from echorin.models.detection import Detection


class RangeAngleView(QWidget):
    """Display [bearing, range] power with cell and detection inspection."""

    cell_selected = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product: RangeAngleProduct | None = None
        self._display_stride = 1
        layout = QVBoxLayout(self)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(("dB", "Linear"))
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Power scale"))
        controls.addWidget(self.scale_combo)
        controls.addStretch()
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.plot = pg.PlotWidget(background="#101820")
        self.plot.setLabel("bottom", "Range", units="m")
        self.plot.setLabel("left", "Bearing", units="deg")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.detection_item = pg.ScatterPlotItem(
            symbol="x", size=11, pen=pg.mkPen("r", width=2)
        )
        self.plot.addItem(self.detection_item)
        self.cursor_x = pg.InfiniteLine(angle=90, movable=False, pen="w")
        self.cursor_y = pg.InfiniteLine(angle=0, movable=False, pen="w")
        self.plot.addItem(self.cursor_x)
        self.plot.addItem(self.cursor_y)
        self.cursor_x.hide()
        self.cursor_y.hide()
        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.image_item)
        self.colorbar.gradient.loadPreset("viridis")
        row.addWidget(self.plot, stretch=1)
        row.addWidget(self.colorbar)
        layout.addLayout(row, stretch=1)
        self.readout = QLabel("Select a Range-Angle cell")
        layout.addWidget(self.readout)
        self.scale_combo.currentTextChanged.connect(self._render)
        self.plot.scene().sigMouseMoved.connect(self._hover)
        self.plot.scene().sigMouseClicked.connect(self._click)

    def set_product(self, product: RangeAngleProduct) -> None:
        """Plot power and physical axes from a beamformed sensor product."""
        self.product = product
        ranges = product.ranges_m
        angles = np.rad2deg(product.bearings_rad)
        self._display_stride = max(1, int(np.ceil(len(ranges) / 2048)))
        dx = float(ranges[1] - ranges[0]) * self._display_stride
        dy = float(angles[1] - angles[0])
        last_displayed = ranges[::self._display_stride][-1]
        self.image_item.setRect(
            QRectF(
                float(ranges[0] - dx / 2),
                float(angles[0] - dy / 2),
                float(last_displayed - ranges[0] + dx),
                float(angles[-1] - angles[0] + dy),
            )
        )
        self._render()

    def set_detections(self, detections: Iterable[Detection]) -> None:
        """Mark derived range-angle detections without consulting truth."""
        values = [d for d in detections if np.isfinite(d.bearing_rad)]
        self.detection_item.setData(
            x=[d.range_m for d in values], y=[np.rad2deg(d.bearing_rad) for d in values]
        )

    def clear_product(self) -> None:
        self.product = None
        self.image_item.clear()
        self.detection_item.setData(x=[], y=[])
        self.cursor_x.hide()
        self.cursor_y.hide()
        self.readout.setText("Select a Range-Angle cell")

    def _render(self) -> None:
        if self.product is None:
            return
        display_power = self.product.power[:, ::self._display_stride]
        image = (
            to_db(np.sqrt(display_power))
            if self.scale_combo.currentText() == "dB"
            else display_power
        )
        low, high = float(np.min(image)), float(np.max(image))
        if high <= low:
            high = low + 1.0
        self.image_item.setImage(image, autoLevels=False, levels=(low, high))
        self.colorbar.setLevels(low, high)

    def select_cell_at(self, range_m: float, bearing_deg: float) -> bool:
        """Select the closest cell inside the physical product bounds."""
        if self.product is None:
            return False
        ranges = self.product.ranges_m
        angles = np.rad2deg(self.product.bearings_rad)
        dx, dy = float(ranges[1] - ranges[0]), float(angles[1] - angles[0])
        if not (
            ranges[0] - dx / 2 <= range_m < ranges[-1] + dx / 2
            and angles[0] - dy / 2 <= bearing_deg < angles[-1] + dy / 2
        ):
            return False
        r = int(np.argmin(abs(ranges - range_m)))
        a = int(np.argmin(abs(angles - bearing_deg)))
        self.cursor_x.setPos(float(ranges[r]))
        self.cursor_y.setPos(float(angles[a]))
        self.cursor_x.show()
        self.cursor_y.show()
        level = float(to_db(np.sqrt(self.product.power[a : a + 1, r : r + 1]))[0, 0])
        self.readout.setText(
            f"Range {ranges[r]:.2f} m | bearing {angles[a]:.1f} deg "
            f"| {level:.1f} dB | bin ({a}, {r})"
        )
        self.cell_selected.emit(a, r)
        return True

    def _coordinates(self, position: object) -> tuple[float, float] | None:
        if not self.plot.sceneBoundingRect().contains(position):
            return None
        mapped = self.plot.plotItem.vb.mapSceneToView(position)
        return mapped.x(), mapped.y()

    def _hover(self, position: object) -> None:
        coordinates = self._coordinates(position)
        if coordinates is not None and self.product is not None:
            self.select_cell_at(*coordinates)

    def _click(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            coordinates = self._coordinates(event.scenePos())
            if coordinates is not None:
                self.select_cell_at(*coordinates)
