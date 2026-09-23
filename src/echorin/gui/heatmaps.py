"""Interactive Range-Doppler display of the existing coherent FFT product."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from echorin.dsp.doppler import DopplerProduct
from echorin.gui.visualization_data import (
    RangeDopplerCell,
    RangeDopplerImage,
    range_doppler_image,
)
from echorin.models.detection import Detection


class RangeDopplerView(QWidget):
    """Full velocity-by-range heatmap with color scale and cell inspection."""

    cell_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product: RangeDopplerImage | None = None
        self._axis_extent: tuple[float, float, float, float] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        controls = QGridLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(("dB", "Linear"))
        controls.addWidget(QLabel("Scale"), 0, 0)
        controls.addWidget(self.scale_combo, 0, 1)
        self.limits: dict[str, QDoubleSpinBox] = {}
        for column, (key, label) in enumerate(
            (
                ("range_min", "Range min (m)"),
                ("range_max", "Range max (m)"),
                ("velocity_min", "Velocity min (m/s)"),
                ("velocity_max", "Velocity max (m/s)"),
            ),
            start=2,
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(6)
            self.limits[key] = spin
            controls.addWidget(QLabel(label), 0, column * 2)
            controls.addWidget(spin, 0, column * 2 + 1)
            spin.valueChanged.connect(self._apply_limits)
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.plot = pg.PlotWidget(background="#101820")
        self.plot.setLabel("bottom", "Range", units="m")
        self.plot.setLabel("left", "Radial velocity", units="m/s")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.cursor_x = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("w", width=1))
        self.cursor_y = pg.InfiniteLine(angle=0, movable=False,
                                        pen=pg.mkPen("w", width=1))
        self.plot.addItem(self.cursor_x)
        self.plot.addItem(self.cursor_y)
        self.cursor_x.hide()
        self.cursor_y.hide()
        self.selection_item = pg.ScatterPlotItem(
            symbol="o", size=12, pen=pg.mkPen("y", width=2), brush=None
        )
        self.plot.addItem(self.selection_item)
        self.detection_item = pg.ScatterPlotItem(
            symbol="x", size=11, pen=pg.mkPen("r", width=2)
        )
        self.plot.addItem(self.detection_item)
        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.image_item)
        self.colorbar.gradient.loadPreset("viridis")
        row.addWidget(self.plot, stretch=1)
        color_column = QVBoxLayout()
        self.color_label = QLabel("Magnitude (dB re 1)")
        color_column.addWidget(self.color_label)
        color_column.addWidget(self.colorbar, stretch=1)
        row.addLayout(color_column)
        layout.addLayout(row, stretch=1)
        self.readout = QLabel("Select a Range-Doppler cell")
        layout.addWidget(self.readout)
        self.scale_combo.currentTextChanged.connect(self._render)
        self.plot.scene().sigMouseMoved.connect(self._hover)
        self.plot.scene().sigMouseClicked.connect(self._click)

    def set_product(self, product: DopplerProduct) -> None:
        """Display every processed range and velocity cell."""
        self.product = range_doppler_image(product)
        x, y = self.product.ranges_m, self.product.velocities_mps
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        self.image_item.setRect(
            QRectF(float(x[0] - dx / 2), float(y[0] - dy / 2),
                   float(x[-1] - x[0] + dx), float(y[-1] - y[0] + dy))
        )
        extent = (float(x[0] - dx / 2), float(x[-1] + dx / 2),
                  float(y[0] - dy / 2), float(y[-1] + dy / 2))
        if extent != self._axis_extent:
            for key, value in zip(self.limits, extent, strict=True):
                spin = self.limits[key]
                with QSignalBlocker(spin):
                    spin.setValue(value)
            self._axis_extent = extent
        self._render()
        self._apply_limits()

    def set_detections(self, detections: Iterable[Detection]) -> None:
        """Overlay physical range and velocity of sensor-derived detections."""
        finite = [
            detection for detection in detections
            if detection.radial_velocity_mps is not None
            and np.isfinite(detection.range_m)
            and np.isfinite(detection.radial_velocity_mps)
        ]
        self.detection_item.setData(
            x=[detection.range_m for detection in finite],
            y=[detection.radial_velocity_mps for detection in finite],
        )

    def _render(self) -> None:
        if self.product is None:
            return
        image = (
            self.product.level_db
            if self.scale_combo.currentText() == "dB"
            else self.product.magnitude
        )
        finite = image[np.isfinite(image)]
        levels = (
            (float(np.min(finite)), float(np.max(finite)))
            if finite.size and float(np.min(finite)) < float(np.max(finite))
            else (0.0, 1.0)
        )
        self.image_item.setImage(image, autoLevels=False, levels=levels)
        self.colorbar.setLevels(*levels)
        self.color_label.setText(
            "Magnitude (dB re 1)" if self.scale_combo.currentText() == "dB"
            else "Magnitude (linear)"
        )

    def _apply_limits(self) -> None:
        x0 = self.limits["range_min"].value()
        x1 = self.limits["range_max"].value()
        y0 = self.limits["velocity_min"].value()
        y1 = self.limits["velocity_max"].value()
        if x0 < x1 and y0 < y1:
            self.plot.setXRange(x0, x1, padding=0.0)
            self.plot.setYRange(y0, y1, padding=0.0)

    def select_cell_at(self, range_m: float, velocity_mps: float) -> bool:
        """Select the nearest physical cell and publish its sensor-derived data."""
        if self.product is None:
            return False
        indices = self.product.cell_at(range_m, velocity_mps)
        if indices is None:
            return False
        cell: RangeDopplerCell = self.product.selected_cell(*indices)
        self.selection_item.setData(x=[cell.range_m], y=[cell.radial_velocity_mps])
        self.readout.setText(
            f"Range {cell.range_m:.2f} m | velocity {cell.radial_velocity_mps:.3f} "
            f"m/s | magnitude {cell.magnitude:.4g} | {cell.level_db:.1f} dB "
            f"| bin ({cell.velocity_bin}, {cell.range_bin})"
        )
        self.cell_selected.emit(cell)
        return True

    def _scene_coordinates(self, position: object) -> tuple[float, float] | None:
        if not self.plot.sceneBoundingRect().contains(position):
            return None
        mapped = self.plot.plotItem.vb.mapSceneToView(position)
        return mapped.x(), mapped.y()

    def _hover(self, position: object) -> None:
        coordinates = self._scene_coordinates(position)
        if coordinates is None or self.product is None:
            return
        indices = self.product.cell_at(*coordinates)
        if indices is not None:
            cell = self.product.selected_cell(*indices)
            self.cursor_x.setPos(cell.range_m)
            self.cursor_y.setPos(cell.radial_velocity_mps)
            self.cursor_x.show()
            self.cursor_y.show()
            self.readout.setText(
                f"Range {cell.range_m:.2f} m | velocity "
                f"{cell.radial_velocity_mps:.3f} m/s | {cell.level_db:.1f} dB"
            )

    def _click(self, event: object) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        coordinates = self._scene_coordinates(event.scenePos())
        if coordinates is not None:
            self.select_cell_at(*coordinates)
