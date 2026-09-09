"""Signal-product plot widgets, populated by later DSP milestones."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from echorin.dsp.doppler import DopplerProduct
from echorin.dsp.range_processing import RangeProfile


class SignalPlots(QWidget):
    """Range-profile plot container reserved for sensor products."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.range_plot = pg.PlotWidget(background="#101418")
        self.range_plot.setLabel("bottom", "Range", units="m")
        self.range_plot.setLabel("left", "Magnitude")
        self.profile_curve = self.range_plot.plot(
            pen=pg.mkPen("c", width=1.5), name="Matched filter"
        )
        self.threshold_curve = self.range_plot.plot(
            pen=pg.mkPen("m", width=1.2), name="CA-CFAR threshold"
        )
        self.detection_item = pg.ScatterPlotItem(
            symbol="t", size=10, pen=pg.mkPen("y"), brush=pg.mkBrush("y")
        )
        self.range_plot.addItem(self.detection_item)
        layout.addWidget(self.range_plot)
        self.doppler_plot = pg.PlotWidget(background="#101418")
        self.doppler_plot.setLabel("bottom", "Radial velocity", units="m/s")
        self.doppler_plot.setLabel("left", "Magnitude")
        self.doppler_curve = self.doppler_plot.plot(pen=pg.mkPen("g", width=1.5))
        layout.addWidget(self.doppler_plot)

    def set_range_product(
        self,
        profile: RangeProfile,
        threshold: np.ndarray,
        detection_bins: Iterable[int],
    ) -> None:
        """Display matched-filter magnitude, adaptive threshold, and detections."""
        if threshold.shape != profile.magnitude.shape:
            raise ValueError("threshold and range profile must have equal shape")
        bins = np.asarray(list(detection_bins), dtype=np.intp)
        if bins.size and (np.any(bins < 0) or np.any(bins >= profile.response.size)):
            raise ValueError("detection bin lies outside the range profile")
        magnitude = profile.magnitude
        self.profile_curve.setData(profile.ranges_m, magnitude)
        self.threshold_curve.setData(profile.ranges_m, threshold)
        self.detection_item.setData(
            x=profile.ranges_m[bins] if bins.size else [],
            y=magnitude[bins] if bins.size else [],
        )

    def set_doppler_product(
        self, product: DopplerProduct, selected_range_bin: int
    ) -> None:
        """Display the velocity spectrum at one selected range bin."""
        if not 0 <= selected_range_bin < product.spectrum.shape[1]:
            raise ValueError("selected_range_bin lies outside Doppler product")
        self.doppler_curve.setData(
            product.radial_velocity_mps,
            product.magnitude[:, selected_range_bin],
        )
