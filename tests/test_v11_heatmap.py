"""Full Range-Doppler coordinate and GUI behavior."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import SensorConfig
from echorin.dsp.doppler import doppler_spectrum
from echorin.gui.heatmaps import RangeDopplerView
from echorin.gui.visualization_data import range_doppler_image, to_db
from echorin.models.detection import Detection


def test_range_doppler_axes_and_peak_coordinates_for_radar_and_sonar() -> None:
    for config in (SensorConfig.radar(), SensorConfig.sonar()):
        pulses = np.zeros((16, 25), dtype=np.complex128)
        slow_time = np.arange(16)
        pulses[:, 7] = np.exp(2j * np.pi * 3 * slow_time / 16)
        product = doppler_spectrum(pulses, config, apply_window=False)
        image = range_doppler_image(product)
        peak = np.unravel_index(np.argmax(image.magnitude), image.magnitude.shape)

        assert peak == (11, 7)
        assert image.ranges_m[7] == pytest.approx(
            7 * config.propagation_speed_mps / (2 * config.sample_rate_hz)
        )
        assert image.velocities_mps[11] == pytest.approx(
            3 * config.prf_hz / 16
            * config.propagation_speed_mps / config.carrier_frequency_hz / 2
        )
        assert image.cell_at(image.ranges_m[7], image.velocities_mps[11]) == peak
        assert image.cell_at(-1_000, 0) is None


def test_db_conversion_handles_zero_tiny_and_nonfinite_samples() -> None:
    values = np.array([[1.0, 0.1, 0.0, np.nan, np.inf]])
    converted = to_db(values, floor_db=-80.0)

    np.testing.assert_allclose(converted, [[0.0, -20.0, -80.0, -80.0, -80.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        to_db(np.array([-1.0]))


def test_heatmap_uses_full_product_and_selects_physical_cell() -> None:
    app = QApplication.instance() or QApplication([])
    config = SensorConfig.sonar()
    pulses = np.zeros((8, 20), dtype=np.complex128)
    pulses[:, 6] = np.exp(2j * np.pi * np.arange(8) / 8)
    product = doppler_spectrum(pulses, config, apply_window=False)
    view = RangeDopplerView()
    selected = []
    view.cell_selected.connect(selected.append)

    view.set_product(product)
    assert view.image_item.image.shape == (8, 20)
    assert view.colorbar is not None
    view.select_cell_at(product.ranges_m[6], product.radial_velocity_mps[5])
    assert selected[-1].range_bin == 6
    assert selected[-1].velocity_bin == 5
    view.scale_combo.setCurrentText("Linear")
    assert view.image_item.image[5, 6] == pytest.approx(1.0)
    view.limits["range_max"].setValue(product.ranges_m[12])
    view.set_product(product)
    assert view.limits["range_max"].value() == pytest.approx(product.ranges_m[12])
    view.set_detections(
        [Detection(0.0, float(product.ranges_m[6]), 0.0,
                   float(product.radial_velocity_mps[5]), 1.0, 12.0, 0.8, 6)]
    )
    marker_x, marker_y = view.detection_item.getData()
    assert marker_x[0] == pytest.approx(product.ranges_m[6])
    assert marker_y[0] == pytest.approx(product.radial_velocity_mps[5])
    view.close()
    app.processEvents()
