"""v1.2 array physics, beamforming, and production-path acceptance."""

from __future__ import annotations

import numpy as np
import pytest

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.beamforming import bartlett_spectrum, steering_vectors
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_angle import range_angle_product
from echorin.dsp.range_processing import SignalProcessor
from echorin.sensors.array import ArrayGeometry, uniform_linear_array
from echorin.sensors.factory import create_sensor
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker


def test_ula_geometry_and_analytical_phase() -> None:
    config = SensorConfig.radar()
    wavelength = config.propagation_speed_mps / config.carrier_frequency_hz
    array = uniform_linear_array(8, wavelength / 2)
    assert array.element_positions_m.shape == (8, 2)
    with pytest.raises(ValueError, match="half wavelength"):
        ArrayGeometry(
            array.element_positions_m * 2,
            carrier_frequency_hz=config.carrier_frequency_hz,
            propagation_speed_mps=config.propagation_speed_mps,
        )
    vector = steering_vectors(array, wavelength, np.array([np.deg2rad(30.0)]))[:, 0]
    np.testing.assert_allclose(vector[1] / vector[0], 1j, atol=1e-12)


def test_bartlett_and_range_angle_resolve_two_same_range_targets() -> None:
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.0)
    )
    sensor = create_sensor(config, random_seed=17)
    radius = 300.0
    targets = tuple(
        Target(str(index), radius * np.cos(angle), radius * np.sin(angle))
        for index, angle in enumerate((-0.45, 0.45))
    )
    frame = sensor.acquire_array_pulse_train(targets, 0.0, 16)
    assert frame.samples.shape == (8, 16, config.acquisition_samples)
    assert not hasattr(frame, "target_ids")
    processor = SignalProcessor(config)
    responses = np.asarray(
        [
            processor.pulse_matrix_range_responses(element, frame.transmitted_signal)
            for element in frame.samples
        ]
    )
    angles = np.linspace(-np.pi / 2, np.pi / 2, 181)
    product = range_angle_product(
        responses,
        processor.range_axis(responses.shape[-1]),
        frame.array_geometry,
        config,
        angles,
        0.0,
    )
    range_bin = int(np.argmin(abs(product.ranges_m - radius)))
    angular_power = product.power[:, range_bin]
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(angular_power, height=angular_power.max() * 0.5, distance=10)
    measured = sorted(product.bearings_rad[peaks])
    assert len(measured) == 2
    np.testing.assert_allclose(measured, [-0.45, 0.45], atol=np.deg2rad(1.0))
    direct = bartlett_spectrum(
        responses[:, :, range_bin],
        frame.array_geometry,
        config.propagation_speed_mps / config.carrier_frequency_hz,
        angles,
    )
    np.testing.assert_allclose(direct.power, angular_power)


def test_production_detections_derive_bearing_from_composite_array() -> None:
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    angle = np.deg2rad(28.0)
    target = Target("private", 300 * np.cos(angle), 300 * np.sin(angle))
    result = process_frame(
        (target,),
        0.1,
        create_sensor(config, random_seed=4),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        16,
    )
    assert result.range_angle_product.power.ndim == 2
    near = [d for d in result.detections if abs(d.range_m - 300) < 16]
    assert near
    assert min(abs(d.bearing_rad - angle) for d in near) <= np.deg2rad(2)
    assert all(not hasattr(d, "target_id") for d in result.detections)


def test_two_targets_same_range_produce_two_sensor_bearings() -> None:
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    angles = (-0.45, 0.45)
    targets = tuple(
        Target(str(i), 300 * np.cos(a), 300 * np.sin(a)) for i, a in enumerate(angles)
    )
    result = process_frame(
        targets,
        0.1,
        create_sensor(config),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        16,
    )
    near = sorted(
        (d for d in result.detections if abs(d.range_m - 300) < 16),
        key=lambda d: d.bearing_rad,
    )
    assert len(near) == 2
    np.testing.assert_allclose(
        [d.bearing_rad for d in near], angles, atol=np.deg2rad(1)
    )
    assert near[0].source_bin == near[1].source_bin


def test_range_angle_gui_labels_units_and_inspects_cells(tmp_path) -> None:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from echorin.gui.main_window import MainWindow
    from echorin.simulation.scenarios import single_stationary_target

    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        world=single_stationary_target(range_m=100),
        settings=QSettings(str(tmp_path / "array.ini"), QSettings.Format.IniFormat),
    )
    window.step_once()
    assert window.last_range_angle_product is not None
    assert window.range_angle_view.product is not None
    assert window.range_angle_view.select_cell_at(100, 0)
    assert "bearing" in window.range_angle_view.readout.text()
    window.close()
    app.processEvents()


def test_large_range_angle_display_retains_full_resolution_inspection() -> None:
    from PySide6.QtWidgets import QApplication

    from echorin.dsp.range_angle import RangeAngleProduct
    from echorin.gui.range_angle_view import RangeAngleView

    app = QApplication.instance() or QApplication([])
    ranges = np.arange(10_000, dtype=np.float64)
    angles = np.linspace(-1, 1, 5)
    product = RangeAngleProduct(ranges, angles, np.ones((5, 10_000)), 0.0)
    view = RangeAngleView()
    view.set_product(product)
    assert view.image_item.image.shape[1] <= 2048
    assert view.select_cell_at(9_999, np.rad2deg(angles[2]))
    assert "9999.00 m" in view.readout.text()
    view.close()
    app.processEvents()
