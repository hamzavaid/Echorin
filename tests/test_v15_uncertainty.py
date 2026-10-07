"""Two-leg attenuation and measurement-space uncertainty references."""

import numpy as np
import pytest

from echorin.config import NoiseConfig, SensorConfig
from echorin.models.detection import Detection
from echorin.models.geometry import SensorPose
from echorin.propagation.geometry import receiver_range_from_path
from echorin.sensors.components import Emitter, Receiver
from echorin.sensors.factory import create_link_sensor
from echorin.signals.propagation import amplitude_at_range
from echorin.signals.waveform import generate_waveform
from echorin.simulation.target import Target
from echorin.tracking.tracker import measurement_covariance_for_detection


def test_attenuation_uses_both_legs_and_scalar_acquisition_uses_path_delay():
    config = SensorConfig(noise_model=NoiseConfig(0))
    tx, rx = SensorPose(-2000, 0), SensorPose()
    target = Target("hidden", 1500, 0)
    sensor = create_link_sensor(Emitter("tx", config), Receiver("rx", config), tx, rx)
    raw = sensor.acquire_array_pulse_train((target,), 0, 8)
    expected = np.sqrt(amplitude_at_range(3500, 1) * amplitude_at_range(1500, 1))
    waveform = generate_waveform(config)
    assert abs(raw.samples).max() == pytest.approx(abs(waveform).max() * expected)
    np.testing.assert_allclose(
        sensor.acquire((target,), 0).received_signal, raw.samples[0, 0].real
    )


def test_bistatic_covariance_uses_ellipse_inverse_jacobian():
    tx, rx = SensorPose(-100, 30), SensorPose(0, 0)
    length, bearing = 450.0, 0.3
    r = receiver_range_from_path(length, bearing, tx, rx)
    detection = Detection(
        0,
        r,
        bearing,
        None,
        1,
        20,
        0.9,
        1,
        sensor_pose=rx,
        transmitter_pose=tx,
        path_length_m=length,
    )
    covariance = measurement_covariance_for_detection(detection, 2, 0.01)

    def position(length_m, b):
        rr = receiver_range_from_path(length_m, b, tx, rx)
        return rr * np.array([np.cos(b), np.sin(b)])

    eps = 1e-5
    jl = (position(length + eps, bearing) - position(length - eps, bearing)) / (2 * eps)
    jb = (position(length, bearing + eps) - position(length, bearing - eps)) / (2 * eps)
    expected = 4**2 * np.outer(jl, jl) + 0.01**2 * np.outer(jb, jb)
    np.testing.assert_allclose(covariance, expected, rtol=1e-6)
