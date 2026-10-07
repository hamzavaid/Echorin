"""Generalized acquisition must use the existing real DSP, without truth."""

from dataclasses import replace

import numpy as np
import pytest

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.models.geometry import SensorPose
from echorin.propagation.geometry import path_geometry
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.components import Emitter, Receiver
from echorin.sensors.factory import create_link_sensor, create_sensor
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_explicit_monostatic_pair_preserves_every_sample(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.02))
    pose = SensorPose(vx_mps=0.005)
    target = Target("not-a-source-id", config.max_range_m / 3, 0, vx_mps=0.01)
    legacy = create_sensor(config, pose, random_seed=21)
    link = create_link_sensor(
        Emitter("tx", config), Receiver("rx", config), pose, pose, random_seed=21
    )
    expected = legacy.acquire_array_pulse_train((target,), 0, 8)
    actual = link.acquire_array_pulse_train((target,), 0, 8)
    np.testing.assert_array_equal(actual.samples, expected.samples)
    assert (actual.emitter_id, actual.receiver_id) == ("tx", "rx")
    link.reset()
    np.testing.assert_array_equal(
        actual.samples, link.acquire_array_pulse_train((target,), 0, 8).samples
    )


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_moving_bistatic_measurements_and_products_have_correct_semantics(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.00001))
    scale = config.max_range_m / 10
    tx = SensorPose(-scale, 0, vx_mps=0.02)
    rx = SensorPose(0, 0, vy_mps=0.01)
    target = Target("secret-target", 3 * scale, scale, vx_mps=0.03, vy_mps=-0.01)
    link = create_link_sensor(
        Emitter("illuminator", config), Receiver("array", config), tx, rx
    )
    result = process_frame(
        (target,),
        0.1,
        link,
        SignalProcessor(config),
        CaCfarDetector(CfarConfig(edge_mode="adaptive")),
        MultiTargetTracker(),
        32,
    )
    reference = path_geometry(
        tx,
        rx,
        (target.x_m, target.y_m),
        (target.vx_mps, target.vy_mps),
        config.propagation_speed_mps,
        config.carrier_frequency_hz,
    )
    distance_bin = config.propagation_speed_mps / config.sample_rate_hz
    assert result.detections
    detection = max(result.detections, key=lambda d: d.amplitude)
    assert detection.path_length_m == pytest.approx(
        reference.path_length_m, abs=distance_bin
    )
    assert detection.radial_velocity_mps is None
    rate_bin = (
        config.propagation_speed_mps / config.carrier_frequency_hz * config.prf_hz / 32
    )
    assert detection.path_rate_mps == pytest.approx(
        reference.path_rate_mps, abs=rate_bin
    )
    assert detection.range_m == pytest.approx(
        reference.receiver_range_m, abs=2 * distance_bin
    )
    assert detection.bearing_rad == pytest.approx(
        reference.receiver_bearing_rad, abs=np.deg2rad(1)
    )
    assert (
        np.linalg.norm(np.array(detection.world_position_m) - [target.x_m, target.y_m])
        < 0.02 * config.max_range_m
    )
    for product in (result.doppler_product, result.range_angle_product):
        assert product.emitter_id == "illuminator"
        assert product.receiver_id == "array"
        assert product.is_bistatic
    assert "secret-target" not in repr(result.detections)


def test_distinct_transmit_and_receive_beams_and_power():
    config = SensorConfig(noise_model=NoiseConfig(0))
    tx, rx = SensorPose(0, 1000), SensorPose()
    target = Target("private", 1500, 0)
    emitter, receiver = Emitter("tx", config), Receiver("rx", config)
    baseline = create_link_sensor(emitter, receiver, tx, rx)
    data = baseline.acquire_array_pulse_train((target,), 0, 8).samples
    blocked = create_link_sensor(
        replace(emitter, beam=BeamConfig(kind="sector", width_rad=0.1)),
        receiver,
        tx,
        rx,
    )
    assert not np.any(blocked.acquire_array_pulse_train((target,), 0, 8).samples)
    powerful = create_link_sensor(
        replace(emitter, transmit_power_scale=4), receiver, tx, rx
    )
    np.testing.assert_allclose(
        powerful.acquire_array_pulse_train((target,), 0, 8).samples, data * 2
    )


def test_incompatible_medium_or_waveform_sampling_rejected():
    radar = SensorConfig()
    for config in (SensorConfig.sonar(), replace(radar, prf_hz=500)):
        with pytest.raises(ValueError, match="compatible"):
            create_link_sensor(
                Emitter("tx", radar), Receiver("rx", config), SensorPose(), SensorPose()
            )
