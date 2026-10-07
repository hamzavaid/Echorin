"""Analytic generalized TX-target-RX references, before sensing integration."""

import numpy as np
import pytest

from echorin.config import SensorConfig
from echorin.models.geometry import SensorPose
from echorin.models.platform import MountTransform, PlatformState
from echorin.propagation.geometry import path_geometry, receiver_range_from_path
from echorin.sensors.components import Emitter, Receiver, SensorPlatform


def test_monostatic_reduction_and_relative_motion():
    pose = SensorPose(10, 20, vx_mps=3, vy_mps=4)
    result = path_geometry(pose, pose, (310, 420), (9, 12), 1500, 20000)
    assert result.path_length_m == 1000
    assert result.delay_s == pytest.approx(2 / 3)
    assert result.path_rate_mps == pytest.approx(20)
    assert result.doppler_hz == pytest.approx(20 * 20000 / 1500)
    assert result.receiver_bearing_rad == pytest.approx(np.arctan2(400, 300))


def test_moving_bistatic_doppler_is_derivative_of_total_path():
    tx, rx = SensorPose(-100, 20, vx_mps=8), SensorPose(80, -30, vy_mps=-3)
    position, velocity = np.array([300.0, 170.0]), np.array([5.0, 2.0])
    result = path_geometry(tx, rx, position, velocity, 1500, 20000)
    dt = 1e-5
    tx2 = np.array([tx.x_m, tx.y_m]) + dt * np.array([tx.vx_mps, tx.vy_mps])
    rx2 = np.array([rx.x_m, rx.y_m]) + dt * np.array([rx.vx_mps, rx.vy_mps])
    p2 = position + velocity * dt
    derivative = (
        np.linalg.norm(p2 - tx2) + np.linalg.norm(p2 - rx2) - result.path_length_m
    ) / dt
    assert result.path_rate_mps == pytest.approx(derivative, abs=1e-5)
    assert result.doppler_hz == pytest.approx(derivative * 20000 / 1500, abs=1e-3)
    inferred = receiver_range_from_path(
        result.path_length_m, result.receiver_bearing_rad, tx, rx
    )
    assert inferred == pytest.approx(np.linalg.norm(position - [rx.x_m, rx.y_m]))


@pytest.mark.parametrize("length,bearing", [(50, 0), (100, 0), (np.nan, 0)])
def test_impossible_or_degenerate_measurement_rejected(length, bearing):
    with pytest.raises(ValueError):
        receiver_range_from_path(length, bearing, SensorPose(100, 0), SensorPose())


def test_collocated_target_and_invalid_physics_rejected():
    for speed, carrier, position in [
        (0, 1, (1, 0)),
        (1, 0, (1, 0)),
        (1, 1, (0, 0)),
        (1, 1, (np.nan, 0)),
    ]:
        with pytest.raises(ValueError):
            path_geometry(SensorPose(), SensorPose(), position, (0, 0), speed, carrier)


def test_components_mount_velocity_and_monostatic_factory():
    config = SensorConfig()
    state = PlatformState(np.zeros(2), np.array([3, 4]), np.zeros(2), 0, 2, 0)
    platform = SensorPlatform.monostatic("ship", state, config)
    assert isinstance(platform.emitters[0], Emitter)
    assert isinstance(platform.receivers[0], Receiver)
    assert platform.emitter_pose(platform.emitters[0]) == platform.receiver_pose(
        platform.receivers[0]
    )
    emitter = Emitter("offset", config, mount=MountTransform(np.array([5, 0])))
    pose = platform.emitter_pose(emitter)
    assert (pose.vx_mps, pose.vy_mps) == (3, 14)
    with pytest.raises(ValueError):
        Emitter("", config)
    with pytest.raises(ValueError):
        Emitter("bad", config, transmit_power_scale=-1)
