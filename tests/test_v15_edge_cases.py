"""Generalized geometry and config privacy boundaries at invalid inputs."""

from dataclasses import replace

import numpy as np
import pytest

from echorin.application.sensor_network import SensorNetwork
from echorin.config import NoiseConfig, SensorConfig
from echorin.models.geometry import SensorPose
from echorin.models.platform import MountTransform
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.components import Emitter, Receiver, SensorPlatform
from echorin.sensors.factory import create_link_sensor
from echorin.simulation.scenarios import bistatic_scenario
from echorin.simulation.target import Target


def test_nonfinite_device_pose_rejected_before_acquisition():
    config = SensorConfig()
    with pytest.raises(ValueError, match="finite"):
        create_link_sensor(
            Emitter("tx", config),
            Receiver("rx", config),
            SensorPose(vx_mps=np.nan),
            SensorPose(),
        )


def test_collocated_independent_heading_is_monostatic_but_transmit_beam_frame_differs():
    config = SensorConfig(noise_model=NoiseConfig(0))
    sensor = create_link_sensor(
        Emitter("tx", config, beam=BeamConfig(kind="sector", width_rad=0.2)),
        Receiver("rx", config),
        SensorPose(heading_rad=1),
        SensorPose(),
    )
    assert not sensor.is_bistatic
    assert not np.any(
        sensor.acquire_array_pulse_train((Target("private", 1500, 0),), 0, 8).samples
    )


def test_link_work_limit_is_enforced_before_array_allocation():
    config = SensorConfig()
    state = bistatic_scenario(config).sensor_platforms[0].state
    platforms = tuple(
        SensorPlatform.monostatic(str(i), state, config) for i in range(5)
    )
    with pytest.raises(ValueError, match="work limit"):
        SensorNetwork(platforms)


def test_network_checkpoint_preserves_mounted_devices_and_platform_trails():
    config = SensorConfig()
    world = bistatic_scenario(config)
    p = world.sensor_platforms[0]
    e = replace(p.emitters[0], mount=MountTransform(np.array([4.0, 2.0]), 0.3))
    world.update_device_configs((replace(p, emitters=(e,)), world.sensor_platforms[1]))
    before = world.sensor_platforms[0].emitter_pose(e)
    world.advance(0.1)
    assert len(world.sensor_platform_histories[p.platform_id]) == 2
    world.reset()
    after = world.sensor_platforms[0].emitter_pose(e)
    assert after == before
    assert len(world.sensor_platform_histories[p.platform_id]) == 1
