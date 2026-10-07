"""Multistatic scheduling, motion, persistence and replay source isolation."""

import json

import numpy as np
import pytest

from echorin.application.sensor_network import SensorNetwork
from echorin.config import NoiseConfig, SensorConfig, SimulationConfig
from echorin.environment.clutter import ClutterConfig
from echorin.environment.config import EnvironmentConfig
from echorin.recording import load_scenario_json, save_scenario_json
from echorin.sensors.components import SensorPlatform
from echorin.sensors.serialization import platforms_to_data
from echorin.simulation.scenarios import bistatic_scenario


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_every_tx_rx_pair_processed_independently_and_reset_repeatable(mode):
    config = getattr(SensorConfig, mode)(
        max_range_m=3000 if mode == "radar" else 60, noise_model=NoiseConfig(0.0001)
    )
    world = bistatic_scenario(config, multistatic=True)
    network = SensorNetwork(world.sensor_platforms, world.environment_config, seed=11)
    assert len(network.links) == 4
    initial = [p.state.position_m.copy() for p in world.sensor_platforms]
    world.advance(0.1)
    assert all(
        np.any(p.state.position_m != before)
        for p, before in zip(world.sensor_platforms, initial, strict=True)
    )
    network.sync_poses(world.sensor_platforms)
    first = network.process(world.targets, world.time_s, 16)
    assert len(first) == 4
    for key, result in first.items():
        assert result.detections
        assert (
            result.range_angle_product.emitter_id,
            result.range_angle_product.receiver_id,
        ) == key
        assert {(d.emitter_id, d.receiver_id) for d in result.detections} == {key}
        assert all(t.track_id >= 1 for t in result.tracks)
    world.reset()
    network = SensorNetwork(world.sensor_platforms, world.environment_config, seed=11)
    world.advance(0.1)
    network.sync_poses(world.sensor_platforms)
    second = network.process(world.targets, world.time_s, 16)
    for key in first:
        np.testing.assert_array_equal(
            first[key].sensor_frame.received_signal,
            second[key].sensor_frame.received_signal,
        )
        assert first[key].detections == second[key].detections


def test_schema_four_roundtrip_and_schema_three_legacy_defaults(tmp_path):
    config = SensorConfig(noise_model=NoiseConfig(0.001))
    world = bistatic_scenario(config, multistatic=True)
    world.advance(0.1)
    path = tmp_path / "network.json"
    save_scenario_json(path, world, SimulationConfig(), config)
    data = json.loads(path.read_text())
    assert data["schema_version"] == 4
    restored, _, _ = load_scenario_json(path)
    assert len(restored.sensor_platforms) == 4
    for left, right in zip(
        world.sensor_platforms, restored.sensor_platforms, strict=True
    ):
        assert left.platform_id == right.platform_id
        np.testing.assert_array_equal(left.state.position_m, right.state.position_m)
        assert platforms_to_data((left,)) == platforms_to_data((right,))
    world.advance(0.2)
    restored.advance(0.2)
    np.testing.assert_allclose(
        world.sensor_platforms[0].state.position_m,
        restored.sensor_platforms[0].state.position_m,
    )
    data["schema_version"] = 3
    data.pop("sensor_platforms")
    path.write_text(json.dumps(data))
    legacy, _, _ = load_scenario_json(path)
    assert legacy.sensor_platforms == ()


def test_network_rejects_duplicate_ids_and_excessive_work():
    config = SensorConfig()
    world = bistatic_scenario(config)
    platform = world.sensor_platforms[0]
    with pytest.raises(ValueError, match="unique"):
        SensorNetwork((platform, platform, world.sensor_platforms[1]))
    lone = SensorPlatform("empty", platform.state)
    with pytest.raises(ValueError, match="emitter.*receiver"):
        SensorNetwork((lone,))


def test_all_links_observe_same_persistent_environment_field():
    config = SensorConfig(max_range_m=3000)
    world = bistatic_scenario(config, multistatic=True)
    environment = EnvironmentConfig(clutter=(ClutterConfig(density_per_m=0.001),))
    network = SensorNetwork(world.sensor_platforms, environment)
    fields = [link.sensor._field for link in network.links.values()]
    assert fields[0]
    assert all(field is fields[0] for field in fields)
