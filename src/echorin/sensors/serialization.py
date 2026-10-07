"""JSON-safe explicit platform/device configuration (scenario schema 4)."""

from dataclasses import asdict
from typing import Any

import numpy as np

from echorin.config import ArrayConfig, NoiseConfig, SensorConfig, SensorMode
from echorin.environment.config import ReceiverNoiseConfig
from echorin.environment.serialization import beam_from_dict
from echorin.models.platform import MountTransform, PlatformState
from echorin.sensors.components import Emitter, Receiver, SensorPlatform
from echorin.simulation.trajectories import PlatformTrajectory, TrajectoryKind, Waypoint


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def platforms_to_data(platforms: tuple[SensorPlatform, ...]) -> list[dict[str, Any]]:
    """Export known device states/configs, never target identities."""
    return [_plain(asdict(platform)) for platform in platforms]


def _sensor(data: dict[str, Any]) -> SensorConfig:
    values = dict(data)
    values["mode"] = SensorMode(values["mode"])
    values["noise_model"] = NoiseConfig(**values["noise_model"])
    return SensorConfig(**values)


def platforms_from_data(data: list[dict[str, Any]]) -> tuple[SensorPlatform, ...]:
    """Reconstruct typed config with normal validation, rejecting unknown fields."""
    platforms = []
    for entry in data:
        values = dict(entry)
        values["state"] = PlatformState(**values["state"])
        trajectory = values.get("trajectory", {})
        values["trajectory"] = PlatformTrajectory(
            TrajectoryKind(trajectory.get("kind", "stationary")),
            tuple(Waypoint(**p) for p in trajectory.get("waypoints", [])),
        )
        emitters, receivers = [], []
        for device_data in values.get("emitters", []):
            device = dict(device_data)
            device["config"] = _sensor(device["config"])
            device["mount"] = MountTransform(**device.get("mount", {}))
            device["beam"] = beam_from_dict(device.get("beam", {}))
            emitters.append(Emitter(**device))
        for device_data in values.get("receivers", []):
            device = dict(device_data)
            device["config"] = _sensor(device["config"])
            device["mount"] = MountTransform(**device.get("mount", {}))
            device["beam"] = beam_from_dict(device.get("beam", {}))
            device["array_config"] = ArrayConfig(**device.get("array_config", {}))
            if device.get("noise_model") is not None:
                device["noise_model"] = ReceiverNoiseConfig(**device["noise_model"])
            receivers.append(Receiver(**device))
        values["emitters"], values["receivers"] = tuple(emitters), tuple(receivers)
        platforms.append(SensorPlatform(**values))
    return tuple(platforms)
