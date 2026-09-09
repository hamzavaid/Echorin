"""Mode-dispatched sensor construction."""

from __future__ import annotations

from echorin.config import SensorConfig, SensorMode
from echorin.models.geometry import SensorPose
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.radar import RadarSensor
from echorin.sensors.sonar import SonarSensor


def create_sensor(
    config: SensorConfig,
    pose: SensorPose | None = None,
    random_seed: int = 7,
) -> SyntheticMonostaticSensor:
    """Construct the concrete sensor selected by validated configuration."""
    if config.mode is SensorMode.RADAR:
        return RadarSensor(config, pose, random_seed)
    if config.mode is SensorMode.SONAR:
        return SonarSensor(config, pose, random_seed)
    raise ValueError(f"unsupported sensor mode: {config.mode}")
