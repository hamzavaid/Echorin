"""Mode-dispatched sensor construction."""

from __future__ import annotations

from echorin.config import SensorConfig, SensorMode
from echorin.models.geometry import SensorPose
from echorin.sensors.array import ArrayGeometry
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.radar import RadarSensor
from echorin.sensors.sonar import SonarSensor


def create_sensor(
    config: SensorConfig,
    pose: SensorPose | None = None,
    random_seed: int = 7,
    array_geometry: ArrayGeometry | None = None,
) -> SyntheticMonostaticSensor:
    """Construct the concrete sensor selected by validated configuration."""
    if config.mode is SensorMode.RADAR:
        sensor = RadarSensor(config, pose, random_seed)
        if array_geometry is not None:
            sensor.array_geometry = array_geometry
        return sensor
    if config.mode is SensorMode.SONAR:
        sensor = SonarSensor(config, pose, random_seed)
        if array_geometry is not None:
            sensor.array_geometry = array_geometry
        return sensor
    raise ValueError(f"unsupported sensor mode: {config.mode}")
