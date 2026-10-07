"""Mode-dispatched sensor construction."""

from __future__ import annotations

from dataclasses import replace

from echorin.config import SensorConfig, SensorMode
from echorin.environment.config import EnvironmentConfig
from echorin.models.geometry import SensorPose
from echorin.sensors.array import ArrayGeometry, geometry_for_config
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.components import Emitter, Receiver
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.radar import RadarSensor
from echorin.sensors.sonar import SonarSensor


def create_sensor(
    config: SensorConfig,
    pose: SensorPose | None = None,
    random_seed: int = 7,
    array_geometry: ArrayGeometry | None = None,
    environment: EnvironmentConfig | None = None,
    beam: BeamConfig | None = None,
) -> SyntheticMonostaticSensor:
    """Construct the concrete sensor selected by validated configuration."""
    if config.mode is SensorMode.RADAR:
        sensor = RadarSensor(config, pose, random_seed)
        if array_geometry is not None:
            sensor.array_geometry = array_geometry
        sensor.configure_environment(
            environment or EnvironmentConfig(), beam or BeamConfig()
        )
        return sensor
    if config.mode is SensorMode.SONAR:
        sensor = SonarSensor(config, pose, random_seed)
        if array_geometry is not None:
            sensor.array_geometry = array_geometry
        sensor.configure_environment(
            environment or EnvironmentConfig(), beam or BeamConfig()
        )
        return sensor
    raise ValueError(f"unsupported sensor mode: {config.mode}")


def create_link_sensor(
    emitter: Emitter,
    receiver: Receiver,
    transmitter_pose: SensorPose,
    receiver_pose: SensorPose,
    *,
    random_seed: int = 7,
    environment: EnvironmentConfig | None = None,
) -> SyntheticMonostaticSensor:
    """Build one synchronized time-division TX/RX CPI using shared synthesis.

    Each receiver sums every target/path/field return before processing.
    Different transmitters are scheduled in separate labeled slots, not
    separated using hidden target channels or idealized signal attribution.
    """
    fields = (
        "mode",
        "sample_rate_hz",
        "carrier_frequency_hz",
        "bandwidth_hz",
        "pulse_width_s",
        "prf_hz",
        "propagation_speed_mps",
    )
    if any(getattr(emitter.config, k) != getattr(receiver.config, k) for k in fields):
        raise ValueError("TX/RX must have compatible medium and waveform sampling")
    environment = environment or EnvironmentConfig()
    if receiver.noise_model is not None:
        environment = replace(environment, receiver_noise=receiver.noise_model)
    sensor = create_sensor(
        receiver.config,
        receiver_pose,
        random_seed,
        geometry_for_config(receiver.array_config, receiver.config),
        environment,
        receiver.beam,
    )
    sensor.transmitter_pose = transmitter_pose
    sensor.transmitter_beam = emitter.beam
    sensor.waveform_kind = emitter.waveform_kind
    sensor.transmit_power_scale = emitter.transmit_power_scale
    sensor.emitter_id, sensor.receiver_id = emitter.emitter_id, receiver.receiver_id
    return sensor
