"""Synthetic monostatic radar."""

from __future__ import annotations

from echorin.config import SensorConfig, SensorMode
from echorin.models.geometry import SensorPose
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.signals.waveform import WaveformKind


class RadarSensor(SyntheticMonostaticSensor):
    """Radar specialization of the shared monostatic echo sensor."""

    def __init__(
        self,
        config: SensorConfig,
        pose: SensorPose | None = None,
        random_seed: int = 7,
        waveform_kind: WaveformKind = WaveformKind.LFM,
        bearing_noise_std_rad: float = 0.002,
    ) -> None:
        super().__init__(
            config,
            SensorMode.RADAR,
            pose,
            random_seed,
            waveform_kind,
            bearing_noise_std_rad,
        )
