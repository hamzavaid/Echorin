"""Synthetic monostatic radar implementation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from echorin.config import SensorConfig, SensorMode
from echorin.models import SensorPose
from echorin.sensors.base import ReflectiveTarget, Sensor, SensorFrame
from echorin.signals.noise import add_awgn
from echorin.signals.propagation import delayed_echo
from echorin.signals.waveform import WaveformKind, generate_waveform
from echorin.simulation.kinematics import relative_geometry


class RadarSensor(Sensor):
    """Generate delayed, attenuated target echoes and additive Gaussian noise."""

    def __init__(
        self,
        config: SensorConfig,
        pose: SensorPose | None = None,
        random_seed: int = 7,
        waveform_kind: WaveformKind = WaveformKind.LFM,
    ) -> None:
        if config.mode is not SensorMode.RADAR:
            raise ValueError("RadarSensor requires radar mode configuration")
        self.config = config
        self.pose = pose or SensorPose()
        self.waveform_kind = waveform_kind
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def reset(self) -> None:
        """Restore the seeded noise sequence for deterministic scenario replay."""
        self._rng = np.random.default_rng(self._random_seed)

    def acquire(
        self, targets: Iterable[ReflectiveTarget], timestamp_s: float
    ) -> SensorFrame:
        """Synthesize raw received samples; publish no perfect detections."""
        transmitted = generate_waveform(self.config, self.waveform_kind)
        noiseless = np.zeros(self.config.acquisition_samples, dtype=np.float64)
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m <= self.config.max_range_m:
                noiseless += delayed_echo(
                    transmitted,
                    geometry.range_m,
                    self.config,
                    reflectivity=target.reflectivity,
                )
        received = add_awgn(noiseless, self.config.noise_model, self._rng)
        return SensorFrame(timestamp_s, transmitted, received)
