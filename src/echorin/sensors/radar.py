"""Synthetic monostatic radar implementation."""

from __future__ import annotations

from collections.abc import Iterable
from math import pi

import numpy as np

from echorin.config import SensorConfig, SensorMode
from echorin.dsp.doppler import radial_velocity_to_doppler_hz
from echorin.models import SensorPose
from echorin.sensors.base import (
    DirectionalPulseTrainFrame,
    DirectionalSensorFrame,
    PulseTrainFrame,
    ReflectiveTarget,
    Sensor,
    SensorFrame,
)
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
        bearing_noise_std_rad: float = 0.002,
    ) -> None:
        if config.mode is not SensorMode.RADAR:
            raise ValueError("RadarSensor requires radar mode configuration")
        self.config = config
        self.pose = pose or SensorPose()
        self.waveform_kind = waveform_kind
        if bearing_noise_std_rad < 0.0:
            raise ValueError("bearing_noise_std_rad must be nonnegative")
        self.bearing_noise_std_rad = bearing_noise_std_rad
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

    def acquire_directional(
        self, targets: Iterable[ReflectiveTarget], timestamp_s: float
    ) -> tuple[DirectionalSensorFrame, ...]:
        """Synthesize ideal beamformed angular channels with noisy bearing.

        Each channel contains one angularly resolved return. It deliberately
        carries no simulation identity or exact state; downstream algorithms see
        only samples and the sensor's noisy angle-of-arrival measurement.
        """
        transmitted = generate_waveform(self.config, self.waveform_kind)
        frames: list[DirectionalSensorFrame] = []
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m > self.config.max_range_m:
                continue
            echo = delayed_echo(
                transmitted,
                geometry.range_m,
                self.config,
                reflectivity=target.reflectivity,
            )
            received = add_awgn(echo, self.config.noise_model, self._rng)
            measured_bearing = geometry.bearing_rad + float(
                self._rng.normal(0.0, self.bearing_noise_std_rad)
            )
            measured_bearing = (measured_bearing + pi) % (2.0 * pi) - pi
            frames.append(
                DirectionalSensorFrame(
                    timestamp_s,
                    transmitted,
                    received,
                    measured_bearing,
                )
            )
        return tuple(frames)

    def acquire_pulse_train(
        self,
        targets: Iterable[ReflectiveTarget],
        timestamp_s: float,
        pulse_count: int,
    ) -> PulseTrainFrame:
        """Synthesize a coherent CPI under the stop-and-hop narrowband model."""
        if pulse_count < 2:
            raise ValueError("pulse_count must be at least two")
        transmitted = generate_waveform(self.config, self.waveform_kind)
        received = np.zeros(
            (pulse_count, self.config.acquisition_samples), dtype=np.complex128
        )
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m > self.config.max_range_m:
                continue
            received += self._coherent_target_return(
                transmitted,
                geometry.range_m,
                geometry.radial_velocity_mps,
                target.reflectivity,
                pulse_count,
            )
        noisy = add_awgn(received, self.config.noise_model, self._rng)
        return PulseTrainFrame(
            timestamp_s,
            transmitted,
            np.asarray(noisy, dtype=np.complex128),
        )

    def acquire_directional_pulse_trains(
        self,
        targets: Iterable[ReflectiveTarget],
        timestamp_s: float,
        pulse_count: int,
    ) -> tuple[DirectionalPulseTrainFrame, ...]:
        """Create coherent, angularly resolved sensor channels without IDs."""
        if pulse_count < 2:
            raise ValueError("pulse_count must be at least two")
        transmitted = generate_waveform(self.config, self.waveform_kind)
        frames: list[DirectionalPulseTrainFrame] = []
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m > self.config.max_range_m:
                continue
            received = self._coherent_target_return(
                transmitted,
                geometry.range_m,
                geometry.radial_velocity_mps,
                target.reflectivity,
                pulse_count,
            )
            noisy = add_awgn(received, self.config.noise_model, self._rng)
            measured_bearing = geometry.bearing_rad + float(
                self._rng.normal(0.0, self.bearing_noise_std_rad)
            )
            measured_bearing = (measured_bearing + pi) % (2.0 * pi) - pi
            frames.append(
                DirectionalPulseTrainFrame(
                    timestamp_s,
                    transmitted,
                    np.asarray(noisy, dtype=np.complex128),
                    measured_bearing,
                )
            )
        return tuple(frames)

    def _coherent_target_return(
        self,
        transmitted: np.ndarray,
        range_m: float,
        radial_velocity_mps: float,
        reflectivity: float,
        pulse_count: int,
    ) -> np.ndarray:
        doppler_hz = float(
            radial_velocity_to_doppler_hz(radial_velocity_mps, self.config)
        )
        if abs(doppler_hz) >= self.config.prf_hz / 2.0:
            raise ValueError("target Doppler exceeds the unambiguous Doppler interval")
        echo = delayed_echo(
            transmitted,
            range_m,
            self.config,
            reflectivity=reflectivity,
        ).astype(np.complex128)
        slow_time_s = np.arange(pulse_count, dtype=np.float64) / self.config.prf_hz
        phase = np.exp(1j * 2.0 * np.pi * doppler_hz * slow_time_s)
        return phase[:, None] * echo[None, :]
