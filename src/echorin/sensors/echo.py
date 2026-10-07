"""Shared active TX-target-RX echo mechanics for radar and sonar."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from math import pi

import numpy as np
from numpy.typing import NDArray

from echorin.config import SensorConfig, SensorMode
from echorin.dsp.doppler import radial_velocity_to_doppler_hz
from echorin.environment.config import EnvironmentConfig
from echorin.environment.field import build_field
from echorin.models.geometry import SensorPose
from echorin.propagation.geometry import path_geometry
from echorin.propagation.multipath import expand_paths
from echorin.sensors.array import ArrayGeometry, uniform_linear_array
from echorin.sensors.base import (
    ArrayPulseData,
    DirectionalPulseTrainFrame,
    DirectionalSensorFrame,
    PulseTrainFrame,
    ReflectiveTarget,
    Sensor,
    SensorFrame,
)
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.scan import ScanScheduler
from echorin.signals.noise import add_awgn
from echorin.signals.propagation import amplitude_at_range, delayed_echo, sample_delay
from echorin.signals.waveform import WaveformKind, generate_waveform
from echorin.simulation.kinematics import relative_geometry


class SyntheticMonostaticSensor(Sensor):
    """Common sampled echo pipeline parameterized by propagation mode."""

    def __init__(
        self,
        config: SensorConfig,
        expected_mode: SensorMode,
        pose: SensorPose | None = None,
        random_seed: int = 7,
        waveform_kind: WaveformKind = WaveformKind.LFM,
        bearing_noise_std_rad: float = 0.002,
        array_geometry: ArrayGeometry | None = None,
    ) -> None:
        if config.mode is not expected_mode:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"{expected_mode.value} mode configuration"
            )
        if bearing_noise_std_rad < 0.0:
            raise ValueError("bearing_noise_std_rad must be nonnegative")
        self.config = config
        self.pose = pose or SensorPose()
        self.transmitter_pose: SensorPose | None = None
        self.transmitter_beam: BeamConfig | None = None
        self.transmitter_orientation_rad = 0.0
        self.transmit_power_scale = 1.0
        self.emitter_id: str | None = None
        self.receiver_id = "receiver-0"
        self.waveform_kind = waveform_kind
        self.bearing_noise_std_rad = bearing_noise_std_rad
        wavelength = config.propagation_speed_mps / config.carrier_frequency_hz
        self.array_geometry = array_geometry or uniform_linear_array(
            8,
            wavelength / 2,
            carrier_frequency_hz=config.carrier_frequency_hz,
            propagation_speed_mps=config.propagation_speed_mps,
        )
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._environment_origin = self.pose
        self.environment = EnvironmentConfig()
        self.beam = BeamConfig()
        self.configure_environment(self.environment, self.beam)

    def configure_environment(
        self, environment: EnvironmentConfig, beam: BeamConfig
    ) -> None:
        """Validate and seed optional effects independently of legacy AWGN.

        Called at construction/rebuild, never while an acquisition is running.
        Field RNG does not consume the legacy noise stream. The CPI uses the
        boresight at its timestamp (stop-and-hop scan approximation).
        """
        for tone in environment.interference:
            if abs(tone.frequency_hz) >= self.config.sample_rate_hz / 2:
                raise ValueError("interference exceeds receiver Nyquist")
        field_rng = np.random.default_rng(
            np.random.SeedSequence(self._random_seed, spawn_key=(1,))
        )
        field = build_field(
            environment, self.config.max_range_m, self._environment_origin, field_rng
        )
        noise = environment.receiver_noise.model(self.config.noise_model)
        pattern, scan = beam.pattern(), ScanScheduler(beam.scan)
        self.environment, self.beam = environment, beam
        self._noise, self._beam_pattern, self._scan, self._field = (
            noise,
            pattern,
            scan,
            field,
        )

    def reset(self) -> None:
        """Restore the seeded noise sequence for deterministic replay."""
        self._rng = np.random.default_rng(self._random_seed)

    @property
    def is_bistatic(self) -> bool:
        """Whether two independent poses require generalized measurement axes."""
        return self.transmitter_pose is not None and self.transmitter_pose != self.pose

    def acquire(
        self, targets: Iterable[ReflectiveTarget], timestamp_s: float
    ) -> SensorFrame:
        """Synthesize one composite raw signal without perfect detections."""
        targets = tuple(targets)
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
        if (
            self.environment != EnvironmentConfig()
            or self.beam != BeamConfig()
            or self.transmitter_pose is not None
        ):
            noiseless = self._synthesize(targets, timestamp_s, 1)[0].real
        received = self._disturb(noiseless, timestamp_s)
        return SensorFrame(
            timestamp_s, transmitted, np.asarray(received, dtype=np.float64)
        )

    def acquire_directional(
        self, targets: Iterable[ReflectiveTarget], timestamp_s: float
    ) -> tuple[DirectionalSensorFrame, ...]:
        """Synthesize ideal angular channels carrying no truth identity/state."""
        transmitted = generate_waveform(self.config, self.waveform_kind)
        frames: list[DirectionalSensorFrame] = []
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m > self.config.max_range_m:
                continue
            echo = self._synthesize((target,), timestamp_s, 1)[0].real
            received = self._disturb(echo, timestamp_s)
            frames.append(
                DirectionalSensorFrame(
                    timestamp_s,
                    transmitted,
                    np.asarray(received, dtype=np.float64),
                    self._measure_bearing(geometry.bearing_rad),
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
        self._validate_pulse_count(pulse_count)
        transmitted = generate_waveform(self.config, self.waveform_kind)
        received = self._synthesize(targets, timestamp_s, pulse_count)
        noisy = self._disturb(received, timestamp_s)
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
        """Create coherent, angularly resolved channels without truth IDs."""
        self._validate_pulse_count(pulse_count)
        transmitted = generate_waveform(self.config, self.waveform_kind)
        frames: list[DirectionalPulseTrainFrame] = []
        for target in targets:
            geometry = relative_geometry(self.pose, target)
            if geometry.range_m > self.config.max_range_m:
                continue
            received = self._synthesize((target,), timestamp_s, pulse_count)
            noisy = self._disturb(received, timestamp_s)
            frames.append(
                DirectionalPulseTrainFrame(
                    timestamp_s,
                    transmitted,
                    np.asarray(noisy, dtype=np.complex128),
                    self._measure_bearing(geometry.bearing_rad),
                )
            )
        return tuple(frames)

    def acquire_array_pulse_train(
        self,
        targets: Iterable[ReflectiveTarget],
        timestamp_s: float,
        pulse_count: int,
    ) -> ArrayPulseData:
        """Accumulate all target echoes in physical receiver channels before DSP."""
        self._validate_pulse_count(pulse_count)
        transmitted = generate_waveform(self.config, self.waveform_kind)
        received = self._synthesize(targets, timestamp_s, pulse_count, array=True)
        noisy = self._disturb(received, timestamp_s)
        return ArrayPulseData(
            timestamp_s,
            transmitted,
            np.asarray(noisy, dtype=np.complex128),
            self.config.sample_rate_hz,
            self.config.prf_hz,
            self.config.carrier_frequency_hz,
            self.pose,
            self.array_geometry,
            self.receiver_id,
            self.emitter_id,
            self.transmitter_pose or self.pose,
            self.is_bistatic,
        )

    def _synthesize(
        self,
        targets: Iterable[ReflectiveTarget],
        timestamp_s: float,
        pulse_count: int,
        *,
        array: bool = False,
    ) -> NDArray[np.complex128]:
        """Accumulate target/path/field returns before noise or numerical DSP."""
        transmitted = generate_waveform(self.config, self.waveform_kind)
        elements = self.array_geometry.element_count if array else 1
        received = np.zeros(
            (elements, pulse_count, self.config.acquisition_samples),
            dtype=np.complex128,
        )
        wavelength = (
            self.config.propagation_speed_mps / self.config.carrier_frequency_hz
        )
        relative = (
            self.array_geometry.element_positions_m
            - self.array_geometry.element_positions_m[
                self.array_geometry.reference_element
            ]
        )
        sources = (
            *targets,
            *(
                p.at(timestamp_s - self._environment_origin.timestamp_s)
                for p in self._field
            ),
        )
        boresight = self._scan.boresight(timestamp_s)
        tx_beam = self.transmitter_beam or self.beam
        tx_pattern = tx_beam.pattern()
        tx_boresight = ScanScheduler(tx_beam.scan).boresight(timestamp_s)
        for target in sources:
            geometry = relative_geometry(self.pose, target)
            # Preserve exact legacy arithmetic for co-located devices. Both
            # branches use the same array accumulation and downstream DSP.
            apparent_direct_range = geometry.range_m
            equivalent_velocity = geometry.radial_velocity_mps
            transmit_angle = (
                geometry.bearing_rad
                - self.pose.heading_rad
                - self.array_geometry.orientation_rad
            )
            if self.transmitter_beam is not None:
                transmit_angle += self.array_geometry.orientation_rad
            if self.is_bistatic:
                g = path_geometry(
                    self.transmitter_pose,
                    self.pose,
                    (target.x_m, target.y_m),
                    (target.vx_mps, target.vy_mps),
                    self.config.propagation_speed_mps,
                    self.config.carrier_frequency_hz,
                )
                apparent_direct_range = g.path_length_m / 2
                equivalent_velocity = g.path_rate_mps / 2
                transmit_angle = (
                    g.transmitter_bearing_rad
                    - self.transmitter_pose.heading_rad
                    - self.transmitter_orientation_rad
                )
            if apparent_direct_range > self.config.max_range_m:
                continue
            local_angle = (
                geometry.bearing_rad
                - self.pose.heading_rad
                - self.array_geometry.orientation_rad
            )
            for path in expand_paths(
                apparent_direct_range,
                self.config.propagation_speed_mps,
                self.environment.multipath,
            ):
                path = replace(
                    path, transmitter_id=self.emitter_id, receiver_id=self.receiver_id
                )
                apparent_range = path.path_length_m / 2
                if apparent_range > self.config.max_range_m:
                    continue
                angle = local_angle + path.angle_offset_rad
                gain = self._beam_pattern.gain(
                    angle - boresight, self.config.carrier_frequency_hz
                ) * tx_pattern.gain(
                    transmit_angle + path.angle_offset_rad - tx_boresight,
                    self.config.carrier_frequency_hz,
                )
                if gain == 0:
                    continue
                direction = np.array([np.cos(angle), np.sin(angle)])
                phase = (
                    np.exp(2j * np.pi * (relative @ direction) / wavelength)
                    if array
                    else np.ones(1)
                )
                scale = (
                    gain
                    * np.sqrt(self.transmit_power_scale)
                    * path.amplitude_scale
                    * 10
                    ** (-self.environment.absorption_db_per_m * path.path_length_m / 20)
                )
                amplitude = amplitude_at_range(
                    apparent_range,
                    target.reflectivity * scale,
                    exponent=self.environment.attenuation_exponent,
                )
                if self.is_bistatic:
                    extra_leg = (path.path_length_m - g.path_length_m) / 2
                    # Each leg contributes half the configured amplitude
                    # exponent. The legacy near-range cap applies per leg.
                    amplitude = (
                        target.reflectivity
                        * scale
                        * np.sqrt(
                            amplitude_at_range(
                                g.transmitter_range_m + extra_leg,
                                1,
                                exponent=self.environment.attenuation_exponent,
                            )
                            * amplitude_at_range(
                                g.receiver_range_m + extra_leg,
                                1,
                                exponent=self.environment.attenuation_exponent,
                            )
                        )
                    )
                pulse = (transmitted * amplitude).astype(np.complex128)
                slow_phase = self._doppler_phase(equivalent_velocity, pulse_count)
                response = slow_phase[:, None] * pulse[None, :]
                delay = sample_delay(apparent_range, self.config)
                extra_phase = path.phase_rad + getattr(target, "phase_rad", 0.0)
                # Only the occupied pulse span contributes. This preserves the
                # dense reference samples but avoids a full array-sized zero
                # temporary for every scatterer/secondary path.
                received[:, :, delay : delay + transmitted.size] += (
                    phase[:, None, None]
                    * response[None, :, :]
                    * np.exp(1j * extra_phase)
                )
        return received if array else received[0]

    def _disturb(self, received: NDArray, timestamp_s: float) -> NDArray:
        """Add coherent tones, then noise; validate drift at every acquisition."""
        result = received.copy()
        fast = np.arange(result.shape[-1]) / self.config.sample_rate_hz
        times = timestamp_s + fast
        if result.ndim >= 2:
            times = (
                times[None, :]
                + np.arange(result.shape[-2])[:, None] / self.config.prf_hz
            )
        for tone in self.environment.interference:
            instantaneous = tone.frequency_hz + tone.drift_hz_s * times
            if np.any(abs(instantaneous) >= self.config.sample_rate_hz / 2):
                raise ValueError("drifting interference exceeds receiver Nyquist")
            values = tone.samples(times, complex_output=np.iscomplexobj(result))
            if tone.arrival_angle_rad is not None:
                values *= self._beam_pattern.gain(
                    tone.arrival_angle_rad - self._scan.boresight(timestamp_s),
                    tone.frequency_hz,
                )
            if result.ndim == 3 and tone.arrival_angle_rad is not None:
                direction = np.array(
                    [np.cos(tone.arrival_angle_rad), np.sin(tone.arrival_angle_rad)]
                )
                relative = (
                    self.array_geometry.element_positions_m
                    - self.array_geometry.element_positions_m[
                        self.array_geometry.reference_element
                    ]
                )
                delays = ((relative @ direction) / self.config.propagation_speed_mps)[
                    :, None, None
                ]
                # Evaluate phase(t + delay) - phase(t) analytically, avoiding
                # cancellation when absolute carrier phase is very large.
                cycles = instantaneous[None, :, :] * delays + (
                    0.5 * tone.drift_hz_s * delays**2
                )
                phase = np.exp(2j * np.pi * cycles)
                values = phase * values[None, :, :]
            result += values
        if result.ndim < 3 and self.environment.receiver_noise.kind == "correlated":
            return add_awgn(result, self.config.noise_model, self._rng)
        return self._noise.add(result, self._rng)

    def _measure_bearing(self, true_bearing_rad: float) -> float:
        measured = true_bearing_rad + float(
            self._rng.normal(0.0, self.bearing_noise_std_rad)
        )
        return (measured + pi) % (2.0 * pi) - pi

    @staticmethod
    def _validate_pulse_count(pulse_count: int) -> None:
        if pulse_count < 2:
            raise ValueError("pulse_count must be at least two")

    def _coherent_target_return(
        self,
        transmitted: NDArray[np.float64],
        range_m: float,
        radial_velocity_mps: float,
        reflectivity: float,
        pulse_count: int,
    ) -> NDArray[np.complex128]:
        phase = self._doppler_phase(radial_velocity_mps, pulse_count)
        echo = delayed_echo(
            transmitted,
            range_m,
            self.config,
            reflectivity=reflectivity,
            attenuation_exponent=self.environment.attenuation_exponent,
        ).astype(np.complex128)
        return np.asarray(phase[:, None] * echo[None, :], dtype=np.complex128)

    def _doppler_phase(
        self, radial_velocity_mps: float, pulse_count: int
    ) -> NDArray[np.complex128]:
        """Shared coherent phase law for compact and dense reference synthesis."""
        doppler_hz = float(
            radial_velocity_to_doppler_hz(radial_velocity_mps, self.config)
        )
        if abs(doppler_hz) >= self.config.prf_hz / 2.0:
            raise ValueError("target Doppler exceeds the unambiguous Doppler interval")
        slow_time_s = np.arange(pulse_count, dtype=np.float64) / self.config.prf_hz
        return np.exp(1j * 2.0 * np.pi * doppler_hz * slow_time_s)
