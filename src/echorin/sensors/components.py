"""Explicit mounted transmitter/receiver composition without DSP or Qt."""

from dataclasses import dataclass, field, replace

import numpy as np

from echorin.config import ArrayConfig, SensorConfig
from echorin.environment.config import ReceiverNoiseConfig
from echorin.models.geometry import SensorPose
from echorin.models.platform import MountTransform, PlatformState
from echorin.sensors.beam_pattern import BeamConfig
from echorin.signals.waveform import WaveformKind
from echorin.simulation.trajectories import PlatformTrajectory


@dataclass(frozen=True, slots=True)
class Emitter:
    """One active source with independent waveform, mount and transmit beam.

    transmit_power_scale is relative power; synthesis uses its square root
    as an amplitude multiplier. Carrier/waveform parameters live in config.
    """

    emitter_id: str
    config: SensorConfig
    mount: MountTransform = field(default_factory=MountTransform)
    waveform_kind: WaveformKind = WaveformKind.LFM
    transmit_power_scale: float = 1.0
    beam: BeamConfig = field(default_factory=BeamConfig)

    def __post_init__(self) -> None:
        if not self.emitter_id.strip():
            raise ValueError("emitter ID must not be empty")
        if not np.isfinite(self.transmit_power_scale) or self.transmit_power_scale < 0:
            raise ValueError("transmit power must be finite and nonnegative")
        object.__setattr__(self, "waveform_kind", WaveformKind(self.waveform_kind))


@dataclass(frozen=True, slots=True)
class Receiver:
    """One sampled receiver; array spacing derives from its sampling medium."""

    receiver_id: str
    config: SensorConfig
    mount: MountTransform = field(default_factory=MountTransform)
    array_config: ArrayConfig = field(default_factory=ArrayConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    noise_model: ReceiverNoiseConfig | None = None

    def __post_init__(self) -> None:
        if not self.receiver_id.strip():
            raise ValueError("receiver ID must not be empty")


@dataclass(frozen=True, slots=True)
class SensorPlatform:
    """Independent dynamic platform carrying zero or more TX/RX devices."""

    platform_id: str
    state: PlatformState
    emitters: tuple[Emitter, ...] = ()
    receivers: tuple[Receiver, ...] = ()
    trajectory: PlatformTrajectory = field(default_factory=PlatformTrajectory)

    def __post_init__(self) -> None:
        if not self.platform_id.strip():
            raise ValueError("platform ID must not be empty")
        for devices, key in (
            (self.emitters, "emitter_id"),
            (self.receivers, "receiver_id"),
        ):
            ids = [getattr(device, key) for device in devices]
            if len(set(ids)) != len(ids):
                raise ValueError("device IDs must be unique")

    def emitter_pose(self, emitter: Emitter) -> SensorPose:
        return self.state.sensor_pose(emitter.mount)

    def receiver_pose(self, receiver: Receiver) -> SensorPose:
        return self.state.sensor_pose(receiver.mount)

    def advance(self, dt_s: float) -> "SensorPlatform":
        return replace(self, state=self.trajectory.advance(self.state, dt_s))

    @classmethod
    def monostatic(
        cls, platform_id: str, state: PlatformState, config: SensorConfig
    ) -> "SensorPlatform":
        """Convenience co-located active sensor using the generalized models."""
        return cls(
            platform_id,
            state,
            (Emitter(f"{platform_id}-tx", config),),
            (Receiver(f"{platform_id}-rx", config),),
        )
