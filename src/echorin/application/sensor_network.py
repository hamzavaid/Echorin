"""Bounded synchronous multistatic acquisition; no fusion or GUI dependency."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from echorin.application.frame_pipeline import FrameComputation, process_frame
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.config import EnvironmentConfig
from echorin.environment.field import build_field
from echorin.sensors.base import ReflectiveTarget
from echorin.sensors.components import (
    Emitter,
    Receiver,
    SensorPlatform,
    validate_platforms,
)
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.factory import create_link_sensor
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig

LinkKey = tuple[str, str]


@dataclass(slots=True)
class SensorLink:
    """One source-labeled stream with an independent legacy tracker."""

    emitter_platform_id: str
    receiver_platform_id: str
    emitter: Emitter
    receiver: Receiver
    sensor: SyntheticMonostaticSensor
    processor: SignalProcessor
    detector: CaCfarDetector
    tracker: MultiTargetTracker


class SensorNetwork:
    """Time-division transmitter slots, observed by every configured receiver.

    All slots share the simulation epoch (stop-and-hop); wall-clock scheduling
    is sequential and bounded. Every stream sums all sources before DSP. Local
    track IDs are scoped by (emitter_id, receiver_id), never fused in v1.5.
    """

    def __init__(
        self,
        platforms: Sequence[SensorPlatform],
        environment: EnvironmentConfig | None = None,
        *,
        seed: int = 7,
    ) -> None:
        validate_platforms(platforms)
        self.links: dict[LinkKey, SensorLink] = {}
        for tx_platform in platforms:
            for emitter in tx_platform.emitters:
                for rx_platform in platforms:
                    for receiver in rx_platform.receivers:
                        config = receiver.config
                        sensor = create_link_sensor(
                            emitter,
                            receiver,
                            tx_platform.emitter_pose(emitter),
                            rx_platform.receiver_pose(receiver),
                            random_seed=seed + len(self.links),
                            environment=environment,
                        )
                        self.links[(emitter.emitter_id, receiver.receiver_id)] = (
                            SensorLink(
                                tx_platform.platform_id,
                                rx_platform.platform_id,
                                emitter,
                                receiver,
                                sensor,
                                SignalProcessor(config),
                                CaCfarDetector(
                                    CfarConfig(
                                        training_cells=16,
                                        guard_cells=4,
                                        false_alarm_probability=1e-3,
                                        edge_mode="adaptive",
                                        minimum_separation_bins=max(
                                            1, config.pulse_samples // 8
                                        ),
                                    )
                                ),
                                MultiTargetTracker(
                                    TrackerConfig(
                                        measurement_std_m=max(
                                            1,
                                            config.propagation_speed_mps
                                            / (2 * config.sample_rate_hz),
                                        ),
                                        bearing_std_rad=receiver.array_config.nominal_bearing_std_rad,
                                    )
                                ),
                            )
                        )

        # A network sees one world field, not a newly seeded set of nuisance
        # sources for each transmitter slot. Noise streams remain independent.
        reference = next(iter(self.links.values())).sensor
        origin = reference.pose
        field = build_field(
            environment or EnvironmentConfig(),
            max(link.sensor.config.max_range_m for link in self.links.values()),
            origin,
            np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))),
        )
        for link in self.links.values():
            link.sensor.set_environment_field(field, origin)

    def sync_poses(self, platforms: Sequence[SensorPlatform]) -> None:
        """Apply a world epoch before dispatch; never during worker execution."""
        lookup = {p.platform_id: p for p in platforms}
        for link in self.links.values():
            link.sensor.pose = lookup[link.receiver_platform_id].receiver_pose(
                link.receiver
            )
            link.sensor.transmitter_pose = lookup[
                link.emitter_platform_id
            ].emitter_pose(link.emitter)

    def process(
        self, targets: Sequence[ReflectiveTarget], timestamp_s: float, pulse_count: int
    ) -> dict[LinkKey, FrameComputation]:
        """Process all source streams with the shared numerical pipeline."""
        return {
            key: process_frame(
                targets,
                timestamp_s,
                link.sensor,
                link.processor,
                link.detector,
                link.tracker,
                pulse_count,
            )
            for key, link in self.links.items()
        }
