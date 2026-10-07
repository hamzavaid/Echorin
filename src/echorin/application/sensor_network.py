"""Bounded synchronous multistatic acquisition; no fusion or GUI dependency."""

from collections.abc import Sequence
from dataclasses import dataclass

from echorin.application.frame_pipeline import FrameComputation, process_frame
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.config import EnvironmentConfig
from echorin.sensors.base import ReflectiveTarget
from echorin.sensors.components import Emitter, Receiver, SensorPlatform
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.factory import create_link_sensor
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig

LinkKey = tuple[str, str]
MAX_LINKS = 16


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


def validate_platforms(platforms: Sequence[SensorPlatform]) -> None:
    """Validate identity and work bounds before constructing raw buffers."""
    for ids in (
        [p.platform_id for p in platforms],
        [d.emitter_id for p in platforms for d in p.emitters],
        [d.receiver_id for p in platforms for d in p.receivers],
    ):
        if len(set(ids)) != len(ids):
            raise ValueError("platform and device IDs must be globally unique")
    transmitters = sum(len(p.emitters) for p in platforms)
    receivers = sum(len(p.receivers) for p in platforms)
    if not transmitters or not receivers:
        raise ValueError("network requires an emitter and a receiver")
    if transmitters * receivers > MAX_LINKS:
        raise ValueError(f"network exceeds {MAX_LINKS}-link work limit")
    if len({p.state.timestamp_s for p in platforms}) != 1:
        raise ValueError("network platform timestamps must match")


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
    ):
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
        for link in self.links.values():
            link.sensor._field = reference._field
            link.sensor._environment_origin = reference._environment_origin

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
