"""Scenario serialization and replayable frame-data export."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from echorin.config import (
    ArrayConfig,
    NoiseConfig,
    SensorConfig,
    SensorMode,
    SimulationConfig,
)
from echorin.models.detection import Detection
from echorin.models.frame import FrameResult
from echorin.models.geometry import SensorPose
from echorin.models.platform import MountTransform, PlatformState
from echorin.models.track import Track
from echorin.simulation.target import Target
from echorin.simulation.trajectories import PlatformTrajectory, TrajectoryKind, Waypoint
from echorin.simulation.world import World


def save_scenario_json(
    path: str | Path,
    world: World,
    simulation_config: SimulationConfig,
    sensor_config: SensorConfig,
) -> None:
    """Serialize a complete repeatable scenario using public model fields."""
    payload = {
        "schema_version": 2,
        "simulation": asdict(simulation_config),
        "sensor": {
            **asdict(sensor_config),
            "mode": sensor_config.mode.value,
            "noise_model": asdict(sensor_config.noise_model),
        },
        "sensor_pose": asdict(world.sensor_pose),
        "platform": {
            "state": {
                "position_m": world.platform_state.position_m.tolist(),
                "velocity_mps": world.platform_state.velocity_mps.tolist(),
                "acceleration_mps2": world.platform_state.acceleration_mps2.tolist(),
                "heading_rad": world.platform_state.heading_rad,
                "angular_velocity_rad_s": world.platform_state.angular_velocity_rad_s,
                "timestamp_s": world.platform_state.timestamp_s,
            },
            "trajectory": {
                "kind": world.platform_trajectory.kind.value,
                "waypoints": [
                    {
                        "timestamp_s": point.timestamp_s,
                        "position_m": point.position_m.tolist(),
                    }
                    for point in world.platform_trajectory.waypoints
                ],
            },
            "sensor_mount": {
                "position_m": world.sensor_mount.position_m.tolist(),
                "heading_rad": world.sensor_mount.heading_rad,
            },
            "array_config": asdict(world.array_config),
        },
        "targets": [asdict(target) for target in world.targets],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scenario_json(
    path: str | Path,
) -> tuple[World, SimulationConfig, SensorConfig]:
    """Load a scenario created by :func:`save_scenario_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported scenario schema version: {version}")
    sensor_data = dict(payload["sensor"])
    sensor_data["mode"] = SensorMode(sensor_data["mode"])
    sensor_data["noise_model"] = NoiseConfig(**sensor_data["noise_model"])
    targets = [Target(**target) for target in payload["targets"]]
    if version == 1:
        world = World(targets=targets, sensor_pose=SensorPose(**payload["sensor_pose"]))
    else:
        platform = payload["platform"]
        trajectory_data = platform["trajectory"]
        trajectory = PlatformTrajectory(
            TrajectoryKind(trajectory_data["kind"]),
            tuple(Waypoint(**point) for point in trajectory_data["waypoints"]),
        )
        world = World(
            targets=targets,
            platform_state=PlatformState(**platform["state"]),
            platform_trajectory=trajectory,
            sensor_mount=MountTransform(**platform["sensor_mount"]),
            array_config=ArrayConfig(**platform.get("array_config", {})),
        )
    return (
        world,
        SimulationConfig(**payload["simulation"]),
        SensorConfig(**sensor_data),
    )


class FrameRecorder:
    """Capture truth-free detection/track summaries for export and replay."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records: list[dict[str, Any]] = records or []

    def capture(self, frame: FrameResult) -> None:
        """Append JSON-safe outputs from one published processing frame."""
        self.records.append(
            {
                "timestamp_s": frame.timestamp_s,
                "detections": [
                    asdict(detection)
                    for detection in frame.detections
                    if isinstance(detection, Detection)
                ],
                "tracks": [self._track_record(track) for track in frame.tracks],
                "platform_states": [
                    self._platform_record(state) for state in frame.platform_states
                ],
                "receiver_pose": (
                    asdict(frame.receiver_pose)
                    if isinstance(frame.receiver_pose, SensorPose)
                    else None
                ),
                "timing_metrics_s": dict(frame.timing_metrics_s),
            }
        )

    def save_json(self, path: str | Path) -> None:
        """Write captured records in replayable JSON form."""
        payload = {"schema_version": 2, "frames": self.records}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> FrameRecorder:
        """Restore captured frame records without simulation truth."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if payload.get("schema_version") != 2:
                raise ValueError("unsupported frame recording schema version")
            records = payload["frames"]
        else:
            records = payload
        if not isinstance(records, list):
            raise ValueError("frame recording must contain a JSON list")
        return cls(records)

    def export_csv(self, path: str | Path) -> None:
        """Write one flat row per detection and track."""
        fieldnames = (
            "timestamp_s",
            "entity_type",
            "id",
            "range_m",
            "bearing_rad",
            "radial_velocity_mps",
            "x_m",
            "y_m",
            "vx_mps",
            "vy_mps",
            "snr_db",
            "confidence",
            "status",
        )
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for frame in self.records:
                for detection in frame["detections"]:
                    writer.writerow(
                        {
                            "timestamp_s": frame["timestamp_s"],
                            "entity_type": "detection",
                            "range_m": detection["range_m"],
                            "bearing_rad": detection["bearing_rad"],
                            "radial_velocity_mps": detection["radial_velocity_mps"],
                            "snr_db": detection["snr_db"],
                            "confidence": detection["confidence"],
                        }
                    )
                for track in frame["tracks"]:
                    writer.writerow(
                        {
                            "timestamp_s": frame["timestamp_s"],
                            "entity_type": "track",
                            "id": track["track_id"],
                            "x_m": track["x_m"],
                            "y_m": track["y_m"],
                            "vx_mps": track["vx_mps"],
                            "vy_mps": track["vy_mps"],
                            "status": track["status"],
                        }
                    )

    @staticmethod
    def _track_record(track: Any) -> dict[str, Any]:
        if not isinstance(track, Track):
            raise TypeError("frame tracks must contain Track objects")
        return {
            "track_id": track.track_id,
            "status": track.status.value,
            "x_m": track.x_m,
            "y_m": track.y_m,
            "vx_mps": track.vx_mps,
            "vy_mps": track.vy_mps,
            "hits": track.hits,
            "misses": track.misses,
        }

    @staticmethod
    def _platform_record(state: Any) -> dict[str, Any]:
        if not isinstance(state, PlatformState):
            raise TypeError("frame platform_states must contain PlatformState objects")
        return {
            "position_m": state.position_m.tolist(),
            "velocity_mps": state.velocity_mps.tolist(),
            "acceleration_mps2": state.acceleration_mps2.tolist(),
            "heading_rad": state.heading_rad,
            "angular_velocity_rad_s": state.angular_velocity_rad_s,
            "timestamp_s": state.timestamp_s,
        }
