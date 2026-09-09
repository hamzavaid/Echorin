"""Scenario serialization and replayable frame-data export."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from echorin.config import NoiseConfig, SensorConfig, SensorMode, SimulationConfig
from echorin.models.detection import Detection
from echorin.models.frame import FrameResult
from echorin.models.geometry import SensorPose
from echorin.models.track import Track
from echorin.simulation.target import Target
from echorin.simulation.world import World


def save_scenario_json(
    path: str | Path,
    world: World,
    simulation_config: SimulationConfig,
    sensor_config: SensorConfig,
) -> None:
    """Serialize a complete repeatable scenario using public model fields."""
    payload = {
        "simulation": asdict(simulation_config),
        "sensor": {
            **asdict(sensor_config),
            "mode": sensor_config.mode.value,
            "noise_model": asdict(sensor_config.noise_model),
        },
        "sensor_pose": asdict(world.sensor_pose),
        "targets": [asdict(target) for target in world.targets],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scenario_json(
    path: str | Path,
) -> tuple[World, SimulationConfig, SensorConfig]:
    """Load a scenario created by :func:`save_scenario_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sensor_data = dict(payload["sensor"])
    sensor_data["mode"] = SensorMode(sensor_data["mode"])
    sensor_data["noise_model"] = NoiseConfig(**sensor_data["noise_model"])
    world = World(
        targets=[Target(**target) for target in payload["targets"]],
        sensor_pose=SensorPose(**payload["sensor_pose"]),
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
                "timing_metrics_s": dict(frame.timing_metrics_s),
            }
        )

    def save_json(self, path: str | Path) -> None:
        """Write captured records in replayable JSON form."""
        Path(path).write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> FrameRecorder:
        """Restore captured frame records without simulation truth."""
        records = json.loads(Path(path).read_text(encoding="utf-8"))
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
