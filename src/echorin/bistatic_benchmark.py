"""Reproducible signal-derived moving TX/RX validation (truth only in evaluation)."""

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from echorin.application.sensor_network import SensorNetwork
from echorin.config import NoiseConfig, SensorConfig
from echorin.propagation.geometry import path_geometry
from echorin.simulation.scenarios import bistatic_scenario


def benchmark_config(mode: str) -> SensorConfig:
    """Small release configurations with resolvable nonzero bistatic Doppler."""
    if mode == "radar":
        return SensorConfig.radar(
            max_range_m=3000,
            carrier_frequency_hz=4e6,
            prf_hz=20,
            noise_model=NoiseConfig(0.0001),
        )
    if mode == "sonar":
        return SensorConfig.sonar(max_range_m=60, noise_model=NoiseConfig(0.0001))
    raise ValueError("mode must be radar or sonar")


def run_bistatic_benchmark(*, seed: int = 7, frames: int = 4) -> dict[str, Any]:
    """Compare moving multistatic measurements to both-leg analytic references.

    Bounded gates follow actual fast-time, angle and FFT bin widths; no hidden
    reference enters production sensing/detection/tracking. Timings are evidence,
    not portable performance guarantees. Every pair is also a bistatic scenario.
    """
    if frames < 1 or frames > 100:
        raise ValueError("frames must be in [1,100]")
    cases = []
    for mode in ("radar", "sonar"):
        config = benchmark_config(mode)
        world = bistatic_scenario(config, multistatic=True)
        network = SensorNetwork(world.sensor_platforms, seed=seed)
        rows = {}
        for key in network.links:
            rows[key] = {
                "mode": mode,
                "emitter_id": key[0],
                "receiver_id": key[1],
                "config": asdict(config),
                "seed": seed,
                "pulse_count": 32,
                "dt_s": 0.05,
                "hits": 0,
                "detections_per_frame": [],
                "measured_path_m": [],
                "reference_path_m": [],
                "measured_path_rate_mps": [],
                "reference_path_rate_mps": [],
                "max_path_error_m": 0.0,
                "max_path_rate_error_mps": 0.0,
                "max_bearing_error_rad": 0.0,
                "mean_processing_ms": 0.0,
                "path_bin_m": config.propagation_speed_mps / config.sample_rate_hz,
                "path_rate_bin_mps": config.propagation_speed_mps
                / config.carrier_frequency_hz
                * config.prf_hz
                / 32,
                "angle_bin_rad": float(np.deg2rad(1)),
            }
        for _ in range(frames):
            world.advance(0.05)
            network.sync_poses(world.sensor_platforms)
            started = perf_counter()
            products = network.process(world.targets, world.time_s, 32)
            elapsed = perf_counter() - started
            target = world.targets[0]
            for key, result in products.items():
                link, row = network.links[key], rows[key]
                reference = path_geometry(
                    link.sensor.transmitter_pose,
                    link.sensor.pose,
                    (target.x_m, target.y_m),
                    (target.vx_mps, target.vy_mps),
                    config.propagation_speed_mps,
                    config.carrier_frequency_hz,
                )
                row["detections_per_frame"].append(len(result.detections))
                row["mean_processing_ms"] += elapsed * 1000 / frames
                if not result.detections:
                    continue
                detection = min(
                    result.detections,
                    key=lambda d: abs(d.path_length_m - reference.path_length_m),
                )
                row["hits"] += 1
                row["measured_path_m"].append(detection.path_length_m)
                row["reference_path_m"].append(reference.path_length_m)
                row["measured_path_rate_mps"].append(detection.path_rate_mps)
                row["reference_path_rate_mps"].append(reference.path_rate_mps)
                for field, error in (
                    (
                        "max_path_error_m",
                        abs(detection.path_length_m - reference.path_length_m),
                    ),
                    (
                        "max_path_rate_error_mps",
                        abs(detection.path_rate_mps - reference.path_rate_mps),
                    ),
                    (
                        "max_bearing_error_rad",
                        abs(
                            detection.bearing_rad
                            + link.sensor.pose.heading_rad
                            - reference.receiver_bearing_rad
                        ),
                    ),
                ):
                    row[field] = max(row[field], error)
        cases.extend(rows.values())
    return {
        "seed": seed,
        "frames_per_case": frames,
        "schedule": "time_division_stop_and_hop",
        "cases": cases,
    }
