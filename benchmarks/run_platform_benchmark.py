"""Reproducible moving-receiver range, bearing, and Doppler benchmark."""

from __future__ import annotations

import json
from math import atan2, hypot

import numpy as np

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.models.platform import PlatformState
from echorin.sensors.factory import create_sensor
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.target import Target
from echorin.simulation.trajectories import PlatformTrajectory, TrajectoryKind
from echorin.simulation.world import World
from echorin.tracking.tracker import MultiTargetTracker


def run(seed: int = 7) -> dict[str, float | int]:
    """Compare signal-derived measurements to analytic moving-platform geometry."""
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    target = Target("evaluation-truth", 400.0, 100.0)
    world = World(
        (target,),
        platform_state=PlatformState(
            np.zeros(2), np.array([10_000.0, 0.0]), np.zeros(2), 0.0, 0.0, 0.0
        ),
        platform_trajectory=PlatformTrajectory(TrajectoryKind.CONSTANT_VELOCITY),
    )
    world.advance(0.01)
    truth = relative_geometry(world.sensor_pose, target)
    result = process_frame(
        world.targets,
        world.time_s,
        create_sensor(config, world.sensor_pose, random_seed=seed),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        32,
    )
    detection = min(
        result.detections,
        key=lambda item: (
            abs(item.range_m - truth.range_m)
            + 100.0 * abs(item.bearing_rad - truth.bearing_rad)
        ),
    )
    velocity_axis = result.doppler_product.radial_velocity_mps
    return {
        "seed": seed,
        "receiver_x_m": world.sensor_pose.x_m,
        "analytic_range_m": hypot(target.x_m - world.sensor_pose.x_m, target.y_m),
        "analytic_bearing_rad": atan2(target.y_m, target.x_m - world.sensor_pose.x_m),
        "analytic_radial_velocity_mps": truth.radial_velocity_mps,
        "measured_range_m": detection.range_m,
        "measured_bearing_rad": detection.bearing_rad,
        "measured_radial_velocity_mps": float(detection.radial_velocity_mps),
        "range_error_m": abs(detection.range_m - truth.range_m),
        "bearing_error_rad": abs(detection.bearing_rad - truth.bearing_rad),
        "radial_velocity_error_mps": abs(
            float(detection.radial_velocity_mps) - truth.radial_velocity_mps
        ),
        "range_bin_m": config.propagation_speed_mps / (2 * config.sample_rate_hz),
        "angle_bin_rad": float(np.diff(result.range_angle_product.bearings_rad)[0]),
        "velocity_bin_mps": float(abs(velocity_axis[1] - velocity_axis[0])),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
