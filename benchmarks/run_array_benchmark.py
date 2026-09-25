"""Print a reproducible two-target angular-resolution benchmark as JSON."""

from __future__ import annotations

import json
from math import cos, sin

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.sensors.factory import create_sensor
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker


def run(seed: int = 7) -> dict[str, object]:
    """Measure two bearings at one range using only the composite array data."""
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    bearings = (-0.45, 0.45)
    targets = tuple(
        Target(str(i), 300 * cos(a), 300 * sin(a)) for i, a in enumerate(bearings)
    )
    result = process_frame(
        targets,
        0.1,
        create_sensor(config, random_seed=seed),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        16,
    )
    near = sorted(
        (d for d in result.detections if abs(d.range_m - 300) < 16),
        key=lambda detection: detection.bearing_rad,
    )
    return {
        "seed": seed,
        "mode": config.mode.value,
        "target_range_m": 300.0,
        "target_bearings_rad": list(bearings),
        "measured_bearings_rad": [d.bearing_rad for d in near],
        "range_bin_m": config.propagation_speed_mps / (2 * config.sample_rate_hz),
        "angle_bin_rad": float(
            result.range_angle_product.bearings_rad[1]
            - result.range_angle_product.bearings_rad[0]
        ),
        "array_elements": 8,
        "pulse_count": 16,
        "timing_metrics_s": result.timing_metrics_s,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
