"""Seeded v1.4 evaluation through the production pipeline, with explicit truth ROI."""

from __future__ import annotations

from dataclasses import asdict
from math import cos, radians, sin
from typing import Any

import numpy as np

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.presets import environment_preset
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.factory import create_sensor
from echorin.sensors.scan import ScanConfig
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker


def run_environment_benchmark(seed: int = 7, frames: int = 4) -> dict[str, Any]:
    """Measure every selectable effect; runtime is evidence, not a CI time limit.

    A stationary point at one-third maximum range, bearing 20 degrees, zero
    Doppler is the evaluation ROI. Only this evaluation layer compares outputs
    to truth. Scanning deliberately blanks alternating dwells. Additional
    detections include genuine clutter/multipath, so are not all false alarms.
    """
    if frames < 1:
        raise ValueError("benchmark frames must be positive")
    rows: list[dict[str, Any]] = []
    for mode in ("radar", "sonar"):
        config = getattr(SensorConfig, mode)(
            noise_model=NoiseConfig(0.001),
            max_range_m=3_000 if mode == "radar" else 60,
        )
        range_m, bearing = config.max_range_m / 3, radians(20)
        target = Target(
            "evaluation-only", range_m * cos(bearing), range_m * sin(bearing)
        )
        beam = BeamConfig(scan=ScanConfig(boresight_rad=bearing))
        cases = [("baseline", EnvironmentConfig(), beam)]
        cases.extend(
            (
                kind,
                EnvironmentConfig(receiver_noise=ReceiverNoiseConfig(kind=kind)),
                beam,
            )
            for kind in ("colored", "impulsive", "correlated")
        )
        cases.extend(
            (name.lower(), environment_preset(preset, config), beam)
            for name, preset in (
                ("interference", "Interference"),
                ("clutter", "Clutter / Reverberation"),
                ("multipath", "Multipath"),
            )
        )
        cases.extend(
            (f"beam_{kind}", EnvironmentConfig(), BeamConfig(kind=kind, scan=beam.scan))
            for kind in ("sector", "gaussian", "sinc")
        )
        cases.append(
            (
                "scan",
                EnvironmentConfig(),
                BeamConfig(
                    kind="sector",
                    width_rad=radians(30),
                    scan=ScanConfig(
                        kind="step",
                        dwell_s=0.1,
                        step_angles_rad=(bearing, bearing + radians(70)),
                    ),
                ),
            )
        )
        for name, environment, pattern in cases:
            sensor = create_sensor(
                config, random_seed=seed, environment=environment, beam=pattern
            )
            tracker = MultiTargetTracker()
            processor = SignalProcessor(config)
            detector = CaCfarDetector(CfarConfig(edge_mode="adaptive"))
            durations, counts, errors, ghost_hits = [], [], [], 0
            pulse_count = 16 if mode == "sonar" else 32
            range_bin = config.propagation_speed_mps / (2 * config.sample_rate_hz)
            angle_bin = radians(1)
            velocity_bin = (
                config.propagation_speed_mps
                * config.prf_hz
                / (2 * config.carrier_frequency_hz * pulse_count)
            )
            for index in range(frames):
                result = process_frame(
                    (target,),
                    0.05 + index * 0.1,
                    sensor,
                    processor,
                    detector,
                    tracker,
                    pulse_count,
                )
                counts.append(len(result.detections))
                durations.append(sum(result.timing_metrics_s.values()) * 1000)
                nearby = [
                    d
                    for d in result.detections
                    if abs(d.range_m - range_m) <= 2 * range_bin
                    and abs(d.bearing_rad - bearing) <= 2 * angle_bin
                ]
                if nearby:
                    measured = min(nearby, key=lambda d: abs(d.range_m - range_m))
                    errors.append(
                        (
                            abs(measured.range_m - range_m),
                            abs(measured.bearing_rad - bearing),
                            abs(float(measured.radial_velocity_mps)),
                        )
                    )
                if environment.multipath:
                    ghost = range_m + environment.multipath[0].extra_path_length_m / 2
                    ghost_hits += any(
                        abs(d.range_m - ghost) <= 2 * range_bin
                        for d in result.detections
                    )
            maximum = np.max(errors, axis=0).tolist() if errors else [None] * 3
            rows.append(
                {
                    "mode": mode,
                    "case": name,
                    "sensor": asdict(config),
                    "environment": asdict(environment),
                    "beam": asdict(pattern),
                    "pulse_count": pulse_count,
                    "detections_per_frame": counts,
                    "target_hits": len(errors),
                    "ghost_hits": ghost_hits,
                    "max_range_error_m": maximum[0],
                    "max_bearing_error_rad": maximum[1],
                    "max_velocity_error_mps": maximum[2],
                    "range_bin_m": range_bin,
                    "angle_bin_rad": angle_bin,
                    "velocity_bin_mps": velocity_bin,
                    "mean_processing_ms": float(np.mean(durations)),
                    "max_processing_ms": max(durations),
                }
            )
    return {"seed": seed, "frames_per_case": frames, "cases": rows}
