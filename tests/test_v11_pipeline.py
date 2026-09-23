"""GUI-independent frame controller behavior for responsive live updates."""

from __future__ import annotations

import numpy as np

from echorin.application.frame_pipeline import process_frame
from echorin.config import SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.sensors.factory import create_sensor
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig


def test_frame_pipeline_is_deterministic_and_publishes_truth_free_products() -> None:
    config = SensorConfig.radar(max_range_m=500.0)

    def run():
        return process_frame(
            targets=(Target("private", 100.0, 0.0),),
            timestamp_s=0.1,
            sensor=create_sensor(config, random_seed=8),
            signal_processor=SignalProcessor(config),
            cfar_detector=CaCfarDetector(CfarConfig()),
            tracker=MultiTargetTracker(TrackerConfig()),
            pulse_count=16,
        )

    first, second = run(), run()
    np.testing.assert_array_equal(
        first.doppler_product.spectrum, second.doppler_product.spectrum
    )
    assert first.detections == second.detections
    assert all(not hasattr(detection, "target_id") for detection in first.detections)
    assert first.doppler_product.ranges_m is not None
    assert set(first.timing_metrics_s) == {"sensing_s", "dsp_s", "tracking_s"}
