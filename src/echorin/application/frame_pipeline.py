"""Synchronous, deterministic processing of one acquired simulation frame."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from echorin.dsp.cfar import CaCfarDetector, CfarResult
from echorin.dsp.doppler import (
    DopplerProduct,
    doppler_spectrum,
    enrich_detections_with_velocity,
)
from echorin.dsp.range_processing import RangeProfile, SignalProcessor
from echorin.models.detection import Detection
from echorin.models.track import Track
from echorin.sensors.base import ReflectiveTarget, SensorFrame
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.tracking.tracker import MultiTargetTracker


@dataclass(frozen=True, slots=True)
class FrameComputation:
    """Sensor-derived products with no target identity or GUI dependency."""

    timestamp_s: float
    sensor_frame: SensorFrame
    range_profile: RangeProfile
    cfar_result: CfarResult
    doppler_product: DopplerProduct
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    timing_metrics_s: dict[str, float]


def process_frame(
    targets: Sequence[ReflectiveTarget],
    timestamp_s: float,
    sensor: SyntheticMonostaticSensor,
    signal_processor: SignalProcessor,
    cfar_detector: CaCfarDetector,
    tracker: MultiTargetTracker,
    pulse_count: int,
) -> FrameComputation:
    """Synthesize, filter, detect and track one frame outside the Qt thread."""
    phase_started = perf_counter()
    directional_pulse_trains = sensor.acquire_directional_pulse_trains(
        targets, timestamp_s=timestamp_s, pulse_count=pulse_count
    )
    if directional_pulse_trains:
        combined_pulses = sum(
            (frame.received_pulses for frame in directional_pulse_trains),
            start=np.zeros(
                (pulse_count, sensor.config.acquisition_samples),
                dtype=np.complex128,
            ),
        )
        transmitted = directional_pulse_trains[0].transmitted_signal
    else:
        pulse_train = sensor.acquire_pulse_train(
            (), timestamp_s=timestamp_s, pulse_count=pulse_count
        )
        combined_pulses = pulse_train.received_pulses
        transmitted = pulse_train.transmitted_signal
    sensing_s = perf_counter() - phase_started

    phase_started = perf_counter()
    sensor_frame = SensorFrame(timestamp_s, transmitted, combined_pulses[0])
    range_profile = signal_processor.range_profile(
        sensor_frame.received_signal, sensor_frame.transmitted_signal
    )
    # The combined channel has no bearing estimate; directional frames carry
    # their existing noisy angular measurement without target identity.
    cfar_result = cfar_detector.detect(
        range_profile, timestamp_s=timestamp_s, bearing_rad=float("nan")
    )
    detections: list[Detection] = []
    for directional_frame in directional_pulse_trains:
        range_responses = signal_processor.pulse_matrix_range_responses(
            directional_frame.received_pulses,
            directional_frame.transmitted_signal,
        )
        directional_profile = RangeProfile(
            signal_processor.range_axis(range_responses.shape[1]),
            range_responses[0],
        )
        directional_result = cfar_detector.detect(
            directional_profile,
            timestamp_s=timestamp_s,
            bearing_rad=directional_frame.bearing_rad,
        )
        directional_doppler = doppler_spectrum(range_responses, sensor.config)
        detections.extend(
            enrich_detections_with_velocity(
                directional_result.detections, directional_doppler
            )
        )
    combined_responses = signal_processor.pulse_matrix_range_responses(
        combined_pulses, transmitted
    )
    doppler_product = doppler_spectrum(combined_responses, sensor.config)
    dsp_s = perf_counter() - phase_started

    phase_started = perf_counter()
    tracks = tracker.update(detections, timestamp_s=timestamp_s)
    tracking_s = perf_counter() - phase_started
    return FrameComputation(
        timestamp_s=timestamp_s,
        sensor_frame=sensor_frame,
        range_profile=range_profile,
        cfar_result=cfar_result,
        doppler_product=doppler_product,
        detections=tuple(detections),
        tracks=tuple(deepcopy(track) for track in tracks),
        timing_metrics_s={
            "sensing_s": sensing_s,
            "dsp_s": dsp_s,
            "tracking_s": tracking_s,
        },
    )
