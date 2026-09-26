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
from echorin.dsp.range_angle import (
    RangeAngleProduct,
    angle_detections,
    range_angle_product,
)
from echorin.dsp.range_processing import RangeProfile, SignalProcessor
from echorin.dsp.sidelobes import suppress_matched_filter_sidelobes
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
    range_angle_product: RangeAngleProduct
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
    array_frame = sensor.acquire_array_pulse_train(
        targets, timestamp_s=timestamp_s, pulse_count=pulse_count
    )
    combined_pulses = array_frame.samples[array_frame.array_geometry.reference_element]
    transmitted = array_frame.transmitted_signal
    sensing_s = perf_counter() - phase_started

    phase_started = perf_counter()
    sensor_frame = SensorFrame(timestamp_s, transmitted, combined_pulses[0])
    range_profile = signal_processor.range_profile(
        sensor_frame.received_signal, sensor_frame.transmitted_signal
    )
    cfar_result = cfar_detector.detect(
        range_profile, timestamp_s=timestamp_s, bearing_rad=float("nan")
    )
    cfar_result = suppress_matched_filter_sidelobes(
        cfar_result, range_profile, transmitted
    )
    reference_element = array_frame.array_geometry.reference_element
    combined_responses = signal_processor.array_range_responses(
        array_frame.samples[reference_element : reference_element + 1], transmitted
    )[0]
    doppler_product = doppler_spectrum(combined_responses, sensor.config)
    angle_responses = signal_processor.array_range_responses(
        array_frame.samples[:, :1, :], transmitted
    )
    range_angle = range_angle_product(
        angle_responses,
        signal_processor.range_axis(angle_responses.shape[-1]),
        array_frame.array_geometry,
        sensor.config,
        np.linspace(-np.pi / 2, np.pi / 2, 181),
        timestamp_s,
        receiver_pose=array_frame.receiver_pose,
    )
    detections = enrich_detections_with_velocity(
        angle_detections(cfar_result.detections, range_angle), doppler_product
    )
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
        range_angle_product=range_angle,
        detections=tuple(detections),
        tracks=tuple(deepcopy(track) for track in tracks),
        timing_metrics_s={
            "sensing_s": sensing_s,
            "dsp_s": dsp_s,
            "tracking_s": tracking_s,
        },
    )
