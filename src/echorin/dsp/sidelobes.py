"""Known-waveform matched-filter sidelobe rejection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve

from echorin.dsp.cfar import CfarResult
from echorin.dsp.range_processing import RangeProfile


def suppress_matched_filter_sidelobes(
    cfar: CfarResult,
    profile: RangeProfile,
    transmitted_signal: ArrayLike,
) -> CfarResult:
    """Keep CFAR peaks that remain significant after stronger echo subtraction.

    A delayed point echo produces the known waveform's autocorrelation in the
    complex matched-filter profile. Greedily subtract that point-spread
    function at each accepted peak, then test weaker candidates against their
    original local CFAR thresholds. This removes deterministic range sidelobes
    without imposing a blanket minimum target separation.
    """
    waveform = np.asarray(transmitted_signal, dtype=np.complex128)
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("transmitted_signal must be a nonempty 1D waveform")
    if not cfar.detections:
        return cfar
    correlation = fftconvolve(waveform, np.conjugate(waveform[::-1]), mode="full")
    center = waveform.size - 1
    template = correlation / correlation[center]
    residual = np.asarray(profile.response, dtype=np.complex128).copy()
    accepted: set[int] = set()
    for detection in sorted(
        cfar.detections, key=lambda item: item.amplitude, reverse=True
    ):
        index = detection.source_bin
        amplitude = residual[index]
        if abs(amplitude) <= cfar.threshold[index]:
            continue
        accepted.add(index)
        start = max(0, index - center)
        stop = min(residual.size, index + center + 1)
        residual[start:stop] -= (
            amplitude * template[start - index + center : stop - index + center]
        )
    return CfarResult(
        cfar.threshold,
        cfar.noise_power,
        tuple(
            detection
            for detection in cfar.detections
            if detection.source_bin in accepted
        ),
    )
