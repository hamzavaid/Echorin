"""Matched filtering with physical delay-lag alignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve


def matched_filter(
    received_signal: ArrayLike, transmitted_signal: ArrayLike
) -> NDArray[np.floating] | NDArray[np.complexfloating]:
    """Correlate received samples against the known transmitted waveform.

    The returned array has the received signal's length and starts at correlation
    lag zero, so index ``d`` directly represents a ``d``-sample echo delay.
    """
    received = np.asarray(received_signal)
    transmitted = np.asarray(transmitted_signal)
    if received.ndim != 1 or transmitted.ndim != 1:
        raise ValueError("matched-filter inputs must be one-dimensional")
    if received.size == 0 or transmitted.size == 0:
        raise ValueError("matched-filter inputs must be nonempty")
    correlation = fftconvolve(received, np.conjugate(transmitted[::-1]), mode="full")
    zero_lag = transmitted.size - 1
    return correlation[zero_lag : zero_lag + received.size]
