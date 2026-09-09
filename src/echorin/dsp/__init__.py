"""Digital signal-processing algorithms."""

from echorin.dsp.cfar import CaCfarDetector, CfarConfig, CfarResult
from echorin.dsp.doppler import DopplerProduct, doppler_spectrum
from echorin.dsp.matched_filter import matched_filter
from echorin.dsp.range_processing import (
    FixedThresholdDetector,
    RangeProfile,
    SignalProcessor,
)

__all__ = [
    "CaCfarDetector",
    "CfarConfig",
    "CfarResult",
    "DopplerProduct",
    "FixedThresholdDetector",
    "RangeProfile",
    "SignalProcessor",
    "matched_filter",
    "doppler_spectrum",
]
