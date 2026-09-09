"""Digital signal-processing algorithms."""

from echorin.dsp.cfar import CaCfarDetector, CfarConfig, CfarResult
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
    "FixedThresholdDetector",
    "RangeProfile",
    "SignalProcessor",
    "matched_filter",
]
