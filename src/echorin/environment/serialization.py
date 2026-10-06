"""Strict model reconstruction shared by scenario IO and engineering controls."""

from typing import Any

from echorin.environment.clutter import ClutterConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.propagation.multipath import MultipathComponent
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.scan import ScanConfig


def environment_from_dict(payload: dict[str, Any]) -> EnvironmentConfig:
    """Validate all nested settings; dataclasses reject unknown fields."""
    data = dict(payload)
    data["receiver_noise"] = ReceiverNoiseConfig(**data.get("receiver_noise", {}))
    data["interference"] = tuple(
        NarrowbandInterference(**item) for item in data.get("interference", ())
    )
    data["clutter"] = tuple(ClutterConfig(**item) for item in data.get("clutter", ()))
    data["multipath"] = tuple(
        MultipathComponent(**item) for item in data.get("multipath", ())
    )
    return EnvironmentConfig(**data)


def beam_from_dict(payload: dict[str, Any]) -> BeamConfig:
    """Reconstruct a beam including tuple-valued discrete dwell angles."""
    data = dict(payload)
    data["scan"] = ScanConfig(**data.get("scan", {}))
    return BeamConfig(**data)
