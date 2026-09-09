"""Milestone 0 acceptance tests for package structure and base models."""

from __future__ import annotations

import importlib

import pytest

from echorin.config import NoiseConfig, SensorConfig, SensorMode, SimulationConfig
from echorin.models import FrameResult, SensorPose


def test_public_package_and_architecture_modules_import() -> None:
    modules = (
        "echorin",
        "echorin.simulation",
        "echorin.sensors",
        "echorin.signals",
        "echorin.dsp",
        "echorin.tracking",
        "echorin.models",
        "echorin.gui",
    )

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_default_configuration_is_valid_and_typed() -> None:
    sensor = SensorConfig()
    simulation = SimulationConfig()

    assert sensor.mode is SensorMode.RADAR
    assert sensor.sample_rate_hz > 0.0
    assert sensor.acquisition_samples > sensor.pulse_samples > 1
    assert simulation.dt_s > 0.0
    assert isinstance(simulation.random_seed, int)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_rate_hz", 0.0),
        ("bandwidth_hz", -1.0),
        ("pulse_width_s", 0.0),
        ("prf_hz", -2.0),
        ("max_range_m", 0.0),
    ),
)
def test_sensor_configuration_rejects_nonpositive_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        SensorConfig(**{field: value})


def test_sensor_configuration_rejects_undersampling_and_short_pri() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        SensorConfig(sample_rate_hz=1_000.0, bandwidth_hz=1_000.0)

    with pytest.raises(ValueError, match="unambiguous"):
        SensorConfig(prf_hz=100_000.0, max_range_m=10_000.0)


def test_noise_and_simulation_configuration_validation() -> None:
    with pytest.raises(ValueError, match="standard_deviation"):
        NoiseConfig(standard_deviation=-0.1)
    with pytest.raises(ValueError, match="dt_s"):
        SimulationConfig(dt_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        SimulationConfig(duration_s=-1.0)


def test_frame_result_uses_factories_for_mutable_collections() -> None:
    first = FrameResult(timestamp_s=0.0)
    second = FrameResult(timestamp_s=1.0)
    first.detections.append("marker")

    assert second.detections == []
    assert SensorPose().x_m == SensorPose().y_m == 0.0
