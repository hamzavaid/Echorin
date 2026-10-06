"""Release numerical evidence and truth-free pipeline boundaries."""

import ast
from pathlib import Path

import numpy as np
import pytest

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.config import EnvironmentConfig
from echorin.environment_benchmark import run_environment_benchmark
from echorin.propagation.multipath import MultipathComponent
from echorin.sensors.factory import create_sensor
from echorin.simulation.target import Target
from echorin.tracking.tracker import MultiTargetTracker


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_normal_detector_and_tracker_accept_direct_and_multipath_returns(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.0001))
    target_range = config.max_range_m / 4
    ghost_range = config.max_range_m / 2
    sensor = create_sensor(
        config,
        environment=EnvironmentConfig(
            multipath=(MultipathComponent(2 * (ghost_range - target_range), 0.8),)
        ),
    )
    result = process_frame(
        (Target("truth-must-not-escape", target_range, 0),),
        0.1,
        sensor,
        SignalProcessor(config),
        CaCfarDetector(CfarConfig(edge_mode="adaptive")),
        MultiTargetTracker(),
        16,
    )
    bin_m = config.propagation_speed_mps / (2 * config.sample_rate_hz)
    for expected in (target_range, ghost_range):
        assert any(abs(d.range_m - expected) <= bin_m for d in result.detections)
    assert len(result.tracks) >= 2
    assert "truth-must-not-escape" not in repr(result.detections)


def test_environment_benchmark_covers_models_and_preserves_numerical_accuracy():
    first = run_environment_benchmark(seed=7, frames=2)
    second = run_environment_benchmark(seed=7, frames=2)
    assert first["seed"] == 7
    assert first["frames_per_case"] == 2
    assert {row["mode"] for row in first["cases"]} == {"radar", "sonar"}
    assert {row["case"] for row in first["cases"]} >= {
        "baseline",
        "colored",
        "impulsive",
        "correlated",
        "interference",
        "clutter",
        "multipath",
        "scan",
    }
    for a, b in zip(first["cases"], second["cases"], strict=True):
        assert a["environment"] == b["environment"]
        assert a["detections_per_frame"] == b["detections_per_frame"]
        assert a["target_hits"] == b["target_hits"]
        assert a["mean_processing_ms"] >= 0
        assert np.isfinite(a["mean_processing_ms"])
        if a["case"] == "baseline":
            assert a["target_hits"] == 2
            assert a["max_range_error_m"] <= a["range_bin_m"]
            assert a["max_bearing_error_rad"] <= a["angle_bin_rad"]
            assert a["max_velocity_error_mps"] <= a["velocity_bin_mps"]


def test_numerical_environment_and_propagation_have_no_gui_dependencies():
    source = Path(__file__).parents[1] / "src" / "echorin"
    for directory in ("environment", "propagation", "sensors", "dsp", "tracking"):
        for path in (source / directory).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    name = node.module or ""
                    assert not name.startswith(("echorin.gui", "PySide6", "pyqtgraph"))
