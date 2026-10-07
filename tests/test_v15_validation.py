"""Moving bistatic/multistatic release evidence and public boundaries."""

import numpy as np
import pytest

from echorin.bistatic_benchmark import run_bistatic_benchmark


def test_moving_bistatic_benchmark_matches_analytic_bins_and_is_repeatable():
    first = run_bistatic_benchmark(seed=17, frames=3)
    second = run_bistatic_benchmark(seed=17, frames=3)
    assert first["seed"] == 17
    assert {row["mode"] for row in first["cases"]} == {"radar", "sonar"}
    assert len(first["cases"]) == 8
    for a, b in zip(first["cases"], second["cases"], strict=True):
        assert a["detections_per_frame"] == b["detections_per_frame"]
        assert a["measured_path_m"] == b["measured_path_m"]
        assert a["measured_path_rate_mps"] == b["measured_path_rate_mps"]
        assert a["hits"] == 3
        assert a["max_path_error_m"] <= a["path_bin_m"]
        assert a["max_path_rate_error_mps"] <= a["path_rate_bin_mps"]
        assert a["max_bearing_error_rad"] <= a["angle_bin_rad"]
        assert np.isfinite(a["mean_processing_ms"])
        assert a["mean_processing_ms"] >= 0
        assert a["config"] == b["config"]
    assert any(
        abs(rate) > 0
        for row in first["cases"]
        for rate in row["measured_path_rate_mps"]
    )


def test_benchmark_rejects_invalid_frame_count():
    with pytest.raises(ValueError):
        run_bistatic_benchmark(frames=0)
