"""Milestone 9 release, reproducibility, recording, and documentation tests."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import echorin
from echorin.benchmark import run_tracking_benchmark
from echorin.config import SensorConfig, SimulationConfig
from echorin.gui.main_window import MainWindow
from echorin.models import Detection, FrameResult, Track
from echorin.recording import FrameRecorder, load_scenario_json, save_scenario_json
from echorin.simulation.scenarios import crossing_targets, single_stationary_target

ROOT = Path(__file__).parents[1]


def test_tracking_benchmark_is_repeatable_and_improves_rmse() -> None:
    first = run_tracking_benchmark(seed=44, sample_count=160)
    second = run_tracking_benchmark(seed=44, sample_count=160)

    assert first.raw_position_rmse_m == second.raw_position_rmse_m
    assert first.filtered_position_rmse_m == second.filtered_position_rmse_m
    assert first.filtered_position_rmse_m < first.raw_position_rmse_m * 0.55
    assert first.average_processing_time_ms >= 0.0
    np.testing.assert_array_equal(
        first.filtered_positions_m, second.filtered_positions_m
    )


def test_scenario_json_round_trip_preserves_configs_pose_and_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scenario.json"
    world = crossing_targets(seed=19)
    simulation = SimulationConfig(dt_s=0.125, random_seed=19, duration_s=12.0)
    sensor = SensorConfig.sonar(max_range_m=150.0)

    save_scenario_json(path, world, simulation, sensor)
    restored_world, restored_simulation, restored_sensor = load_scenario_json(path)

    assert restored_simulation == simulation
    assert restored_sensor == sensor
    assert restored_world.sensor_pose == world.sensor_pose
    assert [target.state.tolist() for target in restored_world.targets] == [
        target.state.tolist() for target in world.targets
    ]
    assert [target.target_id for target in restored_world.targets] == [
        target.target_id for target in world.targets
    ]


def test_frame_recorder_exports_replayable_json_and_tabular_csv(tmp_path: Path) -> None:
    recorder = FrameRecorder()
    track = Track.initial(1, np.array([100.0, 50.0]), timestamp_s=1.0)
    detection = Detection(1.0, 111.8, 0.46, -4.0, 8.0, 14.0, 0.8, 10)
    recorder.capture(
        FrameResult(
            timestamp_s=1.0,
            detections=[detection],
            tracks=[track],
            timing_metrics_s={"total_s": 0.002},
        )
    )
    json_path = tmp_path / "frames.json"
    csv_path = tmp_path / "frames.csv"

    recorder.save_json(json_path)
    recorder.export_csv(csv_path)
    replayed = FrameRecorder.load_json(json_path)

    assert replayed.records == recorder.records
    assert replayed.records[0]["detections"][0]["range_m"] == detection.range_m
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["entity_type"] for row in rows} == {"detection", "track"}


def test_main_window_reports_separate_processing_and_gui_timings() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(world=single_stationary_target(range_m=1_000.0))

    window.step_once()

    required = {
        "simulation_s",
        "sensing_s",
        "dsp_s",
        "tracking_s",
        "gui_refresh_s",
        "total_s",
    }
    assert required <= window.last_timing_metrics_s.keys()
    assert all(window.last_timing_metrics_s[name] >= 0.0 for name in required)
    assert window.last_frame_result is not None
    assert window.last_frame_result.timing_metrics_s == window.last_timing_metrics_s
    window.close()
    app.processEvents()


def test_release_controls_expose_seed_timestep_noise_waveform_and_preset() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(world=single_stationary_target(range_m=100.0))

    assert window.controls.dt_spin.value() > 0.0
    assert window.controls.seed_spin.value() == window.simulation_config.random_seed
    assert window.controls.noise_spin.value() >= 0.0
    assert window.controls.waveform_combo.count() >= 2
    assert window.controls.preset_combo.count() >= 2

    window.controls.seed_spin.setValue(1234)
    window.controls.dt_spin.setValue(0.2)
    window.controls.waveform_combo.setCurrentText("Rectangular")
    assert window.simulation_config.random_seed == 1234
    assert window.simulation_config.dt_s == 0.2
    assert window.sensor.waveform_kind.value == "rectangular"
    window.close()
    app.processEvents()


def test_sonar_refresh_interval_accounts_for_larger_acquisition_workload() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(world=single_stationary_target(range_m=100.0))

    window.controls.mode_combo.setCurrentText("Sonar")

    assert window.timer.interval() >= 250
    window.close()
    app.processEvents()


def test_release_metadata_public_docs_and_assets_are_complete() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    theory = (ROOT / "docs" / "theory.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    benchmark_results = json.loads(
        (ROOT / "benchmarks" / "tracking_results.json").read_text(encoding="utf-8")
    )

    assert echorin.__version__ == "1.0.0"
    assert 'version = "1.0.0"' in pyproject
    for term in ("Radar", "Sonar", "Kalman", "Doppler", "CA-CFAR", "pytest"):
        assert term in readme
    for term in ("two-way", "matched filter", "Doppler", "CA-CFAR", "Kalman"):
        assert term.lower() in theory.lower()
    for term in ("simulation", "signals", "DSP", "tracking", "GUI", "ground truth"):
        assert term.lower() in architecture.lower()
    assert (
        benchmark_results["filtered_position_rmse_m"]
        < benchmark_results["raw_position_rmse_m"]
    )

    png = (ROOT / "screenshots" / "echorin-main.png").read_bytes()
    gif = (ROOT / "screenshots" / "echorin-demo.gif").read_bytes()
    svg = (ROOT / "benchmarks" / "tracking_benchmark.svg").read_text(encoding="utf-8")
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 10_000
    assert gif.startswith((b"GIF87a", b"GIF89a")) and len(gif) > 10_000
    assert "<svg" in svg and "Filtered track" in svg


def test_private_documents_are_narrowly_ignored_but_public_docs_are_not() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "docs/requirements.md" in ignore
    assert "docs/log.md" in ignore
    assert "docs/nextup.md" in ignore
    assert "docs/*.docx" in ignore
    assert "\ndocs/\n" not in ignore


def test_linux_ci_installs_qt_runtime_libraries_before_running_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )

    assert "runner.os == 'Linux'" in workflow
    assert "libegl1" in workflow.lower()
    assert workflow.index("libegl1") < workflow.index("python -m pytest")
