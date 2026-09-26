"""v1.3 moving-platform physics and integration acceptance tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig, SimulationConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.models.detection import Detection
from echorin.models.platform import MountTransform, PlatformState
from echorin.recording import load_scenario_json, save_scenario_json
from echorin.sensors.array import ArrayConfig
from echorin.sensors.factory import create_sensor
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.target import Target
from echorin.simulation.trajectories import (
    PlatformTrajectory,
    TrajectoryKind,
    Waypoint,
)
from echorin.simulation.world import World
from echorin.tracking.tracker import MultiTargetTracker, detection_to_cartesian


def test_moving_platform_benchmark_is_deterministic_and_bounded() -> None:
    from benchmarks.run_platform_benchmark import run

    first = run(seed=7)
    second = run(seed=7)
    assert first == second
    assert first["range_error_m"] <= first["range_bin_m"]
    assert first["bearing_error_rad"] <= 2 * first["angle_bin_rad"]
    assert first["radial_velocity_error_mps"] <= first["velocity_bin_mps"]


def test_body_mount_transform_and_rotational_velocity() -> None:
    platform = PlatformState(
        position_m=np.array([10.0, 20.0]),
        velocity_mps=np.array([2.0, 3.0]),
        acceleration_mps2=np.zeros(2),
        heading_rad=np.pi / 2,
        angular_velocity_rad_s=0.5,
        timestamp_s=4.0,
    )
    pose = platform.sensor_pose(MountTransform(np.array([2.0, 0.0]), 0.25))
    assert (pose.x_m, pose.y_m) == pytest.approx((10.0, 22.0))
    assert (pose.vx_mps, pose.vy_mps) == pytest.approx((1.0, 3.0))
    assert pose.heading_rad == pytest.approx(np.pi / 2 + 0.25)
    assert pose.timestamp_s == 4.0


def test_cv_ca_turn_and_waypoint_trajectories_are_analytical() -> None:
    initial = PlatformState(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
        0.0,
        np.pi / 2,
        0.0,
    )
    stationary = PlatformTrajectory(TrajectoryKind.STATIONARY).advance(initial, 2.0)
    np.testing.assert_allclose(stationary.position_m, [0.0, 0.0])
    np.testing.assert_allclose(stationary.velocity_mps, [0.0, 0.0])
    assert stationary.angular_velocity_rad_s == 0.0
    cv = PlatformTrajectory(TrajectoryKind.CONSTANT_VELOCITY).advance(initial, 2.0)
    np.testing.assert_allclose(cv.position_m, [4.0, 0.0])
    ca = PlatformTrajectory(TrajectoryKind.CONSTANT_ACCELERATION).advance(initial, 2.0)
    np.testing.assert_allclose(ca.position_m, [6.0, 0.0])
    np.testing.assert_allclose(ca.velocity_mps, [4.0, 0.0])
    turn = PlatformTrajectory(TrajectoryKind.COORDINATED_TURN).advance(initial, 1.0)
    np.testing.assert_allclose(turn.position_m, [4 / np.pi, 4 / np.pi], atol=1e-12)
    np.testing.assert_allclose(turn.velocity_mps, [0.0, 2.0], atol=1e-12)
    waypoints = PlatformTrajectory(
        TrajectoryKind.WAYPOINT,
        (Waypoint(0.0, np.array([0.0, 0.0])), Waypoint(4.0, np.array([8.0, 4.0]))),
    )
    at_two = waypoints.advance(initial, 2.0)
    np.testing.assert_allclose(at_two.position_m, [4.0, 2.0])
    np.testing.assert_allclose(at_two.velocity_mps, [2.0, 1.0])


def test_moving_sensor_changes_range_rate_and_array_measurement() -> None:
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    platform = PlatformState(
        np.array([0.0, 0.0]),
        np.array([10_000.0, 0.0]),
        np.zeros(2),
        0.0,
        0.0,
        0.1,
    )
    pose = platform.sensor_pose()
    target = Target("truth-only", 300.0, 0.0)
    geometry = relative_geometry(pose, target)
    assert geometry.radial_velocity_mps == pytest.approx(-10_000.0)
    result = process_frame(
        (target,),
        0.1,
        create_sensor(config, pose, random_seed=3),
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        32,
    )
    near = [d for d in result.detections if abs(d.range_m - 300) < 16]
    assert near
    velocity_bin = abs(
        result.doppler_product.radial_velocity_mps[1]
        - result.doppler_product.radial_velocity_mps[0]
    )
    assert abs(near[0].radial_velocity_mps + 10_000) <= velocity_bin
    assert near[0].sensor_pose == pose


def test_local_bearing_and_platform_pose_transform_to_world_for_tracking() -> None:
    platform = PlatformState(
        np.array([100.0, 200.0]), np.zeros(2), np.zeros(2), np.pi / 2, 0.0, 1.0
    )
    pose = platform.sensor_pose()
    detection = Detection(1.0, 50.0, 0.0, None, 2.0, 20.0, 0.9, 5, sensor_pose=pose)
    np.testing.assert_allclose(detection_to_cartesian(detection), [100.0, 250.0])
    track = MultiTargetTracker().update([detection], timestamp_s=1.0)[0]
    np.testing.assert_allclose(track.state[:2], [100.0, 250.0])


def test_world_platform_reset_and_versioned_scenario_migration(tmp_path) -> None:
    initial = PlatformState(
        np.array([10.0, 20.0]), np.array([3.0, 0.0]), np.zeros(2), 0.0, 0.0, 0.0
    )
    world = World(
        (Target("a", 100.0, 0.0),),
        platform_state=initial,
        platform_trajectory=PlatformTrajectory(TrajectoryKind.CONSTANT_VELOCITY),
    )
    world.advance(2.0)
    assert world.sensor_pose.x_m == pytest.approx(16.0)
    world.reset()
    assert world.sensor_pose.x_m == pytest.approx(10.0)
    path = tmp_path / "platform.json"
    save_scenario_json(path, world, SimulationConfig(), SensorConfig())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    restored, _, _ = load_scenario_json(path)
    assert restored.platform_trajectory.kind is TrajectoryKind.CONSTANT_VELOCITY
    np.testing.assert_array_equal(restored.platform_state.position_m, [10.0, 20.0])
    payload.pop("schema_version")
    payload.pop("platform")
    path.write_text(json.dumps(payload), encoding="utf-8")
    legacy, _, _ = load_scenario_json(path)
    assert legacy.sensor_pose.x_m == pytest.approx(10.0)
    assert legacy.platform_trajectory.kind is TrajectoryKind.STATIONARY


def test_invalid_world_step_does_not_advance_targets_or_platform() -> None:
    world = World((Target("a", 100.0, 0.0, vx_mps=2.0),))
    with pytest.raises(ValueError, match="finite and positive"):
        world.advance(float("nan"))
    assert world.get_target("a").x_m == 100.0
    assert world.time_s == 0.0


def test_gui_frame_uses_advanced_platform_pose_and_presents_trajectory(
    tmp_path,
) -> None:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from echorin.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    platform = PlatformState(
        np.array([-10.0, 0.0]), np.array([10.0, 0.0]), np.zeros(2), 0.0, 0.0, 0.0
    )
    world = World(
        (Target("only-truth", 300.0, 0.0),),
        platform_state=platform,
        platform_trajectory=PlatformTrajectory(TrajectoryKind.CONSTANT_VELOCITY),
    )
    window = MainWindow(
        world=world,
        simulation_config=SimulationConfig(dt_s=1.0),
        sensor_config=SensorConfig.radar(max_range_m=600.0),
        settings=QSettings(str(tmp_path / "platform.ini"), QSettings.Format.IniFormat),
    )
    window.step_once()
    assert window.world.sensor_pose.x_m == pytest.approx(0.0)
    assert window.last_detections
    near = min(window.last_detections, key=lambda d: abs(d.range_m - 300))
    assert near.sensor_pose is not None
    assert near.sensor_pose.x_m == pytest.approx(0.0)
    assert near.world_position_m[0] == pytest.approx(300, abs=16)
    assert window.ppi_view.sensor_item.data[0][0] == pytest.approx(0.0)
    assert window.platform_dock.windowTitle() == "Sensor Platform"
    window.close()
    app.processEvents()


def test_turned_array_local_bearing_is_transformed_into_world() -> None:
    config = SensorConfig.radar(
        max_range_m=600.0, noise_model=NoiseConfig(standard_deviation=0.001)
    )
    platform = PlatformState(
        np.array([100.0, 200.0]), np.zeros(2), np.zeros(2), np.pi / 2, 0.0, 0.1
    )
    sensor = create_sensor(config, platform.sensor_pose(), random_seed=12)
    result = process_frame(
        (Target("private", 100.0, 500.0),),
        0.1,
        sensor,
        SignalProcessor(config),
        CaCfarDetector(CfarConfig()),
        MultiTargetTracker(),
        16,
    )
    near = min(result.detections, key=lambda d: abs(d.range_m - 300))
    assert abs(near.bearing_rad) <= np.deg2rad(1)
    np.testing.assert_allclose(near.world_position_m, [100, 500], atol=16)


def test_waypoint_mount_and_array_config_survive_scenario_round_trip(tmp_path) -> None:
    mount = MountTransform(np.array([3.0, -2.0]), 0.2)
    trajectory = PlatformTrajectory(
        TrajectoryKind.WAYPOINT,
        (Waypoint(0.0, np.array([0.0, 0.0])), Waypoint(2.0, np.array([10.0, 5.0]))),
    )
    world = World(
        platform_state=PlatformState(
            np.zeros(2), np.zeros(2), np.zeros(2), 0.1, 0.0, 0.0
        ),
        platform_trajectory=trajectory,
        sensor_mount=mount,
        array_config=ArrayConfig(element_count=6, spacing_wavelengths=0.4),
    )
    path = tmp_path / "waypoint.json"
    save_scenario_json(path, world, SimulationConfig(), SensorConfig())
    restored, _, _ = load_scenario_json(path)
    assert restored.array_config.element_count == 6
    assert restored.array_config.spacing_wavelengths == 0.4
    np.testing.assert_allclose(restored.sensor_mount.position_m, [3, -2])
    assert restored.platform_trajectory.kind is TrajectoryKind.WAYPOINT
    restored.advance(1.0)
    np.testing.assert_allclose(restored.platform_state.position_m, [5.0, 2.5])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_scenario_json(path)


def test_array_config_rejects_grating_lobe_spacing() -> None:
    with pytest.raises(ValueError, match="half wavelength"):
        ArrayConfig(element_count=8, spacing_wavelengths=0.75)


def test_recorded_empty_frame_keeps_platform_pose(tmp_path) -> None:
    from echorin.models.frame import FrameResult
    from echorin.recording import FrameRecorder

    platform = PlatformState(
        np.array([12.0, -7.0]), np.array([2.0, 0.0]), np.zeros(2), 0.5, 0.0, 1.5
    )
    recorder = FrameRecorder()
    recorder.capture(
        FrameResult(
            timestamp_s=1.5,
            platform_states=[platform],
            receiver_pose=platform.sensor_pose(MountTransform(np.array([2.0, 0.0]))),
        )
    )
    path = tmp_path / "frames.json"
    recorder.save_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    restored = FrameRecorder.load_json(path)
    assert restored.records[0]["platform_states"][0]["position_m"] == [12, -7]
    assert restored.records[0]["platform_states"][0]["timestamp_s"] == 1.5
    assert restored.records[0]["receiver_pose"]["timestamp_s"] == 1.5
    path.write_text(json.dumps(recorder.records), encoding="utf-8")
    assert FrameRecorder.load_json(path).records == recorder.records


def test_platform_controls_apply_motion_and_mount(tmp_path) -> None:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from echorin.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        world=World((Target("a", 300.0, 0.0),)),
        settings=QSettings(str(tmp_path / "controls.ini"), QSettings.Format.IniFormat),
    )
    editor = window.platform_controls
    editor.fields["x"].setValue(10.0)
    editor.fields["vx"].setValue(2.0)
    editor.fields["mount_x"].setValue(3.0)
    editor.element_spin.setValue(6)
    editor.trajectory_combo.setCurrentIndex(
        editor.trajectory_combo.findData(TrajectoryKind.CONSTANT_VELOCITY)
    )
    editor.apply_button.click()
    assert window.world.array_config.element_count == 6
    assert window.sensor.array_geometry.element_count == 6
    assert window.world.sensor_pose.x_m == pytest.approx(13.0)
    window.step_once()
    assert window.world.sensor_pose.x_m == pytest.approx(13.1)
    assert len(window.world.platform_history) == 2
    window.reset()
    assert window.world.sensor_pose.x_m == pytest.approx(13.0)
    editor.trajectory_combo.setCurrentIndex(
        editor.trajectory_combo.findData(TrajectoryKind.WAYPOINT)
    )
    editor.waypoints_edit.setPlainText("invalid")
    editor.apply_button.click()
    assert "waypoints" in editor.error_label.text()
    window.close()
    app.processEvents()
