"""Milestone 1 tests for kinematics, geometry, worlds, and scenarios."""

from __future__ import annotations

from math import atan2, hypot, pi

import pytest

from echorin.models import SensorPose
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.scenarios import crossing_targets, single_stationary_target
from echorin.simulation.target import Target
from echorin.simulation.world import World


def test_constant_velocity_motion_matches_analytical_solution() -> None:
    target = Target("alpha", x_m=100.0, y_m=-50.0, vx_mps=12.5, vy_mps=-3.0)

    for _ in range(40):
        target.advance(0.025)

    assert target.x_m == pytest.approx(112.5)
    assert target.y_m == pytest.approx(-53.0)
    assert target.vx_mps == 12.5
    assert target.vy_mps == -3.0


@pytest.mark.parametrize(
    ("target_xy", "expected_bearing"),
    (
        ((4.0, 3.0), atan2(3.0, 4.0)),
        ((-4.0, 3.0), atan2(3.0, -4.0)),
        ((-4.0, -3.0), atan2(-3.0, -4.0)),
        ((0.0, -5.0), -pi / 2.0),
    ),
)
def test_range_and_bearing_match_known_geometry(
    target_xy: tuple[float, float], expected_bearing: float
) -> None:
    sensor = SensorPose(x_m=10.0, y_m=-20.0, heading_rad=0.25)
    absolute_xy = (sensor.x_m + target_xy[0], sensor.y_m + target_xy[1])
    target = Target("geometry", x_m=absolute_xy[0], y_m=absolute_xy[1])

    geometry = relative_geometry(sensor, target)

    assert geometry.range_m == pytest.approx(hypot(*target_xy))
    assert geometry.bearing_rad == pytest.approx(expected_bearing)


def test_world_advances_all_targets_and_supports_crud() -> None:
    first = Target("one", 0.0, 0.0, 2.0, 0.0)
    second = Target("two", 5.0, 5.0, 0.0, -1.0)
    world = World(targets=[first])
    world.add_target(second)

    with pytest.raises(ValueError, match="one"):
        world.add_target(Target("one", 99.0, 99.0))

    world.advance(0.5)
    assert world.time_s == pytest.approx(0.5)
    assert world.get_target("one").x_m == pytest.approx(1.0)
    assert world.get_target("two").y_m == pytest.approx(4.5)
    assert world.remove_target("two").target_id == "two"
    with pytest.raises(KeyError):
        world.get_target("two")


def test_world_reset_restores_deep_initial_state() -> None:
    original = Target("reset-me", 20.0, -10.0, 4.0, 1.0)
    world = World(targets=[original])
    world.advance(3.0)
    world.get_target("reset-me").reflectivity = 8.0

    world.reset()

    restored = world.get_target("reset-me")
    assert world.time_s == 0.0
    assert (restored.x_m, restored.y_m) == (20.0, -10.0)
    assert restored.reflectivity == 1.0
    assert restored is not original


def test_scenario_presets_are_repeatable_and_independent() -> None:
    first = crossing_targets(seed=42)
    second = crossing_targets(seed=42)
    different = crossing_targets(seed=43)

    def state(world: World) -> list[list[float]]:
        return [target.state.tolist() for target in world.targets]

    assert state(first) == state(second)
    assert state(first) != state(different)

    first.advance(1.0)
    assert state(first) != state(second)


def test_stationary_scenario_reports_expected_console_geometry() -> None:
    world = single_stationary_target(range_m=1_500.0, bearing_rad=pi / 6.0)
    lines = world.console_frames(steps=2, dt_s=0.1)

    assert len(lines) == 2
    assert "t=0.100s" in lines[0]
    assert "target-1" in lines[0]
    assert "range=1500.000m" in lines[0]
    assert "bearing=30.000deg" in lines[0]


def test_target_validation_rejects_invalid_identity_and_reflectivity() -> None:
    with pytest.raises(ValueError, match="target_id"):
        Target("", 0.0, 0.0)
    with pytest.raises(ValueError, match="reflectivity"):
        Target("bad", 0.0, 0.0, reflectivity=-1.0)
