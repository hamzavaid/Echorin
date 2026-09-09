"""Milestone 2 offscreen integration tests for the desktop interface."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from echorin.config import SimulationConfig
from echorin.gui.controls import SimulationControls, TargetEditor
from echorin.gui.main_window import MainWindow
from echorin.gui.ppi_view import PpiView
from echorin.simulation.scenarios import single_stationary_target
from echorin.simulation.target import Target


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_ppi_draws_configured_range_rings_and_targets(app: QApplication) -> None:
    view = PpiView(max_range_m=5_000.0, ring_count=4)
    targets = (
        Target("one", 1_000.0, 500.0),
        Target("two", -750.0, -250.0),
    )

    view.set_targets(targets)

    assert len(view.range_rings) == 4
    assert view.max_range_m == 5_000.0
    x_data, y_data = view.target_item.getData()
    assert list(x_data) == [1_000.0, -750.0]
    assert list(y_data) == [500.0, -250.0]
    view.close()


def test_simulation_controls_emit_each_command(app: QApplication) -> None:
    controls = SimulationControls()
    calls: list[str] = []
    truth_states: list[bool] = []
    controls.start_requested.connect(lambda: calls.append("start"))
    controls.pause_requested.connect(lambda: calls.append("pause"))
    controls.step_requested.connect(lambda: calls.append("step"))
    controls.reset_requested.connect(lambda: calls.append("reset"))
    controls.ground_truth_toggled.connect(truth_states.append)

    controls.start_button.click()
    controls.pause_button.click()
    controls.step_button.click()
    controls.reset_button.click()
    controls.ground_truth_checkbox.setChecked(False)

    assert calls == ["start", "pause", "step", "reset"]
    assert truth_states == [False]
    controls.close()


def test_target_editor_emits_valid_edits_additions_and_removals(
    app: QApplication,
) -> None:
    editor = TargetEditor()
    targets = [Target("editable", 10.0, 20.0, 1.0, 2.0)]
    edits: list[tuple[str, Target]] = []
    additions: list[Target] = []
    removals: list[str] = []
    editor.target_edited.connect(lambda old_id, target: edits.append((old_id, target)))
    editor.target_added.connect(additions.append)
    editor.target_removed.connect(removals.append)
    editor.set_targets(targets)

    editor.table.item(0, editor.X_COLUMN).setText("123.5")
    editor.add_button.click()
    editor.table.selectRow(0)
    editor.remove_button.click()

    assert edits[-1][0] == "editable"
    assert edits[-1][1].x_m == pytest.approx(123.5)
    assert additions[0].target_id.startswith("target-")
    assert removals == ["editable"]
    editor.close()


def test_main_window_steps_runs_pauses_edits_and_resets(app: QApplication) -> None:
    world = single_stationary_target(range_m=1_500.0)
    window = MainWindow(world=world, simulation_config=SimulationConfig(dt_s=0.2))

    window.controls.step_button.click()
    assert world.time_s == pytest.approx(0.2)

    window.controls.start_button.click()
    assert window.timer.isActive()
    window.controls.pause_button.click()
    assert not window.timer.isActive()

    window.controls.ground_truth_checkbox.setChecked(False)
    assert not window.ppi_view.target_item.isVisible()
    window.controls.ground_truth_checkbox.setChecked(True)
    assert window.ppi_view.target_item.isVisible()

    window.target_editor.table.item(0, window.target_editor.X_COLUMN).setText("1800")
    assert world.get_target("target-1").x_m == pytest.approx(1_800.0)
    window.controls.step_button.click()
    window.controls.reset_button.click()
    assert world.time_s == 0.0
    assert world.get_target("target-1").x_m == pytest.approx(1_800.0)
    window.close()


def test_numerical_layers_do_not_import_gui_dependencies() -> None:
    source_root = Path(__file__).parents[1] / "src" / "echorin"
    core_directories = ("simulation", "signals", "dsp", "tracking", "sensors", "models")

    for directory in core_directories:
        for path in (source_root / directory).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "PySide6" not in text
            assert "pyqtgraph" not in text
