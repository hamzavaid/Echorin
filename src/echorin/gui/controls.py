"""Simulation controls and editable ground-truth target table."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echorin.models.track import Track
from echorin.simulation.target import Target


class SimulationControls(QGroupBox):
    """Start, pause, single-step, and reset command panel."""

    start_requested = Signal()
    pause_requested = Signal()
    step_requested = Signal()
    reset_requested = Signal()
    ground_truth_toggled = Signal(bool)
    detections_toggled = Signal(bool)
    tracks_toggled = Signal(bool)
    trails_toggled = Signal(bool)
    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Simulation", parent)
        layout = QHBoxLayout(self)
        self.start_button = QPushButton("Run")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        for button in (
            self.start_button,
            self.pause_button,
            self.step_button,
            self.reset_button,
        ):
            layout.addWidget(button)
        self.ground_truth_checkbox = QCheckBox("Ground truth")
        self.ground_truth_checkbox.setChecked(True)
        layout.addWidget(self.ground_truth_checkbox)
        self.detections_checkbox = QCheckBox("Detections")
        self.tracks_checkbox = QCheckBox("Tracks")
        self.trails_checkbox = QCheckBox("Trails")
        for checkbox in (
            self.detections_checkbox,
            self.tracks_checkbox,
            self.trails_checkbox,
        ):
            checkbox.setChecked(True)
            layout.addWidget(checkbox)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(("Radar", "Sonar"))
        layout.addWidget(self.mode_combo)
        self.start_button.clicked.connect(self.start_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        self.step_button.clicked.connect(self.step_requested)
        self.reset_button.clicked.connect(self.reset_requested)
        self.ground_truth_checkbox.toggled.connect(self.ground_truth_toggled)
        self.detections_checkbox.toggled.connect(self.detections_toggled)
        self.tracks_checkbox.toggled.connect(self.tracks_toggled)
        self.trails_checkbox.toggled.connect(self.trails_toggled)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)


class TrackTable(QGroupBox):
    """Read-only summary of sensor-derived active tracks."""

    HEADERS = ("ID", "Status", "X (m)", "Y (m)", "Vx (m/s)", "Vy (m/s)")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Active Tracks", parent)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def set_tracks(self, tracks: Iterable[Track]) -> None:
        """Replace table rows with current track estimates."""
        track_list = list(tracks)
        self.table.setRowCount(len(track_list))
        for row, track in enumerate(track_list):
            values = (
                track.track_id,
                track.status.value,
                f"{track.x_m:.1f}",
                f"{track.y_m:.1f}",
                f"{track.vx_mps:.1f}",
                f"{track.vy_mps:.1f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))


class TargetEditor(QGroupBox):
    """Table editor for simulation truth; it never exposes data to detectors."""

    target_added = Signal(object)
    target_edited = Signal(str, object)
    target_removed = Signal(str)

    ID_COLUMN = 0
    X_COLUMN = 1
    Y_COLUMN = 2
    VX_COLUMN = 3
    VY_COLUMN = 4
    REFLECTIVITY_COLUMN = 5
    HEADERS = ("ID", "X (m)", "Y (m)", "Vx (m/s)", "Vy (m/s)", "Reflectivity")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Scenario Targets", parent)
        self._populating = False
        self._known_ids: list[str] = []
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add Target")
        self.remove_button = QPushButton("Remove Selected")
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        self.table.cellChanged.connect(self._emit_row_edit)
        self.add_button.clicked.connect(self._request_add)
        self.remove_button.clicked.connect(self._request_remove)

    def set_targets(self, targets: Iterable[Target]) -> None:
        """Replace table content without emitting edit requests."""
        target_list = list(targets)
        self._populating = True
        try:
            self.table.setRowCount(len(target_list))
            self._known_ids = [target.target_id for target in target_list]
            for row, target in enumerate(target_list):
                values = (
                    target.target_id,
                    target.x_m,
                    target.y_m,
                    target.vx_mps,
                    target.vy_mps,
                    target.reflectivity,
                )
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
        finally:
            self._populating = False

    def _target_from_row(self, row: int) -> Target | None:
        try:
            values = [self.table.item(row, column).text() for column in range(6)]
            return Target(
                values[0],
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
            )
        except (AttributeError, ValueError):
            return None

    def _emit_row_edit(self, row: int, _column: int) -> None:
        if self._populating or row >= len(self._known_ids):
            return
        target = self._target_from_row(row)
        if target is not None:
            self.target_edited.emit(self._known_ids[row], target)

    def _request_add(self) -> None:
        suffix = 1
        known = set(self._known_ids)
        while f"target-{suffix}" in known:
            suffix += 1
        self.target_added.emit(Target(f"target-{suffix}", 1_000.0, 0.0))

    def _request_remove(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._known_ids):
            self.target_removed.emit(self._known_ids[row])
