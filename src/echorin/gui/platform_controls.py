"""Editable sensor platform and body-mount settings."""

from __future__ import annotations

from math import degrees, radians

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echorin.config import ArrayConfig
from echorin.models.platform import MountTransform, PlatformState
from echorin.simulation.trajectories import (
    PlatformTrajectory,
    TrajectoryKind,
    Waypoint,
)
from echorin.simulation.world import World


class PlatformControls(QGroupBox):
    """Configure a moving receiver and fixed array mount at scenario time zero."""

    platform_changed = Signal(object, object, object, object)
    trail_toggled = Signal(bool)
    field_of_view_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Sensor Platform", parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields: dict[str, QDoubleSpinBox] = {}
        for key, label in (
            ("x", "East (m)"),
            ("y", "North (m)"),
            ("vx", "East speed (m/s)"),
            ("vy", "North speed (m/s)"),
            ("ax", "East accel (m/s²)"),
            ("ay", "North accel (m/s²)"),
            ("heading", "Heading (deg)"),
            ("turn_rate", "Turn rate (deg/s)"),
            ("mount_x", "Array offset east (m)"),
            ("mount_y", "Array offset north (m)"),
            ("mount_heading", "Array mount angle (deg)"),
            ("array_orientation", "Array orientation (deg)"),
        ):
            field = QDoubleSpinBox()
            field.setRange(-1e7, 1e7)
            field.setDecimals(4)
            self.fields[key] = field
            form.addRow(label, field)
        self.trajectory_combo = QComboBox()
        for kind in TrajectoryKind:
            self.trajectory_combo.addItem(kind.value.replace("_", " ").title(), kind)
        form.addRow("Trajectory", self.trajectory_combo)
        self.element_spin = QSpinBox()
        self.element_spin.setRange(2, 32)
        self.element_spin.setValue(8)
        form.addRow("Array elements", self.element_spin)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.01, 0.5)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setValue(0.5)
        form.addRow("Spacing (wavelengths)", self.spacing_spin)
        layout.addLayout(form)
        layout.addWidget(
            QLabel("Waypoints: time (s), east (m), north (m); one per line")
        )
        self.waypoints_edit = QPlainTextEdit()
        self.waypoints_edit.setMaximumHeight(65)
        layout.addWidget(self.waypoints_edit)
        self.trail_checkbox = QCheckBox("Platform trajectory")
        self.trail_checkbox.setChecked(True)
        self.fov_checkbox = QCheckBox("Array view sector")
        self.fov_checkbox.setChecked(True)
        layout.addWidget(self.trail_checkbox)
        layout.addWidget(self.fov_checkbox)
        self.error_label = QLabel("")
        layout.addWidget(self.error_label)
        self.apply_button = QPushButton("Apply platform")
        layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self._apply)
        self.trail_checkbox.toggled.connect(self.trail_toggled)
        self.fov_checkbox.toggled.connect(self.field_of_view_toggled)

    def set_from_world(self, world: World) -> None:
        """Populate controls from the editable scenario baseline."""
        state, mount = world.platform_state, world.sensor_mount
        values = {
            "x": state.position_m[0],
            "y": state.position_m[1],
            "vx": state.velocity_mps[0],
            "vy": state.velocity_mps[1],
            "ax": state.acceleration_mps2[0],
            "ay": state.acceleration_mps2[1],
            "heading": degrees(state.heading_rad),
            "turn_rate": degrees(state.angular_velocity_rad_s),
            "mount_x": mount.position_m[0],
            "mount_y": mount.position_m[1],
            "mount_heading": degrees(mount.heading_rad),
            "array_orientation": degrees(world.array_config.orientation_rad),
        }
        for key, value in values.items():
            self.fields[key].setValue(float(value))
        index = self.trajectory_combo.findData(world.platform_trajectory.kind)
        self.trajectory_combo.setCurrentIndex(index)
        self.element_spin.setValue(world.array_config.element_count)
        self.spacing_spin.setValue(world.array_config.spacing_wavelengths)
        self.waypoints_edit.setPlainText(
            "\n".join(
                f"{point.timestamp_s:g}, {point.position_m[0]:g}, "
                f"{point.position_m[1]:g}"
                for point in world.platform_trajectory.waypoints
            )
        )

    def _apply(self) -> None:
        """Validate all fields before publishing a coherent scenario edit."""
        value = {key: field.value() for key, field in self.fields.items()}
        kind = TrajectoryKind(self.trajectory_combo.currentData())
        try:
            waypoints: tuple[Waypoint, ...] = ()
            if kind is TrajectoryKind.WAYPOINT:
                rows = [
                    line.strip()
                    for line in self.waypoints_edit.toPlainText().splitlines()
                    if line.strip()
                ]
                try:
                    parsed = [
                        tuple(float(part.strip()) for part in row.split(","))
                        for row in rows
                    ]
                except ValueError as error:
                    raise ValueError(
                        "waypoints require numeric time, east, north"
                    ) from error
                if any(len(row) != 3 for row in parsed):
                    raise ValueError("waypoints require time, east, north")
                waypoints = tuple(Waypoint(t, np.array([x, y])) for t, x, y in parsed)
            trajectory = PlatformTrajectory(kind, waypoints)
            state = PlatformState(
                np.array([value["x"], value["y"]]),
                np.array([value["vx"], value["vy"]]),
                np.array([value["ax"], value["ay"]]),
                radians(value["heading"]),
                radians(value["turn_rate"]),
                0.0,
            )
            mount = MountTransform(
                np.array([value["mount_x"], value["mount_y"]]),
                radians(value["mount_heading"]),
            )
            array = ArrayConfig(
                element_count=self.element_spin.value(),
                spacing_wavelengths=self.spacing_spin.value(),
                orientation_rad=radians(value["array_orientation"]),
            )
        except ValueError as error:
            self.error_label.setText(str(error))
            return
        self.error_label.setText("")
        self.platform_changed.emit(state, trajectory, mount, array)
