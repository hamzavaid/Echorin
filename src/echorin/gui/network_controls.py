"""Typed device configuration editor and source-scoped product selector."""

import json

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echorin.sensors.components import SensorPlatform
from echorin.sensors.serialization import platforms_from_data, platforms_to_data


class NetworkControls(QWidget):
    """Configure independent mounts/trajectories/beams without numerical GUI code."""

    configuration_requested = Signal(object)
    preset_requested = Signal(str)
    source_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        note = QLabel(
            "Time-division TX slots; independent source-scoped tracks.\n"
            "Half-path = (TX leg + RX leg)/2; no fusion.\n"
            "Edit mounts, motion, power, arrays and per-device beams below."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(("Monostatic", "Bistatic", "Multistatic"))
        self.preset_button = QPushButton("Load scenario")
        row.addWidget(self.preset_combo)
        row.addWidget(self.preset_button)
        layout.addLayout(row)
        self.source_combo = QComboBox()
        layout.addWidget(QLabel("Active TX → RX products / tracks"))
        layout.addWidget(self.source_combo)
        self.config_edit = QPlainTextEdit()
        self.config_edit.setMinimumHeight(150)
        layout.addWidget(self.config_edit, stretch=1)
        self.apply_button = QPushButton("Apply device configuration (restart)")
        layout.addWidget(self.apply_button)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.preset_button.clicked.connect(
            lambda: self.preset_requested.emit(self.preset_combo.currentText())
        )
        self.apply_button.clicked.connect(self._apply)
        self.source_combo.currentIndexChanged.connect(
            lambda: self.source_changed.emit(self.source_combo.currentData())
        )

    def set_configuration(
        self,
        platforms: tuple[SensorPlatform, ...],
        sources: list[tuple[str, str]],
        selected: tuple[str, str] | None,
    ) -> None:
        self.config_edit.setPlainText(
            json.dumps(platforms_to_data(platforms), indent=2)
        )
        self.error_label.clear()
        self.preset_combo.setCurrentText(
            "Multistatic"
            if len(sources) > 1
            else "Bistatic"
            if platforms
            else "Monostatic"
        )
        with QSignalBlocker(self.source_combo):
            self.source_combo.clear()
            for source in sources:
                self.source_combo.addItem(f"{source[0]} → {source[1]}", source)
            if not sources:
                self.source_combo.addItem("Legacy monostatic", None)
            index = sources.index(selected) if selected in sources else 0
            self.source_combo.setCurrentIndex(index)

    def _apply(self) -> None:
        try:
            data = json.loads(self.config_edit.toPlainText())
            if not isinstance(data, list):
                raise ValueError("device configuration must be a platform list")
            self.configuration_requested.emit(platforms_from_data(data))
        except (ValueError, TypeError, KeyError) as error:
            self.error_label.setText(str(error))
