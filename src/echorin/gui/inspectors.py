"""Dockable detection and track inspector widget."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echorin.gui.inspection_data import InspectionDetails, InspectionModel
from echorin.models.detection import Detection
from echorin.models.track import Track


class MeasurementTrackInspector(QWidget):
    """Inspect selectable published measurements and persistent track states."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = InspectionModel()
        layout = QVBoxLayout(self)
        self.selector = QComboBox()
        self.selector.addItem("Select a detection or track", None)
        layout.addWidget(self.selector)
        self.title = QLabel("No sensor estimate selected")
        layout.addWidget(self.title)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(("Field", "Value"))
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.selector.currentIndexChanged.connect(self._selection_changed)

    def set_frame(
        self, detections: Sequence[Detection], tracks: Sequence[Track]
    ) -> None:
        """Refresh choices and retain track-ID selection across frames."""
        self.model.update(detections, tracks)
        with QSignalBlocker(self.selector):
            self.selector.clear()
            self.selector.addItem("Select a detection or track", None)
            for index, detection in enumerate(self.model.detections):
                self.selector.addItem(
                    f"Detection {index + 1}: {detection.range_m:.1f} m",
                    ("detection", index),
                )
            for track in self.model.tracks:
                self.selector.addItem(
                    f"Track {track.track_id}: {track.status.value}",
                    ("track", track.track_id),
                )
            key = (self.model.selected_kind, self.model.selected_id)
            matching = next(
                (i for i in range(self.selector.count())
                 if self.selector.itemData(i) == key), 0
            )
            self.selector.setCurrentIndex(matching)
        self._render(self.model.current)

    def clear(self) -> None:
        self.model.clear()
        self.set_frame((), ())

    def select_detection(self, index: int) -> None:
        self._render(self.model.select_detection(index))
        self._sync_selector()

    def select_track(self, track_id: int) -> None:
        self._render(self.model.select_track(track_id))
        self._sync_selector()

    def _sync_selector(self) -> None:
        key = (self.model.selected_kind, self.model.selected_id)
        for index in range(self.selector.count()):
            if self.selector.itemData(index) == key:
                with QSignalBlocker(self.selector):
                    self.selector.setCurrentIndex(index)
                return

    def _selection_changed(self, _index: int) -> None:
        choice = self.selector.currentData()
        if choice is None:
            self.model.selected_kind = None
            self.model.selected_id = None
            self.model.current = None
            self._render(None)
        elif choice[0] == "detection":
            self.select_detection(choice[1])
        else:
            self.select_track(choice[1])

    def _render(self, details: InspectionDetails | None) -> None:
        self.title.setText(
            details.title if details is not None else "No sensor estimate selected"
        )
        fields = details.fields if details is not None else {}
        self.table.setRowCount(len(fields))
        for row, (name, value) in enumerate(fields.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(value))

    def visible_fields(self) -> dict[str, str]:
        """Expose currently displayed fields for integration tests/accessibility."""
        return {
            self.table.item(row, 0).text(): self.table.item(row, 1).text()
            for row in range(self.table.rowCount())
        }
