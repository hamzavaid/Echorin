"""v1.1 engineering workspace acceptance tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QDockWidget

from echorin.gui.main_window import MainWindow


def test_dock_workspace_restores_layout_and_keeps_legacy_widgets(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    first = MainWindow(settings=settings)

    assert first.centralWidget() is first.ppi_view
    assert first.controls.parent() is not None
    assert first.target_editor.parent() is not None
    assert first.track_table.parent() is not None
    assert first.signal_plots.parent() is not None
    assert len(first.findChildren(QDockWidget)) >= 4
    assert all(dock.objectName() for dock in first.findChildren(QDockWidget))

    first.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, first.signal_dock)
    first.resize(1300, 850)
    first.close()
    settings.sync()

    second = MainWindow(settings=settings)
    assert second.dockWidgetArea(second.signal_dock) == (
        Qt.DockWidgetArea.RightDockWidgetArea
    )
    second.close()
    app.processEvents()
