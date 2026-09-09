"""Capture reproducible GUI screenshot and animated demonstration assets."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL.ImageQt import fromqimage
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from echorin.gui.main_window import MainWindow


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "screenshots"
    output_directory.mkdir(exist_ok=True)
    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Arial", 9))
    window = MainWindow()
    window.resize(1_200, 800)
    window.show()
    app.processEvents()
    frames = []
    for frame_index in range(12):
        window.step_once()
        app.processEvents()
        image = fromqimage(window.grab().toImage()).convert("RGB")
        frames.append(image)
        if frame_index == 7:
            image.save(output_directory / "echorin-main.png")
    frames[0].save(
        output_directory / "echorin-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    window.close()


if __name__ == "__main__":
    main()
