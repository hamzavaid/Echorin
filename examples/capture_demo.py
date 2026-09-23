"""Capture reproducible GUI screenshot and animated demonstration assets."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL.ImageQt import fromqimage
from PySide6.QtCore import QSettings
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
    with TemporaryDirectory(prefix="echorin-capture-") as settings_directory:
        settings = QSettings(
            str(Path(settings_directory) / "workspace.ini"),
            QSettings.Format.IniFormat,
        )
        window = MainWindow(settings=settings)
        window.resize(1_500, 900)
        window.show()
        app.processEvents()
        frames = []
        for frame_index in range(12):
            window.step_once()
            if frame_index == 7 and window.last_tracks:
                window.inspector.select_track(window.last_tracks[0].track_id)
            if frame_index == 8 and window.last_doppler_product is not None:
                product = window.last_doppler_product
                velocity_bin, range_bin = np.unravel_index(
                    np.argmax(product.magnitude), product.magnitude.shape
                )
                window.range_doppler_view.select_cell_at(
                    float(product.ranges_m[range_bin]),
                    float(product.radial_velocity_mps[velocity_bin]),
                )
            app.processEvents()
            image = fromqimage(window.grab().toImage()).convert("RGB")
            frames.append(image)
            if frame_index == 7:
                image.save(output_directory / "echorin-main.png")
                window.range_doppler_dock.raise_()
                app.processEvents()
            if frame_index == 8:
                image.save(output_directory / "echorin-range-doppler.png")
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
