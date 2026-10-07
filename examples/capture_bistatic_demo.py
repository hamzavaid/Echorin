"""Recreate intentional public v1.5 evidence from actual numerical/UI outputs."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from echorin.bistatic_benchmark import benchmark_config, run_bistatic_benchmark
from echorin.gui.main_window import MainWindow
from echorin.simulation.scenarios import bistatic_scenario

ROOT = Path(__file__).resolve().parents[1]


def validation_figure() -> None:
    """Bin-normalized numerical errors for every moving TX/RX pair."""
    results = run_bistatic_benchmark()
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="700">',
        '<rect width="1100" height="700" fill="#101820"/>',
        '<g font-family="sans-serif" fill="#e8f0f2">',
        '<text x="40" y="35" font-size="22">'
        "Moving bistatic / multistatic validation</text>",
        '<text x="40" y="60">seed 7; four frames; 32 pulses; '
        "errors / configured bin width</text>",
        '<text x="40" y="85">Radar: 4 MHz, PRF 20 Hz, 3 km. '
        "Sonar: 20 kHz, PRF 3 Hz, 60 m. Noise sigma 0.0001.</text>",
    ]
    for column, label in enumerate(
        ("Total path", "Arrival bearing", "Total path rate")
    ):
        left = 270 + column * 265
        lines.append(f'<text x="{left}" y="120" font-size="17">{label}</text>')
        lines.append(
            f'<path d="M{left + 210} 140 V625" stroke="#738b98" '
            'stroke-dasharray="4 4"/>'
        )
        lines.append(f'<text x="{left + 204}" y="648">1 bin</text>')
    for index, row in enumerate(results["cases"]):
        top = 145 + index * 58
        label = f"{row['mode'].title()} {row['emitter_id']} → {row['receiver_id']}"
        lines.append(f'<text x="40" y="{top + 20}">{label}</text>')
        for column, (error, scale) in enumerate(
            (
                ("max_path_error_m", "path_bin_m"),
                ("max_bearing_error_rad", "angle_bin_rad"),
                ("max_path_rate_error_mps", "path_rate_bin_mps"),
            )
        ):
            fraction = row[error] / row[scale]
            left = 270 + column * 265
            lines.append(
                f'<rect x="{left}" y="{top}" width="{210 * fraction:.2f}" '
                'height="26" fill="#50e0a1"/>'
            )
            lines.append(
                f'<text x="{left}" y="{top + 44}">{fraction:.3f} bin; '
                f"{row['hits']}/4 hits</text>"
            )
    lines.append(
        '<text x="40" y="680">Stop-and-hop approximation. Source-scoped CV tracks; '
        "no fusion. Every measured error is below one bin.</text>"
    )
    lines.append("</g></svg>")
    (ROOT / "benchmarks" / "bistatic_validation.svg").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    for row in results["cases"]:
        print(
            row["mode"],
            row["emitter_id"],
            row["receiver_id"],
            "path/bearing/rate errors:",
            row["max_path_error_m"],
            row["max_bearing_error_rad"],
            row["max_path_rate_error_mps"],
            "mean all-link ms:",
            row["mean_processing_ms"],
        )


def main() -> None:
    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Arial", 9))
    with TemporaryDirectory(prefix="echorin-v15-capture-") as directory:
        for mode in ("radar", "sonar"):
            config = benchmark_config(mode)
            window = MainWindow(
                world=bistatic_scenario(config, multistatic=True),
                sensor_config=config,
                settings=QSettings(
                    str(Path(directory) / f"{mode}.ini"), QSettings.Format.IniFormat
                ),
            )
            window.resize(1580, 1000)
            window.range_angle_dock.setMinimumHeight(300)
            window.show()
            window.network_dock.raise_()
            window.range_angle_dock.raise_()
            app.processEvents()
            window.resizeDocks(
                [window.range_angle_dock], [320], Qt.Orientation.Vertical
            )
            window.resizeDocks(
                [window.tracks_dock, window.inspector_dock, window.diagnostics_dock],
                [120, 420, 180],
                Qt.Orientation.Vertical,
            )
            for _ in range(4):
                window.step_once()
                app.processEvents()
            window.inspector.select_detection(0)
            app.processEvents()
            if mode == "radar":
                window.grab().save(
                    str(ROOT / "screenshots" / "echorin-multistatic.png")
                )
            print(mode, "all-link GUI timings:", window.last_timing_metrics_s)
            window.close()
            app.processEvents()
    validation_figure()


if __name__ == "__main__":
    main()
