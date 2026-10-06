"""Generate intentional public v1.4 evidence from real sensor/GUI outputs."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from echorin.application.frame_pipeline import process_frame
from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.presets import environment_preset
from echorin.gui.main_window import MainWindow
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.factory import create_sensor
from echorin.simulation.scenarios import single_stationary_target
from echorin.tracking.tracker import MultiTargetTracker

ROOT = Path(__file__).resolve().parents[1]


def comparison_figure() -> None:
    """SVG from numerical profiles/thresholds; independent curves, no fake data."""
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="740" '
        'viewBox="0 0 1100 740">',
        '<rect width="1100" height="740" fill="#101820"/>',
        '<g font-family="sans-serif" fill="#e8f0f2">',
        '<text x="65" y="30" font-size="20">Environment / CA-CFAR comparison</text>',
        '<text x="65" y="53">seed 7; noise sigma 0.001; '
        "field density 20 / max range</text>",
    ]
    for row, mode in enumerate(("radar", "sonar")):
        config = getattr(SensorConfig, mode)(
            max_range_m=3000 if mode == "radar" else 60, noise_model=NoiseConfig(0.001)
        )
        world = single_stationary_target(range_m=config.max_range_m / 3)
        products = []
        for env in (
            EnvironmentConfig(),
            environment_preset("Clutter / Reverberation", config),
        ):
            result = process_frame(
                world.targets,
                0.05,
                create_sensor(config, environment=env, random_seed=7),
                SignalProcessor(config),
                CaCfarDetector(CfarConfig(edge_mode="adaptive")),
                MultiTargetTracker(),
                16,
            )
            products.extend(
                (result.range_profile.magnitude, result.cfar_result.threshold)
            )
        top, height, left, width = 100 + row * 330, 220, 85, 940
        lines.append(
            f'<text x="85" y="{top - 16}" font-size="17">{mode.title()} — '
            "matched filter / adaptive CA-CFAR (dB re 1)</text>"
        )
        lines.append(
            f'<path d="M{left} {top} V{top + height} H{left + width}" '
            'fill="none" stroke="#a8c0cc"/>'
        )
        floor, ceiling = -70, 60
        for tick in (-60, -30, 0, 30, 60):
            y = top + height * (ceiling - tick) / (ceiling - floor)
            lines.append(f'<text x="45" y="{y + 4:.1f}">{tick}</text>')
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            x = left + width * fraction
            lines.append(
                f'<text x="{x - 10:.1f}" y="{top + height + 20}">'
                f"{config.max_range_m * fraction:g}</text>"
            )
        lines.append(f'<text x="500" y="{top + height + 43}">Range (m)</text>')
        for product, color, title in zip(
            products,
            ("#68baff", "#a4d3fa", "#ffb64c", "#ff7979"),
            ("Baseline", "Baseline threshold", "Field", "Field threshold"),
            strict=True,
        ):
            indices = np.arange(len(product))
            ranges = (
                indices * config.propagation_speed_mps / (2 * config.sample_rate_hz)
            )
            valid = np.isfinite(product) & (ranges <= config.max_range_m)
            level = np.clip(
                20 * np.log10(np.maximum(product[valid], 1e-12)), floor, ceiling
            )
            points = " ".join(
                f"{left + width * r / config.max_range_m:.2f},"
                f"{top + height * (ceiling - db) / (ceiling - floor):.2f}"
                for r, db in zip(ranges[valid], level, strict=True)
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="1"/>'
            )
            column = (
                "Baseline",
                "Baseline threshold",
                "Field",
                "Field threshold",
            ).index(title)
            lines.append(
                f'<text x="{100 + column * 220}" y="{top + height + 66}" '
                f'fill="{color}">{title}</text>'
            )
    lines.append("</g></svg>")
    (ROOT / "benchmarks" / "environment_comparison.svg").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    """Capture default-scale engineering UI and report real GUI frame timings."""
    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Arial", 9))
    with TemporaryDirectory(prefix="echorin-v14-capture-") as directory:
        window = MainWindow(
            settings=QSettings(
                str(Path(directory) / "capture.ini"), QSettings.Format.IniFormat
            )
        )
        environment = environment_preset("Multipath", window.sensor_config)
        from dataclasses import replace

        environment = replace(
            environment, receiver_noise=ReceiverNoiseConfig(kind="colored")
        )
        window._set_environment(environment, BeamConfig(kind="gaussian"))
        window.environment_controls.set_configs(environment, window.world.beam_config)
        window.resize(1550, 980)
        window.show()
        window.environment_dock.raise_()
        for _ in range(5):
            window.step_once()
            app.processEvents()
        window.grab().save(str(ROOT / "screenshots" / "echorin-environment.png"))
        print("Radar GUI timings:", window.last_timing_metrics_s)
        window._set_environment(EnvironmentConfig(), BeamConfig())
        window.controls.mode_combo.setCurrentText("Sonar")
        window.controls.preset_combo.setCurrentText("Single target")
        window._set_environment(
            environment_preset("Clutter / Reverberation", window.sensor_config),
            BeamConfig(),
        )
        window.step_once()
        app.processEvents()
        print("Sonar GUI timings:", window.last_timing_metrics_s)
        window.close()
        app.processEvents()
    comparison_figure()


if __name__ == "__main__":
    main()
