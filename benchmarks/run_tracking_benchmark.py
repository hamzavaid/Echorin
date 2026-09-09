"""Generate the public deterministic tracking benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from echorin.benchmark import TrackingBenchmarkResult, run_tracking_benchmark


def _polyline(points: np.ndarray, width: int, height: int) -> str:
    x = points[:, 0]
    y = points[:, 1]
    x_scaled = 55.0 + (x - x.min()) / max(float(np.ptp(x)), 1.0) * (width - 90.0)
    y_scaled = 30.0 + (y.max() - y) / max(float(np.ptp(y)), 1.0) * (height - 75.0)
    return " ".join(
        f"{px:.1f},{py:.1f}" for px, py in zip(x_scaled, y_scaled, strict=True)
    )


def render_svg(result: TrackingBenchmarkResult) -> str:
    """Render dependency-free trajectory comparison SVG."""
    width, height = 900, 480
    truth = _polyline(result.truth_positions_m, width, height)
    measured = _polyline(result.measured_positions_m, width, height)
    filtered = _polyline(result.filtered_positions_m, width, height)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#08151d"/>\n'
        '<text x="35" y="25" fill="white" font-family="sans-serif" '
        'font-size="18">Echorin Tracking Benchmark</text>\n'
        f'<polyline points="{measured}" fill="none" stroke="#d26a6a" '
        'stroke-width="1" opacity="0.45"/>\n'
        f'<polyline points="{truth}" fill="none" stroke="#f4c95d" '
        'stroke-width="3"/>\n'
        f'<polyline points="{filtered}" fill="none" stroke="#38d996" '
        'stroke-width="2"/>\n'
        '<text x="35" y="455" fill="#d26a6a" font-family="sans-serif">'
        "Noisy measurements</text>\n"
        '<text x="225" y="455" fill="#f4c95d" font-family="sans-serif">'
        "Ground truth</text>\n"
        '<text x="350" y="455" fill="#38d996" font-family="sans-serif">'
        "Filtered track</text>\n"
        '<text x="535" y="455" fill="white" font-family="sans-serif">'
        f"RMSE: {result.raw_position_rmse_m:.2f} m → "
        f"{result.filtered_position_rmse_m:.2f} m</text>\n</svg>"
    )


def main() -> None:
    output_directory = Path(__file__).resolve().parent
    result = run_tracking_benchmark(seed=44, sample_count=200)
    (output_directory / "tracking_results.json").write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    (output_directory / "tracking_benchmark.svg").write_text(
        render_svg(result), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
