"""Validated environmental controls with no numerical implementation in Qt."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import degrees, radians

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from echorin.config import SensorConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.presets import environment_preset
from echorin.environment.serialization import environment_from_dict
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.scan import ScanConfig


class EnvironmentControls(QScrollArea):
    """Common controls plus JSON lists for arbitrary tones, fields and paths."""

    environment_changed = Signal(object, object)
    beam_toggled = Signal(bool)

    def __init__(self, sensor_config: SensorConfig | None = None) -> None:
        super().__init__()
        self.sensor_config = sensor_config or SensorConfig()
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        form = QFormLayout()
        layout.addLayout(form)
        self.preset_combo = self._combo(
            (
                "Custom",
                "Baseline",
                "Clutter / Reverberation",
                "Multipath",
                "Interference",
            )
        )
        form.addRow("Effects template (edit before Apply)", self.preset_combo)
        self.noise_combo = self._combo(("awgn", "colored", "impulsive", "correlated"))
        self.beam_combo = self._combo(("isotropic", "sector", "gaussian", "sinc"))
        self.scan_combo = self._combo(("fixed", "rotation", "sector", "step"))
        form.addRow("Receiver noise", self.noise_combo)
        self.fields: dict[str, QDoubleSpinBox] = {}
        for key, label, low, high, decimals in (
            ("correlation", "AR(1) correlation", -0.999, 0.999, 3),
            ("event_probability", "Impulse probability / sample", 0, 1, 5),
            ("impulse_amplitude", "Impulse amplitude", 0, 1e4, 3),
            ("array_correlation", "Receiver correlation", 0, 1, 3),
            ("width", "Full beamwidth (deg)", 0.01, 360, 2),
            ("sidelobe", "Sector sidelobe amplitude", 0, 1, 3),
            ("aperture", "Sinc aperture (wavelengths)", 0.01, 1000, 2),
            ("boresight", "Fixed/rotation origin (deg)", -360, 360, 2),
            ("rate", "Scan rate (deg/s)", -3600, 3600, 2),
            ("start", "Sector start (deg)", -180, 180, 2),
            ("stop", "Sector stop (deg)", -180, 180, 2),
            ("dwell", "Step dwell (s)", 0.001, 10000, 3),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            self.fields[key] = spin
            if key == "width":
                form.addRow("Beam pattern", self.beam_combo)
            if key == "boresight":
                form.addRow("Scan mode", self.scan_combo)
            form.addRow(label, spin)
        self.ping_pong = QCheckBox("Ping-pong sector (unchecked: reset sweep)")
        self.steps_edit = QLineEdit()
        form.addRow(self.ping_pong)
        form.addRow("Step angles (deg, comma-separated)", self.steps_edit)
        self.beam_checkbox = QCheckBox("Show beam / boresight on PPI")
        self.beam_checkbox.setChecked(True)
        layout.addWidget(self.beam_checkbox)
        help_label = QLabel(
            "Effects JSON: interference, clutter, multipath lists; "
            "attenuation_exponent, absorption_db_per_m. "
            "JSON angles are radians, distances metres. See docs/scenarios.md. "
            "Noise sigma is in Sensor controls."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.effects_edit = QPlainTextEdit()
        self.effects_edit.setMinimumHeight(180)
        layout.addWidget(self.effects_edit)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.apply_button = QPushButton("Apply environment / beam")
        layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self._apply)
        self.preset_combo.currentTextChanged.connect(self._preset)
        self.beam_checkbox.toggled.connect(self.beam_toggled)
        self.set_configs(EnvironmentConfig(), BeamConfig())

    def _preset(self, name: str) -> None:
        if name != "Custom":
            data = asdict(environment_preset(name, self.sensor_config))
            data.pop("receiver_noise")
            self.effects_edit.setPlainText(json.dumps(data, indent=2))

    @staticmethod
    def _combo(items: tuple[str, ...]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    def set_configs(self, environment: EnvironmentConfig, beam: BeamConfig) -> None:
        """Show scenario settings without applying or mutating the sensor."""
        self.noise_combo.setCurrentText(environment.receiver_noise.kind)
        self.beam_combo.setCurrentText(beam.kind)
        self.scan_combo.setCurrentText(beam.scan.kind)
        noise, scan = environment.receiver_noise, beam.scan
        values = {
            "correlation": noise.correlation,
            "event_probability": noise.event_probability,
            "impulse_amplitude": noise.impulse_amplitude,
            "array_correlation": noise.array_correlation,
            "width": degrees(beam.width_rad),
            "sidelobe": beam.sidelobe_gain,
            "aperture": beam.aperture_wavelengths,
            "boresight": degrees(scan.boresight_rad),
            "rate": degrees(scan.rate_rad_s),
            "start": degrees(scan.start_rad),
            "stop": degrees(scan.stop_rad),
            "dwell": scan.dwell_s,
        }
        for key, value in values.items():
            self.fields[key].setValue(value)
        self.ping_pong.setChecked(scan.ping_pong)
        self.steps_edit.setText(
            ", ".join(f"{degrees(a):.8g}" for a in scan.step_angles_rad)
        )
        effects = asdict(environment)
        effects.pop("receiver_noise")
        self.effects_edit.setPlainText(json.dumps(effects, indent=2))
        self.error_label.clear()

    def _apply(self) -> None:
        try:
            v = {key: field.value() for key, field in self.fields.items()}
            effects = json.loads(self.effects_edit.toPlainText())
            effects["receiver_noise"] = asdict(
                ReceiverNoiseConfig(
                    self.noise_combo.currentText(),
                    v["correlation"],
                    v["event_probability"],
                    v["impulse_amplitude"],
                    v["array_correlation"],
                )
            )
            environment = environment_from_dict(effects)
            scan = ScanConfig(
                kind=self.scan_combo.currentText(),
                boresight_rad=radians(v["boresight"]),
                rate_rad_s=radians(v["rate"]),
                start_rad=radians(v["start"]),
                stop_rad=radians(v["stop"]),
                ping_pong=self.ping_pong.isChecked(),
                dwell_s=v["dwell"],
                step_angles_rad=tuple(
                    radians(float(a.strip())) for a in self.steps_edit.text().split(",")
                ),
            )
            beam = BeamConfig(
                self.beam_combo.currentText(),
                radians(v["width"]),
                v["sidelobe"],
                v["aperture"],
                scan,
            )
        except (ValueError, TypeError, KeyError) as error:
            self.error_label.setText(str(error))
            return
        self.error_label.clear()
        self.environment_changed.emit(environment, beam)
