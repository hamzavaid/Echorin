"""Top-level Echorin desktop application window."""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from echorin.config import SensorConfig, SensorMode, SimulationConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig, CfarResult
from echorin.dsp.doppler import (
    DopplerProduct,
    doppler_spectrum,
    enrich_detections_with_velocity,
)
from echorin.dsp.range_processing import RangeProfile, SignalProcessor
from echorin.gui.controls import SimulationControls, TargetEditor, TrackTable
from echorin.gui.ppi_view import PpiView
from echorin.gui.signal_plots import SignalPlots
from echorin.models.detection import Detection
from echorin.models.track import Track
from echorin.sensors.base import SensorFrame
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.factory import create_sensor
from echorin.simulation.scenarios import crossing_targets
from echorin.simulation.target import Target
from echorin.simulation.world import World
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig


class MainWindow(QMainWindow):
    """Coordinate UI commands and world updates without numerical coupling."""

    def __init__(
        self,
        world: World | None = None,
        simulation_config: SimulationConfig | None = None,
        sensor_config: SensorConfig | None = None,
    ) -> None:
        super().__init__()
        self.world = world or crossing_targets()
        self.simulation_config = simulation_config or SimulationConfig()
        self.sensor_config = sensor_config or SensorConfig()
        self.sensor: SyntheticMonostaticSensor = create_sensor(
            self.sensor_config,
            self.world.sensor_pose,
            random_seed=self.simulation_config.random_seed,
        )
        self.signal_processor = SignalProcessor(self.sensor_config)
        self.cfar_detector = CaCfarDetector(
            CfarConfig(
                training_cells=16,
                guard_cells=4,
                false_alarm_probability=1e-3,
                minimum_separation_bins=max(1, self.sensor_config.pulse_samples // 8),
            )
        )
        self.last_sensor_frame: SensorFrame | None = None
        self.last_range_profile: RangeProfile | None = None
        self.last_cfar_result: CfarResult | None = None
        self.tracker = MultiTargetTracker(
            TrackerConfig(
                measurement_std_m=max(
                    5.0,
                    self.sensor_config.propagation_speed_mps
                    / (2.0 * self.sensor_config.sample_rate_hz),
                )
            )
        )
        self.last_detections: tuple[Detection, ...] = ()
        self.last_tracks: tuple[Track, ...] = ()
        self.doppler_pulse_count = 32
        self.last_doppler_product: DopplerProduct | None = None
        self.setWindowTitle("ECHORIN - Radar Signal Processing Simulator")
        self.resize(1_200, 800)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        splitter = QSplitter()
        self.ppi_view = PpiView(max_range_m=self.sensor_config.max_range_m)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        self.controls = SimulationControls()
        self.target_editor = TargetEditor()
        self.track_table = TrackTable()
        side_layout.addWidget(self.controls)
        side_layout.addWidget(self.target_editor)
        side_layout.addWidget(self.track_table)
        splitter.addWidget(self.ppi_view)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, stretch=3)
        self.signal_plots = SignalPlots()
        root_layout.addWidget(self.signal_plots, stretch=1)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setInterval(max(1, round(self.simulation_config.dt_s * 1_000.0)))
        self.timer.timeout.connect(self.step_once)
        self.controls.start_requested.connect(self.timer.start)
        self.controls.pause_requested.connect(self.timer.stop)
        self.controls.step_requested.connect(self.step_once)
        self.controls.reset_requested.connect(self.reset)
        self.controls.ground_truth_toggled.connect(
            self.ppi_view.set_ground_truth_visible
        )
        self.controls.detections_toggled.connect(self.ppi_view.set_detections_visible)
        self.controls.tracks_toggled.connect(self.ppi_view.set_tracks_visible)
        self.controls.trails_toggled.connect(self.ppi_view.set_trails_visible)
        self.controls.mode_changed.connect(self._set_sensor_mode)
        self.target_editor.target_added.connect(self._add_target)
        self.target_editor.target_edited.connect(self._edit_target)
        self.target_editor.target_removed.connect(self._remove_target)
        self._refresh_world_views()

    def step_once(self) -> None:
        """Advance one world frame and run the observation/DSP/detection chain."""
        self.world.advance(self.simulation_config.dt_s)
        directional_pulse_trains = self.sensor.acquire_directional_pulse_trains(
            self.world.targets,
            timestamp_s=self.world.time_s,
            pulse_count=self.doppler_pulse_count,
        )
        if directional_pulse_trains:
            combined_pulses = sum(
                (frame.received_pulses for frame in directional_pulse_trains),
                start=np.zeros(
                    (
                        self.doppler_pulse_count,
                        self.sensor_config.acquisition_samples,
                    ),
                    dtype=np.complex128,
                ),
            )
            transmitted = directional_pulse_trains[0].transmitted_signal
        else:
            pulse_train = self.sensor.acquire_pulse_train(
                (),
                timestamp_s=self.world.time_s,
                pulse_count=self.doppler_pulse_count,
            )
            combined_pulses = pulse_train.received_pulses
            transmitted = pulse_train.transmitted_signal
        self.last_sensor_frame = SensorFrame(
            self.world.time_s,
            transmitted,
            combined_pulses[0],
        )
        self.last_range_profile = self.signal_processor.range_profile(
            self.last_sensor_frame.received_signal,
            self.last_sensor_frame.transmitted_signal,
        )
        # A single omnidirectional channel cannot infer bearing; NaN records that
        # limitation rather than leaking the target's true angle.
        self.last_cfar_result = self.cfar_detector.detect(
            self.last_range_profile,
            timestamp_s=self.world.time_s,
            bearing_rad=float("nan"),
        )
        detections: list[Detection] = []
        for directional_frame in directional_pulse_trains:
            range_responses = self.signal_processor.pulse_matrix_range_responses(
                directional_frame.received_pulses,
                directional_frame.transmitted_signal,
            )
            directional_profile = RangeProfile(
                self.signal_processor.range_axis(range_responses.shape[1]),
                range_responses[0],
            )
            directional_result = self.cfar_detector.detect(
                directional_profile,
                timestamp_s=self.world.time_s,
                bearing_rad=directional_frame.bearing_rad,
            )
            directional_doppler = doppler_spectrum(range_responses, self.sensor_config)
            detections.extend(
                enrich_detections_with_velocity(
                    directional_result.detections, directional_doppler
                )
            )
        combined_range_responses = self.signal_processor.pulse_matrix_range_responses(
            combined_pulses, transmitted
        )
        self.last_doppler_product = doppler_spectrum(
            combined_range_responses, self.sensor_config
        )
        self.last_detections = tuple(detections)
        self.last_tracks = self.tracker.update(
            self.last_detections, timestamp_s=self.world.time_s
        )
        self.signal_plots.set_range_product(
            self.last_range_profile,
            self.last_cfar_result.threshold,
            [detection.source_bin for detection in self.last_cfar_result.detections],
        )
        selected_range_bin = (
            max(
                self.last_cfar_result.detections,
                key=lambda detection: detection.amplitude,
            ).source_bin
            if self.last_cfar_result.detections
            else int(np.argmax(self.last_range_profile.magnitude))
        )
        self.signal_plots.set_doppler_product(
            self.last_doppler_product, selected_range_bin
        )
        self.ppi_view.set_detections(self.last_detections)
        self.ppi_view.set_tracks(self.last_tracks)
        self.track_table.set_tracks(self.last_tracks)
        self._refresh_world_views(update_editor=False)

    def reset(self) -> None:
        """Pause and restore the edited scenario baseline."""
        self.timer.stop()
        self.world.reset()
        self.sensor.reset()
        self.last_sensor_frame = None
        self.last_range_profile = None
        self.last_cfar_result = None
        self.last_doppler_product = None
        self.tracker.reset()
        self.last_detections = ()
        self.last_tracks = ()
        self.ppi_view.set_detections(())
        self.ppi_view.set_tracks(())
        self.track_table.set_tracks(())
        self._refresh_world_views()

    def _add_target(self, target: Target) -> None:
        self.world.add_target(target)
        self.world.checkpoint_reset_state()
        self._refresh_world_views()

    def _edit_target(self, old_id: str, target: Target) -> None:
        self.world.replace_target(old_id, target)
        self.world.checkpoint_reset_state()
        self._refresh_world_views()

    def _remove_target(self, target_id: str) -> None:
        self.world.remove_target(target_id)
        self.world.checkpoint_reset_state()
        self._refresh_world_views()

    def _set_sensor_mode(self, mode_text: str) -> None:
        """Rebuild mode-specific sensor/DSP state while preserving the world."""
        mode = SensorMode(mode_text.lower())
        if mode is self.sensor_config.mode:
            return
        self.timer.stop()
        self.sensor_config = (
            SensorConfig.radar() if mode is SensorMode.RADAR else SensorConfig.sonar()
        )
        self.sensor = create_sensor(
            self.sensor_config,
            self.world.sensor_pose,
            random_seed=self.simulation_config.random_seed,
        )
        self.signal_processor = SignalProcessor(self.sensor_config)
        self.cfar_detector = CaCfarDetector(
            CfarConfig(
                training_cells=16,
                guard_cells=4,
                false_alarm_probability=1e-3,
                minimum_separation_bins=max(1, self.sensor_config.pulse_samples // 8),
            )
        )
        self.tracker = MultiTargetTracker(
            TrackerConfig(
                measurement_std_m=max(
                    1.0,
                    self.sensor_config.propagation_speed_mps
                    / (2.0 * self.sensor_config.sample_rate_hz),
                )
            )
        )
        self.doppler_pulse_count = 32 if mode is SensorMode.RADAR else 16
        self.last_sensor_frame = None
        self.last_range_profile = None
        self.last_cfar_result = None
        self.last_doppler_product = None
        self.last_detections = ()
        self.last_tracks = ()
        self.ppi_view.set_max_range(self.sensor_config.max_range_m)
        self.ppi_view.set_detections(())
        self.ppi_view.set_tracks(())
        self.track_table.set_tracks(())

    def _refresh_world_views(self, update_editor: bool = True) -> None:
        self.ppi_view.set_targets(self.world.targets)
        if update_editor:
            self.target_editor.set_targets(self.world.targets)
        self.statusBar().showMessage(
            f"t={self.world.time_s:.2f}s | {len(self.world.targets)} targets"
        )


def run_application() -> int:
    """Create the Qt application and display Echorin's main window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
