"""Top-level Echorin desktop application window."""

from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QWidget,
)

from echorin.application.frame_pipeline import FrameComputation, process_frame
from echorin.config import NoiseConfig, SensorConfig, SensorMode, SimulationConfig
from echorin.dsp.cfar import CaCfarDetector, CfarConfig, CfarResult
from echorin.dsp.doppler import DopplerProduct
from echorin.dsp.range_angle import RangeAngleProduct
from echorin.dsp.range_processing import RangeProfile, SignalProcessor
from echorin.gui.controls import SimulationControls, TargetEditor, TrackTable
from echorin.gui.heatmaps import RangeDopplerView
from echorin.gui.inspectors import MeasurementTrackInspector
from echorin.gui.ppi_view import PpiView
from echorin.gui.range_angle_view import RangeAngleView
from echorin.gui.signal_plots import SignalPlots
from echorin.gui.visualization_data import RangeDopplerCell
from echorin.models.detection import Detection
from echorin.models.frame import FrameResult
from echorin.models.track import Track
from echorin.sensors.base import SensorFrame
from echorin.sensors.echo import SyntheticMonostaticSensor
from echorin.sensors.factory import create_sensor
from echorin.signals.waveform import WaveformKind
from echorin.simulation.scenarios import crossing_targets, single_stationary_target
from echorin.simulation.target import Target
from echorin.simulation.world import World
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig


class MainWindow(QMainWindow):
    """Coordinate UI commands and world updates without numerical coupling."""

    frame_ready = Signal(int, object, float, float)

    def __init__(
        self,
        world: World | None = None,
        simulation_config: SimulationConfig | None = None,
        sensor_config: SensorConfig | None = None,
        settings: QSettings | None = None,
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
        self.last_range_angle_product: RangeAngleProduct | None = None
        self.last_timing_metrics_s: dict[str, float] = {}
        self.last_frame_result: FrameResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="echorin")
        self._frame_generation = 0
        self._inflight = False
        self.settings = settings or QSettings("Echorin", "Echorin")
        self.setWindowTitle(
            f"ECHORIN - {self.sensor_config.mode.value.title()} Engineering Workspace"
        )
        self.resize(1_400, 900)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )
        self.ppi_view = PpiView(max_range_m=self.sensor_config.max_range_m)
        self.setCentralWidget(self.ppi_view)
        self.controls = SimulationControls()
        self.controls.dt_spin.setValue(self.simulation_config.dt_s)
        self.controls.seed_spin.setValue(self.simulation_config.random_seed)
        self.controls.noise_spin.setValue(
            self.sensor_config.noise_model.standard_deviation
        )
        self.controls.mode_combo.setCurrentText(self.sensor_config.mode.value.title())
        self.target_editor = TargetEditor()
        self.track_table = TrackTable()
        self.signal_plots = SignalPlots()
        self.controls_dock = self._add_dock(
            "Sensor / Simulation", "controls", self.controls,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.scenario_dock = self._add_dock(
            "Scenario", "scenario", self.target_editor,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.tracks_dock = self._add_dock(
            "Tracks", "tracks", self.track_table,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.signal_dock = self._add_dock(
            "Signal Products", "signals", self.signal_plots,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.range_doppler_view = RangeDopplerView()
        self.range_doppler_dock = self._add_dock(
            "Range-Doppler", "range_doppler", self.range_doppler_view,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self.signal_dock, self.range_doppler_dock)
        self.range_angle_view = RangeAngleView()
        self.range_angle_dock = self._add_dock(
            "Range-Angle", "range_angle", self.range_angle_view,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self.signal_dock, self.range_angle_dock)
        self.signal_dock.raise_()
        self.inspector = MeasurementTrackInspector()
        self.inspector_dock = self._add_dock(
            "Measurement / Track Inspector", "inspector", self.inspector,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.diagnostics = QLabel("No frame processed yet")
        self.diagnostics.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.diagnostics.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.diagnostics.setMargin(10)
        self.diagnostics_dock = self._add_dock(
            "Performance / Diagnostics", "diagnostics", self.diagnostics,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.resizeDocks(
            [self.tracks_dock, self.inspector_dock, self.diagnostics_dock],
            [130, 250, 150], Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.signal_dock], [380], Qt.Orientation.Vertical
        )
        self.resizeDocks(
            [self.inspector_dock], [300], Qt.Orientation.Horizontal
        )
        self.resizeDocks(
            [self.controls_dock, self.scenario_dock], [390, 280],
            Qt.Orientation.Vertical,
        )
        saved_geometry = self.settings.value("workspace/geometry")
        saved_state = self.settings.value("workspace/state")
        if saved_geometry is not None:
            self.restoreGeometry(saved_geometry)
        if saved_state is not None:
            self.restoreState(saved_state)
        self.view_menu = self.menuBar().addMenu("View")
        self.ppi_focus = False
        self._pre_focus_state: bytes | None = None
        self.ppi_focus_action = self.view_menu.addAction("Focus PPI")
        self.ppi_focus_action.setCheckable(True)
        self.ppi_focus_action.toggled.connect(self.set_ppi_focus)
        self.view_menu.addSeparator()
        for dock in self.findChildren(QDockWidget):
            self.view_menu.addAction(dock.toggleViewAction())
        self.view_menu.addSeparator()
        self.theme_menu = self.view_menu.addMenu("Theme")
        for name in ("dark", "light"):
            action = self.theme_menu.addAction(name.title())
            action.triggered.connect(lambda checked=False, choice=name:
                                     self.apply_theme(choice))
        self.apply_theme(str(self.settings.value("workspace/theme", "dark")))

        self.timer = QTimer(self)
        self._update_timer_interval()
        self.timer.timeout.connect(self._request_live_step)
        self.frame_ready.connect(
            self._finish_async_step, Qt.ConnectionType.QueuedConnection
        )
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
        self.controls.labels_toggled.connect(self.ppi_view.set_labels_visible)
        self.controls.vectors_toggled.connect(self.ppi_view.set_vectors_visible)
        self.controls.uncertainty_toggled.connect(
            self.ppi_view.set_uncertainty_visible
        )
        self.controls.mode_changed.connect(self._set_sensor_mode)
        self.controls.dt_changed.connect(self._set_dt)
        self.controls.seed_changed.connect(self._set_seed)
        self.controls.noise_changed.connect(self._set_noise)
        self.controls.waveform_changed.connect(self._set_waveform)
        self.controls.preset_changed.connect(self._set_preset)
        self.target_editor.target_added.connect(self._add_target)
        self.target_editor.target_edited.connect(self._edit_target)
        self.target_editor.target_removed.connect(self._remove_target)
        self.ppi_view.track_selected.connect(self.inspector.select_track)
        self.ppi_view.detection_selected.connect(self.inspector.select_detection)
        self.track_table.track_selected.connect(self.inspector.select_track)
        self.range_doppler_view.cell_selected.connect(self._inspect_doppler_cell)
        self._refresh_world_views()

    def _inspect_doppler_cell(self, cell: RangeDopplerCell) -> None:
        if self.last_doppler_product is not None:
            self.signal_plots.set_doppler_product(
                self.last_doppler_product, cell.range_bin
            )

    def _add_dock(
        self, title: str, name: str, widget: QWidget, area: Qt.DockWidgetArea
    ) -> QDockWidget:
        """Create a persistent, movable engineering panel."""
        dock = QDockWidget(title, self)
        dock.setObjectName(f"echorin_{name}_dock")
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(area, dock)
        return dock

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist geometry and panel arrangement between sessions."""
        self.timer.stop()
        self._invalidate_pending()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.ppi_focus:
            self.set_ppi_focus(False)
        self.settings.setValue("workspace/geometry", self.saveGeometry())
        self.settings.setValue("workspace/state", self.saveState())
        super().closeEvent(event)

    def set_ppi_focus(self, enabled: bool) -> None:
        """Temporarily enlarge the central PPI without losing dock layout."""
        if enabled == self.ppi_focus:
            return
        self.ppi_focus = enabled
        if enabled:
            self._pre_focus_state = self.saveState()
            for dock in self.findChildren(QDockWidget):
                dock.hide()
        elif self._pre_focus_state is not None:
            self.restoreState(self._pre_focus_state)
            self._pre_focus_state = None
        self.ppi_focus_action.setChecked(enabled)

    def apply_theme(self, theme: str) -> None:
        """Apply and persist a readable dark or light engineering palette."""
        if theme not in {"dark", "light"}:
            raise ValueError("theme must be dark or light")
        self.theme = theme
        dark = theme == "dark"
        background = "#101820" if dark else "#ffffff"
        foreground = "#e8f0f2" if dark else "#192a35"
        panel = "#1a2a33" if dark else "#eaf0f4"
        control = "#16242d" if dark else "#dce7ed"
        border = "#39515e" if dark else "#9aadb8"
        selected = "#176b8f" if dark else "#b8dcec"
        self.setStyleSheet(
            "QMainWindow, QDockWidget, QMenuBar, QMenu, QLabel, QGroupBox, "
            f"QWidget {{ background: {panel}; color: {foreground}; }} "
            f"QTableWidget {{ background: {background}; color: {foreground}; "
            f"gridline-color: {border}; selection-background-color: {selected}; "
            f"selection-color: {foreground}; }} "
            f"QHeaderView::section, QTableCornerButton::section {{ "
            f"background: {control}; color: {foreground}; border: 0; "
            f"border-right: 1px solid {border}; border-bottom: 1px solid {border}; "
            "padding: 4px; } "
            f"QTabBar::tab {{ background: {control}; color: {foreground}; "
            f"border: 1px solid {border}; padding: 5px 10px; }} "
            f"QTabBar::tab:selected {{ background: {selected}; color: {foreground}; }} "
            f"QTabBar::tab:hover {{ background: {selected}; }} "
            f"QComboBox, QDoubleSpinBox, QSpinBox {{ background: {background}; "
            f"color: {foreground}; border: 1px solid {border}; }} "
            f"QComboBox QAbstractItemView {{ background: {background}; "
            f"color: {foreground}; selection-background-color: {selected}; "
            f"selection-color: {foreground}; }}"
        )
        for plot in (
            self.ppi_view.plot, self.signal_plots.range_plot,
            self.signal_plots.doppler_plot, self.range_doppler_view.plot,
            self.range_angle_view.plot,
        ):
            plot.setBackground(background)
            for axis_name in ("bottom", "left"):
                axis = plot.getAxis(axis_name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
        self.settings.setValue("workspace/theme", theme)

    def step_once(self) -> None:
        """Run one deterministic manual frame and present its products."""
        if self._inflight:
            return
        targets, timestamp_s, simulation_s, started = self._prepare_step()
        result = process_frame(
            targets, timestamp_s, self.sensor, self.signal_processor,
            self.cfar_detector, self.tracker, self.doppler_pulse_count,
        )
        self._present_frame(result, simulation_s, started)

    def _prepare_step(self) -> tuple[tuple[Target, ...], float, float, float]:
        """Advance world time on the UI thread and snapshot target states."""
        started = perf_counter()
        self.world.advance(self.simulation_config.dt_s)
        targets = tuple(deepcopy(target) for target in self.world.targets)
        return targets, self.world.time_s, perf_counter() - started, started

    def _request_live_step(self) -> None:
        """Queue at most one expensive frame while leaving Qt free to repaint."""
        if self._inflight:
            return
        targets, timestamp_s, simulation_s, started = self._prepare_step()
        self._inflight = True
        generation = self._frame_generation
        future = self._executor.submit(
            process_frame, targets, timestamp_s, self.sensor,
            self.signal_processor, self.cfar_detector, self.tracker,
            self.doppler_pulse_count,
        )
        future.add_done_callback(
            lambda finished: self.frame_ready.emit(
                generation, finished, simulation_s, started
            )
        )

    def _finish_async_step(
        self, generation: int, future: Future[FrameComputation],
        simulation_s: float, started: float,
    ) -> None:
        """Publish worker results only if the scenario configuration still matches."""
        if generation != self._frame_generation:
            return
        try:
            result = future.result()
            self._present_frame(result, simulation_s, started)
        except Exception as error:
            self.timer.stop()
            self.statusBar().showMessage(f"Processing error: {error}")
        finally:
            self._inflight = False

    def _invalidate_pending(self) -> None:
        """Discard in-flight output after a reset or configuration change."""
        self._frame_generation += 1
        self._inflight = False

    def _present_frame(
        self, result: FrameComputation, simulation_s: float, frame_started: float
    ) -> None:
        """Update Qt widgets from one completed numerical frame on the UI thread."""
        self.last_sensor_frame = result.sensor_frame
        self.last_range_profile = result.range_profile
        self.last_cfar_result = result.cfar_result
        self.last_doppler_product = result.doppler_product
        self.last_range_angle_product = result.range_angle_product
        self.last_detections = result.detections
        self.last_tracks = result.tracks
        sensing_s = result.timing_metrics_s["sensing_s"]
        dsp_s = result.timing_metrics_s["dsp_s"]
        tracking_s = result.timing_metrics_s["tracking_s"]
        phase_started = perf_counter()
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
        self.range_doppler_view.set_product(self.last_doppler_product)
        self.range_doppler_view.set_detections(self.last_detections)
        self.range_angle_view.set_product(self.last_range_angle_product)
        self.range_angle_view.set_detections(self.last_detections)
        self.ppi_view.set_detections(self.last_detections)
        self.ppi_view.set_tracks(self.last_tracks)
        self.track_table.set_tracks(self.last_tracks)
        self.inspector.set_frame(self.last_detections, self.last_tracks)
        gui_refresh_s = perf_counter() - phase_started
        self.last_timing_metrics_s = {
            "simulation_s": simulation_s,
            "sensing_s": sensing_s,
            "dsp_s": dsp_s,
            "tracking_s": tracking_s,
            "gui_refresh_s": gui_refresh_s,
            "total_s": perf_counter() - frame_started,
        }
        self.last_frame_result = FrameResult(
            timestamp_s=result.timestamp_s,
            transmitted_signal=self.last_sensor_frame.transmitted_signal,
            received_signal=self.last_sensor_frame.received_signal,
            range_profile=self.last_range_profile,
            range_doppler_product=self.last_doppler_product,
            range_angle_product=self.last_range_angle_product,
            detections=list(self.last_detections),
            tracks=list(self.last_tracks),
            timing_metrics_s=self.last_timing_metrics_s,
        )
        self.diagnostics.setText(
            f"Mode: {self.sensor_config.mode.value.title()}\n"
            f"Simulation: {simulation_s * 1e3:.2f} ms\n"
            f"Sensing: {sensing_s * 1e3:.2f} ms\n"
            f"DSP: {dsp_s * 1e3:.2f} ms\n"
            f"Tracking: {tracking_s * 1e3:.2f} ms\n"
            f"GUI refresh: {gui_refresh_s * 1e3:.2f} ms\n"
            f"Total: {self.last_timing_metrics_s['total_s'] * 1e3:.2f} ms\n"
            f"Detections: {len(self.last_detections)}\n"
            f"Tracks: {len(self.last_tracks)}"
        )
        self._refresh_world_views(update_editor=False)

    def reset(self) -> None:
        """Pause and restore the edited scenario baseline."""
        self.timer.stop()
        self.world.reset()
        self._rebuild_sensor_preserving_mode()
        self.last_sensor_frame = None
        self.last_range_profile = None
        self.last_cfar_result = None
        self.last_doppler_product = None
        self.range_doppler_view.clear_product()
        self.last_range_angle_product = None
        self.range_angle_view.clear_product()
        self.last_timing_metrics_s = {}
        self.last_frame_result = None
        self.diagnostics.setText("No frame processed yet")
        self.last_detections = ()
        self.last_tracks = ()
        self.ppi_view.set_detections(())
        self.ppi_view.set_tracks(())
        self.track_table.set_tracks(())
        self.inspector.clear()
        self.range_doppler_view.set_detections(())
        self._refresh_world_views()

    def _add_target(self, target: Target) -> None:
        self.world.add_target(target)
        self.world.checkpoint_reset_state()
        self._rebuild_sensor_preserving_mode()
        self._refresh_world_views()

    def _edit_target(self, old_id: str, target: Target) -> None:
        self.world.replace_target(old_id, target)
        self.world.checkpoint_reset_state()
        self._rebuild_sensor_preserving_mode()
        self._refresh_world_views()

    def _remove_target(self, target_id: str) -> None:
        self.world.remove_target(target_id)
        self.world.checkpoint_reset_state()
        self._rebuild_sensor_preserving_mode()
        self._refresh_world_views()

    def _set_sensor_mode(self, mode_text: str) -> None:
        """Rebuild mode-specific sensor/DSP state while preserving the world."""
        mode = SensorMode(mode_text.lower())
        if mode is self.sensor_config.mode:
            return
        self.timer.stop()
        self._invalidate_pending()
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
        self._update_timer_interval()
        self.last_sensor_frame = None
        self.last_range_profile = None
        self.last_cfar_result = None
        self.last_doppler_product = None
        self.range_doppler_view.clear_product()
        self.last_range_angle_product = None
        self.range_angle_view.clear_product()
        self.last_detections = ()
        self.last_tracks = ()
        self.ppi_view.set_max_range(self.sensor_config.max_range_m)
        self.ppi_view.set_detections(())
        self.ppi_view.set_tracks(())
        self.track_table.set_tracks(())
        self.inspector.clear()
        self.range_doppler_view.set_detections(())
        self.setWindowTitle(
            f"ECHORIN - {self.sensor_config.mode.value.title()} "
            "Engineering Workspace"
        )

    def _set_dt(self, dt_s: float) -> None:
        self.simulation_config = replace(self.simulation_config, dt_s=dt_s)
        self._update_timer_interval()

    def _update_timer_interval(self) -> None:
        """Keep timer cadence above measured mode processing workload."""
        mode_floor_ms = 400 if self.sensor_config.mode is SensorMode.SONAR else 1
        self.timer.setInterval(
            max(mode_floor_ms, round(self.simulation_config.dt_s * 1_000.0))
        )

    def _set_seed(self, random_seed: int) -> None:
        self.simulation_config = replace(
            self.simulation_config, random_seed=random_seed
        )
        self._rebuild_sensor_preserving_mode()

    def _set_noise(self, standard_deviation: float) -> None:
        self.sensor_config = replace(
            self.sensor_config,
            noise_model=NoiseConfig(standard_deviation=standard_deviation),
        )
        self._rebuild_sensor_preserving_mode()

    def _set_waveform(self, waveform_text: str) -> None:
        waveform_kind = (
            WaveformKind.LFM if waveform_text == "LFM" else WaveformKind.RECTANGULAR
        )
        self._rebuild_sensor_preserving_mode(waveform_kind=waveform_kind)

    def _set_preset(self, preset_text: str) -> None:
        self.timer.stop()
        self.world = (
            crossing_targets(seed=self.simulation_config.random_seed)
            if preset_text == "Crossing"
            else single_stationary_target(
                range_m=min(100.0, self.sensor_config.max_range_m / 2.0)
            )
        )
        self._rebuild_sensor_preserving_mode()
        self.ppi_view.set_targets(self.world.targets)
        self.target_editor.set_targets(self.world.targets)

    def _rebuild_sensor_preserving_mode(
        self, waveform_kind: WaveformKind | None = None
    ) -> None:
        self._invalidate_pending()
        waveform_kind = waveform_kind or self.sensor.waveform_kind
        self.sensor = create_sensor(
            self.sensor_config,
            self.world.sensor_pose,
            random_seed=self.simulation_config.random_seed,
        )
        self.sensor.waveform_kind = waveform_kind
        self.tracker = MultiTargetTracker(self.tracker.config)
        self.last_detections = ()
        self.last_tracks = ()

    def _refresh_world_views(self, update_editor: bool = True) -> None:
        self.ppi_view.set_targets(self.world.targets)
        if update_editor:
            self.target_editor.set_targets(self.world.targets)
        self.statusBar().showMessage(
            f"{self.sensor_config.mode.value.title()} | t={self.world.time_s:.2f} s"
            f" | {len(self.last_detections)} detections"
            f" | {len(self.last_tracks)} tracks"
            f" | frame {self.last_timing_metrics_s.get('total_s', 0.0) * 1e3:.1f} ms"
        )


def run_application() -> int:
    """Create the Qt application and display Echorin's main window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
