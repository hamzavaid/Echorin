# Architecture

Echorin follows a layered, one-way dependency structure. Ground truth remains
inside simulation and sensor synthesis; downstream DSP and tracking consume only
sampled observations and derived measurements.

```text
simulation World / Targets (ground truth)
        |
        v
signals + sensors (waveform, propagation, AWGN, composite array samples)
        |
        v
DSP (matched filter, range, Bartlett angle, Doppler, CA-CFAR)
        |
        v
models Detection -> tracking (association, Kalman, lifecycle)
        |
        v
application/frame_pipeline -> GUI / recording / benchmarks
```

## Package boundaries

- `config.py` contains validated radar, sonar, noise, and simulation settings.
- `simulation/` owns target identity, kinematics, scenarios, and the world clock.
- `signals/` implements sampled waveforms, delay/attenuation, and Gaussian noise.
- `sensors/` provides the common synthetic monostatic engine plus thin Radar and
  Sonar specializations. Production frames contain composite element/pulse/
  sample observations and receiver geometry, without truth identity or state.
- `dsp/` performs correlation, physical range mapping, Bartlett beamforming,
  CA-CFAR, and slow-time FFT.
- `models/` contains data-only detections, frame results, geometry, and tracks.
- `tracking/` converts polar detections to Cartesian measurements, performs
  Mahalanobis-gated association, and maintains Kalman track lifecycle.
- `application/frame_pipeline.py` runs the existing sensor, DSP, detection, and
  tracking chain synchronously with no Qt imports. Both manual and live updates
  call this same function.
- `gui/` is the only package that imports PySide6 or PyQtGraph. It orchestrates
  public layer APIs and renders PPI, range, threshold, Doppler, and track views.
  `main_window.py` coordinates the application frame. `visualization_data.py`
  and `inspection_data.py` adapt only published DSP, detection, and track data;
  `heatmaps.py`, `ppi_view.py`, and `inspectors.py` render those adapters.
- `recording.py` serializes scenarios and truth-free frame outputs.
- `benchmark.py` compares noisy observations and filtered tracks against hidden
  truth strictly as an evaluation boundary.

## Processing cycle

Each timer event advances simulation time, acquires a composite coherent array
pulse train, applies matched filtering, forms range/angle/Doppler products, runs CA-CFAR,
associates detections, updates tracks, publishes a `FrameResult`, and refreshes
the GUI. Timings for simulation, sensing, DSP, tracking, and GUI refresh are
measured separately. Manual Step invokes the synchronous application pipeline.
Live Run snapshots the advanced world state, computes one frame at a time on a
worker thread, and publishes completed results back on the Qt event thread.
Reset or configuration changes invalidate in-flight results. The worker never
updates widgets or reads mutable world targets directly.

## v1.1 engineering workspace

The central PPI is surrounded by named `QDockWidget` panels for sensor/scenario
controls, tracks, signal products, the full Range-Doppler map, measurement/track
inspection, and performance diagnostics. Users can move, float, hide, or resize
panels. Qt settings persist dock geometry, selected tabs, and theme. The View
menu includes temporary PPI focus mode and dark/light themes.

The coherent Doppler FFT remains the only Range-Doppler processing path. Its
`DopplerProduct` carries the physical range axis alongside the existing velocity
axis and `[velocity, range]` spectrum. The GUI adapter floors dB values and
maps image cells to physical coordinates. Selecting a heatmap cell updates the
legacy one-dimensional Doppler spectrum. Detection markers, inspector fields,
track vectors, and covariance ellipses derive from detection/track outputs;
ground truth remains an optional, separately controlled PPI layer.

The v1.2 Range-Angle dock consumes only the beamformed product. Range CFAR
selects candidate bins; angular peaks in Bartlett power supply bearings to the
existing tracker. The older directional-frame API remains callable for
compatibility but is not used by the production frame controller.

## Simplifying assumptions

Targets are point reflectors with constant Cartesian velocity. Propagation uses
integer-sample two-way delay, bounded inverse-power amplitude, AWGN, no clutter,
and no multipath. Array phase uses the far-field narrowband approximation and
has ULA front/back ambiguity. Coherent pulse trains use a stop-and-hop assumption,
so range migration within one coherent processing interval is ignored.
