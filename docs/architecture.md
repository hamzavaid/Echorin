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
The production frame pipeline now applies a known-waveform residual sidelobe
filter after range CFAR and a noise-calibrated angular threshold after
beamforming. Both operate only on sampled signal products. Array-enabled
tracking propagates range/bearing measurement uncertainty into world Cartesian
covariance for gating, correction and track initialization; legacy tracker
callers retain their fixed-variance path.
The GUI configures edge-adaptive CA-CFAR for near-range acquisition, while
the reusable detector's default masked-edge policy is backward compatible.

## v1.3 moving-platform boundary

`World` owns a timestamped `PlatformState`, validated motion law, body-frame
`MountTransform`, array configuration, and a bounded platform path. It advances
the target and platform clocks together before each frame. The receiver pose
is a rigid transform of platform position/heading; its velocity includes the
angular-rate cross mount-offset term. Sensor synthesis uses this pose to form
receiver-relative range rate and Doppler. A coherent frame holds the pose fixed
under the existing stop-and-hop approximation.

The array product and each detection carry the acquisition-time receiver pose.
Detection bearings remain receiver-local; `Detection.world_position_m` is the
single local-to-world conversion used by tracking, PPI and inspection adapters.
Numerical modules do not import Qt and receive no target ID or truth bearing.
`PlatformControls` edits scenario state through the main-window controller; the
PPI gets an optional platform path/heading/FOV overlay separately from sensor
measurements. Scenario and frame JSON use schema version 2 and readers migrate
the earlier unversioned files. A platform pose is recorded even when a frame
has no detections.

## Simplifying assumptions

Targets are point reflectors with constant Cartesian velocity. Default propagation
uses integer-sample two-way delay, bounded inverse-power amplitude and AWGN.
Optional v1.4 environmental effects are described below. Array phase uses the
far-field narrowband approximation and
has ULA front/back ambiguity. Coherent pulse trains use a stop-and-hop assumption,
so range migration within one coherent processing interval is ignored.

## v1.4 environment and propagation boundary

`World` owns immutable `EnvironmentConfig` and `BeamConfig` alongside platform
and array settings. The factory composes these into the shared monostatic sensor.
`environment/` supplies noise, coherent tones and persistent scattering fields;
`propagation/multipath.py` expands direct/secondary paths. `sensors/beam_pattern.py`
and `sensors/scan.py` provide one-way gains and a stateless simulation-clock scan.
No numerical module imports Qt. No DSP/detection/tracking API acquires truth IDs,
exact target coordinates or truth bearings.

The sensor accumulates delayed, attenuated, beam-weighted target/field/path
returns in `[element, pulse, sample]`, adds coherent interference, then receiver
noise. Existing matched filtering, Range-Doppler, Bartlett, CA-CFAR and tracking
consume the same raw-array interface. Secondary paths and clutter are not
injected as artificial detections. Receiver noise is a `NoiseModel` protocol;
AWGN delegates to the existing real/complex/SNR implementation.

The original AWGN RNG stream is retained. Field randomness uses an independent
`SeedSequence(seed, spawn_key=(1,))`, so adding a field does not reorder receiver
noise draws. Fields are anchored in world coordinates at sensor construction;
their phases/velocities persist across acquisitions. Reset/rebuild restores the
seeded state. Changing noise kind intentionally changes its own draw sequence.

The Environment / Beam / Scan dock validates edits before replacing the sensor,
then restarts the scenario. Invalid mode changes leave the prior sensor intact.
Live processing uses the existing single-worker pipeline and generation-based
stale-result rejection. Beam overlay uses configured boresight/width, not truth.
Scenario schema 3 persists all environment and scan settings; schema 1/2 load
with empty environment/isotropic defaults. Frame export remains schema 2 and
truth-free. Saved scenarios describe a fresh seeded field anchored to their
starting receiver pose, not a checkpoint of a previously evolved field.

## v1.5 generalized TX/RX boundary

`sensors/components.py` defines typed `Emitter`, `Receiver` and `SensorPlatform`
composition. The world owns independent platform states/trajectories and bounded
histories, while mounts produce acquisition-time device poses. The legacy
single-platform interface remains unchanged for old scenes. Centralized
`propagation/geometry.py` evaluates the two moving legs and bistatic Doppler.
`create_link_sensor` configures the **same** raw-array synthesis, with independent
TX/RX beam gains, per-leg attenuation and source IDs. No ideal per-target output
enters the detector; all targets/scatterers/paths sum in physical array channels.

`application/sensor_network.py` schedules bounded, synchronized time-division
TX slots and processes every TX/RX pair through `process_frame`. Each stream has
its own unchanged CV tracker; no fusion exists. `process_workspace_frame` keeps
the legacy call path or publishes the selected stream plus `source_frames` for
all raw arrays/DSP products/detections/tracks. This synchronous layer has no Qt
imports; live execution still uses one worker and stale-generation rejection.

Range products use half total path length. Doppler's legacy numeric velocity
axis is half path rate in bistatic mode, marked with `is_bistatic` and labeled by
the GUI. Detections carry measured total path/rate and emitter/receiver IDs.
Known device poses plus measured arrival angle intersect the bistatic ellipse
to infer receiver distance; no target position/bearing is read downstream.
The conversion Jacobian propagates measurement covariance into world coordinates.
Bistatic receiver radial velocity is unavailable, not secretly inferred from truth.

The Transmitters / Receivers dock validates scenario/device JSON atomically,
offers medium-scaled moving presets, and selects source-scoped views. The PPI
shows known device positions, headings, beams, platform paths and TX/RX baseline.
Track IDs are local to the selected source pair. Scenario schema 4 adds explicit
platform/device configuration; legacy worlds still write schema 3. Frame schema
2 gains optional truth-free source summaries; CSV flattens all streams and adds
source/path columns. `bistatic_benchmark.py` alone uses target truth to validate
measurements against analytic references. See [v1.5 limits](releases/v1.5.md).
