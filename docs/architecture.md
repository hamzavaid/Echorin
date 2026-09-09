# Architecture

Echorin follows a layered, one-way dependency structure. Ground truth remains
inside simulation and sensor synthesis; downstream DSP and tracking consume only
sampled observations and derived measurements.

```text
simulation World / Targets (ground truth)
        |
        v
signals + sensors (waveform, propagation, AWGN, angular channels)
        |
        v
DSP (matched filter, range, Doppler, CA-CFAR)
        |
        v
models Detection -> tracking (association, Kalman, lifecycle)
        |
        v
GUI / recording / benchmarks
```

## Package boundaries

- `config.py` contains validated radar, sonar, noise, and simulation settings.
- `simulation/` owns target identity, kinematics, scenarios, and the world clock.
- `signals/` implements sampled waveforms, delay/attenuation, and Gaussian noise.
- `sensors/` provides the common synthetic monostatic engine plus thin Radar and
  Sonar specializations. Sensor frames contain signals and measured bearing, but
  no truth identity or target state.
- `dsp/` performs correlation, physical range mapping, CA-CFAR, and slow-time FFT.
- `models/` contains data-only detections, frame results, geometry, and tracks.
- `tracking/` converts polar detections to Cartesian measurements, performs
  Mahalanobis-gated association, and maintains Kalman track lifecycle.
- `gui/` is the only package that imports PySide6 or PyQtGraph. It orchestrates
  public layer APIs and renders PPI, range, threshold, Doppler, and track views.
- `recording.py` serializes scenarios and truth-free frame outputs.
- `benchmark.py` compares noisy observations and filtered tracks against hidden
  truth strictly as an evaluation boundary.

## Processing cycle

Each timer event advances simulation time, acquires coherent angular pulse
trains, applies matched filtering, forms range/Doppler products, runs CA-CFAR,
associates detections, updates tracks, publishes a `FrameResult`, and refreshes
the GUI. Timings for simulation, sensing, DSP, tracking, and GUI refresh are
measured separately.

## Simplifying assumptions

Targets are point reflectors with constant Cartesian velocity. Propagation uses
integer-sample two-way delay, bounded inverse-power amplitude, AWGN, no clutter,
and no multipath. Angular channels model an ideal beamformer with configurable
bearing noise. Coherent pulse trains use a stop-and-hop narrowband assumption,
so range migration within one coherent processing interval is ignored.
