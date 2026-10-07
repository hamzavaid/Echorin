# Deterministic sensing scenarios

## v1.2 same-range angular resolution

Run `python benchmarks/run_array_benchmark.py` from an installed source tree.
The script reports its seed, Radar configuration, target range/bearings, measured
bearings, range and angle bin widths, array size, pulse count, and stage timings
as JSON. Two point targets are placed at 300 m and bearings -0.45 and +0.45
radians. Both occupy the same range bin. At seed 7 and receiver noise sigma
0.001, each reported bearing must be within one 1-degree scan bin of its
configured position. The target identifiers are used only to construct truth;
the production array frame and detector receive no identifier or truth bearing.

This is a high-SNR educational far-field example, not a general angular
resolution guarantee. Array aperture, SNR, separation, and ULA ambiguity affect
other cases.

## v1.3 moving receiver validation

Run `python benchmarks/run_platform_benchmark.py`. With seed 7, the platform
moves at +10 km/s from the origin for 0.01 s, placing the receiver at (100, 0)
metres. A stationary point target is at (400, 100) metres. Analytic range,
bearing and range rate are compared with the signal-derived array detection;
errors must fit the reported range, angular and Doppler bin tolerances. The
fast platform speed produces an easily measurable Doppler shift in the small
educational Radar configuration; it is not intended as a realistic vehicle.
The benchmark uses target coordinates only at its evaluation boundary.

## Crossing preset observability

The Crossing preset's two seeded trajectories lie in the receiver's forward
half-plane. Their projected paths intersect at different times, keeping both
echoes separately observable through the demonstration. The former opposite-
side layout placed one target behind the uniform linear array; front/back
ambiguity made it appear at the other's bearing. The preset does not pretend
that a linear array can resolve two targets at exactly the same range and
bearing. Regression tests follow the default seed for 160 frames and additional
seeds for tracker-ID continuity.

## Single preset range

The Radar Single preset uses a stationary point reflector at 1,500 m, near
the Crossing preset's scale rather than almost at the PPI origin. Sonar caps
the preset at half its configured maximum range (100 m by default). Edge-
adaptive CA-CFAR separately validates that an explicitly configured 100-m
Radar target is detected; moving the preset does not hide a near-range blind
spot in the processing chain.

## v1.4 environmental engineering workspace

Open **View → Environment / Beam / Scan** (also tabbed beside Scenario and
Sensor Platform). Select a noise kind, beam pattern and scan mode; edit angular
controls in **degrees**. Choose an effects template, inspect/edit its JSON,
then click **Apply environment / beam**. Apply pauses and resets the scenario
so the edited configuration starts reproducibly. Existing Sensor controls set
receiver noise sigma. The beam checkbox changes only overlay visibility, not
signal gains. The separate ULA FOV guide is not the beam envelope.

JSON effects support multiple entries. Example for Radar with two tones, a
sparse field, and a secondary path (JSON angles are **radians**):

```json
{
  "interference": [
    {"frequency_hz": 1000000, "amplitude": 0.02, "arrival_angle_rad": 0.2},
    {"frequency_hz": 800000, "amplitude": 0.01, "drift_hz_s": 10}
  ],
  "clutter": [{"density_per_m": 0.002, "reflectivity_scale": 0.1}],
  "multipath": [{"extra_path_length_m": 2000, "attenuation_scale": 0.4,
                 "phase_offset_rad": 0.3, "angle_offset_rad": 0.05}],
  "attenuation_exponent": 2,
  "absorption_db_per_m": 0
}
```

For Sonar use frequencies below 48 kHz at the default sample rate, path lengths
appropriate to its 200-m maximum range, and `kind: "reverberation"` with
`decay_range_m`. The Clutter / Reverberation and Multipath templates scale to
the selected medium. Interference drift is checked against sample Nyquist on
every acquisition; config/work-limit errors are shown without replacing the
sensor. Increasing field density increases processing work. Doppler-aliasing
limits also apply to field velocities and moving-platform relative motion.

Programmatic use shares the exact same sensor pipeline:

```python
from echorin.config import SensorConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.propagation.multipath import MultipathComponent
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.factory import create_sensor
from echorin.sensors.scan import ScanConfig

environment = EnvironmentConfig(
    receiver_noise=ReceiverNoiseConfig(kind="colored", correlation=0.85),
    multipath=(MultipathComponent(2000, 0.4),),
)
beam = BeamConfig(kind="gaussian", width_rad=1.0,
                  scan=ScanConfig(kind="sector"))
sensor = create_sensor(SensorConfig(), random_seed=7,
                       environment=environment, beam=beam)
raw = sensor.acquire_array_pulse_train((), timestamp_s=0.1, pulse_count=32)
```

For scenario persistence set `world.environment_config` / `world.beam_config`
before calling the existing `save_scenario_json`; schema 3 preserves all
parameters, including discrete scan angles. Legacy schemas 1/2 load as AWGN,
no extra effects and isotropic beam. Frame recordings remain truth-free schema 2.

### Reproducible environmental benchmark

Run `python benchmarks/run_environment_benchmark.py`. Seed 7, four frames per
case, noise sigma 0.001, eight half-wavelength receivers, Radar max range 3 km
and 32 pulses, Sonar max range 60 m and 16 pulses. A stationary target is at
one-third maximum range and bearing 20 degrees. The JSON includes full physical,
environment and beam configuration, detection counts, target/ghost hits, bin
widths, errors and timings. The scan case alternates on/off dwells. Noise cases
include AWGN, AR(1), impulses and receiver correlation; field case is Rayleigh
clutter in Radar and exponentially decaying reverberation in Sonar. Density,
phase, interference and secondary paths affect raw signals; they are never
injected as perfect measurements.

The [comparison figure](../benchmarks/environment_comparison.svg) shows one
baseline versus field-contaminated matched-filter profile and the corresponding
CA-CFAR thresholds in both modes. Regenerate it and the environment workspace
capture with `python examples/capture_environment_demo.py` after installing
the optional `release` dependencies. The older release captures remain intact.

## v1.5 moving bistatic / multistatic references

In View → Transmitters / Receivers choose Bistatic (one moving TX, one moving RX)
or Multistatic (two of each), then Load scenario. The target starts at three-
tenths maximum range east and one-tenth north. Transmitters/receivers move under
independent constant-velocity trajectories; mounts and other trajectory types
can be edited in the validated JSON. Default Radar target speed is capped at
300 m/s; Sonar speeds respect its narrow unambiguous Doppler interval. Reload a
preset after changing medium to obtain appropriate range/speed scales.

The source selector displays one TX→RX stream's signal products/measurements
and local tracks; all pairs are processed. Legacy Single/Crossing controls
switch back to their original monostatic scenes. Ground-truth targets remain
an optional overlay. Device IDs/positions/paths are known configuration, not
hidden target measurements. Platform trail visibility also applies to network
platform paths. Per-device beams are edited in network JSON; legacy platform
controls apply only to legacy scenes.

Run `python benchmarks/run_bistatic_benchmark.py` for full moving two-TX/two-RX
evidence in both media (32 pulses, seed 7, four 0.05-s frames). Radar deliberately
uses carrier 4 MHz and PRF 20 Hz to resolve nonzero Doppler; other Radar defaults
remain unchanged. Each link must detect every frame and match analytic total
path, arrival bearing and total path rate within their configured physical bins.
Repeat runs compare identical measured outputs, excluding timing. The
[error figure](../benchmarks/bistatic_validation.svg) normalizes all errors by
these bin widths. Recreate it and the
[workspace screenshot](../screenshots/echorin-multistatic.png) with
`python examples/capture_bistatic_demo.py`.

Scenario schema 4 serializes explicit devices/platforms; empty networks remain
schema 3 and schemas 1–3 still migrate. Frame schema 2 adds optional
`source_frames` with source IDs, known device poses and truth-free measurements/
local tracks; CSV exports all source rows with pair identity. No cross-source
fusion is performed. Simplified secondary paths/clutter can generate nuisance
tracks exactly as in the monostatic simulation.
