"""Environmental returns must enter the real sampled sensor/DSP pipeline."""

import numpy as np
import pytest

from echorin.config import NoiseConfig, SensorConfig
from echorin.dsp.range_processing import SignalProcessor
from echorin.environment.clutter import ClutterConfig
from echorin.environment.config import EnvironmentConfig, ReceiverNoiseConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.propagation.multipath import MultipathComponent
from echorin.sensors.beam_pattern import BeamConfig
from echorin.sensors.factory import create_sensor
from echorin.sensors.scan import ScanConfig
from echorin.signals.noise import add_awgn
from echorin.signals.waveform import generate_waveform
from echorin.simulation.kinematics import relative_geometry
from echorin.simulation.target import Target


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_default_environment_preserves_baseline_and_reset(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.02))
    target = Target("private", config.max_range_m / 2, 0)
    legacy = create_sensor(config, random_seed=8)
    new = create_sensor(config, random_seed=8, environment=EnvironmentConfig())
    first = new.acquire_array_pulse_train((target,), 0, 8).samples
    np.testing.assert_array_equal(
        first, legacy.acquire_array_pulse_train((target,), 0, 8).samples
    )
    new.reset()
    np.testing.assert_array_equal(
        first, new.acquire_array_pulse_train((target,), 0, 8).samples
    )


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_array_synthesis_matches_dense_reference_exactly(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.02))
    sensor = create_sensor(config, random_seed=13)
    targets = (
        Target("a", config.max_range_m / 2, 0),
        Target("b", config.max_range_m / 3, config.max_range_m / 5, reflectivity=0.6),
    )
    reference = np.zeros((8, 8, config.acquisition_samples), complex)
    waveform = generate_waveform(config, sensor.waveform_kind)
    positions = sensor.array_geometry.element_positions_m
    relative = positions - positions[sensor.array_geometry.reference_element]
    wavelength = config.propagation_speed_mps / config.carrier_frequency_hz
    for target in targets:
        geometry = relative_geometry(sensor.pose, target)
        phase = np.exp(
            2j
            * np.pi
            * (
                relative
                @ np.array([np.cos(geometry.bearing_rad), np.sin(geometry.bearing_rad)])
            )
            / wavelength
        )
        response = sensor._coherent_target_return(
            waveform,
            geometry.range_m,
            geometry.radial_velocity_mps,
            target.reflectivity,
            8,
        )
        reference += phase[:, None, None] * response[None, :, :]
    expected = add_awgn(reference, config.noise_model, np.random.default_rng(13))
    np.testing.assert_array_equal(
        sensor.acquire_array_pulse_train(targets, 0.1, 8).samples, expected
    )


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_multipath_ghost_is_delayed_and_measured_by_matched_filter(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0))
    range_m = config.max_range_m / 4
    extra = config.max_range_m / 3
    env = EnvironmentConfig(multipath=(MultipathComponent(extra, 0.5),))
    sensor = create_sensor(config, environment=env)
    raw = sensor.acquire_array_pulse_train((Target("hidden", range_m, 0),), 0, 8)
    profile = SignalProcessor(config).range_profile(
        raw.samples[0, 0], raw.transmitted_signal
    )
    ghost_bin = np.argmin(abs(profile.ranges_m - (range_m + extra / 2)))
    direct_bin = np.argmin(abs(profile.ranges_m - range_m))
    assert profile.magnitude[ghost_bin] > profile.magnitude[direct_bin] * 0.1
    assert np.argmax(profile.magnitude) == pytest.approx(direct_bin, abs=1)


def test_scan_changes_raw_echo_strength_and_static_beam_two_way_gain():
    config = SensorConfig(noise_model=NoiseConfig(0))
    target = Target("private", 1500, 0)
    beam = BeamConfig(
        kind="sector",
        width_rad=0.2,
        scan=ScanConfig(kind="step", dwell_s=1, step_angles_rad=(0, 1)),
    )
    sensor = create_sensor(config, beam=beam)
    assert np.max(abs(sensor.acquire_array_pulse_train((target,), 0, 8).samples)) > 0
    assert not np.any(sensor.acquire_array_pulse_train((target,), 1, 8).samples)
    baseline = create_sensor(config).acquire_array_pulse_train((target,), 0, 8)
    gaussian = create_sensor(
        config,
        beam=BeamConfig(
            kind="gaussian", width_rad=0.4, scan=ScanConfig(boresight_rad=0.2)
        ),
    )
    response = gaussian.acquire_array_pulse_train((target,), 0, 8)
    np.testing.assert_allclose(response.samples, 0.5 * baseline.samples, atol=1e-12)


@pytest.mark.parametrize("mode", ["radar", "sonar"])
def test_all_environment_effects_are_seeded_and_flow_through_acquisition(mode):
    config = getattr(SensorConfig, mode)(noise_model=NoiseConfig(0.001))
    env = EnvironmentConfig(
        receiver_noise=ReceiverNoiseConfig(kind="colored"),
        interference=(NarrowbandInterference(100, 0.01),),
        clutter=(
            ClutterConfig(
                kind="reverberation" if mode == "sonar" else "clutter",
                density_per_m=20 / config.max_range_m,
            ),
        ),
    )
    sensor = create_sensor(config, environment=env, random_seed=12)
    raw = sensor.acquire_array_pulse_train((), 0.3, 8)
    sensor.reset()
    np.testing.assert_array_equal(
        raw.samples, sensor.acquire_array_pulse_train((), 0.3, 8).samples
    )
    assert np.all(np.isfinite(raw.samples))
    # No truth objects are necessary to generate or process clutter.
    profile = SignalProcessor(config).range_profile(
        raw.samples[0, 0], raw.transmitted_signal
    )
    assert np.max(profile.magnitude) > 0


def test_interference_spatial_phase_and_independent_rng_streams():
    config = SensorConfig(noise_model=NoiseConfig(0.02))
    env = EnvironmentConfig(
        interference=(
            NarrowbandInterference(1_000_000, 0.1, arrival_angle_rad=np.pi / 6),
        )
    )
    base = create_sensor(config, random_seed=3)
    sensor = create_sensor(config, environment=env, random_seed=3)
    difference = (
        sensor.acquire_array_pulse_train((), 0, 8).samples
        - base.acquire_array_pulse_train((), 0, 8).samples
    )
    np.testing.assert_allclose(difference[1] / difference[0], 1j, atol=1e-12)


def test_environment_work_and_aliasing_validation():
    with pytest.raises(ValueError, match="Nyquist"):
        create_sensor(
            SensorConfig(),
            environment=EnvironmentConfig(interference=(NarrowbandInterference(1e9),)),
        )
    with pytest.raises(ValueError, match="work limit"):
        create_sensor(
            SensorConfig(),
            environment=EnvironmentConfig(clutter=(ClutterConfig(density_per_m=100),)),
        )


def test_rejected_sensor_reconfiguration_is_atomic():
    sensor = create_sensor(SensorConfig())
    old = sensor.environment
    with pytest.raises(ValueError, match="work limit"):
        sensor.configure_environment(
            EnvironmentConfig(
                receiver_noise=ReceiverNoiseConfig(kind="colored"),
                clutter=(ClutterConfig(density_per_m=100),),
            ),
            BeamConfig(kind="sector"),
        )
    assert sensor.environment == old
    assert sensor.beam == BeamConfig()


def test_drifting_angle_interference_uses_retarded_phase_in_each_receiver():
    config = SensorConfig(noise_model=NoiseConfig(0))
    tone = NarrowbandInterference(1e6, 0.1, drift_hz_s=1000, arrival_angle_rad=0.3)
    sensor = create_sensor(config, environment=EnvironmentConfig(interference=(tone,)))
    raw = sensor.acquire_array_pulse_train((), 0.2, 8)
    fast = np.arange(config.acquisition_samples) / config.sample_rate_hz
    times = 0.2 + np.arange(8)[:, None] / config.prf_hz + fast[None, :]
    positions = sensor.array_geometry.element_positions_m
    relative = positions - positions[sensor.array_geometry.reference_element]
    delays = (
        relative @ np.array([np.cos(0.3), np.sin(0.3)]) / config.propagation_speed_mps
    )
    expected = tone.samples(times[None, :, :] + delays[:, None, None])
    np.testing.assert_allclose(raw.samples, expected, atol=1e-10)


def test_single_pulse_environment_supports_generator_targets_and_path_phase():
    config = SensorConfig(noise_model=NoiseConfig(0))
    env = EnvironmentConfig(multipath=(MultipathComponent(3000, 0.5, np.pi),))
    sensor = create_sensor(config, environment=env)
    target = Target("private", 1500, 0)
    np.testing.assert_array_equal(
        sensor.acquire((p for p in (target,)), 0.1).received_signal,
        sensor.acquire((target,), 0.1).received_signal,
    )
    # Same-delay phase inversion cancels a direct echo before DSP.
    cancel = create_sensor(
        config,
        environment=EnvironmentConfig(multipath=(MultipathComponent(0, 1, np.pi),)),
    )
    assert (
        np.max(abs(cancel.acquire_array_pulse_train((target,), 0.1, 8).samples)) < 1e-12
    )


def test_angle_interference_is_receive_beam_weighted_but_common_mode_is_not():
    config = SensorConfig(noise_model=NoiseConfig(0))
    beam = BeamConfig(kind="sector", width_rad=0.2)
    external = create_sensor(
        config,
        beam=beam,
        environment=EnvironmentConfig(
            interference=(NarrowbandInterference(1e6, 0.1, arrival_angle_rad=0.5),)
        ),
    )
    assert not np.any(external.acquire_array_pulse_train((), 0, 8).samples)
    internal = create_sensor(
        config,
        beam=beam,
        environment=EnvironmentConfig(interference=(NarrowbandInterference(1e6, 0.1),)),
    )
    assert np.max(
        abs(internal.acquire_array_pulse_train((), 0, 8).samples)
    ) == pytest.approx(0.1)
