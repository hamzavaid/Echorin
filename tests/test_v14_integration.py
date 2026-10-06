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
