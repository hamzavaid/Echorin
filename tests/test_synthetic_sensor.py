"""Milestone 3 numerical and integration tests for synthetic radar returns."""

from __future__ import annotations

import numpy as np
import pytest

from echorin.config import NoiseConfig, SensorConfig
from echorin.models import SensorPose
from echorin.sensors.radar import RadarSensor
from echorin.signals.noise import add_awgn
from echorin.signals.propagation import (
    amplitude_at_range,
    delayed_echo,
    round_trip_delay_s,
    sample_delay,
)
from echorin.signals.waveform import lfm_chirp, rectangular_pulse
from echorin.simulation.target import Target


def quiet_config(**overrides: object) -> SensorConfig:
    values: dict[str, object] = {
        "noise_model": NoiseConfig(standard_deviation=0.0),
    }
    values.update(overrides)
    return SensorConfig(**values)


def test_rectangular_pulse_matches_sampled_carrier_formula() -> None:
    config = quiet_config()
    pulse = rectangular_pulse(config, amplitude=2.0)
    time_s = np.arange(config.pulse_samples) / config.sample_rate_hz
    expected = 2.0 * np.cos(2.0 * np.pi * config.carrier_frequency_hz * time_s)

    np.testing.assert_allclose(pulse, expected, rtol=0.0, atol=1e-12)


def test_lfm_chirp_matches_linear_frequency_modulation_formula() -> None:
    config = quiet_config()
    chirp = lfm_chirp(config)
    time_s = np.arange(config.pulse_samples) / config.sample_rate_hz
    start_hz = config.carrier_frequency_hz - config.bandwidth_hz / 2.0
    slope_hz_s = config.bandwidth_hz / config.pulse_width_s
    expected = np.cos(2.0 * np.pi * (start_hz * time_s + 0.5 * slope_hz_s * time_s**2))

    np.testing.assert_allclose(chirp, expected, rtol=0.0, atol=1e-12)
    assert len(chirp) == config.pulse_samples


def test_configuration_rejects_aliasing_of_real_passband_waveform() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        quiet_config(sample_rate_hz=10e6, carrier_frequency_hz=5e6)


def test_round_trip_delay_and_sample_quantization_match_physics() -> None:
    config = quiet_config()
    range_m = 1_500.0
    delay_s = round_trip_delay_s(range_m, config.propagation_speed_mps)
    delay = sample_delay(range_m, config)

    assert delay_s == pytest.approx(2.0 * range_m / config.propagation_speed_mps)
    assert abs(delay / config.sample_rate_hz - delay_s) <= 0.5 / config.sample_rate_hz


def test_delayed_echo_has_physical_delay_and_inverse_power_amplitude() -> None:
    config = quiet_config()
    tx = lfm_chirp(config)
    at_one_km = amplitude_at_range(1_000.0, reflectivity=2.0)
    at_two_km = amplitude_at_range(2_000.0, reflectivity=2.0)
    echo = delayed_echo(tx, range_m=2_000.0, config=config, reflectivity=2.0)
    delay = sample_delay(2_000.0, config)

    assert at_two_km == pytest.approx(at_one_km / 4.0)
    assert np.count_nonzero(echo[:delay]) == 0
    np.testing.assert_allclose(echo[delay : delay + len(tx)], tx * at_two_km)


def test_awgn_is_seeded_and_meets_requested_snr_statistically() -> None:
    signal = np.ones(200_000, dtype=np.float64)
    config = NoiseConfig(standard_deviation=99.0, snr_db=20.0)
    first = add_awgn(signal, config, np.random.default_rng(123))
    second = add_awgn(signal, config, np.random.default_rng(123))
    noise = first - signal
    measured_snr_db = 10.0 * np.log10(np.mean(signal**2) / np.mean(noise**2))

    np.testing.assert_array_equal(first, second)
    assert measured_snr_db == pytest.approx(20.0, abs=0.1)


def test_radar_acquisition_superposes_echoes_and_is_seed_repeatable() -> None:
    config = SensorConfig(noise_model=NoiseConfig(standard_deviation=0.01))
    targets = (
        Target("near", 1_000.0, 0.0, reflectivity=1.0),
        Target("far", 2_000.0, 0.0, reflectivity=0.8),
    )
    first_sensor = RadarSensor(config, SensorPose(), random_seed=17)
    second_sensor = RadarSensor(config, SensorPose(), random_seed=17)

    frame = first_sensor.acquire(targets, timestamp_s=2.5)
    repeated = second_sensor.acquire(targets, timestamp_s=2.5)

    assert frame.timestamp_s == 2.5
    assert frame.transmitted_signal.shape == (config.pulse_samples,)
    assert frame.received_signal.shape == (config.acquisition_samples,)
    np.testing.assert_array_equal(frame.received_signal, repeated.received_signal)
    assert not hasattr(frame, "detections")
    assert not hasattr(frame, "target_ids")


def test_noiseless_radar_frame_equals_sum_of_delayed_target_echoes() -> None:
    config = quiet_config()
    targets = (
        Target("a", 1_000.0, 0.0, reflectivity=1.0),
        Target("b", 0.0, 2_000.0, reflectivity=0.5),
    )
    sensor = RadarSensor(config, SensorPose(), random_seed=1)
    frame = sensor.acquire(targets, timestamp_s=0.0)
    expected = sum(
        (
            delayed_echo(
                frame.transmitted_signal,
                range_m=float(np.hypot(target.x_m, target.y_m)),
                config=config,
                reflectivity=target.reflectivity,
            )
            for target in targets
        ),
        start=np.zeros(config.acquisition_samples),
    )

    np.testing.assert_allclose(frame.received_signal, expected)
