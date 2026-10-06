"""Statistical contracts for configurable receiver disturbances."""

import numpy as np
import pytest
from scipy.signal import welch

from echorin.config import NoiseConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.environment.noise import (
    AwgnNoise,
    ColoredNoise,
    CorrelatedArrayNoise,
    ImpulsiveNoise,
)
from echorin.signals.noise import add_awgn


def test_awgn_adapter_preserves_real_complex_and_seed_contract():
    for signal in (np.zeros(200_000), np.zeros(200_000, complex)):
        result = AwgnNoise(NoiseConfig(0.3)).add(signal, np.random.default_rng(7))
        np.testing.assert_array_equal(
            result, add_awgn(signal, NoiseConfig(0.3), np.random.default_rng(7))
        )
        assert abs(np.mean(result)) < 0.003
        assert np.var(result) == pytest.approx(0.09, rel=0.015)


def test_colored_noise_has_configured_ar1_psd_and_variance():
    model = ColoredNoise(NoiseConfig(0.2), correlation=0.85)
    result = model.add(np.zeros(131_072), np.random.default_rng(5))
    frequencies, psd = welch(result, nperseg=2048)
    assert psd[frequencies < 0.05].mean() > 30 * psd[frequencies > 0.4].mean()
    assert np.var(result) == pytest.approx(0.04, rel=0.05)
    np.testing.assert_array_equal(
        result, model.add(np.zeros(result.size), np.random.default_rng(5))
    )
    channels = model.add(np.zeros((2, 100_000), complex), np.random.default_rng(4))
    assert abs(np.corrcoef(channels.real)[0, 1]) < 0.03


def test_impulse_event_rate_amplitude_and_complex_power():
    model = ImpulsiveNoise(NoiseConfig(0), event_probability=0.02, amplitude=3)
    for signal in (np.zeros(200_000), np.zeros(200_000, complex)):
        result = model.add(signal, np.random.default_rng(7))
        events = np.abs(result) > 0
        assert events.mean() == pytest.approx(0.02, abs=0.001)
        np.testing.assert_allclose(np.abs(result[events]), 3)


def test_tone_frequency_drift_and_phase_are_physical():
    tone = NarrowbandInterference(100, 2, phase_rad=0.3, drift_hz_s=10)
    times = np.arange(1000) / 1000
    result = tone.samples(times, complex_output=True)
    np.testing.assert_allclose(
        result, 2 * np.exp(1j * (0.3 + 2 * np.pi * (100 * times + 5 * times**2)))
    )
    np.testing.assert_allclose(tone.samples(times, complex_output=False), result.real)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ColoredNoise(correlation=1),
        lambda: ImpulsiveNoise(event_probability=-0.1),
        lambda: ImpulsiveNoise(amplitude=float("nan")),
        lambda: NarrowbandInterference(float("inf"), 1),
    ],
)
def test_invalid_disturbance_parameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_correlated_array_noise_matches_requested_covariance():
    model = CorrelatedArrayNoise(NoiseConfig(0.3), correlation=0.6)
    result = model.add(np.zeros((4, 100_000), complex), np.random.default_rng(7))
    covariance = result @ result.conj().T / result.shape[-1]
    expected = 0.09 * (0.4 * np.eye(4) + 0.6 * np.ones((4, 4)))
    np.testing.assert_allclose(covariance, expected, atol=0.001)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_noise_scale_rejects_nonfinite_or_negative_values(value):
    with pytest.raises(ValueError):
        NoiseConfig(value)
