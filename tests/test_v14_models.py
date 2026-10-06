"""Physical gain, timing, clutter statistics and delayed path contracts."""

import numpy as np
import pytest

from echorin.environment.clutter import ClutterConfig, scatterers
from echorin.propagation.multipath import MultipathComponent, expand_paths
from echorin.sensors.beam_pattern import (
    ApertureSincBeam,
    GaussianBeam,
    IsotropicBeam,
    SectorBeam,
)
from echorin.sensors.scan import ScanConfig, ScanScheduler


def test_beam_patterns_are_amplitude_gains_with_physical_half_power_width():
    assert IsotropicBeam().gain(2) == 1
    sector = SectorBeam(np.pi / 3, sidelobe_gain=0.02)
    assert sector.gain(0) == 1
    assert sector.gain(np.pi) == 0.02
    gaussian = GaussianBeam(np.pi / 3)
    assert gaussian.gain(0) == 1
    assert gaussian.gain(np.pi / 6) ** 2 == pytest.approx(0.5)
    assert gaussian.gain(2 * np.pi) == 1
    aperture = ApertureSincBeam(4)
    assert aperture.gain(0) == 1
    assert aperture.gain(np.arcsin(1 / 4)) < 1e-12
    assert aperture.gain(0.4) == pytest.approx(aperture.gain(-0.4))


def test_scan_modes_follow_simulation_time_not_wall_clock():
    stationary = ScanScheduler(ScanConfig(rate_rad_s=0))
    assert stationary.boresight(10) == 0
    fixed = ScanScheduler(ScanConfig(boresight_rad=0.2))
    assert fixed.boresight(123) == 0.2
    rotating = ScanScheduler(ScanConfig(kind="rotation", rate_rad_s=1))
    assert rotating.boresight(7) == pytest.approx(7 - 2 * np.pi)
    sector = ScanScheduler(
        ScanConfig(kind="sector", start_rad=-1, stop_rad=1, rate_rad_s=1)
    )
    np.testing.assert_allclose(
        [sector.boresight(t) for t in (0, 1, 2, 3, 4)], [-1, 0, 1, 0, -1]
    )
    reset = ScanScheduler(
        ScanConfig(
            kind="sector", start_rad=-1, stop_rad=1, rate_rad_s=1, ping_pong=False
        )
    )
    assert reset.boresight(2) == -1
    steps = ScanScheduler(
        ScanConfig(kind="step", dwell_s=0.5, step_angles_rad=(-0.5, 0, 0.5))
    )
    np.testing.assert_allclose(
        [steps.boresight(t) for t in (0, 0.499, 0.5, 1, 1.5)],
        [-0.5, -0.5, 0, 0.5, -0.5],
    )


def test_multipath_total_length_delay_phase_and_angle():
    component = MultipathComponent(60, 0.4, np.pi / 2, 0.1)
    paths = expand_paths(100, 1500, (component,))
    assert paths[0].path_length_m == 200
    assert paths[0].delay_s == pytest.approx(200 / 1500)
    assert paths[1].path_length_m == 260
    assert paths[1].delay_s == pytest.approx(260 / 1500)
    assert paths[1].amplitude_scale == 0.4
    assert paths[1].phase_rad == np.pi / 2
    assert paths[1].angle_offset_rad == 0.1


def test_clutter_is_seeded_distributed_and_reverberation_is_distinct():
    config = ClutterConfig(density_per_m=2, reflectivity_scale=0.1, maximum_speed_mps=3)
    first = scatterers(config, 100, np.random.default_rng(3))
    assert 150 < len(first) < 250
    assert first == scatterers(config, 100, np.random.default_rng(3))
    assert all(0 <= p.range_m <= 100 and abs(p.radial_velocity_mps) <= 3 for p in first)
    assert np.mean([p.reflectivity**2 for p in first]) == pytest.approx(0.02, rel=0.25)
    reverberation = scatterers(
        ClutterConfig(
            kind="reverberation",
            density_per_m=20,
            reflectivity_scale=0.1,
            decay_range_m=10,
        ),
        100,
        np.random.default_rng(3),
    )
    near = [p.reflectivity for p in reverberation if p.range_m < 20]
    far = [p.reflectivity for p in reverberation if p.range_m > 80]
    assert np.mean(near) > 100 * np.mean(far)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianBeam(0),
        lambda: SectorBeam(1, -1),
        lambda: ApertureSincBeam(float("nan")),
        lambda: ScanConfig(kind="sector", start_rad=1, stop_rad=-1),
        lambda: ScanConfig(dwell_s=0),
        lambda: ScanConfig(step_angles_rad=()),
        lambda: MultipathComponent(-1),
        lambda: ClutterConfig(density_per_m=-1),
    ],
)
def test_invalid_physical_model_parameters_fail(factory):
    with pytest.raises(ValueError):
        factory()
