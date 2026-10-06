"""Small medium-scaled examples, not claims of calibrated real-world conditions."""

from echorin.config import SensorConfig, SensorMode
from echorin.environment.clutter import ClutterConfig
from echorin.environment.config import EnvironmentConfig
from echorin.environment.interference import NarrowbandInterference
from echorin.propagation.multipath import MultipathComponent


def environment_preset(name: str, sensor: SensorConfig) -> EnvironmentConfig:
    """Create optional effects with similar workloads in Radar and Sonar."""
    if name == "Clutter / Reverberation":
        return EnvironmentConfig(
            clutter=(
                ClutterConfig(
                    kind="reverberation"
                    if sensor.mode is SensorMode.SONAR
                    else "clutter",
                    density_per_m=20 / sensor.max_range_m,
                    reflectivity_scale=0.15,
                    decay_range_m=sensor.max_range_m / 2,
                ),
            )
        )
    if name == "Multipath":
        return EnvironmentConfig(
            multipath=(MultipathComponent(sensor.max_range_m / 4, 0.4, 0.3, 0.05),)
        )
    if name == "Interference":
        return EnvironmentConfig(
            interference=(
                NarrowbandInterference(
                    sensor.carrier_frequency_hz, 0.05, arrival_angle_rad=0.3
                ),
            )
        )
    if name == "Baseline":
        return EnvironmentConfig()
    raise ValueError("unknown environment preset")
