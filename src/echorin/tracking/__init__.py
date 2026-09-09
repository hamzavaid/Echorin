"""Kalman filtering, data association, and track lifecycle management."""

from echorin.tracking.association import AssociationResult, associate_nearest_neighbor
from echorin.tracking.kalman import ConstantVelocityKalmanFilter
from echorin.tracking.tracker import MultiTargetTracker, TrackerConfig

__all__ = [
    "AssociationResult",
    "ConstantVelocityKalmanFilter",
    "MultiTargetTracker",
    "TrackerConfig",
    "associate_nearest_neighbor",
]
