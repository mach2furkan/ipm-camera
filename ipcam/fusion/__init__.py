"""Phase 7: multi-camera fusion, distributed Re-ID, master-slave PTZ and the C2 picture."""

from .global_tracker import EntityStatus, FusionConfig, FusionEvent, GlobalEntity, GlobalTrackManager, LocalObservation
from .homography import CameraCalibration, fit_homography, fit_homography_ransac, project
from .ptz import (
    EngagementState,
    ISAPIPTZ,
    PTZMount,
    PTZPose,
    ServoGains,
    SlewToCueEngagement,
    VisualServo,
    aim,
    lead_aim,
)
from .reid import EmbeddingGallery, OnnxReIDExtractor, ReIDGate, ReIDGateConfig, l2_normalize
from .topology import CameraTopology, TransitionModel
from .vector_index import FlatIndex, HNSWIndex, create_index
from .worldkf import WorldKF

__all__ = [
    "CameraCalibration",
    "CameraTopology",
    "EmbeddingGallery",
    "EngagementState",
    "EntityStatus",
    "FlatIndex",
    "FusionConfig",
    "FusionEvent",
    "GlobalEntity",
    "GlobalTrackManager",
    "HNSWIndex",
    "ISAPIPTZ",
    "LocalObservation",
    "OnnxReIDExtractor",
    "PTZMount",
    "PTZPose",
    "ReIDGate",
    "ReIDGateConfig",
    "ServoGains",
    "SlewToCueEngagement",
    "TransitionModel",
    "VisualServo",
    "WorldKF",
    "aim",
    "create_index",
    "fit_homography",
    "fit_homography_ransac",
    "l2_normalize",
    "lead_aim",
    "project",
]
