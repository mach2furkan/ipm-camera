"""Phase 5: multi-object tracking (ByteTrack) and spatial security analytics."""

from .bytetrack import ByteTrackConfig, ByteTracker, Track, TrackerOutput, TrackState
from .config import build_rules
from .engine import AnalyticsEngine, AnalyticsResult
from .geometry import point_in_polygon, points_in_polygon, segments_intersect, side
from .heuristics import HeuristicConfig, HeuristicFilter, Verdict
from .kalman import KalmanBoxFilter, KalmanParams
from .matching import iou_matrix, linear_assignment
from .rules import EventKind, PolygonZone, Rule, SecurityEvent, TrackView, Tripwire

__all__ = [
    "AnalyticsEngine",
    "AnalyticsResult",
    "ByteTrackConfig",
    "ByteTracker",
    "EventKind",
    "HeuristicConfig",
    "HeuristicFilter",
    "KalmanBoxFilter",
    "KalmanParams",
    "PolygonZone",
    "Rule",
    "SecurityEvent",
    "Track",
    "TrackState",
    "TrackView",
    "TrackerOutput",
    "Tripwire",
    "Verdict",
    "build_rules",
    "iou_matrix",
    "linear_assignment",
    "point_in_polygon",
    "points_in_polygon",
    "segments_intersect",
    "side",
]
