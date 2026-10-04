"""Hikvision IP camera streaming and hardware integration layer (Phase 1)."""

from .backoff import ExponentialBackoff
from .bus import EventBus
from .config import CameraConfig, DecoderOptions, HWAccel, LowLatencyOptions, StreamProfile, StreamRole
from .errors import (
    AuthenticationError,
    DecoderError,
    FailureKind,
    IPCamError,
    ISAPIError,
    NotSupportedError,
    StreamError,
    StreamOpenError,
    StreamStalledError,
    classify_failure,
)
from .isapi import AlertEvent, AlertStreamListener, HikvisionISAPIClient, IRCutFilterState, IRCutMode
from .node import CameraNode, ViolationResponse
from .resources import live_handles
from .stream import (
    DualStreamManager,
    EvidenceRecorder,
    Frame,
    FrameResult,
    LatencyBreakdown,
    RTSPStreamReader,
    StreamWatchdog,
)
from .vision import Illumination, IlluminationState, IRStateResolver

__version__ = "0.1.0"

__all__ = [
    "AlertEvent",
    "AlertStreamListener",
    "AuthenticationError",
    "CameraConfig",
    "CameraNode",
    "DecoderError",
    "DecoderOptions",
    "DualStreamManager",
    "EventBus",
    "EvidenceRecorder",
    "ExponentialBackoff",
    "FailureKind",
    "Frame",
    "FrameResult",
    "HWAccel",
    "HikvisionISAPIClient",
    "IPCamError",
    "IRCutFilterState",
    "IRCutMode",
    "IRStateResolver",
    "ISAPIError",
    "Illumination",
    "IlluminationState",
    "LatencyBreakdown",
    "LowLatencyOptions",
    "NotSupportedError",
    "RTSPStreamReader",
    "StreamError",
    "StreamOpenError",
    "StreamProfile",
    "StreamRole",
    "StreamStalledError",
    "StreamWatchdog",
    "ViolationResponse",
    "classify_failure",
    "live_handles",
]
