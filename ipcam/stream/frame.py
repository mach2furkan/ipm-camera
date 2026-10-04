"""Decoded frame container with lazy materialisation and latency accounting."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import StreamRole


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Where a frame's delay was spent, in milliseconds.

    network_lag_ms  buffering ahead of the demuxer relative to the best observed delivery
                    (PTS-drift estimate; grows when the camera, network or socket queues up).
    decode_ms       compressed packet demuxed -> picture emitted by the decoder.
    convert_ms      lazy pixel conversion / host download, paid by the first reader.
    queue_ms        picture published -> handed to the consumer.
    """

    network_lag_ms: float
    decode_ms: float
    convert_ms: float
    queue_ms: float

    @property
    def pipeline_ms(self) -> float:
        """Host-side latency (arrival -> consumer), fully measured on one clock."""
        return _nansum(self.decode_ms, self.convert_ms, self.queue_ms)

    @property
    def total_ms(self) -> float:
        return _nansum(self.network_lag_ms, self.decode_ms, self.convert_ms, self.queue_ms)


def _nansum(*values: float) -> float:
    return sum(v for v in values if not math.isnan(v))


class Frame:
    """A decoded picture. Pixel data is produced on first access to :attr:`image`.

    Under the drop-oldest policy most decoded pictures are overwritten before anyone
    reads them; deferring the YUV->BGR conversion (and the GPU->host download for FFmpeg
    hwaccel) to the reader means superseded frames never pay for it.
    """

    __slots__ = (
        "_convert_ms",
        "_image",
        "_lock",
        "_materialize",
        "_release",
        "arrival_ns",
        "decoded_ns",
        "device",
        "frames_since_keyframe",
        "height",
        "keyframe",
        "network_lag_ms",
        "pts",
        "pts_time",
        "published_ns",
        "role",
        "seq",
        "wall_time",
        "width",
    )

    def __init__(
        self,
        *,
        seq: int,
        role: StreamRole,
        pts: int | None,
        pts_time: float | None,
        keyframe: bool,
        frames_since_keyframe: int,
        width: int,
        height: int,
        device: str,
        arrival_ns: int,
        decoded_ns: int,
        wall_time: float,
        network_lag_ms: float,
        materialize: Callable[[], Any] | None = None,
        image: Any = None,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.seq = seq
        self.role = role
        self.pts = pts
        self.pts_time = pts_time
        self.keyframe = keyframe
        self.frames_since_keyframe = frames_since_keyframe
        self.width = width
        self.height = height
        self.device = device
        self.arrival_ns = arrival_ns
        self.decoded_ns = decoded_ns
        self.published_ns = decoded_ns
        self.wall_time = wall_time
        self.network_lag_ms = network_lag_ms
        self._materialize = materialize
        self._image = image
        self._convert_ms = 0.0 if image is not None else math.nan
        self._lock = threading.Lock()
        self._release = release

    @property
    def image(self) -> Any:
        """``numpy.ndarray`` (H, W, 3) on CPU, or ``torch.Tensor`` (3, H, W) on CUDA."""
        img = self._image
        if img is not None:
            return img
        with self._lock:
            if self._image is None:
                fn = self._materialize
                if fn is None:
                    raise RuntimeError("frame has been released")
                t0 = time.perf_counter_ns()
                self._image = fn()
                self._convert_ms = (time.perf_counter_ns() - t0) / 1e6
                self._materialize = None  # drop the reference to the native AVFrame
            return self._image

    @property
    def is_materialized(self) -> bool:
        return self._image is not None

    def release(self) -> None:
        """Drop pixel buffers and native references early (optional; GC also works)."""
        with self._lock:
            self._materialize = None
            self._image = None
            rel, self._release = self._release, None
        if rel is not None:
            rel()

    def latency(self, now_ns: int | None = None) -> LatencyBreakdown:
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        return LatencyBreakdown(
            network_lag_ms=self.network_lag_ms,
            decode_ms=(self.decoded_ns - self.arrival_ns) / 1e6,
            convert_ms=self._convert_ms,
            queue_ms=max(0.0, (now_ns - self.published_ns) / 1e6),
        )

    def age_ms(self, now_ns: int | None = None) -> float:
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        return (now_ns - self.arrival_ns) / 1e6

    def __repr__(self) -> str:
        return (f"Frame(seq={self.seq}, role={self.role.value}, {self.width}x{self.height}, "
                f"device={self.device}, key={self.keyframe}, gop_pos={self.frames_since_keyframe})")


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Return type of ``RTSPStreamReader.get_latest_frame``."""

    frame: Frame
    timestamp: float              # host wall-clock (epoch s) when the frame's data arrived
    latency: LatencyBreakdown
    seq: int
    skipped: int                  # frames superseded since the caller's previous read

    @property
    def image(self) -> Any:
        return self.frame.image

    @property
    def latency_ms(self) -> float:
        return self.latency.total_ms
