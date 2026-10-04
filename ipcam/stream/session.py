"""Media sessions: one RTSP connection lifetime, from handshake to teardown.

A session is single-threaded: it is created, iterated and closed on the same worker
thread. Every native resource it opens is wrapped in a :class:`NativeHandle` and closed in
``__exit__`` -- including on exceptions and when the watchdog requests a teardown.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from types import TracebackType
from typing import Any, Protocol

import av

from ..config import DecoderOptions, StreamProfile
from ..errors import StreamError, StreamOpenError
from ..resources import NativeHandle
from .decoders import DecodedPicture, VideoDecoder, create_decoder

log = logging.getLogger(__name__)

try:
    _LAVF_MAJOR: int | None = int(av.library_versions["libavformat"][0])
except Exception:  # noqa: BLE001
    _LAVF_MAJOR = None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    codec: str
    width: int
    height: int
    fps: float | None
    time_base: Fraction | None
    handshake_ms: float
    backend: str


class MediaSession(Protocol):
    supports_remux: bool
    info: SessionInfo

    def __enter__(self) -> MediaSession: ...

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None,
                 tb: TracebackType | None) -> None: ...

    def packets(self) -> Iterator[Any]: ...

    def decode(self, packet: Any) -> list[DecodedPicture]: ...

    def reset_decoder(self) -> None: ...

    @property
    def decoder_name(self) -> str: ...

    @property
    def video_stream(self) -> Any: ...


class PyAVSession:
    """Low-latency RTSP-over-TCP demux with a pluggable decoder."""

    supports_remux = True

    def __init__(self, url: str, profile: StreamProfile, *, redacted_url: str, abort: threading.Event) -> None:
        self._url = url
        self._redacted = redacted_url
        self._profile = profile
        self._abort = abort
        self._stack = ExitStack()
        self._container: NativeHandle[Any] | None = None
        self._stream: Any = None
        self._decoder: NativeHandle[VideoDecoder] | None = None
        self.info: SessionInfo = SessionInfo("?", 0, 0, None, None, 0.0, "pyav")

    def __enter__(self) -> PyAVSession:
        opts = self._profile.latency.to_ffmpeg(libavformat_major=_LAVF_MAJOR)
        t0 = time.perf_counter()
        try:
            container = av.open(
                self._url,
                mode="r",
                options=opts,
                timeout=(self._profile.open_timeout_s, self._profile.read_timeout_s),
            )
        except Exception as exc:
            # FFmpeg error strings embed the URL; never let credentials reach the logs.
            # ``from None`` keeps the raw exception (and its URL) out of chained tracebacks;
            # errno is carried over so failure classification still works.
            msg = str(exc).replace(self._url, self._redacted)
            err = StreamOpenError(f"open {self._redacted} failed: {msg}")
            err.errno = getattr(exc, "errno", None)  # type: ignore[attr-defined]
            raise err from None
        self._container = self._stack.enter_context(NativeHandle(container, _close_container, "av.input"))
        try:
            stream = container.streams.video[0]
        except IndexError:
            self._stack.close()
            raise StreamOpenError(f"{self._redacted}: no video stream in SDP") from None
        self._stream = stream
        cc = stream.codec_context
        rate = stream.average_rate or stream.guessed_rate
        self.info = SessionInfo(
            codec=cc.name,
            width=cc.width,
            height=cc.height,
            fps=float(rate) if rate else None,
            time_base=stream.time_base,
            handshake_ms=(time.perf_counter() - t0) * 1000.0,
            backend="pyav",
        )
        return self

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.reset_decoder()
        self._stream = None
        self._stack.close()
        self._container = None

    @property
    def video_stream(self) -> Any:
        return self._stream

    @property
    def decoder_name(self) -> str:
        return self._decoder.obj.name if self._decoder is not None and not self._decoder.closed else "-"

    def packets(self) -> Iterator[av.Packet]:
        if self._container is None:
            raise StreamError("session not open")
        container = self._container.obj
        for packet in container.demux(self._stream):
            if self._abort.is_set():
                return
            if packet.size == 0:  # demuxer flush packet at EOF
                continue
            yield packet

    def _ensure_decoder(self) -> VideoDecoder:
        if self._decoder is None or self._decoder.closed:
            dec = create_decoder(self._stream, self._profile.decoder)
            self._decoder = NativeHandle(dec, lambda d: d.close(), "decoder")
            log.info("%s decoder: %s", self._redacted, dec.name)
        return self._decoder.obj

    def decode(self, packet: av.Packet) -> list[DecodedPicture]:
        return self._ensure_decoder().decode(packet)

    def reset_decoder(self) -> None:
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None


def _close_container(container: Any) -> None:
    container.close()


# --------------------------------------------------------------------------- GStreamer

@dataclass(slots=True)
class _GstPacket:
    image: Any
    pts_ms: float
    is_keyframe: bool = True
    is_corrupt: bool = False
    size: int = 1

    @property
    def pts(self) -> int:
        return int(self.pts_ms)

    @property
    def time_base(self) -> Fraction:
        return Fraction(1, 1000)


def build_gstreamer_pipeline(url: str, codec: str, platform: str, *, latency_ms: int = 0,
                             tcp_timeout_us: int = 3_000_000) -> str:
    """Hardware-decoding GStreamer pipeline ending in a 1-buffer leaky appsink.

    platform: ``jetson`` (nvv4l2decoder), ``nvcodec`` (desktop NVDEC), ``va`` (new VA
    plugin), ``vaapi`` (legacy gstreamer-vaapi), ``v4l2`` (stateful V4L2 M2M, Rockchip/RPi).
    """
    c = "h265" if codec in ("hevc", "h265") else "h264"
    src = (f'rtspsrc location="{url}" protocols=tcp latency={latency_ms} drop-on-latency=true '
           f"tcp-timeout={tcp_timeout_us} do-rtcp=true ! rtp{c}depay ! {c}parse config-interval=-1")
    decode = {
        "jetson": "nvv4l2decoder enable-max-performance=1 disable-dpb=true ! nvvidconv ! video/x-raw,format=BGRx",
        "nvcodec": f"nv{c}dec ! cudadownload ! video/x-raw,format=NV12",
        "va": f"va{c}dec ! vapostproc ! video/x-raw,format=BGRx",
        "vaapi": f"vaapi{c}dec low-latency=true ! vaapipostproc ! video/x-raw,format=BGRx",
        "v4l2": f"v4l2{c}dec ! video/x-raw",
    }.get(platform)
    if decode is None:
        raise ValueError(f"unknown GStreamer platform {platform!r}")
    sink = "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false emit-signals=false"
    return f"{src} ! {decode} ! {sink}"


class GStreamerSession:
    """OpenCV CAP_GSTREAMER session for SoCs whose HW decoder is only exposed to GStreamer
    (Jetson nvv4l2decoder, VA plugins). Frames arrive decoded, so packet taps / remux are
    unavailable on this backend."""

    supports_remux = False

    def __init__(self, pipeline: str, *, redacted: str, abort: threading.Event,
                 decoder_opts: DecoderOptions | None = None) -> None:
        self._pipeline = pipeline
        self._redacted = redacted
        self._abort = abort
        self._cap: NativeHandle[Any] | None = None
        self._size = decoder_opts.output_size if decoder_opts else None
        self.info = SessionInfo("?", 0, 0, None, Fraction(1, 1000), 0.0, "gstreamer")

    def __enter__(self) -> GStreamerSession:
        import cv2

        t0 = time.perf_counter()
        cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        handle = NativeHandle(cap, lambda c: c.release(), "cv2.capture")
        if not cap.isOpened():
            handle.close()
            raise StreamOpenError(f"GStreamer pipeline failed to open for {self._redacted}")
        self._cap = handle
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        self.info = SessionInfo("gst", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, Fraction(1, 1000),
                                (time.perf_counter() - t0) * 1000.0, "gstreamer")
        return self

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None,
                 tb: TracebackType | None) -> None:
        if self._cap is not None:
            self._cap.close()
            self._cap = None

    @property
    def video_stream(self) -> Any:
        return None

    @property
    def decoder_name(self) -> str:
        return "gstreamer"

    def packets(self) -> Iterator[_GstPacket]:
        import cv2

        assert self._cap is not None
        cap = self._cap.obj
        while not self._abort.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                raise StreamError(f"GStreamer read failed for {self._redacted}")
            if self._size is not None and (img.shape[1], img.shape[0]) != self._size:
                img = cv2.resize(img, self._size, interpolation=cv2.INTER_LINEAR)
            yield _GstPacket(img, cap.get(cv2.CAP_PROP_POS_MSEC))

    def decode(self, packet: _GstPacket) -> list[DecodedPicture]:
        h, w = packet.image.shape[:2]
        return [DecodedPicture(pts=packet.pts, time_base=packet.time_base, keyframe=True,
                               width=w, height=h, device="cpu", image=packet.image)]

    def reset_decoder(self) -> None:
        return None
