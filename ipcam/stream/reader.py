"""RTSPStreamReader: background capture with drop-oldest delivery and keyframe gating.

Data path per packet (single worker thread, owned by :class:`StreamWatchdog`)::

    demux (TCP, nobuffer) ──► packet taps (evidence ring / recorder; never block)
                         └──► keyframe gate ──► decoder (HW) ──► Frame ──► LatestSlot (size 1)
                                                                              │
                         get_latest_frame() / wait_frame()  ◄─────────────────┘

Guarantees
* The consumer always receives the newest decodable picture; older ones are overwritten.
* No picture is published unless its reference chain back to an IDR is intact: after a
  (re)connect, a corrupt packet or a decoder error the gate drops everything until the
  next keyframe, so the model never sees grey/smeared macroblocks.
* When decoding is switched on mid-stream (main stream for SAHI), the cached GOP is
  replayed so the first frame is available immediately rather than one GOP later.
* Pictures from an abandoned (zombie) session can never reach the slot: every session
  carries a generation number that must match the reader's current one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import CameraConfig, StreamRole
from ..errors import DecoderError
from ..metrics import DriftLagEstimator, RateMeter, RollingWindow
from .decoders import DecodedPicture
from .frame import Frame, FrameResult
from .session import MediaSession, PyAVSession, SessionInfo
from .slot import LatestSlot

log = logging.getLogger(__name__)

PacketTap = Callable[[Any, int, SessionInfo], None]
SessionFactory = Callable[[threading.Event], MediaSession]

_MAX_CONSECUTIVE_DECODE_ERRORS = 25
_MAX_GOP_CACHE = 600  # packets; ~20 s at 30 fps, bounds memory if keyframes never arrive


class Phase(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    AWAITING_KEYFRAME = "awaiting_keyframe"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class Heartbeat:
    phase: Phase
    generation: int
    session_started_ns: int
    last_packet_ns: int
    last_frame_ns: int
    decode_started_ns: int
    decode_enabled: bool


@dataclass(frozen=True, slots=True)
class ReaderStats:
    role: StreamRole
    phase: Phase
    decoder: str
    sessions: int
    packets: int
    bytes: int
    frames_decoded: int
    frames_published: int
    frames_overwritten: int
    gated_packets: int
    decode_errors: int
    corrupt_packets: int
    packet_rate: float
    decode_rate: float
    decode_ms_p50: float
    decode_ms_p95: float
    info: SessionInfo | None


class RTSPStreamReader:
    def __init__(
        self,
        camera: CameraConfig,
        role: StreamRole = StreamRole.SUB,
        *,
        decode: bool = True,
        session_factory: SessionFactory | None = None,
        name: str | None = None,
    ) -> None:
        self.camera = camera
        self.role = role
        self.profile = camera.profile(role)
        self.name = name or f"{camera.label}/{role.value}"
        self._url = camera.rtsp_url(role)
        self.redacted_url = camera.redacted_rtsp_url(role)
        self._session_factory = session_factory or self._default_session
        self._slot: LatestSlot[Frame] = LatestSlot()

        self._lock = threading.Lock()
        self._decode_refs = 1 if decode else 0
        self._taps: tuple[PacketTap, ...] = ()
        self._connect_callbacks: tuple[Callable[[SessionInfo], None], ...] = ()

        self._generation = 0
        self._phase = Phase.IDLE
        self._session_started_ns = 0
        self._last_packet_ns = 0
        self._last_frame_ns = 0
        self._decode_started_ns = 0
        self._info: SessionInfo | None = None
        self._decoder_name = "-"
        self._last_taken_seq = 0

        self._lag = DriftLagEstimator()
        self._packet_rate = RateMeter()
        self._decode_rate = RateMeter()
        self._decode_ms = RollingWindow(512)
        self._sessions = 0
        self._packets = 0
        self._bytes = 0
        self._frames_decoded = 0
        self._gated = 0
        self._decode_errors = 0
        self._corrupt = 0

    # ------------------------------------------------------------------ public API

    def get_latest_frame(self, max_age_ms: float | None = None) -> FrameResult | None:
        """Newest decoded frame with timestamp and latency breakdown (non-blocking)."""
        seq, frame = self._slot.take()
        return self._result(seq, frame, max_age_ms)

    def wait_frame(self, after_seq: int = 0, timeout: float | None = 1.0,
                   max_age_ms: float | None = None) -> FrameResult | None:
        """Block until a frame newer than ``after_seq`` is available."""
        seq, frame = self._slot.wait_newer(after_seq, timeout)
        if seq <= after_seq:
            return None
        return self._result(seq, frame, max_age_ms)

    def _result(self, seq: int, frame: Frame | None, max_age_ms: float | None) -> FrameResult | None:
        if frame is None:
            return None
        now = time.perf_counter_ns()
        if max_age_ms is not None and frame.age_ms(now) > max_age_ms:
            return None
        with self._lock:
            skipped = max(0, seq - self._last_taken_seq - 1)
            self._last_taken_seq = seq
        return FrameResult(frame=frame, timestamp=frame.wall_time, latency=frame.latency(now),
                           seq=seq, skipped=skipped)

    @property
    def decode_enabled(self) -> bool:
        return self._decode_refs > 0

    def acquire_decode(self) -> None:
        """Reference-counted decode enable (e.g. a SAHI lease on the main stream)."""
        with self._lock:
            self._decode_refs += 1

    def release_decode(self) -> None:
        with self._lock:
            self._decode_refs = max(0, self._decode_refs - 1)

    def add_packet_tap(self, tap: PacketTap) -> None:
        """Register a non-blocking callback receiving every compressed packet."""
        with self._lock:
            self._taps = (*self._taps, tap)

    def remove_packet_tap(self, tap: PacketTap) -> None:
        with self._lock:
            self._taps = tuple(t for t in self._taps if t is not tap)

    def on_connected(self, callback: Callable[[SessionInfo], None]) -> None:
        """Called on the worker thread right after each successful handshake."""
        with self._lock:
            self._connect_callbacks = (*self._connect_callbacks, callback)

    @property
    def info(self) -> SessionInfo | None:
        return self._info

    @property
    def generation(self) -> int:
        return self._generation

    def heartbeat(self) -> Heartbeat:
        return Heartbeat(self._phase, self._generation, self._session_started_ns, self._last_packet_ns,
                         self._last_frame_ns, self._decode_started_ns, self.decode_enabled)

    def stats(self) -> ReaderStats:
        d = self._decode_ms.summary()
        return ReaderStats(
            role=self.role, phase=self._phase, decoder=self._decoder_name, sessions=self._sessions,
            packets=self._packets, bytes=self._bytes, frames_decoded=self._frames_decoded,
            frames_published=self._slot.published, frames_overwritten=self._slot.overwritten,
            gated_packets=self._gated, decode_errors=self._decode_errors, corrupt_packets=self._corrupt,
            packet_rate=self._packet_rate.rate(), decode_rate=self._decode_rate.rate(),
            decode_ms_p50=d.p50, decode_ms_p95=d.p95, info=self._info,
        )

    def sanitize(self, text: str) -> str:
        return text.replace(self._url, self.redacted_url)

    # ------------------------------------------------------------------ watchdog interface

    def next_generation(self) -> int:
        """Invalidate the current session (called by the watchdog before (re)starting)."""
        with self._lock:
            self._generation += 1
            return self._generation

    def mark_idle(self) -> None:
        self._phase = Phase.IDLE
        self._slot.clear()

    def close(self) -> None:
        self.next_generation()
        self._phase = Phase.IDLE
        self._slot.close()

    def run_session(self, abort: threading.Event, generation: int) -> None:
        """Run one RTSP session to completion on the calling thread.

        Returns on clean EOF or abort; raises on any failure. All native resources are
        released before this returns or raises.
        """
        self._phase = Phase.CONNECTING
        self._session_started_ns = time.perf_counter_ns()
        self._slot.reopen()
        with self._session_factory(abort) as session:
            if generation != self._generation:
                return
            self._sessions += 1
            self._info = session.info
            self._lag.reset()
            self._last_packet_ns = time.perf_counter_ns()
            log.info("%s connected: %s %dx%d @ %s fps, handshake %.0f ms", self.name, session.info.codec,
                     session.info.width, session.info.height,
                     f"{session.info.fps:.2f}" if session.info.fps else "?", session.info.handshake_ms)
            for cb in self._connect_callbacks:
                try:
                    cb(session.info)
                except Exception:
                    log.exception("%s on_connected callback failed", self.name)
            try:
                self._pump(session, abort, generation)
            finally:
                self._decoder_name = session.decoder_name
                if generation == self._generation:
                    self._slot.clear()

    # ------------------------------------------------------------------ worker loop

    def _default_session(self, abort: threading.Event) -> MediaSession:
        return PyAVSession(self._url, self.profile, redacted_url=self.redacted_url, abort=abort)

    def _pump(self, session: MediaSession, abort: threading.Event, generation: int) -> None:
        awaiting_key = True
        decoding = False
        gop_pos = -1
        consecutive_errors = 0
        gop_cache: list[Any] = []
        self._phase = Phase.AWAITING_KEYFRAME if self.decode_enabled else Phase.STREAMING
        info = session.info

        for pkt in session.packets():
            if abort.is_set() or generation != self._generation:
                return
            arrival_ns = time.perf_counter_ns()
            self._last_packet_ns = arrival_ns
            self._packets += 1
            self._bytes += pkt.size
            self._packet_rate.tick(arrival_ns / 1e9)
            is_key = bool(pkt.is_keyframe)
            pts_time = _pts_seconds(pkt)
            lag_ms = self._lag.update(arrival_ns / 1e9, pts_time)

            for tap in self._taps:
                try:
                    tap(pkt, arrival_ns, info)
                except Exception:
                    log.exception("%s packet tap failed", self.name)

            if not self.decode_enabled:
                if decoding:
                    session.reset_decoder()
                    decoding = False
                    self._slot.clear()
                self._phase = Phase.STREAMING
                if session.supports_remux:
                    if is_key:
                        gop_cache = [pkt]
                    elif gop_cache and len(gop_cache) < _MAX_GOP_CACHE:
                        gop_cache.append(pkt)
                continue

            if not decoding:
                decoding = True
                awaiting_key = True
                self._decode_started_ns = arrival_ns
                self._phase = Phase.AWAITING_KEYFRAME
                if gop_cache and gop_cache[0].is_keyframe:
                    # Instant-on: rebuild decoder state from the cached GOP and publish
                    # only its last picture, then continue with the live packet.
                    gop_pos = self._replay_gop(session, gop_cache, generation, lag_ms)
                    awaiting_key = gop_pos < 0
                gop_cache = []

            if awaiting_key:
                if not is_key:
                    self._gated += 1
                    self._phase = Phase.AWAITING_KEYFRAME
                    continue
                awaiting_key = False

            if getattr(pkt, "is_corrupt", False):
                self._corrupt += 1
                session.reset_decoder()
                awaiting_key = True
                continue

            try:
                pictures = session.decode(pkt)
                consecutive_errors = 0
            except DecoderError:
                raise
            except Exception as exc:  # noqa: BLE001 - FFmpeg raises InvalidDataError & co.
                self._decode_errors += 1
                consecutive_errors += 1
                log.debug("%s decode error: %s", self.name, self.sanitize(str(exc)))
                session.reset_decoder()
                awaiting_key = True
                if consecutive_errors >= _MAX_CONSECUTIVE_DECODE_ERRORS:
                    raise DecoderError(f"{consecutive_errors} consecutive decode errors") from None
                continue

            for pic in pictures:
                if pic.corrupt:
                    self._corrupt += 1
                    session.reset_decoder()
                    awaiting_key = True
                    break
                gop_pos = 0 if pic.keyframe else gop_pos + 1
                self._publish(pic, arrival_ns, lag_ms, gop_pos, generation)
            self._decoder_name = session.decoder_name

    def _replay_gop(self, session: MediaSession, packets: list[Any], generation: int, lag_ms: float) -> int:
        last: DecodedPicture | None = None
        pos = -1
        t0 = time.perf_counter_ns()
        try:
            for p in packets:
                for pic in session.decode(p):
                    if pic.corrupt:
                        session.reset_decoder()
                        return -1
                    pos = 0 if pic.keyframe else pos + 1
                    last = pic
        except Exception:  # noqa: BLE001
            session.reset_decoder()
            return -1
        if last is not None:
            self._publish(last, t0, lag_ms, pos, generation)
            log.debug("%s instant-on replayed %d cached packets", self.name, len(packets))
        return pos

    def _publish(self, pic: DecodedPicture, arrival_ns: int, lag_ms: float, gop_pos: int,
                 generation: int) -> None:
        if generation != self._generation:
            return
        decoded_ns = time.perf_counter_ns()
        self._frames_decoded += 1
        self._decode_rate.tick(decoded_ns / 1e9)
        self._decode_ms.add((decoded_ns - arrival_ns) / 1e6)
        frame = Frame(
            seq=self._slot.seq + 1,
            role=self.role,
            pts=pic.pts,
            pts_time=pic.pts_time,
            keyframe=pic.keyframe,
            frames_since_keyframe=gop_pos,
            width=pic.width,
            height=pic.height,
            device=pic.device,
            arrival_ns=arrival_ns,
            decoded_ns=decoded_ns,
            wall_time=time.time() - (decoded_ns - arrival_ns) / 1e9,
            network_lag_ms=lag_ms,
            materialize=pic.materialize,
            image=pic.image,
        )
        self._slot.publish(frame)
        self._last_frame_ns = decoded_ns
        self._phase = Phase.STREAMING


def _pts_seconds(pkt: Any) -> float | None:
    pts = getattr(pkt, "pts", None)
    tb = getattr(pkt, "time_base", None)
    if pts is None or tb is None:
        return None
    return float(pts * tb)
