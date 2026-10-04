"""Data-plane tests driven by a synthetic H.264/MPEG-4 clip (no camera required)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import av
import pytest

from ipcam.backoff import ExponentialBackoff
from ipcam.config import CameraConfig, DecoderOptions, HWAccel, StreamRole
from ipcam.resources import NativeHandle, live_handles
from ipcam.stream.decoders import PyAVDecoder
from ipcam.stream.reader import RTSPStreamReader
from ipcam.stream.recorder import EvidenceRecorder, PacketRing
from ipcam.stream.session import SessionInfo
from ipcam.stream.watchdog import LinkState, StreamWatchdog

from .conftest import FPS, FRAMES, GOP, SIZE

SW = DecoderOptions(hwaccel=HWAccel.NONE)


class ReplaySession:
    """File-backed MediaSession with fault injection."""

    supports_remux = True

    def __init__(self, path: Path, abort: threading.Event, *, skip: int = 0, stall_after: int | None = None,
                 ignore_abort: bool = False, realtime: bool = True, opts: DecoderOptions = SW) -> None:
        self.path, self.abort = path, abort
        self.skip, self.stall_after, self.ignore_abort, self.realtime = skip, stall_after, ignore_abort, realtime
        self.opts = opts
        self._decoder: PyAVDecoder | None = None

    def __enter__(self) -> ReplaySession:
        self._h = NativeHandle(av.open(str(self.path)), lambda c: c.close(), "av.input")
        self.stream = self._h.obj.streams.video[0]
        self.info = SessionInfo(self.stream.codec_context.name, SIZE[0], SIZE[1], FPS, self.stream.time_base, 1.0,
                                "replay")
        return self

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None) -> None:
        self.reset_decoder()
        self._h.close()

    @property
    def video_stream(self) -> Any:
        return self.stream

    @property
    def decoder_name(self) -> str:
        return "replay"

    def packets(self) -> Any:
        for i, p in enumerate(self._h.obj.demux(self.stream)):
            if p.size == 0 or i < self.skip:
                continue
            if self.stall_after is not None and i >= self.stall_after:
                end = time.monotonic() + (1.5 if self.ignore_abort else 30)
                while time.monotonic() < end and (self.ignore_abort or not self.abort.is_set()):
                    time.sleep(0.01)
                return
            if self.abort.is_set():
                return
            yield p
            if self.realtime:
                time.sleep(1 / FPS / 4)

    def decode(self, p: Any) -> Any:
        if self._decoder is None:
            self._decoder = PyAVDecoder(self.stream, self.opts, None)
        return self._decoder.decode(p)

    def reset_decoder(self) -> None:
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None


CFG = CameraConfig("127.0.0.1", "admin", "secret")


def fast_watchdog(reader: RTSPStreamReader, **kw: Any) -> StreamWatchdog:
    return StreamWatchdog(reader, stall_timeout_s=kw.pop("stall", 0.5), keyframe_wait_s=0.6,
                          backoff=ExponentialBackoff(0.05, 2.0, 0.2, jitter=0.0),
                          hard_kill_grace_s=kw.pop("grace", 2.0), **kw)


def assert_no_leaks(timeout: float = 3.0) -> None:
    """Workers release handles on their own thread; allow them a moment under CI load."""
    deadline = time.monotonic() + timeout
    while live_handles() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not live_handles(), live_handles()


def collect(reader: RTSPStreamReader, n: int, timeout: float = 10.0) -> list[Any]:
    out, seq, end = [], 0, time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        res = reader.wait_frame(seq, timeout=0.5)
        if res is not None:
            seq = res.seq
            out.append(res)
    return out


def test_pyav_decoder_lazy_and_resize(clip: Path) -> None:
    with av.open(str(clip)) as c:
        s = c.streams.video[0]
        dec = PyAVDecoder(s, DecoderOptions(hwaccel=HWAccel.NONE, output_size=(160, 90)), None)
        pics = [pic for p in c.demux(s) if p.size for pic in dec.decode(p)]
        dec.close()
    assert len(pics) >= FRAMES - 2
    assert sum(p.keyframe for p in pics) == pytest.approx(FRAMES / GOP, abs=1)
    img = pics[-1].materialize()
    assert img.shape == (90, 160, 3)


def test_reader_publishes_latest_with_gop_positions(clip: Path) -> None:
    reader = RTSPStreamReader(CFG, StreamRole.SUB, session_factory=lambda ab: ReplaySession(clip, ab))
    with fast_watchdog(reader) as wd:
        assert wd.wait_streaming(5)
        got = collect(reader, 20)
        assert len(got) == 20
        assert all(r.image.shape == (SIZE[1], SIZE[0], 3) for r in got)
        assert all(0 <= r.frame.frames_since_keyframe < GOP for r in got)
        lat = got[-1].latency
        assert lat.decode_ms >= 0 and lat.queue_ms >= 0
        stats = reader.stats()
        # frames the consumer never read were never converted
        assert stats.frames_published >= len(got)
    assert_no_leaks()


def test_keyframe_gating_after_mid_gop_join(clip: Path) -> None:
    reader = RTSPStreamReader(CFG, StreamRole.SUB,
                              session_factory=lambda ab: ReplaySession(clip, ab, skip=3, realtime=False))
    with fast_watchdog(reader):
        got = collect(reader, 1)
        assert got and got[0].frame.keyframe
        time.sleep(0.2)
    assert reader.stats().gated_packets >= GOP - 3


def test_stall_triggers_teardown_and_reconnect(clip: Path) -> None:
    events: list[LinkState] = []
    reader = RTSPStreamReader(CFG, StreamRole.SUB,
                              session_factory=lambda ab: ReplaySession(clip, ab, stall_after=15))
    wd = fast_watchdog(reader, on_event=lambda e: events.append(e.state))
    with wd:
        deadline = time.monotonic() + 8
        while (wd.stats().stalls < 2 or wd.stats().reconnects < 2) and time.monotonic() < deadline:
            time.sleep(0.05)
    st = wd.stats()
    assert st.stalls >= 2 and st.reconnects >= 2
    assert LinkState.STALLED in events and LinkState.BACKOFF in events
    assert reader.stats().sessions >= 2
    assert st.zombies == 0
    assert_no_leaks()


def test_unresponsive_worker_is_quarantined(clip: Path) -> None:
    reader = RTSPStreamReader(
        CFG, StreamRole.SUB,
        session_factory=lambda ab: ReplaySession(clip, ab, stall_after=12, ignore_abort=True))
    wd = fast_watchdog(reader, grace=0.3)
    with wd:
        deadline = time.monotonic() + 8
        while wd.stats().zombies < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        gen_after = reader.generation
    assert wd.stats().zombies >= 1
    assert gen_after > 1
    # The quarantined worker eventually unblocks and still releases its native handles.
    assert_no_leaks(timeout=5.0)


def test_decode_toggle_instant_on(clip: Path) -> None:
    reader = RTSPStreamReader(CFG, StreamRole.MAIN, decode=False,
                              session_factory=lambda ab: ReplaySession(clip, ab))
    with fast_watchdog(reader) as wd:
        assert wd.wait_streaming(5)
        time.sleep(0.4)                        # mid-GOP, packets only
        assert reader.get_latest_frame() is None
        reader.acquire_decode()
        t0 = time.monotonic()
        got = collect(reader, 1, timeout=3)
        reader.release_decode()
    assert got, "no frame after enabling decode"
    # Cached-GOP replay delivers a frame well before the next keyframe would arrive.
    assert time.monotonic() - t0 < 2.0


class TapReader:
    """Stands in for RTSPStreamReader in recorder tests."""

    def __init__(self) -> None:
        self.taps: list[Any] = []

    def add_packet_tap(self, t: Any) -> None:
        self.taps.append(t)

    def remove_packet_tap(self, t: Any) -> None:
        self.taps.remove(t)


def test_evidence_recorder_remux_with_preroll(clip: Path, tmp_path: Path) -> None:
    ring = PacketRing(seconds=0.5)
    fake = TapReader()
    with av.open(str(clip)) as c:
        s = c.streams.video[0]
        packets = [p for p in c.demux(s) if p.size]
        t = time.perf_counter_ns()
        step = int(1e9 / FPS)
        for i, p in enumerate(packets[:30]):           # history before the trigger
            ring.push(p, t - (30 - i) * step)
        rec = EvidenceRecorder(fake, tmp_path / "ev.mkv", duration_s=2.0, preroll=ring)  # type: ignore[arg-type]
        fut = rec.start()
        for p in packets[30:]:
            for tap in list(fake.taps):
                tap(p, time.perf_counter_ns(), None)
        rec.cancel()
        result = fut.result(timeout=10)
    assert result.ok, result.error
    assert result.preroll_s > 0
    with av.open(str(result.path)) as out:
        frames = sum(1 for _ in out.decode(video=0))
    assert frames == result.packets
    assert frames >= len(packets) - 30
    assert not fake.taps
