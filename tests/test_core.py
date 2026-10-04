from __future__ import annotations

import errno
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ipcam.backoff import ExponentialBackoff
from ipcam.config import CameraConfig, LowLatencyOptions, StreamRole
from ipcam.errors import FailureKind, StreamOpenError, StreamStalledError, classify_failure
from ipcam.isapi.audio import pcm16_to_alaw, pcm16_to_ulaw
from ipcam.metrics import DriftLagEstimator, RollingWindow
from ipcam.stream.recorder import PacketRing
from ipcam.stream.slot import LatestSlot
from ipcam.vision.ir_state import Illumination, IRStateResolver


def test_backoff_sequence_and_reset() -> None:
    b = ExponentialBackoff(1.0, 2.0, 15.0, jitter=0.0, stable_after_s=10.0)
    assert [b.next_delay() for _ in range(6)] == [1, 2, 4, 8, 15, 15]
    b.mark_healthy(100.0)
    b.mark_healthy(105.0)
    assert b.peek_base() == 15               # not yet stable
    b.mark_healthy(110.5)
    assert b.peek_base() == 1


def test_backoff_jitter_bounds() -> None:
    b = ExponentialBackoff(jitter=0.2, seed=1)
    for expected in (1, 2, 4, 8, 15, 15):
        d = b.next_delay()
        assert expected * 0.8 - 1e-9 <= d <= min(15, expected * 1.2) + 1e-9


def test_slot_drop_oldest() -> None:
    s: LatestSlot[int] = LatestSlot()
    for i in range(5):
        s.publish(i)
    seq, item = s.take()
    assert (seq, item) == (5, 4)
    assert s.overwritten == 4
    got: list[int | None] = []
    t = threading.Thread(target=lambda: got.append(s.wait_newer(5, 2.0)[1]))
    t.start()
    time.sleep(0.05)
    s.publish(99)
    t.join()
    assert got == [99]
    assert s.wait_newer(6, 0.05)[0] == 6     # timeout returns current seq


def test_rtsp_url_escaping_and_redaction() -> None:
    cfg = CameraConfig("10.0.0.5", "admin", "p@ss:w/rd#1")
    assert cfg.rtsp_url(StreamRole.SUB) == "rtsp://admin:p%40ss%3Aw%2Frd%231@10.0.0.5:554/Streaming/Channels/102"
    assert cfg.rtsp_url(StreamRole.MAIN).endswith("/Streaming/Channels/101")
    assert "p@ss" not in cfg.redacted_rtsp_url(StreamRole.SUB)
    assert "p@ss" not in repr(cfg)


def test_ffmpeg_options_mandated_values() -> None:
    o = LowLatencyOptions().to_ffmpeg(libavformat_major=61)
    assert o["rtsp_transport"] == "tcp"
    assert o["fflags"] == "nobuffer+discardcorrupt"
    assert o["flags"] == "low_delay"
    assert (o["max_delay"], o["reorder_queue_size"], o["probesize"], o["analyzeduration"]) == \
        ("500000", "0", "32768", "500000")
    assert "timeout" in o and "stimeout" not in o
    legacy = LowLatencyOptions().to_ffmpeg(libavformat_major=58)
    assert "stimeout" in legacy and "timeout" not in legacy


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"), FailureKind.NETWORK),
        (BrokenPipeError(errno.EPIPE, "Broken pipe"), FailureKind.NETWORK),
        (StreamOpenError("open failed: Server returned 404 Not Found"), FailureKind.NOT_FOUND),
        (StreamOpenError("Server returned 401 Unauthorized"), FailureKind.AUTH),
        (StreamStalledError("no packets"), FailureKind.TIMEOUT),
        (OSError("[Errno 1414092869] Immediate exit requested"), FailureKind.TIMEOUT),
        (None, FailureKind.ENDED),
        # the RTSP port/path inside the URL must not be read as a 5xx/1xx status
        (StreamOpenError("open rtsp://admin:***@10.0.0.5:554/Streaming/Channels/102 failed: "
                         "Connection refused"), FailureKind.NETWORK),
    ],
)
def test_classify_failure(exc: BaseException | None, kind: FailureKind) -> None:
    assert classify_failure(exc) is kind


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="MSVC CRT errno mapping")
def test_classify_windows_crt_errno() -> None:
    exc = StreamOpenError("open rtsp://admin:***@127.0.0.1:554/Streaming/Channels/102 failed: "
                          "[Errno 138] Error number -138 occurred: 'rtsp://admin:***@127.0.0.1:554/x'")
    assert classify_failure(exc) is FailureKind.TIMEOUT
    exc.errno = 107  # type: ignore[attr-defined]
    assert classify_failure(exc) is FailureKind.NETWORK


def test_g711_reference_points() -> None:
    pcm = np.array([0, 32767, -32768, 1000, -1000], dtype=np.int16)
    u = pcm16_to_ulaw(pcm)
    assert u[0] == 0xFF and u[1] == 0x80 and u[2] == 0x00
    assert u[3] ^ u[4] == 0x80                   # symmetric magnitude, opposite sign bit
    a = pcm16_to_alaw(pcm)
    assert a[0] == 0xD5
    assert a[3] ^ a[4] == 0x80


def test_drift_lag_estimator() -> None:
    est = DriftLagEstimator(window_s=10)
    for i in range(50):
        est.update(100.0 + i * 0.04, i * 0.04)
    # 200 ms of queueing appears as 200 ms lag relative to the best delivery
    assert est.update(100.0 + 50 * 0.04 + 0.2, 50 * 0.04) == pytest.approx(200.0, abs=0.5)
    # a PTS reset (reconnect) re-baselines instead of reporting a huge lag
    assert est.update(200.0, 0.0) == pytest.approx(0.0)


def test_rolling_percentiles() -> None:
    w = RollingWindow(100)
    for v in range(1, 101):
        w.add(float(v))
    s = w.summary()
    assert s.p50 == pytest.approx(50.5) and s.p99 == pytest.approx(99.01) and s.maximum == 100


def test_packet_ring_is_gop_aligned() -> None:
    ring = PacketRing(seconds=1.0)
    ns = 1_000_000_000
    pkt = lambda key: SimpleNamespace(is_keyframe=key)
    ring.push(pkt(False), 0)                             # leading P-frame ignored
    for i in range(100):                                 # 25 fps, GOP of 10 (0.4 s)
        ring.push(pkt(i % 10 == 0), int(i * 0.04 * ns) + 1)
    snap = ring.snapshot()
    assert snap[0][1].is_keyframe
    assert 1.0 <= ring.span_s <= 1.0 + 0.4


def test_ir_resolver_hysteresis() -> None:
    r = IRStateResolver(dwell_s=2.0, analyse_every_s=0.0)
    color = np.zeros((64, 64, 3), np.uint8)
    color[..., 2] = 200
    gray = np.full((64, 64, 3), 90, np.uint8)
    assert r.observe_frame(color, now=0.0).mode is Illumination.COLOR
    assert r.observe_frame(gray, now=1.0).mode is Illumination.COLOR    # dwell not satisfied
    assert r.observe_frame(gray, now=2.0).mode is Illumination.COLOR
    assert r.observe_frame(gray, now=3.1).mode is Illumination.IR
