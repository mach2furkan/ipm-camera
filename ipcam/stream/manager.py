"""DualStreamManager: always-on sub stream for inference, on-demand main stream.

* Sub stream (x02): connected and decoding permanently; feeds the inference queue.
* Main stream (x01): consumed through *leases*. A SAHI tiler takes a decode lease and
  gets full-resolution frames; an evidence recording takes a packet-only lease (remux,
  no decode). When the last lease ends the main session lingers ``main_idle_linger_s``
  (bursty alarms re-use the warm session) and is then torn down to save camera
  sessions, bandwidth and decoder memory.
* With ``main.preroll_s > 0`` the main stream stays connected in packet-only mode so the
  GOP ring always holds pre-event footage; decoding still only runs under a lease.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

from ..config import CameraConfig, StreamRole
from .reader import RTSPStreamReader, SessionFactory
from .recorder import EvidenceRecorder, PacketRing, RecordingResult, evidence_path
from .session import SessionInfo
from .watchdog import StreamWatchdog, WatchdogEvent

log = logging.getLogger(__name__)


class DualStreamManager:
    def __init__(
        self,
        camera: CameraConfig,
        *,
        evidence_dir: str | Path = "evidence",
        on_watchdog_event: Callable[[StreamRole, WatchdogEvent], None] | None = None,
        sub_session_factory: SessionFactory | None = None,
        main_session_factory: SessionFactory | None = None,
    ) -> None:
        self.camera = camera
        self.evidence_dir = Path(evidence_dir)
        cb = on_watchdog_event

        self.sub = RTSPStreamReader(camera, StreamRole.SUB, decode=True, session_factory=sub_session_factory)
        self.sub_watchdog = StreamWatchdog(
            self.sub, on_event=(lambda e: cb(StreamRole.SUB, e)) if cb else None)

        self.main = RTSPStreamReader(camera, StreamRole.MAIN, decode=False, session_factory=main_session_factory)
        self.main_watchdog = StreamWatchdog(
            self.main, on_event=(lambda e: cb(StreamRole.MAIN, e)) if cb else None)

        self.preroll: PacketRing | None = None
        if camera.main.preroll_s > 0:
            self.preroll = PacketRing(camera.main.preroll_s)
            self.main.add_packet_tap(self.preroll.push)
            # Timestamps restart with every RTSP session; never mix GOPs across sessions.
            self.main.on_connected(lambda _info: self.preroll.clear() if self.preroll else None)

        self._lock = threading.Lock()
        self._leases = 0
        self._main_running = False
        self._linger: threading.Timer | None = None
        self._recorders: set[EvidenceRecorder] = set()
        self._started = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> DualStreamManager:
        self.sub_watchdog.start()
        if self.preroll is not None:
            with self._lock:
                self._ensure_main_locked()
        self._started = True
        return self

    def stop(self) -> None:
        with self._lock:
            if self._linger is not None:
                self._linger.cancel()
                self._linger = None
            recorders = list(self._recorders)
        for rec in recorders:
            rec.cancel()
        self.sub_watchdog.stop()
        self.main_watchdog.stop()
        with self._lock:
            self._main_running = False
        self._started = False

    def __enter__(self) -> DualStreamManager:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def on_connected(self, role: StreamRole, callback: Callable[[SessionInfo], None]) -> None:
        (self.main if role is StreamRole.MAIN else self.sub).on_connected(callback)

    # ------------------------------------------------------------------ leases

    def _ensure_main_locked(self) -> None:
        if self._linger is not None:
            self._linger.cancel()
            self._linger = None
        if not self._main_running:
            self.main_watchdog.start()
            self._main_running = True
            log.info("%s main stream started on demand", self.camera.label)

    def _acquire(self, decode: bool) -> None:
        with self._lock:
            self._leases += 1
            self._ensure_main_locked()
            if decode:
                self.main.acquire_decode()

    def _release(self, decode: bool) -> None:
        with self._lock:
            self._leases = max(0, self._leases - 1)
            if decode:
                self.main.release_decode()
            if self._leases == 0 and self.preroll is None and self._main_running:
                self._linger = threading.Timer(self.camera.main_idle_linger_s, self._linger_expired)
                self._linger.daemon = True
                self._linger.start()

    def _linger_expired(self) -> None:
        # Stopping under the lock serialises against a concurrent _acquire(): a lease that
        # arrives mid-teardown waits and then restarts the session cleanly.
        with self._lock:
            if self._leases or not self._main_running:
                return
            self._main_running = False
            self._linger = None
            log.info("%s main stream idle -> stopping", self.camera.label)
            self.main_watchdog.stop()

    @contextmanager
    def main_stream(self, *, decode: bool = True, wait_s: float | None = 5.0) -> Iterator[RTSPStreamReader]:
        """Lease the main stream (decoded frames for SAHI if ``decode``).

        ``wait_s`` blocks until the session is streaming (instant if it was warm).
        """
        self._acquire(decode)
        try:
            if wait_s:
                self.main_watchdog.wait_streaming(wait_s)
            yield self.main
        finally:
            self._release(decode)

    def record_evidence(self, duration_s: float = 15.0, *, reason: str = "event",
                        path: str | Path | None = None) -> Future[RecordingResult]:
        """Start a bit-exact main-stream recording (pre-roll + ``duration_s``)."""
        self._acquire(decode=False)
        target = Path(path) if path else evidence_path(self.evidence_dir, self.camera.label, reason)
        rec = EvidenceRecorder(self.main, target, duration_s=duration_s, reason=reason, preroll=self.preroll)
        with self._lock:
            self._recorders.add(rec)

        def done(_f: Future[RecordingResult]) -> None:
            with self._lock:
                self._recorders.discard(rec)
            self._release(decode=False)

        fut = rec.start()
        fut.add_done_callback(done)
        return fut

    @property
    def main_active(self) -> bool:
        return self._main_running
