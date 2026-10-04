"""StreamWatchdog: supervises an RTSPStreamReader, detects stalls, reconnects with backoff.

Supervisor thread state machine::

    ┌──────────► CONNECTING ──handshake──► STREAMING ──no frame 3 s──► STALLED ─┐
    │                │                         │                               │
    │             failure                 error / EOF                       teardown
    │                ▼                         ▼                               │
    └──backoff── BACKOFF ◄─────────────────────┴───────────────────────────────┘

Each session runs on a fresh worker thread. A stall or failure sets the session's abort
event; FFmpeg's socket I/O timeout (3 s) bounds how long a blocked ``av_read_frame`` can
ignore it. If the worker still has not exited after ``hard_kill_grace_s`` (driver hang,
dead NFS, stuck GStreamer state change) it is *quarantined*: its generation is revoked so
it can never publish again, and a new session is started without waiting for it. Python
cannot kill threads safely; quarantining keeps the pipeline live and the zombie count is
exported so it can be alerted on.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from ..backoff import ExponentialBackoff
from ..errors import FailureKind, StreamStalledError, classify_failure
from .reader import Phase, RTSPStreamReader

log = logging.getLogger(__name__)


class LinkState(str, Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    STALLED = "stalled"
    BACKOFF = "backoff"


@dataclass(frozen=True, slots=True)
class WatchdogEvent:
    state: LinkState
    reason: str
    failure: FailureKind | None
    attempt: int
    next_retry_s: float | None


@dataclass(frozen=True, slots=True)
class WatchdogStats:
    state: LinkState
    reconnects: int
    stalls: int
    zombies: int
    last_failure: FailureKind | None
    last_error: str | None
    uptime_s: float


class StreamWatchdog:
    def __init__(
        self,
        reader: RTSPStreamReader,
        *,
        stall_timeout_s: float | None = None,
        keyframe_wait_s: float | None = None,
        connect_timeout_s: float | None = None,
        backoff: ExponentialBackoff | None = None,
        hard_kill_grace_s: float = 5.0,
        poll_interval_s: float = 0.1,
        on_event: Callable[[WatchdogEvent], None] | None = None,
    ) -> None:
        p = reader.profile
        self.reader = reader
        self._stall = stall_timeout_s if stall_timeout_s is not None else p.stall_timeout_s
        self._kf_wait = max(self._stall, keyframe_wait_s if keyframe_wait_s is not None else p.keyframe_wait_s)
        # Handshake budget: socket open + DESCRIBE/SETUP/PLAY + probe, plus slack.
        self._connect = connect_timeout_s if connect_timeout_s is not None else p.open_timeout_s + p.read_timeout_s + 2
        self._backoff = backoff or ExponentialBackoff(1.0, 2.0, 15.0)
        self._grace = hard_kill_grace_s
        self._poll = poll_interval_s
        self._on_event = on_event

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = LinkState.STOPPED
        self._reconnects = 0
        self._stalls = 0
        self._zombies = 0
        self._last_failure: FailureKind | None = None
        self._last_error: str | None = None
        self._streaming_since: float | None = None
        self._connected_evt = threading.Event()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> StreamWatchdog:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, name=f"watchdog[{self.reader.name}]", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                log.error("%s watchdog did not stop within %.1fs", self.reader.name, timeout)
        self._thread = None
        self.reader.close()
        self._set_state(LinkState.STOPPED, "stopped", None, None)

    def __enter__(self) -> StreamWatchdog:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def wait_streaming(self, timeout: float | None = None) -> bool:
        return self._connected_evt.wait(timeout)

    @property
    def state(self) -> LinkState:
        return self._state

    def stats(self) -> WatchdogStats:
        up = time.monotonic() - self._streaming_since if self._streaming_since else 0.0
        return WatchdogStats(self._state, self._reconnects, self._stalls, self._zombies,
                             self._last_failure, self._last_error, up)

    # ------------------------------------------------------------------ supervision

    def _set_state(self, state: LinkState, reason: str, failure: FailureKind | None,
                   retry: float | None) -> None:
        changed = state is not self._state
        self._state = state
        if state is LinkState.STREAMING:
            self._connected_evt.set()
            if self._streaming_since is None:
                self._streaming_since = time.monotonic()
        else:
            self._connected_evt.clear()
            self._streaming_since = None
        if changed and self._on_event is not None:
            try:
                self._on_event(WatchdogEvent(state, reason, failure, self._backoff.attempt, retry))
            except Exception:
                log.exception("watchdog event callback failed")

    def _supervise(self) -> None:
        reader = self.reader
        while not self._stop.is_set():
            generation = reader.next_generation()
            abort = threading.Event()
            outcome: dict[str, BaseException | None] = {"exc": None}

            def work(gen: int = generation, ab: threading.Event = abort,
                     out: dict[str, BaseException | None] = outcome) -> None:
                try:
                    reader.run_session(ab, gen)
                except BaseException as exc:  # noqa: BLE001 - reported to supervisor
                    out["exc"] = exc

            self._set_state(LinkState.CONNECTING, "connecting", None, None)
            worker = threading.Thread(target=work, name=f"rtsp[{reader.name}]#{generation}", daemon=True)
            started = time.monotonic()
            worker.start()

            stall_exc = self._monitor(worker, abort, started)
            if stall_exc is not None:
                outcome["exc"] = stall_exc

            worker.join(self._grace)
            if worker.is_alive():
                self._zombies += 1
                reader.next_generation()  # revoke: the zombie may never publish again
                log.critical("%s worker %s unresponsive after teardown; quarantined (zombies=%d)",
                             reader.name, worker.name, self._zombies)
            reader.mark_idle()

            if self._stop.is_set():
                break

            err = outcome["exc"]
            kind = classify_failure(err)
            self._last_failure = kind
            self._last_error = reader.sanitize(str(err)) if err else "stream ended"
            self._reconnects += 1
            delay = self._backoff.next_delay()
            if kind is FailureKind.AUTH:
                # Hammering wrong credentials gets the host IP locked out by the camera.
                delay = max(delay, 60.0)
            self._set_state(LinkState.BACKOFF, self._last_error, kind, delay)
            log.warning("%s session ended [%s] %s; reconnect #%d in %.1fs", reader.name, kind.value,
                        self._last_error, self._reconnects, delay)
            if self._stop.wait(delay):
                break

    def _monitor(self, worker: threading.Thread, abort: threading.Event,
                 started: float) -> BaseException | None:
        """Poll the reader heartbeat until the worker exits; return a stall error if torn down."""
        reader = self.reader
        while worker.is_alive():
            if self._stop.wait(self._poll):
                abort.set()
                return None
            hb = reader.heartbeat()
            now_ns = time.perf_counter_ns()
            now = time.monotonic()

            if hb.phase is Phase.CONNECTING:
                if now - started > self._connect:
                    return self._teardown(abort, f"handshake exceeded {self._connect:.1f}s")
                continue

            since_packet = (now_ns - hb.last_packet_ns) / 1e9
            if since_packet > self._stall:
                return self._teardown(abort, f"no packets for {since_packet:.1f}s")

            if hb.decode_enabled:
                ref_ns = max(hb.last_frame_ns, hb.session_started_ns, hb.decode_started_ns)
                since_frame = (now_ns - ref_ns) / 1e9
                limit = self._kf_wait if hb.phase is Phase.AWAITING_KEYFRAME else self._stall
                if since_frame > limit:
                    return self._teardown(abort, f"no decoded frame for {since_frame:.1f}s")
                if hb.phase is Phase.STREAMING:
                    self._set_state(LinkState.STREAMING, "streaming", None, None)
                    self._backoff.mark_healthy(now)
            else:
                self._set_state(LinkState.STREAMING, "streaming (packets only)", None, None)
                self._backoff.mark_healthy(now)
        return None

    def _teardown(self, abort: threading.Event, reason: str) -> StreamStalledError:
        self._stalls += 1
        log.warning("%s stalled: %s -> forcing teardown", self.reader.name, reason)
        self._set_state(LinkState.STALLED, reason, FailureKind.TIMEOUT, None)
        abort.set()
        return StreamStalledError(reason)
