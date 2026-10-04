"""Evidence recording by remux (no decode, no re-encode).

Main-stream packets are copied bit-exact into a container, so a 4K H.265 clip costs a
few percent of one CPU core and the evidence keeps the camera's original quality (and
is defensible as unaltered). An optional GOP-aligned pre-roll ring supplies the seconds
*before* the trigger.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import av

from ..resources import NativeHandle
from .reader import RTSPStreamReader
from .session import SessionInfo

log = logging.getLogger(__name__)


class PacketRing:
    """Holds the last ``seconds`` of compressed packets, always starting at a keyframe.

    Trimming only ever discards whole GOPs, so a snapshot is decodable from its first
    packet. Retention is therefore between ``seconds`` and ``seconds + one GOP``.
    """

    def __init__(self, seconds: float, *, max_packets: int = 50_000) -> None:
        self._horizon_ns = int(seconds * 1e9)
        self._max = max_packets
        self._packets: deque[tuple[int, int, Any]] = deque()   # (seq, arrival_ns, packet)
        self._keys: deque[tuple[int, int]] = deque()           # (seq, arrival_ns) of keyframes
        self._seq = 0
        self._lock = threading.Lock()

    def push(self, packet: Any, arrival_ns: int, _info: SessionInfo | None = None) -> None:
        with self._lock:
            self._seq += 1
            is_key = bool(packet.is_keyframe)
            if not self._packets and not is_key:
                return  # never start a ring on a P-frame
            self._packets.append((self._seq, arrival_ns, packet))
            if is_key:
                self._keys.append((self._seq, arrival_ns))
            self._trim(arrival_ns)

    def _trim(self, now_ns: int) -> None:
        horizon = now_ns - self._horizon_ns
        keys, pkts = self._keys, self._packets
        while len(keys) >= 2 and (keys[1][1] <= horizon or len(pkts) > self._max):
            cut_seq = keys[1][0]
            while pkts and pkts[0][0] < cut_seq:
                pkts.popleft()
            keys.popleft()

    def snapshot(self) -> list[tuple[int, Any]]:
        with self._lock:
            return [(a, p) for _, a, p in self._packets]

    def clear(self) -> None:
        with self._lock:
            self._packets.clear()
            self._keys.clear()

    @property
    def span_s(self) -> float:
        with self._lock:
            if len(self._packets) < 2:
                return 0.0
            return (self._packets[-1][1] - self._packets[0][1]) / 1e9


@dataclass(frozen=True, slots=True)
class RecordingResult:
    path: Path
    reason: str
    started_wall: float
    duration_s: float
    preroll_s: float
    packets: int
    bytes: int
    discontinuities: int
    overflowed: bool
    ok: bool
    error: str | None = None


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def evidence_path(directory: str | Path, camera_label: str, reason: str, suffix: str = ".mkv") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name = f"{_SAFE.sub('-', camera_label)}_{ts}_{_SAFE.sub('-', reason) or 'event'}{suffix}"
    return Path(directory) / name


class EvidenceRecorder:
    """Records ``preroll + duration_s`` of the reader's packets to ``path``.

    The packet tap only enqueues (never blocks the demux thread); muxing and disk I/O run
    on the recorder's own thread. If the queue overflows (disk stall) the recording drops
    up to the next keyframe and is flagged ``overflowed`` rather than corrupting the file.
    """

    def __init__(
        self,
        reader: RTSPStreamReader,
        path: str | Path,
        *,
        duration_s: float,
        reason: str = "event",
        preroll: PacketRing | None = None,
        max_queue: int = 4096,
        start_timeout_s: float = 15.0,
    ) -> None:
        self.reader = reader
        self.path = Path(path)
        self.duration_s = duration_s
        self.reason = reason
        self._preroll = preroll
        self._queue: queue.Queue[tuple[int, Any] | None] = queue.Queue(maxsize=max_queue)
        self._start_timeout = start_timeout_s
        self._overflowed = False
        self._resync = False
        self._accepting = False
        self._thread: threading.Thread | None = None
        self.future: Future[RecordingResult] = Future()

    def start(self) -> Future[RecordingResult]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._trigger_ns = time.perf_counter_ns()
        self._trigger_wall = time.time()
        # Tap first, snapshot second: nothing falls in between; overlap is de-duplicated
        # by arrival time in ``_run``.
        self._accepting = True
        self.reader.add_packet_tap(self._tap)
        backlog = self._preroll.snapshot() if self._preroll is not None else []
        self._thread = threading.Thread(target=self._run, args=(backlog,), name=f"evidence[{self.path.name}]",
                                        daemon=True)
        self._thread.start()
        return self.future

    def _tap(self, packet: Any, arrival_ns: int, _info: SessionInfo) -> None:
        if not self._accepting:
            return
        if self._resync:
            if not packet.is_keyframe:
                return
            self._resync = False
        try:
            self._queue.put_nowait((arrival_ns, packet))
        except queue.Full:
            self._overflowed = True
            self._resync = True

    def _run(self, backlog: list[tuple[int, Any]]) -> None:
        packets = written = discont = 0
        end_ns = self._trigger_ns + int(self.duration_s * 1e9)
        first_arrival: int | None = None
        error: str | None = None
        fmt = "mp4" if self.path.suffix.lower() == ".mp4" else "matroska"
        options = {"movflags": "frag_keyframe+empty_moov+default_base_moof"} if fmt == "mp4" else {}
        try:
            with NativeHandle(av.open(str(self.path), mode="w", format=fmt, options=options),
                              lambda c: c.close(), "av.output") as out_h:
                out = out_h.obj
                out_stream: Any = None
                in_stream: Any = None
                offset: int | None = None
                last_dts: int | None = None
                last_in_pts: int | None = None
                started = False

                backlog_end = backlog[-1][0] if backlog else -1

                def source() -> Any:
                    yield from backlog
                    deadline = time.monotonic() + self._start_timeout
                    while True:
                        timeout = max(0.05, min(1.0, deadline - time.monotonic())) if not started else 1.0
                        try:
                            item = self._queue.get(timeout=timeout)
                        except queue.Empty:
                            if not started and time.monotonic() > deadline:
                                raise TimeoutError("no keyframe received from main stream") from None
                            if time.perf_counter_ns() >= end_ns:
                                return
                            continue
                        if item is None:
                            return
                        if item[0] <= backlog_end:
                            continue
                        yield item

                for arrival_ns, pkt in source():
                    if arrival_ns >= end_ns:
                        break
                    if not started:
                        if not pkt.is_keyframe:
                            continue
                        started = True
                        first_arrival = arrival_ns
                    if out_stream is None:
                        in_stream = pkt.stream
                        out_stream = _add_stream_from(out, in_stream)
                    tb = pkt.time_base
                    pts = pkt.pts if pkt.pts is not None else pkt.dts
                    dts = pkt.dts if pkt.dts is not None else pts
                    if pts is None or dts is None:
                        continue
                    # Rebase to zero; on a PTS jump (reconnect mid-recording) continue the
                    # timeline one tick after the last written packet.
                    if offset is None:
                        offset = dts
                    elif last_in_pts is not None and (pts < last_in_pts or float((pts - last_in_pts) * tb) > 5.0):
                        discont += 1
                        offset = dts - ((last_dts or 0) + 1)
                    last_in_pts = pts
                    new_dts = dts - offset
                    if last_dts is not None and new_dts <= last_dts:
                        new_dts = last_dts + 1
                    new_pts = max(pts - offset, new_dts)
                    copy = av.Packet(bytes(pkt))
                    copy.pts, copy.dts, copy.time_base = new_pts, new_dts, tb
                    try:
                        copy.is_keyframe = bool(pkt.is_keyframe)
                    except AttributeError:
                        pass
                    copy.stream = out_stream
                    out.mux(copy)
                    last_dts = new_dts
                    packets += 1
                    written += pkt.size
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            log.error("evidence recording %s failed: %s", self.path, exc)
        finally:
            self._accepting = False
            self.reader.remove_packet_tap(self._tap)
            self._drain()

        preroll_s = max(0.0, (self._trigger_ns - first_arrival) / 1e9) if first_arrival else 0.0
        result = RecordingResult(
            path=self.path, reason=self.reason, started_wall=self._trigger_wall - preroll_s,
            duration_s=self.duration_s + preroll_s, preroll_s=preroll_s, packets=packets, bytes=written,
            discontinuities=discont, overflowed=self._overflowed, ok=error is None and packets > 0, error=error,
        )
        log.info("evidence %s: %d packets, %.1f MB, pre-roll %.1fs, ok=%s", self.path.name, packets,
                 written / 1e6, preroll_s, result.ok)
        self.future.set_result(result)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def cancel(self) -> None:
        self._accepting = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._drain()
            self._queue.put_nowait(None)


def _add_stream_from(out: Any, template: Any) -> Any:
    add = getattr(out, "add_stream_from_template", None)
    if add is not None:
        return add(template)
    return out.add_stream(template=template)
