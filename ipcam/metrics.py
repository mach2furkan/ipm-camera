"""Lock-protected rolling statistics used for latency/FPS telemetry."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Summary:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float

    def fmt(self, unit: str = "ms") -> str:
        if not self.count:
            return "n/a"
        return (f"p50={self.p50:.1f}{unit} p95={self.p95:.1f}{unit} p99={self.p99:.1f}{unit} "
                f"min={self.minimum:.1f} max={self.maximum:.1f} n={self.count}")


_EMPTY = Summary(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)


class RollingWindow:
    """Fixed-size sample window. Percentiles sort a copy: O(n log n) with n<=window, cheap
    enough at telemetry rates and free of the bias of streaming quantile sketches."""

    __slots__ = ("_lock", "_samples", "_total")

    def __init__(self, size: int = 1024) -> None:
        self._samples: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()
        self._total = 0

    def add(self, value: float) -> None:
        if math.isfinite(value):
            with self._lock:
                self._samples.append(value)
                self._total += 1

    @property
    def total(self) -> int:
        return self._total

    def summary(self) -> Summary:
        with self._lock:
            data = sorted(self._samples)
        if not data:
            return _EMPTY
        n = len(data)

        def pct(q: float) -> float:
            # Linear interpolation between closest ranks (numpy "linear" method).
            pos = (n - 1) * q
            lo = math.floor(pos)
            hi = min(lo + 1, n - 1)
            return data[lo] + (data[hi] - data[lo]) * (pos - lo)

        return Summary(n, sum(data) / n, pct(0.50), pct(0.95), pct(0.99), data[0], data[-1])


class RateMeter:
    """Events/second over a sliding time window (robust to bursty arrival).`n`n    Timestamps are ``time.perf_counter()`` seconds (same clock as the stream layer's`n    ``perf_counter_ns`` stamps); ``time.monotonic()`` ticks in 15.6 ms steps on Windows.`n    """

    __slots__ = ("_lock", "_stamps", "_window")

    def __init__(self, window_s: float = 2.0) -> None:
        self._stamps: deque[float] = deque()
        self._window = window_s
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._stamps.append(now)
            self._evict(now)

    def rate(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._evict(now)
            n = len(self._stamps)
            if n < 2:
                return 0.0
            span = now - self._stamps[0]
            return (n - 1) / span if span > 0 else 0.0

    def _evict(self, now: float) -> None:
        horizon = now - self._window
        stamps = self._stamps
        while stamps and stamps[0] < horizon:
            stamps.popleft()


class SlidingMin:
    """Monotonic-deque sliding-window minimum over time, O(1) amortised."""

    __slots__ = ("_dq", "_window")

    def __init__(self, window_s: float) -> None:
        self._window = window_s
        self._dq: deque[tuple[float, float]] = deque()

    def push(self, t: float, value: float) -> float:
        dq = self._dq
        while dq and dq[-1][1] >= value:
            dq.pop()
        dq.append((t, value))
        horizon = t - self._window
        while dq[0][0] < horizon:
            dq.popleft()
        return dq[0][1]

    def clear(self) -> None:
        self._dq.clear()


class DriftLagEstimator:
    """Estimates buffering lag from the drift between host arrival time and stream PTS.

    For a live source ``arrival - pts`` is a constant (clock offset + network base delay)
    plus whatever queueing was added on the way. Tracking the sliding minimum of that
    difference as the zero-queue baseline turns every sample into "ms behind real time
    compared with the best observed delivery" -- exactly the lag the drop-oldest policy is
    meant to bound. Camera/host clock skew (~tens of ppm) is absorbed by the window.
    A PTS discontinuity (reconnect, camera reboot, 33-bit wrap) resets the baseline.
    """

    __slots__ = ("_discontinuity_s", "_last_arrival", "_last_pts", "_min")

    def __init__(self, window_s: float = 30.0, discontinuity_s: float = 2.0) -> None:
        self._min = SlidingMin(window_s)
        self._last_pts: float | None = None
        self._last_arrival: float | None = None
        self._discontinuity_s = discontinuity_s

    def reset(self) -> None:
        self._min.clear()
        self._last_pts = None
        self._last_arrival = None

    def update(self, arrival_s: float, pts_s: float | None) -> float:
        if pts_s is None or not math.isfinite(pts_s):
            return math.nan
        if self._last_pts is not None and self._last_arrival is not None:
            d_pts = pts_s - self._last_pts
            d_arr = arrival_s - self._last_arrival
            if d_pts < -1e-3 or abs(d_pts - d_arr) > self._discontinuity_s:
                self._min.clear()
        self._last_pts = pts_s
        self._last_arrival = arrival_s
        delta = arrival_s - pts_s
        baseline = self._min.push(arrival_s, delta)
        return (delta - baseline) * 1000.0
