"""Capped exponential backoff with bounded jitter and stability-based reset."""

from __future__ import annotations

import random
import threading
import time


class ExponentialBackoff:
    """Produces 1s, 2s, 4s, 8s, 15s, 15s ... (defaults) with +/- ``jitter`` spread.

    Jitter de-synchronises reconnect storms when a PoE switch reboots and dozens of
    cameras come back at the same instant. ``mark_healthy`` resets the sequence only
    once a session has stayed healthy for ``stable_after_s``; a camera that accepts the
    handshake and then drops after 200 ms must not be hammered at the 1 s floor.
    """

    __slots__ = (
        "_attempt",
        "_factor",
        "_healthy_since",
        "_initial",
        "_jitter",
        "_lock",
        "_maximum",
        "_rng",
        "_stable_after",
    )

    def __init__(
        self,
        initial_s: float = 1.0,
        factor: float = 2.0,
        maximum_s: float = 15.0,
        jitter: float = 0.1,
        stable_after_s: float = 10.0,
        *,
        seed: int | None = None,
    ) -> None:
        if initial_s <= 0 or factor < 1.0 or maximum_s < initial_s:
            raise ValueError("invalid backoff parameters")
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        self._initial = initial_s
        self._factor = factor
        self._maximum = maximum_s
        self._jitter = jitter
        self._stable_after = stable_after_s
        self._attempt = 0
        self._healthy_since: float | None = None
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    @property
    def attempt(self) -> int:
        return self._attempt

    def peek_base(self) -> float:
        """Un-jittered delay the next call to :meth:`next_delay` is centred on."""
        return min(self._maximum, self._initial * (self._factor ** self._attempt))

    def next_delay(self) -> float:
        with self._lock:
            base = self.peek_base()
            self._attempt += 1
            self._healthy_since = None
            if self._jitter:
                spread = base * self._jitter
                base = base + self._rng.uniform(-spread, spread)
            return max(0.0, min(base, self._maximum))

    def mark_healthy(self, now: float | None = None) -> None:
        """Call while the protected resource is working; resets after the stability window."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._healthy_since is None:
                self._healthy_since = now
            elif now - self._healthy_since >= self._stable_after:
                self._attempt = 0

    def reset(self) -> None:
        with self._lock:
            self._attempt = 0
            self._healthy_since = None
