"""Single-slot overwrite buffer: the size-1 drop-oldest queue between decoder and model.

The producer never blocks and never accumulates: publishing replaces the previous item,
so a consumer that falls behind always resumes at the most recent picture rather than
working through a backlog (the cause of "the detector is 6 seconds behind reality").
"""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    __slots__ = ("_closed", "_cond", "_item", "_read_seq", "_seq", "overwritten", "published")

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._item: T | None = None
        self._seq = 0
        self._read_seq = 0
        self.published = 0
        self.overwritten = 0
        self._closed = False

    @property
    def seq(self) -> int:
        return self._seq

    def publish(self, item: T) -> int:
        with self._cond:
            if self._item is not None and self._read_seq != self._seq:
                self.overwritten += 1
            self._item = item
            self._seq += 1
            self.published += 1
            self._cond.notify_all()
            return self._seq

    def peek(self) -> tuple[int, T | None]:
        with self._cond:
            return self._seq, self._item

    def take(self) -> tuple[int, T | None]:
        """Latest item, marking it consumed (for overwrite accounting)."""
        with self._cond:
            self._read_seq = self._seq
            return self._seq, self._item

    def wait_newer(self, after_seq: int, timeout: float | None) -> tuple[int, T | None]:
        """Block until an item newer than ``after_seq`` exists (or timeout/close)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._seq <= after_seq and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self._seq > after_seq:
                self._read_seq = self._seq
            return self._seq, self._item

    def clear(self) -> None:
        with self._cond:
            self._item = None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._item = None
            self._cond.notify_all()

    def reopen(self) -> None:
        with self._cond:
            self._closed = False
