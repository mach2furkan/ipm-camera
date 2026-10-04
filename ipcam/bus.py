"""In-process asyncio event bus with per-subscriber drop-oldest back-pressure.

A slow subscriber (e.g. an MQTT bridge during a broker outage) must never stall the
alert-stream reader or other subscribers, so each subscription owns a bounded queue and
the oldest event is discarded when it is full. ``publish`` is safe to call from any
thread; cross-thread calls are marshalled onto the bus loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class Subscription(Generic[T]):
    def __init__(
        self,
        bus: EventBus[T],
        maxsize: int,
        predicate: Callable[[T], bool] | None,
        name: str,
    ) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[T | None] = asyncio.Queue(maxsize=maxsize)
        self._predicate = predicate
        self.name = name
        self.dropped = 0
        self._closed = False

    def _offer(self, event: T) -> None:
        if self._closed:
            return
        if self._predicate is not None:
            try:
                if not self._predicate(event):
                    return
            except Exception:
                log.exception("subscription %s predicate failed", self.name)
                return
        q = self._queue
        while True:
            try:
                q.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass

    async def get(self, timeout: float | None = None) -> T:
        item = await asyncio.wait_for(self._queue.get(), timeout)
        if item is None:
            raise StopAsyncIteration
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self)
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        return await self.get()

    async def __aenter__(self) -> Subscription[T]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()


class EventBus(Generic[T]):
    def __init__(self, default_maxsize: int = 256) -> None:
        self._subs: list[Subscription[T]] = []
        self._lock = threading.Lock()
        self._default_maxsize = default_maxsize
        self._loop: asyncio.AbstractEventLoop | None = None
        self.published = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    def subscribe(
        self,
        *,
        maxsize: int | None = None,
        predicate: Callable[[T], bool] | None = None,
        name: str = "sub",
    ) -> Subscription[T]:
        if self._loop is None:
            self.bind_loop()
        sub = Subscription(self, maxsize or self._default_maxsize, predicate, name)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription[T]) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def publish(self, event: T) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fanout(event)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: T) -> None:
        self.published += 1
        with self._lock:
            subs = tuple(self._subs)
        for sub in subs:
            sub._offer(event)

    def close(self) -> None:
        with self._lock:
            subs = tuple(self._subs)
        for sub in subs:
            sub.close()
