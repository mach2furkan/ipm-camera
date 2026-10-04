"""Message bus abstraction with NATS subject semantics.

* ``InProcessBus``  single-process deployments and tests (asyncio, drop-oldest per subscriber).
* ``NatsJetStreamBus``  multi-host deployments: edge nodes publish to
  ``events.camera.<camera_id>`` / ``alerts.camera.<camera_id>``; JetStream persists the
  stream so a restarting fusion engine resumes from its durable consumer position.

Subjects use NATS wildcards: ``*`` matches one token, ``>`` matches the rest.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Message:
    subject: str
    data: bytes


def subject_matches(pattern: str, subject: str) -> bool:
    p = pattern.split(".")
    s = subject.split(".")
    for i, tok in enumerate(p):
        if tok == ">":
            return len(s) > i
        if i >= len(s) or (tok != "*" and tok != s[i]):
            return False
    return len(p) == len(s)


class Subscription(Protocol):
    def __aiter__(self) -> AsyncIterator[Message]: ...

    async def close(self) -> None: ...


class MessageBus(Protocol):
    async def publish(self, subject: str, data: bytes) -> None: ...

    async def subscribe(self, pattern: str, *, durable: str | None = None) -> Subscription: ...

    async def close(self) -> None: ...


class _LocalSub:
    def __init__(self, bus: InProcessBus, pattern: str, maxsize: int) -> None:
        self.bus = bus
        self.pattern = pattern
        self.q: asyncio.Queue[Message | None] = asyncio.Queue(maxsize)
        self.dropped = 0

    def offer(self, msg: Message) -> None:
        while True:
            try:
                self.q.put_nowait(msg)
                return
            except asyncio.QueueFull:
                self.q.get_nowait()
                self.dropped += 1

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        m = await self.q.get()
        if m is None:
            raise StopAsyncIteration
        return m

    async def close(self) -> None:
        self.bus._subs.discard(self)
        self.offer(None)  # type: ignore[arg-type]


class InProcessBus:
    def __init__(self, maxsize: int = 4096) -> None:
        self._subs: set[_LocalSub] = set()
        self._maxsize = maxsize
        self.published = 0

    async def publish(self, subject: str, data: bytes) -> None:
        self.published += 1
        msg = Message(subject, data)
        for s in tuple(self._subs):
            if subject_matches(s.pattern, subject):
                s.offer(msg)

    async def subscribe(self, pattern: str, *, durable: str | None = None) -> _LocalSub:
        sub = _LocalSub(self, pattern, self._maxsize)
        self._subs.add(sub)
        return sub

    async def close(self) -> None:
        for s in tuple(self._subs):
            await s.close()


class NatsJetStreamBus:
    """NATS JetStream transport (``pip install nats-py``)."""

    def __init__(self, servers: str | list[str] = "nats://127.0.0.1:4222", *, stream: str = "SURVEILLANCE",
                 subjects: tuple[str, ...] = ("events.camera.*", "alerts.camera.*"), max_age_s: float = 3600.0,
                 memory_storage: bool = True) -> None:
        self._servers = servers
        self._stream = stream
        self._subjects = list(subjects)
        self._max_age = max_age_s
        self._memory = memory_storage
        self._nc: Any = None
        self._js: Any = None

    async def connect(self) -> NatsJetStreamBus:
        import nats
        from nats.js.api import StorageType, StreamConfig

        self._nc = await nats.connect(self._servers, max_reconnect_attempts=-1, reconnect_time_wait=1)
        self._js = self._nc.jetstream()
        cfg = StreamConfig(name=self._stream, subjects=self._subjects, max_age=self._max_age,
                           storage=StorageType.MEMORY if self._memory else StorageType.FILE)
        try:
            await self._js.add_stream(cfg)
        except Exception:  # noqa: BLE001 - stream exists: update in place
            await self._js.update_stream(cfg)
        return self

    async def publish(self, subject: str, data: bytes) -> None:
        await self._js.publish(subject, data)

    async def subscribe(self, pattern: str, *, durable: str | None = None) -> Subscription:
        sub = await self._js.subscribe(pattern, durable=durable, manual_ack=True)

        class _NatsSub:
            def __aiter__(self) -> AsyncIterator[Message]:
                return self._gen()

            async def _gen(self) -> AsyncIterator[Message]:
                async for m in sub.messages:
                    await m.ack()
                    yield Message(m.subject, m.data)

            async def close(self) -> None:
                await sub.unsubscribe()

        return _NatsSub()

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
