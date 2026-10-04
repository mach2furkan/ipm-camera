"""Background listener for ``/ISAPI/Event/notification/alertStream``.

* one long-lived chunked GET, framed by :mod:`multipart` (or bare-XML fallback);
* the httpx read timeout acts as an idle watchdog -- Hikvision sends a ``videoloss``
  heartbeat roughly every 10 s, so silence beyond ``idle_timeout_s`` means a dead TCP
  path (cable pulled, NAT entry expired) even if no RST was ever received;
* exponential-backoff reconnect; authentication failures back off at the cap to stay
  clear of the device's illegal-login lockout;
* raw "active" repeats are debounced into START / UPDATE / END phases.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union

import httpx

from ..backoff import ExponentialBackoff
from ..bus import EventBus
from ..errors import AuthenticationError, ISAPIError
from . import xmlutil as X
from .client import HikvisionISAPIClient
from .models import AlertAttachment, AlertEvent, EventPhase, EventState
from .multipart import MultipartStreamParser, Part, XMLDocumentSplitter, boundary_from_content_type

log = logging.getLogger(__name__)

AlertMessage = Union[AlertEvent, AlertAttachment]


@dataclass(slots=True)
class _Track:
    first: AlertEvent
    last: AlertEvent
    last_seen: float


@dataclass(slots=True)
class EventDebouncer:
    """Turns Hikvision's repeated ``active`` posts into a START/UPDATE*/END lifecycle.

    An END is emitted on an explicit ``inactive`` post or when no ``active`` repeat has
    arrived for ``hold_s`` (VMD and smart events rarely send a closing message).
    """

    hold_s: float = 3.0
    emit_updates: bool = False
    _tracks: dict[tuple[str, int | None, tuple[str, ...]], _Track] = field(default_factory=dict)

    def feed(self, ev: AlertEvent, now: float) -> list[AlertEvent]:
        out: list[AlertEvent] = []
        if ev.is_heartbeat:
            return out
        key = ev.key
        track = self._tracks.get(key)
        if ev.state is EventState.INACTIVE:
            if track is not None:
                del self._tracks[key]
                out.append(ev.with_phase(EventPhase.END))
            return out
        if track is None:
            self._tracks[key] = _Track(ev, ev, now)
            out.append(ev.with_phase(EventPhase.START))
        else:
            track.last, track.last_seen = ev, now
            if self.emit_updates:
                out.append(ev.with_phase(EventPhase.UPDATE))
        return out

    def expire(self, now: float) -> list[AlertEvent]:
        out: list[AlertEvent] = []
        for key, track in list(self._tracks.items()):
            if now - track.last_seen >= self.hold_s:
                del self._tracks[key]
                out.append(track.last.with_phase(EventPhase.END))
        return out

    def flush(self) -> list[AlertEvent]:
        out = [t.last.with_phase(EventPhase.END) for t in self._tracks.values()]
        self._tracks.clear()
        return out


class AlertStreamListener:
    PATH = "/ISAPI/Event/notification/alertStream"

    def __init__(
        self,
        client: HikvisionISAPIClient,
        bus: EventBus[AlertMessage],
        *,
        idle_timeout_s: float = 35.0,
        debounce_hold_s: float = 3.0,
        emit_updates: bool = False,
        publish_heartbeats: bool = False,
        backoff: ExponentialBackoff | None = None,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._bus = bus
        self._idle = idle_timeout_s
        self._debouncer = EventDebouncer(debounce_hold_s, emit_updates)
        self._publish_hb = publish_heartbeats
        self._backoff = backoff or ExponentialBackoff()
        self._on_state = on_state
        self._task: asyncio.Task[None] | None = None
        self.connected = False
        self.events_received = 0
        self.reconnects = 0
        self._last_event: AlertEvent | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._bus.bind_loop()
            self._task = asyncio.create_task(self.run(), name="isapi-alert-stream")
        return self._task

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def _state(self, s: str) -> None:
        if self._on_state is not None:
            try:
                self._on_state(s)
            except Exception:
                log.exception("alert-stream state callback failed")

    async def run(self) -> None:
        expiry = asyncio.create_task(self._expiry_loop(), name="isapi-alert-debounce")
        auth_failures = 0
        try:
            while True:
                try:
                    await self._session()
                    reason = "stream ended"
                except asyncio.CancelledError:
                    raise
                except AuthenticationError as exc:
                    # Default Hikvision policy locks the client IP after ~5 failed logins;
                    # credential errors get their own slow schedule (30 s .. 5 min).
                    auth_failures += 1
                    delay = min(300.0, 30.0 * 2 ** (auth_failures - 1))
                    log.error("alert stream: %s; retrying in %.0fs", exc, delay)
                    self._mark_down()
                    await asyncio.sleep(delay)
                    continue
                except (httpx.TimeoutException, httpx.TransportError, ISAPIError, ValueError) as exc:
                    reason = repr(exc)
                if self.connected:
                    auth_failures = 0
                self._mark_down()
                self.reconnects += 1
                delay = self._backoff.next_delay()
                log.warning("alert stream down (%s); reconnecting in %.1fs", reason, delay)
                await asyncio.sleep(delay)
        finally:
            expiry.cancel()
            for ev in self._debouncer.flush():
                self._bus.publish(ev)
            self._mark_down()

    def _mark_down(self) -> None:
        if self.connected:
            self.connected = False
            self._state("disconnected")
        # An outage hides "inactive" posts: close every open lifecycle instead of leaving
        # downstream consumers believing an intrusion is still in progress.
        for ev in self._debouncer.flush():
            self._bus.publish(ev)

    async def _expiry_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0.5)
            for ev in self._debouncer.expire(loop.time()):
                self._bus.publish(ev)

    async def _session(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._client.open_stream(self.PATH, idle_timeout_s=self._idle) as resp:
            boundary = boundary_from_content_type(resp.headers.get("content-type", ""))
            parser: MultipartStreamParser | XMLDocumentSplitter
            parser = MultipartStreamParser(boundary) if boundary else XMLDocumentSplitter()
            self.connected = True
            self._state("connected")
            log.info("alert stream connected (%s framing)", "multipart" if boundary else "bare-xml")
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                self._backoff.mark_healthy(loop.time())
                for part in parser.feed(chunk):
                    self._dispatch(part, loop.time())

    def _dispatch(self, part: Part, now: float) -> None:
        ctype = part.content_type
        body = part.body.strip()
        if not body:
            return
        if ctype.startswith("image/") or ctype == "application/octet-stream":
            disp = part.headers.get("content-disposition", "")
            m = re.search(r'name="?([^";]+)"?', disp)
            self._bus.publish(AlertAttachment(ctype or "application/octet-stream", part.body,
                                              related_event=self._last_event, name=m.group(1) if m else None,
                                              content_id=part.headers.get("content-id")))
            return
        try:
            if "json" in ctype or body[:1] == b"{":
                ev = AlertEvent.from_json(body)
            else:
                root = X.parse(body)
                if X.local(root.tag) != "EventNotificationAlert":
                    return
                ev = AlertEvent.from_xml(root)
        except (ValueError, ET.ParseError, KeyError, TypeError):
            log.debug("unparseable alert part (%s, %d bytes)", ctype, len(body))
            return
        self.events_received += 1
        self._last_event = ev
        if ev.is_heartbeat:
            if self._publish_hb:
                self._bus.publish(ev)
            return
        for out in self._debouncer.feed(ev, now):
            self._bus.publish(out)
