"""Bridge from analytics events (inference thread) to physical actions (asyncio, ISAPI).

``dispatch`` never blocks the inference loop: it applies per-rule cooldowns and schedules
``CameraNode.respond_to_violation`` on the node's event loop. Event-to-relay latency is
measured so the "< 200 ms alarm -> relay" KPI can be monitored in production.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..metrics import RollingWindow
from ..node import CameraNode, ViolationResponse
from .rules import EventKind, SecurityEvent

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    relay_output: int | None = 1
    relay_s: float = 2.0
    audio_id: int | None = None
    record_s: float | None = 15.0
    cooldown_s: float = 10.0
    kinds: frozenset[EventKind] = field(
        default_factory=lambda: frozenset({EventKind.TRIPWIRE, EventKind.INTRUSION, EventKind.LOITERING}))


class EventActionDispatcher:
    def __init__(self, node: CameraNode, policies: dict[str, ActionSpec], *,
                 default: ActionSpec | None = None, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._node = node
        self._policies = policies
        self._default = default
        self._loop = loop
        self._last: dict[str, float] = {}
        self.latency_ms = RollingWindow(256)
        self.dispatched = 0
        self.suppressed = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def dispatch(self, events: Iterable[SecurityEvent]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        now = time.monotonic()
        for ev in events:
            spec = self._policies.get(ev.rule, self._default)
            if spec is None or ev.kind not in spec.kinds:
                continue
            if now - self._last.get(ev.rule, -1e18) < spec.cooldown_s:
                self.suppressed += 1
                continue
            self._last[ev.rule] = now
            self.dispatched += 1
            fut = asyncio.run_coroutine_threadsafe(self._run(ev, spec, time.perf_counter()), loop)
            fut.add_done_callback(lambda f: f.cancelled() or f.exception() is None
                                  or log.error("action failed: %s", f.exception()))

    async def _run(self, ev: SecurityEvent, spec: ActionSpec, t0: float) -> ViolationResponse:
        log.warning("ALARM %s", ev)
        relay_task = None
        if spec.relay_output is not None:
            # The relay goes first and on its own: it is the latency-critical actuation.
            relay_task = asyncio.create_task(self._node.isapi.pulse_alarm_output(spec.relay_output, spec.relay_s))
            relay_task.add_done_callback(lambda _t: self.latency_ms.add((time.perf_counter() - t0) * 1000.0))
        resp = await self._node.respond_to_violation(
            relay_output=None, audio_id=spec.audio_id, record_s=spec.record_s,
            reason=f"{ev.kind.value}-{ev.rule}-t{ev.track_id}")
        relay_exc: BaseException | None = None
        if relay_task is not None:
            try:
                await relay_task
            except Exception as exc:  # noqa: BLE001
                relay_exc = exc
        return ViolationResponse(relay=relay_exc, audio=resp.audio, evidence=resp.evidence)
