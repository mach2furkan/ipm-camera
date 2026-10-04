"""CameraNode: one camera's data plane + control plane behind a single async facade."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bus import EventBus
from .config import CameraConfig, StreamRole
from .errors import IPCamError
from .isapi.alert_stream import AlertMessage, AlertStreamListener
from .isapi.client import HikvisionISAPIClient
from .stream.frame import FrameResult
from .stream.manager import DualStreamManager
from .stream.recorder import RecordingResult
from .stream.session import SessionInfo
from .stream.watchdog import WatchdogEvent
from .vision.ir_state import IlluminationState, IRStateResolver

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ViolationResponse:
    relay: BaseException | None
    audio: BaseException | str | None
    evidence: RecordingResult | BaseException | None


class CameraNode:
    """Usage::

        async with CameraNode(CameraConfig.from_env()) as node:
            while True:
                res = node.latest_frame()
                ...
                if violation:
                    await node.respond_to_violation(audio_id=1)
    """

    def __init__(
        self,
        camera: CameraConfig,
        *,
        evidence_dir: str | Path = "evidence",
        ir_poll_s: float = 10.0,
        listen_alerts: bool = True,
        request_keyframe_on_connect: bool = True,
        on_watchdog_event: Callable[[StreamRole, WatchdogEvent], None] | None = None,
    ) -> None:
        self.camera = camera
        self.isapi = HikvisionISAPIClient.from_config(camera)
        self.streams = DualStreamManager(camera, evidence_dir=evidence_dir, on_watchdog_event=on_watchdog_event)
        self.events: EventBus[AlertMessage] = EventBus()
        self.ir = IRStateResolver()
        self._ir_poll = ir_poll_s
        self._listen_alerts = listen_alerts
        self._kf_on_connect = request_keyframe_on_connect
        self._listener: AlertStreamListener | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> CameraNode:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.events.bind_loop(self._loop)
        if self._kf_on_connect:
            for role in (StreamRole.SUB, StreamRole.MAIN):
                self.streams.on_connected(role, self._keyframe_requester(role))
        await asyncio.to_thread(self.streams.start)
        self._tasks.append(asyncio.create_task(self._ir_loop(), name=f"{self.camera.label}-ir"))
        if self._listen_alerts:
            self._listener = AlertStreamListener(self.isapi, self.events)
            self._tasks.append(self._listener.start())

    async def stop(self) -> None:
        if self._listener is not None:
            await self._listener.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        await asyncio.to_thread(self.streams.stop)
        self.events.close()
        await self.isapi.aclose()

    def _keyframe_requester(self, role: StreamRole) -> Callable[[SessionInfo], None]:
        stream_id = self.camera.channel * 100 + (1 if role is StreamRole.MAIN else 2)

        def request(_info: SessionInfo) -> None:
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            fut = asyncio.run_coroutine_threadsafe(self.isapi.request_keyframe(stream_id), loop)

            def done(f: Any) -> None:
                if not f.cancelled() and f.exception() is not None:
                    log.debug("requestKeyFrame %d failed: %s", stream_id, f.exception())

            fut.add_done_callback(done)

        return request

    async def _ir_loop(self) -> None:
        while True:
            try:
                self.ir.update_isapi(await self.isapi.get_ircut_filter(self.camera.channel))
            except IPCamError as exc:
                log.debug("ircutFilter poll failed: %s", exc)
            await asyncio.sleep(self._ir_poll)

    # ------------------------------------------------------------------ data plane

    def latest_frame(self, *, max_age_ms: float | None = 500.0, observe_ir: bool = True) -> FrameResult | None:
        res = self.streams.sub.get_latest_frame(max_age_ms=max_age_ms)
        if res is not None and observe_ir:
            self.ir.observe_frame(res.image)
        return res

    @property
    def illumination(self) -> IlluminationState:
        return self.ir.state()

    # ------------------------------------------------------------------ actuation

    async def respond_to_violation(
        self,
        *,
        relay_output: int | None = 1,
        relay_s: float = 5.0,
        audio_id: int | None = None,
        record_s: float | None = 15.0,
        reason: str = "intrusion",
    ) -> ViolationResponse:
        """Fire relay, audio warning and evidence recording concurrently; never raises."""

        async def relay() -> None:
            if relay_output is not None:
                await self.isapi.pulse_alarm_output(relay_output, relay_s)

        async def audio() -> str | None:
            return await self.isapi.play_audio(audio_id) if audio_id is not None else None

        async def evidence() -> RecordingResult | None:
            if record_s is None:
                return None
            return await asyncio.wrap_future(self.streams.record_evidence(record_s, reason=reason))

        r, a, e = await asyncio.gather(relay(), audio(), evidence(), return_exceptions=True)
        return ViolationResponse(
            relay=r if isinstance(r, BaseException) else None,
            audio=a,
            evidence=e,
        )
