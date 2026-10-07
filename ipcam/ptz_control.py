"""Manual bounded PTZ pulses. Independent of detection, no automatic target tracking."""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor

import httpx
from ipcam.isapi.client import HikvisionISAPIClient

DIRECTIONS = {"left": (-1, 0, 0), "right": (1, 0, 0), "up": (0, 1, 0),
              "down": (0, -1, 0), "zoom_in": (0, 0, 1), "zoom_out": (0, 0, -1)}
DAHUA_CODES = dict(left="Left", right="Right", up="Up", down="Down", zoom_in="ZoomTele", zoom_out="ZoomWide")


_session = ContextVar("ptz_session", default=None)


@asynccontextmanager
async def command_client(camera, protocol):
    """Reuse a worker's authenticated connection on its owning event loop."""
    existing = _session.get()
    if existing is not None:
        yield existing
        return
    if protocol == "ISAPI":
        client = HikvisionISAPIClient.from_config(camera, timeout_s=2, retries=0)
    else:
        host = f"[{camera.host}]" if ":" in camera.host else camera.host
        base = f"{'https' if camera.https else 'http'}://{host}:{camera.http_port}"
        client = httpx.AsyncClient(base_url=base, auth=httpx.DigestAuth(camera.username, camera.password),
            timeout=2, trust_env=False, follow_redirects=False, verify=camera.verify_tls)
    async with client:
        yield client


async def send_stop(camera, protocol, channel, direction, speed):
    async with command_client(camera, protocol) as client:
        if protocol == "ISAPI":
            await client.ptz_continuous(0, 0, 0, channel=channel)
        else:
            response = await client.get("/cgi-bin/ptz.cgi", params=dict(action="stop", channel=channel,
                code=DAHUA_CODES[direction], arg1=0, arg2=max(1, min(8, round(speed*8/100))), arg3=0))
            response.raise_for_status()
            if response.text.strip() != "OK":
                raise ValueError("Camera rejected PTZ stop")


def validate_command(protocol, channel, direction, speed, duration):
    if direction not in DIRECTIONS or protocol not in ("ISAPI", "Dahua"):
        raise ValueError("Unsupported PTZ command")
    if not 1 <= speed <= 100 or not .05 <= duration <= .5:
        raise ValueError("Invalid PTZ speed or pulse duration")
    if not 0 <= channel <= 999 or (protocol == "ISAPI" and channel == 0):
        raise ValueError("Invalid PTZ channel")


async def move_pulse(camera, protocol, channel, direction, speed, duration, stop_event):
    """Every start attempt is followed by a stop attempt, including timeouts/errors."""
    validate_command(protocol, channel, direction, speed, duration)
    if stop_event.is_set():
        return
    if protocol == "ISAPI":
        async with command_client(camera, protocol) as client:
            try:
                pan, tilt, zoom = (v*speed for v in DIRECTIONS[direction])
                await client.ptz_continuous(pan, tilt, zoom, channel=channel)
                await asyncio.to_thread(stop_event.wait, duration)
            finally:
                await client.ptz_continuous(0, 0, 0, channel=channel)
    else:
        params = dict(channel=channel, code=DAHUA_CODES[direction], arg1=0,
                      arg2=max(1, min(8, round(speed*8/100))), arg3=0)
        async with command_client(camera, protocol) as client:
            async def command(action):
                response = await client.get("/cgi-bin/ptz.cgi", params={**params, "action": action})
                response.raise_for_status()
                if response.text.strip() != "OK":
                    raise ValueError("Camera rejected PTZ command")
            try:
                await command("start")
                await asyncio.to_thread(stop_event.wait, duration)
            finally:
                await command("stop")


class ManualPTZ:
    """Only one pending movement; busy clicks are rejected rather than queued."""
    def __init__(self, camera, protocol="ISAPI", channel=1):
        self.camera, self.protocol, self.channel = camera, protocol, channel
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manual-ptz")
        self.stop_event = threading.Event()
        self.future = None
        self.closed = False
        self.lock = threading.Lock()
        self.last_command = None
        self.motion_serial = 0
        self._loop = None
        self._client_context = None
        self._client = None

    def _execute(self, operation, *args):
        # All connection creation, requests and cleanup run on one worker/loop.
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        async def run():
            if self._client is None:
                self._client_context = command_client(self.camera, self.protocol)
                self._client = await self._client_context.__aenter__()
            token = _session.set(self._client)
            try:
                return await operation(*args)
            finally:
                _session.reset(token)
        return self._loop.run_until_complete(run())

    def _cleanup(self):
        if self._loop is not None:
            try:
                if self._client_context is not None:
                    self._loop.run_until_complete(self._client_context.__aexit__(None, None, None))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
            finally:
                self._loop.close()

    def pulse(self, direction, speed=30, duration=.25):
        with self.lock:
            if self.closed or (self.future is not None and not self.future.done()):
                return False
            validate_command(self.protocol, self.channel, direction, speed, duration)
            self.stop_event.clear()
            self.last_command = (direction, speed)
            self.motion_serial += 1
            self.future = self.executor.submit(self._execute,
                move_pulse, self.camera, self.protocol, self.channel, direction, speed, duration, self.stop_event)
            return True

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if (not self.closed and self.last_command is not None
                    and (self.future is None or self.future.done())):
                direction, speed = self.last_command
                self.future = self.executor.submit(self._execute,
                    send_stop, self.camera, self.protocol, self.channel, direction, speed)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop_event.set()
        cleanup = self.executor.submit(self._cleanup)
        try:
            cleanup.result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)
