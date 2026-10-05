"""Manual bounded PTZ pulses. Independent of detection, no automatic target tracking."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
from ipcam.isapi.client import HikvisionISAPIClient

DIRECTIONS = {"left": (-1, 0, 0), "right": (1, 0, 0), "up": (0, 1, 0),
              "down": (0, -1, 0), "zoom_in": (0, 0, 1), "zoom_out": (0, 0, -1)}
DAHUA_CODES = dict(left="Left", right="Right", up="Up", down="Down", zoom_in="ZoomTele", zoom_out="ZoomWide")


async def send_stop(camera, protocol, channel, direction, speed):
    if protocol == "ISAPI":
        async with HikvisionISAPIClient.from_config(camera, timeout_s=2, retries=0) as client:
            await client.ptz_continuous(0, 0, 0, channel=channel)
    else:
        host = f"[{camera.host}]" if ":" in camera.host else camera.host
        base = f"{'https' if camera.https else 'http'}://{host}:{camera.http_port}"
        async with httpx.AsyncClient(base_url=base, auth=httpx.DigestAuth(camera.username, camera.password),
            timeout=2, trust_env=False, follow_redirects=False, verify=camera.verify_tls) as client:
            response = await client.get("/cgi-bin/ptz.cgi", params=dict(action="stop", channel=channel,
                code=DAHUA_CODES[direction], arg1=0, arg2=max(1, min(8, round(speed*8/100))), arg3=0))
            response.raise_for_status()
            if response.text.strip() != "OK":
                raise ValueError("Camera rejected PTZ stop")


async def move_pulse(camera, protocol, channel, direction, speed, duration, stop_event):
    """Every start attempt is followed by a stop attempt, including timeouts/errors."""
    if direction not in DIRECTIONS or protocol not in ("ISAPI", "Dahua"):
        raise ValueError("Unsupported PTZ command")
    if not 1 <= speed <= 100 or not .05 <= duration <= .5:
        raise ValueError("Invalid PTZ speed or pulse duration")
    if not 0 <= channel <= 999 or (protocol == "ISAPI" and channel == 0):
        raise ValueError("Invalid PTZ channel")
    if stop_event.is_set():
        return
    if protocol == "ISAPI":
        async with HikvisionISAPIClient.from_config(camera, timeout_s=2, retries=0) as client:
            try:
                pan, tilt, zoom = (v*speed for v in DIRECTIONS[direction])
                await client.ptz_continuous(pan, tilt, zoom, channel=channel)
                await asyncio.to_thread(stop_event.wait, duration)
            finally:
                await client.ptz_continuous(0, 0, 0, channel=channel)
    else:
        host = f"[{camera.host}]" if ":" in camera.host else camera.host
        base = f"{'https' if camera.https else 'http'}://{host}:{camera.http_port}"
        params = dict(channel=channel, code=DAHUA_CODES[direction], arg1=0,
                      arg2=max(1, min(8, round(speed*8/100))), arg3=0)
        async with httpx.AsyncClient(base_url=base, auth=httpx.DigestAuth(camera.username, camera.password),
            timeout=2, trust_env=False, follow_redirects=False, verify=camera.verify_tls) as client:
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

    def pulse(self, direction, speed=30, duration=.25):
        with self.lock:
            if self.closed or (self.future is not None and not self.future.done()):
                return False
            self.stop_event.clear()
            self.last_command = (direction, speed)
            self.motion_serial += 1
            self.future = self.executor.submit(asyncio.run,
                move_pulse(self.camera, self.protocol, self.channel, direction, speed, duration, self.stop_event))
            return True

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if (not self.closed and self.last_command is not None
                    and (self.future is None or self.future.done())):
                direction, speed = self.last_command
                self.future = self.executor.submit(asyncio.run,
                    send_stop(self.camera, self.protocol, self.channel, direction, speed))

    def close(self):
        with self.lock:
            self.closed = True
            self.stop_event.set()
        self.executor.shutdown(wait=True, cancel_futures=False)
