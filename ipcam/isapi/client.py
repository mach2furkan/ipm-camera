"""Asynchronous Hikvision ISAPI client (control plane).

All calls share one pooled ``httpx.AsyncClient`` (HTTP keep-alive) and one Digest
authenticator, so after the first challenge every request is a single round trip.
The client is safe to use from many tasks concurrently; physical outputs are serialised
per port so concurrent violations cannot interleave high/low commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from ..config import CameraConfig
from ..errors import AuthenticationError, ISAPIError, NotSupportedError
from . import xmlutil as X
from .audio import encode_g711, load_wav_mono16
from .digest import DigestAuth
from .models import DeviceInfo, IRCutFilterState, ResponseStatus, StreamingChannelInfo

log = logging.getLogger(__name__)

_IDEMPOTENT = frozenset({"GET", "HEAD", "PUT", "DELETE"})
_NOT_SUPPORTED_SUBCODES = frozenset({"notSupport", "notSupported", "invalidOperation", "methodNotAllowed"})


@dataclass(slots=True)
class _PulseState:
    deadline: float
    task: asyncio.Task[None]


class HikvisionISAPIClient:
    """Digest-authenticated ISAPI client.

    Usage::

        async with HikvisionISAPIClient("192.168.1.64", "admin", "secret") as cam:
            ir = await cam.get_ircut_filter()
            await cam.pulse_alarm_output(1, duration_s=5)
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int = 80,
        https: bool = False,
        verify_tls: bool = False,
        timeout_s: float = 5.0,
        max_connections: int = 8,
        retries: int = 2,
        audio_endpoint_order: tuple[str, ...] = ("system_audio_play", "audio_alarm"),
        tls_fingerprint_sha256: str | None = None,
        namespace: str | None = None,
    ) -> None:
        scheme = "https" if https else "http"
        netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host
        self.base_url = f"{scheme}://{netloc}:{port}"
        self._auth = DigestAuth(username, password)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=self._auth,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections,
                                keepalive_expiry=30.0),
            verify=verify_tls and tls_fingerprint_sha256 is None,
            headers={"User-Agent": "ipcam-isapi/0.1", "Accept": "application/xml, application/json, */*"},
            follow_redirects=False,
            event_hooks={"response": [self._check_pin]} if tls_fingerprint_sha256 else None,
        )
        # Devices ship self-signed certificates (HTTPS is on by default on this generation),
        # so CA validation is meaningless; pinning the leaf certificate's SHA-256 is the
        # practical defence against a man-in-the-middle on the camera VLAN.
        self._pin = tls_fingerprint_sha256.replace(":", "").lower() if tls_fingerprint_sha256 else None
        # ISAPI XML namespace: learnt from the first device response unless given.
        self.namespace = namespace or X.HIK_NS
        self._ns_locked = namespace is not None
        self._retries = retries
        self._output_locks: dict[int, asyncio.Lock] = {}
        self._pulses: dict[int, _PulseState] = {}
        self._audio_order = audio_endpoint_order
        self._audio_strategy: str | None = None
        self._closed = False

    @classmethod
    def from_config(cls, cfg: CameraConfig, **kwargs: Any) -> HikvisionISAPIClient:
        return cls(cfg.host, cfg.username, cfg.password, port=cfg.http_port, https=cfg.https,
                   verify_tls=cfg.verify_tls, **kwargs)

    # ------------------------------------------------------------------ lifecycle

    async def __aenter__(self) -> HikvisionISAPIClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release every relay still held by a pulse (fail-safe), then close the pool."""
        if self._closed:
            return
        self._closed = True
        # Cancelling a release task jumps straight into its ``finally`` which de-energises
        # the relay over the still-open connection pool.
        tasks = [st.task for st in self._pulses.values() if not st.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before its first step never runs its ``finally``; any output
        # still registered here was not released, so release it explicitly.
        for output_id in list(self._pulses):
            with contextlib.suppress(Exception):
                await self.set_alarm_output(output_id, False)
            self._pulses.pop(output_id, None)
        await self._client.aclose()

    # ------------------------------------------------------------------ transport

    async def request(
        self,
        method: str,
        path: str,
        *,
        content: bytes | AsyncIterator[bytes] | None = None,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        method = method.upper()
        req_headers = dict(headers or {})
        body: bytes | AsyncIterator[bytes] | None = content
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode()
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(content, (bytes, bytearray)) and content[:1] == b"<":
            req_headers.setdefault("Content-Type", "application/xml; charset=UTF-8")

        replayable = not (body is not None and not isinstance(body, (bytes, bytearray)))
        attempts = 1 + (self._retries if method in _IDEMPOTENT and replayable else 0)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = await self._client.request(
                    method, path, content=body, params=params, headers=req_headers,
                    timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                )
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
                    httpx.PoolTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.2 * (2 ** attempt))
                    continue
                raise ISAPIError(f"transport failure: {exc!r}", path=path) from exc
            self._raise_for_status(resp, path)
            return resp
        raise ISAPIError(f"transport failure: {last_exc!r}", path=path)

    async def _check_pin(self, resp: httpx.Response) -> None:
        stream = resp.extensions.get("network_stream")
        ssl_obj = stream.get_extra_info("ssl_object") if stream is not None else None
        if ssl_obj is None:
            raise ISAPIError("TLS pinning configured but the connection is not TLS", path=resp.url.path)
        import hashlib

        der = ssl_obj.getpeercert(binary_form=True) or b""
        if hashlib.sha256(der).hexdigest() != self._pin:
            raise ISAPIError("TLS certificate fingerprint mismatch (possible interception)", path=resp.url.path)

    def xml(self, root: str, fields: dict[str, Any]) -> bytes:
        """Build a request body in the namespace this device speaks."""
        return X.build(root, fields, namespace=self.namespace)

    def _learn_namespace(self, root: ET.Element) -> None:
        if self._ns_locked:
            return
        ns = X.namespace_of(root)
        if ns in (X.HIK_NS, X.ISAPI_NS):
            self.namespace = ns
            self._ns_locked = True

    @staticmethod
    def _error_fields(resp: httpx.Response) -> dict[str, Any]:
        """errorCode / errorMsg of newer firmwares (JSON and XML variants)."""
        try:
            body = resp.content
            if body.lstrip()[:1] == b"{":
                obj = json.loads(body)
                obj = obj.get("ResponseStatus", obj)
                code = obj.get("errorCode")
                return {"error_code": int(code) if code is not None else None, "error_msg": obj.get("errorMsg")}
            root = X.parse(body)
            code = X.text(root, "errorCode")
            return {"error_code": int(code, 0) if code else None, "error_msg": X.text(root, "errorMsg")}
        except (ValueError, ET.ParseError, json.JSONDecodeError, AttributeError):
            return {}

    @staticmethod
    def _parse_status(resp: httpx.Response) -> ResponseStatus | None:
        body = resp.content
        if not body:
            return None
        ctype = resp.headers.get("content-type", "")
        try:
            if "json" in ctype or body.lstrip()[:1] == b"{":
                obj = json.loads(body)
                obj = obj.get("ResponseStatus", obj)
                if "statusCode" not in obj:
                    return None
                return ResponseStatus(int(obj.get("statusCode", -1)), str(obj.get("statusString", "")),
                                      str(obj.get("subStatusCode", "")), obj.get("requestURL"))
            root = X.parse(body)
            if X.local(root.tag) != "ResponseStatus":
                return None
            return ResponseStatus.from_xml(root)
        except (ValueError, ET.ParseError, json.JSONDecodeError):
            return None

    def _raise_for_status(self, resp: httpx.Response, path: str) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("digest authentication rejected (check credentials; repeated "
                                      "failures trigger the device's IP lockout)", http_status=401, path=path)
        if resp.status_code < 400:
            return
        status = self._parse_status(resp)
        kwargs = dict(
            http_status=resp.status_code,
            status_code=status.status_code if status else None,
            sub_status_code=status.sub_status_code if status else None,
            path=path,
            **self._error_fields(resp),
        )
        msg = status.status_string if status and status.status_string else resp.reason_phrase
        if resp.status_code in (404, 405, 501) or (status and status.sub_status_code in _NOT_SUPPORTED_SUBCODES):
            raise NotSupportedError(msg or "not supported", **kwargs)
        raise ISAPIError(msg or "ISAPI request failed", **kwargs)

    async def get_xml(self, path: str, **kw: Any) -> ET.Element:
        resp = await self.request("GET", path, **kw)
        root = X.parse(resp.content)
        self._learn_namespace(root)
        return root

    async def get_json(self, path: str, params: Mapping[str, str] | None = None, **kw: Any) -> Any:
        q = {"format": "json", **(params or {})}
        resp = await self.request("GET", path, params=q, **kw)
        return json.loads(resp.content or b"{}")

    async def send_json(self, method: str, path: str, body: Mapping[str, Any],
                        params: Mapping[str, str] | None = None, **kw: Any) -> Any:
        q = {"format": "json", **(params or {})}
        resp = await self.request(method, path, params=q, json_body=body, **kw)
        return json.loads(resp.content) if resp.content.lstrip()[:1] in (b"{", b"[") else None

    async def put_form(self, path: str, parts: list[tuple[str, bytes, str]], **kw: Any) -> ResponseStatus | None:
        """PUT a multipart/form-data body; `parts` = [(name, payload, content_type)].

        Some configuration APIs (e.g. fire detection on the thermal PT series) accept their
        XML only as a named form unit rather than as a plain XML body.
        """
        import os

        boundary = os.urandom(16).hex()
        chunks: list[bytes] = []
        for name, payload, ctype in parts:
            chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n"
                           f"Content-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n\r\n").encode() + payload + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        resp = await self.request("PUT", path, content=b"".join(chunks),
                                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, **kw)
        return self._parse_status(resp)

    async def put_xml(self, path: str, body: bytes, **kw: Any) -> ResponseStatus | None:
        resp = await self.request("PUT", path, content=body, **kw)
        return self._parse_status(resp)

    @contextlib.asynccontextmanager
    async def open_stream(self, path: str, *, idle_timeout_s: float, params: Mapping[str, str] | None = None
                          ) -> AsyncIterator[httpx.Response]:
        """Long-lived GET whose read timeout doubles as an idle watchdog."""
        timeout = httpx.Timeout(connect=5.0, read=idle_timeout_s, write=5.0, pool=5.0)
        async with self._client.stream("GET", path, params=params, timeout=timeout) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_for_status(resp, path)
            yield resp

    # ------------------------------------------------------------------ device / streaming

    async def get_device_info(self) -> DeviceInfo:
        return DeviceInfo.from_xml(await self.get_xml("/ISAPI/System/deviceInfo"))

    async def get_streaming_channel(self, stream_id: int) -> StreamingChannelInfo:
        """``stream_id`` is the RTSP channel id, e.g. 101 (main) or 102 (sub)."""
        return StreamingChannelInfo.from_xml(await self.get_xml(f"/ISAPI/Streaming/channels/{stream_id}"))

    async def request_keyframe(self, stream_id: int) -> None:
        """Force an immediate IDR so a freshly (re)connected decoder starts in < 1 frame
        instead of waiting out a multi-second GOP."""
        await self.request("PUT", f"/ISAPI/Streaming/channels/{stream_id}/requestKeyFrame")

    # ------------------------------------------------------------------ image / IR

    async def get_ircut_filter(self, channel: int = 1) -> IRCutFilterState:
        try:
            root = await self.get_xml(f"/ISAPI/Image/channels/{channel}/ircutFilter")
        except NotSupportedError:
            root = await self.get_xml(f"/ISAPI/Image/channels/{channel}/IrcutFilter")
        return IRCutFilterState.from_xml(root)

    # ------------------------------------------------------------------ PTZ

    async def ptz_absolute(self, azimuth_deg: float, elevation_deg: float, zoom: float, *, channel: int = 1) -> None:
        """3D absolute positioning (``AbsoluteHigh``): angles in degrees, zoom in x."""
        body = self.xml("PTZData", {"AbsoluteHigh": {
            "elevation": int(round(elevation_deg * 10)),
            "azimuth": int(round(azimuth_deg * 10)) % 3600,
            "absoluteZoom": max(10, int(round(zoom * 10))),
        }})
        await self.put_xml(f"/ISAPI/PTZCtrl/channels/{channel}/absolute", body)

    async def ptz_continuous(self, pan: int, tilt: int, zoom: int = 0, *, channel: int = 1) -> None:
        """Continuous move, speeds in [-100, 100]; (0, 0, 0) stops."""
        clamp = lambda v: max(-100, min(100, int(v)))  # noqa: E731
        body = self.xml("PTZData", {"pan": clamp(pan), "tilt": clamp(tilt), "zoom": clamp(zoom)})
        await self.put_xml(f"/ISAPI/PTZCtrl/channels/{channel}/continuous", body)

    async def get_ptz_status(self, *, channel: int = 1) -> Any:
        from ..fusion.ptz import PTZPose

        root = await self.get_xml(f"/ISAPI/PTZCtrl/channels/{channel}/status")
        ah = X.find(root, "AbsoluteHigh")
        if ah is None:
            return None
        num = lambda k: float(X.text(ah, k) or 0) / 10.0  # noqa: E731
        return PTZPose(num("azimuth"), num("elevation"), num("absoluteZoom"))

    async def ptz_goto_preset(self, preset: int, *, channel: int = 1) -> None:
        await self.request("PUT", f"/ISAPI/PTZCtrl/channels/{channel}/presets/{preset}/goto")

    # ------------------------------------------------------------------ alarm outputs

    def _lock_for(self, output_id: int) -> asyncio.Lock:
        lock = self._output_locks.get(output_id)
        if lock is None:
            lock = self._output_locks[output_id] = asyncio.Lock()
        return lock

    async def set_alarm_output(self, output_id: int = 1, active: bool = True) -> ResponseStatus | None:
        body = self.xml("IOPortData", {"outputState": "high" if active else "low"})
        async with self._lock_for(output_id):
            return await self.put_xml(f"/ISAPI/System/IO/outputs/{output_id}/trigger", body)

    async def get_alarm_output_active(self, output_id: int = 1) -> bool:
        root = await self.get_xml(f"/ISAPI/System/IO/outputs/{output_id}/status")
        return (X.text(root, "ioState") or "").lower() == "active"

    async def pulse_alarm_output(self, output_id: int = 1, duration_s: float = 5.0) -> None:
        """Retriggerable monostable: energise the relay for ``duration_s``.

        A new violation during an active pulse extends the deadline instead of toggling
        the relay (no siren chatter). Release is guaranteed on cancellation and on
        :meth:`aclose`, and is retried because a relay stuck "high" is the worst outcome.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_s
        state = self._pulses.get(output_id)
        if state is not None and not state.task.done():
            state.deadline = max(state.deadline, deadline)
            return
        await self.set_alarm_output(output_id, True)
        task = asyncio.create_task(self._pulse_release(output_id), name=f"relay-{output_id}-release")
        self._pulses[output_id] = _PulseState(deadline, task)

    async def _pulse_release(self, output_id: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                st = self._pulses.get(output_id)
                if st is None:
                    return
                remaining = st.deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
        finally:
            for attempt in range(5):
                try:
                    await asyncio.shield(self.set_alarm_output(output_id, False))
                    break
                except asyncio.CancelledError:
                    continue
                except Exception:
                    log.warning("relay %d release attempt %d failed", output_id, attempt + 1, exc_info=True)
                    await asyncio.sleep(0.5 * (attempt + 1))
            self._pulses.pop(output_id, None)

    # ------------------------------------------------------------------ audio

    async def play_audio(self, audio_id: int, *, channel: int = 1, volume: int | None = None,
                         repeat: int = 1) -> str:
        """Play a clip pre-loaded on the camera; returns the endpoint strategy that worked.

        The endpoint differs by product line, so strategies are probed in order and the
        first successful one is cached:

        * ``system_audio_play``: ``PUT /ISAPI/System/Audio/channels/<ch>/play``
        * ``audio_alarm``: configure ``/ISAPI/Event/triggers/notifications/AudioAlarm`` and
          fire its ``AudioTest`` (AcuSense / ColorVu "Strobe light & audible warning").
        """
        order = (self._audio_strategy,) if self._audio_strategy else self._audio_order
        last: Exception | None = None
        for strategy in order:
            try:
                if strategy == "system_audio_play":
                    fields: dict[str, Any] = {"audioID": audio_id, "playTimes": repeat}
                    if volume is not None:
                        fields["volume"] = volume
                    await self.put_xml(f"/ISAPI/System/Audio/channels/{channel}/play",
                                       self.xml("AudioPlay", fields))
                elif strategy == "audio_alarm":
                    cfg: dict[str, Any] = {"audioID": audio_id, "alarmTimes": repeat}
                    if volume is not None:
                        cfg["audioVolume"] = volume
                    await self.request("PUT", "/ISAPI/Event/triggers/notifications/AudioAlarm",
                                       params={"format": "json"}, json_body={"AudioAlarm": cfg})
                    await self.request("PUT", "/ISAPI/Event/triggers/notifications/AudioAlarm/AudioTest",
                                       params={"format": "json"}, json_body={"AudioTest": cfg})
                else:
                    raise ValueError(f"unknown audio strategy {strategy!r}")
            except NotSupportedError as exc:
                last = exc
                continue
            self._audio_strategy = strategy
            return strategy
        raise NotSupportedError(f"no audio playback endpoint accepted the request ({last})",
                                path="audio")

    async def stream_pcm(
        self,
        pcm: npt.NDArray[np.int16],
        sample_rate: int,
        *,
        channel: int = 1,
        chunk_ms: int = 500,
        lead_ms: int = 300,
    ) -> None:
        """Speak arbitrary PCM through the camera speaker via the two-way audio channel.

        Audio is transcoded to the codec the camera advertises (G.711 µ-law/A-law, 8 kHz)
        and uploaded in real-time paced chunks, staying ``lead_ms`` ahead of playback so the
        device jitter buffer neither starves nor overflows.
        """
        base = f"/ISAPI/System/TwoWayAudio/channels/{channel}"
        info = await self.get_xml(base)
        codec = X.text(info, "audioCompressionType") or "G.711ulaw"
        payload = encode_g711(pcm, sample_rate, codec)
        opened = await self.request("PUT", f"{base}/open")
        session_id = None
        with contextlib.suppress(Exception):
            session_id = X.text(X.parse(opened.content), "sessionId")
        params = {"sessionId": session_id} if session_id else None
        bytes_per_ms = 8  # 8 kHz * 1 byte/sample
        step = chunk_ms * bytes_per_ms
        t0 = time.monotonic()
        try:
            for offset in range(0, len(payload), step):
                chunk = payload[offset: offset + step]
                await self.request("PUT", f"{base}/audioData", content=chunk, params=params,
                                   headers={"Content-Type": "application/octet-stream"})
                sent_ms = (offset + len(chunk)) / bytes_per_ms
                wait = (sent_ms - lead_ms) / 1000.0 - (time.monotonic() - t0)
                if wait > 0:
                    await asyncio.sleep(wait)
            tail = len(payload) / bytes_per_ms / 1000.0 - (time.monotonic() - t0)
            if tail > 0:
                await asyncio.sleep(tail)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.shield(self.request("PUT", f"{base}/close", params=params))

    async def play_wav(self, path: str | Path, *, channel: int = 1) -> None:
        pcm, rate = await asyncio.to_thread(load_wav_mono16, path)
        await self.stream_pcm(pcm, rate, channel=channel)
