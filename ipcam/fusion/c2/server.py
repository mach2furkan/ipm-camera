"""Minimal, dependency-free C2 web server for the operator station.

Routes
  GET  /                       operator console (``static/index.html``)
  GET  /api/cop                Common Operating Picture snapshot (JSON)
  GET  /api/stream             Server-Sent Events, one snapshot per fusion tick
  GET  /api/thermal/state      thermal statistics, hotspots, ROI state, temperature grid
  GET  /api/thermal/frame.png  false-colour frame (``?palette=iron|white_hot|...``)
  POST /api/ptz                PTZ command (JSON), requires header ``X-C2-Token``
  GET  /healthz                liveness

Built on ``asyncio.start_server`` so fusion, thermal analytics and the UI share one event
loop. Control requests must carry a per-process token in a custom header: browsers cannot
add custom headers to cross-origin requests without a CORS pre-flight (which this server
never grants), so a page on another origin cannot drive the PTZ. Bind to a management
network; there is no user authentication of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from ..service import FusionService
from .thermal_panel import ThermalPanel

log = logging.getLogger(__name__)

_STATIC = Path(__file__).with_name("static")
_MAX_BODY = 64 * 1024

PTZControl = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _clean(obj: Any) -> Any:
    """NaN/inf are not valid JSON; browsers' JSON.parse rejects them."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _dumps(obj: Any) -> bytes:
    return json.dumps(_clean(obj), default=_json_default, separators=(",", ":"), allow_nan=False).encode()


class C2Server:
    def __init__(self, service: FusionService, *, host: str = "127.0.0.1", port: int = 8080,
                 thermal: ThermalPanel | None = None, ptz_control: PTZControl | None = None,
                 extra_state: Callable[[], dict[str, Any]] | None = None) -> None:
        self.service = service
        self.host = host
        self.port = port
        self.thermal = thermal
        self.ptz_control = ptz_control
        self.extra_state = extra_state
        self.token = secrets.token_urlsafe(24)
        self._server: asyncio.base_events.Server | None = None
        self._index_tpl = (_STATIC / "index.html").read_text(encoding="utf-8")

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("C2 operator station on http://%s:%d", self.host, self.port)
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def snapshot(self) -> dict[str, Any]:
        snap = self.service.snapshot()
        snap["capabilities"] = {"thermal": self.thermal is not None, "ptz_control": self.ptz_control is not None}
        if self.extra_state is not None:
            snap.update(self.extra_state())
        return snap

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        method = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else "/"
        path, _, query = target.partition("?")
        headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if ":" in l)}
        try:
            if method == "GET":
                await self._get(writer, path, parse_qs(query))
                if path == "/api/stream":
                    return
            elif method == "POST" and path == "/api/ptz":
                await self._ptz(reader, writer, headers)
            else:
                await self._send(writer, 405 if path in ("/api/ptz",) else 404, b"not found", "text/plain")
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _get(self, writer: asyncio.StreamWriter, path: str, q: dict[str, list[str]]) -> None:
        if path in ("/", "/index.html"):
            page = self._index_tpl.replace("__C2_TOKEN__", self.token).encode()
            await self._send(writer, 200, page, "text/html; charset=utf-8")
        elif path == "/api/cop":
            await self._send(writer, 200, _dumps(self.snapshot()), "application/json")
        elif path == "/api/stream":
            await self._sse(writer)
        elif path == "/api/thermal/state":
            body = _dumps(self.thermal.state() if self.thermal else {"available": False})
            await self._send(writer, 200, body, "application/json")
        elif path == "/api/thermal/frame.png":
            png = self.thermal.png(q.get("palette", ["iron"])[0]) if self.thermal else None
            if png is None:
                await self._send(writer, 503, b"no thermal frame", "text/plain")
            else:
                await self._send(writer, 200, png, "image/png")
        elif path == "/healthz":
            await self._send(writer, 200, b"ok", "text/plain")
        else:
            await self._send(writer, 404, b"not found", "text/plain")

    async def _ptz(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        if not secrets.compare_digest(headers.get("x-c2-token", ""), self.token):
            await self._send(writer, 403, b'{"error":"token"}', "application/json")
            return
        if self.ptz_control is None:
            await self._send(writer, 501, b'{"error":"no ptz control configured"}', "application/json")
            return
        n = int(headers.get("content-length", "0") or 0)
        if n <= 0 or n > _MAX_BODY:
            await self._send(writer, 400, b'{"error":"body"}', "application/json")
            return
        try:
            cmd = json.loads(await asyncio.wait_for(reader.readexactly(n), 5))
            if not isinstance(cmd, dict):
                raise ValueError("object expected")
            result = await self.ptz_control(cmd)
            await self._send(writer, 200, _dumps({"ok": True, **result}), "application/json")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            await self._send(writer, 400, _dumps({"error": str(exc)}), "application/json")
        except Exception as exc:  # noqa: BLE001 - surface device errors to the operator
            log.warning("PTZ command failed: %s", exc)
            await self._send(writer, 502, _dumps({"error": str(exc)}), "application/json")

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, status: int, body: bytes, ctype: str) -> None:
        reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                  501: "Not Implemented", 502: "Bad Gateway", 503: "Service Unavailable"}.get(status, "OK")
        writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                     f"Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n"
                     .encode() + body)
        await writer.drain()

    async def _sse(self, writer: asyncio.StreamWriter) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                     b"Connection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n")
        q = self.service.subscribe_cop()
        try:
            writer.write(b"data: " + _dumps(self.snapshot()) + b"\n\n")
            await writer.drain()
            while True:
                await q.get()
                writer.write(b"data: " + _dumps(self.snapshot()) + b"\n\n")
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.service.unsubscribe_cop(q)
            with contextlib.suppress(Exception):
                writer.close()
