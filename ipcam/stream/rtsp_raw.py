"""Minimal asyncio RTSP/1.0 client for non-video payloads (thermal raw data, ISAPI metadata).

FFmpeg demuxes H.264/H.265 well but drops unknown RTP payloads such as the thermal PT
series' ``thermalStream`` (PT 109, trackID=5) and ``isapi.metadata`` (PT 107). This client
speaks just enough RTSP to get them, robustly:

* RTP over the RTSP TCP connection (interleaved ``$`` framing) -- no UDP loss, no NAT issues;
* Digest (RFC 2617/7616) and Basic authentication, with nonce reuse across requests;
* SDP parsing and track selection by ``rtpmap`` encoding name;
* RTP header parsing including CSRC lists, header extensions and padding;
* frame reassembly by marker bit with sequence-gap detection: a frame missing a fragment is
  discarded whole instead of delivering a shifted temperature matrix;
* session keep-alive (GET_PARAMETER, falling back to OPTIONS) at half the session timeout.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

from ..isapi.digest import DigestChallenge, compute_response, parse_challenges

log = logging.getLogger(__name__)


class RtspError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# ====================================================================== SDP

@dataclass
class SdpMedia:
    media: str
    payload_type: int
    encoding: str = ""
    clock_rate: int = 0
    control: str = ""
    attrs: dict[str, str] = field(default_factory=dict)


def parse_sdp(text: str) -> tuple[dict[str, str], list[SdpMedia]]:
    session: dict[str, str] = {}
    medias: list[SdpMedia] = []
    cur: SdpMedia | None = None
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if len(line) < 2 or line[1] != "=":
            continue
        k, v = line[0], line[2:].strip()
        if k == "m":
            parts = v.split()
            cur = SdpMedia(parts[0], int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1)
            medias.append(cur)
        elif k == "a":
            name, _, val = v.partition(":")
            target = cur.attrs if cur is not None else session
            target[name] = val
            if cur is not None:
                if name == "rtpmap":
                    m = re.match(r"(\d+)\s+([^/\s]+)(?:/(\d+))?", val)
                    if m:
                        cur.encoding = m.group(2)
                        cur.clock_rate = int(m.group(3) or 0)
                elif name == "control":
                    cur.control = val
    return session, medias


def resolve_control(base: str, control: str) -> str:
    if not control or control == "*":
        return base
    if control.startswith("rtsp://") or control.startswith("rtsps://"):
        return control
    return base.rstrip("/") + "/" + control.lstrip("/")


# ====================================================================== RTP

@dataclass(frozen=True, slots=True)
class RtpPacket:
    marker: bool
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    extension: bytes | None = None


def parse_rtp(data: bytes) -> RtpPacket:
    if len(data) < 12:
        raise ValueError("RTP packet shorter than fixed header")
    b0, b1, seq, ts, ssrc = struct.unpack_from("!BBHII", data, 0)
    if b0 >> 6 != 2:
        raise ValueError(f"unsupported RTP version {b0 >> 6}")
    padding = bool(b0 & 0x20)
    has_ext = bool(b0 & 0x10)
    cc = b0 & 0x0F
    off = 12 + 4 * cc
    ext = None
    if has_ext:
        if len(data) < off + 4:
            raise ValueError("truncated RTP header extension")
        _profile, words = struct.unpack_from("!HH", data, off)
        ext = data[off + 4: off + 4 + 4 * words]
        off += 4 + 4 * words
    end = len(data)
    if padding:
        pad = data[-1]
        if pad == 0 or pad > end - off:
            raise ValueError("invalid RTP padding")
        end -= pad
    if off > end:
        raise ValueError("RTP header longer than packet")
    return RtpPacket(bool(b1 & 0x80), b1 & 0x7F, seq, ts, ssrc, data[off:end], ext)


@dataclass(frozen=True, slots=True)
class AssembledFrame:
    payload_type: int
    timestamp: int
    data: bytes
    packets: int


class FrameAssembler:
    """Concatenates RTP payloads until the marker bit; drops frames with sequence gaps."""

    def __init__(self, *, max_frame_bytes: int = 64 * 1024 * 1024) -> None:
        self._parts: list[bytes] = []
        self._size = 0
        self._ts: int | None = None
        self._expect: int | None = None
        self._corrupt = False
        self._max = max_frame_bytes
        self.frames = 0
        self.dropped = 0
        self.lost_packets = 0

    def push(self, pkt: RtpPacket) -> AssembledFrame | None:
        if self._expect is not None and pkt.sequence != self._expect:
            gap = (pkt.sequence - self._expect) & 0xFFFF
            self.lost_packets += gap if gap < 0x8000 else 0
            self._corrupt = True
        self._expect = (pkt.sequence + 1) & 0xFFFF
        if self._ts is not None and pkt.timestamp != self._ts and self._parts:
            # A new timestamp without a marker on the previous frame: its tail was lost.
            self._reset(drop=True)
        self._ts = pkt.timestamp
        self._parts.append(pkt.payload)
        self._size += len(pkt.payload)
        if self._size > self._max:
            self._reset(drop=True)
            return None
        if not pkt.marker:
            return None
        frame = None
        if not self._corrupt:
            frame = AssembledFrame(pkt.payload_type, pkt.timestamp, b"".join(self._parts), len(self._parts))
            self.frames += 1
        self._reset(drop=frame is None)
        return frame

    def _reset(self, *, drop: bool) -> None:
        if drop and self._parts:
            self.dropped += 1
        self._parts, self._size, self._ts, self._corrupt = [], 0, None, False


# ====================================================================== RTSP client

@dataclass
class RtspResponse:
    status: int
    reason: str
    headers: dict[str, str]
    body: bytes


class RtspClient:
    def __init__(self, url: str, *, username: str | None = None, password: str | None = None,
                 timeout_s: float = 5.0, user_agent: str = "ipcam-rtsp/0.1") -> None:
        parts = urlsplit(url)
        if parts.scheme != "rtsp":
            raise ValueError("only rtsp:// URLs are supported")
        self.host = parts.hostname or ""
        self.port = parts.port or 554
        self.username = username if username is not None else (unquote(parts.username) if parts.username else None)
        self.password = password if password is not None else (unquote(parts.password) if parts.password else None)
        path = parts.path or "/"
        self.url = f"rtsp://{self.host}:{self.port}{path}" + (f"?{parts.query}" if parts.query else "")
        self.timeout = timeout_s
        self.user_agent = user_agent
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._cseq = 0
        self._session: str | None = None
        self.session_timeout_s = 60.0
        self._challenge: DigestChallenge | None = None
        self._basic = False
        self._nc = 0
        self._pending: dict[int, asyncio.Future[RtspResponse]] = {}
        self._rtp: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue(maxsize=4096)
        self._read_task: asyncio.Task[None] | None = None
        self._ka_task: asyncio.Task[None] | None = None
        self._ka_method = "GET_PARAMETER"
        self.medias: list[SdpMedia] = []
        self.content_base = self.url
        self.dropped_rtp = 0

    # ------------------------------------------------------------------ connection

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), self.timeout)
        self._read_task = asyncio.create_task(self._read_loop(), name=f"rtsp-read-{self.host}")

    async def close(self) -> None:
        for t in (self._ka_task, self._read_task):
            if t is not None:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._writer = self._reader = None

    async def __aenter__(self) -> RtspClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            if self._session:
                await asyncio.wait_for(self.request("TEARDOWN", self.url), 2.0)
        await self.close()

    # ------------------------------------------------------------------ framing

    async def _read_loop(self) -> None:
        r = self._reader
        assert r is not None
        try:
            while True:
                first = await r.readexactly(1)
                if first == b"$":
                    ch, ln = struct.unpack("!BH", await r.readexactly(3))
                    data = await r.readexactly(ln)
                    try:
                        self._rtp.put_nowait((ch, data))
                    except asyncio.QueueFull:
                        self.dropped_rtp += 1   # consumer too slow: shed load, never block the socket
                    continue
                head = first + await r.readuntil(b"\r\n\r\n")
                text = head.decode("latin-1")
                lines = text.split("\r\n")
                m = re.match(r"RTSP/\d\.\d\s+(\d{3})\s*(.*)", lines[0])
                if not m:
                    # Server-initiated request (e.g. ANNOUNCE) or garbage: skip its headers.
                    continue
                headers: dict[str, str] = {}
                for line in lines[1:]:
                    k, sep, v = line.partition(":")
                    if sep:
                        key = k.strip().lower()
                        headers[key] = (headers[key] + ", " + v.strip()) if key in headers else v.strip()
                body = b""
                if (cl := headers.get("content-length")) and cl.isdigit():
                    body = await r.readexactly(int(cl))
                resp = RtspResponse(int(m.group(1)), m.group(2), headers, body)
                cseq = int(headers.get("cseq", "-1").split(",")[0]) if headers.get("cseq", "").split(",")[0].strip().isdigit() else -1
                fut = self._pending.pop(cseq, None)
                if fut is not None and not fut.done():
                    fut.set_result(resp)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError) as exc:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RtspError(f"connection lost: {exc!r}"))
            self._pending.clear()
            with contextlib.suppress(asyncio.QueueFull):
                self._rtp.put_nowait(None)

    # ------------------------------------------------------------------ requests

    def _authorization(self, method: str, uri: str) -> str | None:
        if self.username is None:
            return None
        if self._basic:
            tok = base64.b64encode(f"{self.username}:{self.password or ''}".encode()).decode()
            return f"Basic {tok}"
        ch = self._challenge
        if ch is None:
            return None
        self._nc += 1
        cnonce = os.urandom(8).hex()
        response, qop = compute_response(challenge=ch, username=self.username, password=self.password or "",
                                         method=method, uri=uri, nc=self._nc, cnonce=cnonce)
        fields = [f'username="{self.username}"', f'realm="{ch.realm}"', f'nonce="{ch.nonce}"', f'uri="{uri}"',
                  f'response="{response}"']
        if ch.algorithm and ch.algorithm != "MD5":
            fields.append(f"algorithm={ch.algorithm}")
        if ch.opaque is not None:
            fields.append(f'opaque="{ch.opaque}"')
        if qop:
            fields += [f"qop={qop}", f"nc={self._nc:08x}", f'cnonce="{cnonce}"']
        return "Digest " + ", ".join(fields)

    async def request(self, method: str, uri: str, headers: dict[str, str] | None = None,
                      *, _retry: bool = True) -> RtspResponse:
        if self._writer is None:
            raise RtspError("not connected")
        self._cseq += 1
        cseq = self._cseq
        lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {cseq}", f"User-Agent: {self.user_agent}"]
        auth = self._authorization(method, uri)
        if auth:
            lines.append(f"Authorization: {auth}")
        if self._session and method not in ("OPTIONS", "DESCRIBE"):
            lines.append(f"Session: {self._session}")
        for k, v in (headers or {}).items():
            lines.append(f"{k}: {v}")
        fut: asyncio.Future[RtspResponse] = asyncio.get_running_loop().create_future()
        self._pending[cseq] = fut
        self._writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
        await self._writer.drain()
        try:
            resp = await asyncio.wait_for(fut, self.timeout)
        finally:
            self._pending.pop(cseq, None)
        if resp.status == 401 and _retry and self.username is not None:
            hdr = resp.headers.get("www-authenticate", "")
            challenges = parse_challenges([hdr])
            stale = any(c.stale for c in challenges)
            if challenges and (self._challenge is None or stale or challenges[0].nonce != self._challenge.nonce):
                self._challenge = max(challenges, key=lambda c: c.strength)
                self._nc = 0
                return await self.request(method, uri, headers, _retry=False)
            if hdr.lower().startswith("basic") and not self._basic:
                self._basic = True
                return await self.request(method, uri, headers, _retry=False)
        if resp.status >= 400:
            raise RtspError(f"{method} {self._redact(uri)} -> {resp.status} {resp.reason}", resp.status)
        if "session" in resp.headers:
            sess, *params = resp.headers["session"].split(";")
            self._session = sess.strip()
            for p in params:
                k, _, v = p.strip().partition("=")
                if k.lower() == "timeout" and v.strip().isdigit():
                    self.session_timeout_s = float(v)
        return resp

    def _redact(self, uri: str) -> str:
        return re.sub(r"//[^/@]+@", "//***@", uri)

    # ------------------------------------------------------------------ high level

    async def describe(self) -> list[SdpMedia]:
        resp = await self.request("DESCRIBE", self.url, {"Accept": "application/sdp"})
        base = resp.headers.get("content-base") or resp.headers.get("content-location") or self.url
        self.content_base = base
        _, self.medias = parse_sdp(resp.body.decode("utf-8", "replace"))
        return self.medias

    async def setup(self, media: SdpMedia, interleaved: tuple[int, int]) -> None:
        uri = resolve_control(self.content_base, media.control)
        await self.request("SETUP", uri, {"Transport": f"RTP/AVP/TCP;unicast;interleaved={interleaved[0]}-{interleaved[1]}"})

    async def play(self) -> None:
        await self.request("PLAY", self.url, {"Range": "npt=0.000-"})
        self._ka_task = asyncio.create_task(self._keepalive(), name="rtsp-keepalive")

    async def _keepalive(self) -> None:
        period = max(5.0, self.session_timeout_s / 2)
        while True:
            await asyncio.sleep(period)
            try:
                await self.request(self._ka_method, self.url)
            except RtspError as exc:
                if self._ka_method == "GET_PARAMETER" and exc.status in (405, 451, 501):
                    self._ka_method = "OPTIONS"
                    continue
                log.warning("RTSP keep-alive failed: %s", exc)
                return

    async def packets(self) -> AsyncIterator[tuple[int, bytes]]:
        while True:
            item = await self._rtp.get()
            if item is None:
                raise RtspError("connection closed by server")
            yield item


def with_credentials(url: str, username: str, password: str) -> str:
    p = urlsplit(url)
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{p.hostname}:{p.port or 554}"
    return p._replace(netloc=netloc).geturl()
