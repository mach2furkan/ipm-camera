"""Synthetic radiometric scene and an RTSP server that streams it like a thermal PT camera.

Used by the test-suite and the demo to exercise the complete thermal path -- RTSP
signalling, Digest auth, interleaved RTP, fragmentation, layout inference, analytics --
without hardware. The scene is physically motivated rather than random:

* sky colder than ground, ground with a fixed texture and a warmer road band;
* people rendered at their angular size for the configured lens, with apparent
  temperature attenuated toward ambient with distance (atmospheric transmission);
* an engine hotspot and an optional flickering fire;
* temporal noise at the detector NETD plus fixed-pattern noise.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..isapi.digest import DigestChallenge, compute_response

F32 = npt.NDArray[np.float32]


@dataclass
class SimPerson:
    distance_m: float
    lateral_m: float
    speed_lateral_mps: float = 0.0
    speed_radial_mps: float = 0.0
    body_c: float = 33.0


@dataclass
class ThermalSceneConfig:
    width: int = 384
    height: int = 288
    hfov_deg: float = 25.0
    camera_height_m: float = 8.0
    pitch_deg: float = 4.0               # depression of the optical axis
    ambient_c: float = 12.0
    sky_c: float = -18.0
    netd_c: float = 0.05
    attenuation_m: float = 900.0         # 1/e distance of thermal contrast loss
    seed: int = 0


@dataclass
class SyntheticThermalScene:
    cfg: ThermalSceneConfig = field(default_factory=ThermalSceneConfig)
    people: list[SimPerson] = field(default_factory=list)
    fire_at: tuple[float, float] | None = None      # (distance_m, lateral_m)
    fire_start_s: float = 0.0
    vehicle_at: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        from scipy.ndimage import gaussian_filter

        self._fpn = rng.normal(0, 0.03, (c.height, c.width)).astype(np.float32)
        # Terrain texture: spatially correlated (patches of grass, soil, gravel), several scales.
        tex = sum(gaussian_filter(rng.normal(0, 1, (c.height, c.width)), s) * a for s, a in ((2, 4.0), (8, 10.0), (24, 18.0)))
        self._tex = (tex / max(float(np.std(tex)), 1e-6) * 0.9).astype(np.float32)
        self._rng = rng
        self._f = c.width / 2 / math.tan(math.radians(c.hfov_deg) / 2)

    def _row_of_distance(self, d: float) -> float:
        c = self.cfg
        dep = math.degrees(math.atan2(c.camera_height_m, d))
        return c.height / 2 + self._f * math.tan(math.radians(dep - c.pitch_deg))

    def _col_of(self, d: float, lateral: float) -> float:
        return self.cfg.width / 2 + self._f * lateral / d

    def project_person(self, p: SimPerson, t: float) -> tuple[float, float, float, float]:
        d = max(3.0, p.distance_m + p.speed_radial_mps * t)
        lat = p.lateral_m + p.speed_lateral_mps * t
        foot_row = self._row_of_distance(d)
        h_px = self._f * 1.75 / d
        w_px = max(1.0, h_px * 0.38)
        cx = self._col_of(d, lat)
        return cx - w_px / 2, foot_row - h_px, cx + w_px / 2, foot_row

    def render(self, t: float) -> F32:
        c = self.cfg
        rows = np.arange(c.height, dtype=np.float32)[:, None]
        horizon = c.height / 2 - self._f * math.tan(math.radians(c.pitch_deg))
        ground_mix = np.clip((rows - horizon) / 6.0, 0.0, 1.0)
        img = (c.sky_c + (c.ambient_c - c.sky_c) * ground_mix) * np.ones((c.height, c.width), np.float32)
        img = img + self._tex * ground_mix
        road_top, road_bot = self._row_of_distance(120.0), self._row_of_distance(80.0)
        img[int(max(road_top, 0)): int(max(road_bot, 0))] += 3.0
        for p in self.people:
            x1, y1, x2, y2 = self.project_person(p, t)
            d = max(3.0, p.distance_m + p.speed_radial_mps * t)
            app = c.ambient_c + (p.body_c - c.ambient_c) * math.exp(-d / c.attenuation_m)
            self._ellipse(img, (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2, app)
        if self.vehicle_at is not None:
            d, lat = self.vehicle_at
            cx, foot = self._col_of(d, lat), self._row_of_distance(d)
            w = self._f * 4.5 / d
            h = self._f * 1.5 / d
            img[int(foot - h):int(foot), int(cx - w / 2):int(cx + w / 2)] = c.ambient_c + 4.0
            self._ellipse(img, cx - w * 0.35, foot - h * 0.4, w * 0.12, h * 0.25, 65.0)
        if self.fire_at is not None and t >= self.fire_start_s:
            d, lat = self.fire_at
            cx, foot = self._col_of(d, lat), self._row_of_distance(d)
            grow = min(1.0, (t - self.fire_start_s) / 3.0)
            r = max(1.5, self._f * 0.8 / d * grow)
            flame = 280.0 + 220.0 * grow + 80.0 * math.sin(t * 9.1) * self._rng.random()
            self._ellipse(img, cx, foot - r, r, r * 1.6, flame)
        img += self._fpn + self._rng.normal(0, c.netd_c, img.shape).astype(np.float32)
        return img.astype(np.float32)

    @staticmethod
    def _ellipse(img: F32, cx: float, cy: float, rx: float, ry: float, value: float) -> None:
        h, w = img.shape
        x0, x1 = max(0, int(cx - rx - 1)), min(w, int(cx + rx + 2))
        y0, y1 = max(0, int(cy - ry - 1)), min(h, int(cy + ry + 2))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = ((xx + 0.5 - cx) / max(rx, 0.5)) ** 2 + ((yy + 0.5 - cy) / max(ry, 0.5)) ** 2 <= 1.0
        img[y0:y1, x0:x1][inside] = value


def encode_payload(temps: F32, frame_id: int) -> bytes:
    """64-byte header + float32 LE matrix (the decoder must infer the header length)."""
    h, w = temps.shape
    header = struct.pack("<4sIIIQ", b"THRM", w, h, frame_id, time.time_ns()).ljust(64, b"\x00")
    return header + temps.astype("<f4").tobytes()


class FakeThermalRtspServer:
    """Speaks the RTSP subset a thermal PT camera uses for ``thermalStream``."""

    REALM = "IP Camera(SIM01)"

    def __init__(self, scene: SyntheticThermalScene, *, username: str = "admin", password: str = "sim-pass",
                 fps: float = 8.0, mtu: int = 1400, drop_fragment_every: int | None = None,
                 metadata_every_s: float | None = 1.0) -> None:
        self.scene = scene
        self.user = username
        self.password = password
        self.fps = fps
        self.mtu = mtu
        self.drop_every = drop_fragment_every
        self.meta_every = metadata_every_s
        self.nonce = os.urandom(8).hex()
        self.port = 0
        self._server: asyncio.base_events.Server | None = None
        self.frames_sent = 0
        self.auth_failures = 0

    async def start(self, host: str = "127.0.0.1") -> int:
        self._server = await asyncio.start_server(self._client, host, 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def url(self, channel: int = 2, data_type: str = "pixel-to-pixel_thermometry_data") -> str:
        return f"rtsp://127.0.0.1:{self.port}/ISAPI/Streaming/thermal/channels/{channel}/streamType/{data_type}"

    def _sdp(self) -> str:
        c = self.scene.cfg
        return "\r\n".join([
            "v=0", "o=- 1 1 IN IP4 127.0.0.1", "s=Media Presentation", "t=0 0", "a=control:*",
            "m=video 0 RTP/AVP 96", f"a=x-dimensions:{c.width},{c.height}", "a=control:trackID=1",
            "a=rtpmap:96 H264/90000",
            "m=application 0 RTP/AVP 107", "a=control:trackID=3", "a=rtpmap:107 isapi.metadata/90000",
            "m=application 0 RTP/AVP 109", "a=control:trackID=5", "a=rtpmap:109 thermalStream/90000", "",
        ])

    def _authorized(self, method: str, header: str | None) -> bool:
        if not header or not header.startswith("Digest "):
            return False
        params = dict(re.findall(r'(\w+)="?([^",]*)"?', header[7:]))
        ch = DigestChallenge(realm=self.REALM, nonce=self.nonce)
        expected, _ = compute_response(challenge=ch, username=self.user, password=self.password, method=method,
                                       uri=params.get("uri", ""), nc=1, cnonce="")
        return params.get("username") == self.user and params.get("response") == expected

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        channels: dict[str, int] = {}
        streamer: asyncio.Task[None] | None = None
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                method, uri, _ = lines[0].split(" ", 2)
                hdr = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if ":" in l)}
                cseq = hdr.get("cseq", "0")
                if not self._authorized(method, hdr.get("authorization")):
                    self.auth_failures += 1
                    writer.write((f"RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\n"
                                  f'WWW-Authenticate: Digest realm="{self.REALM}", nonce="{self.nonce}", stale="FALSE"\r\n\r\n').encode())
                    await writer.drain()
                    continue
                extra = ""
                body = b""
                if method == "OPTIONS":
                    extra = "Public: OPTIONS, DESCRIBE, SETUP, PLAY, GET_PARAMETER, TEARDOWN\r\n"
                elif method == "DESCRIBE":
                    body = self._sdp().encode()
                    extra = f"Content-Type: application/sdp\r\nContent-Base: {uri.rstrip('/')}/\r\n"
                elif method == "SETUP":
                    m = re.search(r"interleaved=(\d+)-(\d+)", hdr.get("transport", ""))
                    ch = int(m.group(1)) if m else 0
                    channels["meta" if uri.endswith("trackID=3") else "thermal" if uri.endswith("trackID=5") else "video"] = ch
                    extra = f"Session: 1234567;timeout=60\r\nTransport: RTP/AVP/TCP;unicast;interleaved={ch}-{ch + 1}\r\n"
                elif method == "PLAY":
                    extra = "Session: 1234567\r\nRTP-Info: url=trackID=5;seq=0\r\n"
                    streamer = asyncio.create_task(self._stream(writer, channels))
                elif method == "TEARDOWN":
                    writer.write(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n\r\n".encode())
                    await writer.drain()
                    break
                writer.write((f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n{extra}Content-Length: {len(body)}\r\n\r\n").encode()
                             + body)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            if streamer is not None:
                streamer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await streamer
            writer.close()

    async def _stream(self, writer: asyncio.StreamWriter, channels: dict[str, int]) -> None:
        seq = {"thermal": 0, "meta": 0}
        t0 = time.monotonic()
        last_meta = -1e9
        frame_id = 0
        while True:
            t = time.monotonic() - t0
            frame_id += 1
            temps = self.scene.render(t)
            if "thermal" in channels:
                payload = encode_payload(temps, frame_id)
                chunks = [payload[i:i + self.mtu] for i in range(0, len(payload), self.mtu)]
                for i, chunk in enumerate(chunks):
                    seq["thermal"] = (seq["thermal"] + 1) & 0xFFFF
                    if self.drop_every and frame_id % self.drop_every == 0 and i == len(chunks) // 2:
                        continue                                        # simulated loss
                    self._send(writer, channels["thermal"], 109, seq["thermal"], int(t * 90000), chunk,
                               marker=i == len(chunks) - 1)
                self.frames_sent += 1
            if "meta" in channels and self.meta_every is not None and t - last_meta >= self.meta_every:
                last_meta = t
                doc = json.dumps({"Metadata": {"type": "thermometry", "frameID": frame_id,
                                               "maxTemperature": round(float(temps.max()), 1)}}).encode()
                seq["meta"] = (seq["meta"] + 1) & 0xFFFF
                self._send(writer, channels["meta"], 107, seq["meta"], int(t * 90000), doc, marker=True)
            await writer.drain()
            await asyncio.sleep(max(0.0, frame_id / self.fps - (time.monotonic() - t0)))

    @staticmethod
    def _send(writer: asyncio.StreamWriter, ch: int, pt: int, seq: int, ts: int, payload: bytes, *, marker: bool) -> None:
        rtp = struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | pt, seq, ts & 0xFFFFFFFF, 0x1234ABCD) + payload
        writer.write(b"$" + struct.pack("!BH", ch, len(rtp)) + rtp)
