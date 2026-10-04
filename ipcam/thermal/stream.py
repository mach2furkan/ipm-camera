"""Real-time radiometric stream of the thermal PT series over RTSP.

URL: ``rtsp://<host>:554/ISAPI/Streaming/thermal/channels/<ch>/streamType/<type>`` with
``<type>`` one of ``thermal_raw_data``, ``pixel-to-pixel_thermometry_data`` (temperatures),
``real-time_raw_data``. The SDP advertises ``a=rtpmap:109 thermalStream/90000`` on
``trackID=5`` and ``isapi.metadata`` (PT 107) on ``trackID=3``.

The binary layout of one assembled ``thermalStream`` frame is not specified in the ISAPI
PT document, so :class:`ThermalPayloadDecoder` infers it once from the data -- detector
geometry, element type (float32 degC or uint16 counts) and header length -- by requiring
an exact size fit *and* a physically plausible temperature distribution, then locks the
layout. A known layout can be supplied explicitly to skip inference.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..backoff import ExponentialBackoff
from ..isapi import xmlutil as X
from ..isapi.pt_thermal import THERMAL_SENSOR_SIZES, decode_u16_temperatures
from ..metrics import RateMeter
from ..stream.rtsp_raw import FrameAssembler, RtspClient, RtspError, SdpMedia, parse_rtp
from ..stream.slot import LatestSlot

log = logging.getLogger(__name__)

F32 = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ThermalLayout:
    offset: int            # bytes of header before the matrix
    width: int
    height: int
    dtype: str             # "<f4" (degC) or "<u2" (radiometric counts)

    @property
    def nbytes(self) -> int:
        return self.width * self.height * (4 if self.dtype == "<f4" else 2)


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    temps: F32                 # (H, W) degrees Celsius
    t: float                   # host monotonic seconds at reassembly
    wall: float                # host epoch seconds
    rtp_timestamp: int
    frame_id: int
    header: bytes = field(repr=False, default=b"")

    @property
    def shape(self) -> tuple[int, int]:
        return self.temps.shape  # type: ignore[return-value]


def _plausible(t: F32) -> float:
    """Score in [0, 1]: fraction of finite pixels inside the physical range of real scenes
    (sky can be -60 degC, fire 1500 degC), penalising constant or noise-only matrices."""
    if t.size == 0:
        return 0.0
    finite = np.isfinite(t)
    if finite.mean() < 0.99:
        return 0.0
    v = t[finite]
    inside = float(((v > -80.0) & (v < 2000.0)).mean())
    if inside < 0.95:                  # reinterpreted garbage: reject before any arithmetic overflows
        return 0.0
    med = float(np.median(v))
    if not -50.0 <= med <= 90.0:
        return 0.0
    spread = float(np.percentile(v, 95) - np.percentile(v, 5))
    if spread < 0.05:              # a uniform matrix is almost certainly a misparse
        return inside * 0.2
    # Thermal images are spatially smooth; random bytes reinterpreted are not.
    rough = float(np.median(np.abs(np.diff(t, axis=1)))) / (spread + 1e-6)
    return inside * (1.0 if rough < 0.2 else 0.3)


class ThermalPayloadDecoder:
    def __init__(self, *, width: int | None = None, height: int | None = None,
                 layout: ThermalLayout | None = None, max_header: int = 65536) -> None:
        self.layout = layout
        self._sizes = [(width, height)] if width and height else sorted(THERMAL_SENSOR_SIZES, key=lambda s: -s[0] * s[1])
        self._max_header = max_header

    def _apply(self, payload: bytes, lay: ThermalLayout) -> F32:
        data = payload[lay.offset: lay.offset + lay.nbytes]
        if len(data) != lay.nbytes:
            raise ValueError(f"payload {len(payload)} B shorter than layout {lay}")
        arr = np.frombuffer(data, dtype=lay.dtype).reshape(lay.height, lay.width)
        return arr.astype(np.float32) if lay.dtype == "<f4" else decode_u16_temperatures(arr)

    def infer(self, payload: bytes) -> ThermalLayout:
        best: tuple[float, ThermalLayout] | None = None
        n = len(payload)
        for w, h in self._sizes:
            for dtype, size in (("<f4", 4), ("<u2", 2)):
                need = w * h * size
                extra = n - need
                if extra < 0 or extra > self._max_header:
                    continue
                for offset in {extra, 0}:              # header in front, or trailer behind
                    lay = ThermalLayout(offset, w, h, dtype)
                    try:
                        score = _plausible(self._apply(payload, lay))
                    except ValueError:
                        continue
                    score -= extra / (10 * self._max_header)   # prefer the tightest fit
                    if best is None or score > best[0]:
                        best = (score, lay)
        if best is None or best[0] < 0.8:
            raise ValueError(f"could not infer thermal payload layout from {n} bytes")
        return best[1]

    def decode(self, payload: bytes) -> tuple[F32, bytes]:
        if self.layout is None:
            self.layout = self.infer(payload)
            log.info("thermal payload layout inferred: %s", self.layout)
        lay = self.layout
        return self._apply(payload, lay), payload[: lay.offset]


def decode_metadata(payload: bytes) -> dict[str, Any] | None:
    """ISAPI metadata packets: XML or JSON documents (one per assembled RTP frame)."""
    text = payload.strip(b"\x00 \r\n\t")
    if not text:
        return None
    try:
        if text[:1] in (b"{", b"["):
            return json.loads(text)
        if text[:1] == b"<":
            root = X.parse(text)
            return {X.local(root.tag): X.to_dict(root)}
    except (ValueError, json.JSONDecodeError):
        pass
    return {"binary": len(payload)}


class ThermalStreamReader:
    """Connects, selects the thermal (and optionally metadata) track, and keeps the newest
    temperature frame in a drop-oldest slot. Reconnects with exponential backoff."""

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        decoder: ThermalPayloadDecoder | None = None,
        with_metadata: bool = False,
        stall_timeout_s: float = 3.0,
        on_frame: Callable[[ThermalFrame], None] | None = None,
        on_metadata: Callable[[dict[str, Any]], None] | None = None,
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        self.url = url
        self.username = username
        self.password = password
        self.decoder = decoder or ThermalPayloadDecoder()
        self.with_metadata = with_metadata
        self.stall = stall_timeout_s
        self.on_frame = on_frame
        self.on_metadata = on_metadata
        self.backoff = backoff or ExponentialBackoff()
        self.slot: LatestSlot[ThermalFrame] = LatestSlot()
        self.rate = RateMeter()
        self.frames = 0
        self.decode_errors = 0
        self.reconnects = 0
        self.connected = False
        self.last_error: str | None = None
        self.assembler_stats: dict[str, int] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="thermal-stream")
        return self._task

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def latest(self) -> ThermalFrame | None:
        return self.slot.peek()[1]

    @staticmethod
    def pick(medias: list[SdpMedia], encoding: str, pt: int, track: str) -> SdpMedia | None:
        for m in medias:
            if m.encoding.lower() == encoding.lower():
                return m
        for m in medias:
            if m.payload_type == pt or m.control.endswith(track):
                return m
        return None

    async def run(self) -> None:
        while True:
            try:
                await self._session()
                self.last_error = "stream ended"
            except asyncio.CancelledError:
                raise
            except (RtspError, OSError, asyncio.TimeoutError, ValueError) as exc:
                self.last_error = str(exc)
            self.connected = False
            self.reconnects += 1
            delay = self.backoff.next_delay()
            log.warning("thermal stream down (%s); reconnect in %.1fs", self.last_error, delay)
            await asyncio.sleep(delay)

    async def _session(self) -> None:
        client = RtspClient(self.url, username=self.username, password=self.password)
        async with client:
            medias = await client.describe()
            thermal = self.pick(medias, "thermalStream", 109, "trackID=5")
            if thermal is None:
                raise RtspError("SDP has no thermalStream track (is the thermal stream enabled?)")
            channels: dict[int, tuple[str, FrameAssembler]] = {}
            await client.setup(thermal, (0, 1))
            channels[0] = ("thermal", FrameAssembler())
            if self.with_metadata:
                meta = self.pick(medias, "isapi.metadata", 107, "trackID=3")
                if meta is not None:
                    await client.setup(meta, (2, 3))
                    channels[2] = ("metadata", FrameAssembler(max_frame_bytes=4 * 1024 * 1024))
            await client.play()
            self.connected = True
            frame_id = 0
            it = client.packets().__aiter__()
            while True:
                ch, data = await asyncio.wait_for(it.__anext__(), self.stall)
                entry = channels.get(ch)
                if entry is None:
                    continue                     # RTCP (odd channels) or unrequested tracks
                kind, asm = entry
                try:
                    pkt = parse_rtp(data)
                except ValueError:
                    continue
                frame = asm.push(pkt)
                if frame is None:
                    continue
                if kind == "metadata":
                    doc = decode_metadata(frame.data)
                    if doc is not None and self.on_metadata is not None:
                        self.on_metadata(doc)
                    continue
                try:
                    temps, header = self.decoder.decode(frame.data)
                except ValueError as exc:
                    self.decode_errors += 1
                    log.debug("thermal decode failed: %s", exc)
                    continue
                frame_id += 1
                tf = ThermalFrame(temps, time.monotonic(), time.time(), frame.timestamp, frame_id, header)
                self.frames += 1
                self.rate.tick()
                self.backoff.mark_healthy()
                self.slot.publish(tf)
                self.assembler_stats = {"frames": asm.frames, "dropped": asm.dropped, "lost": asm.lost_packets}
                if self.on_frame is not None:
                    self.on_frame(tf)
