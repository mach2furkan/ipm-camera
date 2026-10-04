"""Dependency-free proto3 wire codec for ``proto/surveillance/v1/event.proto``.

Byte-compatible with protoc-generated code (verified against ``google.protobuf`` in the
test-suite), but avoids the generated-code/runtime version coupling of ``*_pb2.py`` on edge
devices. Packed float fields are emitted straight from the numpy buffer and decoded with
``np.frombuffer`` -- a 512-d embedding is never boxed into 512 Python floats.

Unknown fields are skipped on decode, so newer edge nodes can talk to older engines.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

_VARINT, _I64, _LEN, _I32 = 0, 1, 2, 5


# --------------------------------------------------------------------------- primitives

def _varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _tag(num: int, wt: int) -> bytes:
    return _varint((num << 3) | wt)


def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


class _W:
    __slots__ = ("parts",)

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def string(self, n: int, s: str) -> None:
        if s:
            b = s.encode("utf-8")
            self.parts += [_tag(n, _LEN), _varint(len(b)), b]

    def uint(self, n: int, v: int) -> None:
        if v:
            self.parts += [_tag(n, _VARINT), _varint(int(v))]

    def boolean(self, n: int, v: bool) -> None:
        if v:
            self.parts += [_tag(n, _VARINT), b"\x01"]

    def f32(self, n: int, v: float) -> None:
        if v:
            self.parts += [_tag(n, _I32), struct.pack("<f", v)]

    def f64(self, n: int, v: float) -> None:
        if v:
            self.parts += [_tag(n, _I64), struct.pack("<d", v)]

    def packed_f32(self, n: int, arr: npt.ArrayLike | None) -> None:
        if arr is None:
            return
        b = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        if b:
            self.parts += [_tag(n, _LEN), _varint(len(b)), b]

    def packed_uint(self, n: int, vals: list[int]) -> None:
        if vals:
            b = b"".join(_varint(int(v)) for v in vals)
            self.parts += [_tag(n, _LEN), _varint(len(b)), b]

    def message(self, n: int, payload: bytes | None) -> None:
        if payload is not None:
            self.parts += [_tag(n, _LEN), _varint(len(payload)), payload]

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def _fields(data: bytes | memoryview):
    """Yield (field_number, wire_type, value) where LEN values are memoryviews."""
    buf = memoryview(data)
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _read_varint(buf, pos)
        num, wt = key >> 3, key & 7
        if wt == _VARINT:
            v, pos = _read_varint(buf, pos)
            yield num, wt, v
        elif wt == _I64:
            yield num, wt, buf[pos:pos + 8]
            pos += 8
        elif wt == _LEN:
            ln, pos = _read_varint(buf, pos)
            if pos + ln > n:
                raise ValueError("truncated length-delimited field")
            yield num, wt, buf[pos:pos + ln]
            pos += ln
        elif wt == _I32:
            yield num, wt, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _floats(v: memoryview | bytes, wt: int) -> F32:
    if wt == _LEN:
        return np.frombuffer(v, dtype="<f4")      # zero-copy view on the received buffer
    return np.frombuffer(v, dtype="<f4", count=1)


# --------------------------------------------------------------------------- messages

@dataclass(slots=True)
class DetectionEvent:
    camera_id: str = ""
    timestamp_ns: int = 0
    local_track_id: int = 0
    bbox: tuple[float, float, float, float, float] | None = None       # x1, y1, x2, y2, conf
    world: tuple[float, float, float] | None = None
    reid_embedding: F32 | None = None
    is_occluded: bool = False
    world_covariance: tuple[float, float, float] | None = None          # xx, xy, yy
    world_valid: bool = False
    class_id: int = 0

    def encode(self) -> bytes:
        w = _W()
        w.string(1, self.camera_id)
        w.uint(2, self.timestamp_ns)
        w.uint(3, self.local_track_id)
        if self.bbox is not None:
            b = _W()
            for i, v in enumerate(self.bbox, start=1):
                b.f32(i, float(v))
            w.message(4, b.bytes())
        if self.world is not None:
            g = _W()
            for i, v in enumerate(self.world, start=1):
                g.f64(i, float(v))
            w.message(5, g.bytes())
        w.packed_f32(6, self.reid_embedding)
        w.boolean(7, self.is_occluded)
        if self.world_covariance is not None:
            w.packed_f32(8, np.asarray(self.world_covariance, np.float32))
        w.boolean(9, self.world_valid)
        w.uint(10, self.class_id)
        return w.bytes()

    @classmethod
    def decode(cls, data: bytes | memoryview) -> DetectionEvent:
        ev = cls()
        for num, wt, v in _fields(data):
            if num == 1:
                ev.camera_id = bytes(v).decode("utf-8")
            elif num == 2:
                ev.timestamp_ns = _signed64(v)
            elif num == 3:
                ev.local_track_id = v
            elif num == 4:
                vals = [0.0] * 5
                for n2, _, v2 in _fields(v):
                    if 1 <= n2 <= 5:
                        vals[n2 - 1] = struct.unpack("<f", v2)[0]
                ev.bbox = tuple(vals)  # type: ignore[assignment]
            elif num == 5:
                g = [0.0] * 3
                for n2, _, v2 in _fields(v):
                    if 1 <= n2 <= 3:
                        g[n2 - 1] = struct.unpack("<d", v2)[0]
                ev.world = tuple(g)  # type: ignore[assignment]
            elif num == 6:
                arr = _floats(v, wt)
                ev.reid_embedding = arr if ev.reid_embedding is None else np.concatenate([ev.reid_embedding, arr])
            elif num == 7:
                ev.is_occluded = bool(v)
            elif num == 8:
                c = _floats(v, wt)
                if len(c) >= 3:
                    ev.world_covariance = (float(c[0]), float(c[1]), float(c[2]))
            elif num == 9:
                ev.world_valid = bool(v)
            elif num == 10:
                ev.class_id = v
        return ev


@dataclass(slots=True)
class FrameBatch:
    camera_id: str = ""
    timestamp_ns: int = 0
    sequence: int = 0
    detections: list[DetectionEvent] = field(default_factory=list)
    ended_track_ids: list[int] = field(default_factory=list)

    def encode(self) -> bytes:
        w = _W()
        w.string(1, self.camera_id)
        w.uint(2, self.timestamp_ns)
        w.uint(3, self.sequence)
        for d in self.detections:
            w.message(4, d.encode())
        w.packed_uint(5, self.ended_track_ids)
        return w.bytes()

    @classmethod
    def decode(cls, data: bytes | memoryview) -> FrameBatch:
        fb = cls()
        for num, wt, v in _fields(data):
            if num == 1:
                fb.camera_id = bytes(v).decode("utf-8")
            elif num == 2:
                fb.timestamp_ns = _signed64(v)
            elif num == 3:
                fb.sequence = v
            elif num == 4:
                fb.detections.append(DetectionEvent.decode(v))
            elif num == 5:
                if wt == _LEN:
                    pos = 0
                    while pos < len(v):
                        x, pos = _read_varint(v, pos)
                        fb.ended_track_ids.append(x)
                else:
                    fb.ended_track_ids.append(v)
        return fb


@dataclass(slots=True)
class SecurityAlert:
    camera_id: str = ""
    timestamp_ns: int = 0
    local_track_id: int = 0
    kind: str = ""
    rule: str = ""
    direction: str = ""
    dwell_s: float = 0.0

    def encode(self) -> bytes:
        w = _W()
        w.string(1, self.camera_id)
        w.uint(2, self.timestamp_ns)
        w.uint(3, self.local_track_id)
        w.string(4, self.kind)
        w.string(5, self.rule)
        w.string(6, self.direction)
        w.f32(7, self.dwell_s)
        return w.bytes()

    @classmethod
    def decode(cls, data: bytes | memoryview) -> SecurityAlert:
        a = cls()
        for num, _wt, v in _fields(data):
            if num == 1:
                a.camera_id = bytes(v).decode("utf-8")
            elif num == 2:
                a.timestamp_ns = _signed64(v)
            elif num == 3:
                a.local_track_id = v
            elif num == 4:
                a.kind = bytes(v).decode("utf-8")
            elif num == 5:
                a.rule = bytes(v).decode("utf-8")
            elif num == 6:
                a.direction = bytes(v).decode("utf-8")
            elif num == 7:
                a.dwell_s = struct.unpack("<f", v)[0]
        return a
