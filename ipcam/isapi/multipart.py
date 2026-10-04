"""Incremental multipart/mixed parser for long-lived ISAPI alert streams.

Real devices deviate from RFC 2046 in several ways this parser tolerates:

* the first delimiter may lack the preceding CRLF, or the body may start mid-preamble;
* ``Content-Length`` may be present (fast path, exact read) or absent (scan for the
  next delimiter);
* some firmwares terminate parts with ``\\r\\n`` before the delimiter, others not;
* very old firmwares send bare concatenated ``<EventNotificationAlert>`` documents with no
  multipart framing at all (handled by :class:`XMLDocumentSplitter`).

Memory is bounded: a part larger than ``max_part_size`` aborts the stream instead of
growing the buffer without limit when a device emits garbage.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

_CT_BOUNDARY = re.compile(r'boundary\s*=\s*"?([^";,\s]+)"?', re.IGNORECASE)


def boundary_from_content_type(content_type: str) -> str | None:
    m = _CT_BOUNDARY.search(content_type or "")
    return m.group(1) if m else None


@dataclass(slots=True)
class Part:
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";", 1)[0].strip().lower()


class MultipartFramingError(ValueError):
    pass


class MultipartStreamParser:
    _HEADER_END = b"\r\n\r\n"

    def __init__(self, boundary: str, *, max_part_size: int = 8 * 1024 * 1024) -> None:
        b = boundary.encode("latin-1")
        if b.startswith(b"--"):
            b = b[2:]
        self._delim = b"--" + b
        self._buf = bytearray()
        self._max = max_part_size
        self._state = "preamble"
        self._headers: dict[str, str] = {}
        self._length: int | None = None

    def feed(self, chunk: bytes) -> Iterator[Part]:
        self._buf += chunk
        while True:
            part = self._step()
            if part is None:
                break
            if part.body or part.headers:
                yield part
        if len(self._buf) > self._max + 4096:
            raise MultipartFramingError(f"part exceeds {self._max} bytes")

    def _step(self) -> Part | None:
        buf = self._buf
        if self._state == "preamble":
            idx = buf.find(self._delim)
            if idx < 0:
                # Keep a tail long enough to contain a split delimiter.
                keep = len(self._delim) + 2
                if len(buf) > keep:
                    del buf[:-keep]
                return None
            del buf[: idx + len(self._delim)]
            self._state = "after_delim"
            return Part()

        if self._state == "after_delim":
            if len(buf) < 2:
                return None
            if buf[:2] == b"--":  # close delimiter; keep listening for a new preamble
                del buf[:2]
                self._state = "preamble"
                return Part()
            # Consume the transport padding + CRLF that ends the delimiter line.
            eol = buf.find(b"\n")
            if eol < 0:
                return None
            del buf[: eol + 1]
            self._state = "headers"
            return Part()

        if self._state == "headers":
            if len(buf) < 2:
                return None
            if buf[:2] == b"\r\n" or buf[:1] == b"\n":  # part without headers
                del buf[: 2 if buf[:1] == b"\r" else 1]
                self._headers, self._length = {}, None
                self._state = "body"
                return Part()
            end = buf.find(self._HEADER_END)
            alt = buf.find(b"\n\n")
            if end < 0 and alt < 0:
                return None
            if end < 0 or (0 <= alt < end):
                header_blob, cut = bytes(buf[:alt]), alt + 2
            else:
                header_blob, cut = bytes(buf[:end]), end + 4
            del buf[:cut]
            headers: dict[str, str] = {}
            for line in header_blob.decode("latin-1").splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    headers[key.strip().lower()] = value.strip()
            self._headers = headers
            cl = headers.get("content-length", "")
            self._length = int(cl) if cl.isdigit() else None
            if self._length is not None and self._length > self._max:
                raise MultipartFramingError(f"declared part length {self._length} too large")
            self._state = "body"
            return Part()

        # body
        if self._length is not None:
            if len(buf) < self._length:
                return None
            body = bytes(buf[: self._length])
            del buf[: self._length]
        else:
            idx = buf.find(self._delim)
            if idx < 0:
                return None
            body = bytes(buf[:idx])
            del buf[:idx]
            if body.endswith(b"\r\n"):
                body = body[:-2]
            elif body.endswith(b"\n"):
                body = body[:-1]
        part = Part(self._headers, body)
        self._headers, self._length = {}, None
        self._state = "preamble"
        return part


class XMLDocumentSplitter:
    """Fallback framing for unframed streams of concatenated XML alert documents."""

    _OPEN = re.compile(rb"<(?:\w+:)?EventNotificationAlert\b")
    _CLOSE = re.compile(rb"</(?:\w+:)?EventNotificationAlert\s*>")

    def __init__(self, *, max_doc_size: int = 1024 * 1024) -> None:
        self._buf = bytearray()
        self._max = max_doc_size

    def feed(self, chunk: bytes) -> Iterator[Part]:
        self._buf += chunk
        while True:
            m_open = self._OPEN.search(self._buf)
            if not m_open:
                if len(self._buf) > 64:
                    del self._buf[:-64]
                return
            m_close = self._CLOSE.search(self._buf, m_open.end())
            if not m_close:
                if len(self._buf) - m_open.start() > self._max:
                    raise MultipartFramingError("unterminated XML alert document")
                if m_open.start():
                    del self._buf[: m_open.start()]
                return
            doc = bytes(self._buf[m_open.start(): m_close.end()])
            del self._buf[: m_close.end()]
            yield Part({"content-type": "application/xml"}, doc)
