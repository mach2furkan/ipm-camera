"""Exception hierarchy and failure classification for the streaming/control planes."""

from __future__ import annotations

import errno
import re
import sys
from enum import Enum


class IPCamError(Exception):
    """Base class for every error raised by this package."""


class StreamError(IPCamError):
    """RTSP/data-plane failure."""


class StreamOpenError(StreamError):
    """The RTSP handshake (DESCRIBE/SETUP/PLAY) or stream probing failed."""


class StreamStalledError(StreamError):
    """No media arrived within the heartbeat window; the session was torn down."""


class DecoderError(StreamError):
    """A decoder backend could not be created or failed irrecoverably."""


class ISAPIError(IPCamError):
    """Hikvision ISAPI returned a non-success ResponseStatus or HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        status_code: int | None = None,
        sub_status_code: str | None = None,
        path: str | None = None,
        error_code: int | None = None,
        error_msg: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.status_code = status_code
        self.sub_status_code = sub_status_code
        self.path = path
        self.error_code = error_code          # e.g. 0x40002068 (two-way audio in progress)
        self.error_msg = error_msg

    @property
    def hint(self) -> str | None:
        return ERROR_HINTS.get(self.sub_status_code or "") or (
            ERROR_HINTS.get(f"0x{self.error_code:08X}") if self.error_code is not None else None)

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.path:
            parts.append(f"path={self.path}")
        if self.http_status is not None:
            parts.append(f"http={self.http_status}")
        if self.status_code is not None:
            parts.append(f"statusCode={self.status_code}")
        if self.sub_status_code:
            parts.append(f"subStatusCode={self.sub_status_code}")
        if self.error_code is not None:
            parts.append(f"errorCode=0x{self.error_code:08X}")
        if self.hint:
            parts.append(f"({self.hint})")
        return " ".join(parts)


# Operator-facing hints for sub-status / error codes that come up in field integration.
ERROR_HINTS: dict[str, str] = {
    "notSupport": "firmware/model does not implement this API; check the capability set first",
    "invalidOperation": "operation rejected in the current device state",
    "numberReachedLlimit": "configured item limit reached; delete unused items",
    "twoWayAudioInProgressPleaseWait": "another client holds the two-way audio channel",
    "deviceBusy": "device busy (PTZ moving, upgrade or calibration in progress); retry later",
    "badXmlFormat": "malformed request body",
    "badXmlContent": "value out of range for this model; consult the capability ranges",
    "badParameters": "parameter rejected; consult the capability ranges",
    "lowPrivilege": "account lacks permission (operator/user); use an administrator account",
    "userLocked": "account locked after failed logins; wait for the lock to expire",
    "0x40002068": "two-way audio in progress",
    "0x40001119": "configured item limit reached",
}


class AuthenticationError(ISAPIError):
    """Credentials rejected. Retrying aggressively risks the device's illegal-login lockout."""


class NotSupportedError(ISAPIError):
    """The device/firmware does not implement the requested capability."""


class FailureKind(str, Enum):
    """Coarse failure classes; each maps to a different reconnect policy."""

    NETWORK = "network"          # reset/refused/unreachable/broken pipe
    TIMEOUT = "timeout"          # socket I/O timeout or heartbeat stall
    NOT_FOUND = "not_found"      # RTSP 404 (wrong channel, stream disabled)
    AUTH = "auth"                # RTSP/HTTP 401/403
    SERVER = "server"            # 5xx / 453 Not Enough Bandwidth / session limit
    DECODE = "decode"            # corrupt bitstream / decoder failure
    ENDED = "ended"              # clean EOF (camera closed the session)
    UNKNOWN = "unknown"


_NETWORK_ERRNOS = {
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ECONNABORTED,
    errno.EPIPE,
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENOTCONN,
}
_TIMEOUT_ERRNOS = {errno.ETIMEDOUT, errno.EAGAIN}

if sys.platform.startswith("win"):
    # FFmpeg on Windows reports MSVC CRT errno values (AVERROR(ECONNREFUSED) = -107, ...),
    # whereas Python's errno module exposes the Winsock WSAE* numbers (10061, ...).
    _NETWORK_ERRNOS |= {106, 107, 108, 110, 116, 118, 126, 32}
    _TIMEOUT_ERRNOS |= {138, 140}
    _WIN_ERRNO_RE = re.compile(r"\[errno -?(\d+)\]")
else:
    _WIN_ERRNO_RE = None

_URL_RE = re.compile(r"[a-z][a-z0-9+.-]*://\S+")

# Status codes are matched as whole words so FFmpeg's large negative errno values
# (e.g. "[Errno -1094995529]") can never be mistaken for an RTSP status.
_MESSAGE_RULES: tuple[tuple[FailureKind, re.Pattern[str]], ...] = (
    (FailureKind.AUTH, re.compile(r"\b40[13]\b|unauthori[sz]ed|forbidden")),
    (FailureKind.NOT_FOUND, re.compile(r"\b404\b|not found|no such file")),
    (FailureKind.SERVER, re.compile(r"\b(453|5\d\d)\b|not enough bandwidth|server returned 5")),
    (FailureKind.TIMEOUT, re.compile(r"timed out|timeout|etimedout|immediate exit requested")),
    (
        FailureKind.NETWORK,
        re.compile(
            r"connection reset|broken pipe|connection refused|network is unreachable|"
            r"no route to host|end of file|i/o error|connection aborted"
        ),
    ),
    (FailureKind.DECODE, re.compile(r"invalid data found|decod|corrupt")),
)


def classify_failure(exc: BaseException | None) -> FailureKind:
    """Map an exception raised anywhere in a streaming session to a FailureKind.

    PyAV maps FFmpeg AVERROR codes onto OSError subclasses carrying ``errno``; HTTP-ish
    RTSP statuses only surface through the message text, so both are inspected.
    """
    if exc is None:
        return FailureKind.ENDED
    if isinstance(exc, StreamStalledError):
        return FailureKind.TIMEOUT
    if isinstance(exc, AuthenticationError):
        return FailureKind.AUTH
    if isinstance(exc, DecoderError):
        return FailureKind.DECODE
    if isinstance(exc, (TimeoutError,)):
        return FailureKind.TIMEOUT
    if isinstance(exc, (ConnectionError, BrokenPipeError)):
        return FailureKind.NETWORK
    # URLs carry ports (":554") and paths ("/Channels/102") that would otherwise be
    # matched as status codes.
    text = _URL_RE.sub("", str(exc).lower())
    err = getattr(exc, "errno", None)
    if not isinstance(err, int) and _WIN_ERRNO_RE is not None:
        m = _WIN_ERRNO_RE.search(text)
        err = int(m.group(1)) if m else None
    if isinstance(err, int):
        err = abs(err)
        if err in _NETWORK_ERRNOS:
            return FailureKind.NETWORK
        if err in _TIMEOUT_ERRNOS:
            return FailureKind.TIMEOUT
    for kind, pattern in _MESSAGE_RULES:
        if pattern.search(text):
            return kind
    if isinstance(exc, EOFError):
        return FailureKind.ENDED
    return FailureKind.UNKNOWN
