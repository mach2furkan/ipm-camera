"""HTTP Digest Access Authentication (RFC 7616, backwards compatible with RFC 2617).

Implemented as an ``httpx.Auth`` flow so it works with both the sync and async clients.
Compared with ``httpx.DigestAuth`` this implementation:

* chooses the strongest offered challenge (SHA-512-256 > SHA-256 > MD5) when the device
  sends several ``WWW-Authenticate`` headers, as RFC 7616 section 3.7 recommends;
* supports ``-sess`` variants, ``qop=auth-int``, ``userhash`` and ``stale=true`` nonce
  rotation without surfacing a spurious 401 to the caller;
* authenticates pre-emptively with the cached nonce and a monotonically increasing nonce
  count, saving one round trip per ISAPI call and making non-replayable streaming
  uploads (two-way audio) possible;
* refuses to retry a definitively rejected credential: Hikvision locks the client IP for
  30 minutes after a handful of failed logins.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass

import httpx

_HASHES: dict[str, Callable[[bytes], str]] = {
    "MD5": lambda b: hashlib.md5(b).hexdigest(),
    "SHA-256": lambda b: hashlib.sha256(b).hexdigest(),
}
try:  # OpenSSL builds without truncated SHA-512 exist (e.g. some Alpine images).
    hashlib.new("sha512_256", b"")
    _HASHES["SHA-512-256"] = lambda b: hashlib.new("sha512_256", b).hexdigest()
except ValueError:
    pass

_STRENGTH = {"MD5": 1, "SHA-256": 2, "SHA-512-256": 3}

_PARAM_RE = re.compile(
    r"""\s*([A-Za-z0-9_\-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^\s,]*)\s*(?:,|$)"""
)
_SCHEME_RE = re.compile(r"(?:^|,)\s*([A-Za-z][A-Za-z0-9_\-]*)\s+(?=[A-Za-z0-9_\-]+\s*=)")


@dataclass(frozen=True, slots=True)
class DigestChallenge:
    realm: str
    nonce: str
    algorithm: str = "MD5"
    qop: tuple[str, ...] = ()
    opaque: str | None = None
    stale: bool = False
    userhash: bool = False
    charset: str | None = None

    @property
    def base_algorithm(self) -> str:
        return self.algorithm.upper().removesuffix("-SESS")

    @property
    def is_session(self) -> bool:
        return self.algorithm.upper().endswith("-SESS")

    @property
    def strength(self) -> int:
        return _STRENGTH.get(self.base_algorithm, 0)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return re.sub(r"\\(.)", r"\1", value[1:-1])
    return value


def parse_challenges(header_values: list[str]) -> list[DigestChallenge]:
    """Parse every Digest challenge contained in one or more WWW-Authenticate values.

    A single header may carry several comma-separated challenges of different schemes
    (``Basic realm="x", Digest realm="y", nonce="..."``), hence the scheme split.
    """
    out: list[DigestChallenge] = []
    for raw in header_values:
        starts = [(m.start(1), m.group(1)) for m in _SCHEME_RE.finditer(raw)]
        if not starts and raw.strip():
            scheme, _, rest = raw.strip().partition(" ")
            starts = [(0, scheme)]
            raw = f"{scheme} {rest}"
        for idx, (pos, scheme) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(raw)
            if scheme.lower() != "digest":
                continue
            body = raw[pos + len(scheme):end].strip().rstrip(",")
            params = {k.lower(): _unquote(v) for k, v in _PARAM_RE.findall(body)}
            if "nonce" not in params:
                continue
            algorithm = params.get("algorithm", "MD5").upper()
            if algorithm.removesuffix("-SESS") not in _HASHES:
                continue
            qop = tuple(q.strip().lower() for q in params.get("qop", "").split(",") if q.strip())
            out.append(
                DigestChallenge(
                    realm=params.get("realm", ""),
                    nonce=params["nonce"],
                    algorithm=algorithm,
                    qop=qop,
                    opaque=params.get("opaque"),
                    stale=params.get("stale", "false").lower() == "true",
                    userhash=params.get("userhash", "false").lower() == "true",
                    charset=params.get("charset"),
                )
            )
    return out


def compute_response(
    *,
    challenge: DigestChallenge,
    username: str,
    password: str,
    method: str,
    uri: str,
    nc: int,
    cnonce: str,
    body: bytes | None = None,
) -> tuple[str, str | None]:
    """Return ``(response, qop_used)`` per RFC 7616 section 3.4."""
    h = _HASHES[challenge.base_algorithm]
    enc = "utf-8" if (challenge.charset or "").upper() == "UTF-8" else "latin-1"

    def b(s: str) -> bytes:
        try:
            return s.encode(enc)
        except UnicodeEncodeError:
            return s.encode("utf-8")

    ha1 = h(b(f"{username}:{challenge.realm}:{password}"))
    if challenge.is_session:
        ha1 = h(b(f"{ha1}:{challenge.nonce}:{cnonce}"))

    qop: str | None = None
    if "auth" in challenge.qop:
        qop = "auth"
    elif "auth-int" in challenge.qop and body is not None:
        qop = "auth-int"

    if qop == "auth-int":
        ha2 = h(b(f"{method}:{uri}:") + h(body or b"").encode())
    else:
        ha2 = h(b(f"{method}:{uri}"))

    if qop:
        response = h(b(f"{ha1}:{challenge.nonce}:{nc:08x}:{cnonce}:{qop}:{ha2}"))
    else:
        response = h(b(f"{ha1}:{challenge.nonce}:{ha2}"))
    return response, qop


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class DigestAuth(httpx.Auth):
    """Thread- and task-safe Digest authenticator with nonce caching."""

    # Streaming bodies (two-way audio) must not be buffered by httpx; auth-int simply
    # degrades to qop=auth when the body is not materialised.
    requires_request_body = False

    def __init__(self, username: str, password: str, *, preemptive: bool = True) -> None:
        self._username = username
        self._password = password
        self._preemptive = preemptive
        self._lock = threading.Lock()
        self._challenge: DigestChallenge | None = None
        self._nc = 0

    @property
    def challenge(self) -> DigestChallenge | None:
        return self._challenge

    def _authorization(self, request: httpx.Request, challenge: DigestChallenge) -> str:
        with self._lock:
            if self._challenge is challenge:
                self._nc += 1
                nc = self._nc
            else:
                nc = 1
        cnonce = os.urandom(16).hex()
        uri = request.url.raw_path.decode("ascii")
        body: bytes | None
        try:
            body = request.content
        except httpx.RequestNotRead:
            body = None
        response, qop = compute_response(
            challenge=challenge,
            username=self._username,
            password=self._password,
            method=request.method,
            uri=uri,
            nc=nc,
            cnonce=cnonce,
            body=body,
        )
        if challenge.userhash:
            user = _HASHES[challenge.base_algorithm](f"{self._username}:{challenge.realm}".encode())
        else:
            user = self._username
        fields = [
            f"username={_quote(user)}",
            f"realm={_quote(challenge.realm)}",
            f"nonce={_quote(challenge.nonce)}",
            f"uri={_quote(uri)}",
            f"algorithm={challenge.algorithm}",
            f"response={_quote(response)}",
        ]
        if challenge.opaque is not None:
            fields.append(f"opaque={_quote(challenge.opaque)}")
        if qop:
            fields += [f"qop={qop}", f"nc={nc:08x}", f"cnonce={_quote(cnonce)}"]
        if challenge.userhash:
            fields.append("userhash=true")
        return "Digest " + ", ".join(fields)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        cached = self._challenge if self._preemptive else None
        if cached is not None:
            request.headers["Authorization"] = self._authorization(request, cached)

        response = yield request
        if response.status_code != 401:
            return

        challenges = parse_challenges(response.headers.get_list("www-authenticate"))
        if not challenges:
            return
        best = max(challenges, key=lambda c: c.strength)

        # The cached nonce was valid and the server did not flag it stale: the credentials
        # themselves are wrong. Surfacing the 401 avoids burning a lockout attempt.
        if cached is not None and not best.stale and best.nonce == cached.nonce:
            return

        with self._lock:
            self._challenge = best
            self._nc = 0
        request.headers["Authorization"] = self._authorization(request, best)
        yield request
