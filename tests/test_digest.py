from __future__ import annotations

import httpx
import pytest

from ipcam.isapi.digest import DigestAuth, DigestChallenge, compute_response, parse_challenges

# RFC 7616 section 3.9.1
RFC = dict(
    username="Mufasa",
    password="Circle of Life",
    method="GET",
    uri="/dir/index.html",
    nc=1,
    cnonce="f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ",
)
REALM = "http-auth@example.org"
NONCE = "7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v"


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("MD5", "8ca523f5e9506fed4657c9700eebdbec"),
        ("SHA-256", "753927fa0e85d155564e2e272a28d1802ca10daf4496794697cf8db5856cb6c1"),
    ],
)
def test_rfc7616_vectors(algorithm: str, expected: str) -> None:
    ch = DigestChallenge(realm=REALM, nonce=NONCE, algorithm=algorithm, qop=("auth", "auth-int"))
    response, qop = compute_response(challenge=ch, **RFC)
    assert qop == "auth"
    assert response == expected


def test_rfc2617_vector_without_session() -> None:
    # RFC 2617 section 3.5
    ch = DigestChallenge(realm="testrealm@host.com", nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093",
                         qop=("auth",), opaque="5ccc069c403ebaf9f0171e9517f40e41")
    response, _ = compute_response(challenge=ch, username="Mufasa", password="Circle Of Life",
                                   method="GET", uri="/dir/index.html", nc=1, cnonce="0a4f113b")
    assert response == "6629fae49393a05397450978507c4ef1"


def test_parse_multiple_challenges_picks_strongest() -> None:
    headers = [
        'Basic realm="x", Digest realm="cam", nonce="n1", qop="auth", algorithm=MD5',
        'Digest realm="cam", nonce="n2", qop="auth", algorithm=SHA-256, stale=TRUE, opaque="o"',
    ]
    chs = parse_challenges(headers)
    assert [c.algorithm for c in chs] == ["MD5", "SHA-256"]
    best = max(chs, key=lambda c: c.strength)
    assert best.nonce == "n2" and best.stale and best.opaque == "o"


class FakeCamera:
    """Minimal Digest server: validates the response and rotates nonces on demand."""

    def __init__(self, password: str = "Circle of Life") -> None:
        self.password = password
        self.nonce = "nonce-1"
        self.requests = 0
        self.stale_next = False
        self.seen_nc: list[str] = []

    def challenge(self, stale: bool = False) -> httpx.Response:
        hdr = f'Digest realm="IP Camera", nonce="{self.nonce}", qop="auth", algorithm=MD5'
        if stale:
            hdr += ", stale=TRUE"
        return httpx.Response(401, headers={"WWW-Authenticate": hdr})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        auth = request.headers.get("authorization")
        if not auth:
            return self.challenge()
        params = dict(p.strip().split("=", 1) for p in auth[len("Digest "):].split(","))
        params = {k: v.strip('"') for k, v in params.items()}
        if self.stale_next:
            self.stale_next = False
            self.nonce = "nonce-2"
            return self.challenge(stale=True)
        if params["nonce"] != self.nonce:
            return self.challenge(stale=True)
        ch = DigestChallenge(realm="IP Camera", nonce=self.nonce, qop=("auth",))
        expected, _ = compute_response(challenge=ch, username="admin", password=self.password,
                                       method=request.method, uri=params["uri"], nc=int(params["nc"], 16),
                                       cnonce=params["cnonce"])
        if params["response"] != expected:
            return self.challenge()
        self.seen_nc.append(params["nc"])
        return httpx.Response(200, text="<ok/>")


def test_flow_preemptive_and_stale_rotation() -> None:
    cam = FakeCamera()
    client = httpx.Client(transport=httpx.MockTransport(cam), auth=DigestAuth("admin", "Circle of Life"),
                          base_url="http://cam")
    assert client.get("/ISAPI/System/deviceInfo").status_code == 200
    assert cam.requests == 2                      # challenge + authorised
    assert client.get("/ISAPI/System/deviceInfo").status_code == 200
    assert cam.requests == 3                      # pre-emptive: one round trip
    assert cam.seen_nc == ["00000001", "00000002"]
    cam.stale_next = True
    assert client.get("/x").status_code == 200    # stale nonce transparently renewed
    assert cam.seen_nc[-1] == "00000001"


def test_wrong_password_does_not_loop() -> None:
    cam = FakeCamera(password="other")
    client = httpx.Client(transport=httpx.MockTransport(cam), auth=DigestAuth("admin", "Circle of Life"),
                          base_url="http://cam")
    assert client.get("/x").status_code == 401
    assert cam.requests == 2
    before = cam.requests
    assert client.get("/x").status_code == 401   # cached nonce, not stale -> give up at once
    assert cam.requests - before == 1
