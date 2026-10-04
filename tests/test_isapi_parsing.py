from __future__ import annotations

import random

import httpx
import pytest

from ipcam.errors import AuthenticationError, NotSupportedError
from ipcam.isapi import xmlutil as X
from ipcam.isapi.alert_stream import EventDebouncer
from ipcam.isapi.client import HikvisionISAPIClient
from ipcam.isapi.models import (
    AlertEvent,
    EventPhase,
    EventState,
    IRCutFilterState,
    IRCutMode,
    StreamingChannelInfo,
)
from ipcam.isapi.multipart import MultipartStreamParser, XMLDocumentSplitter, boundary_from_content_type

NS = 'xmlns="http://www.hikvision.com/ver20/XMLSchema"'


def alert_xml(event: str = "VMD", state: str = "active", count: int = 1, region: str | None = None) -> bytes:
    regions = (f"<DetectionRegionList><DetectionRegionEntry><regionID>{region}</regionID>"
               f"<detectionTarget>human</detectionTarget></DetectionRegionEntry></DetectionRegionList>"
               if region else "")
    return (f'<?xml version="1.0" encoding="UTF-8"?><EventNotificationAlert version="2.0" {NS}>'
            f"<ipAddress>192.168.1.64</ipAddress><channelID>1</channelID>"
            f"<dateTime>2026-10-04T12:00:00+03:00</dateTime><activePostCount>{count}</activePostCount>"
            f"<eventType>{event}</eventType><eventState>{state}</eventState>"
            f"<eventDescription>{event} alarm</eventDescription>{regions}</EventNotificationAlert>").encode()


def multipart_body(parts: list[bytes], *, with_length: bool) -> bytes:
    out = b""
    for p in parts:
        out += b"--boundary\r\nContent-Type: application/xml; charset=\"UTF-8\"\r\n"
        if with_length:
            out += f"Content-Length: {len(p)}\r\n".encode()
        out += b"\r\n" + p + b"\r\n"
    return out


@pytest.mark.parametrize("with_length", [True, False])
def test_multipart_parser_random_chunking(with_length: bool) -> None:
    docs = [alert_xml(count=i) for i in range(1, 8)]
    body = multipart_body(docs, with_length=with_length) + b"--boundary\r\n"
    rng = random.Random(7)
    parser = MultipartStreamParser(boundary_from_content_type('multipart/mixed; boundary="boundary"'))
    got = []
    i = 0
    while i < len(body):
        n = rng.randint(1, 50)
        got += [p.body for p in parser.feed(body[i:i + n])]
        i += n
    assert got == docs


def test_bare_xml_splitter() -> None:
    docs = [alert_xml(count=i) for i in range(3)]
    stream = b"garbage" + b"\r\n".join(docs)
    sp = XMLDocumentSplitter()
    got = [p.body for chunk in (stream[:100], stream[100:333], stream[333:]) for p in sp.feed(chunk)]
    assert [AlertEvent.from_xml(X.parse(d)).active_post_count for d in got] == [0, 1, 2]


def test_alert_event_model() -> None:
    ev = AlertEvent.from_xml(X.parse(alert_xml("linedetection", region="2")))
    assert ev.event_type == "linedetection"
    assert ev.state is EventState.ACTIVE
    assert ev.channel_id == 1
    assert ev.region_ids == ("2",)
    assert ev.target_type == "human"
    assert ev.timestamp.utcoffset() is not None
    hb = AlertEvent.from_xml(X.parse(alert_xml("videoloss", "inactive")))
    assert hb.is_heartbeat


def test_debouncer_lifecycle() -> None:
    d = EventDebouncer(hold_s=3.0)
    a1 = AlertEvent.from_xml(X.parse(alert_xml(count=1)))
    a2 = AlertEvent.from_xml(X.parse(alert_xml(count=2)))
    assert [e.phase for e in d.feed(a1, 0.0)] == [EventPhase.START]
    assert d.feed(a2, 1.0) == []
    assert d.expire(3.5) == []
    ended = d.expire(4.1)
    assert [e.phase for e in ended] == [EventPhase.END] and ended[0].active_post_count == 2
    d.feed(a1, 10.0)
    inactive = AlertEvent.from_xml(X.parse(alert_xml(state="inactive")))
    assert [e.phase for e in d.feed(inactive, 10.5)] == [EventPhase.END]


def test_ircut_and_channel_models() -> None:
    ir = IRCutFilterState.from_xml(X.parse(
        f'<IrcutFilter {NS}><IrcutFilterType>auto</IrcutFilterType>'
        f"<nightToDayFilterLevel>4</nightToDayFilterLevel><nightToDayFilterTime>5</nightToDayFilterTime>"
        f"</IrcutFilter>".encode()))
    assert ir.mode is IRCutMode.AUTO and ir.night_to_day_level == 4

    ch = StreamingChannelInfo.from_xml(X.parse(
        f"<StreamingChannel {NS}><id>102</id><enabled>true</enabled><Video>"
        f"<videoCodecType>H.265</videoCodecType><videoResolutionWidth>640</videoResolutionWidth>"
        f"<videoResolutionHeight>360</videoResolutionHeight><maxFrameRate>2500</maxFrameRate>"
        f"<GovLength>50</GovLength><vbrUpperCap>512</vbrUpperCap></Video></StreamingChannel>".encode()))
    assert (ch.ffmpeg_codec, ch.width, ch.height, ch.max_fps, ch.gop_frames) == ("hevc", 640, 360, 25.0, 50)


def test_xml_rejects_entities() -> None:
    with pytest.raises(ValueError):
        X.parse(b'<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><a>&x;</a>')


async def test_client_relay_pulse_and_errors() -> None:
    calls: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        if request.url.path.endswith("/outputs/9/trigger"):
            return httpx.Response(403, content=(
                f"<ResponseStatus {NS}><statusCode>4</statusCode><statusString>Invalid Operation"
                f"</statusString><subStatusCode>notSupport</subStatusCode></ResponseStatus>").encode())
        if request.url.path == "/denied":
            return httpx.Response(401)
        return httpx.Response(200, content=f"<ResponseStatus {NS}><statusCode>1</statusCode>"
                                           f"<statusString>OK</statusString></ResponseStatus>".encode())

    cam = HikvisionISAPIClient("cam", "admin", "x")
    cam._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://cam")
    import asyncio

    await cam.pulse_alarm_output(1, duration_s=0.2)
    await cam.pulse_alarm_output(1, duration_s=0.3)       # retrigger extends, no extra "high"
    await asyncio.sleep(0.5)
    states = [b"high" in body for m, path, body in calls if path.endswith("/outputs/1/trigger")]
    assert states == [True, False]

    with pytest.raises(NotSupportedError):
        await cam.set_alarm_output(9, True)
    with pytest.raises(AuthenticationError):
        await cam.request("GET", "/denied")

    await cam.pulse_alarm_output(2, duration_s=60)
    await cam.aclose()                                     # fail-safe release on shutdown
    assert b"low" in [body for m, path, body in calls if path.endswith("/outputs/2/trigger")][-1]
