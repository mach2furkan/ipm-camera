"""Thermal PT series: RTSP raw stream, radiometry, bi-spectral fusion, geolocation, ISAPI."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import zlib

import httpx
import numpy as np
import pytest

from ipcam.errors import ISAPIError
from ipcam.fusion.geo import (
    LensModel,
    LocalTangentPlane,
    PTGeoMount,
    PTGeolocator,
    calibrate_north,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from ipcam.fusion.pt_bridge import PTEventGeoBridge
from ipcam.isapi import xmlutil as X
from ipcam.isapi.client import HikvisionISAPIClient
from ipcam.isapi.events_pt import pt_details
from ipcam.isapi.models import AlertEvent
from ipcam.isapi.pt_thermal import PTThermalCamera, decode_u16_temperatures, parse_realtime_thermometry
from ipcam.stream.rtsp_raw import FrameAssembler, RtpPacket, parse_rtp, parse_sdp
from ipcam.thermal import (
    ChannelRegistration,
    FireDetector,
    HotspotDetector,
    PersonCandidates,
    PlateauAGC,
    RoiMonitor,
    ThermalConfirmation,
    ThermalPayloadDecoder,
    ThermalStreamReader,
    colorize,
    encode_png,
)
from ipcam.thermal.sim import FakeThermalRtspServer, SimPerson, SyntheticThermalScene, encode_payload

NS = X.ISAPI_NS


def scene(**kw: object) -> SyntheticThermalScene:
    return SyntheticThermalScene(people=[SimPerson(60.0, -6.0, 1.2), SimPerson(110.0, 9.0)],
                                 vehicle_at=(150.0, 0.0), **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- RTP / SDP

def test_rtp_parse_csrc_extension_padding() -> None:
    payload = b"thermal-bytes"
    hdr = struct.pack("!BBHII", 0x80 | 0x20 | 0x10 | 2, 0x80 | 109, 7, 90000, 0xABCD)
    csrc = struct.pack("!II", 1, 2)
    ext = struct.pack("!HH", 0xBEDE, 1) + b"\x01\x02\x03\x04"
    pkt = parse_rtp(hdr + csrc + ext + payload + b"\x00\x00\x03")
    assert (pkt.marker, pkt.payload_type, pkt.sequence, pkt.payload) == (True, 109, 7, payload)
    assert pkt.extension == b"\x01\x02\x03\x04"
    with pytest.raises(ValueError):
        parse_rtp(b"\x40" + b"\x00" * 11)       # version 1


def test_sdp_and_assembler_gap() -> None:
    _, medias = parse_sdp("v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=control:trackID=1\r\n"
                          "m=application 0 RTP/AVP 109\r\na=control:trackID=5\r\na=rtpmap:109 thermalStream/90000\r\n")
    assert medias[1].encoding == "thermalStream" and medias[1].control == "trackID=5" and medias[1].clock_rate == 90000
    asm = FrameAssembler()
    mk = lambda seq, m, ts=1: RtpPacket(m, 109, seq, ts, 1, bytes([seq]))  # noqa: E731
    assert asm.push(mk(1, False)) is None
    assert asm.push(mk(2, True)).data == b"\x01\x02"
    asm.push(mk(3, False, 2))
    assert asm.push(mk(5, True, 2)) is None            # seq 4 lost -> whole frame dropped
    assert asm.dropped == 1 and asm.lost_packets == 1
    assert asm.push(mk(6, True, 3)).data == b"\x06"


def test_payload_layout_inference_float_and_u16() -> None:
    temps = scene().render(0.0)
    dec = ThermalPayloadDecoder()
    out, header = dec.decode(encode_payload(temps, 1))
    assert dec.layout.offset == 64 and (dec.layout.width, dec.layout.height, dec.layout.dtype) == (384, 288, "<f4")
    assert np.allclose(out, temps) and header[:4] == b"THRM"
    deci_k = np.round((temps + 273.15) * 10).astype("<u2")
    dec2 = ThermalPayloadDecoder(width=384, height=288)
    out2, _ = dec2.decode(b"\x00" * 32 + deci_k.tobytes())
    assert dec2.layout.dtype == "<u2" and np.abs(out2 - temps).max() < 0.06
    with pytest.raises(ValueError):
        ThermalPayloadDecoder().decode(np.random.default_rng(0).bytes(384 * 288 * 4 + 64))


# --------------------------------------------------------------------------- RTSP end to end

async def test_rtsp_thermal_stream_end_to_end() -> None:
    srv = FakeThermalRtspServer(scene(), fps=20)
    await srv.start()
    meta: list[dict] = []
    reader = ThermalStreamReader(srv.url(), username="admin", password="sim-pass", with_metadata=True,
                                 on_metadata=meta.append)
    reader.start()
    try:
        for _ in range(100):
            if reader.frames >= 6 and meta:
                break
            await asyncio.sleep(0.05)
        assert reader.frames >= 6, reader.last_error
        frame = reader.latest()
        assert frame.shape == (288, 384)
        assert float(np.median(frame.temps)) == pytest.approx(12.0, abs=6.0)
        assert frame.temps.max() > 25.0                      # people / engine present
        assert meta and meta[0]["Metadata"]["type"] == "thermometry"
    finally:
        await reader.stop()
        await srv.stop()


async def test_rtsp_wrong_password_and_packet_loss() -> None:
    srv = FakeThermalRtspServer(scene(), fps=20, drop_fragment_every=3)
    await srv.start()
    bad = ThermalStreamReader(srv.url(), username="admin", password="wrong")
    good = ThermalStreamReader(srv.url(), username="admin", password="sim-pass")
    bad.start()
    good.start()
    try:
        for _ in range(120):
            if good.frames >= 8 and good.assembler_stats.get("dropped", 0) > 0:
                break
            await asyncio.sleep(0.05)
        assert bad.frames == 0 and "401" in (bad.last_error or "")
        assert good.assembler_stats["dropped"] > 0           # incomplete frames discarded ...
        assert good.decode_errors == 0                        # ... never decoded as shifted garbage
    finally:
        await bad.stop()
        await good.stop()
        await srv.stop()


# --------------------------------------------------------------------------- rendering

def test_plateau_agc_keeps_small_target_contrast_and_png() -> None:
    temps = np.full((120, 160), 10.0, np.float32) + np.random.default_rng(0).normal(0, 0.05, (120, 160)).astype(np.float32)
    temps[:40] = -20.0                          # big cold sky
    temps[80:84, 70:72] = 13.0                  # tiny person 3 degC above the field
    temps[100:110, 10:20] = 300.0               # fire saturating a linear stretch
    gray = PlateauAGC(smoothing=1.0)(temps)
    linear = ((temps - temps.min()) / (temps.max() - temps.min()) * 255).astype(np.uint8)
    contrast = lambda g: int(g[81, 70]) - int(np.median(g[60:78, 40:60]))  # noqa: E731
    assert contrast(gray) > 25 and contrast(linear) < 5
    png = encode_png(colorize(gray, "iron"))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack("!II", png[16:24])
    idat = png[png.index(b"IDAT") + 4: png.index(b"IEND") - 8]
    assert (w, h) == (160, 120) and len(zlib.decompress(idat)) == h * (w * 3 + 1)


# --------------------------------------------------------------------------- radiometric analytics

def test_hotspots_people_and_fire() -> None:
    sc = scene(fire_at=(90.0, -15.0), fire_start_s=5.0)
    t0 = sc.render(0.0)
    people = PersonCandidates()(t0)
    assert len(people) >= 2
    exp = [sc.project_person(p, 0.0) for p in sc.people]
    for x1, y1, x2, y2 in exp:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        assert any(abs(b.centroid[0] - cx) < 4 and abs(b.centroid[1] - cy) < 6 for b in people)
    assert HotspotDetector()(t0)[0].max_c > 50      # engine is the hottest object before the fire
    fd = FireDetector()
    fires = []
    for k in range(40):
        t = k * 0.25
        fires = fd(sc.render(t), t)
        if t < 5.0:
            assert not fires
    assert fires and fires[0].max_c > 150


def test_roi_monitor_levels_dwell_hysteresis_and_rise() -> None:
    roi = RoiMonitor("trafo", [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], prealarm_c=60, alarm_c=80,
                     dwell_s=2.0, hysteresis_c=3.0, rise_c_per_min=5.0, rise_window_s=30.0)
    temps = np.full((40, 40), 20.0, np.float32)
    events = []
    for k in range(400):
        t = k * 0.5
        v = 40 + 0.25 * k if k < 200 else 90 - 0.3 * (k - 200)   # ramp up 30 degC/min, then cool
        temps[20, 20] = v
        events += [(e.level, round(e.t, 1)) for e in roi.update(temps, t)]
    levels = [e[0] for e in events]
    assert levels.index("rise") < levels.index("prealarm") < levels.index("alarm")   # early warning first
    t_pre = dict((lv, t) for lv, t in events)["prealarm"]
    assert t_pre >= (60 - 40) / 0.25 * 0.5 + 2.0 - 0.5      # dwell honoured
    assert levels[-1] == "clear"


def test_bispectral_registration_and_thermal_confirmation() -> None:
    reg = ChannelRegistration.from_fov((1920, 1080), 30.0, (384, 288), 25.0, baseline_m=0.12)
    c = reg.optical_to_thermal([[960, 540]])[0]
    assert np.allclose(c, [192, 144], atol=0.5)
    near = reg.optical_to_thermal([[960, 540]], range_m=10.0)[0]
    far = reg.optical_to_thermal([[960, 540]], range_m=1000.0)[0]
    assert near[0] - c[0] > 3 * (far[0] - c[0])            # parallax only matters up close
    temps = np.full((288, 384), 12.0, np.float32) + np.random.default_rng(1).normal(0, 0.1, (288, 384)).astype(np.float32)
    temps[130:160, 185:197] = 31.0                         # person in the thermal view
    conf = ThermalConfirmation(reg)
    warm = conf.assess(temps, (940, 470, 1000, 640))
    cold = conf.assess(temps, (300, 470, 360, 640))        # a shadow / headlight reflection
    assert warm.warm and not cold.warm


# --------------------------------------------------------------------------- geodesy / geolocation

def test_ecef_roundtrip_and_enu_scale() -> None:
    for lat, lon, alt in [(39.92, 32.85, 900.0), (-33.9, 151.2, 5.0), (70.0, -150.0, 0.0)]:
        la, lo, al = ecef_to_geodetic(*geodetic_to_ecef(lat, lon, alt))
        assert abs(la - lat) < 1e-9 and abs(lo - lon) < 1e-9 and abs(al - alt) < 1e-3
    site = LocalTangentPlane(39.92, 32.85, 900.0)
    e, n, u = site.to_enu(39.93, 32.85, 900.0)
    assert abs(e) < 0.05 and n == pytest.approx(1110.9, abs=2.0) and abs(u) < 0.2


def test_pt_geolocation_range_terrain_and_north_calibration() -> None:
    site = LocalTangentPlane(39.92, 32.85, 900.0)
    mount = PTGeoMount("pt-1", 39.92, 32.85, 912.0, height_above_ground_m=12.0, north_offset_deg=37.5,
                       sigma_range_m=1.0, sigma_azimuth_deg=0.05)
    loc = PTGeolocator(mount, site)
    for e, n in [(250.0, 400.0), (-800.0, 120.0), (60.0, -35.0)]:
        az, el, rng = loc.aim(e, n, aim_height_m=0.0)
        g = loc.locate(az, el, range_m=rng)
        assert math.hypot(g.enu[0] - e, g.enu[1] - n) < 0.05 and g.method == "range"
        g2 = loc.locate(az, el)                             # flat-terrain intersection, no range
        assert math.hypot(g2.enu[0] - e, g2.enu[1] - n) < 0.5 and g2.method == "terrain"
    # Along-beam variance is the range error, cross-beam grows with distance.
    az, el, rng = loc.aim(0.0, 1000.0, aim_height_m=0.0)
    g = loc.locate(az, el, range_m=rng)
    w, v = np.linalg.eigh(g.cov_en)
    major = v[:, int(np.argmax(w))]
    assert math.sqrt(w.max()) == pytest.approx(1.0, rel=0.05)      # laser range sigma along the beam
    assert abs(major[1]) > 0.999                                     # target due north -> beam along N
    assert math.sqrt(w.min()) == pytest.approx(1000 * math.radians(0.05), rel=0.05)   # cross-beam
    # Box offset in the image shifts the bearing by the right angle.
    lens = LensModel(9.6, 7.2, 640, 480, 25.0)
    g_center = loc.locate(az, el, range_m=rng)
    g_right = loc.locate(az, el, range_m=rng, rect=(0.75, 0.4, 0.05, 0.1), lens=lens)
    hfov = lens.fov_deg(1.0)[0]
    expect = math.degrees(math.atan2(0.275, 0.5 / math.tan(math.radians(hfov) / 2)))
    assert (g_right.bearing_deg - g_center.bearing_deg) == pytest.approx(expect, abs=1e-6)
    # North offset from two landmarks.
    sightings = []
    for e, n in [(500.0, 900.0), (-700.0, -200.0)]:
        lat, lon, _ = site.to_geodetic(e, n, -12.0)
        sightings.append((loc.aim(e, n)[0], lat, lon))
    blank = PTGeoMount("pt-1", 39.92, 32.85, 912.0, 12.0, north_offset_deg=0.0)
    off, rms = calibrate_north(blank, site, sightings)
    assert off == pytest.approx(37.5, abs=0.01) and rms < 0.01


# --------------------------------------------------------------------------- events (document samples)

FIELD_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<EventNotificationAlert xmlns="{NS}" version="2.0">
 <ipAddress>172.6.64.7</ipAddress><channelID>1</channelID><dateTime>2026-10-04T21:30:08+03:00</dateTime>
 <activePostCount>1</activePostCount><eventType>fielddetection</eventType><eventState>active</eventState>
 <eventDescription>fielddetection alarm</eventDescription>
 <DetectionRegionList><DetectionRegionEntry><regionID>1</regionID><sensitivityLevel>50</sensitivityLevel>
  <RegionCoordinatesList><RegionCoordinates><positionX>100</positionX><positionY>500</positionY></RegionCoordinates>
  <RegionCoordinates><positionX>900</positionX><positionY>500</positionY></RegionCoordinates>
  <RegionCoordinates><positionX>900</positionX><positionY>950</positionY></RegionCoordinates></RegionCoordinatesList>
  <detectionTarget>human</detectionTarget>
  <TargetRect><X>0.475</X><Y>0.40</Y><width>0.05</width><height>0.10</height></TargetRect>
 </DetectionRegionEntry></DetectionRegionList>
 <targetSpeed>2</targetSpeed><targetDistance>400</targetDistance>
 <visibleLightAbsoluteHigh><elevation>{{el}}</elevation><azimuth>{{az}}</azimuth><absoluteZoom>4.0</absoluteZoom><focus>1200</focus></visibleLightAbsoluteHigh>
 <deviceLocation><longitudeType>E</longitudeType><latitudeType>N</latitudeType>
  <longitude><degree>32</degree><minute>51</minute><sec>0.0</sec></longitude>
  <latitude><degree>39</degree><minute>55</minute><sec>12.0</sec></latitude></deviceLocation>
 <laserRanging>{{rng}}</laserRanging>
</EventNotificationAlert>"""

TDA_JSON = {"ipAddress": "172.6.64.7", "channelID": 2, "dateTime": "2026-10-04T21:30:08+03:00", "activePostCount": 1,
            "eventType": "TDA", "eventState": "active", "eventDescription": "Temperature Diff Alarm",
            "DetectionRegionList": [{"DetectionRegionEntry": {
                "AlarmRuleList": [{"alarmID": 1, "RegionCoordinatesList": [
                    {"RegionCoordinates": {"positionX": 0.2, "positionY": 0.3}}]}],
                "TDA": {"thermometryUnit": "celsius", "ruleTemperatureDiff": 10.0, "currTemperatureDiff": 14.5,
                        "ruleCalibType": "region", "alarmType": "MaxTemperature", "alarmRule": "greater",
                        "AbsoluteHigh": {"elevation": 125.0, "azimuth": 2345.0, "absoluteZoom": 40.0},
                        "presetNo": 3, "visibleChannel": 1}}}]}


def test_pt_event_details_and_geo_bridge() -> None:
    site = LocalTangentPlane(39.92, 32.85, 900.0)
    mount = PTGeoMount("pt-1", 39.92, 32.85, 912.0, height_above_ground_m=12.0, north_offset_deg=10.0)
    loc = PTGeolocator(mount, site)
    az, el, rng = loc.aim(300.0, 250.0, aim_height_m=0.0)
    ev = AlertEvent.from_xml(X.parse(FIELD_XML.format(az=f"{az:.3f}", el=f"{el:.3f}", rng=f"{rng:.0f}")))
    d = pt_details(ev)
    assert d.category == "perimeter" and d.target_type == "human" and d.laser_range_m == pytest.approx(rng, abs=1)
    assert d.target_rect == (0.475, 0.40, 0.05, 0.10) and d.region_polygon[0] == (0.1, 0.5)
    assert d.device_location.lat == pytest.approx(39.92, abs=1e-6)
    assert d.visible_pose.azimuth == pytest.approx(az, abs=1e-3)
    bridge = PTEventGeoBridge("pt-1", loc, visible_lens=LensModel(5.37, 3.02, 1920, 1080, 4.8))
    obs, geo, ended = bridge.convert(ev, d)
    assert math.hypot(obs.xy[0] - 300.0, obs.xy[1] - 250.0) < 2.0   # box centred, laser range
    obs2, _, _ = bridge.convert(ev, d)
    assert obs2.local_track_id == obs.local_track_id                 # repeated posts -> same pseudo track

    tda = AlertEvent.from_json(json.dumps(TDA_JSON).encode())
    t = pt_details(tda)
    assert t.category == "temperature" and t.temperature.current == 14.5 and t.temperature.preset == 3
    assert t.visible_pose.azimuth == pytest.approx(234.5) and t.visible_pose.elevation == pytest.approx(12.5)


# --------------------------------------------------------------------------- ISAPI client against a mock device

class MockPT:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, bytes, str]] = []
        self.fire = (f'<FireDetection xmlns="{NS}" version="2.0"><enabled>true</enabled><sensitivity>50</sensitivity>'
                     "<fireComfirmTime>5</fireComfirmTime><detectionMode>multipleFarme</detectionMode></FireDetection>")

    def __call__(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        self.sent.append((req.method, p, req.content, req.headers.get("content-type", "")))
        x = lambda body: httpx.Response(200, content=body.encode(), headers={"content-type": "application/xml"})  # noqa: E731
        if p == "/ISAPI/System/deviceInfo":
            return x(f'<DeviceInfo xmlns="{NS}" version="2.0"><deviceName>pt</deviceName><model>DS-2TD6267-75C4L/W</model>'
                     "<firmwareVersion>V5.5.80</firmwareVersion></DeviceInfo>")
        if p == "/ISAPI/System/capabilities":
            return x(f'<DeviceCap xmlns="{NS}" version="2.0"><SysCap><isSupportThermalStreamData>true</isSupportThermalStreamData>'
                     "</SysCap><ThermalCap><isSupportFireDetection>true</isSupportFireDetection>"
                     "<isSupportRealtimeThermometry>true</isSupportRealtimeThermometry></ThermalCap></DeviceCap>")
        if p.startswith("/ISAPI/Thermal/channels/2/") and p.endswith("/capabilities"):
            return x(f'<Cap xmlns="{NS}"/>')
        if p.startswith("/ISAPI/Thermal/channels/1/"):
            return httpx.Response(403, json={"statusCode": 4, "statusString": "Invalid Operation",
                                             "subStatusCode": "notSupport", "errorCode": 1073741825, "errorMsg": "notSupport"})
        if p == "/ISAPI/PTZCtrl/channels/1/capabilities":
            return x(f'<PTZChanelCap xmlns="{NS}"><isSupportAbsoluteEx>true</isSupportAbsoluteEx>'
                     "<isSupportPosition3D>true</isSupportPosition3D></PTZChanelCap>")
        if p == "/ISAPI/PTZCtrl/channels/1/absoluteEx/capabilities":
            return x(f'<PTZAbsoluteEx xmlns="{NS}"><elevation min="-20.000" max="90.000">0</elevation>'
                     '<azimuth min="0" max="360.000">0</azimuth><absoluteZoom min="1" max="32.00">1</absoluteZoom>'
                     '<focalLen min="25" max="225">25</focalLen><objectDistance min="1" max="5000.0">1</objectDistance>'
                     "</PTZAbsoluteEx>")
        if p == "/ISAPI/PTZCtrl/channels/1/absoluteEx" and req.method == "GET":
            return x(f'<PTZAbsoluteEx xmlns="{NS}"><elevation>3.250</elevation><azimuth>271.125</azimuth>'
                     "<absoluteZoom>8.50</absoluteZoom><focalLen>100</focalLen><lookDownUpAngle>3.31</lookDownUpAngle>"
                     "</PTZAbsoluteEx>")
        if p == "/ISAPI/Thermal/channels/2/fireDetection" and req.method == "GET":
            return x(self.fire)
        if p == "/ISAPI/Streaming/channels/1/metadata/subscribeType":
            return httpx.Response(200, json={"rtspURI": "/ISAPI/Streaming/channels/101?type=personalTrack"})
        if p == "/ISAPI/Thermal/channels/2/thermometry/realTimethermometry/rules":
            return httpx.Response(200, json={"RealTimeThermometry": {"ThermometryRulesList": [{"ThermometryRule": {
                "ruleID": 1, "ruleName": "trafo", "ruleCalibType": "region", "maxTemperature": 71.5,
                "minTemperature": 30.2, "averageTemperature": 44.0,
                "MaxTemperaturePoint": {"positionX": 0.41, "positionY": 0.62}}}]}})
        if req.method == "PUT":
            return x(f'<ResponseStatus xmlns="{NS}"><statusCode>1</statusCode><statusString>OK</statusString></ResponseStatus>')
        return httpx.Response(404)


async def test_pt_thermal_client_against_mock_device() -> None:
    dev = MockPT()
    c = HikvisionISAPIClient("cam", "admin", "x")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(dev), base_url="http://cam")
    cam = PTThermalCamera(c)
    prof = await cam.discover()
    assert (prof.visible_channel, prof.thermal_channel, prof.ptz_channel) == (1, 2, 1)
    assert prof.namespace == NS and prof.supports_absolute_ex and prof.supports_fire
    assert prof.ex_range("elevation") == (-20.0, 90.0)

    await cam.move_absolute(361.5, 120.0, focal_len_mm=300, object_distance_m=420.0)
    method, path, body, _ = dev.sent[-1]
    root = X.parse(body)
    assert path.endswith("/absoluteEx") and X.namespace_of(root) == NS
    assert X.text(root, "azimuth") == "1.500" and X.text(root, "elevation") == "90.000"   # wrapped / clamped
    assert X.text(root, "focalLen") == "225" and X.text(root, "zoomType") == "focalLen"
    assert X.text(root, "objectDistance") == "420.0"

    pose = await cam.get_pose()
    assert (pose.azimuth, pose.focal_len_mm, pose.pitch_sensor_deg) == (271.125, 100.0, 3.31)

    await cam.center_on(1.0, 0.5)
    assert b"<positionX>255</positionX>" in dev.sent[-1][2] and b"<positionY>128</positionY>" in dev.sent[-1][2]

    await cam.update_fire_detection({"sensitivity": 80})
    _, path, body, ctype = dev.sent[-1]
    assert ctype.startswith("multipart/form-data") and b'name="FireDetection"' in body
    assert b"<sensitivity>80</sensitivity>" in body and b"<fireComfirmTime>5</fireComfirmTime>" in body
    assert b"ns0:" not in body

    assert await cam.subscribe_metadata(["thermometry", "fireDetection"]) == "/ISAPI/Streaming/channels/101?type=personalTrack"
    rules = await cam.realtime_rules()
    assert rules[0].max_c == 71.5 and rules[0].max_pos == (0.41, 0.62)

    with pytest.raises(ISAPIError) as ei:
        await c.get_xml("/ISAPI/Thermal/channels/1/fireDetection")
    assert ei.value.error_code == 1073741825 and "capability" in str(ei.value)
    await c.aclose()


async def test_c2_thermal_endpoints_and_ptz_token() -> None:
    import time as _time

    from ipcam.fusion import GlobalTrackManager
    from ipcam.fusion.c2 import C2Server
    from ipcam.fusion.c2.thermal_panel import ThermalPanel
    from ipcam.fusion.service import FusionService
    from ipcam.fusion.transport import InProcessBus
    from ipcam.thermal import ThermalFrame

    sc = scene(fire_at=(90.0, -15.0), fire_start_s=0.0)
    state = {"i": 0}

    def source() -> ThermalFrame:
        state["i"] += 1
        return ThermalFrame(sc.render(state["i"] * 0.25), _time.monotonic(), _time.time(), 0, state["i"])

    alarms: list[tuple[str, str]] = []
    panel = ThermalPanel(source, rois=[RoiMonitor("r", [[0, 0], [1, 0], [1, 1], [0, 1]], prealarm_c=50, alarm_c=100,
                                                  dwell_s=0.0)], on_alarm=lambda k, t: alarms.append((k, t)))
    for _ in range(4):
        panel.ingest(source())                       # unattended analysis raises alarms without a viewer
    assert any(k == "fire" for k, _ in alarms) and any(k.startswith("roi_") for k, _ in alarms)
    commands: list[dict] = []

    async def ptz_control(cmd: dict) -> dict:
        commands.append(cmd)
        if cmd.get("action") not in ("stop", "slew"):
            raise ValueError("unknown")
        return {"message": "ok"}

    svc = FusionService(InProcessBus(), GlobalTrackManager())
    srv = C2Server(svc, port=0, thermal=panel, ptz_control=ptz_control)
    port = await srv.start()
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            st = (await c.get("/api/thermal/state")).json()
            assert st["available"] and st["width"] == 384 and len(st["grid"]) == st["grid_w"] * st["grid_h"]
            assert st["max"] > 150 and st["fires"]
            png = await c.get("/api/thermal/frame.png", params={"palette": "white_hot"})
            assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
            html = (await c.get("/")).text
            assert "__C2_TOKEN__" not in html and srv.token in html
            assert (await c.post("/api/ptz", json={"action": "stop"})).status_code == 403
            ok = await c.post("/api/ptz", json={"action": "slew", "x": 10, "y": 5}, headers={"X-C2-Token": srv.token})
            assert ok.status_code == 200 and ok.json()["ok"]
            bad = await c.post("/api/ptz", json={"action": "x"}, headers={"X-C2-Token": srv.token})
            assert bad.status_code == 400
            assert (await c.get("/api/cop")).json()["capabilities"] == {"thermal": True, "ptz_control": True}
        assert commands == [{"action": "slew", "x": 10, "y": 5}, {"action": "x"}]
    finally:
        await srv.stop()


def test_u16_decoding_and_capability_parser() -> None:
    t = np.array([[20.0, 25.5], [-5.0, 60.0]], np.float32)
    assert np.allclose(decode_u16_temperatures(np.round((t + 273.15) * 10).astype(np.uint16)), t, atol=0.06)
    cap = X.Capability.from_xml(X.parse(
        f'<PTZAbsoluteEx xmlns="{NS}"><zoomType opt="absoluteZoom,focalLen" def="absoluteZoom">x</zoomType>'
        '<elevation min="-90.000" max="270.000">0</elevation><objectInfo><id min="1" max="9"/></objectInfo></PTZAbsoluteEx>'))
    assert cap.options("zoomType") == ("absoluteZoom", "focalLen") and cap.range("elevation") == (-90.0, 270.0)
    assert "objectInfo" in cap and cap.range("..id") == (1.0, 9.0)
    assert parse_realtime_thermometry({"RealTimeThermometry": {"ThermometryRulesList": []}}) == []
