"""ISAPI integration for bi-spectrum thermal pan-tilt cameras (DS-2TD / HM-TD PT series).

What differs from a plain IP camera, and is handled here:

* **Two video channels on one head** -- an optical and a thermal channel share the pan-tilt;
  which is which is *probed*, not assumed (thermal APIs only answer on the thermal channel).
* **High-precision PTZ** -- ``absoluteEx`` takes float degrees (0.001 deg step), zoom either as
  ratio or *focal length in mm*, focus by *object distance in metres*, and reports the
  inclinometer pitch (``lookDownUpAngle``) next to the motor-step tilt. Long thermal lenses
  (75-300 mm) have depth of field of a few metres, so commanding focus by range is the
  difference between a sharp and an unusable image after a slew.
* **Radiometry** -- thermometry mode, rules, real-time rule temperatures, pixel-to-pixel
  thermometry and raw thermal stream parameters.
* **Fire / smoke detection** -- configured through a multipart form body.
* **Metadata** -- per-type enable switches plus an RTSP subscription URI.

Every capability-gated call checks the device's capability documents first and raises
:class:`NotSupportedError` locally instead of sending requests a model will reject.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt

from ..errors import ISAPIError, NotSupportedError
from . import xmlutil as X
from .client import HikvisionISAPIClient
from .models import DeviceInfo
from .multipart import MultipartStreamParser, boundary_from_content_type

log = logging.getLogger(__name__)

# Native thermal detector formats (the encoded thermal stream is often upscaled).
THERMAL_SENSOR_SIZES = {(160, 120), (256, 192), (320, 240), (384, 288), (640, 480), (640, 512), (1280, 1024)}


@dataclass(frozen=True, slots=True)
class PTZPoseEx:
    azimuth: float                 # degrees [0, 360)
    elevation: float               # degrees; positive = down on most heads (see mount config)
    zoom: float                    # zoom ratio
    focus: int | None = None
    focal_len_mm: float | None = None
    object_distance_m: float | None = None
    pitch_sensor_deg: float | None = None   # inclinometer (lookDownUpAngle), more accurate than tilt
    rotate: float | None = None


@dataclass(frozen=True, slots=True)
class ThermometryRuleReading:
    rule_id: int
    name: str
    kind: str                      # point | line | region
    max_c: float | None
    min_c: float | None
    avg_c: float | None
    max_pos: tuple[float, float] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClockOffset:
    offset_s: float                # device_time - host_time
    rtt_s: float
    device_time: datetime
    uncertainty_s: float           # +/- rtt/2 plus the device's 1 s time resolution


@dataclass
class DeviceProfile:
    device: DeviceInfo
    system: X.Capability
    visible_channel: int
    thermal_channel: int | None
    ptz_channel: int
    ptz: X.Capability | None = None
    absolute_ex: X.Capability | None = None
    thermal: X.Capability | None = None
    namespace: str = X.HIK_NS
    notes: list[str] = field(default_factory=list)

    # -- capability shortcuts -------------------------------------------------
    @property
    def supports_absolute_ex(self) -> bool:
        return self.ptz is not None and (self.ptz.supports("isSupportAbsoluteEx") or self.absolute_ex is not None)

    @property
    def supports_position3d(self) -> bool:
        return self.ptz is not None and self.ptz.supports("isSupportPosition3D")

    @property
    def supports_onepush_focus(self) -> bool:
        return self.ptz is not None and self.ptz.supports("isSupportOnepushfoucsStart")

    def thermal_flag(self, name: str) -> bool:
        return self.system.supports(f"ThermalCap.{name}") or (self.thermal is not None and self.thermal.supports(name))

    @property
    def supports_fire(self) -> bool:
        return self.thermal_flag("isSupportFireDetection")

    @property
    def supports_thermometry(self) -> bool:
        return self.thermal_flag("isSupportThermometry")

    @property
    def supports_realtime_thermometry(self) -> bool:
        return self.thermal_flag("isSupportRealtimeThermometry")

    @property
    def supports_thermal_stream(self) -> bool:
        return self.system.supports("SysCap.isSupportThermalStreamData") or self.thermal_flag("isSupportThermalStreamParam")

    def ex_range(self, name: str) -> tuple[float, float] | None:
        return self.absolute_ex.range(name) if self.absolute_ex is not None else None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.device.model, "firmware": self.device.firmware_version,
            "visible_channel": self.visible_channel, "thermal_channel": self.thermal_channel,
            "ptz_channel": self.ptz_channel, "namespace": self.namespace,
            "absoluteEx": self.supports_absolute_ex, "position3D": self.supports_position3d,
            "onePushFocus": self.supports_onepush_focus, "fireDetection": self.supports_fire,
            "thermometry": self.supports_thermometry, "realtimeThermometry": self.supports_realtime_thermometry,
            "thermalStream": self.supports_thermal_stream,
            "elevation_range": self.ex_range("elevation"), "zoom_range": self.ex_range("absoluteZoom"),
            "focal_len_range": self.ex_range("focalLen"), "notes": list(self.notes),
        }


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


class PTThermalCamera:
    """High-level API over :class:`HikvisionISAPIClient` for the thermal PT series."""

    def __init__(self, client: HikvisionISAPIClient) -> None:
        self.c = client
        self.profile: DeviceProfile | None = None

    # ================================================================== discovery

    async def _try_xml(self, path: str) -> ET.Element | None:
        try:
            return await self.c.get_xml(path)
        except (ISAPIError, ValueError, ET.ParseError):
            return None

    async def discover(self, *, candidate_channels: tuple[int, ...] = (1, 2)) -> DeviceProfile:
        device = DeviceInfo.from_xml(await self.c.get_xml("/ISAPI/System/deviceInfo"))
        sys_root = await self._try_xml("/ISAPI/System/capabilities")
        system = X.Capability.from_xml(sys_root) if sys_root is not None else X.Capability("DeviceCap")
        notes: list[str] = []

        # Thermal channel: the one that answers thermal capability queries.
        thermal_ch: int | None = None
        for ch in sorted(candidate_channels, reverse=True):          # thermal is usually 2
            for probe in (f"/ISAPI/Thermal/channels/{ch}/thermometry/basicParam/capabilities",
                          f"/ISAPI/Thermal/channels/{ch}/fireDetection/capabilities",
                          f"/ISAPI/Thermal/channels/{ch}/streamParam/capabilities"):
                if await self._try_xml(probe) is not None:
                    thermal_ch = ch
                    break
            if thermal_ch is not None:
                break
        if thermal_ch is None:
            notes.append("no channel answered thermal capability probes")
        visible_ch = next((ch for ch in candidate_channels if ch != thermal_ch), candidate_channels[0])

        ptz_ch, ptz_cap = visible_ch, None
        for ch in (visible_ch, thermal_ch):
            if ch is None:
                continue
            root = await self._try_xml(f"/ISAPI/PTZCtrl/channels/{ch}/capabilities")
            if root is not None:
                ptz_ch, ptz_cap = ch, X.Capability.from_xml(root)
                break
        absolute_ex = None
        if ptz_cap is not None:
            root = await self._try_xml(f"/ISAPI/PTZCtrl/channels/{ptz_ch}/absoluteEx/capabilities")
            absolute_ex = X.Capability.from_xml(root) if root is not None else None
        thermal_root = await self._try_xml("/ISAPI/Thermal/capabilities")
        self.profile = DeviceProfile(
            device=device, system=system, visible_channel=visible_ch, thermal_channel=thermal_ch,
            ptz_channel=ptz_ch, ptz=ptz_cap, absolute_ex=absolute_ex,
            thermal=X.Capability.from_xml(thermal_root) if thermal_root is not None else None,
            namespace=self.c.namespace, notes=notes)
        log.info("PT thermal profile: %s", self.profile.summary())
        return self.profile

    def _need(self, ok: bool, what: str) -> None:
        if self.profile is not None and not ok:
            raise NotSupportedError(f"{self.profile.device.model or 'device'} does not support {what}", path=what)

    def _ptz_ch(self, channel: int | None) -> int:
        return channel if channel is not None else (self.profile.ptz_channel if self.profile else 1)

    def _th_ch(self, channel: int | None) -> int:
        if channel is not None:
            return channel
        if self.profile is None or self.profile.thermal_channel is None:
            raise NotSupportedError("thermal channel unknown (run discover())", path="thermal")
        return self.profile.thermal_channel

    # ================================================================== PTZ

    def _clamp_ex(self, name: str, v: float) -> float:
        r = self.profile.ex_range(name) if self.profile else None
        return min(max(v, r[0]), r[1]) if r else v

    async def get_pose(self, channel: int | None = None) -> PTZPoseEx:
        ch = self._ptz_ch(channel)
        if self.profile is None or self.profile.supports_absolute_ex:
            root = await self._try_xml(f"/ISAPI/PTZCtrl/channels/{ch}/absoluteEx")
            if root is not None and X.text(root, "azimuth") is not None:
                return PTZPoseEx(
                    azimuth=_f(X.text(root, "azimuth")) or 0.0, elevation=_f(X.text(root, "elevation")) or 0.0,
                    zoom=_f(X.text(root, "absoluteZoom")) or 1.0,
                    focus=int(_f(X.text(root, "focus")) or 0) or None,
                    focal_len_mm=_f(X.text(root, "focalLen")),
                    object_distance_m=_f(X.text(root, "objectDistance")),
                    pitch_sensor_deg=_f(X.text(root, "lookDownUpAngle")),
                    rotate=_f(X.text(root, "absoluteRotate")))
        root = await self.c.get_xml(f"/ISAPI/PTZCtrl/channels/{ch}/status")
        ah = X.find(root, "AbsoluteHigh")
        if ah is None:
            raise ISAPIError("PTZ status without AbsoluteHigh", path="status")
        return PTZPoseEx((_f(X.text(ah, "azimuth")) or 0) / 10, (_f(X.text(ah, "elevation")) or 0) / 10,
                         (_f(X.text(ah, "absoluteZoom")) or 10) / 10)

    async def move_absolute(self, azimuth: float, elevation: float, *, zoom: float | None = None,
                            focal_len_mm: float | None = None, object_distance_m: float | None = None,
                            speed_dps: float | None = None, channel: int | None = None) -> None:
        """Absolute move with float precision where supported.

        ``focal_len_mm`` zooms by focal length (exact FOV control); ``object_distance_m``
        pre-focuses at the target range so the image is sharp on arrival.
        """
        ch = self._ptz_ch(channel)
        az = azimuth % 360.0
        if self.profile is None or self.profile.supports_absolute_ex:
            fields: dict[str, Any] = {
                "elevation": X.fmt_float(self._clamp_ex("elevation", elevation)),
                "azimuth": X.fmt_float(self._clamp_ex("azimuth", az)),
            }
            if focal_len_mm is not None:
                fields["focalLen"] = int(round(self._clamp_ex("focalLen", focal_len_mm)))
                fields["zoomType"] = "focalLen"
            elif zoom is not None:
                fields["absoluteZoom"] = X.fmt_float(self._clamp_ex("absoluteZoom", zoom), 2)
                fields["zoomType"] = "absoluteZoom"
            if speed_dps is not None:
                fields["horizontalSpeed"] = X.fmt_float(abs(speed_dps), 2)
                fields["verticalSpeed"] = X.fmt_float(abs(speed_dps), 2)
            if object_distance_m is not None:
                fields["objectDistance"] = X.fmt_float(self._clamp_ex("objectDistance", object_distance_m), 1)
            await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{ch}/absoluteEx", self.c.xml("PTZAbsoluteEx", fields))
            return
        await self.c.ptz_absolute(az, elevation, zoom if zoom is not None else 1.0, channel=ch)

    async def center_on(self, x: float, y: float, channel: int | None = None) -> None:
        """Centre the image on a normalised point (0..1) -- 3D positioning."""
        self._need(self.profile is None or self.profile.supports_position3d, "position3D")
        px, py = (max(0, min(255, int(round(v * 255)))) for v in (x, y))
        pt = {"positionX": px, "positionY": py}
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/position3D",
                             self.c.xml("Position3D", {"StartPoint": pt, "EndPoint": dict(pt)}))

    async def zoom_to_rect(self, x1: float, y1: float, x2: float, y2: float, *, zoom_in: bool = True,
                           channel: int | None = None) -> None:
        """Centre and zoom on a normalised rectangle. Zoom-out is signalled by StartX > EndX."""
        self._need(self.profile is None or self.profile.supports_position3d, "position3D")
        lx, rx = sorted((x1, x2))
        ty, by = sorted((y1, y2))
        a, b = ((lx, ty), (rx, by)) if zoom_in else ((rx, ty), (lx, by))
        q = lambda v: max(0, min(255, int(round(v * 255))))  # noqa: E731
        await self.c.put_xml(
            f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/position3D",
            self.c.xml("Position3D", {"StartPoint": {"positionX": q(a[0]), "positionY": q(a[1])},
                                      "EndPoint": {"positionX": q(b[0]), "positionY": q(b[1])}}))

    async def continuous(self, pan: int, tilt: int, zoom: int = 0, rotate: int | None = None,
                         channel: int | None = None) -> None:
        clamp = lambda v: max(-100, min(100, int(v)))  # noqa: E731
        fields: dict[str, Any] = {"pan": clamp(pan), "tilt": clamp(tilt), "zoom": clamp(zoom)}
        if rotate is not None:
            fields["rotate"] = clamp(rotate)
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/continuous",
                             self.c.xml("PTZData", fields))

    async def stop(self, channel: int | None = None) -> None:
        await self.continuous(0, 0, 0, channel=channel)

    async def one_touch_focus(self, channel: int | None = None) -> None:
        self._need(self.profile is None or self.profile.supports_onepush_focus, "one-touch focus")
        await self.c.request("PUT", f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/onepushfoucs/start")

    async def reset_focus(self, channel: int | None = None) -> None:
        await self.c.request("PUT", f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/onepushfoucs/reset")

    async def set_home(self, channel: int | None = None) -> None:
        """Define the current pose as PT(0, 0) -- the reference of all absolute angles."""
        await self.c.request("PUT", f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/homeposition")

    async def goto_home(self, channel: int | None = None) -> None:
        await self.c.request("PUT", f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/homeposition/goto")

    async def lock(self, seconds: int, channel: int | None = None) -> None:
        """Lock PTZ against other operators while an engagement is running."""
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/lockPTZ",
                             self.c.xml("LockPTZ", {"lockTime": int(seconds)}))

    async def set_aux(self, aux_id: int, aux_type: str, on: bool, channel: int | None = None) -> None:
        """Wiper / light / heater: ``aux_type`` as reported by the capability (e.g. LIGHT, WIPER)."""
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/auxcontrols/{aux_id}",
                             self.c.xml("PTZAux", {"id": aux_id, "type": aux_type, "status": "on" if on else "off"}))

    async def list_presets(self, channel: int | None = None) -> list[dict[str, Any]]:
        root = await self.c.get_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/presets")
        return [X.to_dict(p) for p in X.findall(root, "PTZPreset")]

    async def goto_preset(self, preset: int, channel: int | None = None) -> None:
        await self.c.request("PUT", f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/presets/{preset}/goto")

    async def save_preset(self, preset: int, name: str, channel: int | None = None) -> None:
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/presets/{preset}",
                             self.c.xml("PTZPreset", {"id": preset, "presetName": name, "enabled": True}))

    async def hand_over_geo_target(self, target_id: int, lat: float, lon: float, *, course_deg: float,
                                   speed_mps: float, channel: int | None = None) -> None:
        """Feed a fused geographic track to the head's own continuous tracking (radar/PTZ
        style linkage), where the model supports ``objectInfo`` in ``absoluteEx``."""
        caps = self.profile.absolute_ex if self.profile else None
        self._need(caps is None or "objectInfo" in caps, "absoluteEx objectInfo tracking")

        def dms(v: float) -> dict[str, Any]:
            a = abs(v)
            d = int(a)
            m = int((a - d) * 60)
            s = (a - d - m / 60) * 3600
            return {"degree": d, "minute": m, "sec": X.fmt_float(s, 2)}

        body = self.c.xml("PTZAbsoluteEx", {
            "isContinuousTrackingEnabled": True,
            "objectInfo": {
                "id": target_id,
                "positionInfo": {"longitudeType": "E" if lon >= 0 else "W", "latitudeType": "N" if lat >= 0 else "S",
                                 "longitude": dms(lon), "latitude": dms(lat)},
                "motionDirection": X.fmt_float(course_deg % 360, 1),
                "motionSpeed": X.fmt_float(max(0.0, speed_mps), 2),
            },
        })
        await self.c.put_xml(f"/ISAPI/PTZCtrl/channels/{self._ptz_ch(channel)}/absoluteEx", body)

    # ================================================================== thermography

    async def get_thermometry_mode(self, channel: int | None = None) -> dict[str, Any]:
        return X.to_dict(await self.c.get_xml(f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/thermometryMode"))

    async def set_thermometry_mode(self, mode: str, *, roi: bool | None = None, channel: int | None = None) -> None:
        if mode not in ("normal", "expert", "AI"):
            raise ValueError("mode must be normal | expert | AI")
        fields: dict[str, Any] = {"mode": mode}
        if roi is not None:
            fields["thermometryROIEnabled"] = roi
        await self.c.put_xml(f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/thermometryMode",
                             self.c.xml("ThermometryMode", fields))

    async def update_basic_param(self, changes: dict[str, Any], channel: int | None = None) -> None:
        """Read-modify-write of ``thermometry/basicParam`` (emissivity, distance, unit ...),
        keeping every field the model returned that we do not touch."""
        path = f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/thermometry/basicParam"
        root = await self.c.get_xml(path)
        for key, value in changes.items():
            X.set_text(root, key, value)
        await self.c.put_xml(path, X.serialize(root))

    async def realtime_rules(self, channel: int | None = None, rule_id: int | None = None
                             ) -> list[ThermometryRuleReading]:
        self._need(self.profile is None or self.profile.supports_realtime_thermometry, "real-time thermometry")
        base = f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/thermometry/realTimethermometry/rules"
        obj = await self.c.get_json(base if rule_id is None else f"{base}/{rule_id}")
        return parse_realtime_thermometry(obj)

    async def jpeg_with_temperatures(self, channel: int | None = None) -> ThermalSnapshot:
        """Snapshot + per-pixel temperature matrix (``jpegPicWithAppendData``)."""
        resp = await self.c.request("GET", f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/thermometry/"
                                           "jpegPicWithAppendData", params={"format": "json"}, timeout=15.0)
        return parse_jpeg_with_append_data(resp.content, resp.headers.get("content-type", ""))

    async def get_stream_param(self, channel: int | None = None) -> dict[str, Any]:
        return X.to_dict(await self.c.get_xml(f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/streamParam"))

    async def set_stream_data_type(self, data_type: str, channel: int | None = None) -> None:
        """Switch the raw thermal stream payload (field name as reported by the device)."""
        path = f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/streamParam"
        root = await self.c.get_xml(path)
        node = next((n for n in root.iter() if X.local(n.tag).lower().endswith("type")), None)
        if node is None:
            raise NotSupportedError("streamParam has no data type field", path=path)
        node.text = data_type
        await self.c.put_xml(path, X.serialize(root))

    @staticmethod
    def thermal_stream_path(channel: int, data_type: str = "pixel-to-pixel_thermometry_data") -> str:
        if data_type not in ("thermal_raw_data", "pixel-to-pixel_thermometry_data", "real-time_raw_data"):
            raise ValueError(f"unknown thermal stream type {data_type!r}")
        return f"/ISAPI/Streaming/thermal/channels/{channel}/streamType/{data_type}"

    # ================================================================== fire / smoke

    async def get_fire_detection(self, channel: int | None = None) -> dict[str, Any]:
        return X.to_dict(await self.c.get_xml(f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/fireDetection"))

    async def update_fire_detection(self, changes: dict[str, Any], channel: int | None = None) -> None:
        """Read-modify-write; the device expects the XML as form unit ``FireDetection``."""
        self._need(self.profile is None or self.profile.supports_fire, "fire detection")
        path = f"/ISAPI/Thermal/channels/{self._th_ch(channel)}/fireDetection"
        root = await self.c.get_xml(path)
        for key, value in changes.items():
            X.set_text(root, key, value)
        body = X.serialize(root)
        try:
            await self.c.put_form(path, [("FireDetection", body, "application/xml")])
        except ISAPIError as exc:
            if exc.http_status not in (400, 415):
                raise
            await self.c.put_xml(path, body)        # older firmwares take a plain XML body

    # ================================================================== metadata

    async def enable_metadata(self, event_type: str, enable: bool = True, channel: int | None = None) -> None:
        ch = channel if channel is not None else (self.profile.visible_channel if self.profile else 1)
        await self.c.put_xml(f"/ISAPI/Streaming/channels/{ch}/metadata/{event_type}",
                             self.c.xml("SingleMetadataCfg", {"Metadata": {"type": event_type, "enable": enable}}))

    async def subscribe_metadata(self, types: list[str], channel: int | None = None) -> str:
        """Returns the RTSP URI whose ``isapi.metadata`` track carries the subscribed types."""
        ch = channel if channel is not None else (self.profile.visible_channel if self.profile else 1)
        obj = await self.c.send_json("POST", f"/ISAPI/Streaming/channels/{ch}/metadata/subscribeType",
                                     {"type": list(types)})
        uri = (obj or {}).get("rtspURI")
        if not uri:
            raise ISAPIError("metadata subscription returned no rtspURI", path="subscribeType")
        return str(uri)

    # ================================================================== system

    async def get_units(self) -> dict[str, Any]:
        return await self.c.get_json("/ISAPI/System/unitConfig")

    async def set_units(self, *, temperature: str = "degreeCentigrade", distance: str = "meter") -> None:
        await self.c.send_json("PUT", "/ISAPI/System/unitConfig",
                               {"enabled": True, "temperatureRange": temperature, "distanceUnit": distance})

    async def measure_clock_offset(self, samples: int = 5) -> ClockOffset:
        """NTP-style offset of the device clock against the host (minimum-RTT sample).

        Event ``dateTime`` values come from the device clock; fusing them with host-side
        tracks needs the offset, even when both claim NTP sync.
        """
        best: ClockOffset | None = None
        for _ in range(samples):
            t0 = time.time()
            root = await self.c.get_xml("/ISAPI/System/time")
            t1 = time.time()
            txt = X.text(root, "localTime")
            if not txt:
                raise ISAPIError("device time without localTime", path="/ISAPI/System/time")
            dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            rtt = t1 - t0
            off = dt.timestamp() - (t0 + rtt / 2)
            cand = ClockOffset(off, rtt, dt, rtt / 2 + 0.5)
            if best is None or cand.rtt_s < best.rtt_s:
                best = cand
            await asyncio.sleep(0.05)
        assert best is not None
        return best


# ====================================================================== payload parsers

def parse_realtime_thermometry(obj: Any) -> list[ThermometryRuleReading]:
    """Normalise the real-time thermometry JSON (field names vary slightly by firmware)."""
    root = obj.get("RealTimeThermometry", obj) if isinstance(obj, dict) else {}
    rules = root.get("ThermometryRulesList") or root.get("rulesList") or root.get("ThermometryRules") or []
    if isinstance(rules, dict):
        rules = [rules]
    out: list[ThermometryRuleReading] = []
    for item in rules:
        r = item.get("ThermometryRule", item) if isinstance(item, dict) else {}

        def num(*keys: str) -> float | None:
            for k in keys:
                if k in r and r[k] is not None:
                    try:
                        return float(r[k])
                    except (TypeError, ValueError):
                        pass
            return None

        pos = r.get("MaxTemperaturePoint") or r.get("HighestPoint") or r.get("maxTemperaturePoint")
        max_pos = (float(pos.get("positionX", 0)), float(pos.get("positionY", 0))) if isinstance(pos, dict) else None
        out.append(ThermometryRuleReading(
            rule_id=int(num("ruleID", "id") or 0), name=str(r.get("ruleName", r.get("name", ""))),
            kind=str(r.get("ruleCalibType", r.get("type", "region"))),
            max_c=num("maxTemperature", "highestTemperature", "maxTemp"),
            min_c=num("minTemperature", "lowestTemperature", "minTemp"),
            avg_c=num("averageTemperature", "avgTemperature", "averageTemp"),
            max_pos=max_pos, raw=r))
    return out


@dataclass
class ThermalSnapshot:
    meta: dict[str, Any]
    jpeg: bytes | None
    temperatures: npt.NDArray[np.float32] | None     # (H, W) degrees Celsius
    width: int
    height: int


def parse_jpeg_with_append_data(body: bytes, content_type: str) -> ThermalSnapshot:
    """Decode the multipart response: JSON header, JPEG, and the temperature matrix.

    The ISAPI PT-series document names this API but does not specify the binary layout.
    The parser therefore accepts the two layouts seen in practice -- float32 degC and
    uint16 radiometric counts -- and cross-checks them against the geometry in the JSON
    header (``jpegPicWidth``/``jpegPicHeight``), failing loudly when nothing fits rather
    than returning a plausible-looking but wrong matrix. Verify on the target firmware.
    """
    boundary = boundary_from_content_type(content_type)
    parts = list(MultipartStreamParser(boundary).feed(body)) if boundary else []
    meta: dict[str, Any] = {}
    jpeg = None
    blob = None
    for p in parts:
        ct = p.content_type
        if "json" in ct or p.body.lstrip()[:1] == b"{":
            with contextlib.suppress(ValueError):
                meta = json.loads(p.body)
                meta = meta.get("JpegPictureWithAppendData", meta)
        elif ct.startswith("image/") or p.body[:2] == b"\xff\xd8":
            jpeg = p.body
        else:
            blob = p.body
    w = int(meta.get("jpegPicWidth") or meta.get("width") or 0)
    h = int(meta.get("jpegPicHeight") or meta.get("height") or 0)
    temps = None
    if blob is not None and w and h:
        if len(blob) >= w * h * 4:
            temps = np.frombuffer(blob[: w * h * 4], dtype="<f4").reshape(h, w).copy()
        elif len(blob) >= w * h * 2:
            temps = decode_u16_temperatures(np.frombuffer(blob[: w * h * 2], dtype="<u2").reshape(h, w))
        else:
            raise ValueError(f"temperature payload {len(blob)} B does not fit {w}x{h}")
    return ThermalSnapshot(meta, jpeg, temps, w, h)


def decode_u16_temperatures(raw: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """uint16 radiometric matrices come either as deci-Kelvin or centi-degree with offset.

    The encoding is picked by physical plausibility of the scene median (a real scene sits
    between -40 and +80 degC) -- documented ambiguity across firmwares, resolved by data.
    """
    med = float(np.median(raw))
    candidates = {
        "deciK": raw.astype(np.float32) / 10.0 - 273.15,
        "centiK": raw.astype(np.float32) / 100.0 - 273.15,
        "centiC_offset": raw.astype(np.float32) / 100.0 - 100.0,
    }
    for name, arr in candidates.items():
        m = {"deciK": med / 10 - 273.15, "centiK": med / 100 - 273.15, "centiC_offset": med / 100 - 100}[name]
        if -40.0 <= m <= 80.0:
            return arr
    raise ValueError(f"cannot infer uint16 temperature encoding (median raw {med:.0f})")
