"""Typed ISAPI payload models."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from . import xmlutil as X


class IRCutMode(str, Enum):
    DAY = "day"        # IR-cut filter engaged: colour imaging
    NIGHT = "night"    # filter removed, IR illuminator on: monochrome
    AUTO = "auto"      # light-sensor driven; current state not reported by this endpoint
    SCHEDULE = "schedule"
    EVENT_TRIGGER = "eventTrigger"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IRCutFilterState:
    mode: IRCutMode
    raw_type: str
    night_to_day_level: int | None = None
    night_to_day_time_s: int | None = None
    schedule_begin: str | None = None
    schedule_end: str | None = None

    @property
    def is_forced_night(self) -> bool:
        return self.mode is IRCutMode.NIGHT

    @property
    def is_forced_day(self) -> bool:
        return self.mode is IRCutMode.DAY

    @classmethod
    def from_xml(cls, root: ET.Element) -> IRCutFilterState:
        raw = X.text(root, "IrcutFilterType") or X.text(root, "IrcutFilterTime") or "unknown"
        try:
            mode = IRCutMode(raw)
        except ValueError:
            mode = IRCutMode.UNKNOWN
        level = X.text(root, "nightToDayFilterLevel")
        ttime = X.text(root, "nightToDayFilterTime")
        return cls(
            mode=mode,
            raw_type=raw,
            night_to_day_level=int(level) if level and level.isdigit() else None,
            night_to_day_time_s=int(ttime) if ttime and ttime.isdigit() else None,
            schedule_begin=X.text(root, "Schedule", "TimeRange", "beginTime"),
            schedule_end=X.text(root, "Schedule", "TimeRange", "endTime"),
        )


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device_name: str
    model: str
    serial_number: str
    mac_address: str
    firmware_version: str
    firmware_released: str
    device_type: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_xml(cls, root: ET.Element) -> DeviceInfo:
        g = lambda k: X.text(root, k) or ""
        return cls(
            device_name=g("deviceName"),
            model=g("model"),
            serial_number=g("serialNumber"),
            mac_address=g("macAddress"),
            firmware_version=g("firmwareVersion"),
            firmware_released=g("firmwareReleasedDate"),
            device_type=g("deviceType"),
            raw=X.to_dict(root),
        )


@dataclass(frozen=True, slots=True)
class StreamingChannelInfo:
    channel_id: int
    enabled: bool
    codec: str            # "H.264" | "H.265" | "MJPEG"
    width: int
    height: int
    max_fps: float
    gop_frames: int | None
    bitrate_kbps: int | None
    smart_codec: bool

    @property
    def ffmpeg_codec(self) -> str:
        c = self.codec.upper().replace(".", "").replace("-", "")
        return {"H264": "h264", "H265": "hevc", "HEVC": "hevc", "MJPEG": "mjpeg"}.get(c, "h264")

    @classmethod
    def from_xml(cls, root: ET.Element) -> StreamingChannelInfo:
        vid = X.find(root, "Video")
        if vid is None:
            raise ValueError("StreamingChannel without <Video>")

        def i(*path: str) -> int | None:
            v = X.text(vid, *path)
            return int(v) if v and v.lstrip("-").isdigit() else None

        # maxFrameRate is expressed in 1/100 fps (2500 -> 25 fps).
        fps_raw = i("maxFrameRate") or 0
        smart = (X.text(vid, "SmartCodec", "enabled") or "false").lower() == "true"
        return cls(
            channel_id=int(X.text(root, "id") or 0),
            enabled=(X.text(root, "enabled") or "true").lower() == "true",
            codec=X.text(vid, "videoCodecType") or "H.264",
            width=i("videoResolutionWidth") or 0,
            height=i("videoResolutionHeight") or 0,
            max_fps=fps_raw / 100.0 if fps_raw > 100 else float(fps_raw),
            gop_frames=i("GovLength"),
            bitrate_kbps=i("vbrUpperCap") or i("constantBitRate"),
            smart_codec=smart,
        )


class EventState(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class EventPhase(str, Enum):
    """Debounced lifecycle synthesised by the alert listener (Hikvision repeats 'active'
    roughly once per second and frequently never sends a closing 'inactive')."""

    START = "start"
    UPDATE = "update"
    END = "end"
    INSTANT = "instant"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    event_type: str                 # VMD, linedetection, fielddetection, shelteralarm, ...
    state: EventState
    channel_id: int | None
    timestamp: datetime
    received_at: datetime
    active_post_count: int
    description: str
    ip_address: str | None
    region_ids: tuple[str, ...] = ()
    target_type: str | None = None
    phase: EventPhase = EventPhase.INSTANT
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def is_heartbeat(self) -> bool:
        # Hikvision emits "videoloss/inactive" every ~10 s as an alert-stream keep-alive.
        return self.event_type.lower() == "videoloss" and self.state is EventState.INACTIVE

    @property
    def key(self) -> tuple[str, int | None, tuple[str, ...]]:
        return (self.event_type.lower(), self.channel_id, self.region_ids)

    def with_phase(self, phase: EventPhase) -> AlertEvent:
        return AlertEvent(
            self.event_type, self.state, self.channel_id, self.timestamp, self.received_at,
            self.active_post_count, self.description, self.ip_address, self.region_ids,
            self.target_type, phase, self.raw,
        )

    @staticmethod
    def _parse_time(value: str | None, fallback: datetime) -> datetime:
        if not value:
            return fallback
        v = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(v)
        except ValueError:
            return fallback
        return dt if dt.tzinfo else dt.replace(tzinfo=fallback.tzinfo)

    @classmethod
    def from_xml(cls, root: ET.Element) -> AlertEvent:
        now = datetime.now(timezone.utc).astimezone()
        channel = X.text(root, "channelID") or X.text(root, "dynChannelID")
        regions = tuple(
            r for r in (X.text(e, "regionID") for e in X.findall(root, "DetectionRegionEntry")) if r
        )
        apc = X.text(root, "activePostCount") or "0"
        return cls(
            event_type=X.text(root, "eventType") or "unknown",
            state=EventState((X.text(root, "eventState") or "active").lower())
            if (X.text(root, "eventState") or "active").lower() in ("active", "inactive")
            else EventState.ACTIVE,
            channel_id=int(channel) if channel and channel.isdigit() else None,
            timestamp=cls._parse_time(X.text(root, "dateTime"), now),
            received_at=now,
            active_post_count=int(apc) if apc.isdigit() else 0,
            description=X.text(root, "eventDescription") or "",
            ip_address=X.text(root, "ipAddress") or X.text(root, "ipv6Address"),
            region_ids=regions,
            target_type=X.text(root, "targetType") or next(
                (t for t in (X.text(e, "detectionTarget") for e in X.findall(root, "DetectionRegionEntry"))
                 if t), None),
            raw=X.to_dict(root),
        )

    @classmethod
    def from_json(cls, data: bytes) -> AlertEvent:
        obj = json.loads(data)
        body = obj.get("EventNotificationAlert", obj)
        now = datetime.now(timezone.utc).astimezone()
        state = str(body.get("eventState", "active")).lower()
        ch = body.get("channelID")
        return cls(
            event_type=str(body.get("eventType", "unknown")),
            state=EventState(state) if state in ("active", "inactive") else EventState.ACTIVE,
            channel_id=int(ch) if isinstance(ch, (int, str)) and str(ch).isdigit() else None,
            timestamp=cls._parse_time(body.get("dateTime"), now),
            received_at=now,
            active_post_count=int(body.get("activePostCount", 0) or 0),
            description=str(body.get("eventDescription", "")),
            ip_address=body.get("ipAddress"),
            raw=body,
        )


@dataclass(frozen=True, slots=True)
class AlertAttachment:
    """Binary part (typically a JPEG snapshot) delivered on the alert stream."""

    content_type: str
    data: bytes = field(repr=False)
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).astimezone())
    related_event: AlertEvent | None = None
    name: str | None = None            # form unit name: visibleLightImage | thermalImage | targetImage | <pId>
    content_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseStatus:
    status_code: int
    status_string: str
    sub_status_code: str
    request_url: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code in (0, 1)

    @classmethod
    def from_xml(cls, root: ET.Element) -> ResponseStatus:
        code = X.text(root, "statusCode") or "0"
        return cls(
            status_code=int(code) if code.isdigit() else -1,
            status_string=X.text(root, "statusString") or "",
            sub_status_code=X.text(root, "subStatusCode") or "",
            request_url=X.text(root, "requestURL"),
        )
