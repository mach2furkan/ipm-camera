"""Rich event details of the thermal PT series (perimeter, temperature, fire).

The generic :class:`AlertEvent` keeps the device payload in ``raw``; this module extracts
the fields that make PT-camera events geolocatable and actionable:

* ``TargetRect`` (normalised) and ``detectionTarget`` (human / vehicle / others),
* ``targetDistance`` / ``laserRanging`` (metres) and ``targetSpeed``,
* ``visibleLightAbsoluteHigh`` / ``thermalAbsoluteHigh`` -- the head pose *at the event*,
* ``deviceLocation`` (GNSS, degrees-minutes-seconds),
* temperature-alarm fields (TMA / TMPA / TDA): rule type, thresholds, current values,
  and the PTZ coordinates / preset the measurement was taken at.

Field names follow the PT-series ISAPI document; parsing is tolerant to the XML vs JSON
and ``ver20`` vs ``isapi.org`` variants because ``raw`` is namespace-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AlertEvent

PERIMETER_EVENTS = frozenset({"fielddetection", "linedetection", "regionentrance", "regionexiting", "loitering"})
TEMPERATURE_EVENTS = frozenset({"tma", "tmpa", "tda", "temperatureevent", "temperatureup", "temperaturedown"})
FIRE_EVENTS = frozenset({"firedetection", "smokedetection", "smokeandfiredetection", "firesmartfiredetect"})


@dataclass(frozen=True, slots=True)
class HeadPose:
    azimuth: float
    elevation: float
    zoom: float | None = None
    focus: int | None = None


@dataclass(frozen=True, slots=True)
class GeoFix:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class TemperatureReading:
    unit: str | None
    alarm_type: str | None          # MaxTemperature | MinTemperature | AverageTemperature
    rule: str | None                # greater | less
    threshold: float | None
    current: float | None
    calib_type: str | None          # point | line | region
    preset: int | None


@dataclass(frozen=True, slots=True)
class PTEventDetail:
    category: str                   # perimeter | temperature | fire | other
    target_type: str | None = None
    target_rect: tuple[float, float, float, float] | None = None    # x, y, w, h (0..1)
    region_polygon: tuple[tuple[float, float], ...] = ()             # 0..1
    distance_m: float | None = None                                  # laser > reported target distance
    laser_range_m: float | None = None
    target_distance_m: float | None = None
    target_speed: float | None = None
    visible_pose: HeadPose | None = None
    thermal_pose: HeadPose | None = None
    device_location: GeoFix | None = None
    temperature: TemperatureReading | None = None
    image_refs: dict[str, str] = field(default_factory=dict)        # visibleLightURL / thermalURL / pId ...

    @property
    def pose(self) -> HeadPose | None:
        return self.visible_pose or self.thermal_pose


def _num(v: Any) -> float | None:
    if isinstance(v, dict):
        return None
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _first(d: Any, *keys: str) -> Any:
    """Breadth-first search for the first key present anywhere in a nested dict/list."""
    queue = [d]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for k in keys:
                if k in cur:
                    return cur[k]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur)
    return None


def _pose(obj: Any, *, tenths: bool = False) -> HeadPose | None:
    if not isinstance(obj, dict):
        return None
    az, el = _num(obj.get("azimuth")), _num(obj.get("elevation"))
    if az is None or el is None:
        return None
    zoom = _num(obj.get("absoluteZoom"))
    focus = _num(obj.get("focus"))
    if tenths:  # AbsoluteHigh inside alarms is in 0.1 degree units (range [0, 3600])
        az, el = az / 10.0, el / 10.0
        zoom = zoom / 10.0 if zoom is not None else None
    return HeadPose(az, el, zoom, int(focus) if focus is not None else None)


def _dms(obj: Any, hemi: str | None, neg: str) -> float | None:
    if not isinstance(obj, dict):
        return None
    d, m, s = (_num(obj.get(k)) or 0.0 for k in ("degree", "minute", "sec"))
    v = d + m / 60.0 + s / 3600.0
    return -v if (hemi or "").upper() == neg else v


def pt_details(ev: AlertEvent) -> PTEventDetail:
    raw = ev.raw
    et = ev.event_type.lower()
    category = ("perimeter" if et in PERIMETER_EVENTS else "temperature" if et in TEMPERATURE_EVENTS
                else "fire" if et in FIRE_EVENTS else "other")

    rect = None
    tr = _first(raw, "TargetRect")
    if isinstance(tr, dict):
        x, y, w, h = (_num(tr.get(k)) for k in ("X", "Y", "width", "height"))
        if None not in (x, y, w, h):
            rect = (x, y, w, h)  # type: ignore[assignment]

    polygon: tuple[tuple[float, float], ...] = ()
    coords = _first(raw, "RegionCoordinatesList")
    if coords is not None:
        items = coords.get("RegionCoordinates", coords) if isinstance(coords, dict) else coords
        items = items if isinstance(items, list) else [items]
        pts = []
        for it in items:
            it = it.get("RegionCoordinates", it) if isinstance(it, dict) else it
            px, py = _num(it.get("positionX")), _num(it.get("positionY"))
            if px is not None and py is not None:
                scale = 1000.0 if max(px, py) > 1.0 else 1.0     # perimeter: 0..1000, TDA: 0..1
                pts.append((px / scale, py / scale))
        polygon = tuple(pts)

    laser = _num(_first(raw, "laserRanging"))
    tdist = _num(_first(raw, "targetDistance"))
    loc_obj = _first(raw, "deviceLocation")
    loc = None
    if isinstance(loc_obj, dict):
        lat = _dms(loc_obj.get("latitude"), loc_obj.get("latitudeType"), "S")
        lon = _dms(loc_obj.get("longitude"), loc_obj.get("longitudeType"), "W")
        if lat is not None and lon is not None:
            loc = GeoFix(lat, lon)

    temperature = None
    if category == "temperature":
        t = _first(raw, ev.event_type, "TDA", "TMA", "TMPA") or raw
        if isinstance(t, dict):
            temperature = TemperatureReading(
                unit=t.get("thermometryUnit"), alarm_type=t.get("alarmType"), rule=t.get("alarmRule"),
                threshold=_num(t.get("ruleTemperature", t.get("ruleTemperatureDiff", t.get("alarmTemperature")))),
                current=_num(t.get("currTemperature", t.get("currTemperatureDiff", t.get("maxTemperature")))),
                calib_type=t.get("ruleCalibType"),
                preset=int(_num(t.get("presetNo")) or 0) or None)
    ah = _first(raw, "AbsoluteHigh")
    refs = {k: str(v) for k in ("visibleLightURL", "thermalURL", "thermalInfoURL", "bkgUrl", "pId")
            if isinstance(v := _first(raw, k), (str, int)) and str(v) not in ("", "test")}
    return PTEventDetail(
        category=category,
        target_type=_first(raw, "detectionTarget", "targetType"),
        target_rect=rect, region_polygon=polygon,
        distance_m=laser if laser else tdist, laser_range_m=laser, target_distance_m=tdist,
        target_speed=_num(_first(raw, "targetSpeed")),
        visible_pose=_pose(_first(raw, "visibleLightAbsoluteHigh")) or (_pose(ah, tenths=True) if ah else None),
        thermal_pose=_pose(_first(raw, "thermalAbsoluteHigh")),
        device_location=loc, temperature=temperature, image_refs=refs,
    )
