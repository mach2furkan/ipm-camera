"""Spatial rule engine: directed tripwires, polygon intrusion and loitering.

All rules reason about the track's *foot point* (bottom-centre of the box): it is the
point that actually stands on the ground plane the operator drew the zone on. The box
centre of a person close to a fence would otherwise "enter" a zone while standing outside.

Rules are stateful per track. Per-track state is released through :meth:`forget` when
the tracker removes the track, so memory stays bounded over weeks of uptime.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .geometry import Point, points_in_polygon, segments_intersect, signed_distance, spread_radius

F64 = npt.NDArray[np.float64]


class EventKind(str, Enum):
    TRIPWIRE = "tripwire"
    INTRUSION = "intrusion"        # confirmed presence inside a zone (after min dwell)
    ZONE_EXIT = "zone_exit"        # end of an intrusion
    LOITERING = "loitering"


@dataclass(frozen=True, slots=True)
class TrackView:
    """Immutable per-frame snapshot of a track handed to the rules."""

    track_id: int
    t: float
    foot: Point
    box: tuple[float, float, float, float]
    velocity: tuple[float, float]
    height: float
    conf: float
    observed: bool          # False while occluded (Kalman extrapolation only)
    valid: bool             # passed heuristic false-alarm suppression
    reasons: tuple[str, ...] = ()
    maturing: bool = False  # invalid only because the track is still young


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    kind: EventKind
    rule: str
    track_id: int
    t: float
    position: Point
    box: tuple[float, float, float, float]
    direction: str | None = None
    dwell_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        extra = f" dir={self.direction}" if self.direction else ""
        dwell = f" dwell={self.dwell_s:.1f}s" if self.dwell_s is not None else ""
        return f"[{self.kind.value}] rule={self.rule} track={self.track_id}{extra}{dwell}"


class Rule(Protocol):
    name: str

    def evaluate(self, tracks: list[TrackView], t: float) -> list[SecurityEvent]: ...

    def forget(self, track_ids: list[int], t: float) -> list[SecurityEvent]: ...


# --------------------------------------------------------------------------- tripwire

@dataclass(slots=True)
class _WireState:
    side: int
    anchor: Point
    last_event_t: float = -math.inf
    pending: tuple[str, float] | None = None   # (direction, crossed_at) awaiting track validation


class Tripwire:
    """Directed virtual fence A->B.

    Crossing logic: a track's side of the line only changes once its foot point is more
    than a hysteresis band away from the line (``max(min_band_px, band_rel * height)``),
    so someone standing on the line cannot generate a burst of crossings. When the side
    flips, the movement from the last position on the old side (the *anchor*) to the
    current position must intersect the **segment** AB; walking around the end of the
    wire is not a crossing.

    Direction: give ``inside`` -- any point on the protected side. ``alarm_on="in"`` then
    fires only for movements towards that side. Without ``inside`` the side where
    ``side(A, B, P) < 0`` is treated as inside.
    """

    def __init__(
        self,
        name: str,
        a: Point,
        b: Point,
        *,
        inside: Point | None = None,
        alarm_on: str = "in",
        band_rel: float = 0.05,
        min_band_px: float = 2.0,
        cooldown_s: float = 2.0,
        pending_s: float = 2.0,
    ) -> None:
        if alarm_on not in ("in", "out", "both"):
            raise ValueError("alarm_on must be 'in', 'out' or 'both'")
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1.0:
            raise ValueError(f"tripwire {name!r} is degenerate")
        self.name = name
        self.a, self.b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
        if inside is not None:
            d = signed_distance(self.a, self.b, inside)
            if d == 0:
                raise ValueError("tripwire 'inside' reference point lies on the line")
            self._inside_sign = 1 if d > 0 else -1
        else:
            self._inside_sign = -1
        self.alarm_on = alarm_on
        self._band_rel = band_rel
        self._min_band = min_band_px
        self._cooldown = cooldown_s
        self._pending_s = pending_s
        self._state: dict[int, _WireState] = {}

    def evaluate(self, tracks: list[TrackView], t: float) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for tv in tracks:
            if not tv.observed:
                continue  # never decide a crossing from an extrapolated position
            st0 = self._state.get(tv.track_id)
            if st0 is not None and st0.pending is not None:
                # A crossing made while the track was still maturing is reported once the
                # track proves to be a person, or dropped if it is rejected / too late.
                direction, crossed_at = st0.pending
                if tv.valid and t - crossed_at <= self._pending_s:
                    st0.pending = None
                    events += self._emit(st0, tv, t, direction, crossed_at)
                elif not tv.maturing or t - crossed_at > self._pending_s:
                    st0.pending = None
            d = signed_distance(self.a, self.b, tv.foot)
            if abs(d) < max(self._min_band, self._band_rel * tv.height):
                continue
            s = 1 if d > 0 else -1
            st = self._state.get(tv.track_id)
            if st is None:
                self._state[tv.track_id] = _WireState(s, tv.foot)
                continue
            if s == st.side:
                st.anchor = tv.foot
                continue
            crossed = segments_intersect(st.anchor, tv.foot, self.a, self.b)
            st.side, st.anchor = s, tv.foot
            if not crossed:
                continue
            direction = "in" if s == self._inside_sign else "out"
            if self.alarm_on not in (direction, "both"):
                continue
            if tv.valid:
                events += self._emit(st, tv, t, direction, t)
            elif tv.maturing:
                st.pending = (direction, t)
        return events

    def _emit(self, st: _WireState, tv: TrackView, t: float, direction: str, crossed_at: float
              ) -> list[SecurityEvent]:
        if t - st.last_event_t < self._cooldown:
            return []
        st.last_event_t = t
        return [SecurityEvent(EventKind.TRIPWIRE, self.name, tv.track_id, t, tv.foot, tv.box,
                              direction=direction, details={"crossed_at": crossed_at})]

    def forget(self, track_ids: list[int], t: float) -> list[SecurityEvent]:
        for tid in track_ids:
            self._state.pop(tid, None)
        return []


# --------------------------------------------------------------------------- polygon zone

@dataclass(slots=True)
class _ZoneState:
    inside: bool = False
    t_enter: float = 0.0
    t_last_inside: float = -math.inf
    intrusion_fired: bool = False
    loiter_fired: bool = False
    points: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=2048))


class PolygonZone:
    """Restricted area with intrusion and loitering detection (any simple polygon).

    intrusion   foot point inside for at least ``min_dwell_s`` (suppresses single-frame
                jitter across the border) -> INTRUSION once; leaving for longer than
                ``exit_grace_s`` -> ZONE_EXIT and re-arm.
    loitering   inside for at least ``loiter_s`` while the foot positions of that window
                stay within ``loiter_radius_px`` of their centroid -> LOITERING once per stay.

    While a track is occluded (lost) its zone state is frozen instead of reset, so a
    person passing behind a pillar keeps accumulating dwell time.
    """

    def __init__(
        self,
        name: str,
        polygon: F64,
        *,
        intrusion: bool = True,
        min_dwell_s: float = 0.5,
        exit_grace_s: float = 1.0,
        loiter_s: float | None = 15.0,
        loiter_radius_px: float = 30.0,
    ) -> None:
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
            raise ValueError(f"zone {name!r} needs at least 3 vertices")
        self.name = name
        self.polygon = poly
        self._bbox = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
        self.intrusion = intrusion
        self.min_dwell = min_dwell_s
        self.exit_grace = exit_grace_s
        self.loiter_s = loiter_s
        self.loiter_radius = loiter_radius_px
        self._state: dict[int, _ZoneState] = {}

    def contains(self, points: F64) -> npt.NDArray[np.bool_]:
        pts = np.atleast_2d(points)
        x0, y0, x1, y1 = self._bbox
        inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if inside.any():
            idx = np.flatnonzero(inside)
            inside[idx] = points_in_polygon(pts[idx], self.polygon)
        return inside

    def evaluate(self, tracks: list[TrackView], t: float) -> list[SecurityEvent]:
        observed = [tv for tv in tracks if tv.observed]
        if not observed:
            return []
        flags = self.contains(np.array([tv.foot for tv in observed]))
        events: list[SecurityEvent] = []
        for tv, inside in zip(observed, flags):
            st = self._state.get(tv.track_id)
            if st is None:
                if not inside:
                    continue  # state is only allocated for tracks that ever enter
                st = self._state[tv.track_id] = _ZoneState()
            if inside:
                if not st.inside:
                    if t - st.t_last_inside > self.exit_grace:
                        st.t_enter = t
                        st.points.clear()
                    st.inside = True
                st.t_last_inside = t
                st.points.append((t, tv.foot[0], tv.foot[1]))
                dwell = t - st.t_enter
                if self.intrusion and not st.intrusion_fired and dwell >= self.min_dwell and tv.valid:
                    st.intrusion_fired = True
                    events.append(SecurityEvent(EventKind.INTRUSION, self.name, tv.track_id, t, tv.foot, tv.box,
                                                dwell_s=dwell))
                if self.loiter_s is not None and not st.loiter_fired and dwell >= self.loiter_s and tv.valid:
                    window = np.array([(x, y) for (pt, x, y) in st.points if pt >= t - self.loiter_s])
                    radius = spread_radius(window)
                    if radius <= self.loiter_radius:
                        st.loiter_fired = True
                        events.append(SecurityEvent(EventKind.LOITERING, self.name, tv.track_id, t, tv.foot, tv.box,
                                                    dwell_s=dwell, details={"radius_px": round(radius, 1)}))
            elif st.inside and t - st.t_last_inside > self.exit_grace:
                events += self._exit(tv.track_id, st, t, tv.foot, tv.box)
            elif not st.inside and not st.intrusion_fired and t - st.t_last_inside > self.exit_grace:
                self._state.pop(tv.track_id, None)
        return events

    def _exit(self, tid: int, st: _ZoneState, t: float, pos: Point,
              box: tuple[float, float, float, float]) -> list[SecurityEvent]:
        out: list[SecurityEvent] = []
        if st.intrusion_fired:
            out.append(SecurityEvent(EventKind.ZONE_EXIT, self.name, tid, t, pos, box,
                                     dwell_s=st.t_last_inside - st.t_enter))
        self._state.pop(tid, None)
        return out

    def forget(self, track_ids: list[int], t: float) -> list[SecurityEvent]:
        out: list[SecurityEvent] = []
        for tid in track_ids:
            st = self._state.get(tid)
            if st is None:
                continue
            last = st.points[-1] if st.points else (t, math.nan, math.nan)
            out += self._exit(tid, st, t, (last[1], last[2]), (math.nan,) * 4)  # type: ignore[arg-type]
        return out

    def occupants(self) -> list[int]:
        return [tid for tid, st in self._state.items() if st.inside]
