"""Bridge: thermal PT camera events -> geolocated observations for the global tracker.

A PT camera's own perimeter detection does not hand out track ids, but its events carry
head pose, target box and range. Each event is geolocated (:mod:`geo`) and associated to a
short-lived *pseudo local track* per camera (nearest within a gate in metres and seconds),
so a person producing repeated ``fielddetection`` posts becomes one local track that the
Global Track Manager fuses with fixed-camera tracks and hands over between views.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..isapi.events_pt import PTEventDetail, pt_details
from ..isapi.models import AlertEvent, EventState
from .geo import GeoObservation, LensModel, PTGeolocator
from .global_tracker import LocalObservation


@dataclass
class _Pseudo:
    lid: int
    t: float
    e: float
    n: float


class PTEventGeoBridge:
    def __init__(self, camera_id: str, locator: PTGeolocator, *, visible_lens: LensModel | None = None,
                 thermal_lens: LensModel | None = None, gate_m: float = 8.0, gate_s: float = 4.0,
                 epoch_offset_s: float = 0.0) -> None:
        self.camera_id = camera_id
        self.loc = locator
        self.visible_lens = visible_lens
        self.thermal_lens = thermal_lens
        self.gate_m = gate_m
        self.gate_s = gate_s
        self.offset = epoch_offset_s          # device clock - host clock (see measure_clock_offset)
        self._ids = itertools.count(1)
        self._live: list[_Pseudo] = []
        self.unlocatable = 0

    def convert(self, ev: AlertEvent, detail: PTEventDetail | None = None
                ) -> tuple[LocalObservation, GeoObservation, list[int]] | None:
        """Returns (observation, geolocation, ended pseudo-track ids) or None."""
        d = detail or pt_details(ev)
        if d.category != "perimeter" or ev.state is EventState.INACTIVE:
            return None
        pose = d.visible_pose or d.thermal_pose
        if pose is None:
            self.unlocatable += 1
            return None
        lens = self.visible_lens if d.visible_pose is not None else self.thermal_lens
        try:
            geo = self.loc.locate(pose.azimuth, pose.elevation, range_m=d.distance_m, rect=d.target_rect,
                                  lens=lens, zoom=pose.zoom)
        except ValueError:
            self.unlocatable += 1
            return None
        t = ev.timestamp.timestamp() - self.offset
        ended = [p.lid for p in self._live if t - p.t > self.gate_s]
        self._live = [p for p in self._live if t - p.t <= self.gate_s]
        e, n = float(geo.enu[0]), float(geo.enu[1])
        best = min(self._live, key=lambda p: math.hypot(p.e - e, p.n - n), default=None)
        gate = max(self.gate_m, 3 * math.sqrt(float(np.trace(geo.cov_en))))
        if best is None or math.hypot(best.e - e, best.n - n) > gate:
            best = _Pseudo(next(self._ids), t, e, n)
            self._live.append(best)
        best.t, best.e, best.n = t, e, n
        obs = LocalObservation(self.camera_id, best.lid, t, np.array([e, n]), geo.cov_en,
                               box=tuple(d.target_rect) if d.target_rect else (0.0, 0.0, 0.0, 0.0),  # type: ignore[arg-type]
                               conf=1.0)
        return obs, geo, ended
