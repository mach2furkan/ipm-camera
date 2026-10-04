"""Radiometric analytics on temperature matrices.

* ``HotspotDetector``   connected regions significantly warmer than the local scene,
                        with an adaptive threshold ``median + k * MAD`` (robust to the
                        sky/ground bimodality) and an absolute floor.
* ``PersonCandidates``  warm blobs whose apparent temperature fits a human seen through
                        the atmosphere (above background, below ~40 degC), with geometry
                        gates -- detector-independent night/fog cueing.
* ``FireDetector``      absolute-temperature *or* rate-of-rise fire candidates confirmed
                        over several frames at a stable location (flame flicker breaks
                        single-frame logic; sun glint on water fails the persistence test).
* ``RoiMonitor``        per-polygon max/min/mean with pre-alarm / alarm thresholds,
                        hysteresis, dwell (``alarmFilteringTime`` semantics of the device),
                        and a least-squares temperature-rise rate over a sliding window --
                        early warning before an absolute threshold is crossed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import ndimage

F32 = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Blob:
    label: int
    bbox: tuple[int, int, int, int]        # x1, y1, x2, y2 (pixels, exclusive end)
    area: int
    max_c: float
    mean_c: float
    centroid: tuple[float, float]
    peak: tuple[int, int]                  # (x, y) of the hottest pixel
    contrast_c: float                      # mean_c - background median

    def norm_bbox(self, w: int, h: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return x1 / w, y1 / h, x2 / w, y2 / h


def robust_background(temps: F32) -> tuple[float, float]:
    v = temps[np.isfinite(temps)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, 0.05)


class HotspotDetector:
    def __init__(self, *, k_mad: float = 6.0, min_delta_c: float = 2.0, min_area: int = 4,
                 abs_floor_c: float | None = None) -> None:
        self.k = k_mad
        self.min_delta = min_delta_c
        self.min_area = min_area
        self.abs_floor = abs_floor_c

    def __call__(self, temps: F32) -> list[Blob]:
        med, mad = robust_background(temps)
        thr = med + max(self.k * mad, self.min_delta)
        if self.abs_floor is not None:     # anything above an absolute trigger is hot regardless of scene
            thr = min(thr, self.abs_floor)
        mask = np.isfinite(temps) & (temps > thr)
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        if n == 0:
            return []
        idx = np.arange(1, n + 1)
        areas = ndimage.sum_labels(np.ones_like(temps), labels, idx)
        maxima = ndimage.maximum(temps, labels, idx)
        means = ndimage.mean(temps, labels, idx)
        peaks = ndimage.maximum_position(temps, labels, idx)
        cents = ndimage.center_of_mass(mask, labels, idx)
        slices = ndimage.find_objects(labels)
        out = []
        for i, lab in enumerate(idx):
            if areas[i] < self.min_area or slices[i] is None:
                continue
            sy, sx = slices[i]
            out.append(Blob(int(lab), (sx.start, sy.start, sx.stop, sy.stop), int(areas[i]), float(maxima[i]),
                            float(means[i]), (float(cents[i][1]), float(cents[i][0])),
                            (int(peaks[i][1]), int(peaks[i][0])), float(means[i]) - med))
        return sorted(out, key=lambda b: -b.max_c)


@dataclass(frozen=True, slots=True)
class PersonCandidateConfig:
    min_contrast_c: float = 1.5
    max_apparent_c: float = 42.0
    min_area: int = 6
    min_aspect: float = 1.1            # height / width (standing); set lower for top-down views
    max_aspect: float = 6.0


class PersonCandidates:
    def __init__(self, config: PersonCandidateConfig | None = None) -> None:
        self.cfg = config or PersonCandidateConfig()
        self._det = HotspotDetector(k_mad=4.0, min_delta_c=self.cfg.min_contrast_c, min_area=self.cfg.min_area)

    def __call__(self, temps: F32) -> list[Blob]:
        c = self.cfg
        out = []
        for b in self._det(temps):
            w = b.bbox[2] - b.bbox[0]
            h = b.bbox[3] - b.bbox[1]
            aspect = h / max(w, 1)
            if b.max_c > c.max_apparent_c or b.contrast_c < c.min_contrast_c:
                continue
            if not c.min_aspect <= aspect <= c.max_aspect:
                continue
            out.append(b)
        return out


@dataclass(frozen=True, slots=True)
class FireCandidate:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    max_c: float
    frames: int
    reason: str                       # "absolute" | "rate"


class FireDetector:
    def __init__(self, *, abs_c: float = 150.0, rate_c_per_s: float = 8.0, min_rate_floor_c: float = 60.0,
                 confirm_frames: int = 3, radius_px: float = 12.0, forget_s: float = 2.0) -> None:
        self.abs_c = abs_c
        self.rate = rate_c_per_s
        self.rate_floor = min_rate_floor_c
        self.confirm = confirm_frames
        self.radius = radius_px
        self.forget = forget_s
        self._tracks: list[dict[str, object]] = []
        self._prev: tuple[float, F32] | None = None
        self._det = HotspotDetector(k_mad=8.0, min_delta_c=15.0, min_area=2)

    def __call__(self, temps: F32, t: float) -> list[FireCandidate]:
        rate_map = None
        if self._prev is not None and self._prev[1].shape == temps.shape and t > self._prev[0]:
            rate_map = (temps - self._prev[1]) / (t - self._prev[0])
        self._prev = (t, temps)
        cands: list[tuple[Blob, str]] = []
        for b in self._det(temps):
            if b.max_c >= self.abs_c:
                cands.append((b, "absolute"))
            elif rate_map is not None and b.max_c >= self.rate_floor:
                x1, y1, x2, y2 = b.bbox
                if float(np.nanmax(rate_map[y1:y2, x1:x2])) >= self.rate:
                    cands.append((b, "rate"))
        confirmed: list[FireCandidate] = []
        alive = []
        for tr in self._tracks:
            if t - float(tr["t"]) <= self.forget:  # type: ignore[arg-type]
                alive.append(tr)
        self._tracks = alive
        for b, reason in cands:
            match = None
            for tr in self._tracks:
                cx, cy = tr["c"]  # type: ignore[misc]
                if np.hypot(cx - b.centroid[0], cy - b.centroid[1]) <= self.radius:
                    match = tr
                    break
            if match is None:
                match = {"c": b.centroid, "n": 0, "t": t}
                self._tracks.append(match)
            match["n"] = int(match["n"]) + 1  # type: ignore[call-overload]
            match["c"], match["t"] = b.centroid, t
            if int(match["n"]) >= self.confirm:  # type: ignore[call-overload]
                confirmed.append(FireCandidate(b.centroid, b.bbox, b.max_c, int(match["n"]), reason))  # type: ignore[call-overload]
        return confirmed


@dataclass(frozen=True, slots=True)
class RoiAlarm:
    roi: str
    level: str                        # prealarm | alarm | rise | clear
    value_c: float
    rate_c_per_min: float | None
    t: float


@dataclass
class _RoiState:
    level: str = "normal"
    candidate: str | None = None
    since: float = 0.0
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=4096))
    rise_active: bool = False


class RoiMonitor:
    """Polygon temperature rules with device-like semantics plus rate-of-rise detection."""

    def __init__(self, name: str, polygon_norm: npt.ArrayLike, *, stat: str = "max", prealarm_c: float = 60.0,
                 alarm_c: float = 80.0, hysteresis_c: float = 2.0, dwell_s: float = 2.0,
                 rise_c_per_min: float | None = 3.0, rise_window_s: float = 60.0) -> None:
        if stat not in ("max", "min", "mean"):
            raise ValueError("stat must be max | min | mean")
        self.name = name
        self.poly = np.asarray(polygon_norm, dtype=np.float64)
        self.stat = stat
        self.pre = prealarm_c
        self.alarm = alarm_c
        self.hyst = hysteresis_c
        self.dwell = dwell_s
        self.rise = rise_c_per_min
        self.window = rise_window_s
        self._mask: npt.NDArray[np.bool_] | None = None
        self._shape: tuple[int, int] | None = None
        self.state = _RoiState()

    def _mask_for(self, shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
        if self._mask is None or self._shape != shape:
            from ..analytics.geometry import points_in_polygon

            h, w = shape
            yy, xx = np.mgrid[0:h, 0:w]
            pts = np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h], axis=1)
            self._mask = points_in_polygon(pts, self.poly).reshape(h, w)
            self._shape = shape
        return self._mask

    def value(self, temps: F32) -> float:
        v = temps[self._mask_for(temps.shape) & np.isfinite(temps)]
        if v.size == 0:
            return float("nan")
        return float(v.max() if self.stat == "max" else v.min() if self.stat == "min" else v.mean())

    def rate_c_per_min(self) -> float | None:
        h = self.state.history
        if len(h) < 5 or h[-1][0] - h[0][0] < self.window * 0.3:
            return None
        t = np.array([p[0] for p in h])
        v = np.array([p[1] for p in h])
        tm = t - t.mean()
        denom = float((tm ** 2).sum())
        return float((tm * (v - v.mean())).sum() / denom) * 60.0 if denom > 0 else None

    def update(self, temps: F32, t: float) -> list[RoiAlarm]:
        val = self.value(temps)
        if not np.isfinite(val):
            return []
        st = self.state
        st.history.append((t, val))
        while st.history and t - st.history[0][0] > self.window:
            st.history.popleft()
        out: list[RoiAlarm] = []
        rate = self.rate_c_per_min()

        # Target level with hysteresis on the way down.
        if val >= self.alarm or (st.level == "alarm" and val >= self.alarm - self.hyst):
            target = "alarm"
        elif val >= self.pre or (st.level in ("prealarm", "alarm") and val >= self.pre - self.hyst):
            target = "prealarm"
        else:
            target = "normal"
        if target != st.level:
            if st.candidate != target:
                st.candidate, st.since = target, t
            if t - st.since >= self.dwell or (target == "normal"):
                st.level, st.candidate = target, None
                out.append(RoiAlarm(self.name, target if target != "normal" else "clear", val, rate, t))
        else:
            st.candidate = None

        if self.rise is not None and rate is not None:
            if rate >= self.rise and not st.rise_active:
                st.rise_active = True
                out.append(RoiAlarm(self.name, "rise", val, rate, t))
            elif rate < self.rise * 0.5:
                st.rise_active = False
        return out
