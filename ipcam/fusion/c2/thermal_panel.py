"""Server-side state for the operator's thermal view: rendering cache, statistics, hotspots.

The browser receives (a) a false-colour PNG rendered once per frame and palette, and (b) a
block-max temperature grid so the cursor read-out shows real radiometric values without
streaming the full float matrix. Fire candidates and ROI alarms are evaluated here on
every new frame so they reach the event log even when no operator is watching.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ...thermal.radiometry import FireDetector, HotspotDetector, RoiAlarm, RoiMonitor
from ...thermal.render import PALETTES, PlateauAGC, colorize, downsample_grid, encode_png
from ...thermal.stream import ThermalFrame


class ThermalPanel:
    def __init__(self, source: Callable[[], ThermalFrame | None], *, label: str = "thermal",
                 rois: list[RoiMonitor] | None = None, on_alarm: Callable[[str, str], None] | None = None,
                 status: Callable[[], dict[str, Any]] | None = None) -> None:
        self.source = source
        self.label = label
        self.rois = rois or []
        self.on_alarm = on_alarm
        self.status = status
        self._agc = {p: PlateauAGC() for p in PALETTES}
        self._png: dict[str, tuple[int, bytes]] = {}
        self._hot = HotspotDetector(k_mad=6.0, min_delta_c=4.0)
        self._fire = FireDetector()
        self._last_id = -1
        self._state: dict[str, Any] = {"available": False}
        self.roi_events: list[RoiAlarm] = []

    def _analyse(self, f: ThermalFrame) -> None:
        if f.frame_id == self._last_id:
            return
        self._last_id = f.frame_id
        t = f.temps
        fires = self._fire(t, f.t)
        for fc in fires:
            if fc.frames == self._fire.confirm and self.on_alarm:
                self.on_alarm("fire", f"{self.label}: yangın adayı {fc.max_c:.0f} °C ({fc.reason})")
        for roi in self.rois:
            for ev in roi.update(t, f.t):
                self.roi_events.append(ev)
                if self.on_alarm:
                    names = {"prealarm": "ön alarm", "alarm": "alarm", "rise": "hızlı ısınma", "clear": "normal"}
                    rate = f", {ev.rate_c_per_min:+.1f} °C/dk" if ev.rate_c_per_min is not None else ""
                    self.on_alarm(f"roi_{ev.level}", f"{self.label}/{ev.roi}: {names[ev.level]} {ev.value_c:.1f} °C{rate}")
        grid, k = downsample_grid(t)
        h, w = t.shape
        hot = self._hot(t)[:8]
        self._state = {
            "available": True, "frame": f.frame_id, "age_ms": round((time.monotonic() - f.t) * 1000),
            "width": w, "height": h, "block": k,
            "min": round(float(np.nanmin(t)), 2), "max": round(float(np.nanmax(t)), 2),
            "mean": round(float(np.nanmean(t)), 2),
            "grid_w": int(grid.shape[1]), "grid_h": int(grid.shape[0]),
            "grid": np.round(grid, 1).ravel().tolist(),
            "hotspots": [{"x": b.peak[0] / w, "y": b.peak[1] / h, "max": round(b.max_c, 1), "area": b.area,
                          "bbox": [round(v, 4) for v in b.norm_bbox(w, h)]} for b in hot],
            "fires": [{"x": c.centroid[0] / w, "y": c.centroid[1] / h, "max": round(c.max_c, 1)} for c in fires],
            "rois": [{"name": r.name, "poly": r.poly.round(4).tolist(), "level": r.state.level,
                      "rate": r.rate_c_per_min()} for r in self.rois],
        }

    def ingest(self, frame: ThermalFrame) -> None:
        """Analyse every frame as it arrives (wire to ``ThermalStreamReader.on_frame``).

        Without this, fire and ROI rules would only run while a browser polls the panel --
        an unattended console must still raise the alarm.
        """
        self._analyse(frame)

    def state(self) -> dict[str, Any]:
        f = self.source()
        if f is not None:
            self._analyse(f)
        st = dict(self._state)
        if self.status is not None:
            st["link"] = self.status()
        return st

    def png(self, palette: str = "iron") -> bytes | None:
        if palette not in PALETTES:
            palette = "iron"
        f = self.source()
        if f is None:
            return None
        cached = self._png.get(palette)
        if cached is not None and cached[0] == f.frame_id:
            return cached[1]
        self._analyse(f)
        data = encode_png(colorize(self._agc[palette](f.temps), palette), level=3)
        self._png[palette] = (f.frame_id, data)
        return data
