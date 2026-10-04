"""AnalyticsEngine: detections -> tracks -> heuristic validation -> spatial events."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..metrics import RollingWindow
from .bytetrack import ByteTracker, Track, TrackerOutput
from .heuristics import HeuristicFilter
from .rules import Rule, SecurityEvent, TrackView

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    t: float
    tracks: list[TrackView]          # observed (matched this frame)
    occluded: list[TrackView]        # confirmed but currently lost
    events: list[SecurityEvent]
    removed_ids: list[int]
    process_ms: float


class AnalyticsEngine:
    """Stateful per-camera analytics. Not thread-safe: call from the inference thread.

    Usage::

        engine = AnalyticsEngine(rules=build_rules(cfg, w, h))
        res = engine.update(detections, t=frame.arrival_ns / 1e9)
        for ev in res.events: dispatcher.dispatch(ev)
    """

    def __init__(
        self,
        *,
        tracker: ByteTracker | None = None,
        rules: Sequence[Rule] = (),
        heuristics: HeuristicFilter | None = None,
    ) -> None:
        self.tracker = tracker or ByteTracker()
        self.rules = list(rules)
        self.heuristics = heuristics or HeuristicFilter()
        self.timing = RollingWindow(1024)

    def _view(self, tr: Track, t: float, observed: bool) -> TrackView:
        verdict = self.heuristics.assess(tr)
        b = tr.box
        return TrackView(
            track_id=tr.track_id, t=t, foot=tr.foot, box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            velocity=tr.velocity, height=tr.height, conf=tr.conf, observed=observed,
            valid=verdict.valid, reasons=verdict.reasons, maturing=verdict.maturing,
        )

    def update(self, detections: F64, t: float) -> AnalyticsResult:
        t0 = time.perf_counter()
        out: TrackerOutput = self.tracker.update(detections, t)
        views = [self._view(tr, t, True) for tr in out.tracks]
        occluded = [self._view(tr, t, False) for tr in out.lost]
        everything = views + occluded
        events: list[SecurityEvent] = []
        for rule in self.rules:
            events += rule.evaluate(everything, t)
        if out.removed_ids:
            for rule in self.rules:
                events += rule.forget(out.removed_ids, t)
        ms = (time.perf_counter() - t0) * 1000.0
        self.timing.add(ms)
        return AnalyticsResult(t, views, occluded, events, out.removed_ids, ms)
