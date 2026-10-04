"""Overlay rendering (tracks, trajectories, zones, alarms) for operator view / re-streaming."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .engine import AnalyticsResult
from .rules import PolygonZone, Rule, Tripwire


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    c = rng.integers(64, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def draw(image: Any, result: AnalyticsResult, rules: Sequence[Rule], *,
         trails: dict[int, list[tuple[float, float]]] | None = None) -> Any:
    """Draw in place on a BGR uint8 image and return it."""
    import cv2

    for rule in rules:
        if isinstance(rule, PolygonZone):
            pts = rule.polygon.astype(np.int32).reshape(-1, 1, 2)
            active = bool(rule.occupants())
            cv2.polylines(image, [pts], True, (0, 0, 255) if active else (0, 200, 255), 2, cv2.LINE_AA)
            x, y = rule.polygon[0]
            cv2.putText(image, rule.name, (int(x) + 4, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
        elif isinstance(rule, Tripwire):
            a = tuple(int(v) for v in rule.a)
            b = tuple(int(v) for v in rule.b)
            cv2.arrowedLine(image, a, b, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.03)
            cv2.putText(image, rule.name, (a[0] + 4, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1, cv2.LINE_AA)

    alarmed = {ev.track_id for ev in result.events}
    for tv in result.tracks + result.occluded:
        color = (0, 0, 255) if tv.track_id in alarmed else _color(tv.track_id)
        x1, y1, x2, y2 = (int(v) for v in tv.box)
        thickness = 2 if tv.observed else 1
        if not tv.valid:
            color = (128, 128, 128)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = f"#{tv.track_id} {tv.conf:.2f}" + ("" if tv.observed else " occl")
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.circle(image, (int(tv.foot[0]), int(tv.foot[1])), 3, color, -1, cv2.LINE_AA)
        if trails is not None:
            trail = trails.setdefault(tv.track_id, [])
            if tv.observed:
                trail.append(tv.foot)
                del trail[:-40]
            if len(trail) > 1:
                cv2.polylines(image, [np.array(trail, np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    if trails is not None:
        for tid in result.removed_ids:
            trails.pop(tid, None)
    return image
