"""Independent vehicle vocabulary and directed line counts using existing ByteTrack/Tripwire.

Inspired by HodenX/python-traffic-counter-with-yolo-and-sort; no source copied.
The desk model's vocabulary and thresholds are deliberately independent.
"""
from __future__ import annotations

import numpy as np
from ipcam.analytics.rules import TrackView, Tripwire

VEHICLE_PROMPTS = ("car", "truck", "bus", "motorcycle", "bicycle")
VEHICLE_LABELS = ("otomobil", "kamyon", "otobüs", "motosiklet", "bisiklet")


def vehicle_detections(boxes, confidences, classes, width, height, floor=.30):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    confidences, classes = np.asarray(confidences), np.asarray(classes)
    good = (np.isfinite(boxes).all(axis=1) & np.isfinite(confidences) &
            np.isfinite(classes) & (classes == np.floor(classes)) &
            (classes >= 0) & (classes < len(VEHICLE_PROMPTS)) &
            (confidences >= floor) & (confidences <= 1))
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    good &= (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    return np.column_stack((boxes[good], confidences[good], classes[good]))


class TrafficCounter:
    """Count a confirmed track at most once; retire state when its tracker retires it.

    A fragmented track can still cause counting errors; this is not identity re-identification.
    Reconnects clear spatial history while retaining totals.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.counts = np.zeros((len(VEHICLE_PROMPTS), 2), dtype=int)
        self.reset_tracks()

    def reset_tracks(self):
        self.wire = Tripwire("traffic", self.a, self.b, alarm_on="both", cooldown_s=0,
                             band_rel=.02, min_band_px=3)
        self.counted = set()

    def reset_counts(self):
        self.counts.fill(0)
        self.reset_tracks()

    def update(self, tracks, removed_ids, t):
        views = []
        classes = {}
        for tr in tracks:
            classes[tr.track_id] = tr.cls
            box = tuple(float(v) for v in tr.box)
            views.append(TrackView(tr.track_id, t, ((box[0]+box[2])/2, box[3]),
                                   box, (0., 0.), box[3]-box[1], tr.conf, True, True))
        for event in self.wire.evaluate(views, t):
            if event.track_id not in self.counted:
                self.counts[classes[event.track_id], 0 if event.direction == "in" else 1] += 1
                self.counted.add(event.track_id)
        self.wire.forget(removed_ids, t)
        self.counted.difference_update(removed_ids)


class TrafficOverlay:
    """Line editing with two clicks after L; count reset R."""
    def __init__(self):
        self.line = ((.1, .5), (.9, .5))
        self.points = None
        self.counter = None
        self.shape = None

    def click(self, event, x, y, flags, param):
        import cv2
        if event != cv2.EVENT_LBUTTONDOWN or self.points is None or self.shape is None:
            return
        h, w = self.shape
        self.points.append((min(1., max(0., x/w)), min(1., max(0., y/h))))
        if len(self.points) == 2:
            a, b = self.points
            if np.hypot((a[0]-b[0])*w, (a[1]-b[1])*h) >= 10:
                self.line = (a, b)
                self.counter = None  # new line starts new totals
                self.points = None
            else:
                self.points.clear()

    def update(self, output, t, height, width):
        if self.counter is None or self.shape != (height, width):
            a, b = [tuple(int(v*s) for v, s in zip(p, (width, height))) for p in self.line]
            totals = self.counter.counts.copy() if self.counter is not None else None
            self.counter = TrafficCounter(a, b)
            if totals is not None:
                self.counter.counts[:] = totals
            self.shape = (height, width)
        if self.points is None:
            self.counter.update(output.tracks, output.removed_ids, t)

    def draw(self, frame, painter):
        import cv2
        if self.counter is None:
            return
        c = self.counter
        cv2.arrowedLine(frame, c.a, c.b, (0, 230, 255), 3, tipLength=.03)
        dx, dy = c.b[0]-c.a[0], c.b[1]-c.a[1]
        length = np.hypot(dx, dy)
        mx, my = (c.a[0]+c.b[0])/2, (c.a[1]+c.b[1])/2
        painter.put(frame, "A", int(mx+dy/length*35), int(my-dx/length*35), (32, 80, 80))
        painter.put(frame, "B", int(mx-dy/length*35), int(my+dx/length*35), (32, 80, 80))
        if self.points is not None:
            text = "Yeni sayım çizgisi için iki noktaya tıkla (yeni sayaç)."
        else:
            text = "Trafik A / B: " + " | ".join(
                f"{name} {int(row[0])}/{int(row[1])}" for name, row in zip(VEHICLE_LABELS, c.counts))
        painter.put(frame, text, 0, 38, (32, 48, 48))
        painter.put(frame, "L: çizgi seç   R: sayacı sıfırla | A/B: işaretli tarafa doğru geçiş", 0, 76, (32, 48, 48))
