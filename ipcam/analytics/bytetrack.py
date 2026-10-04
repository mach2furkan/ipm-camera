"""ByteTrack multi-object tracker, adapted for CCTV and irregular frame timing.

Association per frame (Zhang et al., ByteTrack, ECCV 2022)::

    detections ──► high  (conf >= high_thresh)
               └─► low   (low_thresh <= conf < high_thresh)

    1. tracked + lost  ×  high   cost = 1 - BIoU(b1) [x score]   Hungarian, gated
    2. still-tracked   ×  low    cost = 1 - BIoU(b2)             keeps occluded people alive
    3. tentative       ×  rest of high                            confirm or drop
    4. unmatched high (conf >= new_thresh) ─► new tentative tracks

CCTV adaptations over the reference implementation

* continuous time: Kalman predicts with each track's real ``dt`` and every lifetime
  is expressed in seconds, so the tracker is indifferent to dropped frames or FPS changes;
* buffered IoU (C-BIoU) for small, fast or frame-skipped targets;
* optional Kalman Mahalanobis gating to reject geometrically implausible matches that
  IoU alone allows when people cross each other;
* class-aware association; duplicate suppression between tracked and lost pools.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from .kalman import KalmanBoxFilter, KalmanParams, x_to_xyxy, xs_to_xyxy
from .matching import CHI2_4DOF_95, iou_matrix, linear_assignment

F64 = npt.NDArray[np.float64]


class TrackState(IntEnum):
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    high_thresh: float = 0.60
    low_thresh: float = 0.15
    new_track_thresh: float = 0.70
    match_thresh_high: float = 0.80      # max cost (1 - IoU) in stage 1
    match_thresh_low: float = 0.50       # stage 2
    match_thresh_tentative: float = 0.70 # stage 3
    buffer_high: float = 0.3             # C-BIoU buffer, stage 1
    buffer_low: float = 0.5              # stage 2 (occluded / low-score targets drift more)
    fuse_score: bool = True
    mahalanobis_gate: float | None = CHI2_4DOF_95 * 4
    min_hits: int = 2                    # consecutive matches before a track is confirmed
    confirm_first_frame: bool = True     # reference behavior; disable for strict live alarm confirmation
    lost_ttl_s: float = 1.5              # keep lost tracks this long for re-identification
    duplicate_iou: float = 0.85
    class_aware: bool = True
    history: int = 300                   # trajectory points kept per track
    kalman: KalmanParams = field(default_factory=KalmanParams)


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    t: float
    foot: tuple[float, float]
    box: tuple[float, float, float, float]
    conf: float
    observed: bool        # False for Kalman-only (predicted) positions


class Track:
    __slots__ = (
        "cls",
        "conf",
        "cov",
        "history",
        "hits",
        "last_box",
        "mean",
        "misses",
        "state",
        "t_first",
        "t_last_seen",
        "t_state",
        "track_id",
    )

    def __init__(self, track_id: int, mean: F64, cov: F64, box: F64, conf: float, cls: int, t: float,
                 history: int) -> None:
        self.track_id = track_id
        self.mean = mean
        self.cov = cov
        self.conf = conf
        self.cls = cls
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.misses = 0
        self.t_first = t
        self.t_last_seen = t
        self.t_state = t
        self.last_box = box.copy()
        self.history: deque[TrajectoryPoint] = deque(maxlen=history)
        self._record(t, box, conf, observed=True)

    # ------------------------------------------------------------------ geometry

    @property
    def box(self) -> F64:
        """Kalman-smoothed xyxy box at the last predict/update time."""
        return x_to_xyxy(self.mean)

    @property
    def foot(self) -> tuple[float, float]:
        """Ground-contact point (bottom-centre) -- the reference for every zone rule."""
        b = self.box
        return float((b[0] + b[2]) / 2), float(b[3])

    @property
    def velocity(self) -> tuple[float, float]:
        """Centre velocity in px/s."""
        return float(self.mean[4]), float(self.mean[5])

    @property
    def height(self) -> float:
        b = self.box
        return float(b[3] - b[1])

    @property
    def age_s(self) -> float:
        return self.t_state - self.t_first

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.TRACKED, TrackState.LOST)

    def _record(self, t: float, box: F64, conf: float, observed: bool) -> None:
        self.history.append(TrajectoryPoint(
            t, (float((box[0] + box[2]) / 2), float(box[3])),
            (float(box[0]), float(box[1]), float(box[2]), float(box[3])), conf, observed))

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, {self.state.name}, hits={self.hits}, conf={self.conf:.2f})"


@dataclass(frozen=True, slots=True)
class TrackerOutput:
    tracks: list[Track]            # confirmed and matched this frame
    lost: list[Track]              # confirmed but currently unobserved (occluded)
    removed_ids: list[int]         # tracks deleted this frame (release per-track state!)
    t: float


class ByteTracker:
    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self.cfg = config or ByteTrackConfig()
        self.kf = KalmanBoxFilter(self.cfg.kalman)
        self._ids = itertools.count(1)
        self.tracked: list[Track] = []     # TRACKED + TENTATIVE
        self.lost: list[Track] = []
        self._t: float | None = None
        self.frame_count = 0

    def reset(self) -> None:
        self.tracked.clear()
        self.lost.clear()
        self._t = None
        self.frame_count = 0

    # ------------------------------------------------------------------ main step

    def update(self, detections: F64, t: float) -> TrackerOutput:
        """Associate one frame.

        detections  (N, 5|6) array ``[x1, y1, x2, y2, conf(, cls)]`` in pixels.
        t           capture timestamp in seconds (monotonic; e.g. ``frame.arrival_ns/1e9``).
        """
        cfg = self.cfg
        if not np.isfinite(t):
            raise ValueError('Tracker timestamp must be finite')
        dets = np.asarray(detections, dtype=np.float64)
        if dets.size == 0:
            dets = np.zeros((0, 6))
        elif dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.ndim != 2 or dets.shape[1] not in (5, 6):
            raise ValueError('Detections must be shaped N x 5 or N x 6')
        if dets.shape[1] == 5:
            dets = np.concatenate([dets, np.zeros((len(dets), 1))], axis=1)
        valid = (np.isfinite(dets).all(axis=1) & (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
                 & (dets[:, 4] >= cfg.low_thresh) & (dets[:, 4] <= 1)
                 & (dets[:, 5] >= 0) & (dets[:, 5] == np.floor(dets[:, 5])))
        dets = dets[valid]
        high = dets[dets[:, 4] >= cfg.high_thresh]
        low = dets[dets[:, 4] < cfg.high_thresh]

        if self._t is not None and t < self._t:
            # Clock went backwards (stream restarted): start over rather than mis-predict.
            self.reset()
        self._t = t
        self.frame_count += 1
        # Expire before association: an old ID must not be revived after its lifetime.
        removed = [tr for tr in self.tracked + self.lost if t-tr.t_last_seen > cfg.lost_ttl_s]
        for tr in removed:
            tr.state = TrackState.REMOVED
        self.tracked = [tr for tr in self.tracked if tr.state is not TrackState.REMOVED]
        self.lost = [tr for tr in self.lost if tr.state is not TrackState.REMOVED]

        confirmed = [tr for tr in self.tracked if tr.state is TrackState.TRACKED]
        tentative = [tr for tr in self.tracked if tr.state is TrackState.TENTATIVE]
        pool = confirmed + self.lost
        self._predict(pool + tentative, t)

        activated: list[Track] = []
        refound: list[Track] = []

        # ---- stage 1: confirmed + lost vs high
        matches, um_tracks, um_high = self._associate(pool, high, cfg.buffer_high, cfg.match_thresh_high,
                                                      fuse=cfg.fuse_score)
        for ti, di in matches:
            tr = pool[ti]
            was_lost = tr.state is TrackState.LOST
            self._apply(tr, high[di], t)
            (refound if was_lost else activated).append(tr)

        # ---- stage 2: still-tracked vs low (occlusion / motion blur / IR halo)
        remain = [pool[i] for i in um_tracks if pool[i].state is TrackState.TRACKED]
        matches2, um_remain, _ = self._associate(remain, low, cfg.buffer_low, cfg.match_thresh_low, fuse=False)
        for ti, di in matches2:
            self._apply(remain[ti], low[di], t)
            activated.append(remain[ti])
        newly_lost = []
        for i in um_remain:
            tr = remain[i]
            tr.state = TrackState.LOST
            tr.misses += 1
            newly_lost.append(tr)
        for i in um_tracks:
            tr = pool[i]
            if tr.state is TrackState.LOST and tr not in newly_lost:
                tr.misses += 1
                tr._record(t, tr.box, 0.0, observed=False)

        # ---- stage 3: tentative vs leftover high
        left_high = high[um_high]
        matches3, um_tent, um_new = self._associate(tentative, left_high, cfg.buffer_high,
                                                    cfg.match_thresh_tentative, fuse=cfg.fuse_score)
        for ti, di in matches3:
            tr = tentative[ti]
            self._apply(tr, left_high[di], t)
            if tr.hits >= cfg.min_hits:
                tr.state = TrackState.TRACKED
            activated.append(tr)
        for i in um_tent:
            tentative[i].state = TrackState.REMOVED
            removed.append(tentative[i])

        # ---- stage 4: births
        for di in um_new:
            d = left_high[di]
            if d[4] < cfg.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(d[:4])
            tr = Track(next(self._ids), mean, cov, d[:4], float(d[4]), int(d[5]), t, cfg.history)
            if cfg.min_hits <= 1 or (cfg.confirm_first_frame and self.frame_count == 1):
                tr.state = TrackState.TRACKED
            activated.append(tr)

        # ---- lost-pool expiry
        still_lost: list[Track] = []
        for tr in self.lost + newly_lost:
            if tr.state is not TrackState.LOST:
                continue
            if t - tr.t_last_seen > cfg.lost_ttl_s:
                tr.state = TrackState.REMOVED
                removed.append(tr)
            elif tr not in still_lost:
                still_lost.append(tr)

        self.tracked = [tr for tr in activated + refound if tr.state in (TrackState.TRACKED, TrackState.TENTATIVE)]
        self.lost = [tr for tr in still_lost if tr not in self.tracked]
        removed += self._dedupe()

        out_tracks = [tr for tr in self.tracked if tr.state is TrackState.TRACKED]
        return TrackerOutput(out_tracks, list(self.lost), [tr.track_id for tr in removed], t)

    # ------------------------------------------------------------------ helpers

    def _predict(self, tracks: list[Track], t: float) -> None:
        if not tracks:
            return
        means = np.stack([tr.mean for tr in tracks])
        covs = np.stack([tr.cov for tr in tracks])
        dts = np.array([max(0.0, t - tr.t_state) for tr in tracks])
        means, covs = self.kf.multi_predict(means, covs, dts)
        for tr, m, c in zip(tracks, means, covs):
            tr.mean, tr.cov, tr.t_state = m, c, t

    def _associate(self, tracks: list[Track], dets: F64, buffer: float, thresh: float, *, fuse: bool
                   ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        if not tracks or len(dets) == 0:
            return np.empty((0, 2), np.int64), np.arange(len(tracks)), np.arange(len(dets))
        boxes = xs_to_xyxy(np.stack([tr.mean for tr in tracks]))
        iou = iou_matrix(boxes, dets[:, :4], buffer=buffer)
        if fuse:
            iou = iou * dets[None, :, 4]
        cost = 1.0 - iou
        if self.cfg.class_aware:
            cls = np.array([tr.cls for tr in tracks])
            cost[cls[:, None] != dets[None, :, 5].astype(int)] = np.inf
        gate = self.cfg.mahalanobis_gate
        if gate is not None:
            for i, tr in enumerate(tracks):
                cand = np.flatnonzero(np.isfinite(cost[i]) & (cost[i] <= thresh))
                if len(cand):
                    d2 = self.kf.mahalanobis(tr.mean, tr.cov, dets[cand, :4])
                    cost[i, cand[d2 > gate]] = np.inf
        return linear_assignment(cost, thresh)

    def _apply(self, tr: Track, det: F64, t: float) -> None:
        tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, det[:4])
        tr.conf = float(det[4])
        tr.cls = int(det[5])
        tr.hits += 1
        tr.misses = 0
        tr.t_last_seen = t
        tr.last_box = det[:4].copy()
        if tr.state is TrackState.LOST:
            tr.state = TrackState.TRACKED
        tr._record(t, tr.box, tr.conf, observed=True)

    def _dedupe(self) -> list[Track]:
        """Remove lost tracks that coincide with a currently observed track."""
        if not self.tracked or not self.lost:
            return []
        a = np.stack([tr.box for tr in self.tracked])
        b = np.stack([tr.box for tr in self.lost])
        iou = iou_matrix(a, b)
        # The observed track always wins: it carries this frame's measurement, while the
        # lost one is a pure extrapolation that stage 1 already failed to match.
        duplicates = iou > self.cfg.duplicate_iou
        if self.cfg.class_aware:
            duplicates &= np.array([tr.cls for tr in self.tracked])[:, None] == np.array([tr.cls for tr in self.lost])[None, :]
        drop_l = set(np.nonzero(duplicates)[1].tolist())
        removed = [self.lost[j] for j in drop_l]
        for tr in removed:
            tr.state = TrackState.REMOVED
        self.lost = [tr for j, tr in enumerate(self.lost) if j not in drop_l]
        return removed
