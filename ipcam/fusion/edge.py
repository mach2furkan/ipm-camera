"""Edge node side of Phase 7: per-frame analytics output -> FrameBatch / SecurityAlert on the bus."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..analytics.engine import AnalyticsResult
from .homography import CameraCalibration
from .reid import EmbeddingExtractor, ReIDGate, crop, occlusion_scores
from .transport.bus import MessageBus
from .transport.codec import DetectionEvent, FrameBatch, SecurityAlert


class EdgePublisher:
    """Projects tracks to world coordinates, runs gated Re-ID and publishes one batch per frame.

    ``pixel_sigma_rel`` sets the foot-point uncertainty relative to the box height (the
    bottom edge of a detection box is less certain vertically than horizontally); it is
    propagated through the homography into the world covariance that the fusion engine
    uses as measurement noise.

    Re-ID crops may come from a higher-resolution image than the analytics frame (e.g.
    the main stream while analytics runs on the sub stream): pass ``crop_image`` and the
    ``crop_scale`` between the two.
    """

    def __init__(
        self,
        camera_id: str,
        calibration: CameraCalibration,
        bus: MessageBus,
        *,
        extractor: EmbeddingExtractor | None = None,
        gate: ReIDGate | None = None,
        pixel_sigma_rel: tuple[float, float] = (0.04, 0.06),
        min_sigma_px: float = 1.0,
        extrapolation_inflation: float = 3.0,
    ) -> None:
        self.camera_id = camera_id
        self.cal = calibration
        self.bus = bus
        self.extractor = extractor
        self.gate = gate or ReIDGate()
        self.sigma_rel = pixel_sigma_rel
        self.min_sigma = min_sigma_px
        self.inflation = extrapolation_inflation
        self.sequence = 0
        self.events_subject = f"events.camera.{camera_id}"
        self.alerts_subject = f"alerts.camera.{camera_id}"

    def build(self, result: AnalyticsResult, timestamp_ns: int, *, crop_image: npt.NDArray[Any] | None = None,
              crop_scale: float = 1.0) -> FrameBatch:
        tracks = result.tracks
        boxes = np.array([tv.box for tv in tracks], dtype=np.float64).reshape(-1, 4)
        occl = occlusion_scores(boxes)
        embeddings: dict[int, npt.NDArray[np.float32]] = {}
        if self.extractor is not None and crop_image is not None and len(tracks):
            ids, crops = [], []
            for tv, oc in zip(tracks, occl):
                b = [v * crop_scale for v in tv.box]
                c = crop(crop_image, b)
                ok, _ = self.gate.admit(tv.track_id, b, crop_image.shape, float(oc), result.t, c)
                if ok:
                    ids.append(tv.track_id)
                    crops.append(c)
            if crops:
                for tid, emb in zip(ids, self.extractor(crops)):
                    embeddings[tid] = emb

        dets: list[DetectionEvent] = []
        if len(tracks):
            feet = np.array([tv.foot for tv in tracks])
            world = self.cal.to_world(feet)
            inside = self.cal.in_calibrated_region(feet, margin_px=20.0)
            for i, tv in enumerate(tracks):
                h = max(tv.box[3] - tv.box[1], 1.0)
                k = 1.0 if inside[i] else self.inflation   # outside the surveyed hull: trust less
                su = max(self.min_sigma, self.sigma_rel[0] * h) * k
                sv = max(self.min_sigma, self.sigma_rel[1] * h) * k
                valid = bool(np.all(np.isfinite(world[i])))
                cov = self.cal.world_covariance(tv.foot[0], tv.foot[1], su, sv) if valid else None
                dets.append(DetectionEvent(
                    camera_id=self.camera_id, timestamp_ns=timestamp_ns, local_track_id=tv.track_id,
                    bbox=(*tv.box, tv.conf),  # type: ignore[arg-type]
                    world=(float(world[i, 0]), float(world[i, 1]), 0.0) if valid else None,
                    reid_embedding=embeddings.get(tv.track_id),
                    is_occluded=bool(occl[i] >= self.gate.cfg.max_occlusion),
                    world_covariance=(float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])) if cov is not None else None,
                    world_valid=valid,
                ))
        self.gate.forget(result.removed_ids)
        self.sequence += 1
        return FrameBatch(self.camera_id, timestamp_ns, self.sequence, dets, list(result.removed_ids))

    async def publish(self, result: AnalyticsResult, *, timestamp_ns: int | None = None,
                      crop_image: npt.NDArray[Any] | None = None, crop_scale: float = 1.0) -> FrameBatch:
        ts = time.time_ns() if timestamp_ns is None else timestamp_ns
        batch = self.build(result, ts, crop_image=crop_image, crop_scale=crop_scale)
        await self.bus.publish(self.events_subject, batch.encode())
        for ev in result.events:
            alert = SecurityAlert(self.camera_id, ts, ev.track_id, ev.kind.value, ev.rule, ev.direction or "",
                                  float(ev.dwell_s or 0.0))
            await self.bus.publish(self.alerts_subject, alert.encode())
        return batch


def batch_to_observations(batch: FrameBatch, *, clock_offset_s: float = 0.0) -> Sequence[Any]:
    """Decode helper used by the fusion service (kept here to colocate the field mapping)."""
    from .global_tracker import LocalObservation

    out = []
    for d in batch.detections:
        t = d.timestamp_ns / 1e9 + clock_offset_s
        xy = np.array(d.world[:2]) if d.world_valid and d.world is not None else None
        cov = None
        if d.world_covariance is not None:
            xx, xy_, yy = d.world_covariance
            cov = np.array([[xx, xy_], [xy_, yy]], dtype=np.float64)
        emb = None
        if d.reid_embedding is not None and len(d.reid_embedding):
            emb = np.array(d.reid_embedding, dtype=np.float32)   # detach from the message buffer
        box = tuple(d.bbox[:4]) if d.bbox else (0.0, 0.0, 0.0, 0.0)
        out.append(LocalObservation(d.camera_id or batch.camera_id, int(d.local_track_id), t, xy, cov,
                                    box, float(d.bbox[4]) if d.bbox else 1.0, emb, d.is_occluded))  # type: ignore[arg-type]
    return out
