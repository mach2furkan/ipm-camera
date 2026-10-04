"""Global Track Manager: fuses per-camera local tracks into site-wide identities.

Association of a new local track (camera C, local id L), in order:

1. **Overlap fusion** -- an entity currently seen by a camera whose FOV overlaps C lies
   within 0.8 m (and passes a chi-square gate on the combined covariance) of the
   projected foot point -> same person, seen twice. A Re-ID veto prevents fusing two
   different people standing side by side.
2. **Provisional birth** -- otherwise a new entity is shown on the map immediately,
   flagged *provisional*.
3. **Re-ID handover** -- as soon as the provisional entity has a key-crop embedding it is
   compared (cosine distance on the FAISS/flat index) against *lost* entities that the
   camera topology allows (t_min <= dt <= t_max from their last camera):
   D_C < 0.28 merge at once; 0.28 <= D_C < 0.40 wait for 3 embeddings and decide on their
   median; otherwise it stays a new identity. A merge keeps the older global id, so the
   operator sees one continuous identity across cameras.

Each observation updates the entity's metric Kalman filter with its own homography
covariance (covariance-weighted multi-camera fusion). A local track whose measurements
become persistently inconsistent with its entity (edge tracker identity switch) is split
off rather than dragging the entity across the site.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .reid import EmbeddingGallery, l2_normalize
from .topology import CameraTopology
from .vector_index import create_index
from .worldkf import WorldKF

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
LocalKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class LocalObservation:
    camera_id: str
    local_track_id: int
    t: float
    xy: F64 | None                    # world metres; None if the foot point is not on the ground model
    cov: F64 | None                   # 2x2 world covariance
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    conf: float = 1.0
    embedding: F32 | None = None
    occluded: bool = False


class EntityStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"


@dataclass(slots=True)
class CameraVisit:
    camera_id: str
    t_in: float
    t_out: float | None = None


@dataclass(eq=False)
class GlobalEntity:
    gid: int
    kf: WorldKF
    first_seen: float
    last_seen: float
    last_camera: str
    provisional: bool = True
    status: EntityStatus = EntityStatus.ACTIVE
    links: dict[str, int] = field(default_factory=dict)            # camera -> local id
    last_seen_by: dict[str, float] = field(default_factory=dict)
    gallery: EmbeddingGallery = field(default_factory=EmbeddingGallery)
    breadcrumbs: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=4000))
    visits: list[CameraVisit] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    reid_settled: bool = False
    birth_xy: F64 | None = None
    aliases: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"G-{self.gid}"

    def summary(self, t: float, *, crumbs: int = 300) -> dict[str, Any]:
        pos = self.kf.peek(t) if self.status is EntityStatus.ACTIVE else self.kf.position
        return {
            "gid": self.gid, "label": self.label, "status": self.status.value, "provisional": self.provisional,
            "x": float(pos[0]), "y": float(pos[1]),
            "vx": float(self.kf.mean[2]), "vy": float(self.kf.mean[3]), "speed": round(self.kf.speed, 2),
            "pos_std": float(math.sqrt(max(np.trace(self.kf.cov[:2, :2]) / 2, 0.0))),
            "cameras": sorted(self.links), "first_seen": self.first_seen, "last_seen": self.last_seen,
            "tags": sorted(self.tags), "aliases": list(self.aliases),
            "breadcrumbs": [[round(x, 2), round(y, 2)] for _, x, y in list(self.breadcrumbs)[-crumbs:]],
            "visits": [[v.camera_id, v.t_in, v.t_out] for v in self.visits[-20:]],
            "embeddings": len(self.gallery),
        }


@dataclass(frozen=True, slots=True)
class FusionEvent:
    kind: str            # new | fused | handover | lost | retired | split
    gid: int
    t: float
    camera_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FusionConfig:
    overlap_gate_m: float = 0.8
    overlap_chi2: float = 13.8            # 2 dof, 99.9 %
    overlap_recent_s: float = 0.6
    fuse_window_s: float = 3.0            # provisional entities may still be overlap-merged
    reid_accept: float = 0.28
    reid_ambiguous: float = 0.40
    reid_confirm_n: int = 3
    reid_margin: float = 0.05
    reid_veto: float = 0.45               # never overlap-fuse two embeddings further apart than this
    max_speed_mps: float = 3.5            # kinematic gate: straight-line distance / dt above this is impossible
    transition_weight: float = 0.10       # d_eff = D_C + w * (1 - P(transition | dt) / P_max)
    min_transition_likelihood: float = 0.011  # exp(-3^2/2): beyond 3 sigma of the travel-time model
    provisional_timeout_s: float = 6.0    # give up on re-identification after this long
    stale_link_s: float = 3.0             # edge stopped reporting a local track without ending it
    late_tolerance_s: float = 0.3
    split_chi2: float = 60.0
    split_after: int = 4
    breadcrumb_dt_s: float = 0.2
    embedding_dim: int = 512
    index_backend: str = "flat"
    learn_topology: bool = True


class GlobalTrackManager:
    def __init__(self, topology: CameraTopology | None = None, config: FusionConfig | None = None) -> None:
        self.topology = topology or CameraTopology()
        self.cfg = config or FusionConfig()
        self.entities: dict[int, GlobalEntity] = {}
        self.links: dict[LocalKey, int] = {}
        self.index = create_index(self.cfg.embedding_dim, self.cfg.index_backend)
        self.alias: dict[int, int] = {}
        self.retired: deque[dict[str, Any]] = deque(maxlen=2000)
        self._events: list[FusionEvent] = []
        self._ids = itertools.count(1)
        self._inconsistent: dict[LocalKey, int] = {}
        self.t = 0.0

    # ------------------------------------------------------------------ public API

    def resolve(self, gid: int) -> int:
        while gid in self.alias:
            gid = self.alias[gid]
        return gid

    def gid_of(self, camera_id: str, local_track_id: int) -> int | None:
        return self.links.get((camera_id, local_track_id))

    def drain_events(self) -> list[FusionEvent]:
        ev, self._events = self._events, []
        return ev

    def observe(self, obs: LocalObservation) -> int | None:
        self.t = max(self.t, obs.t)
        key = (obs.camera_id, obs.local_track_id)
        gid = self.links.get(key)
        if gid is not None:
            ent = self.entities[gid]
            self._update(ent, obs, key)
            self._resolve_if_current(key, ent, obs)
            return self.links.get(key)
        if obs.xy is None or obs.cov is None or not np.all(np.isfinite(obs.xy)):
            return None   # cannot place it on the map yet; wait for a usable foot point
        target = self._overlap_candidate(obs, exclude=None)
        if target is not None:
            self._link(key, target, obs)
            self._update(target, obs, key)
            self._emit("fused", target.gid, obs.t, obs.camera_id, {"cameras": sorted(target.links)})
            return target.gid
        ent = self._birth(obs, key)
        self._resolve_if_current(key, ent, obs)
        return self.links.get(key)

    def _resolve_if_current(self, key: LocalKey, ent: GlobalEntity, obs: LocalObservation) -> None:
        # ``_update`` may have merged or split ``ent`` away; only resolve a live link.
        if self.links.get(key) == ent.gid and ent.gid in self.entities and ent.provisional:
            self._try_resolve(ent, obs)

    def end_local(self, camera_id: str, local_track_id: int, t: float) -> None:
        key = (camera_id, local_track_id)
        gid = self.links.pop(key, None)
        self._inconsistent.pop(key, None)
        if gid is None:
            return
        ent = self.entities.get(gid)
        if ent is None:
            return
        if ent.links.get(camera_id) == local_track_id:
            del ent.links[camera_id]
            for v in reversed(ent.visits):
                if v.camera_id == camera_id and v.t_out is None:
                    v.t_out = t
                    break
        if not ent.links:
            ent.status = EntityStatus.LOST
            self._emit("lost", gid, t, camera_id)

    def tick(self, t: float) -> None:
        """Housekeeping: stale links, retirement of entities that left the site."""
        self.t = max(self.t, t)
        for key, gid in list(self.links.items()):
            ent = self.entities.get(gid)
            if ent is None or t - ent.last_seen_by.get(key[0], ent.last_seen) > self.cfg.stale_link_s:
                self.end_local(key[0], key[1], t)
        for ent in list(self.entities.values()):
            if ent.links:
                continue
            horizon = self.topology.t_max_from(ent.last_camera)
            if t - ent.last_seen > horizon:
                self._retire(ent, t)
            elif ent.provisional and t - ent.first_seen > self.cfg.provisional_timeout_s:
                ent.provisional = False

    def snapshot(self, t: float | None = None) -> list[dict[str, Any]]:
        t = self.t if t is None else t
        return [e.summary(t) for e in self.entities.values()]

    # ------------------------------------------------------------------ association internals

    def _emit(self, kind: str, gid: int, t: float, camera: str | None, details: dict[str, Any] | None = None) -> None:
        self._events.append(FusionEvent(kind, gid, t, camera, details or {}))

    def _birth(self, obs: LocalObservation, key: LocalKey) -> GlobalEntity:
        assert obs.xy is not None and obs.cov is not None
        gid = next(self._ids)
        ent = GlobalEntity(gid, WorldKF(obs.xy, obs.cov, obs.t), obs.t, obs.t, obs.camera_id,
                           birth_xy=np.asarray(obs.xy, dtype=np.float64).copy())
        self.entities[gid] = ent
        self._link(key, ent, obs)
        self._update(ent, obs, key)
        self._emit("new", gid, obs.t, obs.camera_id)
        return ent

    def _link(self, key: LocalKey, ent: GlobalEntity, obs: LocalObservation) -> None:
        cam, lid = key
        old = ent.links.get(cam)
        if old is not None and old != lid:
            self.links.pop((cam, old), None)
        ent.links[cam] = lid
        self.links[key] = ent.gid
        ent.status = EntityStatus.ACTIVE
        if not ent.visits or ent.visits[-1].camera_id != cam or ent.visits[-1].t_out is not None:
            ent.visits.append(CameraVisit(cam, obs.t))

    def _update(self, ent: GlobalEntity, obs: LocalObservation, key: LocalKey) -> bool:
        """Apply one observation. Returns False if the local track was split off."""
        cfg = self.cfg
        ent.last_seen = max(ent.last_seen, obs.t)
        ent.last_seen_by[obs.camera_id] = obs.t
        ent.last_camera = obs.camera_id
        ent.status = EntityStatus.ACTIVE
        if obs.embedding is not None:
            ent.gallery.add(obs.embedding, obs.t)
            proto = ent.gallery.prototype
            if proto is not None:
                self.index.upsert(ent.gid, proto)
        if obs.xy is None or obs.cov is None or not np.all(np.isfinite(obs.xy)):
            return True
        if obs.t < ent.kf.t - cfg.late_tolerance_s:
            return True                       # too old to be useful; ordering jitter on the bus
        ent.kf.predict(obs.t)
        d2 = ent.kf.mahalanobis2(obs.xy, obs.cov)
        if d2 > cfg.split_chi2 and len(ent.links) > 1:
            n = self._inconsistent.get(key, 0) + 1
            self._inconsistent[key] = n
            if n >= cfg.split_after:
                self._split(ent, obs, key)
                return False
            return True
        self._inconsistent.pop(key, None)
        ent.kf.update(obs.xy, obs.cov)
        if not ent.breadcrumbs or obs.t - ent.breadcrumbs[-1][0] >= cfg.breadcrumb_dt_s:
            p = ent.kf.position
            ent.breadcrumbs.append((obs.t, float(p[0]), float(p[1])))
        if ent.provisional and obs.t - ent.first_seen <= cfg.fuse_window_s:
            other = self._overlap_candidate(obs, exclude=ent)
            if other is not None:
                older, newer = (other, ent) if other.first_seen <= ent.first_seen else (ent, other)
                self._merge(older, newer, obs.t, kind="fused", camera=obs.camera_id)
        return True

    def _split(self, ent: GlobalEntity, obs: LocalObservation, key: LocalKey) -> None:
        cam, _ = key
        ent.links.pop(cam, None)
        self.links.pop(key, None)
        self._inconsistent.pop(key, None)
        self._emit("split", ent.gid, obs.t, cam)
        self._birth(obs, key)

    def _overlap_candidate(self, obs: LocalObservation, *, exclude: GlobalEntity | None) -> GlobalEntity | None:
        cfg = self.cfg
        if obs.xy is None or obs.cov is None:
            return None
        best: GlobalEntity | None = None
        best_d = cfg.overlap_gate_m
        for ent in self.entities.values():
            if ent is exclude or obs.camera_id in ent.links or not ent.links:
                continue
            if not any(self.topology.overlaps(c, obs.camera_id) and c != obs.camera_id for c in ent.links):
                continue
            if obs.t - ent.last_seen > cfg.overlap_recent_s:
                continue
            pred = ent.kf.peek(obs.t)
            d = float(np.hypot(*(pred - obs.xy)))
            if d >= best_d:
                continue
            if ent.kf.mahalanobis2(obs.xy, obs.cov) > cfg.overlap_chi2 * 4:
                continue
            emb = obs.embedding if obs.embedding is not None else (
                exclude.gallery.prototype if exclude is not None else None)
            proto = ent.gallery.prototype
            if emb is not None and proto is not None and 1.0 - float(np.dot(l2_normalize(emb), proto)) > cfg.reid_veto:
                continue
            best, best_d = ent, d
        return best

    def _try_resolve(self, ent: GlobalEntity, obs: LocalObservation) -> None:
        """Re-identify a provisional entity against lost entities (camera handover)."""
        cfg = self.cfg
        if not ent.provisional or ent.reid_settled or len(ent.gallery) == 0:
            if ent.provisional and obs.t - ent.first_seen > cfg.provisional_timeout_s:
                ent.provisional = False
            return
        allowed: dict[int, float] = {}
        cam = obs.camera_id
        birth = ent.birth_xy if ent.birth_xy is not None else ent.kf.position
        for cand in self.entities.values():
            if cand is ent or cand.links or len(cand.gallery) == 0:
                continue
            dt = ent.first_seen - cand.last_seen
            ok, like = self.topology.gate(cand.last_camera, cam, dt)
            if not ok or like < cfg.min_transition_likelihood:
                continue
            # Kinematic gate: the straight line is a lower bound of the walked path, so
            # dist / dt is a lower bound of the required speed. This catches what a loose
            # topology (t_min = 0 between neighbouring views) cannot: a look-alike appearing
            # behind a person who was last seen walking the other way.
            dist = float(np.hypot(*(cand.kf.position - birth)))
            if dist / max(dt, 0.5) > cfg.max_speed_mps:
                continue
            allowed[cand.gid] = like
        if not allowed:
            if len(ent.gallery) >= cfg.reid_confirm_n or obs.t - ent.first_seen > cfg.provisional_timeout_s:
                ent.provisional = False
            return
        n = len(ent.gallery)
        query = ent.gallery.recent(1)[0] if n < cfg.reid_confirm_n else \
            l2_normalize(np.median(np.stack(ent.gallery.recent(cfg.reid_confirm_n)), axis=0))
        raw = self.index.search(query, k=5, allowed=set(allowed))
        if not raw:
            return
        # Spatio-temporal prior folded into the distance: atypical travel times need a
        # proportionally better appearance match.
        hits = sorted(((g, d + cfg.transition_weight * (1.0 - allowed[g])) for g, d in raw), key=lambda h: h[1])
        best_gid, best_d = hits[0]
        margin = (hits[1][1] - best_d) if len(hits) > 1 else 1.0
        decided = n >= cfg.reid_confirm_n
        accept = best_d < cfg.reid_accept and margin >= cfg.reid_margin
        if not accept and decided:
            accept = best_d < cfg.reid_ambiguous and margin >= cfg.reid_margin
        if accept:
            old = self.entities[best_gid]
            dt = ent.first_seen - old.last_seen
            src = old.last_camera
            self._merge(old, ent, obs.t, kind="handover", camera=cam,
                        details={"from_camera": src, "dt_s": round(dt, 2), "reid_distance": round(best_d, 3),
                                 "transition_likelihood": round(allowed[best_gid], 3)})
            if cfg.learn_topology:
                m = self.topology.model(src, cam)
                if m is not None:
                    m.observe(dt)
        elif decided or best_d >= cfg.reid_ambiguous:
            ent.reid_settled = decided or best_d >= cfg.reid_ambiguous
            if ent.reid_settled:
                ent.provisional = False

    def _merge(self, keep: GlobalEntity, drop: GlobalEntity, t: float, *, kind: str, camera: str | None,
               details: dict[str, Any] | None = None) -> None:
        """Fold ``drop`` into ``keep`` (older id survives)."""
        if keep is drop:
            return
        # The more recently updated filter carries the current position.
        if drop.kf.t >= keep.kf.t and not keep.links:
            keep.kf = drop.kf
        elif drop.links:
            keep.kf.predict(drop.kf.t)
            keep.kf.update(drop.kf.position, drop.kf.cov[:2, :2])
        for cam, lid in drop.links.items():
            keep.links[cam] = lid
            self.links[(cam, lid)] = keep.gid
        for _, emb in list(drop.gallery._items):
            keep.gallery.add(emb)
        if keep.gallery.prototype is not None:
            self.index.upsert(keep.gid, keep.gallery.prototype)
        self.index.remove(drop.gid)
        keep.breadcrumbs.extend(drop.breadcrumbs)
        keep.visits.extend(drop.visits)
        keep.tags |= drop.tags
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        keep.last_seen_by.update(drop.last_seen_by)
        keep.last_camera = drop.last_camera if drop.last_seen >= keep.last_seen else keep.last_camera
        keep.status = EntityStatus.ACTIVE if keep.links else keep.status
        keep.provisional = False
        keep.aliases.append(drop.gid)
        self.alias[drop.gid] = keep.gid
        del self.entities[drop.gid]
        self._emit(kind, keep.gid, t, camera, {"merged": drop.gid, **(details or {})})

    def _retire(self, ent: GlobalEntity, t: float) -> None:
        self.index.remove(ent.gid)
        self.retired.append(ent.summary(t, crumbs=4000))
        del self.entities[ent.gid]
        self._emit("retired", ent.gid, t, ent.last_camera)
