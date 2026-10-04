"""Central fusion service: bus -> Global Track Manager -> PTZ engagement -> Common Operating Picture."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics import RollingWindow
from .edge import batch_to_observations
from .global_tracker import EntityStatus, FusionEvent, GlobalTrackManager
from .homography import CameraCalibration
from .ptz import DetectionProvider, EngagementState, PTZController, PTZMount, PTZPose, SlewToCueEngagement, fov_deg
from .transport.bus import MessageBus
from .transport.codec import FrameBatch, SecurityAlert

log = logging.getLogger(__name__)


@dataclass
class PTZUnit:
    mount: PTZMount
    controller: PTZController
    detections: DetectionProvider
    coverage_m: float = 120.0
    clock: Callable[[], float] | None = None      # engagement time base (monotonic by default)
    pose_provider: Callable[[], "PTZPose | None"] | None = None   # head pose outside engagements
    engagement: SlewToCueEngagement | None = None
    target_gid: int | None = None


@dataclass(frozen=True, slots=True)
class COPEvent:
    t: float
    kind: str
    text: str
    gid: int | None = None
    camera_id: str | None = None


@dataclass
class SiteConfig:
    name: str = "site"
    bounds: tuple[float, float, float, float] | None = None   # xmin, ymin, xmax, ymax (m)
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)
    footprint_range_m: float | None = 60.0     # draw camera coverage only to its useful range


class FusionService:
    def __init__(
        self,
        bus: MessageBus,
        manager: GlobalTrackManager,
        *,
        site: SiteConfig | None = None,
        ptz_units: list[PTZUnit] | None = None,
        tick_hz: float = 5.0,
        engage_on: Callable[[SecurityAlert], bool] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.bus = bus
        self.manager = manager
        self.site = site or SiteConfig()
        self.ptz = ptz_units or []
        self.tick_period = 1.0 / tick_hz
        self.engage_on = engage_on or (lambda a: a.kind in ("intrusion", "tripwire", "loitering"))
        self.clock = clock
        self.feed: deque[COPEvent] = deque(maxlen=500)
        self.e2e_ms = RollingWindow(2048)
        self.batches = 0
        self._tasks: list[asyncio.Task[None]] = []
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        ev = await self.bus.subscribe("events.camera.*", durable="fusion-events")
        al = await self.bus.subscribe("alerts.camera.*", durable="fusion-alerts")
        self._subs = (ev, al)
        self._tasks = [asyncio.create_task(self._consume_events(ev), name="fusion-events"),
                       asyncio.create_task(self._consume_alerts(al), name="fusion-alerts"),
                       asyncio.create_task(self._ticker(), name="fusion-tick")]

    async def stop(self) -> None:
        for unit in self.ptz:
            if unit.engagement is not None:
                await unit.engagement.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        for s in getattr(self, "_subs", ()):
            await s.close()

    # ------------------------------------------------------------------ ingestion

    def ingest_batch(self, batch: FrameBatch) -> None:
        self.batches += 1
        self.e2e_ms.add((self.clock() - batch.timestamp_ns / 1e9) * 1000.0)
        for obs in batch_to_observations(batch):
            self.manager.observe(obs)
        t = batch.timestamp_ns / 1e9
        for lid in batch.ended_track_ids:
            self.manager.end_local(batch.camera_id, int(lid), t)
        self._drain_fusion_events()

    async def _consume_events(self, sub: Any) -> None:
        async for msg in sub:
            try:
                self.ingest_batch(FrameBatch.decode(msg.data))
            except Exception:  # noqa: BLE001 - a malformed message must not kill the engine
                log.exception("bad frame batch on %s", msg.subject)

    async def _consume_alerts(self, sub: Any) -> None:
        async for msg in sub:
            try:
                await self.handle_alert(SecurityAlert.decode(msg.data))
            except Exception:  # noqa: BLE001
                log.exception("bad alert on %s", msg.subject)

    async def handle_alert(self, alert: SecurityAlert) -> None:
        gid = self.manager.gid_of(alert.camera_id, alert.local_track_id)
        label = f"G-{gid}" if gid is not None else f"{alert.camera_id}/#{alert.local_track_id}"
        what = {"intrusion": "yasak bölgeye girdi", "tripwire": f"sanal çiti geçti ({alert.direction})",
                "loitering": f"aylak dolaşıyor ({alert.dwell_s:.0f} s)", "zone_exit": "bölgeden çıktı"}
        text = f"{label} {what.get(alert.kind, alert.kind)} [{alert.rule}]"
        if gid is not None:
            ent = self.manager.entities.get(gid)
            if ent is not None and alert.kind != "zone_exit":
                ent.tags.add(f"{alert.kind}:{alert.rule}")
        if gid is not None and self.engage_on(alert):
            unit = self._assign_ptz(gid)
            if unit is not None:
                text += f" -> {unit.mount.camera_id} angaje oldu"
        self._push(COPEvent(alert.timestamp_ns / 1e9, alert.kind, text, gid, alert.camera_id))

    def _drain_fusion_events(self) -> None:
        for ev in self.manager.drain_events():
            text = self._describe(ev)
            if text:
                self._push(COPEvent(ev.t, ev.kind, text, ev.gid, ev.camera_id))
            if ev.kind in ("handover", "fused") and ev.details.get("merged") is not None:
                for unit in self.ptz:
                    if unit.target_gid == ev.details["merged"]:
                        unit.target_gid = ev.gid

    @staticmethod
    def _describe(ev: FusionEvent) -> str | None:
        if ev.kind == "handover":
            return (f"G-{ev.gid} {ev.details.get('from_camera')} -> {ev.camera_id} geçişi "
                    f"(Δt {ev.details.get('dt_s')} s, D_C {ev.details.get('reid_distance')})")
        if ev.kind == "new":
            return f"G-{ev.gid} ilk kez görüldü ({ev.camera_id})"
        if ev.kind == "retired":
            return f"G-{ev.gid} sahadan ayrıldı"
        if ev.kind == "split":
            return f"G-{ev.gid} tutarsız iz ayrıldı ({ev.camera_id})"
        return None

    def _push(self, ev: COPEvent) -> None:
        self.feed.append(ev)

    def push_event(self, kind: str, text: str, *, camera_id: str | None = None, gid: int | None = None,
                   t: float | None = None) -> None:
        """Add an operator-log entry from another subsystem (thermal analytics, ROI rules ...)."""
        self._push(COPEvent(self.clock() if t is None else t, kind, text, gid, camera_id))

    # ------------------------------------------------------------------ PTZ

    def _target_provider(self, unit: PTZUnit) -> Callable[[], Any]:
        def provide() -> Any:
            if unit.target_gid is None:
                return None
            gid = self.manager.resolve(unit.target_gid)
            ent = self.manager.entities.get(gid)
            if ent is None or ent.status is EntityStatus.LOST and self.manager.t - ent.last_seen > 5.0:
                return None
            return ent.kf.peek(self.manager.t), ent.kf.velocity
        return provide

    def _assign_ptz(self, gid: int) -> PTZUnit | None:
        ent = self.manager.entities.get(gid)
        if ent is None or not self.ptz:
            return None
        pos = ent.kf.position
        best, best_d = None, math.inf
        for unit in self.ptz:
            d = math.hypot(pos[0] - unit.mount.x, pos[1] - unit.mount.y)
            busy = unit.engagement is not None and unit.engagement.state is not EngagementState.IDLE
            if d <= unit.coverage_m and (not busy or unit.target_gid == gid) and d < best_d:
                best, best_d = unit, d
        if best is None:
            return None
        best.target_gid = gid
        if best.engagement is None:
            kw = {"clock": best.clock} if best.clock is not None else {}
            best.engagement = SlewToCueEngagement(best.controller, best.mount, self._target_provider(best),
                                                  best.detections, **kw)
        best.engagement.start()
        return best

    # ------------------------------------------------------------------ COP

    async def _ticker(self) -> None:
        while True:
            # Edge timestamps are wall-clock (epoch) seconds; ``clock`` must be on the same base.
            self.manager.tick(self.clock())
            self._drain_fusion_events()
            if self._subscribers:
                snap = self.snapshot()
                for q in tuple(self._subscribers):
                    if q.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            q.get_nowait()
                    q.put_nowait(snap)
            await asyncio.sleep(self.tick_period)

    def subscribe_cop(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self._subscribers.add(q)
        return q

    def unsubscribe_cop(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def snapshot(self) -> dict[str, Any]:
        t = self.manager.t
        cams = []
        for cid, cal in self.site.cameras.items():
            try:
                fp = cal.ground_footprint(max_range_m=self.site.footprint_range_m)
                fp = fp[np.all(np.isfinite(fp), axis=1)]
            except Exception:  # noqa: BLE001
                fp = np.zeros((0, 2))
            cams.append({"id": cid, "position": list(cal.position) if cal.position else None,
                         "footprint": np.round(fp, 2).tolist(), "rmse_m": cal.rmse_m})
        ptz = []
        for unit in self.ptz:
            eng = unit.engagement
            pose = eng.pose if eng is not None and eng.pose is not None else (
                unit.pose_provider() if unit.pose_provider is not None else None)
            bearing = None
            if pose is not None:
                az = -pose.azimuth_deg if unit.mount.azimuth_clockwise else pose.azimuth_deg
                bearing = (az + unit.mount.azimuth_zero_bearing_deg) % 360
            ptz.append({
                "id": unit.mount.camera_id, "x": unit.mount.x, "y": unit.mount.y,
                "state": eng.state.value if eng else "idle", "target": unit.target_gid,
                "bearing_deg": bearing, "zoom": pose.zoom if pose else None,
                "hfov_deg": fov_deg(unit.mount, pose.zoom)[0] if pose else None,
                "cue_to_lock_ms": eng.stats.cue_to_lock_ms[-1] if eng and eng.stats.cue_to_lock_ms else None,
            })
        e2e = self.e2e_ms.summary()
        return {
            "t": t, "site": {"name": self.site.name, "bounds": self.site.bounds},
            "cameras": cams, "ptz": ptz,
            "entities": self.manager.snapshot(t),
            "events": [{"t": e.t, "kind": e.kind, "text": e.text, "gid": e.gid, "camera": e.camera_id}
                       for e in list(self.feed)[-60:]],
            "stats": {"entities": len(self.manager.entities), "links": len(self.manager.links),
                      "batches": self.batches, "e2e_p50_ms": None if math.isnan(e2e.p50) else round(e2e.p50, 2),
                      "e2e_p95_ms": None if math.isnan(e2e.p95) else round(e2e.p95, 2)},
        }
