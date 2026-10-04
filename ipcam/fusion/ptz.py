"""Master-slave PTZ engagement: slew-to-cue with motion lead, then closed-loop visual servoing.

Geometry (world: metres, Z up, bearings CCW from +X)::

    dX, dY, dZ = target - mount
    bearing    = atan2(dY, dX)
    azimuth    = (bearing - theta_0)          mirrored if the head counts clockwise
    tilt       = atan2(dZ, hypot(dX, dY))      elevation = -tilt if "down is positive"
    focal_req  = fill * slant_range * sensor_h / target_height
    zoom       = focal_req / focal_min

Two refinements over the textbook formulas:

* **Lead compensation** -- the head needs a few hundred ms to slew. Aiming at where the
  target *is* means arriving where it *was*. The aim point is iterated as
  ``p + v * T_slew(p)`` (fixed point, converges in 2-3 steps).
* **Gain scheduling** -- servo errors are converted from pixels to *angles* with the
  current field of view, so the same PD gains are stable at 1x and at 30x zoom
  (in pixel space, the effective loop gain would grow linearly with zoom).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PTZMount:
    camera_id: str
    x: float
    y: float
    z: float
    azimuth_zero_bearing_deg: float = 0.0      # world bearing at which the head reads azimuth 0 (theta_0)
    azimuth_clockwise: bool = True             # Hikvision azimuth grows clockwise seen from above
    elevation_down_positive: bool = True       # Hikvision: 0 = horizon, +90 = straight down
    elevation_min_deg: float = -15.0
    elevation_max_deg: float = 90.0
    zoom_min: float = 1.0
    zoom_max: float = 32.0
    focal_min_mm: float = 4.8
    sensor_width_mm: float = 5.37              # 1/2.8" sensor
    sensor_height_mm: float = 3.02
    pan_speed_dps: float = 200.0               # preset (absolute) slew speeds
    tilt_speed_dps: float = 120.0
    zoom_speed_xps: float = 12.0
    command_latency_s: float = 0.08
    target_height_m: float = 1.7
    aim_height_m: float = 1.0                  # aim at the torso, not the feet
    frame_fill: float = 0.45                   # target height as a fraction of frame height
    continuous_dps_at_100: float = 120.0       # continuous-move angular speed at command 100


@dataclass(frozen=True, slots=True)
class PTZPose:
    azimuth_deg: float
    elevation_deg: float
    zoom: float
    range_m: float = float("nan")

    def isapi_units(self) -> tuple[int, int, int]:
        """(azimuth 0.1 deg in [0, 3600), elevation 0.1 deg, absoluteZoom 0.1x)."""
        return (int(round(self.azimuth_deg * 10)) % 3600, int(round(self.elevation_deg * 10)),
                int(round(self.zoom * 10)))


def _wrap360(a: float) -> float:
    return a % 360.0


def _angdiff(a: float, b: float) -> float:
    """Shortest signed difference a - b in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def fov_deg(mount: PTZMount, zoom: float) -> tuple[float, float]:
    f = mount.focal_min_mm * max(zoom, mount.zoom_min)
    return (math.degrees(2 * math.atan(mount.sensor_width_mm / (2 * f))),
            math.degrees(2 * math.atan(mount.sensor_height_mm / (2 * f))))


def aim(mount: PTZMount, target_xy: Sequence[float], target_z: float | None = None) -> PTZPose:
    dz_target = mount.aim_height_m if target_z is None else target_z
    dx, dy, dz = target_xy[0] - mount.x, target_xy[1] - mount.y, dz_target - mount.z
    horiz = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx))
    az = bearing - mount.azimuth_zero_bearing_deg
    if mount.azimuth_clockwise:
        az = -az
    tilt_up = math.degrees(math.atan2(dz, horiz))
    el = -tilt_up if mount.elevation_down_positive else tilt_up
    el = min(max(el, mount.elevation_min_deg), mount.elevation_max_deg)
    slant = math.hypot(horiz, dz)
    f_req = mount.frame_fill * slant * mount.sensor_height_mm / mount.target_height_m
    zoom = min(max(f_req / mount.focal_min_mm, mount.zoom_min), mount.zoom_max)
    return PTZPose(_wrap360(az), el, zoom, slant)


def slew_time(mount: PTZMount, a: PTZPose | None, b: PTZPose) -> float:
    if a is None:
        return 180.0 / mount.pan_speed_dps + mount.command_latency_s   # unknown start: assume worst half-turn
    return mount.command_latency_s + max(
        abs(_angdiff(b.azimuth_deg, a.azimuth_deg)) / mount.pan_speed_dps,
        abs(b.elevation_deg - a.elevation_deg) / mount.tilt_speed_dps,
        abs(b.zoom - a.zoom) / mount.zoom_speed_xps,
    )


def lead_aim(mount: PTZMount, pos: Sequence[float], vel: Sequence[float], current: PTZPose | None,
             *, iterations: int = 3, extra_latency_s: float = 0.0) -> tuple[PTZPose, float]:
    """Aim at the predicted intercept point; returns (pose, expected arrival delay s)."""
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    t_hit = extra_latency_s
    pose = aim(mount, p)
    for _ in range(iterations):
        pose = aim(mount, p + v * t_hit)
        t_hit = slew_time(mount, current, pose) + extra_latency_s
    return pose, t_hit


# --------------------------------------------------------------------------- visual servo

@dataclass(frozen=True, slots=True)
class ServoGains:
    kp: float = 2.2               # (deg/s) per deg of error
    kd: float = 0.12
    deriv_tau_s: float = 0.15     # low-pass on the derivative (detections are noisy)
    deadband_deg: float = 0.25
    feedforward: bool = True      # add the target's own angular velocity


class VisualServo:
    """PD controller in angle space -> Hikvision continuous pan/tilt commands (-100..100).

    Image convention: e = (target_centre - frame_centre) / half_size, in [-1, 1], x right,
    y down. Positive pan command turns right, positive tilt command turns up.
    """

    def __init__(self, mount: PTZMount, gains: ServoGains | None = None) -> None:
        self.mount = mount
        self.g = gains or ServoGains()
        self._prev: tuple[float, float, float] | None = None   # (t, ang_x, ang_y)
        self._d = np.zeros(2)
        self._v_head = np.zeros(2)

    def reset(self) -> None:
        self._prev = None
        self._d = np.zeros(2)
        self._v_head = np.zeros(2)

    def step(self, ex: float, ey: float, zoom: float, t: float) -> tuple[int, int]:
        hfov, vfov = fov_deg(self.mount, zoom)
        ang = np.array([ex * hfov / 2, ey * vfov / 2])
        if self._prev is not None:
            dt = max(t - self._prev[0], 1e-3)
            raw = (ang - np.array(self._prev[1:])) / dt
            a = dt / (self.g.deriv_tau_s + dt)
            self._d = (1 - a) * self._d + a * raw
        self._prev = (t, float(ang[0]), float(ang[1]))
        fb = self.g.kp * ang + self.g.kd * self._d
        fb[np.abs(ang) < self.g.deadband_deg] = 0.0   # deadband on feedback only
        v = fb
        if self.g.feedforward:
            # error rate = target rate - head rate  =>  target rate = error rate + head rate.
            # Feeding the target's own angular rate forward removes the steady-state lag a
            # pure PD loop has against a walking person.
            v = fb + (self._d + self._v_head)
        scale = 100.0 / self.mount.continuous_dps_at_100
        pan = int(np.clip(round(v[0] * scale), -100, 100))
        tilt = int(np.clip(round(-v[1] * scale), -100, 100))   # image y down -> tilt up negative
        self._v_head = np.array([pan, -tilt], dtype=np.float64) / scale
        return pan, tilt


# --------------------------------------------------------------------------- engagement

class PTZController(Protocol):
    async def absolute(self, pose: PTZPose) -> None: ...

    async def continuous(self, pan: int, tilt: int, zoom: int = 0) -> None: ...

    async def stop(self) -> None: ...

    async def status(self) -> PTZPose | None: ...


class ISAPIPTZ:
    """PTZController on top of HikvisionISAPIClient."""

    def __init__(self, client: Any, channel: int = 1) -> None:
        self._c = client
        self._ch = channel

    async def absolute(self, pose: PTZPose) -> None:
        await self._c.ptz_absolute(pose.azimuth_deg, pose.elevation_deg, pose.zoom, channel=self._ch)

    async def continuous(self, pan: int, tilt: int, zoom: int = 0) -> None:
        await self._c.ptz_continuous(pan, tilt, zoom, channel=self._ch)

    async def stop(self) -> None:
        await self._c.ptz_continuous(0, 0, 0, channel=self._ch)

    async def status(self) -> PTZPose | None:
        return await self._c.get_ptz_status(channel=self._ch)


class ISAPIPTZEx:
    """PTZController for heads with ``absoluteEx`` (thermal PT series).

    Absolute moves use float degrees and pre-focus at the slant range of the cue: with a
    150 mm thermal lens the depth of field at 300 m is a few metres, so a head that arrives
    on target but focused at the previous range shows an unusable blur for the 1-2 s its
    contrast autofocus needs -- exactly the time window of the lock-on.
    """

    def __init__(self, camera: Any, channel: int | None = None, *, focus_by_range: bool = True) -> None:
        self._cam = camera            # ipcam.isapi.pt_thermal.PTThermalCamera
        self._ch = channel
        self._focus = focus_by_range

    async def absolute(self, pose: PTZPose) -> None:
        dist = pose.range_m if self._focus and math.isfinite(pose.range_m) and pose.range_m > 1 else None
        await self._cam.move_absolute(pose.azimuth_deg, pose.elevation_deg, zoom=pose.zoom,
                                      object_distance_m=dist, channel=self._ch)

    async def continuous(self, pan: int, tilt: int, zoom: int = 0) -> None:
        await self._cam.continuous(pan, tilt, zoom, channel=self._ch)

    async def stop(self) -> None:
        await self._cam.stop(channel=self._ch)

    async def status(self) -> PTZPose | None:
        p = await self._cam.get_pose(channel=self._ch)
        return PTZPose(p.azimuth, p.elevation, p.zoom)


class EngagementState(str, Enum):
    IDLE = "idle"
    SLEWING = "slewing"
    ACQUIRING = "acquiring"
    LOCKED = "locked"


TargetProvider = Callable[[], "tuple[Sequence[float], Sequence[float]] | None"]
# Detections in the PTZ frame, normalised [0, 1]: (N, 5) x1, y1, x2, y2, conf.
DetectionProvider = Callable[[], "Awaitable[F64 | None] | F64 | None"]


@dataclass
class EngagementStats:
    cues: int = 0
    locks: int = 0
    reacquisitions: int = 0
    cue_to_lock_ms: list[float] = field(default_factory=list)


class SlewToCueEngagement:
    """IDLE -> SLEWING (absolute, lead-compensated) -> ACQUIRING (find the person near the
    frame centre at the expected size) -> LOCKED (visual servo, continuous moves) and back
    to SLEWING whenever the lock is lost or the PTZ drifts away from the fused position."""

    def __init__(
        self,
        ptz: PTZController,
        mount: PTZMount,
        target: TargetProvider,
        detections: DetectionProvider,
        *,
        servo: VisualServo | None = None,
        rate_hz: float = 10.0,
        acquire_timeout_s: float = 1.5,
        lost_timeout_s: float = 1.0,
        drift_tolerance: float = 1.0,      # re-cue if fused target is > this many half-FOVs away
        keepalive_s: float = 0.5,
        on_state: Callable[[EngagementState], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ptz = ptz
        self.mount = mount
        self.target = target
        self.detections = detections
        self.servo = servo or VisualServo(mount)
        self.period = 1.0 / rate_hz
        self.acquire_timeout = acquire_timeout_s
        self.lost_timeout = lost_timeout_s
        self.drift_tol = drift_tolerance
        self.keepalive = keepalive_s
        self.on_state = on_state
        self.clock = clock
        self.state = EngagementState.IDLE
        self.pose: PTZPose | None = None
        self.stats = EngagementStats()
        self._lock_box: F64 | None = None
        self._last_seen = 0.0
        self._deadline = 0.0
        self._cue_t = 0.0
        self._last_cmd: tuple[int, int] = (0, 0)
        self._last_cmd_t = 0.0
        self._task: asyncio.Task[None] | None = None

    def _set(self, s: EngagementState) -> None:
        if s is not self.state:
            self.state = s
            if self.on_state:
                self.on_state(s)

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._set(EngagementState.SLEWING)
            self._task = asyncio.create_task(self.run(), name=f"ptz-{self.mount.camera_id}")
        return self._task

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.ptz.stop()
        self._set(EngagementState.IDLE)

    async def run(self) -> None:
        try:
            while True:
                t0 = self.clock()
                await self.step()
                await asyncio.sleep(max(0.0, self.period - (self.clock() - t0)))
        finally:
            try:
                await asyncio.shield(self.ptz.stop())
            except Exception:  # noqa: BLE001
                log.debug("ptz stop failed", exc_info=True)

    async def _dets(self) -> F64 | None:
        d = self.detections()
        if asyncio.iscoroutine(d) or isinstance(d, asyncio.Future):
            d = await d
        return None if d is None else np.asarray(d, dtype=np.float64).reshape(-1, 5)

    async def step(self) -> None:
        now = self.clock()
        tgt = self.target()
        if tgt is None and self.state is not EngagementState.LOCKED:
            if self.state is not EngagementState.IDLE:
                await self.ptz.stop()
            self._set(EngagementState.IDLE)
            return
        # While LOCKED the visual servo needs no fused position: the PTZ keeps following
        # the person into blind spots no fixed camera covers -- where it matters most.
        pos, vel = tgt if tgt is not None else (None, None)
        if self.state in (EngagementState.IDLE, EngagementState.SLEWING):
            pose, t_hit = lead_aim(self.mount, pos, vel, self.pose)
            await self.ptz.absolute(pose)
            self.pose = pose
            self.servo.reset()
            self._lock_box = None
            self._cue_t = now
            self._deadline = now + t_hit + self.acquire_timeout
            self.stats.cues += 1
            self._set(EngagementState.ACQUIRING)
            return

        dets = await self._dets()
        zoom = self.pose.zoom if self.pose else self.mount.zoom_min
        if self.state is EngagementState.ACQUIRING:
            pick = self._pick(dets, expect_center=True)
            if pick is not None:
                self._lock_box = pick
                self._last_seen = now
                self.stats.locks += 1
                self.stats.cue_to_lock_ms.append((now - self._cue_t) * 1000.0)
                self._set(EngagementState.LOCKED)
            elif now > self._deadline:
                self.stats.reacquisitions += 1
                self._set(EngagementState.SLEWING)
            return

        # LOCKED
        pick = self._pick(dets, expect_center=False)
        if pick is None:
            if now - self._last_seen > self.lost_timeout:
                await self.ptz.stop()
                self.stats.reacquisitions += 1
                self._set(EngagementState.SLEWING if pos is not None else EngagementState.IDLE)
            elif self._last_cmd != (0, 0):
                await self.ptz.stop()      # never keep spinning blind
                self._last_cmd = (0, 0)
            return
        self._lock_box = pick
        self._last_seen = now
        cx, cy = (pick[0] + pick[2]) / 2, (pick[1] + pick[3]) / 2
        pan, tilt = self.servo.step(2 * cx - 1, 2 * cy - 1, zoom, now)
        if (pan, tilt) != self._last_cmd or now - self._last_cmd_t > self.keepalive:
            await self.ptz.continuous(pan, tilt, 0)
            self._last_cmd, self._last_cmd_t = (pan, tilt), now
        if self.pose is not None:
            # Dead-reckon the head pose for drift checks (and for the next cue's slew estimate).
            dps = self.mount.continuous_dps_at_100 / 100.0
            sign = 1 if self.mount.azimuth_clockwise else -1
            self.pose = PTZPose(_wrap360(self.pose.azimuth_deg + sign * pan * dps * self.period),
                                self.pose.elevation_deg - tilt * dps * self.period * (1 if self.mount.elevation_down_positive else -1),
                                self.pose.zoom)
            if pos is not None:
                expected = aim(self.mount, pos)
                hfov, _ = fov_deg(self.mount, zoom)
                if abs(_angdiff(expected.azimuth_deg, self.pose.azimuth_deg)) > self.drift_tol * hfov / 2 * 3:
                    self.stats.reacquisitions += 1   # locked onto the wrong person: re-cue
                    self._set(EngagementState.SLEWING)

    def _pick(self, dets: F64 | None, *, expect_center: bool) -> F64 | None:
        if dets is None or len(dets) == 0:
            return None
        c = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], axis=1)
        h = dets[:, 3] - dets[:, 1]
        if expect_center or self._lock_box is None:
            # Nearest to the centre with a size compatible with the commanded zoom.
            size_pen = np.abs(np.log(np.maximum(h, 1e-3) / self.mount.frame_fill))
            score = np.hypot(c[:, 0] - 0.5, c[:, 1] - 0.5) + 0.25 * size_pen
            i = int(np.argmin(score))
            return dets[i, :4] if score[i] < 0.6 else None
        lb = self._lock_box
        lc = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
        d = np.hypot(c[:, 0] - lc[0], c[:, 1] - lc[1])
        i = int(np.argmin(d))
        return dets[i, :4] if d[i] < max(0.25, 1.5 * (lb[3] - lb[1])) else None
