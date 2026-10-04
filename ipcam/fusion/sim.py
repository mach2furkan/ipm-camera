"""Synthetic multi-camera site for validating fusion, Re-ID handover and PTZ engagement.

Geometry is real pinhole projection (cameras at 6 m, 25 deg down-tilt), so homography
uncertainty, foreshortening and the horizon behave as on site. Re-ID embeddings model
the two effects that make cross-camera matching hard:

* a per-camera *domain shift* shared by everyone seen through that camera (lighting,
  white balance, viewing angle), and
* per-crop noise,

calibrated so that same-person / cross-camera cosine distances land around 0.15-0.25 and
different people around 0.8-1.0 -- the regime the 0.28 / 0.40 thresholds are designed for.
Optional look-alike pairs (similar clothing) create hard negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .homography import CameraCalibration, pinhole_homography, pinhole_project
from .ptz import PTZMount, PTZPose, _angdiff, _wrap360, fov_deg, slew_time
from .reid import l2_normalize
from .topology import CameraTopology
from .transport.codec import DetectionEvent, FrameBatch

F64 = npt.NDArray[np.float64]


@dataclass
class SimCamera:
    camera_id: str
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float = 25.0
    width: int = 1280
    height: int = 720
    f: float = 800.0
    max_range_m: float = 38.0

    def _kw(self) -> dict[str, float | tuple[float, float, float]]:
        return dict(fx=self.f, fy=self.f, cx=self.width / 2, cy=self.height / 2, position=self.position,
                    yaw_deg=self.yaw_deg, pitch_deg=self.pitch_deg)

    def project(self, xyz: npt.ArrayLike) -> F64:
        return pinhole_project(world_xyz=xyz, **self._kw())  # type: ignore[arg-type]

    def calibration(self, *, survey_noise_m: float = 0.0, pixel_noise: float = 0.0,
                    rng: np.random.Generator | None = None, rows: tuple[float, ...] = (0.24, 0.45, 0.7, 0.95)
                    ) -> CameraCalibration:
        """Calibrate like a field engineer: surveyed ground markers, noisy clicks.

        ``rows`` (fractions of image height) should span the working range: the default
        puts markers from ~38 m down to ~5 m. Near-field-only markers (e.g. rows=(0.6, 0.95))
        reproduce the classic failure of a great RMSE with metres of error far away.
        """
        rng = rng or np.random.default_rng(0)
        hw = np.linalg.inv(pinhole_homography(**self._kw()))  # type: ignore[arg-type]
        px = np.array([[self.width * a, self.height * b] for a in (0.15, 0.5, 0.85) for b in rows])
        q = np.hstack([px, np.ones((len(px), 1))]) @ hw.T
        world = q[:, :2] / q[:, 2:3]
        world_meas = world + rng.normal(0, survey_noise_m, world.shape)
        px_meas = px + rng.normal(0, pixel_noise, px.shape)
        return CameraCalibration.from_points(self.camera_id, px_meas, world_meas, (self.width, self.height),
                                             ransac_threshold_m=None, position=self.position)

    def observe(self, xy: F64, height_m: float = 1.7) -> tuple[float, float, float, float] | None:
        rng_m = math.hypot(xy[0] - self.position[0], xy[1] - self.position[1])
        if rng_m > self.max_range_m:
            return None
        foot = self.project([xy[0], xy[1], 0.0])[0]
        head = self.project([xy[0], xy[1], height_m])[0]
        if not (np.all(np.isfinite(foot)) and np.all(np.isfinite(head))):
            return None
        h = foot[1] - head[1]
        if h < 12:
            return None
        w = 0.4 * h
        box = (foot[0] - w / 2, head[1], foot[0] + w / 2, foot[1])
        if box[0] < 0 or box[2] > self.width or box[3] > self.height or box[1] < 0:
            return None
        return box


@dataclass
class SimPerson:
    pid: int
    waypoints: list[tuple[float, float]]
    speed: float
    t_start: float
    embedding: F64
    height_m: float = 1.75
    _seg: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        pts = np.asarray(self.waypoints, dtype=np.float64)
        self._seg = [0.0] + list(np.cumsum(np.hypot(*np.diff(pts, axis=0).T)))

    @property
    def t_end(self) -> float:
        return self.t_start + self._seg[-1] / self.speed

    def state(self, t: float) -> tuple[F64, F64] | None:
        if not self.t_start <= t <= self.t_end:
            return None
        s = (t - self.t_start) * self.speed
        i = min(int(np.searchsorted(self._seg, s, side="right")) - 1, len(self.waypoints) - 2)
        a, b = np.asarray(self.waypoints[i]), np.asarray(self.waypoints[i + 1])
        seg_len = self._seg[i + 1] - self._seg[i]
        u = (s - self._seg[i]) / seg_len if seg_len > 0 else 0.0
        d = (b - a) / seg_len if seg_len > 0 else np.zeros(2)
        return a + (b - a) * u, d * self.speed


@dataclass
class SimDetection:
    local_id: int
    pid: int
    box: tuple[float, float, float, float]
    conf: float
    embedding: npt.NDArray[np.float32] | None


class SimWorld:
    def __init__(self, cameras: list[SimCamera], persons: list[SimPerson], *, seed: int = 0, dim: int = 512,
                 domain_shift: float = 0.35, crop_noise: float = 0.30, pixel_noise: float = 1.5,
                 miss_prob: float = 0.05, reid_min_px: float = 40.0, reid_interval_s: float = 1.0) -> None:
        self.cameras = {c.camera_id: c for c in cameras}
        self.persons = persons
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.shift = {c.camera_id: l2_normalize(self.rng.normal(size=dim)).astype(np.float64) for c in cameras}
        self.domain_shift = domain_shift
        self.crop_noise = crop_noise
        self.pixel_noise = pixel_noise
        self.miss_prob = miss_prob
        self.reid_min_px = reid_min_px
        self.reid_interval = reid_interval_s
        self._local: dict[tuple[str, int], int] = {}
        self._next_local: dict[str, int] = {c.camera_id: 1 for c in cameras}
        self._last_reid: dict[tuple[str, int], float] = {}
        self.calibrations = {c.camera_id: c.calibration(survey_noise_m=0.05, pixel_noise=1.0, rng=self.rng)
                             for c in cameras}
        self.gt: dict[tuple[str, int], int] = {}          # (camera, local id) -> pid

    def embedding(self, pid: int, camera_id: str) -> npt.NDArray[np.float32]:
        p = self.persons[pid]
        noise = self.rng.normal(size=self.dim) * (self.crop_noise / math.sqrt(self.dim))
        return l2_normalize(p.embedding + self.domain_shift * self.shift[camera_id] + noise)

    def step(self, t: float) -> dict[str, tuple[list[SimDetection], list[int]]]:
        out: dict[str, tuple[list[SimDetection], list[int]]] = {}
        for cid, cam in self.cameras.items():
            dets: list[SimDetection] = []
            visible: set[int] = set()
            for p in self.persons:
                st = p.state(t)
                if st is None:
                    continue
                box = cam.observe(st[0], p.height_m)
                if box is None:
                    continue
                visible.add(p.pid)
                key = (cid, p.pid)
                if key not in self._local:
                    self._local[key] = self._next_local[cid]
                    self.gt[(cid, self._next_local[cid])] = p.pid
                    self._next_local[cid] += 1
                if self.rng.random() < self.miss_prob:
                    continue
                n = self.rng.normal(0, self.pixel_noise, 4)
                b = (box[0] + n[0], box[1] + n[1], box[2] + n[2], box[3] + n[3])
                emb = None
                if box[3] - box[1] >= self.reid_min_px and t - self._last_reid.get(key, -1e9) >= self.reid_interval:
                    emb = self.embedding(p.pid, cid)
                    self._last_reid[key] = t
                dets.append(SimDetection(self._local[key], p.pid, b, float(self.rng.uniform(0.7, 0.95)), emb))
            ended = []
            for (c, pid), lid in list(self._local.items()):
                if c == cid and pid not in visible:
                    ended.append(lid)
                    del self._local[(c, pid)]
                    self._last_reid.pop((c, pid), None)
            out[cid] = (dets, ended)
        return out

    def frame_batch(self, camera_id: str, t: float, dets: list[SimDetection], ended: list[int], seq: int,
                    *, t_offset: float = 0.0) -> FrameBatch:
        cal = self.calibrations[camera_id]
        ts = int((t + t_offset) * 1e9)
        events = []
        for d in dets:
            foot = ((d.box[0] + d.box[2]) / 2, d.box[3])
            w = cal.to_world([foot])[0]
            valid = bool(np.all(np.isfinite(w)))
            h = d.box[3] - d.box[1]
            cov = cal.world_covariance(foot[0], foot[1], max(1.0, 0.04 * h), max(1.0, 0.06 * h)) if valid else None
            events.append(DetectionEvent(
                camera_id, ts, d.local_id, (*d.box, d.conf),  # type: ignore[arg-type]
                (float(w[0]), float(w[1]), 0.0) if valid else None, d.embedding, False,
                (float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])) if cov is not None else None, valid))
        return FrameBatch(camera_id, ts, seq, events, ended)


def default_site(seed: int = 0, *, n_people: int = 12, lookalikes: int = 2, duration_s: float = 150.0,
                 dim: int = 512) -> tuple[list[SimCamera], list[SimPerson], CameraTopology, PTZMount]:
    """130 x 40 m site: A and B overlap around x = 25-35 m, a ~45 m blind gap separates B and C."""
    rng = np.random.default_rng(seed)
    cams = [
        SimCamera("cam-A", (0.0, 20.0, 6.0), 0.0),
        SimCamera("cam-B", (60.0, 20.0, 6.0), 180.0),
        SimCamera("cam-C", (95.0, 20.0, 6.0), 0.0),
    ]
    topo = CameraTopology(default_t_max_s=90.0, allow_unknown_pairs=False)
    topo.add_overlap("cam-A", "cam-B")
    topo.add_transition("cam-A", "cam-B", mu_s=2.0, sigma_s=2.0, t_min_s=0.0, t_max_s=20.0)
    topo.add_transition("cam-B", "cam-C", mu_s=32.0, sigma_s=6.0, t_min_s=11.0, t_max_s=90.0)
    topo.add_transition("cam-A", "cam-C", mu_s=55.0, sigma_s=10.0, t_min_s=18.0, t_max_s=120.0)
    persons: list[SimPerson] = []
    bases = [l2_normalize(rng.normal(size=dim)).astype(np.float64) for _ in range(n_people)]
    for k in range(lookalikes):                       # hard negatives: similar clothing
        i, j = 2 * k, 2 * k + 1
        if j < n_people:
            bases[j] = l2_normalize(bases[i] + 1.1 * l2_normalize(rng.normal(size=dim))).astype(np.float64)
    for pid in range(n_people):
        y0, y1 = rng.uniform(13, 27, 2)
        forward = rng.random() < 0.75
        leaves_in_gap = rng.random() < 0.2
        x_end = rng.uniform(65, 90) if leaves_in_gap else 128.0
        pts = [(4.0, y0), (45.0, (y0 + y1) / 2), (x_end, y1)]
        if leaves_in_gap:
            pts.append((x_end, 39.5))                 # exits the site through the gap
        if not forward:
            pts = list(reversed(pts))
        persons.append(SimPerson(pid, pts, float(rng.uniform(1.1, 1.7)),
                                 float(rng.uniform(0, duration_s * 0.5)), bases[pid]))
    ptz = PTZMount("ptz-1", 60.0, 38.0, 8.0, azimuth_zero_bearing_deg=-90.0)
    return cams, persons, topo, ptz


class SimPTZ:
    """Kinematic PTZ head: absolute moves at preset speed, continuous moves integrate."""

    def __init__(self, mount: PTZMount, clock: object) -> None:
        self.mount = mount
        self.clock = clock  # callable -> seconds
        self.pose = PTZPose(0.0, 10.0, 1.0)
        self._target: PTZPose | None = None
        self._t0 = 0.0
        self._from = self.pose
        self._dur = 0.0
        self._vel = (0, 0)
        self._t_last = 0.0
        self.commands = 0

    def _now(self) -> float:
        return self.clock()  # type: ignore[operator]

    def advance(self) -> PTZPose:
        now = self._now()
        if self._target is not None:
            u = min(1.0, (now - self._t0) / self._dur) if self._dur > 0 else 1.0
            a, b = self._from, self._target
            self.pose = PTZPose(_wrap360(a.azimuth_deg + _angdiff(b.azimuth_deg, a.azimuth_deg) * u),
                                a.elevation_deg + (b.elevation_deg - a.elevation_deg) * u,
                                a.zoom + (b.zoom - a.zoom) * u)
            if u >= 1.0:
                self._target = None
        elif self._vel != (0, 0):
            dt = now - self._t_last
            dps = self.mount.continuous_dps_at_100 / 100.0
            sign = 1 if self.mount.azimuth_clockwise else -1
            self.pose = PTZPose(_wrap360(self.pose.azimuth_deg + sign * self._vel[0] * dps * dt),
                                self.pose.elevation_deg - self._vel[1] * dps * dt, self.pose.zoom)
        self._t_last = now
        return self.pose

    async def absolute(self, pose: PTZPose) -> None:
        self.advance()
        self.commands += 1
        self._from, self._target = self.pose, pose
        self._t0 = self._now() + self.mount.command_latency_s
        self._dur = slew_time(self.mount, self.pose, pose) - self.mount.command_latency_s
        self._vel = (0, 0)

    async def continuous(self, pan: int, tilt: int, zoom: int = 0) -> None:
        self.advance()
        self.commands += 1
        self._target = None
        self._vel = (pan, tilt)

    async def stop(self) -> None:
        await self.continuous(0, 0, 0)

    async def status(self) -> PTZPose | None:
        return self.advance()

    def detections(self, target_xy: F64 | None, *, height_m: float = 1.75, noise: float = 0.004,
                   rng: np.random.Generator | None = None) -> F64:
        """Normalised detections of the target in the PTZ frame at the current pose."""
        pose = self.advance()
        if target_xy is None:
            return np.zeros((0, 5))
        m = self.mount
        dx, dy = target_xy[0] - m.x, target_xy[1] - m.y
        horiz = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        az = bearing - m.azimuth_zero_bearing_deg
        az = -az if m.azimuth_clockwise else az
        hfov, vfov = fov_deg(m, pose.zoom)
        ex = _angdiff(_wrap360(az), pose.azimuth_deg) * (1 if m.azimuth_clockwise else -1) / (hfov / 2)
        el_c = math.degrees(math.atan2(m.z - m.aim_height_m, horiz))
        ey = (el_c - pose.elevation_deg) / (vfov / 2)
        h = math.degrees(math.atan2(height_m, math.hypot(horiz, m.z - m.aim_height_m))) / vfov
        cx, cy = 0.5 + ex / 2, 0.5 + ey / 2
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            return np.zeros((0, 5))
        r = rng or np.random.default_rng()
        cx += r.normal(0, noise)
        cy += r.normal(0, noise)
        w = 0.4 * h * (9 / 16)
        return np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9]])
