"""Planar homography: image foot points -> metric ground-plane coordinates.

* Normalised DLT (Hartley): both point sets are translated to their centroid and scaled
  to mean distance sqrt(2) before the SVD. Without it the design matrix mixes values of
  ~1 and ~10^6 (pixel^2) and the solution degrades badly with real calibration noise.
* RANSAC for surveyed point sets containing a mis-clicked or mis-measured marker.
* First-order uncertainty propagation: the 2x2 world covariance of a projected foot point
  is ``J diag(sigma_u^2, sigma_v^2) J^T`` with J the Jacobian of the homography. Far
  targets (small boxes, grazing view angle) automatically get large covariances, which
  is exactly what weights near/sharp cameras higher in multi-camera fusion.
* Horizon guard: pixels on or above the vanishing line map to infinity or to the wrong
  side of the camera; they are rejected instead of producing teleporting targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def _normalizer(pts: F64) -> F64:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _h(pts: F64) -> F64:
    return np.hstack([pts, np.ones((len(pts), 1))])


def fit_homography(src: npt.ArrayLike, dst: npt.ArrayLike) -> F64:
    """Least-squares homography mapping ``src`` (N>=4, 2) onto ``dst`` (N, 2)."""
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    if s.shape != d.shape or s.shape[0] < 4 or s.shape[1] != 2:
        raise ValueError("need at least 4 point correspondences of shape (N, 2)")
    ts, td = _normalizer(s), _normalizer(d)
    sn = (_h(s) @ ts.T)[:, :2]
    dn = (_h(d) @ td.T)[:, :2]
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sv, vt = np.linalg.svd(np.asarray(rows))
    if len(sv) >= 8 and sv[7] < 1e-12 * max(sv[0], 1e-300):
        raise ValueError("degenerate configuration (collinear calibration points?)")
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h / np.linalg.norm(h)


def convex_hull(pts: npt.ArrayLike) -> F64:
    """Andrew's monotone chain; returns hull vertices counter-clockwise."""
    p = sorted(map(tuple, np.asarray(pts, dtype=np.float64)))
    if len(p) <= 2:
        return np.asarray(p)

    def cross(o: tuple[float, ...], a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, ...]] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[tuple[float, ...]] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1])


def project(h: F64, pts: npt.ArrayLike) -> F64:
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    q = _h(p) @ h.T
    return q[:, :2] / q[:, 2:3]


def fit_homography_ransac(src: npt.ArrayLike, dst: npt.ArrayLike, *, threshold: float = 0.3,
                          iterations: int = 2000, seed: int = 0) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Robust fit; ``threshold`` is the max transfer error in destination units (metres)."""
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    n = len(s)
    if n < 4:
        raise ValueError("need at least 4 points")
    if n == 4:
        return fit_homography(s, d), np.ones(4, bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iterations):
        idx = rng.choice(n, 4, replace=False)
        try:
            h = fit_homography(s[idx], d[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        with np.errstate(all="ignore"):
            err = np.linalg.norm(project(h, s) - d, axis=1)
        inl = np.nan_to_num(err, nan=np.inf) < threshold
        if inl.sum() > best.sum():
            best = inl
            if best.all():
                break
    if best.sum() < 4:
        raise ValueError("RANSAC found no consistent homography")
    return fit_homography(s[best], d[best]), best


def jacobian(h: F64, u: float, v: float) -> F64:
    """d(X, Y) / d(u, v) of the projective map at pixel (u, v)."""
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    x = h[0, 0] * u + h[0, 1] * v + h[0, 2]
    y = h[1, 0] * u + h[1, 1] * v + h[1, 2]
    return np.array([
        [(h[0, 0] * w - x * h[2, 0]) / w ** 2, (h[0, 1] * w - x * h[2, 1]) / w ** 2],
        [(h[1, 0] * w - y * h[2, 0]) / w ** 2, (h[1, 1] * w - y * h[2, 1]) / w ** 2],
    ])


@dataclass
class CameraCalibration:
    """Pixel -> ground-plane calibration of one fixed camera."""

    camera_id: str
    h: F64                                   # pixel -> world (metres)
    image_size: tuple[int, int]              # (width, height) of the frame the pixels refer to
    position: tuple[float, float, float] | None = None   # camera mount (world), for the C2 map
    rmse_m: float = float("nan")
    horizon_margin: float = 0.02             # reject pixels within this fraction of the horizon
    # Convex hull of the surveyed markers in pixels. A low reprojection RMSE only certifies
    # the region *inside* it; outside, errors grow quickly (a near-field calibration with
    # 3 cm RMSE can be ~2 m off at 30 m).
    calibrated_hull: F64 | None = None
    _h_inv: F64 = field(init=False, repr=False)
    _w_sign: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=np.float64)
        self._h_inv = np.linalg.inv(self.h)
        # Sign of the projective denominator on the bottom row (always on the ground).
        w, hh = self.image_size
        self._w_sign = float(np.sign(self.h[2] @ np.array([w / 2, hh - 1, 1.0])))

    @classmethod
    def from_points(cls, camera_id: str, pixels: npt.ArrayLike, world: npt.ArrayLike,
                    image_size: tuple[int, int], *, ransac_threshold_m: float | None = 0.3,
                    position: tuple[float, float, float] | None = None) -> CameraCalibration:
        px = np.asarray(pixels, dtype=np.float64)
        wd = np.asarray(world, dtype=np.float64)
        if ransac_threshold_m is not None and len(px) > 4:
            h, inl = fit_homography_ransac(px, wd, threshold=ransac_threshold_m)
        else:
            h, inl = fit_homography(px, wd), np.ones(len(px), bool)
        err = np.linalg.norm(project(h, px[inl]) - wd[inl], axis=1)
        return cls(camera_id, h, image_size, position, rmse_m=float(np.sqrt(np.mean(err ** 2))),
                   calibrated_hull=convex_hull(px[inl]))

    def in_calibrated_region(self, px: npt.ArrayLike, *, margin_px: float = 0.0) -> npt.NDArray[np.bool_]:
        """True for pixels inside the surveyed markers' hull (interpolation, not extrapolation)."""
        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        if self.calibrated_hull is None or len(self.calibrated_hull) < 3:
            return np.ones(len(p), bool)
        from ..analytics.geometry import points_in_polygon

        hull = self.calibrated_hull
        if margin_px:
            c = hull.mean(axis=0)
            d = hull - c
            hull = c + d * (1 + margin_px / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9))
        return points_in_polygon(p, hull)

    def valid_pixels(self, px: F64) -> npt.NDArray[np.bool_]:
        """True where the pixel lies safely below the horizon line."""
        p = np.atleast_2d(px)
        w = _h(p) @ self.h[2]
        scale = np.abs(self.h[2] @ np.array([self.image_size[0] / 2, self.image_size[1] - 1, 1.0]))
        return w * self._w_sign > self.horizon_margin * scale

    def to_world(self, px: npt.ArrayLike) -> F64:
        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        out = project(self.h, p)
        out[~self.valid_pixels(p)] = np.nan
        return out

    def to_pixel(self, world: npt.ArrayLike) -> F64:
        return project(self._h_inv, world)

    def world_covariance(self, u: float, v: float, sigma_u: float, sigma_v: float) -> F64:
        j = jacobian(self.h, u, v)
        return j @ np.diag([sigma_u ** 2, sigma_v ** 2]) @ j.T

    def ground_footprint(self, *, samples: int = 16, top_fraction: float | None = None,
                         max_range_m: float | None = None) -> F64:
        """World polygon of the visible ground (image border clipped below the horizon).

        ``max_range_m`` clips the far edge to the camera's useful range (needs ``position``);
        without it a slightly tilted camera's footprint runs out to the horizon.
        """
        w, hh = self.image_size
        # Find the highest image row that still maps validly, column by column.
        cols = np.linspace(0, w - 1, samples)
        border: list[tuple[float, float]] = []
        rows = np.linspace(0, hh - 1, 256)
        top_rows = []
        for c in cols:
            pix = np.stack([np.full_like(rows, c), rows], axis=1)
            ok = self.valid_pixels(pix)
            if max_range_m is not None and self.position is not None:
                with np.errstate(all="ignore"):
                    wpts = project(self.h, pix)
                dist = np.hypot(wpts[:, 0] - self.position[0], wpts[:, 1] - self.position[1])
                ok &= np.nan_to_num(dist, nan=np.inf) <= max_range_m
            r = rows[np.argmax(ok)] if ok.any() else hh - 1
            if top_fraction is not None:
                r = max(r, hh * top_fraction)
            top_rows.append(r)
        border += [(c, r) for c, r in zip(cols, top_rows)]
        border += [(w - 1, hh - 1), (0, hh - 1)]
        return project(self.h, np.array(border))

    def to_json(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "h": self.h.tolist(), "image_size": list(self.image_size),
                "position": list(self.position) if self.position else None, "rmse_m": self.rmse_m,
                "calibrated_hull": self.calibrated_hull.tolist() if self.calibrated_hull is not None else None}

    @classmethod
    def from_json(cls, data: dict[str, Any] | str | Path) -> CameraCalibration:
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if "h" in data:
            pos = tuple(data["position"]) if data.get("position") else None
            hull = np.array(data["calibrated_hull"]) if data.get("calibrated_hull") else None
            return cls(data["camera_id"], np.array(data["h"]), tuple(data["image_size"]), pos,  # type: ignore[arg-type]
                       rmse_m=float(data.get("rmse_m", float("nan"))), calibrated_hull=hull)
        return cls.from_points(data["camera_id"], data["pixels"], data["world"], tuple(data["image_size"]),  # type: ignore[arg-type]
                               position=tuple(data["position"]) if data.get("position") else None)


def pinhole_homography(*, fx: float, fy: float, cx: float, cy: float, position: tuple[float, float, float],
                       yaw_deg: float, pitch_deg: float) -> F64:
    """Ground-plane (Z=0) homography of an ideal pinhole camera; world -> pixel.

    Used by simulations and tests to create ground-truth geometry. ``yaw`` is the
    bearing of the optical axis in the world XY plane (0 = +X, CCW), ``pitch`` the
    depression angle below the horizon.
    """
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd])          # world -> camera rotation (rows = camera axes)
    c = np.asarray(position, dtype=np.float64)
    t = -r @ c
    p = k @ np.hstack([r, t[:, None]])        # 3x4
    return p[:, [0, 1, 3]]                    # drop the Z column: ground plane


def pinhole_project(*, fx: float, fy: float, cx: float, cy: float, position: tuple[float, float, float],
                    yaw_deg: float, pitch_deg: float, world_xyz: npt.ArrayLike) -> F64:
    """Full 3-D projection (for head points etc.) of the same ideal camera."""
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd])
    pts = np.atleast_2d(np.asarray(world_xyz, dtype=np.float64)) - np.asarray(position)
    cam = pts @ r.T
    uvw = cam @ k.T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = uvw[:, :2] / uvw[:, 2:3]
    out[cam[:, 2] <= 0] = np.nan
    return out
