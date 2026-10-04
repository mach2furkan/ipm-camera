"""Constant-velocity Kalman filter over x = [u, v, s, r, u', v', s']^T (SORT state space).

u, v  box centre (px)        s  box area (px^2)        r  aspect ratio w/h
u', v', s'  time derivatives in units **per second**.

Time is continuous: every predict takes the real elapsed ``dt`` of the frame that is
being associated. The streaming layer drops frames whenever inference falls behind
(drop-oldest), so the gap between two processed frames is irregular; a per-frame
filter would under-predict motion across a gap and lose the association.

Noise is scale-aware (proportional to the box height, as in DeepSORT): a 15 px person
far from the camera and a 300 px person next to it get proportionally the same
uncertainty, which matters for CCTV where target size spans more than an order of
magnitude in a single scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

_H = np.eye(4, 7)
_I7 = np.eye(7)


@dataclass(frozen=True, slots=True)
class KalmanParams:
    meas_pos: float = 0.05        # std of centre measurement, fraction of box height
    meas_area: float = 0.10       # std of area measurement, fraction of area
    meas_ratio: float = 0.02      # std of aspect-ratio measurement (absolute)
    proc_pos: float = 0.05        # process std of position, fraction of height per sqrt(s)
    proc_vel: float = 0.60        # process std of velocity, heights/s per sqrt(s)
    proc_area: float = 0.05
    proc_area_vel: float = 0.20
    proc_ratio: float = 0.01
    init_vel: float = 2.0         # initial velocity std, heights/s (walking ~1 h/s)
    min_std_px: float = 1.0


def xyxy_to_z(box: F64) -> F64:
    w = max(box[2] - box[0], 1e-3)
    h = max(box[3] - box[1], 1e-3)
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h])


def x_to_xyxy(x: F64) -> F64:
    s = max(float(x[2]), 1e-6)
    r = max(float(x[3]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


def xs_to_xyxy(x: F64) -> F64:
    """Vectorised x_to_xyxy for an (N, 7) state matrix."""
    s = np.maximum(x[:, 2], 1e-6)
    r = np.maximum(x[:, 3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.stack([x[:, 0] - w / 2, x[:, 1] - h / 2, x[:, 0] + w / 2, x[:, 1] + h / 2], axis=1)


def _height(s: npt.ArrayLike, r: npt.ArrayLike) -> F64:
    return np.sqrt(np.maximum(s, 1e-6) / np.maximum(r, 1e-6))


class KalmanBoxFilter:
    def __init__(self, params: KalmanParams | None = None) -> None:
        self.p = params or KalmanParams()

    def initiate(self, box: F64) -> tuple[F64, F64]:
        z = xyxy_to_z(box)
        p = self.p
        h = float(_height(z[2], z[3]))
        mean = np.r_[z, 0.0, 0.0, 0.0]
        std = np.array([
            max(p.min_std_px, 2 * p.meas_pos * h), max(p.min_std_px, 2 * p.meas_pos * h),
            2 * p.meas_area * z[2], 2 * p.meas_ratio,
            p.init_vel * h, p.init_vel * h, p.init_vel * p.meas_area * z[2] * 4,
        ])
        return mean, np.diag(std ** 2)

    def _q(self, s: F64, r: F64, dt: F64) -> F64:
        """(N, 7, 7) process noise, integrated over dt (white-noise random walk)."""
        p = self.p
        h = _height(s, r)
        var = np.stack([
            (p.proc_pos * h) ** 2, (p.proc_pos * h) ** 2, (p.proc_area * s) ** 2,
            np.full_like(s, p.proc_ratio ** 2),
            (p.proc_vel * h) ** 2, (p.proc_vel * h) ** 2, (p.proc_area_vel * s) ** 2,
        ], axis=1)
        var = np.maximum(var, p.min_std_px ** 2 * 1e-2) * np.maximum(dt, 1e-3)[:, None]
        q = np.zeros((len(s), 7, 7))
        idx = np.arange(7)
        q[:, idx, idx] = var
        return q

    def multi_predict(self, means: F64, covs: F64, dts: F64) -> tuple[F64, F64]:
        """Predict N filters at once; ``dts`` is the per-filter elapsed time in seconds."""
        n = len(means)
        if n == 0:
            return means, covs
        means = means.copy()
        # Never let the area extrapolate through zero (SORT safeguard).
        collapse = means[:, 2] + dts * means[:, 6] <= 0
        means[collapse, 6] = 0.0
        f = np.broadcast_to(_I7, (n, 7, 7)).copy()
        f[:, 0, 4] = f[:, 1, 5] = f[:, 2, 6] = dts
        means = np.einsum("nij,nj->ni", f, means)
        covs = np.einsum("nij,njk,nlk->nil", f, covs, f) + self._q(means[:, 2], means[:, 3], dts)
        return means, covs

    def _r(self, z: F64) -> F64:
        p = self.p
        h = float(_height(z[2], z[3]))
        std = np.array([max(p.min_std_px, p.meas_pos * h), max(p.min_std_px, p.meas_pos * h),
                        p.meas_area * z[2], p.meas_ratio])
        return np.diag(std ** 2)

    def update(self, mean: F64, cov: F64, box: F64) -> tuple[F64, F64]:
        z = xyxy_to_z(box)
        s_mat = _H @ cov @ _H.T + self._r(z)
        pht = cov @ _H.T
        k = np.linalg.solve(s_mat, pht.T).T
        innov = z - _H @ mean
        mean = mean + k @ innov
        ikh = _I7 - k @ _H
        cov = ikh @ cov @ ikh.T + k @ self._r(z) @ k.T  # Joseph form: stays symmetric PSD
        return mean, cov

    def mahalanobis(self, mean: F64, cov: F64, boxes: F64) -> F64:
        """Squared Mahalanobis distance of measurements (M, 4 xyxy) to one track (position+scale)."""
        if len(boxes) == 0:
            return np.zeros(0)
        zs = np.array([xyxy_to_z(b) for b in boxes])
        s_mat = _H @ cov @ _H.T + self._r(_H @ mean)
        d = zs - _H @ mean
        chol = np.linalg.cholesky(s_mat)
        sol = np.linalg.solve(chol, d.T)
        return np.sum(sol ** 2, axis=0)
