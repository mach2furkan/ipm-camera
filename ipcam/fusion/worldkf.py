"""Metric constant-velocity Kalman filter for global entities: x = [X, Y, VX, VY] (m, m/s).

Measurements from several cameras update the same filter sequentially, each with its own
2x2 covariance from homography uncertainty propagation. That *is* covariance-weighted
fusion: a sharp nearby camera (small R) pulls the estimate harder than a distant,
grazing-angle one, without any hand-tuned per-camera weights.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


class WorldKF:
    __slots__ = ("accel_std", "cov", "mean", "t")

    def __init__(self, xy: F64, r: F64, t: float, *, accel_std: float = 1.5, init_speed_std: float = 1.5) -> None:
        self.mean = np.array([xy[0], xy[1], 0.0, 0.0])
        self.cov = np.zeros((4, 4))
        self.cov[:2, :2] = r
        self.cov[2:, 2:] = np.eye(2) * init_speed_std ** 2
        self.t = t
        self.accel_std = accel_std

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        # Discrete white-noise acceleration model.
        q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * self.accel_std ** 2
        q = np.zeros((4, 4))
        q[np.ix_([0, 2], [0, 2])] = q1
        q[np.ix_([1, 3], [1, 3])] = q1
        self.mean = f @ self.mean
        self.cov = f @ self.cov @ f.T + q
        self.t = t

    def peek(self, t: float) -> F64:
        """Predicted position at ``t`` without mutating the filter."""
        dt = max(0.0, t - self.t)
        return self.mean[:2] + self.mean[2:] * dt

    def innovation(self, z: F64, r: F64) -> tuple[F64, F64]:
        y = z - _H @ self.mean
        s = _H @ self.cov @ _H.T + r
        return y, s

    def mahalanobis2(self, z: F64, r: F64) -> float:
        y, s = self.innovation(z, r)
        return float(y @ np.linalg.solve(s, y))

    def update(self, z: F64, r: F64) -> None:
        y, s = self.innovation(z, r)
        k = self.cov @ _H.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ y
        ikh = np.eye(4) - k @ _H
        self.cov = ikh @ self.cov @ ikh.T + k @ r @ k.T

    @property
    def position(self) -> F64:
        return self.mean[:2].copy()

    @property
    def velocity(self) -> F64:
        return self.mean[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.hypot(self.mean[2], self.mean[3]))
