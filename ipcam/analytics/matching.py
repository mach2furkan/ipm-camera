"""Vectorised association primitives: (buffered) IoU and gated linear assignment."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

CHI2_4DOF_95 = 9.4877  # gating threshold for a 4-D measurement (Kalman, 95 %)


def iou_matrix(a: F64, b: F64, *, buffer: float = 0.0) -> F64:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M).

    ``buffer > 0`` evaluates *buffered* IoU (C-BIoU): each box is grown by ``buffer`` times
    its width/height on every side before overlap is computed. A 15 px pedestrian that
    moves 20 px between two processed frames has IoU 0 with its own prediction; the
    buffered boxes still overlap, so association survives dropped frames and fast motion
    without falling back to an unbounded distance metric.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if buffer:
        a = _grow(a, buffer)
        b = _grow(b, buffer)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0.0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _grow(boxes: F64, k: float) -> F64:
    wh = boxes[:, 2:] - boxes[:, :2]
    return np.concatenate([boxes[:, :2] - k * wh, boxes[:, 2:] + k * wh], axis=1)


def linear_assignment(cost: F64, threshold: float) -> tuple[I64, I64, I64]:
    """Optimal (Hungarian / Jonker-Volgenant) assignment with a cost ceiling.

    Returns ``(matches[K, 2], unmatched_rows, unmatched_cols)``. Pairs above ``threshold``
    are forbidden *before* solving -- filtering afterwards would let the solver trade one
    good match for a forbidden one and then discard both.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), np.int64), np.arange(n), np.arange(m)
    big = threshold + 1e5
    c = np.where(np.isfinite(cost) & (cost <= threshold), cost, big)
    rows, cols = linear_sum_assignment(c)
    keep = c[rows, cols] <= threshold
    matches = np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64)
    um_r = np.setdiff1d(np.arange(n), matches[:, 0], assume_unique=True)
    um_c = np.setdiff1d(np.arange(m), matches[:, 1], assume_unique=True)
    return matches, um_r, um_c
