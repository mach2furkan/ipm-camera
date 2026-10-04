"""Person re-identification: selective key-crop gating, embedding extraction, galleries.

Re-ID inference is the most expensive per-target operation, and embeddings of blurred,
tiny or half-occluded crops are worse than none (they pull identity prototypes toward
the background). Crops are therefore admitted only when:

* the box is at least 64 x 128 px,
* the Laplacian variance (focus measure) is >= 120,
* less than 25 % of the box is covered by targets standing *in front* of it,
* at most one key-crop per track per second.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


def l2_normalize(x: npt.ArrayLike, axis: int = -1) -> F32:
    a = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, 1e-12)


def laplacian_variance(gray: npt.NDArray[Any]) -> float:
    """Variance of the 4-neighbour Laplacian (higher = sharper)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR -> luma
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def occlusion_scores(boxes: F64) -> F64:
    """Fraction of each box covered by boxes whose foot is lower in the image (closer).

    For a ground-standing scene a lower foot point means nearer to the camera, i.e. the
    occluder. Overlaps from several occluders are capped at 1.
    """
    b = np.asarray(boxes, dtype=np.float64)
    n = len(b)
    if n == 0:
        return np.zeros(0)
    tl = np.maximum(b[:, None, :2], b[None, :, :2])
    br = np.minimum(b[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    in_front = b[None, :, 3] > b[:, None, 3]
    np.fill_diagonal(in_front, False)
    area = np.maximum(np.prod(b[:, 2:4] - b[:, :2], axis=1), 1e-9)
    return np.minimum((inter * in_front).sum(axis=1) / area, 1.0)


@dataclass(frozen=True, slots=True)
class ReIDGateConfig:
    min_width: int = 64
    min_height: int = 128
    min_laplacian_var: float = 120.0
    max_occlusion: float = 0.25
    min_interval_s: float = 1.0
    edge_margin_px: int = 4          # crops touching the frame border are truncated bodies


class ReIDGate:
    def __init__(self, config: ReIDGateConfig | None = None) -> None:
        self.cfg = config or ReIDGateConfig()
        self._last: dict[int, float] = {}
        self.stats: dict[str, int] = {}

    def _reject(self, reason: str) -> tuple[bool, str]:
        self.stats[reason] = self.stats.get(reason, 0) + 1
        return False, reason

    def admit(self, track_id: int, box: Sequence[float], frame_shape: tuple[int, ...], occlusion: float,
              t: float, crop: npt.NDArray[Any] | None = None) -> tuple[bool, str]:
        c = self.cfg
        x1, y1, x2, y2 = box[:4]
        if t - self._last.get(track_id, -1e18) < c.min_interval_s:
            return self._reject("rate")
        if x2 - x1 < c.min_width or y2 - y1 < c.min_height:
            return self._reject("size")
        h, w = frame_shape[:2]
        m = c.edge_margin_px
        if x1 < m or y1 < m or x2 > w - m or y2 > h - m:
            return self._reject("truncated")
        if occlusion >= c.max_occlusion:
            return self._reject("occluded")
        if crop is not None and laplacian_variance(crop) < c.min_laplacian_var:
            return self._reject("blur")
        self._last[track_id] = t
        self.stats["ok"] = self.stats.get("ok", 0) + 1
        return True, "ok"

    def forget(self, track_ids: Sequence[int]) -> None:
        for tid in track_ids:
            self._last.pop(tid, None)


def crop(image: npt.NDArray[Any], box: Sequence[float]) -> npt.NDArray[Any]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    return image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


class EmbeddingExtractor(Protocol):
    dim: int

    def __call__(self, crops: Sequence[npt.NDArray[Any]]) -> F32: ...


class OnnxReIDExtractor:
    """OSNet-AIN / FastReID exported to ONNX, run with ONNX Runtime.

    Provider preference: TensorRT -> CUDA -> CPU. Input is BGR uint8 crops; the
    extractor resizes to the network size (default 256x128), applies ImageNet
    normalisation, batches, and returns L2-normalised (N, D) float32 embeddings.
    """

    def __init__(self, model_path: str, *, input_size: tuple[int, int] = (256, 128),
                 providers: Sequence[str] | None = None, max_batch: int = 32) -> None:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prefs = providers or [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                          "CPUExecutionProvider") if p in avail]
        self._sess = ort.InferenceSession(model_path, providers=list(prefs))
        self._input = self._sess.get_inputs()[0].name
        self._h, self._w = input_size
        self._max_batch = max_batch
        out_shape = self._sess.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 512
        self._mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    def _prep(self, crops: Sequence[npt.NDArray[Any]]) -> F32:
        import cv2

        batch = np.stack([cv2.resize(c, (self._w, self._h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                          for c in crops]).astype(np.float32) / 255.0
        return ((batch.transpose(0, 3, 1, 2) - self._mean) / self._std).astype(np.float32)

    def __call__(self, crops: Sequence[npt.NDArray[Any]]) -> F32:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        outs = []
        for i in range(0, len(crops), self._max_batch):
            x = self._prep(crops[i:i + self._max_batch])
            outs.append(self._sess.run(None, {self._input: x})[0])
        return l2_normalize(np.concatenate(outs).reshape(len(crops), -1))


@dataclass
class EmbeddingGallery:
    """Bounded set of key-crop embeddings for one identity with a robust prototype.

    The prototype is the L2-normalised element-wise median: one embedding polluted by a
    passer-by in the crop shifts a mean noticeably but leaves the median untouched.
    """

    capacity: int = 16
    _items: deque[tuple[float, F32]] = field(default_factory=deque)
    _proto: F32 | None = None

    def add(self, emb: npt.ArrayLike, t: float | None = None) -> None:
        self._items.append((time.monotonic() if t is None else t, l2_normalize(emb)))
        while len(self._items) > self.capacity:
            self._items.popleft()
        self._proto = None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def prototype(self) -> F32 | None:
        if not self._items:
            return None
        if self._proto is None:
            m = np.stack([e for _, e in self._items])
            self._proto = l2_normalize(np.median(m, axis=0))
        return self._proto

    def recent(self, n: int) -> list[F32]:
        return [e for _, e in list(self._items)[-n:]]


def cosine_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    return float(1.0 - np.dot(l2_normalize(a), l2_normalize(b)))
