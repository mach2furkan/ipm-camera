"""Cosine-distance vector index over identity prototypes, keyed by global id.

Backends
* ``flat``  exact search, contiguous float32 matrix, O(1) swap-remove / in-place update.
            One (N x D) @ (D,) GEMV: 10 000 x 512 is ~5 MFLOP, about a millisecond on a
            modern CPU core, with zero recall loss.
* ``hnsw``  FAISS IndexHNSWFlat on inner product. HNSW cannot delete, so removals are
            tombstoned and the graph is rebuilt once tombstones exceed a fraction of the
            index. Worth it beyond ~50 000 identities.

Vectors are L2-normalised on insert, so ``1 - inner product`` is the cosine distance
``D_C`` used by the association thresholds.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .reid import l2_normalize

F32 = npt.NDArray[np.float32]


class FlatIndex:
    def __init__(self, dim: int, capacity: int = 1024) -> None:
        self.dim = dim
        self._mat = np.zeros((capacity, dim), np.float32)
        self._keys = np.zeros(capacity, np.int64)
        self._pos: dict[int, int] = {}
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def __contains__(self, key: int) -> bool:
        return key in self._pos

    def upsert(self, key: int, vec: npt.ArrayLike) -> None:
        v = l2_normalize(vec).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        i = self._pos.get(key)
        if i is None:
            if self._n == len(self._mat):
                self._mat = np.concatenate([self._mat, np.zeros_like(self._mat)])
                self._keys = np.concatenate([self._keys, np.zeros_like(self._keys)])
            i = self._n
            self._n += 1
            self._pos[key] = i
            self._keys[i] = key
        self._mat[i] = v

    def remove(self, key: int) -> None:
        i = self._pos.pop(key, None)
        if i is None:
            return
        last = self._n - 1
        if i != last:
            self._mat[i] = self._mat[last]
            moved = int(self._keys[last])
            self._keys[i] = moved
            self._pos[moved] = i
        self._n -= 1

    def search(self, query: npt.ArrayLike, k: int = 5, *, allowed: set[int] | None = None
               ) -> list[tuple[int, float]]:
        if self._n == 0:
            return []
        q = l2_normalize(query).reshape(-1)
        sims = self._mat[: self._n] @ q
        if allowed is not None:
            allow = np.fromiter(allowed, np.int64, len(allowed))
            sims = np.where(np.isin(self._keys[: self._n], allow), sims, -np.inf)
        k = min(k, self._n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self._keys[i]), float(1.0 - sims[i])) for i in top if np.isfinite(sims[i])]


class HNSWIndex:
    def __init__(self, dim: int, *, m: int = 32, ef_search: int = 64, rebuild_ratio: float = 0.25) -> None:
        import faiss

        self._faiss = faiss
        self.dim = dim
        self._m = m
        self._ef = ef_search
        self._ratio = rebuild_ratio
        self._vecs: dict[int, F32] = {}
        self._row_key: list[int] = []
        self._dead = 0
        self._build()

    def _build(self) -> None:
        idx = self._faiss.IndexHNSWFlat(self.dim, self._m, self._faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efSearch = self._ef
        keys = list(self._vecs)
        if keys:
            idx.add(np.stack([self._vecs[k] for k in keys]))
        self._index = idx
        self._row_key = keys
        self._row_of = {k: i for i, k in enumerate(keys)}
        self._dead = 0

    def __len__(self) -> int:
        return len(self._vecs)

    def __contains__(self, key: int) -> bool:
        return key in self._vecs

    def upsert(self, key: int, vec: npt.ArrayLike) -> None:
        v = l2_normalize(vec).reshape(1, -1)
        if key in self._vecs:
            self._dead += 1
        self._vecs[key] = v[0]
        self._index.add(v)
        self._row_of[key] = len(self._row_key)
        self._row_key.append(key)
        self._maybe_rebuild()

    def remove(self, key: int) -> None:
        if self._vecs.pop(key, None) is not None:
            self._row_of.pop(key, None)
            self._dead += 1
            self._maybe_rebuild()

    def _maybe_rebuild(self) -> None:
        if self._dead > max(64, self._ratio * len(self._row_key)):
            self._build()

    def search(self, query: npt.ArrayLike, k: int = 5, *, allowed: set[int] | None = None
               ) -> list[tuple[int, float]]:
        if not self._vecs:
            return []
        q = l2_normalize(query).reshape(1, -1)
        want = min(len(self._row_key), max(k * 4, k + self._dead))
        sims, rows = self._index.search(q, want)
        out: list[tuple[int, float]] = []
        for s, r in zip(sims[0], rows[0]):
            if r < 0:
                continue
            key = self._row_key[r]
            if self._row_of.get(key) != r:   # tombstoned or superseded row
                continue
            if allowed is not None and key not in allowed:
                continue
            out.append((key, float(1.0 - s)))
            if len(out) == k:
                break
        return out


def create_index(dim: int = 512, backend: str = "flat") -> FlatIndex | HNSWIndex:
    if backend == "hnsw":
        return HNSWIndex(dim)
    if backend == "auto":
        try:
            return HNSWIndex(dim)
        except ImportError:
            return FlatIndex(dim)
    return FlatIndex(dim)
