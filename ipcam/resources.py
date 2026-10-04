"""Deterministic native-resource ownership (RAII for FFmpeg/OpenCV/NVDEC handles).

CPython frees an ``av.container.InputContainer`` or ``cv2.VideoCapture`` only when the
last reference disappears. In a long-running pipeline a reference captured by a traceback,
a logging record or a reference cycle keeps sockets, AVCodecContexts and decoder surfaces
alive indefinitely -- the classic "RSS grows 40 MB per reconnect" leak. Every native
handle in this package is therefore opened through :class:`NativeHandle`, which closes
explicitly, is idempotent, and is tracked in a process-wide registry so tests and the
smoke-test can assert that nothing survives a teardown.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections import Counter
from collections.abc import Callable
from types import TracebackType
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[int, str] = {}
        self._opened: Counter[str] = Counter()
        self._closed: Counter[str] = Counter()

    def add(self, key: int, kind: str) -> None:
        with self._lock:
            self._live[key] = kind
            self._opened[kind] += 1

    def remove(self, key: int) -> None:
        with self._lock:
            kind = self._live.pop(key, None)
            if kind is not None:
                self._closed[kind] += 1

    def live(self) -> Counter[str]:
        with self._lock:
            return Counter(self._live.values())

    def totals(self) -> dict[str, tuple[int, int]]:
        with self._lock:
            return {k: (self._opened[k], self._closed[k]) for k in self._opened}


REGISTRY = _Registry()


def live_handles() -> Counter[str]:
    """Native handles currently open, grouped by kind (e.g. ``{'av.input': 1}``)."""
    return REGISTRY.live()


class NativeHandle(Generic[T]):
    """Owns one native object; ``close()`` runs the closer exactly once.

    A ``weakref.finalize`` backstop closes the handle if the owner is collected without an
    explicit close (and logs it, because that path indicates a missing ``with`` block).
    """

    __slots__ = ("__weakref__", "_closer", "_finalizer", "_kind", "_lock", "_obj")

    def __init__(self, obj: T, closer: Callable[[T], object], kind: str) -> None:
        self._obj: T | None = obj
        self._closer = closer
        self._kind = kind
        self._lock = threading.Lock()
        key = id(self)
        REGISTRY.add(key, kind)
        self._finalizer = weakref.finalize(self, NativeHandle._finalize, obj, closer, kind, key)

    @staticmethod
    def _finalize(obj: object, closer: Callable[[object], object], kind: str, key: int) -> None:
        log.warning("native handle %s reclaimed by GC finalizer (missing explicit close)", kind)
        try:
            closer(obj)
        except Exception:  # noqa: BLE001 - finalizers must never raise
            pass
        REGISTRY.remove(key)

    @property
    def obj(self) -> T:
        obj = self._obj
        if obj is None:
            raise RuntimeError(f"{self._kind} handle already closed")
        return obj

    @property
    def closed(self) -> bool:
        return self._obj is None

    def close(self) -> None:
        with self._lock:
            obj, self._obj = self._obj, None
        if obj is None:
            return
        self._finalizer.detach()
        try:
            self._closer(obj)
        except Exception:
            log.debug("error while closing %s", self._kind, exc_info=True)
        finally:
            REGISTRY.remove(id(self))

    def __enter__(self) -> NativeHandle[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
