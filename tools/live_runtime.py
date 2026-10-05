"""Keep Windows video controls responsive during a single GPU inference job.

One persistent worker, one outstanding job: no frame backlog and no concurrent calls
to a model. The caller renders only the frame belonging to the completed prediction.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import numpy as np


class LatestPreview:
    """Keep one fresh video frame, independent of the frame being analyzed."""

    def __init__(self, refresh_hz=30, stale_s=1):
        if refresh_hz <= 0 or stale_s <= 0:
            raise ValueError('Preview rate and stale interval must be positive')
        self.interval = 1/refresh_hz
        self.stale_s = stale_s
        self.seq = 0
        self.generation = None
        self.image = None
        self.received_at = -float('inf')
        self.last_draw = -float('inf')

    def accept(self, result, generation, now=None):
        now = time.perf_counter() if now is None else now
        if generation != self.generation:
            self.image = None
            self.seq = 0
            self.generation = generation
        if result is None or result.seq <= self.seq:
            return False
        image = result.image
        if not isinstance(image, np.ndarray):
            image = image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
        self.image, self.seq, self.received_at = image, result.seq, now
        return True

    def poll(self, reader, now=None):
        now = time.perf_counter() if now is None else now
        if now-self.last_draw < self.interval:
            return False
        self.last_draw = now
        generation = reader.generation
        if generation != self.generation:
            self.image = None
            self.seq = 0
            self.generation = generation
        result = reader.wait_frame(self.seq, timeout=0, max_age_ms=self.stale_s*1000)
        if reader.generation != generation:
            self.image = None
            self.seq = 0
            self.generation = reader.generation
            return True
        self.accept(result, generation, now)
        if now-self.received_at > self.stale_s:
            self.image = None
        return True


class InferenceCancelled(Exception):
    """The user closed the preview while inference was running."""


class ResponsiveInference:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-inference")

    def run(self, operation, pump):
        future = self.executor.submit(operation)
        while True:
            try:
                return future.result(timeout=.015)
            except TimeoutError:
                # A completed operation can itself have raised TimeoutError.
                if future.done():
                    return future.result()
                if not pump():
                    raise InferenceCancelled()

    def close(self):
        # Complete the current GPU job before releasing model / camera resources.
        self.executor.shutdown(wait=True, cancel_futures=True)


def box_arrays(boxes):
    """Copy detector output GPU->CPU once rather than synchronizing three times."""
    data = getattr(boxes, "data", None)
    if data is not None:
        data = data.cpu().numpy()
        return data[:, :4], data[:, -2], data[:, -1]
    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
