"""Keep Windows video controls responsive during a single GPU inference job.

One persistent worker, one outstanding job: no frame backlog and no concurrent calls
to a model. The caller renders only the frame belonging to the completed prediction.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError


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
