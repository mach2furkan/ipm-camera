import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tools.live_runtime import InferenceCancelled, ResponsiveInference, box_arrays
from tools.live_detect import LabelPainter


def test_slow_inference_pumps_controls_without_parallel_jobs():
    runtime = ResponsiveInference()
    release = threading.Event()
    pumped = []
    def operation():
        assert release.wait(2)
        return "prediction"
    def pump():
        pumped.append(threading.get_ident())
        release.set()
        return True
    try:
        assert runtime.run(operation, pump) == "prediction"
        assert pumped == [threading.get_ident()]
        assert runtime.run(lambda: "next-frame", pump) == "next-frame"
    finally:
        runtime.close()


def test_close_during_inference_cancels_display_and_finishes_worker():
    runtime = ResponsiveInference()
    release = threading.Event()
    done = threading.Event()
    def operation():
        assert release.wait(2)
        done.set()
    def pump():
        release.set()
        return False
    try:
        with pytest.raises(InferenceCancelled):
            runtime.run(operation, pump)
    finally:
        runtime.close()
    assert done.is_set()


def test_worker_errors_reach_the_caller_including_timeouts():
    runtime = ResponsiveInference()
    try:
        for error in (ValueError("bad model"), TimeoutError("model timeout")):
            def operation():
                raise error
            with pytest.raises(type(error), match=str(error)):
                runtime.run(operation, lambda: True)
    finally:
        runtime.close()


@pytest.mark.parametrize("tracked", [False, True])
def test_detector_copy_uses_one_transfer_and_preserves_coordinates(tracked):
    array = np.array([[1., 2., 3., 4., .7, 9.]])
    if tracked:
        array = np.insert(array, 4, 100, axis=1)
    class Tensor:
        calls = 0
        def cpu(self):
            self.calls += 1
            return self
        def numpy(self):
            return array
    tensor = Tensor()
    boxes, conf, cls = box_arrays(SimpleNamespace(data=tensor))
    assert tensor.calls == 1
    np.testing.assert_array_equal(boxes, [[1, 2, 3, 4]])
    assert conf.tolist() == [.7] and cls.tolist() == [9.]


def test_label_cache_evicts_old_entries_without_discarding_recent_labels():
    painter = LabelPainter(12)
    for i in range(512):
        painter.patch(str(i), (20, 20, 20))
    recent = painter.patch("0", (20, 20, 20))
    painter.patch("new", (20, 20, 20))
    assert len(painter._cache) == 512
    assert ("1", (20, 20, 20)) not in painter._cache
    assert painter.patch("0", (20, 20, 20)) is recent
