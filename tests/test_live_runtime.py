import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tools.live_runtime import InferenceCancelled, ResponsiveInference, LatestPreview, box_arrays
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


def test_video_advances_while_model_is_busy_without_modifying_inference_frame():
    preview = LatestPreview(refresh_hz=30)
    runtime = ResponsiveInference()
    source = np.zeros((20, 30, 3), np.uint8)
    result = SimpleNamespace(seq=1, image=source)
    preview.accept(result, 1)
    release = threading.Event()
    class Reader:
        generation = 1
        def wait_frame(self, after_seq, **kwargs):
            assert kwargs['timeout'] == 0
            return SimpleNamespace(seq=after_seq+1, image=np.full_like(source, 200))
    def operation():
        assert release.wait(2)
        assert not source.any()
        return 'done'
    def pump():
        assert preview.poll(Reader())
        assert preview.seq == 2
        assert (preview.image == 200).all()
        release.set()
        return True
    try:
        assert runtime.run(operation, pump) == 'done'
    finally:
        runtime.close()


def test_preview_rate_limit_expiry_and_reconnect_discard_old_frame():
    preview = LatestPreview(refresh_hz=30, stale_s=1)
    class Reader:
        generation = 1
        calls = 0
        result = SimpleNamespace(seq=9, image=np.zeros((2, 2, 3), np.uint8))
        def wait_frame(self, *args, **kwargs):
            self.calls += 1
            result, self.result = self.result, None
            return result
    reader = Reader()
    assert preview.poll(reader, now=0)
    assert not preview.poll(reader, now=.01)
    assert reader.calls == 1
    assert preview.poll(reader, now=1.1)
    assert preview.image is None
    reader.generation = 2
    reader.result = SimpleNamespace(seq=1, image=np.ones((2, 2, 3), np.uint8))
    assert preview.poll(reader, now=1.2)
    assert preview.seq == 1 and preview.image.all()
    reader.generation = 3
    assert preview.poll(reader, now=1.3)
    assert preview.image is None
