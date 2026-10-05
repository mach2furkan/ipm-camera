from types import SimpleNamespace

import numpy as np
import pytest

from ipcam.config import StreamRole
from ipcam.vision.traffic import TrafficCounter, TrafficOverlay, VEHICLE_PROMPTS, vehicle_detections
from tools.camera_setup import selection_from_fields
from tools.live_detect import CLASSES, LabelPainter


def fields(**changes):
    values = dict(host="192.168.1.64", username="my-user", password="p@:/#secret",
                  port="554", channel="2", path="", stream="main", traffic=True)
    values.update(changes)
    return selection_from_fields(**values)


def test_credentials_escape_and_never_appear_in_repr_or_redacted_url():
    result = fields()
    assert "p%40%3A%2F%23secret" in result.camera.rtsp_url(StreamRole.MAIN)
    assert result.camera.rtsp_url(StreamRole.SUB).endswith("/Streaming/Channels/202")
    assert "secret" not in repr(result)
    assert "secret" not in result.camera.redacted_rtsp_url(StreamRole.MAIN)


def test_other_brands_and_ipv6():
    result = fields(host="[2001:db8::1]", path="/cam/realmonitor?channel=1&subtype=0")
    url = result.camera.rtsp_url(StreamRole.MAIN)
    assert "@[2001:db8::1]:554/cam/realmonitor?channel=1&subtype=0" in url
    assert fields(host="camera.local").camera.host == "camera.local"


@pytest.mark.parametrize("change", [dict(host="rtsp://user:pw@camera"), dict(host="999.2.3.4"),
    dict(host="camera/path"), dict(host=""), dict(port="0"), dict(port="65536"),
    dict(port="abc"), dict(channel="0"), dict(password=""), dict(username=""),
    dict(path="//other-host/path"), dict(path="/stream?password=secret"), dict(stream="wrong")])
def test_invalid_camera_input(change):
    with pytest.raises(ValueError):
        fields(**change)


def track(tid, x, foot_y, cls=0):
    return SimpleNamespace(track_id=tid, box=np.array([x-10, foot_y-20, x+10, foot_y]),
                           cls=cls, conf=.8)


def test_counter_hysteresis_direction_and_no_duplicate():
    c = TrafficCounter((20, 50), (180, 50))
    for t, y in enumerate((30, 49, 51, 48, 70, 30, 70)):
        c.update([track(1, 100, y)], [], t)
    assert c.counts[0].tolist() == [0, 1]
    c.update([track(2, 100, 70, 2)], [], 10)
    c.update([track(2, 100, 30, 2)], [], 11)
    assert c.counts[2].tolist() == [1, 0]


def test_no_crossing_around_segment_end_or_after_reconnect():
    c = TrafficCounter((20, 50), (180, 50))
    c.update([track(1, 200, 20)], [], 0)
    c.update([track(1, 200, 80)], [], 1)
    assert c.counts.sum() == 0
    c.update([track(2, 100, 20)], [], 2)
    c.reset_tracks()
    c.update([track(2, 100, 80)], [], 3)
    assert c.counts.sum() == 0


def test_removed_state_is_released_totals_retained_and_reset():
    c = TrafficCounter((20, 50), (180, 50))
    c.update([track(1, 100, 20)], [], 0)
    c.update([track(1, 100, 80)], [], 1)
    c.update([], [1], 2)
    assert not c.counted and not c.wire._state
    c.reset_tracks()
    assert c.counts.sum() == 1
    c.reset_counts()
    assert c.counts.sum() == 0


def test_vehicle_filter_rejects_nonfinite_bad_class_and_degenerate_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, np.nan, 20], [0, 0, 5, 5],
                      [0, 0, 1, 20], [0, 0, 10, 10]])
    output = vehicle_detections(boxes, [.9]*5, [0, 0, 6, 2, .5], 100, 100)
    assert output.shape == (1, 6) and output[0, 5] == 0
    assert vehicle_detections(np.empty((0, 4)), [], [], 100, 100).shape == (0, 6)


def test_original_desk_prompts_thresholds_and_traffic_are_independent():
    assert [(c[0], c[2]) for c in CLASSES] == [
        ("person", .30), ("book", .20), ("laptop", .35), ("computer monitor", .25),
        ("keyboard", .20), ("pen", .12), ("pencil", .12), ("computer mouse", .15),
        ("cell phone", .20), ("cup", .20), ("bottle", .20)]
    assert not set(VEHICLE_PROMPTS).intersection(c[0] for c in CLASSES)


def test_long_overlay_clips_to_small_frame():
    painter = LabelPainter(16)
    frame = np.zeros((12, 20, 3), np.uint8)
    painter.put(frame, "very long traffic counter"*10, 100, 100, (20, 30, 40))
    assert frame.any()


def test_line_edit_and_resolution_change():
    import cv2
    overlay = TrafficOverlay()
    empty = SimpleNamespace(tracks=[], removed_ids=[])
    overlay.update(empty, 0, 100, 200)
    overlay.counter.counts[0, 0] = 3
    overlay.update(empty, 1, 200, 400)
    assert overlay.counter.counts.sum() == 3
    overlay.points = []
    overlay.click(cv2.EVENT_LBUTTONDOWN, 50, 60, 0, None)
    overlay.click(cv2.EVENT_LBUTTONDOWN, 350, 160, 0, None)
    overlay.update(empty, 2, 200, 400)
    assert overlay.counter.counts.sum() == 0
    assert overlay.counter.a == (50, 60) and overlay.counter.b == (350, 160)


@pytest.mark.parametrize("exit_key,expected", [(ord("q"), 0), (ord("c"), 75)])
@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('square', [False, True])
def test_live_loop_uses_raw_frames_separate_vocabulary_and_closes_resources(monkeypatch, exit_key, expected, dual, square):
    import torch
    import ultralytics
    from tools import live_detect
    from ipcam.vision import traffic
    models, readers, watchdogs, overlays, windows = [], [], [], [], []
    original_window = live_detect.RollingWindow
    def window(size):
        result = original_window(size)
        windows.append(result)
        return result
    monkeypatch.setattr(live_detect, "RollingWindow", window)
    original_overlay = traffic.TrafficOverlay
    class Overlay(original_overlay):
        def __init__(self):
            super().__init__()
            overlays.append(self)
    monkeypatch.setattr(traffic, "TrafficOverlay", Overlay)
    class Model:
        def __init__(self, path):
            self.prompts = None
            self.frames = []
            self.options = []
            models.append(self)
        def set_classes(self, prompts):
            self.prompts = list(prompts)
        def predict(self, frame, **kwargs):
            self.options.append(kwargs)
            self.frames.append(frame.copy())
            if self.prompts == list(VEHICLE_PROMPTS):
                y = (25, 30, 40, 60)[len(self.frames)-1]
                boxes = SimpleNamespace(xyxy=torch.tensor([[70., y-30., 110., y]]),
                    conf=torch.tensor([.9]), cls=torch.tensor([0.]))
            else:
                class EmptyBoxes:
                    def __len__(self):
                        return 0
                boxes = EmptyBoxes()
            return [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr(ultralytics, "YOLOWorld", Model)
    class Reader:
        generation = 1
        def __init__(self, camera, *args, **kwargs):
            self.n = 0
            self.camera = camera
            readers.append(self)
        def wait_frame(self, *args, **kwargs):
            self.n += 1
            return SimpleNamespace(seq=self.n, image=np.zeros((100, 200, 3), dtype=np.uint8),
                frame=SimpleNamespace(arrival_ns=self.n*40_000_000,
                    latency=lambda: SimpleNamespace(total_ms=0)))
    class Watchdog:
        def __init__(self, reader):
            self.started = self.stopped = False
            watchdogs.append(self)
        def start(self):
            self.started = True
        def stop(self):
            self.stopped = True
    monkeypatch.setattr(live_detect, "RTSPStreamReader", Reader)
    monkeypatch.setattr(live_detect, "StreamWatchdog", Watchdog)
    for method in ("namedWindow", "setMouseCallback", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(live_detect.cv2, method, lambda *args, **kwargs: None)
    keys = iter((-1, -1, -1, exit_key))
    monkeypatch.setattr(live_detect.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(live_detect.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setenv("HIK_PASS", "local-test-only")
    argv = ["--ip", "192.168.1.64", "--brand", "hikvision", "--windowed", "--traffic"]
    if dual:
        argv += ['--thermal-channel', '2']
    if square:
        argv += ['--square-inference']
    assert live_detect.main(argv) == expected
    assert len(models) == 2
    assert models[0].prompts == [c[0] for c in CLASSES]
    assert models[1].prompts == list(VEHICLE_PROMPTS)
    assert all(not frame.any() for model in models for frame in model.frames)
    assert all(options['rect'] is (not square) for model in models for options in model.options)
    assert len(models[0].frames) == len(models[1].frames) == 4
    assert watchdogs[0].started and watchdogs[0].stopped
    assert len(watchdogs) == (2 if dual else 1)
    if dual:
        assert readers[1].camera.channel == 2
        assert readers[0].camera.rtsp_url(StreamRole.MAIN) != readers[1].camera.rtsp_url(StreamRole.MAIN)
        assert watchdogs[1].started and watchdogs[1].stopped
    assert overlays[0].counter.counts.sum() == 1
    # Frame latency is already sampled after inference: do not add inference twice.
    assert windows[1].summary().p50 == 0
