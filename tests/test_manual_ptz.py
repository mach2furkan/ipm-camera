import asyncio
import threading
from types import SimpleNamespace

import pytest

from ipcam.config import CameraConfig
from ipcam import ptz_control as ptz
from ipcam.vision.traffic import TrafficOverlay


@pytest.mark.asyncio
@pytest.mark.parametrize("direction,expected", [("left", (-30, 0, 0)), ("right", (30, 0, 0)),
    ("up", (0, 30, 0)), ("down", (0, -30, 0)), ("zoom_in", (0, 0, 30)), ("zoom_out", (0, 0, -30))])
async def test_isapi_move_always_stops_and_closes(monkeypatch, direction, expected):
    calls = []
    stop = threading.Event()
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            calls.append("closed")
        async def ptz_continuous(self, *speeds, channel):
            calls.append((speeds, channel))
            stop.set()  # Simulate STOP while a move is active.
    monkeypatch.setattr(ptz.HikvisionISAPIClient, "from_config", lambda *args, **kwargs: Client())
    await ptz.move_pulse(CameraConfig("camera", "admin", "secret"), "ISAPI", 2, direction, 30, .25, stop)
    assert calls == [(expected, 2), ((0, 0, 0), 2), "closed"]


@pytest.mark.asyncio
async def test_start_error_still_attempts_stop(monkeypatch):
    calls = []
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            calls.append("closed")
        async def ptz_continuous(self, *speeds, channel):
            calls.append(speeds)
            if any(speeds):
                raise TimeoutError("start may have reached camera")
    monkeypatch.setattr(ptz.HikvisionISAPIClient, "from_config", lambda *args, **kwargs: Client())
    with pytest.raises(TimeoutError):
        await ptz.move_pulse(CameraConfig("camera", "admin", "secret"), "ISAPI", 1, "left", 30, .25, threading.Event())
    assert calls == [(-30, 0, 0), (0, 0, 0), "closed"]


@pytest.mark.asyncio
async def test_dahua_zoom_start_stop_uses_same_command_and_zero_based_channel(monkeypatch):
    calls = []
    stop = threading.Event()
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            calls.append("closed")
        async def get(self, path, params):
            calls.append((path, params))
            stop.set()
            return SimpleNamespace(text="OK", raise_for_status=lambda: None)
    monkeypatch.setattr(ptz.httpx, "AsyncClient", Client)
    await ptz.move_pulse(CameraConfig("camera", "admin", "secret"), "Dahua", 0, "zoom_in", 30, .25, stop)
    assert calls[0][1]["code"] == "ZoomTele" and calls[0][1]["channel"] == 0
    assert calls[0][1]["action"] == "start" and calls[1][1]["action"] == "stop"
    assert {k: v for k, v in calls[0][1].items() if k != "action"} == {k: v for k, v in calls[1][1].items() if k != "action"}
    assert calls[-1] == "closed"


def test_busy_moves_are_not_queued_and_close_rejects_new_moves(monkeypatch):
    started, release = threading.Event(), threading.Event()
    async def operation(*args):
        started.set()
        await asyncio.to_thread(release.wait, 2)
    monkeypatch.setattr(ptz, "move_pulse", operation)
    controller = ptz.ManualPTZ(CameraConfig("camera", "admin", "secret"))
    try:
        assert controller.pulse("left")
        assert started.wait(2)
        assert not controller.pulse("right")
        assert controller.motion_serial == 1
        controller.stop()
        assert controller.stop_event.is_set()
    finally:
        release.set()
        controller.close()
    assert not controller.pulse("left")


def test_ptz_motion_does_not_count_as_a_vehicle_crossing():
    import numpy as np
    overlay = TrafficOverlay()
    def output(y):
        tr = SimpleNamespace(track_id=1, cls=0, conf=.9, box=np.array([80., y-20, 100., y]))
        return SimpleNamespace(tracks=[tr], removed_ids=[])
    overlay.update(output(20), 0, 100, 200)
    overlay.counter.reset_tracks()
    overlay.update(output(80), 1, 100, 200, count=False)
    assert overlay.paused and overlay.counter.counts.sum() == 0
    overlay.counter.reset_tracks()
    overlay.update(output(80), 2, 100, 200)
    assert not overlay.paused and overlay.counter.counts.sum() == 0


def test_stop_after_completed_or_failed_move_sends_only_stop(monkeypatch):
    calls = []
    async def move(*args):
        calls.append("move")
        raise TimeoutError("unconfirmed stop")
    async def stop(*args):
        calls.append("stop-only")
    monkeypatch.setattr(ptz, "move_pulse", move)
    monkeypatch.setattr(ptz, "send_stop", stop)
    controller = ptz.ManualPTZ(CameraConfig("camera", "admin", "secret"))
    try:
        assert controller.pulse("left")
        with pytest.raises(TimeoutError):
            controller.future.result(timeout=2)
        controller.stop()
        controller.future.result(timeout=2)
    finally:
        controller.close()
    assert calls == ["move", "stop-only"]
