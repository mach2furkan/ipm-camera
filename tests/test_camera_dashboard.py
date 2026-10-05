from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from tools.camera_dashboard import CameraDashboard
from tools.live_detect import LabelPainter
from tools.camera_setup import selection_from_fields


def test_dual_view_preserves_aspect_and_routes_only_optical_clicks():
    calls = []
    dashboard = CameraDashboard(SimpleNamespace(controller=None, show=lambda: None), LabelPainter(16), 'camera', 2, 1,
        SimpleNamespace(click=lambda *args: calls.append(args)))
    optical = np.full((480, 640, 3), (10, 20, 30), np.uint8)
    thermal = np.full((256, 320, 3), (50, 60, 70), np.uint8)
    canvas = dashboard.render(optical, thermal, thermal_live=True)
    assert canvas.shape == (900, 1600, 3)
    left, top, width, height, fw, fh = dashboard.optical_mapping
    assert abs(width/height-fw/fh) < .01
    dashboard.click(cv2.EVENT_LBUTTONDOWN, left+width//2, top+height//2, 0, None)
    assert abs(calls[0][1]-320) <= 1 and abs(calls[0][2]-240) <= 1
    dashboard.click(cv2.EVENT_LBUTTONDOWN, 1000, 400, 0, None)
    assert len(calls) == 1
    dashboard.mode = 'thermal'
    dashboard.render(optical, None, thermal_live=False)
    assert dashboard.optical_mapping is None


def test_ptz_buttons_dispatch_and_speed_is_bounded():
    calls = []
    panel = SimpleNamespace(controller=None, pulse=lambda *args: calls.append(args) or True, show=lambda: None)
    dashboard = CameraDashboard(panel, LabelPainter(16), 'camera')
    dashboard.render(None, optical_live=False)
    dashboard.click(cv2.EVENT_LBUTTONDOWN, 1440, 180, 0, None)
    assert calls == [('up', 30)]
    dashboard.change_speed(1000)
    assert dashboard.speed == 100
    dashboard.change_speed(-1000)
    assert dashboard.speed == 10


@pytest.mark.parametrize('channel', ['abc', '0', '1000', '1'])
def test_reject_invalid_or_duplicate_thermal_channel(channel):
    with pytest.raises(ValueError):
        selection_from_fields('camera', 'admin', 'secret', '554', '1', '', 'main', False,
                              thermal_channel=channel)


def test_dahua_channels_are_separate_and_thermal_custom_path_is_validated():
    selection = selection_from_fields('camera', 'admin', 'secret', '554', '1', '', 'main', False,
                                     brand='Dahua', thermal_channel='2')
    assert selection.thermal_channel == 2
    with pytest.raises(ValueError):
        selection_from_fields('camera', 'admin', 'secret', '554', '1', '', 'main', False,
            thermal_channel='2', thermal_path='/video?password=secret')


def test_repeated_preview_reuses_resize_and_new_frame_invalidates_it(monkeypatch):
    calls = []
    resize = cv2.resize
    def measured(*args, **kwargs):
        calls.append(1)
        return resize(*args, **kwargs)
    monkeypatch.setattr(cv2, 'resize', measured)
    d = CameraDashboard(SimpleNamespace(controller=None, show=lambda: None), LabelPainter(16), 'camera', 2)
    a = np.zeros((480, 640, 3), np.uint8)
    b = np.zeros((256, 320, 3), np.uint8)
    d.render(a, b)
    d.render(a, b)
    assert len(calls) == 2
    d.render(a.copy(), b)
    assert len(calls) == 3
    d.mode = 'optical'
    d.render(a, b)
    assert len(calls) == 4
