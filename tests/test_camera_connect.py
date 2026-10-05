from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipcam.config import CameraConfig, StreamRole
from ipcam.stream.connect import ConnectionCheckError, common_paths, resolve_camera, onvif_candidates, probe_video
from tools.camera_setup import selection_from_fields


def camera():
    return CameraConfig("192.168.1.149", "admin", "secret@:/")


def test_dahua_main_sub_channel_and_selection():
    assert common_paths("Dahua", 3, "main") == ["/cam/realmonitor?channel=3&subtype=0"]
    assert common_paths("Dahua", 3, "sub") == ["/cam/realmonitor?channel=3&subtype=1"]
    s = selection_from_fields("192.168.1.149", "admin", "test", "554", "1", "", "main", False, brand="Dahua")
    assert s.camera.rtsp_url(StreamRole.MAIN).endswith("/cam/realmonitor?channel=1&subtype=0")


def test_auto_dahua_and_hik_paths_are_verified_not_guessed():
    tried = []
    def probe(cam, role):
        tried.append(cam.rtsp_path)
        if not cam.rtsp_path.startswith("/Streaming"):
            raise ValueError("404 Not Found")
    result = resolve_camera(camera(), "sub", probe=probe)
    assert tried == ["/cam/realmonitor?channel=1&subtype=1", "/Streaming/Channels/102"]
    assert result.rtsp_path == tried[-1]


def test_auth_failure_stops_path_attempts_and_hides_secret():
    attempts = []
    def probe(cam, role):
        attempts.append(cam)
        raise ValueError("401 Unauthorized " + cam.rtsp_url(role))
    with pytest.raises(ConnectionCheckError, match="401/403") as error:
        resolve_camera(camera(), "main", probe=probe)
    assert len(attempts) == 1
    assert "secret" not in str(error.value)


def test_custom_path_is_used_without_brand_fallback():
    cfg = replace(camera(), rtsp_path="/vendor/video?streamid=2")
    calls = []
    assert resolve_camera(cfg, "main", probe=lambda c, _: calls.append(c)) == cfg
    assert calls == [cfg]


def test_onvif_fallback_uses_returned_port_and_uri():
    expected = replace(camera(), rtsp_port=8554, rtsp_path="/unusual/video")
    calls = []
    def probe(cam, role):
        calls.append(cam)
        if cam != expected:
            raise ValueError("404 Not Found")
    def discover(cam, stream, port):
        assert port == 8080 and stream == "main"
        return [expected]
    assert resolve_camera(camera(), "main", onvif_port=8080, probe=probe, discover=discover) == expected
    assert len(calls) == 3


def test_onvif_auth_error_is_visible_without_raw_http_exception():
    def discover(*args):
        raise ConnectionCheckError("ONVIF kullanıcı adı/şifre reddedildi.")
    with pytest.raises(ConnectionCheckError, match="ONVIF kullanıcı"):
        resolve_camera(camera(), "main", "ONVIF", probe=lambda *_: None, discover=discover)


def test_onvif_media_flow_and_profile_resolution_order(monkeypatch):
    import httpx
    messages = []
    replies = [
        '<Envelope><Media><XAddr>http://192.168.1.149/onvif/media</XAddr></Media></Envelope>',
        '<Envelope><Profiles token="low"><Resolution><Width>640</Width><Height>360</Height></Resolution></Profiles>'
        '<Profiles token="high"><Resolution><Width>1920</Width><Height>1080</Height></Resolution></Profiles></Envelope>',
        '<Envelope><Uri>rtsp://192.168.1.149:8554/high</Uri></Envelope>',
        '<Envelope><Uri>rtsp://192.168.1.149:8554/low</Uri></Envelope>']
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, endpoint, content, headers):
            messages.append(content.decode())
            return SimpleNamespace(status_code=200, content=replies.pop(0).encode(), raise_for_status=lambda: None)
    monkeypatch.setattr(httpx, "Client", Client)
    results = onvif_candidates(camera(), "main")
    assert [c.rtsp_path for c in results] == ["/high", "/low"]
    assert all(c.rtsp_port == 8554 for c in results)
    assert "<trt:ProfileToken>high</trt:ProfileToken>" in messages[2]
    assert all("secret@:/" not in body for body in messages)


def test_probe_requires_decoded_frame_and_closes_container(monkeypatch):
    import av
    state = []
    class Container:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            state.append("closed")
        def demux(self, **kwargs):
            return [SimpleNamespace(decode=lambda: [SimpleNamespace(width=1920, height=1080)])]
    def open_container(url, **kwargs):
        assert kwargs["options"]["rtsp_transport"] == "tcp"
        return Container()
    monkeypatch.setattr(av, "open", open_container)
    probe_video(camera(), StreamRole.MAIN)
    assert state == ["closed"]
