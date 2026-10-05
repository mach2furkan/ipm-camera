import threading
import pytest

from ipcam.config import CameraConfig
from ipcam.stream.connect import ConnectionCancelled, resolve_camera
from tools.camera_setup import selection_from_fields
from tools import camera_app


def test_cancel_before_probe_prevents_connection():
    cancelled = threading.Event()
    cancelled.set()
    calls = []
    with pytest.raises(ConnectionCancelled):
        resolve_camera(CameraConfig("192.168.1.149", "admin", "secret"), "main",
                       probe=lambda *args: calls.append(args), cancelled=cancelled)
    assert calls == []


def test_cancel_during_probe_prevents_fallbacks():
    cancelled = threading.Event()
    attempts = []
    def probe(*args):
        attempts.append(args)
        cancelled.set()
        raise ValueError("404 Not Found")
    with pytest.raises(ConnectionCancelled):
        resolve_camera(CameraConfig("192.168.1.149", "admin", "secret"), "main",
                       probe=probe, cancelled=cancelled)
    assert len(attempts) == 1


def test_progress_never_includes_password():
    messages = []
    resolve_camera(CameraConfig("192.168.1.149", "admin", "secret"), "main",
                   probe=lambda *_: None, progress=messages.append)
    assert len(messages) == 2 and all("secret" not in text for text in messages)


def test_launcher_reconfigures_with_same_nonsecret_context(monkeypatch):
    calls = []
    def main(args, connection_defaults):
        calls.append(dict(connection_defaults))
        connection_defaults["host"] = "192.168.1.149"
        return 75 if len(calls) == 1 else 0
    monkeypatch.setattr(camera_app, "main", main)
    assert camera_app.run([]) == 0
    assert calls == [{}, {"host": "192.168.1.149"}]


def test_custom_path_is_retained_separately_from_generated_brand_path():
    params = ("192.168.1.149", "admin", "secret", "554", "1")
    auto = selection_from_fields(*params, "", "main", False, brand="Dahua")
    custom = selection_from_fields(*params, "/custom/video", "main", False, brand="Dahua")
    assert auto.custom_path == "" and auto.camera.rtsp_path.startswith("/cam/realmonitor")
    assert custom.custom_path == custom.camera.rtsp_path == "/custom/video"


def test_startup_errors_hide_credentials_and_explain_memory():
    assert "belleği" in camera_app.startup_error_message(MemoryError("secret"))
    assert "secret" not in camera_app.startup_error_message(RuntimeError("rtsp://admin:secret@camera"))
    assert "bulunamadı" in camera_app.startup_error_message(FileNotFoundError("secret"))
