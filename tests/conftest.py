from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

GOP = 10
FRAMES = 60
FPS = 25
SIZE = (320, 240)


def _pick_encoder() -> str:
    for name in ("libx264", "libopenh264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001
            continue
    pytest.skip("no usable video encoder in this PyAV build")


def make_clip(path: Path, *, frames: int = FRAMES, gop: int = GOP, color: bool = True) -> Path:
    encoder = _pick_encoder()
    with av.open(str(path), "w", format="matroska") as out:
        s = out.add_stream(encoder, rate=FPS)
        s.width, s.height = SIZE
        s.pix_fmt = "yuv420p"
        cc = s.codec_context
        cc.gop_size = gop
        cc.max_b_frames = 0
        if encoder == "libx264":
            s.options = {"bf": "0", "tune": "zerolatency", "sc_threshold": "0", "keyint_min": str(gop)}
        w, h = SIZE
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(frames):
            if color:
                img = np.stack([(xx + i * 4) % 256, (yy + i * 2) % 256, np.full_like(xx, 128)], -1)
            else:
                g = (xx + yy + i * 3) % 256
                img = np.stack([g, g, g], -1)
            frame = av.VideoFrame.from_ndarray(img.astype(np.uint8), format="rgb24")
            frame.pts = i
            for p in s.encode(frame):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_clip(tmp_path_factory.mktemp("media") / "clip.mkv")


@pytest.fixture(scope="session")
def gray_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_clip(tmp_path_factory.mktemp("media") / "gray.mkv", color=False)
