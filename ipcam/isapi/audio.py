"""Vectorised G.711 (µ-law / A-law) encoders and PCM preparation for two-way audio.

``audioop`` was removed in Python 3.13, so the ITU-T G.711 segment encoders are
implemented with numpy. Hikvision two-way audio expects 8 kHz mono G.711 by default
(``audioCompressionType`` in /ISAPI/System/TwoWayAudio/channels/<id>).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

_MU_BIAS = 0x84
_MU_CLIP = 32635

_SEG_END = np.array([0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF], dtype=np.int32)


def pcm16_to_ulaw(pcm: npt.NDArray[np.int16]) -> bytes:
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0x00).astype(np.int32)
    mag = np.minimum(np.abs(x), _MU_CLIP) + _MU_BIAS
    # exponent = position of the highest set bit among bits 7..14
    exponent = np.clip(np.floor(np.log2(np.maximum(mag, 1))).astype(np.int32) - 7, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    out = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return out.astype(np.uint8).tobytes()


def pcm16_to_alaw(pcm: npt.NDArray[np.int16]) -> bytes:
    x = pcm.astype(np.int32) >> 3  # 13-bit
    mask = np.where(x >= 0, 0xD5, 0x55).astype(np.int32)
    mag = np.where(x >= 0, x, -x - 1)
    seg = np.searchsorted(_SEG_END[:8] >> 3, mag, side="left").astype(np.int32)
    seg = np.minimum(seg, 8)
    aval = np.where(
        seg >= 8,
        0x7F,
        (seg << 4) | np.where(seg < 2, (mag >> 1) & 0x0F, (mag >> np.maximum(seg, 1)) & 0x0F),
    )
    return ((aval ^ mask) & 0xFF).astype(np.uint8).tobytes()


def resample_linear(pcm: npt.NDArray[np.int16], src_rate: int, dst_rate: int) -> npt.NDArray[np.int16]:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    n_out = int(round(pcm.size * dst_rate / src_rate))
    t_src = np.arange(pcm.size, dtype=np.float64) / src_rate
    t_dst = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(t_dst, t_src, pcm.astype(np.float64)).round().astype(np.int16)


def load_wav_mono16(path: str | Path) -> tuple[npt.NDArray[np.int16], int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        rate = wf.getframerate()
        ch = wf.getnchannels()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).round().astype(np.int16)
    return data.astype(np.int16), rate


def encode_g711(pcm: npt.NDArray[np.int16], sample_rate: int, codec: str) -> bytes:
    pcm8k = resample_linear(pcm, sample_rate, 8000)
    c = codec.lower().replace(".", "").replace("_", "")
    if "ulaw" in c or "mulaw" in c or c.endswith("u"):
        return pcm16_to_ulaw(pcm8k)
    if "alaw" in c or c.endswith("a"):
        return pcm16_to_alaw(pcm8k)
    raise ValueError(f"unsupported two-way audio codec {codec!r} (G.711 µ-law/A-law only)")
