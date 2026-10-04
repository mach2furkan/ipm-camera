"""Display pipeline for radiometric frames: AGC, false-colour palettes, PNG encoding.

* ``PlateauAGC`` -- plateau histogram equalisation, the method thermal cores use on-camera.
  Plain equalisation spends most output levels on the large uniform background (sky,
  ground) and flattens small warm targets; clipping each histogram bin at a plateau
  limits that, so a 20-pixel person 2 degC above the field keeps visible contrast. The
  transfer curve is smoothed over time so the picture does not "breathe" when a hot
  object enters the view.
* Palettes are 256-entry LUTs interpolated from control points (white-hot, black-hot,
  iron, rainbow-HC, arctic). ``isotherm`` highlights a temperature band in a fixed colour
  over a greyscale base -- the operator's tool for "show me everything above 45 degC".
* ``encode_png`` writes RGB/greyscale PNGs with zlib only (no Pillow on edge devices).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]

_STOPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "white_hot": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "black_hot": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
    "iron": [(0.0, (0, 0, 10)), (0.18, (40, 0, 120)), (0.38, (150, 0, 150)), (0.55, (220, 50, 50)),
             (0.72, (250, 140, 0)), (0.88, (255, 220, 40)), (1.0, (255, 255, 230))],
    "rainbow_hc": [(0.0, (0, 0, 40)), (0.15, (0, 0, 200)), (0.32, (0, 170, 220)), (0.48, (0, 200, 60)),
                   (0.64, (230, 230, 0)), (0.8, (250, 120, 0)), (0.92, (230, 0, 0)), (1.0, (255, 255, 255))],
    "arctic": [(0.0, (10, 10, 40)), (0.35, (20, 80, 170)), (0.6, (120, 190, 230)), (0.8, (250, 200, 120)),
               (1.0, (255, 245, 220))],
}

PALETTES = tuple(_STOPS)


def palette_lut(name: str) -> U8:
    stops = _STOPS[name]
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(grid, xs, cols[:, i]) for i in range(3)], axis=1).round().astype(np.uint8)


_LUTS = {name: palette_lut(name) for name in _STOPS}


class PlateauAGC:
    def __init__(self, *, bins: int = 512, plateau: float = 0.02, smoothing: float = 0.15,
                 clip_percent: float = 0.2) -> None:
        self.bins = bins
        self.plateau = plateau          # max share of pixels a single bin may claim
        self.alpha = smoothing          # EMA factor on the transfer curve
        self.clip = clip_percent        # ignore this % of extreme pixels when fixing the range
        self._lo: float | None = None
        self._hi: float | None = None
        self._cdf: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        self._lo = self._hi = self._cdf = None

    def __call__(self, temps: F32) -> U8:
        v = temps[np.isfinite(temps)]
        if v.size == 0:
            return np.zeros(temps.shape, np.uint8)
        lo, hi = np.percentile(v, [self.clip, 100 - self.clip])
        if hi - lo < 0.5:
            mid = (hi + lo) / 2
            lo, hi = mid - 0.25, mid + 0.25
        a = self.alpha
        self._lo = lo if self._lo is None else (1 - a) * self._lo + a * lo
        self._hi = hi if self._hi is None else (1 - a) * self._hi + a * hi
        hist, edges = np.histogram(v, bins=self.bins, range=(self._lo, self._hi))
        cap = max(1.0, self.plateau * v.size)
        hist = np.minimum(hist, cap)
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
        self._cdf = cdf if self._cdf is None else (1 - a) * self._cdf + a * cdf
        idx = np.clip(((temps - self._lo) / (self._hi - self._lo) * self.bins).astype(np.int64), 0, self.bins - 1)
        out = (self._cdf[idx] * 255.0)
        out[~np.isfinite(temps)] = 0
        return out.astype(np.uint8)

    @property
    def window(self) -> tuple[float, float] | None:
        return (self._lo, self._hi) if self._lo is not None and self._hi is not None else None


def colorize(gray: U8, palette: str = "iron") -> U8:
    return _LUTS[palette][gray]


def isotherm(gray: U8, temps: F32, lo_c: float, hi_c: float = float("inf"),
             color: tuple[int, int, int] = (255, 170, 0)) -> U8:
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask = (temps >= lo_c) & (temps <= hi_c)
    rgb[mask] = color
    return rgb


def encode_png(img: U8, *, level: int = 6) -> bytes:
    if img.ndim == 2:
        h, w = img.shape
        color_type, rows = 0, img
    else:
        h, w, c = img.shape
        if c != 3:
            raise ValueError("RGB or greyscale only")
        color_type, rows = 2, img.reshape(h, w * 3)
    raw = np.empty((h, rows.shape[1] + 1), np.uint8)
    raw[:, 0] = 0                          # filter type "None" per scanline
    raw[:, 1:] = rows
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
            + chunk(b"IEND", b""))


def downsample_grid(temps: F32, max_w: int = 96) -> tuple[F32, int]:
    """Block-max downsampling for the operator's hover read-out: a hotspot smaller than a
    block must still show its peak temperature, so the reducer is max, not mean."""
    h, w = temps.shape
    k = max(1, int(np.ceil(w / max_w)))
    hh, ww = h // k, w // k
    t = temps[: hh * k, : ww * k].reshape(hh, k, ww, k)
    return t.max(axis=(1, 3)).astype(np.float32), k
