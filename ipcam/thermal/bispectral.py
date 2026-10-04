"""Optical <-> thermal channel registration and thermal confirmation of optical detections.

The two sensors of a bi-spectrum head sit side by side, so mapping a pixel between them is
a homography *plus a range-dependent parallax*: for a baseline ``b`` and focal length ``f``
the disparity is ``f * b / range`` pixels. At 500 m it vanishes; at 15 m it can be tens of
thermal pixels -- which is exactly where intruders are. The registration model is

    p_thermal = H_inf(p_optical) + f_th * b / range * u_b

with ``H_inf`` the at-infinity homography (fitted from far-field point pairs or derived from
the two fields of view) and ``u_b`` the baseline direction in the thermal image.

``ThermalConfirmation`` checks whether an optical detection box covers a warm signature:
headlight reflections, shadows, moving foliage and printed posters do not radiate at body
temperature. The score is a z-value against the box's surrounding ring rather than a fixed
delta, so it adapts to hot summer afternoons where clothing approaches ambient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..fusion.homography import fit_homography, project

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


@dataclass
class ChannelRegistration:
    h_inf: F64                                   # optical px -> thermal px at infinite range
    optical_size: tuple[int, int]
    thermal_size: tuple[int, int]
    baseline_m: float = 0.0
    baseline_dir: tuple[float, float] = (1.0, 0.0)   # unit vector in the thermal image
    thermal_focal_px: float = 0.0

    @classmethod
    def from_fov(cls, optical_size: tuple[int, int], optical_hfov_deg: float, thermal_size: tuple[int, int],
                 thermal_hfov_deg: float, *, baseline_m: float = 0.0, offset_px: tuple[float, float] = (0.0, 0.0)
                 ) -> ChannelRegistration:
        """Coaxial approximation from the two horizontal fields of view (both tracked by zoom)."""
        ow, oh = optical_size
        tw, th = thermal_size
        fo = ow / 2 / np.tan(np.radians(optical_hfov_deg) / 2)
        ft = tw / 2 / np.tan(np.radians(thermal_hfov_deg) / 2)
        s = ft / fo
        h = np.array([[s, 0, tw / 2 - s * ow / 2 + offset_px[0]],
                      [0, s, th / 2 - s * oh / 2 + offset_px[1]],
                      [0, 0, 1.0]])
        return cls(h, optical_size, thermal_size, baseline_m, (1.0, 0.0), ft)

    @classmethod
    def from_points(cls, optical_px: npt.ArrayLike, thermal_px: npt.ArrayLike, optical_size: tuple[int, int],
                    thermal_size: tuple[int, int], **kw: float) -> ChannelRegistration:
        """Fit H_inf from far-field correspondences (> 200 m, where parallax is negligible)."""
        return cls(fit_homography(optical_px, thermal_px), optical_size, thermal_size, **kw)  # type: ignore[arg-type]

    def optical_to_thermal(self, pts: npt.ArrayLike, range_m: float | None = None) -> F64:
        out = project(self.h_inf, pts)
        if range_m and self.baseline_m and self.thermal_focal_px:
            d = self.thermal_focal_px * self.baseline_m / max(range_m, 1.0)
            out = out + d * np.asarray(self.baseline_dir)
        return out

    def box_to_thermal(self, box: tuple[float, float, float, float], range_m: float | None = None
                       ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        q = self.optical_to_thermal([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], range_m)
        return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


@dataclass(frozen=True, slots=True)
class ThermalEvidence:
    peak_c: float
    background_c: float
    z: float
    warm: bool


class ThermalConfirmation:
    def __init__(self, registration: ChannelRegistration, *, z_min: float = 3.0, min_delta_c: float = 1.0,
                 peak_percentile: float = 95.0, ring_px: int = 4) -> None:
        self.reg = registration
        self.z_min = z_min
        self.min_delta = min_delta_c
        self.pct = peak_percentile
        self.ring = ring_px

    def assess(self, temps: F32, optical_box: tuple[float, float, float, float],
               range_m: float | None = None) -> ThermalEvidence | None:
        h, w = temps.shape
        x1, y1, x2, y2 = self.reg.box_to_thermal(optical_box, range_m)
        ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        ix2, iy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if ix2 - ix1 < 1 or iy2 - iy1 < 1:
            return None
        inner = temps[iy1:iy2, ix1:ix2]
        r = self.ring
        ox1, oy1, ox2, oy2 = max(0, ix1 - r), max(0, iy1 - r), min(w, ix2 + r), min(h, iy2 + r)
        outer = temps[oy1:oy2, ox1:ox2].copy()
        outer[iy1 - oy1: iy2 - oy1, ix1 - ox1: ix2 - ox1] = np.nan
        ring = outer[np.isfinite(outer)]
        if ring.size < 4:
            return None
        bg = float(np.median(ring))
        spread = max(float(np.median(np.abs(ring - bg))) * 1.4826, 0.1)
        peak = float(np.nanpercentile(inner, self.pct))
        z = (peak - bg) / spread
        return ThermalEvidence(peak, bg, z, z >= self.z_min and peak - bg >= self.min_delta)
