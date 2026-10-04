"""Day/night (IR) state for model-weight selection: ISAPI configuration fused with pixels.

``/ISAPI/Image/channels/1/ircutFilter`` reports the *configured* mode. When the mode is
``auto`` (the factory default) it does not say which way the light sensor has switched the
filter right now. The resolver therefore combines:

1. ISAPI ``day`` / ``night`` -> authoritative;
2. ISAPI ``auto`` / ``schedule`` -> chroma analysis of sub-stream frames: in IR mode the
   ISP outputs monochrome, so the U/V planes collapse onto 128. A Schmitt trigger with
   a dwell time prevents flapping at dusk or under passing headlights.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..isapi.models import IRCutFilterState, IRCutMode


class Illumination(str, Enum):
    COLOR = "color"
    IR = "ir"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IlluminationState:
    mode: Illumination
    source: str                 # "isapi" | "chroma" | "none"
    chroma: float               # mean |U-128|+|V-128| of the last analysed frame
    isapi_mode: IRCutMode | None
    since: float                # monotonic time of the last transition

    @property
    def use_ir_weights(self) -> bool:
        return self.mode is Illumination.IR


def chroma_energy(image: Any, *, stride: int = 8) -> float:
    """Mean absolute chroma of a BGR/RGB uint8 frame (numpy HxWx3 or torch 3xHxW)."""
    if hasattr(image, "detach"):  # torch tensor (C, H, W) on any device
        t = image[:, ::stride, ::stride].float()
        r, g, b = t[0], t[1], t[2]
        u = (-0.169 * r - 0.331 * g + 0.5 * b).abs()
        v = (0.5 * r - 0.419 * g - 0.081 * b).abs()
        return float((u + v).mean().item())
    a = np.asarray(image)[::stride, ::stride].astype(np.float32)
    # BGR layout; for monochrome (IR) pixels R=G=B and both chroma terms vanish exactly.
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    u = np.abs(-0.169 * r - 0.331 * g + 0.5 * b)
    v = np.abs(0.5 * r - 0.419 * g - 0.081 * b)
    return float((u + v).mean())


class IRStateResolver:
    def __init__(self, *, enter_ir_below: float = 3.0, exit_ir_above: float = 7.0, dwell_s: float = 5.0,
                 analyse_every_s: float = 0.5) -> None:
        if enter_ir_below >= exit_ir_above:
            raise ValueError("hysteresis band inverted")
        self._lo = enter_ir_below
        self._hi = exit_ir_above
        self._dwell = dwell_s
        self._period = analyse_every_s
        self._lock = threading.Lock()
        self._isapi: IRCutFilterState | None = None
        self._pixel_mode = Illumination.UNKNOWN
        self._candidate: Illumination | None = None
        self._candidate_since = 0.0
        self._last_chroma = float("nan")
        self._last_analysis = 0.0
        self._since = time.monotonic()
        self._last_mode = Illumination.UNKNOWN

    def update_isapi(self, state: IRCutFilterState) -> IlluminationState:
        with self._lock:
            self._isapi = state
        return self.state()

    def observe_frame(self, image: Any, now: float | None = None) -> IlluminationState:
        now = time.monotonic() if now is None else now
        if now - self._last_analysis < self._period:
            return self.state()
        self._last_analysis = now
        c = chroma_energy(image)
        with self._lock:
            self._last_chroma = c
            target: Illumination | None = None
            if c < self._lo:
                target = Illumination.IR
            elif c > self._hi:
                target = Illumination.COLOR
            if target is None or target is self._pixel_mode:
                self._candidate = None
            elif self._pixel_mode is Illumination.UNKNOWN:
                self._pixel_mode = target
            elif self._candidate is not target:
                self._candidate, self._candidate_since = target, now
            elif now - self._candidate_since >= self._dwell:
                self._pixel_mode, self._candidate = target, None
        return self.state()

    def state(self) -> IlluminationState:
        with self._lock:
            isapi = self._isapi
            if isapi is not None and isapi.is_forced_night:
                mode, source = Illumination.IR, "isapi"
            elif isapi is not None and isapi.is_forced_day:
                mode, source = Illumination.COLOR, "isapi"
            elif self._pixel_mode is not Illumination.UNKNOWN:
                mode, source = self._pixel_mode, "chroma"
            else:
                mode, source = Illumination.UNKNOWN, "none"
            if mode is not self._last_mode:
                self._last_mode = mode
                self._since = time.monotonic()
            return IlluminationState(mode, source, self._last_chroma, isapi.mode if isapi else None, self._since)
