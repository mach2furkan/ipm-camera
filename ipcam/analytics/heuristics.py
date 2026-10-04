"""False-alarm suppression on *tracks* (not detections).

Filtering raw detections would starve ByteTrack's second association stage, which
exists precisely to keep low-quality detections of a real person attached to the track.
Instead each track is judged on its observed history, and only tracks that pass may
raise alarms. Verdicts use robust statistics (median / percentile) over a window so a
single mis-shaped box does not flip a person into "animal".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bytetrack import Track


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    min_aspect: float | None = 0.8         # H/W; cats, dogs, car glare are wide. None for top-down cameras.
    min_height_px: float = 8.0
    max_speed_hps: float = 6.0            # body heights per second (sprinting adult ~ 4-5)
    speed_percentile: float = 80.0
    min_observations: int = 3
    min_age_s: float = 0.3
    window: int = 15                       # observed points considered


@dataclass(frozen=True, slots=True)
class Verdict:
    valid: bool
    reasons: tuple[str, ...]

    @property
    def maturing(self) -> bool:
        """Rejected only because the track is still young (may become valid shortly)."""
        return bool(self.reasons) and set(self.reasons) <= _MATURING


_MATURING = {"too_few_observations", "too_young"}


class HeuristicFilter:
    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.cfg = config or HeuristicConfig()

    def assess(self, track: Track) -> Verdict:
        cfg = self.cfg
        obs = [p for p in track.history if p.observed][-cfg.window:]
        reasons: list[str] = []
        if len(obs) < cfg.min_observations:
            reasons.append("too_few_observations")
        if track.age_s < cfg.min_age_s:
            reasons.append("too_young")
        if obs:
            boxes = np.array([p.box for p in obs])
            w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-3)
            h = boxes[:, 3] - boxes[:, 1]
            if float(np.median(h)) < cfg.min_height_px:
                reasons.append("too_small")
            if cfg.min_aspect is not None and float(np.median(h / w)) < cfg.min_aspect:
                reasons.append("aspect")
            if len(obs) >= 2:
                t = np.array([p.t for p in obs])
                foot = np.array([p.foot for p in obs])
                dt = np.diff(t)
                ok = dt > 1e-4
                if ok.any():
                    step = np.hypot(*np.diff(foot, axis=0).T)[ok] / dt[ok]
                    speed = float(np.percentile(step / np.maximum(h[1:][ok], 1.0), cfg.speed_percentile))
                    if speed > cfg.max_speed_hps:
                        reasons.append("implausible_speed")
        return Verdict(not reasons, tuple(reasons))
