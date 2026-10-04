"""Camera network topology: spatio-temporal transition models between fields of view.

For a directed pair (A -> B) the travel time of a person leaving A and appearing in B is
modelled as Gaussian N(mu_AB, sigma_AB^2), truncated to [t_min, t_max]:

* dt < t_min   physically impossible (faster than running) -> hard reject, regardless of
               how similar the embeddings are;
* dt > t_max   the person is assumed to have left the site -> candidate expires.

Models can be seeded from the site plan and refined online from confirmed handovers
(Welford running mean/variance, with a floor on sigma so the model never collapses).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TransitionModel:
    src: str
    dst: str
    mu_s: float
    sigma_s: float
    t_min_s: float
    t_max_s: float
    n: int = 0
    _m2: float = 0.0
    min_sigma_s: float = 1.0

    def allowed(self, dt: float) -> bool:
        return self.t_min_s <= dt <= self.t_max_s

    def pdf(self, dt: float) -> float:
        s = max(self.sigma_s, 1e-6)
        return math.exp(-0.5 * ((dt - self.mu_s) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    def likelihood_ratio(self, dt: float) -> float:
        """pdf normalised to 1 at the mode: 1 = typical travel time, -> 0 = atypical."""
        s = max(self.sigma_s, 1e-6)
        return math.exp(-0.5 * ((dt - self.mu_s) / s) ** 2)

    def observe(self, dt: float, *, prior_weight: int = 5) -> None:
        """Online refinement from a confirmed handover; the seed acts as ``prior_weight`` samples."""
        if not self.allowed(dt):
            return
        if self.n == 0:
            self.n = prior_weight
            self._m2 = self.sigma_s ** 2 * (prior_weight - 1)
        self.n += 1
        delta = dt - self.mu_s
        self.mu_s += delta / self.n
        self._m2 += delta * (dt - self.mu_s)
        self.sigma_s = max(self.min_sigma_s, math.sqrt(self._m2 / max(self.n - 1, 1)))


class CameraTopology:
    def __init__(self, *, default_t_max_s: float = 120.0, allow_unknown_pairs: bool = True) -> None:
        self._models: dict[tuple[str, str], TransitionModel] = {}
        self._overlaps: set[frozenset[str]] = set()
        self.default_t_max_s = default_t_max_s
        self.allow_unknown = allow_unknown_pairs

    def add_transition(self, src: str, dst: str, *, mu_s: float, sigma_s: float, t_min_s: float,
                       t_max_s: float, symmetric: bool = True) -> None:
        self._models[(src, dst)] = TransitionModel(src, dst, mu_s, sigma_s, t_min_s, t_max_s)
        if symmetric:
            self._models[(dst, src)] = TransitionModel(dst, src, mu_s, sigma_s, t_min_s, t_max_s)

    def add_overlap(self, a: str, b: str) -> None:
        self._overlaps.add(frozenset((a, b)))

    def overlaps(self, a: str, b: str) -> bool:
        return a == b or frozenset((a, b)) in self._overlaps

    def model(self, src: str, dst: str) -> TransitionModel | None:
        return self._models.get((src, dst))

    def gate(self, src: str, dst: str, dt: float) -> tuple[bool, float]:
        """(allowed, likelihood in [0, 1]) for a transition after ``dt`` seconds."""
        if src == dst:
            return dt <= self.default_t_max_s, 1.0  # re-entry into the same view
        m = self._models.get((src, dst))
        if m is None:
            if self.overlaps(src, dst):
                return dt <= self.default_t_max_s, 1.0
            return (self.allow_unknown and 0 <= dt <= self.default_t_max_s), 0.5
        return m.allowed(dt), m.likelihood_ratio(dt) if m.allowed(dt) else 0.0

    def t_max_from(self, src: str) -> float:
        outs = [m.t_max_s for (s, _), m in self._models.items() if s == src]
        return max(outs + [self.default_t_max_s if self.allow_unknown or not outs else 0.0])

    @classmethod
    def from_json(cls, data: dict[str, Any] | str | Path) -> CameraTopology:
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        topo = cls(default_t_max_s=float(data.get("default_t_max_s", 120.0)),
                   allow_unknown_pairs=bool(data.get("allow_unknown_pairs", True)))
        for t in data.get("transitions", []):
            topo.add_transition(t["from"], t["to"], mu_s=t["mu_s"], sigma_s=t["sigma_s"], t_min_s=t["t_min_s"],
                                t_max_s=t["t_max_s"], symmetric=t.get("symmetric", True))
        for a, b in data.get("overlaps", []):
            topo.add_overlap(a, b)
        return topo
