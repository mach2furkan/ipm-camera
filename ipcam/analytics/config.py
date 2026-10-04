"""Zone/tripwire configuration in resolution-independent (normalised) coordinates.

Example ``zones.json``::

    {
      "coordinates": "normalized",
      "rules": [
        {"type": "tripwire", "name": "gate", "a": [0.10, 0.62], "b": [0.85, 0.58],
         "inside": [0.5, 0.9], "alarm_on": "in"},
        {"type": "zone", "name": "restricted", "polygon": [[0.6, 0.3], [0.95, 0.3], [0.95, 0.9], [0.6, 0.9]],
         "min_dwell_s": 0.5, "loiter_s": 15, "loiter_radius_px": 30}
      ]
    }

Pixel-valued parameters (``loiter_radius_px``, ``min_band_px``) refer to the frame the
analytics runs on (normally the sub stream).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .geometry import denormalize, is_simple_polygon
from .rules import PolygonZone, Rule, Tripwire

_TRIPWIRE_KEYS = {"alarm_on", "band_rel", "min_band_px", "cooldown_s", "pending_s"}
_ZONE_KEYS = {"intrusion", "min_dwell_s", "exit_grace_s", "loiter_s", "loiter_radius_px"}


def build_rules(config: dict[str, Any] | str | Path, width: int, height: int) -> list[Rule]:
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    normalized = config.get("coordinates", "normalized") == "normalized"

    def px(points: Any) -> Any:
        return denormalize(points, width, height) if normalized else denormalize(points, 1, 1)

    rules: list[Rule] = []
    names: set[str] = set()
    for spec in config.get("rules", []):
        kind = spec.get("type")
        name = spec.get("name") or f"{kind}-{len(rules) + 1}"
        if name in names:
            raise ValueError(f"duplicate rule name {name!r}")
        names.add(name)
        if kind == "tripwire":
            a, b = px([spec["a"], spec["b"]])
            inside = tuple(px([spec["inside"]])[0]) if "inside" in spec else None
            opts = {k: v for k, v in spec.items() if k in _TRIPWIRE_KEYS}
            rules.append(Tripwire(name, tuple(a), tuple(b), inside=inside, **opts))  # type: ignore[arg-type]
        elif kind == "zone":
            poly = px(spec["polygon"])
            if not is_simple_polygon(poly):
                raise ValueError(f"zone {name!r} is self-intersecting")
            opts = {k: v for k, v in spec.items() if k in _ZONE_KEYS}
            rules.append(PolygonZone(name, poly, **opts))
        else:
            raise ValueError(f"unknown rule type {kind!r}")
    return rules
