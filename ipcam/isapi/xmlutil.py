"""Namespace-agnostic helpers for Hikvision ISAPI XML.

Hikvision firmwares disagree on the default namespace (``ver10`` vs ``ver20``, sometimes
none), so lookups match on local names only. Parsing uses the stdlib expat parser with
DTD/entity expansion disabled by construction (ISAPI bodies never declare a DOCTYPE; one
is rejected to rule out entity-expansion bombs from a compromised device).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

HIK_NS = "http://www.hikvision.com/ver20/XMLSchema"
ISAPI_NS = "http://www.isapi.org/ver20/XMLSchema"     # newer firmwares (thermal PT series, 2020+)


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespace_of(el: ET.Element) -> str | None:
    """Default namespace of a parsed element (``{ns}tag`` -> ``ns``)."""
    return el.tag[1:].split("}", 1)[0] if el.tag.startswith("{") else None


def parse(data: bytes | str) -> ET.Element:
    raw = data.encode() if isinstance(data, str) else data
    head = raw[:512].lower()
    if b"<!doctype" in head or b"<!entity" in raw.lower():
        raise ValueError("XML with DOCTYPE/ENTITY declarations is not accepted")
    return ET.fromstring(raw)


def find(el: ET.Element, *path: str) -> ET.Element | None:
    cur: ET.Element | None = el
    for name in path:
        if cur is None:
            return None
        cur = next((c for c in cur if local(c.tag) == name), None)
    return cur


def text(el: ET.Element, *path: str, default: str | None = None) -> str | None:
    node = find(el, *path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def findall(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el.iter() if local(c.tag) == name]


def to_dict(el: ET.Element) -> dict[str, Any]:
    """Recursive element -> dict conversion; repeated children become lists."""
    out: dict[str, Any] = {}
    for child in el:
        key = local(child.tag)
        value: Any = to_dict(child) if len(child) else (child.text or "").strip()
        if key in out:
            existing = out[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[key] = [existing, value]
        else:
            out[key] = value
    return out


def build(root: str, fields: dict[str, Any], *, namespace: str = HIK_NS) -> bytes:
    """Build ``<root version="2.0" xmlns=...>`` with nested dict/list children."""
    el = ET.Element(root, {"version": "2.0", "xmlns": namespace})

    def fill(parent: ET.Element, data: dict[str, Any]) -> None:
        for key, value in data.items():
            items = value if isinstance(value, list) else [value]
            for item in items:
                child = ET.SubElement(parent, key)
                if isinstance(item, dict):
                    fill(child, item)
                elif isinstance(item, bool):
                    child.text = "true" if item else "false"
                else:
                    child.text = str(item)

    fill(el, fields)
    return b'<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(el, encoding="utf-8")


def set_text(el: ET.Element, path: str, value: Any, *, create: bool = True) -> ET.Element:
    """Set ``a/b/c`` (local names) under ``el``; missing nodes are created in el's namespace."""
    ns = namespace_of(el)
    cur = el
    for name in path.split("/"):
        nxt = next((c for c in cur if local(c.tag) == name), None)
        if nxt is None:
            if not create:
                raise KeyError(path)
            nxt = ET.SubElement(cur, f"{{{ns}}}{name}" if ns else name)
        cur = nxt
    cur.text = ("true" if value else "false") if isinstance(value, bool) else str(value)
    return cur


def serialize(el: ET.Element) -> bytes:
    """Serialise a (possibly modified) device document keeping its default namespace
    unprefixed -- ``ns0:`` prefixes are rejected by several firmwares."""
    ns = namespace_of(el)

    def strip(node: ET.Element) -> None:
        if ns and node.tag.startswith("{" + ns + "}"):
            node.tag = node.tag[len(ns) + 2:]
        for c in node:
            strip(c)

    clone = ET.fromstring(ET.tostring(el))
    strip(clone)
    if ns:
        clone.set("xmlns", ns)
    return b'<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(clone, encoding="utf-8")


def fmt_float(v: float, digits: int = 3) -> str:
    """Fixed-point text without exponent (devices reject ``1e-05``) and without ``-0.000``."""
    s = f"{v:.{digits}f}"
    return "0." + "0" * digits if s.lstrip("-") == "0." + "0" * digits else s


class Capability:
    """One node of an ISAPI capability document.

    XML capabilities carry their constraints as attributes on the node
    (``<elevation min="-90.000" max="270.000">``, ``<zoomType opt="absoluteZoom,focalLen">``);
    JSON capabilities as ``@min``/``@max``/``@opt`` keys. Both are normalised here.
    """

    __slots__ = ("children", "default", "max", "min", "name", "opt", "value")

    def __init__(self, name: str) -> None:
        self.name = name
        self.value: str | None = None
        self.min: float | None = None
        self.max: float | None = None
        self.opt: tuple[str, ...] | None = None
        self.default: str | None = None
        self.children: dict[str, Capability] = {}

    @classmethod
    def from_xml(cls, el: ET.Element) -> Capability:
        cap = cls(local(el.tag))
        cap.value = (el.text or "").strip() or None
        a = el.attrib
        for key in ("min", "max"):
            if key in a:
                try:
                    setattr(cap, key, float(a[key]))
                except ValueError:
                    pass
        if "opt" in a:
            cap.opt = tuple(s.strip() for s in a["opt"].split(",") if s.strip())
        cap.default = a.get("def")
        for child in el:
            c = cls.from_xml(child)
            if c.name in cap.children:          # repeated elements: keep the first, index others
                cap.children[f"{c.name}#{len(cap.children)}"] = c
            else:
                cap.children[c.name] = c
        return cap

    @classmethod
    def from_json(cls, name: str, obj: Any) -> Capability:
        cap = cls(name)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "@min":
                    cap.min = float(v)
                elif k == "@max":
                    cap.max = float(v)
                elif k == "@opt":
                    cap.opt = tuple(str(x) for x in v) if isinstance(v, list) else tuple(str(v).split(","))
                elif k == "@def":
                    cap.default = str(v)
                else:
                    cap.children[k] = cls.from_json(k, v)
        elif obj is not None:
            cap.value = str(obj).lower() if isinstance(obj, bool) else str(obj)
        return cap

    def get(self, path: str) -> Capability | None:
        """Dotted path lookup, e.g. ``"ThermalCap.isSupportFireDetection"``; ``"..x"`` searches deep."""
        if path.startswith(".."):
            target = path[2:]
            stack = [self]
            while stack:
                node = stack.pop()
                if node.name == target:
                    return node
                stack.extend(node.children.values())
            return None
        node: Capability | None = self
        for part in path.split("."):
            if node is None:
                return None
            node = node.children.get(part)
        return node

    def supports(self, path: str) -> bool:
        node = self.get(path)
        return node is not None and (node.value or "").lower() == "true"

    def range(self, path: str) -> tuple[float, float] | None:
        node = self.get(path)
        if node is None or node.min is None or node.max is None:
            return None
        return node.min, node.max

    def options(self, path: str) -> tuple[str, ...]:
        node = self.get(path)
        return node.opt or () if node is not None else ()

    def __contains__(self, path: str) -> bool:
        return self.get(path) is not None

    def __repr__(self) -> str:
        return f"Capability({self.name}, children={len(self.children)})"
