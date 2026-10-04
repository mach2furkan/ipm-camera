"""Planar geometry for zone rules (image coordinates: x right, y down)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Point = tuple[float, float]
F64 = npt.NDArray[np.float64]


def side(a: Point, b: Point, p: Point) -> float:
    """2-D cross product (B - A) x (P - A).

    Sign tells which side of the directed line A->B the point lies on; magnitude is
    twice the triangle area, i.e. |AB| times the perpendicular distance.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def signed_distance(a: Point, b: Point, p: Point) -> float:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return side(a, b, p) / length if length > 0 else 0.0


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    """True if segment P1P2 properly crosses segment AB.

    ``side(A,B,P1) * side(A,B,P2) < 0`` alone only tests the infinite *line* AB: a person
    walking past the end of a short tripwire would trigger it. The second pair of
    orientation tests restricts the crossing to within the segment's extent.
    """
    d1 = side(a, b, p1)
    d2 = side(a, b, p2)
    d3 = side(p1, p2, a)
    d4 = side(p1, p2, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def points_in_polygon(points: F64, polygon: F64) -> npt.NDArray[np.bool_]:
    """Even-odd ray casting for (N, 2) points against an (M, 2) polygon, vectorised.

    Works for any simple polygon, convex or not. A horizontal ray is cast to +x and the
    edges it crosses are counted; the half-open rule ``(y_i > y) != (y_j > y)`` counts a
    vertex exactly once, so rays through vertices do not double count.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    poly = np.asarray(polygon, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    straddle = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
    hits = straddle & (x < x_cross)
    return (np.count_nonzero(hits, axis=1) % 2) == 1


def point_in_polygon(p: Point, polygon: F64) -> bool:
    return bool(points_in_polygon(np.array([p]), polygon)[0])


def polygon_area(polygon: F64) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_simple_polygon(polygon: F64) -> bool:
    """No two non-adjacent edges intersect (self-intersecting zones are a config error)."""
    n = len(polygon)
    if n < 3:
        return False
    edges = [(tuple(polygon[i]), tuple(polygon[(i + 1) % n])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True


def denormalize(points: Sequence[Sequence[float]], width: int, height: int) -> F64:
    """Normalised [0, 1] coordinates -> pixels, so one zone config serves main and sub streams."""
    arr = np.asarray(points, dtype=np.float64)
    return arr * np.array([width, height], dtype=np.float64)


def spread_radius(points: F64) -> float:
    """Max distance of a point cloud from its centroid (how far someone wandered)."""
    if len(points) == 0:
        return 0.0
    c = points.mean(axis=0)
    return float(np.max(np.hypot(points[:, 0] - c[0], points[:, 1] - c[1])))
