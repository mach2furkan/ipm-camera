"""Geodesy and single-camera geolocation for pan-tilt heads.

A PT event carries the head pose at the moment of detection (azimuth/elevation, float
degrees), the target box in the image and -- with a laser rangefinder or the device's own
distance estimate -- the slant range. That is enough to place the target on the map from
one camera, without a ground-plane homography:

    bearing    = north_offset + azimuth_reading (+ horizontal angle of the box centre)
    depression = elevation_reading + tilt_bias (+ vertical angle of the box foot)
    horizontal = sqrt(range^2 - dz^2)            (range known)
               = height / tan(depression)        (flat terrain, no range)

Coordinates are produced in a local East-North-Up frame tangent to the WGS-84 ellipsoid at
a site origin (exact ECEF transforms, not a flat-earth approximation), so multi-kilometre
thermal sites stay consistent with GNSS-surveyed camera positions. Uncertainty from the
angular resolution, pixel quantisation and range error is propagated into a 2x2 covariance
that the global tracker uses as measurement noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_EP2 = (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2


def geodetic_to_ecef(lat: float, lon: float, alt: float = 0.0) -> F64:
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo), (n + alt) * math.cos(la) * math.sin(lo),
                     (n * (1 - WGS84_E2) + alt) * math.sin(la)])


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Bowring's closed form (sub-millimetre for terrestrial heights)."""
    p = math.hypot(x, y)
    th = math.atan2(z * WGS84_A, p * WGS84_B)
    lon = math.atan2(y, x)
    lat = math.atan2(z + WGS84_EP2 * WGS84_B * math.sin(th) ** 3, p - WGS84_E2 * WGS84_A * math.cos(th) ** 3)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else abs(z) - WGS84_B
    return math.degrees(lat), math.degrees(lon), alt


class LocalTangentPlane:
    """East-North-Up frame at a geodetic origin."""

    def __init__(self, lat: float, lon: float, alt: float = 0.0) -> None:
        self.origin = (lat, lon, alt)
        self._o = geodetic_to_ecef(lat, lon, alt)
        la, lo = math.radians(lat), math.radians(lon)
        self._r = np.array([
            [-math.sin(lo), math.cos(lo), 0.0],
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
        ])

    def to_enu(self, lat: float, lon: float, alt: float = 0.0) -> F64:
        return self._r @ (geodetic_to_ecef(lat, lon, alt) - self._o)

    def to_geodetic(self, e: float, n: float, u: float = 0.0) -> tuple[float, float, float]:
        x, y, z = self._r.T @ np.array([e, n, u]) + self._o
        return ecef_to_geodetic(float(x), float(y), float(z))


def bearing_deg(dx_east: float, dy_north: float) -> float:
    """Compass bearing (0 = north, clockwise) of an ENU vector."""
    return math.degrees(math.atan2(dx_east, dy_north)) % 360.0


@dataclass(frozen=True, slots=True)
class LensModel:
    sensor_width_mm: float
    sensor_height_mm: float
    image_width: int
    image_height: int
    focal_min_mm: float
    focal_max_mm: float | None = None

    def focal_mm(self, zoom: float | None, focal_len_mm: float | None = None) -> float:
        if focal_len_mm:
            return focal_len_mm
        f = self.focal_min_mm * max(zoom or 1.0, 1.0)
        return min(f, self.focal_max_mm) if self.focal_max_mm else f

    def fov_deg(self, zoom: float | None, focal_len_mm: float | None = None) -> tuple[float, float]:
        f = self.focal_mm(zoom, focal_len_mm)
        return (math.degrees(2 * math.atan(self.sensor_width_mm / (2 * f))),
                math.degrees(2 * math.atan(self.sensor_height_mm / (2 * f))))

    def ifov_mrad(self, zoom: float | None, focal_len_mm: float | None = None) -> float:
        """Instantaneous field of view of one pixel (thermal spec sheets quote this)."""
        f = self.focal_mm(zoom, focal_len_mm)
        return 1000.0 * (self.sensor_width_mm / self.image_width) / f


@dataclass(frozen=True, slots=True)
class PTGeoMount:
    """A pan-tilt head placed on the map."""

    camera_id: str
    lat: float
    lon: float
    alt_m: float                     # mount altitude (ellipsoidal, same datum as targets)
    height_above_ground_m: float     # for flat-terrain intersection when no range is available
    north_offset_deg: float = 0.0    # true bearing when the head reads azimuth 0
    azimuth_clockwise: bool = True
    elevation_down_positive: bool = True
    tilt_bias_deg: float = 0.0       # inclinometer (lookDownUpAngle) minus motor tilt
    sigma_azimuth_deg: float = 0.05
    sigma_elevation_deg: float = 0.05
    sigma_range_m: float = 1.0       # laser rangefinder; device distance estimates are much worse
    target_height_m: float = 0.0     # height of the reference point above ground (feet = 0)


@dataclass(frozen=True, slots=True)
class GeoObservation:
    enu: F64                         # (e, n, u) metres in the site frame
    cov_en: F64                      # 2x2 covariance of (e, n)
    lat: float
    lon: float
    bearing_deg: float
    depression_deg: float
    range_m: float
    horizontal_m: float
    method: str                      # "range" | "terrain"


class PTGeolocator:
    def __init__(self, mount: PTGeoMount, site: LocalTangentPlane) -> None:
        self.mount = mount
        self.site = site
        self.mount_enu = site.to_enu(mount.lat, mount.lon, mount.alt_m)

    def true_bearing(self, azimuth_reading: float) -> float:
        az = azimuth_reading if self.mount.azimuth_clockwise else -azimuth_reading
        return (self.mount.north_offset_deg + az) % 360.0

    def depression(self, elevation_reading: float) -> float:
        el = elevation_reading if self.mount.elevation_down_positive else -elevation_reading
        return el + self.mount.tilt_bias_deg

    def locate(self, azimuth: float, elevation: float, *, range_m: float | None = None,
               rect: tuple[float, float, float, float] | None = None, lens: LensModel | None = None,
               zoom: float | None = None, focal_len_mm: float | None = None) -> GeoObservation:
        """Geolocate the foot point of ``rect`` (x, y, w, h normalised) seen at the given pose.

        Without ``rect`` the optical axis is used (e.g. a laser-ranged point at the centre).
        """
        m = self.mount
        d_az = d_el = 0.0
        pix_sigma = 0.0
        if rect is not None and lens is not None:
            hfov, vfov = lens.fov_deg(zoom, focal_len_mm)
            u = rect[0] + rect[2] / 2 - 0.5               # foot point: bottom-centre of the box
            v = rect[1] + rect[3] - 0.5
            f = 0.5 / math.tan(math.radians(hfov) / 2)    # focal length in normalised width units
            d_az = math.degrees(math.atan2(u, f))
            fv = 0.5 / math.tan(math.radians(vfov) / 2)
            d_el = math.degrees(math.atan2(v, fv))
            pix_sigma = hfov / lens.image_width            # ~1 px of foot-point uncertainty
        bearing = (self.true_bearing(azimuth) + d_az) % 360.0
        dep = self.depression(elevation) + d_el
        # Vertical offset of the target reference point relative to the lens (negative = below).
        dz = m.target_height_m - m.height_above_ground_m
        if range_m is not None and range_m > abs(dz):
            horiz = math.sqrt(range_m ** 2 - dz ** 2)
            method = "range"
            sr = m.sigma_range_m
        else:
            if dep <= 0.2:
                raise ValueError(f"cannot intersect terrain at {dep:.2f} deg depression (above horizon)")
            horiz = (m.height_above_ground_m - m.target_height_m) / math.tan(math.radians(dep))
            range_m = math.hypot(horiz, dz)
            method = "terrain"
            # d(horiz)/d(dep) = -h / sin^2(dep): grazing angles amplify tilt error enormously.
            sr = (m.height_above_ground_m / math.sin(math.radians(dep)) ** 2
                  * math.radians(math.hypot(m.sigma_elevation_deg, pix_sigma)))
        b = math.radians(bearing)
        e = self.mount_enu[0] + horiz * math.sin(b)
        n = self.mount_enu[1] + horiz * math.cos(b)
        u = self.mount_enu[2] + dz
        # Polar -> Cartesian propagation: along-beam = range error, cross-beam = bearing error.
        s_cross = horiz * math.radians(math.hypot(m.sigma_azimuth_deg, pix_sigma))
        rot = np.array([[math.sin(b), math.cos(b)], [math.cos(b), -math.sin(b)]])
        cov = rot @ np.diag([sr ** 2, max(s_cross, 0.05) ** 2]) @ rot.T
        lat, lon, _ = self.site.to_geodetic(e, n, u)
        return GeoObservation(np.array([e, n, u]), cov, lat, lon, bearing, dep, float(range_m), horiz, method)

    def aim(self, e: float, n: float, u: float | None = None, *, aim_height_m: float = 1.0
            ) -> tuple[float, float, float]:
        """Inverse: (azimuth reading, elevation reading, slant range) to look at an ENU point."""
        m = self.mount
        ground_u = self.mount_enu[2] - m.height_above_ground_m
        tu = (ground_u + aim_height_m) if u is None else u
        de, dn, du = e - self.mount_enu[0], n - self.mount_enu[1], tu - self.mount_enu[2]
        horiz = math.hypot(de, dn)
        bearing = bearing_deg(de, dn)
        az = (bearing - m.north_offset_deg) % 360.0
        if not m.azimuth_clockwise:
            az = (-az) % 360.0
        dep = math.degrees(math.atan2(-du, horiz)) - m.tilt_bias_deg
        el = dep if m.elevation_down_positive else -dep
        return az, el, math.hypot(horiz, du)


def calibrate_north(mount: PTGeoMount, site: LocalTangentPlane,
                    sightings: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Estimate ``north_offset_deg`` from landmark sightings.

    ``sightings`` = [(azimuth_reading, landmark_lat, landmark_lon), ...]. Returns
    (offset, RMS residual in degrees) using a circular mean, so readings around 0/360
    average correctly. Two well-separated landmarks are enough; residual RMS above
    ~0.2 deg usually means a mis-surveyed mount position, not a bad offset.
    """
    me = site.to_enu(mount.lat, mount.lon, mount.alt_m)
    diffs = []
    for az, lat, lon in sightings:
        p = site.to_enu(lat, lon, 0.0)
        true_b = bearing_deg(p[0] - me[0], p[1] - me[1])
        reading = az if mount.azimuth_clockwise else -az
        diffs.append(math.radians(true_b - reading))
    if not diffs:
        raise ValueError("need at least one landmark sighting")
    c, s = float(np.mean(np.cos(diffs))), float(np.mean(np.sin(diffs)))
    off = math.degrees(math.atan2(s, c)) % 360.0
    res = [((math.degrees(d) - off + 180) % 360 - 180) for d in diffs]
    return off, float(np.sqrt(np.mean(np.square(res))))
