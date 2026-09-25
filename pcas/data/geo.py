"""Geodetic coordinates to a local, runway-aligned metric frame.

TrajAir's `processed_data` already ships in an airport-centred frame, so this module is
for everything else: TrajAir's `raw_data`, VATSIM, and OpenSky. Models should never see
lat/lon - a degree of longitude is ~85 km at Pittsburgh's latitude and ~111 km at the
equator, so degrees are not a distance and a network trained on them learns nonsense.

Convention here matches TrajAir's: origin at the airport reference point, x along the
runway, and (x, y, z) right-handed with z up - so y points to the *left* of the runway
heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

FEET_TO_METERS = 0.3048


def geodetic_to_ecef(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic to earth-centred, earth-fixed metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)

    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    # Radius of curvature in the prime vertical.
    n = _A / np.sqrt(1 - _E2 * sin_lat**2)

    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1 - _E2) + alt) * sin_lat
    return x, y, z


@dataclass(frozen=True)
class LocalFrame:
    """A tangent-plane frame centred on an airport, rotated to the runway.

    `runway_heading_deg` is the true heading of the runway (0 = north, clockwise). The
    returned axes are always **runway-relative**: x runs along that heading, y to its
    left, z up. So heading 0 gives (north, west, up) and heading 90 gives (east, north,
    up) - not east/north/up regardless of heading.
    """

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    runway_heading_deg: float = 0.0

    def to_local(
        self, lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Geodetic positions to (x, y, z) metres in this frame."""
        ox, oy, oz = geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)
        px, py, pz = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        dx, dy, dz = px - ox, py - oy, pz - oz

        lat = math.radians(self.lat_deg)
        lon = math.radians(self.lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)

        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return self._rotate(east, north) + (up,)

    def _rotate(self, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate east/north so +x runs along the runway heading, +y to its left."""
        theta = math.radians(self.runway_heading_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        along = east * sin_t + north * cos_t
        left = -east * cos_t + north * sin_t
        return along, left


# Pittsburgh-Butler Regional (KBTP): the TrajAir collection site. Runway 8/26, so the
# "8" direction is roughly 080 true. Airport reference point and field elevation.
KBTP = LocalFrame(
    lat_deg=40.7769,
    lon_deg=-79.9498,
    alt_m=380.0,
    runway_heading_deg=80.0,
)
