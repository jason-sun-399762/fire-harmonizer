"""MODIS/VIIRS sinusoidal tile grid: closed-form pixel centres, no reprojection needed.

The projection is equal-area, so every pixel has the same area.
"""

from __future__ import annotations

import re

import numpy as np

R_SIN = 6371007.181          # sphere radius used by the MODIS sinusoidal grid, m
TILE_M = 1111950.5197665     # tile width/height, m
X0 = -20015109.354           # grid upper-left x, m
Y0 = 10007554.677            # grid upper-left y, m

TILE_RE = re.compile(r"\.h(\d{2})v(\d{2})\.")
DATE_RE = re.compile(r"\.A(\d{4})(\d{3})\.")


def pixel_centers(h: int, v: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon (deg) of every pixel centre in tile (h, v) with n x n pixels.

    Pixels that fall off the Earth (tile corners near the edge of the grid) get NaN.
    """
    size = TILE_M / n
    j = np.arange(n)
    x = X0 + h * TILE_M + (j + 0.5) * size
    y = Y0 - v * TILE_M - (j + 0.5) * size
    xx, yy = np.meshgrid(x, y)          # rows = y (north to south), cols = x
    lat = yy / R_SIN
    with np.errstate(divide="ignore", invalid="ignore"):
        lon = xx / (R_SIN * np.cos(lat))
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon)
    bad = (np.abs(lon_d) > 180.0) | ~np.isfinite(lon_d)
    lat_d[bad] = np.nan
    lon_d[bad] = np.nan
    return lat_d, lon_d


def pixel_area_km2(n: int) -> float:
    return (TILE_M / n / 1000.0) ** 2


def parse_tile(filename: str) -> tuple[int, int]:
    m = TILE_RE.search(filename)
    if not m:
        raise ValueError(f"no hXXvYY tile id in {filename!r}")
    return int(m.group(1)), int(m.group(2))


def parse_date(filename: str):
    """AYYYYDDD in a product filename -> numpy datetime64[D]."""
    m = DATE_RE.search(filename)
    if not m:
        raise ValueError(f"no AYYYYDDD date in {filename!r}")
    year, doy = int(m.group(1)), int(m.group(2))
    return np.datetime64(f"{year}-01-01") + np.timedelta64(doy - 1, "D")
