"""Spatial grids. The pipeline only talks to a Grid, so lat/lon 0.25 deg vs H3 res-5 is a config switch.

Cell ids are int64 for both kinds.
"""

from __future__ import annotations

import numpy as np

# Authalic (equal-area) Earth radius, km. 4*pi*R^2 = 510,065,622 km^2.
R_AUTHALIC_KM = 6371.0072


class LatLonGrid:
    """Regular latitude/longitude grid. Cell id = iy * nx + ix, iy counted from the south pole."""

    kind = "latlon"

    def __init__(self, res_deg: float = 0.25):
        n = 360.0 / res_deg
        if abs(n - round(n)) > 1e-9 or abs(180.0 / res_deg - round(180.0 / res_deg)) > 1e-9:
            raise ValueError(f"res_deg {res_deg} must divide 180 and 360 evenly")
        self.res = float(res_deg)
        self.nx = int(round(360.0 / res_deg))
        self.ny = int(round(180.0 / res_deg))

    @property
    def name(self) -> str:
        return "ll" + f"{self.res:g}".replace(".", "p")

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_of(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lon = ((lon + 180.0) % 360.0) - 180.0  # wrap; +180 becomes -180
        ix = np.floor((lon + 180.0) / self.res).astype(np.int64)
        iy = np.floor((lat + 90.0) / self.res).astype(np.int64)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)  # lat = +90 belongs to the top row
        return iy * self.nx + ix

    def center(self, cell) -> tuple[np.ndarray, np.ndarray]:
        cell = np.asarray(cell, dtype=np.int64)
        iy, ix = np.divmod(cell, self.nx)
        return -90.0 + (iy + 0.5) * self.res, -180.0 + (ix + 0.5) * self.res

    def area_km2(self, cell) -> np.ndarray:
        cell = np.asarray(cell, dtype=np.int64)
        iy = cell // self.nx
        lat0 = np.radians(-90.0 + iy * self.res)
        lat1 = np.radians(-90.0 + (iy + 1) * self.res)
        return R_AUTHALIC_KM**2 * np.radians(self.res) * (np.sin(lat1) - np.sin(lat0))

    def parent(self, cell, parent_res_deg: float) -> np.ndarray:
        """Exact parent on a coarser lat/lon grid (parent_res must be a multiple of res)."""
        k = parent_res_deg / self.res
        if abs(k - round(k)) > 1e-9:
            raise ValueError("parent resolution must be a whole multiple of the cell resolution")
        lat, lon = self.center(cell)
        return LatLonGrid(parent_res_deg).cell_of(lat, lon)


class H3Grid:
    """H3 hexagonal grid at a fixed resolution. Cell ids are H3 indexes as int64."""

    kind = "h3"

    def __init__(self, res: int = 5):
        import h3  # noqa: F401  (fail early if missing)

        self.res = int(res)

    @property
    def name(self) -> str:
        return f"h3r{self.res}"

    def cell_of(self, lat, lon) -> np.ndarray:
        import h3

        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        out = np.fromiter(
            (h3.str_to_int(h3.latlng_to_cell(a, b, self.res)) for a, b in zip(lat, lon)),
            dtype=np.int64,
            count=lat.size,
        )
        return out

    def center(self, cell) -> tuple[np.ndarray, np.ndarray]:
        import h3

        pts = [h3.cell_to_latlng(h3.int_to_str(int(c))) for c in np.atleast_1d(cell)]
        arr = np.asarray(pts)
        return arr[:, 0], arr[:, 1]

    def area_km2(self, cell) -> np.ndarray:
        import h3

        return np.asarray([h3.cell_area(h3.int_to_str(int(c)), "km^2") for c in np.atleast_1d(cell)])

    def parent(self, cell, parent_res: int) -> np.ndarray:
        import h3

        return np.asarray(
            [h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), parent_res)) for c in np.atleast_1d(cell)],
            dtype=np.int64,
        )


def make_grid(cfg: dict):
    g = cfg["grid"]
    if g["kind"] == "latlon":
        return LatLonGrid(g["res_deg"])
    if g["kind"] == "h3":
        return H3Grid(g["h3_res"])
    raise ValueError(f"unknown grid kind {g['kind']!r}")


def spatial_block(grid, cell, cfg: dict) -> np.ndarray:
    """Coarse block id for spatial cross-validation."""
    if grid.kind == "latlon":
        return grid.parent(cell, cfg["cv"]["block_latlon_deg"])
    return grid.parent(cell, cfg["cv"]["block_h3_res"])
