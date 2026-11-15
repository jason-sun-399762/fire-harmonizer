"""Regular lat/lon rasters -> grid cells through a cached sparse overlap matrix.

The matrix W[target cell, source pixel] holds true overlap areas (sphere, steradians), built once
per (source grid, target grid) by splitting each source pixel into k x k sub-pixels. With aligned
grids (0.05 -> 0.25, 0.25 -> 0.25) k = 1 is exact; GFWED's 0.5 x 0.625 needs k = 10 to be exact.
Applying it to a new month is three sparse products, so hundreds of time steps stay cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class RasterSpec:
    """Regular lat/lon raster: north-up rows starting at lat0 (top edge), columns from lon0 (left edge)."""

    ny: int
    nx: int
    lat0: float      # top edge latitude of row 0
    lon0: float      # left edge longitude of column 0
    dlat: float      # row height, degrees (positive)
    dlon: float      # column width, degrees

    @classmethod
    def from_centers(cls, lat_c, lon_c) -> tuple["RasterSpec", bool]:
        """Build from 1-D cell-centre arrays. Returns (spec, flip) where flip=True means the
        data rows run south-to-north and must be flipped to north-up."""
        lat_c = np.asarray(lat_c, float)
        lon_c = np.asarray(lon_c, float)
        flip = lat_c[0] < lat_c[-1]
        lat_n = lat_c[::-1] if flip else lat_c
        dlat = abs(lat_n[0] - lat_n[1])
        dlon = abs(lon_c[1] - lon_c[0])
        return cls(lat_c.size, lon_c.size, lat_n[0] + dlat / 2, lon_c[0] - dlon / 2, dlat, dlon), flip


CMG_005 = RasterSpec(3600, 7200, 90.0, -180.0, 0.05, 0.05)  # MODIS/VIIRS 0.05 deg climate-modelling grid


class Regridder:
    def __init__(self, spec: RasterSpec, grid, supersample: int = 1, max_points_per_chunk: int = 4_000_000):
        self.spec, self.grid, self.k = spec, grid, int(supersample)
        rows, cols, vals = [], [], []
        k = self.k
        chunk_rows = max(1, max_points_per_chunk // (spec.nx * k * k))
        for r0 in range(0, spec.ny, chunk_rows):
            r1 = min(spec.ny, r0 + chunk_rows)
            sub_r = np.arange(r0 * k, r1 * k)
            top = spec.lat0 - sub_r * spec.dlat / k
            bot = top - spec.dlat / k
            lat_c = (top + bot) / 2
            band = np.abs(np.sin(np.radians(top)) - np.sin(np.radians(bot)))  # area per unit longitude
            sub_c = np.arange(spec.nx * k)
            lon_c = spec.lon0 + (sub_c + 0.5) * spec.dlon / k
            LAT, LON = np.meshgrid(lat_c, lon_c, indexing="ij")
            W = np.repeat(band[:, None] * np.radians(spec.dlon / k), sub_c.size, axis=1)
            ok = np.abs(LAT) <= 90
            cell = np.full(LAT.shape, -1, dtype=np.int64)
            cell[ok] = grid.cell_of(LAT[ok], LON[ok])
            src = (sub_r[:, None] // k) * spec.nx + (sub_c[None, :] // k)
            # merge sub-pixels that share (cell, source pixel) before storing: keeps memory ~ #pairs
            part = (pd.DataFrame({"c": cell.ravel(), "s": src.ravel(), "w": W.ravel()})
                    .groupby(["c", "s"], sort=False)["w"].sum().reset_index())
            rows.append(part["c"].to_numpy())
            cols.append(part["s"].to_numpy())
            vals.append(part["w"].to_numpy())
        cell = np.concatenate(rows)
        src = np.concatenate(cols)
        w = np.concatenate(vals)
        keep = cell >= 0
        self.cells, tgt = np.unique(cell[keep], return_inverse=True)
        self.W = sp.csr_matrix((w[keep], (tgt, src[keep])), shape=(self.cells.size, spec.ny * spec.nx))
        self.W.sum_duplicates()
        self.total = np.asarray(self.W.sum(axis=1)).ravel()

    def mean(self, data: np.ndarray) -> pd.DataFrame:
        """Area-weighted mean over valid (finite) source pixels, with std and valid fraction."""
        x = np.asarray(data, float).ravel()
        v = np.isfinite(x)
        xv = np.where(v, x, 0.0)
        sw = self.W @ v.astype(float)
        sx = self.W @ xv
        sxx = self.W @ (xv * xv)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = sx / sw
            sd = np.sqrt(np.maximum(sxx / sw - m * m, 0.0))
        out = pd.DataFrame({"cell": self.cells, "mean": m, "std": sd, "valid_frac": sw / self.total})
        return out[out["valid_frac"] > 0].reset_index(drop=True)

    def sum_density(self, density_per_sr: np.ndarray) -> pd.DataFrame:
        """For extensive amounts given as density per unit area: integrate over each cell."""
        x = np.nan_to_num(np.asarray(density_per_sr, float).ravel())
        return pd.DataFrame({"cell": self.cells, "sum": self.W @ x})


def source_area_mean(data: np.ndarray, spec: RasterSpec) -> float:
    """Area-weighted mean of a raster's valid pixels (for the regridding conservation check)."""
    top = spec.lat0 - np.arange(spec.ny) * spec.dlat
    band = np.abs(np.sin(np.radians(top)) - np.sin(np.radians(np.maximum(top - spec.dlat, -90.0))))
    w = np.broadcast_to(band[:, None], (spec.ny, spec.nx))
    ok = np.isfinite(data)
    return float(np.average(data[ok], weights=w[ok])) if ok.any() else float("nan")


def cells_area_mean(df: pd.DataFrame, col: str, valid_col: str, grid) -> float:
    """Area-weighted mean over cells, weighting by cell area x valid fraction. Exact when the source
    raster covers whole cells; use regridded_area_mean where the raster edge cuts cells."""
    ok = np.isfinite(df[col].to_numpy())
    w = grid.area_km2(df["cell"].to_numpy()[ok]) * df[valid_col].to_numpy()[ok]
    return float(np.average(df[col].to_numpy()[ok], weights=w)) if ok.any() else float("nan")


def regridded_area_mean(rg: "Regridder", data: np.ndarray, df: pd.DataFrame, col: str) -> float:
    """Area-weighted mean over cells using the valid source area inside each cell as weight.
    Equals source_area_mean exactly when regridding is conservative, including partly covered cells."""
    sw = pd.Series(rg.W @ np.isfinite(np.asarray(data, float).ravel()).astype(float), index=rg.cells)
    v = df.set_index("cell")[col]
    w = sw.reindex(v.index).to_numpy()
    ok = np.isfinite(v.to_numpy()) & (w > 0)
    return float(np.average(v.to_numpy()[ok], weights=w[ok])) if ok.any() else float("nan")


_CACHE: dict = {}


def regridder(spec: RasterSpec, grid, supersample: int = 1) -> Regridder:
    key = (spec, grid.name, supersample)
    if key not in _CACHE:
        _CACHE[key] = Regridder(spec, grid, supersample)
    return _CACHE[key]
