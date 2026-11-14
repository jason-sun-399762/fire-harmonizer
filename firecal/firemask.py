"""Daily FireMask tiles (MYD14A1, VNP14A1, VJ114A1) -> per-cell clear / cloud / fire pixel counts.

MYD14A1 (HDF4) holds 8 days per file; VNP14A1 / VJ114A1 (HDF5) hold one day.
Classes: 0-2 not processed, 3 water, 4 cloud, 5 non-fire land, 6 unknown, 7-9 fire (low/nominal/high).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sinusoidal import parse_date, parse_tile, pixel_centers

VIIRS_FIREMASK = "/HDFEOS/GRIDS/VIIRS_Grid_Daily_Fire/Data_Fields/FireMask"


def read_firemask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (masks[days, n, n] uint8, dates[days] datetime64[D]) for any supported product."""
    name = Path(path).name
    start = parse_date(name)
    if name.upper().startswith(("MYD14A1", "MOD14A1")):
        from pyhdf.SD import SD, SDC

        sd = SD(str(path), SDC.READ)
        try:
            arr = np.asarray(sd.select("FireMask")[:], dtype=np.uint8)
        finally:
            sd.end()
        if arr.ndim == 2:
            arr = arr[None]
        dates = start + np.arange(arr.shape[0]).astype("timedelta64[D]")
        return arr, dates
    if name.upper().startswith(("VNP14A1", "VJ114A1", "VJ214A1")):
        import h5py

        with h5py.File(path, "r") as f:
            arr = np.asarray(f[VIIRS_FIREMASK][:], dtype=np.uint8)
        return arr[None], np.array([start])
    raise ValueError(f"unrecognised FireMask product: {name}")


class TileCellIndex:
    """Cache of pixel -> cell ids for one tile, so each tile's geometry is computed once."""

    def __init__(self, grid):
        self.grid = grid
        self._cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, h: int, v: int, n: int):
        key = (h, v, n)
        if key not in self._cache:
            lat, lon = pixel_centers(h, v, n)
            ok = np.isfinite(lat).ravel()
            cells = np.full(lat.size, -1, dtype=np.int64)
            cells[ok] = self.grid.cell_of(lat.ravel()[ok], lon.ravel()[ok])
            uniq, inv = np.unique(cells, return_inverse=True)
            self._cache[key] = (uniq, inv)
        return self._cache[key]


def count_tile(masks: np.ndarray, dates: np.ndarray, h: int, v: int, index: TileCellIndex, cfg: dict) -> pd.DataFrame:
    """Per cell x day pixel counts: n_valid, n_clear, n_land_clear, n_cloud, n_fire."""
    fm = cfg["firemask"]
    n = masks.shape[-1]
    uniq, inv = index.get(h, v, n)
    groups = {
        "n_clear": np.isin(masks, fm["clear"]),
        "n_land_clear": np.isin(masks, fm["land_clear"]),
        "n_cloud": np.isin(masks, fm["cloud"]),
        "n_fire": np.isin(masks, fm["fire"]),
        "n_valid": masks >= 3,
    }
    frames = []
    for k in range(masks.shape[0]):
        cols = {name: np.bincount(inv, weights=g[k].ravel(), minlength=uniq.size) for name, g in groups.items()}
        df = pd.DataFrame(cols)
        df["cell"] = uniq
        df["date"] = dates[k]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["cell"] >= 0) & (out["n_valid"] > 0)]
    for c in groups:
        out[c] = out[c].astype(np.int32)
    return out.reset_index(drop=True)


def count_file(path: str | Path, index: TileCellIndex, cfg: dict) -> pd.DataFrame:
    masks, dates = read_firemask(path)
    h, v = parse_tile(Path(path).name)
    out = count_tile(masks, dates, h, v, index, cfg)
    out["product"] = Path(path).name.split(".")[0].upper()
    return out
