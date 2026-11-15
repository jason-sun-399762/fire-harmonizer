"""Daily FireMask tiles (MYD14A1, VNP14A1, VJ114A1) -> per-cell clear / cloud / fire pixel counts.

MYD14A1 (HDF4) holds 8 days per file; VNP14A1 / VJ114A1 (HDF5) hold one day.
Classes: 0-2 not processed, 3 water, 4 cloud, 5 non-fire land, 6 unknown, 7-9 fire (low/nominal/high).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sinusoidal import parse_date, parse_tile, pixel_centers

VIIRS_FIREMASK = "/HDFEOS/GRIDS/VIIRS_Grid_Daily_Fire/Data Fields/FireMask"  # "Data Fields" has a space in real files


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
    """Pixel -> cell lookup per tile, computed once per tile and kept for the most recent
    `max_tiles` tiles (process files grouped by tile so each tile is built once).
    The local index is stored as uint16 when it fits (~3 MB per 1200 x 1200 tile)."""

    def __init__(self, grid, max_tiles: int = 8):
        from collections import OrderedDict

        self.grid = grid
        self.max_tiles = max_tiles
        self._cache: "OrderedDict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, h: int, v: int, n: int):
        key = (h, v, n)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        lat, lon = pixel_centers(h, v, n)
        ok = np.isfinite(lat).ravel()
        cells = np.full(lat.size, -1, dtype=np.int64)
        cells[ok] = self.grid.cell_of(lat.ravel()[ok], lon.ravel()[ok])
        uniq, inv = np.unique(cells, return_inverse=True)
        inv = inv.astype(np.uint16 if uniq.size < 65536 else np.int32)
        self._cache[key] = (uniq, inv)
        while len(self._cache) > self.max_tiles:
            self._cache.popitem(last=False)
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


METRICS = ("n_valid", "n_clear", "n_land_clear", "n_cloud", "n_fire", "frp_fire_mw")


def class_table(cfg: dict) -> np.ndarray:
    """uint8 FireMask class -> bit flags (valid, clear, land_clear, cloud, fire) as a 256 x 5 table."""
    fm = cfg["firemask"]
    t = np.zeros((256, 5), dtype=np.float64)
    t[3:10, 0] = 1
    for col, key in ((1, "clear"), (2, "land_clear"), (3, "cloud"), (4, "fire")):
        t[fm[key], col] = 1
    return t


def read_firemask_frp(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FireMask plus MaxFRP (MW, scale 0.1) for any supported product: (masks, frp, dates)."""
    name = Path(path).name
    start = parse_date(name)
    if name.upper().startswith(("MYD14A1", "MOD14A1")):
        from pyhdf.SD import SD, SDC

        sd = SD(str(path), SDC.READ)
        try:
            m = np.asarray(sd.select("FireMask")[:], dtype=np.uint8)
            f = np.asarray(sd.select("MaxFRP")[:], dtype=np.float32) * 0.1
        finally:
            sd.end()
        if m.ndim == 2:
            m, f = m[None], f[None]
        return m, f, start + np.arange(m.shape[0]).astype("timedelta64[D]")
    import h5py

    with h5py.File(path, "r") as h:
        m = np.asarray(h[VIIRS_FIREMASK][:], dtype=np.uint8)
        g = VIIRS_FIREMASK.rsplit("/", 1)[0] + "/MaxFRP"
        f = np.asarray(h[g][:], dtype=np.float32) * 0.1 if g in h else np.zeros(m.shape, np.float32)
    return m[None], f[None], np.array([start])


def count_fast(masks: np.ndarray, frp: np.ndarray, h: int, v: int, index: TileCellIndex, table: np.ndarray):
    """Vectorised per-day counts: one bincount per metric over a class lookup.
    Returns (cells, counts[days, 6, ncells]) with metrics in METRICS order."""
    uniq, inv = index.get(h, v, masks.shape[-1])
    nd, nc = masks.shape[0], uniq.size
    out = np.zeros((nd, len(METRICS), nc))
    base = inv.astype(np.int64) * 16
    fire_cls = np.flatnonzero(table[:, 4] > 0)
    for k in range(nd):
        m = masks[k].ravel()
        # one integer pass: pixels per (cell, class); classes are 0-9, 16 slots leaves headroom
        per_class = np.bincount(base + np.minimum(m, 15), minlength=nc * 16).reshape(nc, 16)
        out[k, :5] = (per_class @ table[:16, :5]).T
        fi = np.flatnonzero(np.isin(m, fire_cls))
        if fi.size:
            out[k, 5] = np.bincount(inv[fi], weights=np.nan_to_num(frp[k].ravel()[fi]), minlength=nc)
    keep = uniq >= 0
    return uniq[keep], out[:, :, keep]


def count_file(path: str | Path, index: TileCellIndex, cfg: dict) -> pd.DataFrame:
    masks, dates = read_firemask(path)
    h, v = parse_tile(Path(path).name)
    out = count_tile(masks, dates, h, v, index, cfg)
    out["product"] = Path(path).name.split(".")[0].upper()
    return out
