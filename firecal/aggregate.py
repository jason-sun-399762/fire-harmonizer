"""Raster pixels -> grid cells, with one rule per variable type (protocol, section 3).

Every result carries the valid-coverage fraction; means also carry the within-cell spread.
Weights are pixel areas: constant for sinusoidal products, proportional to cos(lat) on lat/lon rasters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group(cells: np.ndarray):
    uniq, inv = np.unique(cells, return_inverse=True)
    return uniq, inv


def intensive(cells, values, weights=None) -> pd.DataFrame:
    """Area-weighted mean over valid (finite) pixels, with std and valid fraction."""
    cells = np.asarray(cells).ravel()
    x = np.asarray(values, dtype=float).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    uniq, inv = _group(cells)
    valid = np.isfinite(x)
    wv = np.where(valid, w, 0.0)
    xv = np.where(valid, x, 0.0)
    sw_all = np.bincount(inv, weights=w, minlength=uniq.size)
    sw = np.bincount(inv, weights=wv, minlength=uniq.size)
    sx = np.bincount(inv, weights=wv * xv, minlength=uniq.size)
    sxx = np.bincount(inv, weights=wv * xv * xv, minlength=uniq.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sx / sw
        var = np.maximum(sxx / sw - mean**2, 0.0)
    return pd.DataFrame({"cell": uniq, "mean": mean, "std": np.sqrt(var), "valid_frac": sw / sw_all})


def extensive(cells, values, weights=None) -> pd.DataFrame:
    """Sum of an amount (counts, km^2, people). weights = fraction of the pixel inside the cell."""
    cells = np.asarray(cells).ravel()
    x = np.asarray(values, dtype=float).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    uniq, inv = _group(cells)
    valid = np.isfinite(x)
    total = np.bincount(inv, weights=np.where(valid, x * w, 0.0), minlength=uniq.size)
    vf = np.bincount(inv, weights=valid * w, minlength=uniq.size) / np.bincount(inv, weights=w, minlength=uniq.size)
    n = np.bincount(inv, weights=valid.astype(float), minlength=uniq.size)
    return pd.DataFrame({"cell": uniq, "sum": total, "n_pixels": n.astype(np.int64), "valid_frac": vf})


def categorical(cells, classes, weights=None, class_values=None) -> pd.DataFrame:
    """Area fraction of each class among valid pixels; one column per class (frac_<value>)."""
    cells = np.asarray(cells).ravel()
    c = np.asarray(classes).ravel()
    w = np.ones(c.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
    valid = c >= 0
    if class_values is None:
        class_values = np.unique(c[valid])
    uniq, inv = _group(cells)
    sw_all = np.bincount(inv, weights=w, minlength=uniq.size)
    sw = np.bincount(inv, weights=w * valid, minlength=uniq.size)
    out = {"cell": uniq}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in class_values:
            out[f"frac_{k}"] = np.bincount(inv, weights=w * (c == k), minlength=uniq.size) / sw
        out["valid_frac"] = sw / sw_all
    return pd.DataFrame(out)


def circular(cells, doy, weights=None, period: float = 365.25) -> pd.DataFrame:
    """Circular mean of day-of-year (or aspect with period=360) and resultant length R in [0, 1]."""
    cells = np.asarray(cells).ravel()
    d = np.asarray(doy, dtype=float).ravel()
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float).ravel()
    valid = np.isfinite(d)
    ang = 2 * np.pi * np.where(valid, d, 0.0) / period
    wv = np.where(valid, w, 0.0)
    uniq, inv = _group(cells)
    sw = np.bincount(inv, weights=wv, minlength=uniq.size)
    c = np.bincount(inv, weights=wv * np.cos(ang), minlength=uniq.size)
    s = np.bincount(inv, weights=wv * np.sin(ang), minlength=uniq.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.arctan2(s, c) % (2 * np.pi)) * period / (2 * np.pi)
        r = np.hypot(c, s) / sw
    vf = sw / np.bincount(inv, weights=w, minlength=uniq.size)
    return pd.DataFrame({"cell": uniq, "circ_mean": mean, "resultant": r, "valid_frac": vf})


def latlon_raster_points(values: np.ndarray, lat_edges: np.ndarray, lon_edges: np.ndarray, supersample: int = 1):
    """Turn a regular lat/lon raster into weighted points for the rules above.

    supersample > 1 splits each source pixel into k x k sub-points, a simple conservative
    remap for sources coarser than the target cells (GLDAS, GFWED). Weights are true areas.
    Returns (lat, lon, values, weights) as flat arrays.
    """
    k = int(supersample)
    lat_e = np.asarray(lat_edges, dtype=float)
    lon_e = np.asarray(lon_edges, dtype=float)
    ny, nx = values.shape
    fy = (np.arange(ny * k) + 0.5) / k
    fx = (np.arange(nx * k) + 0.5) / k
    lat_c = np.interp(fy, np.arange(ny + 1), lat_e)
    lon_c = np.interp(fx, np.arange(nx + 1), lon_e)
    lat0 = np.interp(np.arange(ny * k) / k, np.arange(ny + 1), lat_e)
    lat1 = np.interp((np.arange(ny * k) + 1) / k, np.arange(ny + 1), lat_e)
    dlon = np.abs(np.diff(lon_e)).repeat(k) / k
    band = np.abs(np.sin(np.radians(lat1)) - np.sin(np.radians(lat0)))
    w = np.outer(band, np.radians(dlon))
    vals = np.repeat(np.repeat(values, k, axis=0), k, axis=1)
    LON, LAT = np.meshgrid(lon_c, lat_c)
    return LAT.ravel(), LON.ravel(), vals.ravel(), w.ravel()
