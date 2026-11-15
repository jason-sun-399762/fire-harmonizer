"""Readers for the covariate products in the bucket, returning physical values (NaN = missing)
on a RasterSpec, plus `to_cells` helpers that regrid them with the protocol's rules.

Products: VNP13C2 / VJ113C2 (0.05 deg monthly NDVI, EVI), VNP22C2 (0.05 deg yearly phenology),
MCD12C1 (0.05 deg yearly land-cover percentages), GLDAS_NOAH025_M (0.25 deg monthly weather),
GFWED (MERRA-2 0.5 x 0.625 deg daily, GEOS-5 0.25 x 0.3125 deg daily fire weather).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .rasters import CMG_005, RasterSpec, regridder

VI_GROUP = "HDFEOS/GRIDS/VIIRS_Grid_monthly_VI_CMG/Data Fields/"
PHENO_GROUP = "HDFEOS/GRIDS/Cycle 1/Data Fields/"
PHENO_FILL = 32767
IGBP_NAMES = ["water", "enf", "ebf", "dnf", "dbf", "mixed_forest", "closed_shrub", "open_shrub", "woody_savanna",
              "savanna", "grassland", "wetland", "cropland", "urban", "crop_mosaic", "snow_ice", "barren"]


def product_year(path) -> int:
    m = re.search(r"\.A(\d{4})(\d{3})\.", Path(path).name)
    return int(m.group(1))


def product_ym(path) -> int:
    """yyyymm from AYYYYDDD (VIIRS/MODIS) or AYYYYMM (GLDAS) or .YYYYMMDD. (GFWED) file names."""
    name = Path(path).name
    if m := re.search(r"\.A(\d{4})(\d{3})\.", name):
        d = np.datetime64(f"{m.group(1)}-01-01") + np.timedelta64(int(m.group(2)) - 1, "D")
        y, mo = str(d)[:4], str(d)[5:7]
        return int(y) * 100 + int(mo)
    if m := re.search(r"\.A(\d{4})(\d{2})\.", name):
        return int(m.group(1)) * 100 + int(m.group(2))
    if m := re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc$", name):
        return int(m.group(1)) * 100 + int(m.group(2))
    raise ValueError(f"no date in {name}")


def _scaled(raw: np.ndarray, fills, valid_range, scale: float) -> np.ndarray:
    x = raw.astype(np.float64)
    bad = np.isin(raw, np.atleast_1d(fills)) | (raw < valid_range[0]) | (raw > valid_range[1])
    x[bad] = np.nan
    return x * scale


# ---------------- VIIRS 0.05 deg vegetation indices ----------------

def read_vi_cmg(path, fields=("NDVI", "EVI")) -> dict[str, np.ndarray]:
    import h5py

    out = {}
    with h5py.File(path, "r") as h:
        for f in fields:
            ds = h[VI_GROUP + f"CMG 0.05 Deg monthly {f}"]
            out[f.lower()] = _scaled(ds[:], ds.attrs["_FillValue"], ds.attrs["valid_range"], float(ds.attrs["scale_factor"][0]))
        rel = h[VI_GROUP + "CMG 0.05 Deg monthly pixel reliability"][:]
        out["reliability"] = np.where(rel >= 0, rel, np.nan).astype(float)
    return out


# ---------------- VIIRS 0.05 deg phenology ----------------

PHENO_FIELDS = {
    "greenup_doy": "Onset_Greenness_Increase_1",
    "senescence_doy": "Onset_Greenness_Decrease_1",
    "dormancy_doy": "Onset_Greenness_Minimum_1",
    "season_length": "Growing_Season_Length_1",
}


def read_phenology_cmg(path) -> dict[str, np.ndarray]:
    """Dates are stored as days counted in 366-day years from 2000; convert to day of the product year
    (values outside 1-366 mean the event fell in the previous / next year, kept as is)."""
    import h5py

    year = product_year(path)
    out = {}
    with h5py.File(path, "r") as h:
        for name, field in PHENO_FIELDS.items():
            raw = h[PHENO_GROUP + field][:].astype(np.float64)
            raw[raw == PHENO_FILL] = np.nan
            out[name] = raw if name == "season_length" else raw - 366 * (year - 2000)
    return out


# ---------------- MODIS 0.05 deg land cover ----------------

def read_landcover_cmg(path) -> np.ndarray:
    """IGBP class percentages as uint8, shape (3600, 7200, 17); 255 = unmapped. Kept as integers
    (~440 MB) because a float copy of a global file is ~1.8 GB."""
    from pyhdf.SD import SD, SDC

    sd = SD(str(path), SDC.READ)
    try:
        return np.asarray(sd.select("Land_Cover_Type_1_Percent")[:], dtype=np.uint8)
    finally:
        sd.end()


# ---------------- GLDAS monthly weather ----------------

def read_gldas(path) -> tuple[dict[str, np.ndarray], RasterSpec]:
    """Monthly means plus derived RH and VPD (computed at 0.25 deg before any regridding)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        spec, flip = RasterSpec.from_centers(ds["lat"].values, ds["lon"].values)
        get = lambda v: (ds[v].values[0][::-1] if flip else ds[v].values[0]).astype(np.float64)  # noqa: E731
        t, q, p = get("Tair_f_inst"), get("Qair_f_inst"), get("Psurf_f_inst")
        out = {
            "tair_c": t - 273.15,
            "wind_ms": get("Wind_f_inst"),
            "precip_mm_day": get("Rainf_f_tavg") * 86400.0,
            "soilmoist_0_10_kgm2": get("SoilMoi0_10cm_inst"),
            "rootmoist_kgm2": get("RootMoist_inst"),
        }
    e = q * p / (0.622 + 0.378 * q)                                   # vapour pressure, Pa
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))           # saturation (Bolton 1980), Pa
    out["rh_pct"] = np.clip(100.0 * e / es, 0, 100)
    out["vpd_kpa"] = np.maximum(es - e, 0) / 1000.0
    return out, spec


# ---------------- GFWED daily fire weather ----------------

GFWED_VARS = ("FWI", "FFMC", "DMC", "DC", "ISI", "BUI")


def read_gfwed_daily(path, variables=GFWED_VARS) -> tuple[dict[str, np.ndarray], RasterSpec, str]:
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        prefix = next(v.split("_")[0] for v in ds.data_vars)            # MERRA2 or GEOS-5
        spec, flip = RasterSpec.from_centers(ds["lat"].values, ds["lon"].values)
        out = {}
        for v in variables:
            a = ds[f"{prefix}_{v}"].values[0].astype(np.float64)
            out[v.lower()] = a[::-1] if flip else a
    return out, spec, prefix


def gfwed_month(paths, threshold: float = 30.0) -> tuple[dict[str, np.ndarray], RasterSpec]:
    """One month of daily GFWED files -> per-pixel summaries on the source grid:
    FWI mean, 90th percentile and days above `threshold`; monthly means of the other components."""
    stacks: dict[str, list] = {v.lower(): [] for v in GFWED_VARS}
    spec = None
    for p in sorted(paths):
        d, s, _ = read_gfwed_daily(p)
        if spec is not None and s != spec:
            raise ValueError("GFWED files of one month have different grids")
        spec = s
        for k, a in d.items():
            stacks[k].append(a)
    fwi = np.stack(stacks["fwi"])
    with _quiet():
        out = {
            "fwi_mean": np.nanmean(fwi, axis=0),
            "fwi_p90": np.nanpercentile(fwi, 90, axis=0),
            f"fwi_days_gt{int(threshold)}": np.where(np.isfinite(fwi).any(0), np.sum(fwi > threshold, axis=0), np.nan).astype(float),
            "fwi_ndays": np.where(np.isfinite(fwi).any(0), np.isfinite(fwi).sum(0), np.nan).astype(float),
        }
        for v in ("ffmc", "dmc", "dc", "isi", "bui"):
            out[f"{v}_mean"] = np.nanmean(np.stack(stacks[v]), axis=0)
    return out, spec


def read_gfwed_monthly(path) -> tuple[dict[str, np.ndarray], RasterSpec]:
    """Monthly-mean GFWED file (fallback when daily files are missing): means only."""
    d, spec, _ = read_gfwed_daily(path)
    return {f"{k}_mean": v for k, v in d.items()}, spec


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def auto_supersample(spec: RasterSpec, grid, max_k: int = 20) -> int:
    """Smallest k such that k x k sub-pixels line up with every lat/lon cell edge (exact overlap).
    For H3 (edges are not lat/lon lines) use 4 sub-pixels per side as a close approximation."""
    if grid.kind != "latlon":
        return 4

    def aligned(step, origin, res):
        return abs(res / step - round(res / step)) < 1e-6 and abs(origin / step - round(origin / step)) < 1e-6

    for k in range(1, max_k + 1):
        if aligned(spec.dlat / k, spec.lat0, grid.res) and aligned(spec.dlon / k, spec.lon0, grid.res):
            return k
    return max_k


# ---------------- to grid cells ----------------

def to_cells(fields: dict[str, np.ndarray], spec: RasterSpec, grid, supersample: int | None = None,
             circular: tuple[str, ...] = (), circular_period: float = 365.25) -> pd.DataFrame:
    """Regrid every field (intensive rule: area-weighted mean, std, valid fraction).
    Fields named in `circular` (day-of-year) are averaged on the circle instead."""
    k = supersample or auto_supersample(spec, grid)
    rg = regridder(spec, grid, k)
    out = None
    for name, arr in fields.items():
        if name in circular:
            ang = 2 * np.pi * arr / circular_period
            c = rg.mean(np.cos(ang)).set_index("cell")
            s = rg.mean(np.sin(ang)).set_index("cell")
            df = pd.DataFrame(
                {
                    f"{name}": (np.arctan2(s["mean"], c["mean"]) % (2 * np.pi)) * circular_period / (2 * np.pi),
                    f"{name}_R": np.hypot(s["mean"], c["mean"]),
                    f"{name}_valid": c["valid_frac"],
                }
            )
        else:
            m = rg.mean(arr).set_index("cell")
            df = m.rename(columns={"mean": name, "std": f"{name}_std", "valid_frac": f"{name}_valid"})
        out = df if out is None else out.join(df, how="outer")
    return out.reset_index()


def landcover_to_cells(pct: np.ndarray, grid, groups: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """IGBP percentages (ny, nx, 17) -> per-cell area fractions, merged into groups (default: one
    column per class). Every column is an area-weighted mean of the group's share, so columns sum to
    1 with water. Also returns land_frac = 1 - water fraction."""
    groups = groups or {n: [i] for i, n in enumerate(IGBP_NAMES)}
    rg = regridder(CMG_005, grid, auto_supersample(CMG_005, grid))
    unmapped = pct[..., 0] == 255
    cols = {}
    for name, idx in {"water": [0], **groups}.items():
        share = pct[..., idx].sum(axis=-1, dtype=np.uint16).astype(np.float32) / 100.0
        share[unmapped] = np.nan
        cols[name] = rg.mean(share).set_index("cell")
    base = pd.DataFrame({"lc_valid": cols["water"]["valid_frac"]})
    for name, df in cols.items():
        if name != "water" or "water" in groups:
            base[f"lc_{name}"] = df["mean"]
    base["land_frac"] = 1.0 - cols["water"]["mean"]
    return base.reset_index()


LC_GROUPS = {  # short list used by the model (protocol: map classes to a few physical groups)
    "forest": [1, 2, 3, 4, 5],
    "shrub": [6, 7],
    "savanna": [8, 9],
    "grass": [10],
    "wetland": [11],
    "crop": [12, 14],
    "urban": [13],
    "barren_snow": [15, 16],
}
