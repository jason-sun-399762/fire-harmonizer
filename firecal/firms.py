"""FIRMS hot-spot points: standardize, filter, and aggregate to grid cells.

Accepts two layouts:
  * the raw FIRMS archive (acq_date, acq_time, satellite, confidence, type, ...)
  * the team's bucket layout in clean/firms (sensor, utc_time, conf_ok, conf_high, ...)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .timeutil import add_time_keys, firms_utc, local_solar

CONF_LOW, CONF_NOMINAL, CONF_HIGH = 0, 1, 2
CONF_NAMES = {"low": CONF_LOW, "nominal": CONF_NOMINAL, "high": CONF_HIGH}

MODIS_SATS = {"aqua", "terra"}
VIIRS_SATS = {"snpp", "noaa20", "noaa21"}


def normalize_satellite(values) -> pd.Series:
    """Map the many spellings used by FIRMS and our bucket to aqua/terra/snpp/noaa20/noaa21.

    A bare 'MODIS' (Aqua and Terra merged) maps to 'modis_unknown': the target needs Aqua alone.
    """
    s = pd.Series(values).astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=s.index, dtype=object)
    compact = s.str.replace(r"[\s_\-]", "", regex=True)

    rules = [
        (compact.str.contains("aqua") | (compact == "a"), "aqua"),
        (compact.str.contains("terra") | (compact == "t"), "terra"),
        (compact.str.contains(r"noaa21|n21|jpss2|j2", regex=True) | (compact == "2"), "noaa21"),
        (compact.str.contains(r"noaa20|n20|jpss1|j1", regex=True) | (compact == "1"), "noaa20"),
        (compact.str.contains(r"npp|suomi", regex=True) | (compact == "n"), "snpp"),
    ]
    for mask, name in rules:
        out[(out == "unknown") & mask] = name
    out[(out == "unknown") & compact.str.contains("modis")] = "modis_unknown"
    return out


def instrument_of(sat: pd.Series) -> pd.Series:
    inst = pd.Series("unknown", index=sat.index, dtype=object)
    inst[sat.isin(MODIS_SATS) | (sat == "modis_unknown")] = "modis"
    inst[sat.isin(VIIRS_SATS)] = "viirs"
    return inst


def confidence_class(conf, instrument: pd.Series, modis_cutoffs=(30, 80)) -> pd.Series:
    """MODIS 0-100 -> low/nominal/high by cutoffs; VIIRS l/n/h strings -> same classes."""
    conf = pd.Series(conf, index=instrument.index)
    out = pd.Series(np.int8(-1), index=instrument.index)
    m = instrument == "modis"
    if m.any():
        num = pd.to_numeric(conf[m], errors="coerce")
        lo, hi = modis_cutoffs
        out[m] = np.where(num >= hi, CONF_HIGH, np.where(num >= lo, CONF_NOMINAL, CONF_LOW)).astype(np.int8)
        out[m & num.reindex(out.index).isna()] = -1
    v = instrument == "viirs"
    if v.any():
        first = conf[v].astype(str).str.strip().str.lower().str[0]
        out[v] = first.map({"l": CONF_LOW, "n": CONF_NOMINAL, "h": CONF_HIGH}).fillna(-1).astype(np.int8)
    return out.astype(np.int8)


def standardize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one tidy frame: lat, lon, utc_time, local time keys, satellite, instrument,
    conf_class, frp, daynight, scan, track and (when present) type."""
    d = df.copy()
    lat = d["latitude"].astype(float)
    lon = d["longitude"].astype(float)

    if "utc_time" in d:
        utc = pd.to_datetime(d["utc_time"])
    else:
        utc = firms_utc(d["acq_date"], d["acq_time"])
    utc.index = d.index

    sat_src = d["satellite"] if "satellite" in d else d["sensor"]
    sat = normalize_satellite(sat_src)
    sat.index = d.index
    inst = instrument_of(sat)
    if (sat == "modis_unknown").any():
        warnings.warn(
            "MODIS rows without Aqua/Terra label: the target needs Aqua only. "
            "Re-derive the satellite column before building the target.",
            stacklevel=2,
        )

    if "confidence" in d:
        conf = confidence_class(d["confidence"], inst, tuple(cfg["filters"]["modis_conf_cutoffs"]))
    elif "conf_high" in d or "conf_ok" in d:
        hi = d.get("conf_high", pd.Series(False, index=d.index)).astype(bool)
        ok = d.get("conf_ok", pd.Series(True, index=d.index)).astype(bool)
        conf = pd.Series(np.where(hi, CONF_HIGH, np.where(ok, CONF_NOMINAL, CONF_LOW)), index=d.index).astype(np.int8)
    else:
        conf = pd.Series(np.int8(-1), index=d.index)

    if "daynight" in d:
        dn = d["daynight"].astype(str).str.upper().str[0]
    elif "is_day" in d:
        dn = pd.Series(np.where(d["is_day"].astype(bool), "D", "N"), index=d.index)
    else:
        dn = pd.Series("U", index=d.index)

    out = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "utc_time": utc,
            "local_time": local_solar(utc, lon).set_axis(d.index),
            "satellite": sat,
            "instrument": inst,
            "conf_class": conf,
            "frp": pd.to_numeric(d.get("frp", np.nan), errors="coerce"),
            "daynight": dn,
            "scan": pd.to_numeric(d.get("scan", np.nan), errors="coerce"),
            "track": pd.to_numeric(d.get("track", np.nan), errors="coerce"),
        },
        index=d.index,
    )
    if "type" in d:
        out["type"] = pd.to_numeric(d["type"], errors="coerce").astype("Int8")
    return add_time_keys(out).reset_index(drop=True)


def apply_filters(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Type and confidence filters. Returns the kept rows and a count of what each rule removed."""
    f = cfg["filters"]
    report = {"rows_in": len(df)}
    keep = pd.Series(True, index=df.index)

    if "type" in df:
        t_ok = df["type"].isin(f["firms_types_keep"]) | df["type"].isna()
        report["dropped_type"] = int((~t_ok).sum())
        keep &= t_ok
    else:
        report["dropped_type"] = None  # column absent: rule could not be applied

    min_cls = {k: CONF_NAMES[v] for k, v in f["min_conf_class"].items()}
    thr = df["instrument"].map(min_cls).fillna(0)
    c_ok = (df["conf_class"] < 0) | (df["conf_class"] >= thr)
    report["dropped_confidence"] = int((keep & ~c_ok).sum())
    keep &= c_ok

    out = df[keep].reset_index(drop=True)
    report["rows_out"] = len(out)
    return out, report


def persistent_bins(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Find ~1 km bins lit on many days a year, for several years (gas flares, industry).

    Computed per instrument, because VIIRS sees more flares than MODIS.
    Returns a frame of (instrument, bin_y, bin_x) to drop.
    """
    p = cfg["filters"]["persistent"]
    b = p["pixel_deg"]
    tmp = pd.DataFrame(
        {
            "instrument": df["instrument"],
            "bin_y": np.floor(df["lat"] / b).astype(np.int64),
            "bin_x": np.floor(df["lon"] / b).astype(np.int64),
            "year": df["year"],
            "local_date": df["local_date"],
        }
    )
    days = tmp.groupby(["instrument", "bin_y", "bin_x", "year"])["local_date"].nunique()
    hot_years = (days >= p["min_days_per_year"]).groupby(level=[0, 1, 2]).sum()
    return hot_years[hot_years >= p["min_years"]].reset_index()[["instrument", "bin_y", "bin_x"]]


def drop_persistent(df: pd.DataFrame, bins: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if bins.empty:
        return df
    b = cfg["filters"]["persistent"]["pixel_deg"]
    key = pd.DataFrame(
        {
            "instrument": df["instrument"],
            "bin_y": np.floor(df["lat"] / b).astype(np.int64),
            "bin_x": np.floor(df["lon"] / b).astype(np.int64),
        }
    )
    hit = key.merge(bins.assign(_hit=1), how="left", on=["instrument", "bin_y", "bin_x"])["_hit"].notna().to_numpy()
    return df[~hit].reset_index(drop=True)


def aggregate(df: pd.DataFrame, grid, step: str = "month", pixel_deg: float = 0.01) -> pd.DataFrame:
    """Cell x time x satellite x day/night aggregates.

    n_det        detections (every overpass counts)
    n_pixel_days distinct ~1 km bins x local day x day/night (removes repeat overpasses)

    Fixed bins have an edge effect: two detections of one pixel can straddle a bin border and
    count twice. That is why the MODIS target uses MYD14A1's own pixel grid (pixel_days_l3).
    """
    tkey = ["ym"] if step == "month" else ["year", "week"]
    keys = ["cell", *tkey, "satellite", "daynight"]
    d = df[[*tkey, "satellite", "daynight", "frp", "local_date"]].copy()
    d["cell"] = grid.cell_of(df["lat"].to_numpy(), df["lon"].to_numpy())
    d["is_high"] = (df["conf_class"].to_numpy() == CONF_HIGH).astype(np.int32)
    d["_pix"] = (np.floor(df["lat"].to_numpy() / pixel_deg).astype(np.int64) * 1_000_000
                 + np.floor(df["lon"].to_numpy() / pixel_deg).astype(np.int64))
    for c in ("satellite", "daynight"):
        d[c] = d[c].astype("category")
    out = d.groupby(keys, observed=True, sort=False).agg(
        n_det=("frp", "size"), frp_sum=("frp", "sum"), frp_max=("frp", "max"), n_high=("is_high", "sum"))
    out["n_days"] = d.drop_duplicates([*keys, "local_date"]).groupby(keys, observed=True, sort=False).size()
    out["n_pixel_days"] = (d.drop_duplicates([*keys, "_pix", "local_date"])
                           .groupby(keys, observed=True, sort=False).size())
    out = out.reset_index()
    for c in ("satellite", "daynight"):
        out[c] = out[c].astype(str)
    return out.sort_values(keys, kind="stable").reset_index(drop=True)
