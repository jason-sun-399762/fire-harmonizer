"""Time keys: local solar time, month and the fixed 52-slot week."""

from __future__ import annotations

import numpy as np
import pandas as pd


def firms_utc(acq_date, acq_time) -> pd.Series:
    """FIRMS acq_date (YYYY-MM-DD) + acq_time (HHMM, int or str) -> UTC timestamps."""
    t = pd.Series(acq_time).astype(str).str.zfill(4)
    d = pd.to_datetime(pd.Series(acq_date).astype(str))
    hh = t.str[:2].astype(int)
    mm = t.str[2:].astype(int)
    return d + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")


def local_solar(utc, lon) -> pd.Series:
    """Local solar time = UTC + lon/15 hours."""
    utc = pd.to_datetime(pd.Series(utc))
    offset = pd.to_timedelta(np.asarray(lon, dtype=float) / 15.0 * 3600.0, unit="s")
    return utc + offset


def week_of_year(doy0) -> np.ndarray:
    """52 fixed slots: week = min(doy0 // 7, 51), doy0 = 0 on Jan 1. Last slot holds 8-9 days."""
    doy0 = np.asarray(doy0, dtype=np.int64)
    return np.minimum(doy0 // 7, 51)


def add_time_keys(df: pd.DataFrame, local_col: str = "local_time") -> pd.DataFrame:
    """Add local_date, year, month, ym (yyyymm int), doy0 and week from a local-time column."""
    lt = pd.to_datetime(df[local_col])
    out = df.copy()
    out["local_date"] = lt.dt.normalize()
    out["year"] = lt.dt.year.astype(np.int16)
    out["month"] = lt.dt.month.astype(np.int8)
    out["ym"] = (out["year"].astype(np.int32) * 100 + out["month"]).astype(np.int32)
    out["doy0"] = (lt.dt.day_of_year - 1).astype(np.int16)
    out["week"] = week_of_year(out["doy0"]).astype(np.int8)
    return out


def ym_shift(ym, months: int) -> np.ndarray:
    """Shift yyyymm keys by a number of months (negative = earlier), for lagged joins."""
    ym = np.asarray(ym, dtype=np.int64)
    idx = (ym // 100) * 12 + (ym % 100 - 1) + months
    return (idx // 12) * 100 + idx % 12 + 1
