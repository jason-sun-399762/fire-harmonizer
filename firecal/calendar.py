"""Calendar building blocks: 52-week grid, circular smoothing, season timing."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 365


def week_grid(df: pd.DataFrame, value: str = "count") -> pd.DataFrame:
    """year x week (0-51) table, missing weeks filled with 0."""
    t = df.pivot_table(index="year", columns="week", values=value, aggfunc="sum", fill_value=0)
    return t.reindex(columns=range(52), fill_value=0)


def circular_kde(daily: np.ndarray, bandwidth_days: float = 10.0) -> np.ndarray:
    """Smooth a 365-day profile with a von Mises kernel; December and January join up.

    Returns a density that sums to 1 over the 365 days.
    """
    x = np.asarray(daily, float)
    if x.size != DAYS:
        raise ValueError("expects a 365-day profile (fold Feb 29 into Feb 28 first)")
    if x.sum() <= 0:
        return np.full(DAYS, 1.0 / DAYS)
    kappa = 1.0 / (2 * np.pi * bandwidth_days / DAYS) ** 2
    ang = 2 * np.pi * np.arange(DAYS) / DAYS
    kern = np.exp(kappa * (np.cos(ang) - 1.0))  # centred at day 0, scaled to avoid overflow
    kern /= kern.sum()
    sm = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(kern)))
    sm = np.maximum(sm, 0)
    return sm / sm.sum()


def season_timing(daily: np.ndarray, qs=(0.1, 0.5, 0.9), window: int = 30) -> dict:
    """Days of year (0-based) when cumulative activity reaches each quantile.

    The year is started at the quietest 30-day window, so a season that crosses New Year
    (e.g. Dec-Feb) is measured in one piece. Returns the start offset too.
    """
    x = np.asarray(daily, float)
    roll = np.convolve(np.concatenate([x, x[: window - 1]]), np.ones(window), "valid")[:DAYS]
    start = int((np.argmin(roll) + window // 2) % DAYS)
    rot = np.roll(x, -start)
    cum = np.cumsum(rot) / max(rot.sum(), 1e-12)
    out = {"year_start_doy0": start}
    for q in qs:
        out[f"q{int(q * 100)}_doy0"] = int((np.searchsorted(cum, q) + start) % DAYS)
    return out


def daily_profile(doy0: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """365-day histogram; day 365 (Dec 31 of a leap year) folds into day 364."""
    d = np.minimum(np.asarray(doy0, dtype=int), DAYS - 1)
    return np.bincount(d, weights=weights, minlength=DAYS)[:DAYS].astype(float)
