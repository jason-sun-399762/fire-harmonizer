"""A synthetic world with a known MODIS/VIIRS relationship, for tests and dry runs.

Truth: VIIRS sees fires at a rate lam; MODIS sees a land-cover- and weather-dependent share of them.
A good harmonization model should recover that share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# True MODIS/VIIRS detection share by land cover (crop fires are small, forest fires big).
TRUE_SHARE = {0: 0.15, 1: 0.35, 2: 0.30, 3: 0.50}  # crop, savanna, shrub, forest


def make_world(n_cells: int = 400, years=range(2012, 2023), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame(
        {
            "cell": np.arange(n_cells),
            "lat": rng.uniform(-40, 60, n_cells),
            "lon": rng.uniform(-170, 170, n_cells),
            "landcover": rng.integers(0, 4, n_cells),
            "peak_month": rng.integers(1, 13, n_cells),
            "base_rate": rng.lognormal(0.0, 1.0, n_cells),
        }
    )
    rows = []
    for y in years:
        for m in range(1, 13):
            rows.append(cells.assign(year=y, month=m))
    df = pd.concat(rows, ignore_index=True)
    df["ym"] = df["year"] * 100 + df["month"]
    d = np.minimum(np.abs(df["month"] - df["peak_month"]), 12 - np.abs(df["month"] - df["peak_month"]))
    season = np.exp(-0.5 * (d / 1.5) ** 2)
    df["fwi"] = rng.gamma(2.0, 5.0, len(df)) * (0.5 + season)          # fire weather
    lam = df["base_rate"] * season * (0.5 + df["fwi"] / 20.0)           # true fire activity per clear look
    df["exposure"] = rng.integers(200, 1200, len(df)).astype(float)     # clear pixel-days
    df["viirs"] = rng.poisson(lam * df["exposure"] / 100.0)
    share = df["landcover"].map(TRUE_SHARE) * (0.8 + 0.4 * (df["fwi"] > 15))  # bigger fires in bad weather
    df["y"] = rng.binomial(df["viirs"], np.clip(share, 0, 1))           # MODIS sees a subset
    df["true_share"] = share
    return df
