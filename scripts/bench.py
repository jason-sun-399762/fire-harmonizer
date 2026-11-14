"""Throughput check on realistic volumes (synthetic data). Run: python scripts/bench.py"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firecal import firms  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.firemask import TileCellIndex, count_tile  # noqa: E402
from firecal.grid import H3Grid, LatLonGrid  # noqa: E402


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    print(f"{label:<45s} {time.perf_counter() - t:7.2f} s")
    return out


def main(n_points: int = 2_000_000):
    cfg = load_config()
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "latitude": rng.uniform(-40, 60, n_points),
            "longitude": rng.uniform(-170, 170, n_points),
            "acq_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 366, n_points), unit="D"),
            "acq_time": rng.integers(0, 2400, n_points) // 100 * 100 + rng.integers(0, 60, n_points),
            "satellite": rng.choice(["N", "N20", "N21"], n_points),
            "confidence": rng.choice(["l", "n", "h"], n_points),
            "frp": rng.gamma(1.5, 5.0, n_points),
            "daynight": rng.choice(["D", "N"], n_points),
            "type": 0,
        }
    )
    raw["acq_date"] = raw["acq_date"].dt.strftime("%Y-%m-%d")
    print(f"{n_points:,} VIIRS points")
    d = timed("standardize", lambda: firms.standardize(raw, cfg))
    k, _ = timed("filters", lambda: firms.apply_filters(d, cfg))
    timed("aggregate to 0.25 deg x month", lambda: firms.aggregate(k, LatLonGrid(0.25)))
    timed("lat/lon cell ids, 10M points", lambda: LatLonGrid(0.25).cell_of(rng.uniform(-90, 90, 10_000_000), rng.uniform(-180, 180, 10_000_000)))
    timed("H3 res-5 cell ids, 200k points", lambda: H3Grid(5).cell_of(rng.uniform(-90, 90, 200_000), rng.uniform(-180, 180, 200_000)))

    idx = TileCellIndex(LatLonGrid(0.25))
    masks = rng.choice([3, 4, 5, 7], size=(8, 1200, 1200), p=[0.1, 0.4, 0.48, 0.02]).astype(np.uint8)
    dates = np.datetime64("2020-07-03") + np.arange(8).astype("timedelta64[D]")
    timed("FireMask tile, 8 days (first: builds index)", lambda: count_tile(masks, dates, 20, 8, idx, cfg))
    timed("FireMask tile, 8 days (cached index)", lambda: count_tile(masks, dates, 20, 8, idx, cfg))


if __name__ == "__main__":
    main()
