"""End-to-end dry run on a synthetic world with a known answer.

Runs the whole modelling path the real data will take: spatial-block CV over the model
ladder, the 2019-2021 back-test, 'simplest within 1 SE' selection, a calendar for one
cell, and a provenance-stamped Parquet output. Run: python scripts/dry_run.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firecal import calendar as cal  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.cv import backtest, block_folds, evaluate, simplest_within_1se, summarize  # noqa: E402
from firecal.grid import make_grid, spatial_block  # noqa: E402
from firecal.models import GlobalRatio, LightGBMModel, PoissonGLM, StratifiedRatio  # noqa: E402
from firecal.provenance import read_meta, write_parquet  # noqa: E402
from firecal.synthetic import make_world  # noqa: E402


def ladder():
    return [
        GlobalRatio(),
        StratifiedRatio(["landcover", "month"]),
        PoissonGLM(["viirs", "fwi"], log_features=["viirs"], offset=False),
        LightGBMModel(["viirs", "fwi", "landcover", "month"], objective="tweedie", offset=False, num_boost_round=300),
    ]


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    world = make_world(n_cells=600, years=range(2012, 2023), seed=7)
    world["cell"] = grid.cell_of(world["lat"], world["lon"])
    print(f"synthetic world: {len(world):,} cell-months, grid {grid.name}")

    blocks = spatial_block(grid, world["cell"].to_numpy(), cfg)
    folds = block_folds(blocks, cfg["cv"]["folds"], cfg["cv"]["seed"])
    res = evaluate(ladder, world, folds)
    s = summarize(res)
    print("\nspatial-block CV, Poisson deviance (lower is better):")
    print(s.round(4).to_string())
    pick = simplest_within_1se(s)
    print(f"simplest model within 1 SE of the best: {pick}")

    bt = backtest(ladder, world, cfg["cv"]["backtest_years"])
    print("\nback-test, MODIS hidden for", cfg["cv"]["backtest_years"])
    print(bt.round(4).to_string(index=False))

    # calendar for the busiest cell, from the chosen model's predictions
    model = {m.name: m for m in ladder()}[pick].fit(world)
    world["pred"] = model.predict(world)
    busiest = world.groupby("cell")["y"].sum().idxmax()
    c = world[world["cell"] == busiest]
    mid_doy = pd.to_datetime(dict(year=c["year"], month=c["month"], day=15)).dt.dayofyear - 1
    prof = cal.daily_profile(mid_doy.to_numpy(), c["pred"].to_numpy())
    timing = cal.season_timing(cal.circular_kde(prof, bandwidth_days=20))
    print(f"\nbusiest cell {busiest}: season timing (0-based day of year) {timing}")
    print(f"  true peak month in the synthetic world: {int(c['peak_month'].iloc[0])}")

    out = Path(__file__).resolve().parent.parent / "out" / "dry_run_predictions.parquet"
    write_parquet(world[["cell", "ym", "y", "viirs", "pred"]], out, cfg, "dry_run_predictions", extra={"model": pick})
    print(f"\nwrote {out.name}; provenance: {read_meta(out)['config_hash']=}")


if __name__ == "__main__":
    main()
