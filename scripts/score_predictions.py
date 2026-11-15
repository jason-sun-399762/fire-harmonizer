"""Common benchmark so any model (LightGBM here, a teammate's CNN, ...) is scored on the same rows
with the same metrics as the back-test in train.py.

    # 1) write the benchmark rows once (cell, ym, observed Aqua y, S-NPP exposure, fold, block)
    python scripts/score_predictions.py export --years 2019-2021 --out benchmark_2019_2021.parquet
    # 2) score a prediction file with columns cell, ym, pred  (pred = expected Aqua fire pixel-days
    #    for that 0.25 degree cell-month, made WITHOUT using Aqua data from 2019-2021)
    python scripts/score_predictions.py score --pred cnn_2019_2021.parquet --name cnn

Rows are exactly the back-test rows of train.py: Aqua in-spec window, both sensors' month/FRP
flags OK, both sensors >= 90 % of their usual observed days. Metrics: Poisson deviance, relative
bias, R2 on log1p (cell-month), the same on 10x10 degree block-months, and peak-month agreement
for cells with >= 20 Aqua fires over the test years.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

from firecal.config import load_config  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.models import score  # noqa: E402


def parse_years(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def export(args, cfg, grid):
    """Streams the years (low memory): usual observation rates from 2012-2023, then the test rows."""
    def path(y):
        p = Path(args.cache) / "train_table" / f"{y}.parquet"
        if not p.exists():
            T.io_r2.download(cfg, T.output_key(cfg, grid.name, "train_table", str(y)), p)
        return p

    rates = {"a": [], "s": []}
    for y in range(2012, 2024):
        d = pd.read_parquet(path(y), columns=["cell", "ym", "in_calib", "a_obs", "s_obs"])
        dim = T.dim_of(d.ym.to_numpy())
        for x in "as":
            k = d.in_calib.to_numpy() & (d[f"{x}_obs"].to_numpy() > 0)
            rates[x].append(pd.DataFrame({"cell": d.cell.to_numpy()[k].astype(np.int32),
                                          "r": (d[f"{x}_obs"].to_numpy()[k] / dim[k]).astype(np.float32)}))
    ua, us = (pd.concat(rates[x]).groupby("cell").r.median() for x in "as")
    del rates
    cols = ["cell", "ym", "year", "month", "lat", "lon", "fold", "in_calib", "a_fire", "a_valid", "a_obs", "a_month_ok",
            "a_frp_ok", "s_fire", "s_clear", "s_obs", "s_month_ok", "s_frp_ok"]
    outs = []
    for y in args.years:
        d = pd.read_parquet(path(y), columns=cols)
        d = d[T.training_rows(d, ua, us).to_numpy()]
        outs.append(pd.DataFrame({"cell": d.cell, "ym": d.ym, "year": d.year, "month": d.month, "lat": d.lat, "lon": d.lon,
                                  "y_aqua_fire": d.a_fire, "s_clear": d.s_clear, "s_fire": d.s_fire, "fold": d.fold,
                                  "block": (d.lat // 10).astype(int) * 100 + (d.lon // 10).astype(int)}))
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} benchmark rows ({out.cell.nunique():,} cells, years {args.years}) -> {args.out}")


def score_file(args):
    b = pd.read_parquet(args.bench)
    p = pd.read_parquet(args.pred, columns=["cell", "ym", "pred"])
    j = b.merge(p, on=["cell", "ym"], how="left")
    cov = j.pred.notna().mean()
    j = j[j.pred.notna()]
    te = pd.DataFrame({"y": j.y_aqua_fire.to_numpy(), "ym": j.ym.to_numpy(), "cell": j.cell.to_numpy(), "month": j.month.to_numpy()})
    mu = np.clip(j.pred.to_numpy(float), 0, None)
    r = {"model": args.name, "coverage": round(float(cov), 4), **score(te.y, mu),
         **T.block_month_score(te, mu, j.block.reset_index(drop=True)), **T.peak_month_agreement(te, mu)}
    ratio = j.y_aqua_fire.sum() / j.s_fire.sum()                 # reference: one global ratio, same rows
    ref = {"model": "ref_global_ratio_in_sample", "coverage": round(float(cov), 4), **score(te.y, j.s_fire.to_numpy() * ratio),
           **T.block_month_score(te, j.s_fire.to_numpy() * ratio, j.block.reset_index(drop=True)),
           **T.peak_month_agreement(te, j.s_fire.to_numpy() * ratio)}
    res = pd.DataFrame([r, ref])
    print(res.round(4).to_string(index=False))
    if cov < 0.999:
        print(f"WARNING: prediction covers {cov:.1%} of benchmark rows; scores are on the covered rows only "
              "and are not directly comparable with train.py's backtest.csv")
    if args.out:
        res.to_csv(args.out, index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--years", type=parse_years, default=[2019, 2020, 2021])
    e.add_argument("--out", default="benchmark_2019_2021.parquet")
    e.add_argument("--cache", default=str(ROOT / "cache"))
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--name", default="model")
    s.add_argument("--bench", default="benchmark_2019_2021.parquet")
    s.add_argument("--out")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "export":
        export(args, cfg, make_grid(cfg))
    else:
        score_file(args)


if __name__ == "__main__":
    main()
