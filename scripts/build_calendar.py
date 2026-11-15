"""Harmonized 2002-2026 MODIS-Aqua-equivalent record and the per-cell burning calendar.

    python scripts/build_calendar.py --pred-tag v1 [--cache DIR] [--upload]

Record (per land cell and month, unit = Aqua fire pixel-days for a full month):
  2002-07 .. 2023-01  observed Aqua (MYD14A1), scaled up for missing days: a_fire * clip(usual/obs, 1, 3)
  2023-02 .. 2026-11  model prediction (pred_aqua_<tag>), already on a full-month basis
  Aqua leaves its 13:30 orbit in 2023 (drift), so the in-spec record ends there.
Calendar, from the base period 2003-2025 (complete years):
  m01..m12        mean fire pixel-days per calendar month
  annual_mean     mean yearly total
  peak_month      month with the most activity (1-12)
  start/end_month fractional month when cumulative activity reaches 10 % / 90 %, counted from
                  the quietest month, so seasons crossing New Year stay in one piece
  season_months   end - start (months holding the middle 80 % of activity)
  concentration   mean resultant length of the monthly profile (0 = year-round, 1 = one month)
  trend_pct_yr    Theil-Sen slope of yearly totals, % of the cell mean per year
Outputs go to features/v2/<grid>/harmonized_<tag>/<year>.parquet and calendar_<tag>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firecal import io_r2  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.provenance import output_key, write_parquet  # noqa: E402

AQUA_END_YM = 202301
SNPP_END_YM = 202611      # S-NPP stopped 2026-11-02
BASE = range(2003, 2026)
SCALE_MAX = 3.0
MIN_ANNUAL_WEB = 1.0     # web export keeps cells with >= 1 fire pixel-day per year on average


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def days_in_month(ym: np.ndarray) -> np.ndarray:
    return pd.to_datetime((np.asarray(ym) * 100 + 1).astype(str), format="%Y%m%d").days_in_month.to_numpy()


class Src:
    def __init__(self, cfg, grid, cache):
        self.cfg, self.grid, self.cache = cfg, grid, Path(cache)

    def read(self, ds, y, cols=None):
        p = self.cache / ds / f"{y}.parquet"
        if not p.exists():
            io_r2.download(self.cfg, output_key(self.cfg, self.grid.name, ds, str(y)), p)
        return pd.read_parquet(p, columns=cols)


def season_metrics(clim: np.ndarray) -> dict:
    """clim: (n, 12) mean monthly activity. Vectorised season timing."""
    n = clim.shape[0]
    tot = clim.sum(1)
    peak = clim.argmax(1) + 1
    sm = (np.roll(clim, 1, 1) + clim + np.roll(clim, -1, 1)) / 3     # circular 3-month smoothing
    quiet = sm.argmin(1)                                              # year starts after the quietest month
    idx = (quiet[:, None] + 1 + np.arange(12)[None, :]) % 12
    rot = np.take_along_axis(clim, idx, 1)
    cum = np.cumsum(rot, 1) / np.where(tot > 0, tot, 1)[:, None]

    def reach(q):
        k = (cum < q).sum(1)                                          # first month index reaching q
        k = np.minimum(k, 11)
        prev = np.where(k > 0, np.take_along_axis(cum, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0.0)
        cur = np.take_along_axis(cum, k[:, None], 1)[:, 0]
        frac = np.where(cur > prev, (q - prev) / (cur - prev), 0.0)   # position inside that month
        return k + frac                                               # months since rotated start

    s, e = reach(0.1), reach(0.9)
    start = ((quiet + 1 + s) % 12) + 1
    end = ((quiet + 1 + e) % 12) + 1
    ang = 2 * np.pi * (np.arange(12) + 0.5) / 12
    R = np.hypot((clim * np.cos(ang)).sum(1), (clim * np.sin(ang)).sum(1)) / np.where(tot > 0, tot, 1)
    nan = tot <= 0
    out = {"peak_month": np.where(nan, 0, peak).astype(np.int8),
           "start_month": np.where(nan, np.nan, start), "end_month": np.where(nan, np.nan, end),
           "season_months": np.where(nan, np.nan, e - s), "concentration": np.where(nan, np.nan, R)}
    return {k: np.asarray(v, dtype=np.float32) if v.dtype.kind == "f" else v for k, v in out.items()} | {"_n": n}


def theil_sen_pct(annual: np.ndarray) -> np.ndarray:
    """annual: (n, T). Median pairwise slope / mean, in % per year."""
    T = annual.shape[1]
    i, j = np.triu_indices(T, 1)
    out = np.empty(annual.shape[0], np.float32)
    for a in range(0, annual.shape[0], 50_000):
        blk = annual[a:a + 50_000].astype(np.float32)
        sl = (blk[:, j] - blk[:, i]) / (j - i)
        out[a:a + 50_000] = np.median(sl, 1)
    mean = annual.mean(1)
    return np.where(mean > 0, 100 * out / np.where(mean > 0, mean, 1), np.nan).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-tag", default="v1")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    grid = make_grid(cfg)
    src = Src(cfg, grid, args.cache)
    tag = args.pred_tag

    static = src.read("train_table", 2015, ["cell", "lat", "lon", "persistent"]).drop_duplicates("cell").reset_index(drop=True)
    cells = static.cell.to_numpy()
    pos = pd.Series(np.arange(cells.size), index=cells)
    log(f"{cells.size:,} land cells")

    # pass 1: each cell's usual Aqua observed tile-days per day (median over 2003-2022 months)
    obs_rate = []
    for y in range(2003, 2023):
        a = src.read("firemask_myd14a1", y, ["cell", "ym", "tile_obs_days"])
        a = a[(a.ym // 100 == y) & a.cell.isin(pos.index)]
        obs_rate.append(pd.DataFrame({"cell": a.cell, "r": a.tile_obs_days / days_in_month(a.ym.to_numpy())}))
    usual = pd.concat(obs_rate).groupby("cell").r.median()
    del obs_rate
    log("usual Aqua observation rate per cell done")

    years = range(2002, 2027)
    month_sum = np.zeros((cells.size, 12))
    month_n = np.zeros((cells.size, 12))
    annual = np.full((cells.size, len(BASE)), np.nan, np.float32)
    glob = []
    for y in years:
        a = src.read("firemask_myd14a1", y, ["cell", "ym", "n_fire", "tile_obs_days"])
        a = a[(a.ym // 100 == y) & a.cell.isin(pos.index) & (a.ym <= AQUA_END_YM)]
        yms = sorted(set(a.ym))
        recs = []
        if yms:
            full = pd.DataFrame({"cell": np.repeat(cells, len(yms)), "ym": np.tile(yms, cells.size)})
            full = full.merge(a, on=["cell", "ym"], how="left")
            exp_obs = full.cell.map(usual).to_numpy() * days_in_month(full.ym.to_numpy())
            obs = full.tile_obs_days.fillna(0).to_numpy()
            scale = np.clip(exp_obs / np.where(obs > 0, obs, np.nan), 1, SCALE_MAX)
            val = np.where(obs > 0, full.n_fire.fillna(0).to_numpy() * scale, np.nan)
            recs.append(pd.DataFrame({"cell": full.cell, "ym": full.ym, "fire_pd": val.astype(np.float32),
                                      "source": np.where(obs > 0, "aqua_obs", "none")}))
        if y >= 2012:
            p = src.read(f"pred_aqua_{tag}", y, ["cell", "ym", "pred_aqua_equiv", "pred_source", "pred_from_j"])
            p = p[p.ym > AQUA_END_YM].copy()
            # from S-NPP's last (partial) month on, NOAA-20 is the primary source
            late = (p.ym >= SNPP_END_YM) & p.pred_from_j.notna()
            p.loc[late, "pred_aqua_equiv"] = p.loc[late, "pred_from_j"]
            p.loc[late, "pred_source"] = "noaa20"
            if len(p):
                recs.append(pd.DataFrame({"cell": p.cell, "ym": p.ym, "fire_pd": p.pred_aqua_equiv.astype(np.float32),
                                          "source": np.where(p.pred_source == "none", "none", "model_" + p.pred_source)}))
        rec = pd.concat(recs, ignore_index=True)
        rec["source"] = rec.source.astype("category")
        key = output_key(cfg, grid.name, f"harmonized_{tag}", str(y))
        local = ROOT / "out" / key
        write_parquet(rec, local, cfg, f"harmonized_{tag}", inputs=[f"firemask_myd14a1/{y}", f"pred_aqua_{tag}/{y}"],
                      extra={"aqua_end_ym": AQUA_END_YM, "scale_max": SCALE_MAX, "unit": "Aqua fire pixel-days per full month"})
        if args.upload:
            io_r2.upload(cfg, local, key)
        g = rec.groupby(["ym", "source"], observed=True).fire_pd.sum().unstack(fill_value=0)
        glob.append(g)
        if y in BASE:
            i = pos[rec.cell.to_numpy()].to_numpy()
            m = (rec.ym % 100).to_numpy() - 1
            v = rec.fire_pd.to_numpy()
            ok = np.isfinite(v)
            np.add.at(month_sum, (i[ok], m[ok]), v[ok])
            np.add.at(month_n, (i[ok], m[ok]), 1)
            tot = np.bincount(i[ok], weights=v[ok], minlength=cells.size)
            annual[:, BASE.index(y)] = tot
        log(f"  {y}: {len(rec):,} cell-months, total {np.nansum(rec.fire_pd):,.0f} "
            f"({', '.join(f'{k} {v:,.0f}' for k, v in rec.groupby('source', observed=True).fire_pd.sum().items())})")

    clim = np.where(month_n > 0, month_sum / np.maximum(month_n, 1), 0.0)
    sm = season_metrics(clim)
    cal = static.copy()
    for k in range(12):
        cal[f"m{k + 1:02d}"] = clim[:, k].astype(np.float32)
    cal["annual_mean"] = clim.sum(1).astype(np.float32)
    for k, v in sm.items():
        if k != "_n":
            cal[k] = v
    cal["trend_pct_yr"] = theil_sen_pct(np.nan_to_num(annual))
    cal["n_years"] = np.int8(len(BASE))
    outdir = ROOT / "out" / f"{cfg['r2']['out_prefix']}/{cfg['version']}/{grid.name}/calendar_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    write_parquet(cal, outdir / "climatology.parquet", cfg, f"calendar_{tag}", extra={"base_years": [BASE.start, BASE.stop - 1]})
    web = cal[cal.annual_mean >= MIN_ANNUAL_WEB].copy()
    for c in web.columns:
        if web[c].dtype == np.float32:
            web[c] = web[c].round(3 if c in ("lat", "lon") else 2)
    web.drop(columns=["cell"]).to_csv(outdir / "calendar_cells.csv.gz", index=False)
    gm = pd.concat(glob).fillna(0)
    gm["total"] = gm.sum(axis=1)
    gm.to_csv(outdir / "global_monthly.csv")
    log(f"calendar: {len(cal):,} cells, {len(web):,} with >= {MIN_ANNUAL_WEB} pixel-day/yr; "
        f"global mean annual {cal.annual_mean.sum():,.0f}")
    if args.upload:
        for p in outdir.iterdir():
            io_r2.upload(cfg, p, f"{cfg['r2']['out_prefix']}/{cfg['version']}/{grid.name}/calendar_{tag}/{p.name}")
        log("uploaded")


if __name__ == "__main__":
    main()
