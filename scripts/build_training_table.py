"""Join fire masks and covariates into one cell-month table per year (land cells only).

    python scripts/build_training_table.py --years 2012-2026 [--cache DIR] [--upload]

Inputs are the per-year Parquet files in features/v2/<grid>/<dataset>/<year>.parquet; a local
cache directory with the same <dataset>/<year>.parquet layout is used first, anything missing is
downloaded from R2. Output: features/v2/<grid>/train_table/<year>.parquet.

Per sensor (a_ = Aqua MYD14A1, s_ = S-NPP VNP14A1, j_ = NOAA-20 VJ114A1) the table carries fire,
clear, land-clear, cloud and valid pixel-days, FRP and observed tile-days. A sensor-month with no
file data is NaN (sensor not flying / not yet processed); a cell missing from a month that exists
is 0 (no valid observation that month).

Rules decided from the fire-mask QC (2026-11-14):
  * Aqua year files carry Jan 1-3 of the next year (8-day composites): only the file's own year is used.
  * Land cells: MCD12C1 land fraction >= cfg filters.min_land_fraction, and lat > -60.
  * Sensor-month flag <x>_month_ok: false when > 5 % of that sensor-month's fire pixels fall on
    water cells (VJ1 2018-02 and 2024-02, several S-NPP winter months 2012-2014 and 2023-11).
  * Cell-month flag <x>_frp_ok: false when mean FRP per fire pixel-day > 1500 MW (winter
    high-latitude artifacts; removes ~0 % of fires in normal years).
  * NOAA-20 is not used before 2018-03 (commissioning).
  * persistent: cell had S-NPP fire in >= 98 % of months 2013-2022 (gas flares, volcanoes, industry;
    174 cells, 1.5 % of fire). A flag for sensitivity runs, not a hard drop; FIRMS type=0 is better once points exist.
Covariates: NDVI/EVI at t-1 month (S-NPP and NOAA-20 separately), GLDAS and GFWED same month,
land cover at y-1, phenology at the latest year <= y-1.
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
from firecal.cv import block_folds  # noqa: E402
from firecal.grid import make_grid, spatial_block  # noqa: E402
from firecal.provenance import output_key, write_parquet  # noqa: E402

SENSORS = {"a": "firemask_myd14a1", "s": "firemask_vnp14a1", "j": "firemask_vj114a1"}
FM_COLS = {"n_fire": "fire", "n_clear": "clear", "n_land_clear": "land_clear", "n_cloud": "cloud",
           "n_valid": "valid", "frp_fire_mw": "frp", "tile_obs_days": "obs"}
OCEAN_SHARE_MAX = 0.05
FRP_PER_FIRE_MAX = 1500.0
NOAA20_START_YM = 201803
PERSIST_YEARS = range(2013, 2023)
PERSIST_SHARE = 0.98
GLDAS = ["tair_c", "precip_mm_day", "soilmoist_0_10_kgm2", "rootmoist_kgm2", "rh_pct", "vpd_kpa", "wind_ms"]
GFWED = ["fwi_mean", "fwi_p90", "ffmc_mean", "dmc_mean", "dc_mean", "isi_mean", "bui_mean"]
GFWED_SRC = {"MERRA2_daily": 0, "GEOS5_analysis": 1, "GEOS5_day1": 2, "monthly_mean": 3}
LC = ["lc_forest", "lc_shrub", "lc_savanna", "lc_grass", "lc_wetland", "lc_crop", "lc_urban", "lc_barren_snow", "land_frac"]
PHENO = ["greenup_doy", "senescence_doy", "dormancy_doy", "season_length"]


def parse_years(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class Inputs:
    def __init__(self, cfg, grid, cache: Path):
        self.cfg, self.grid, self.cache = cfg, grid, cache
        self.used: list[str] = []

    def path(self, dataset: str, year: int) -> Path | None:
        p = self.cache / dataset / f"{year}.parquet"
        key = output_key(self.cfg, self.grid.name, dataset, str(year))
        if not p.exists():
            if not io_r2.list_keys(self.cfg, key, limit=1):
                return None
            io_r2.download(self.cfg, key, p)
        self.used.append(key)
        return p

    def read(self, dataset: str, year: int, columns=None) -> pd.DataFrame | None:
        p = self.path(dataset, year)
        return None if p is None else pd.read_parquet(p, columns=columns)


def land_cells(inp: Inputs, cfg, grid) -> np.ndarray:
    lc = inp.read("landcover_mcd12c1", 2020, ["cell", "land_frac"])
    cells = lc.loc[lc.land_frac >= cfg["filters"]["min_land_fraction"], "cell"].to_numpy()
    lat, _ = grid.center(cells)
    return np.sort(cells[lat > -60])


def sensor_qc(inp: Inputs, land: np.ndarray, years) -> tuple[dict, set]:
    """Return {(sensor, ym): ocean_share} for every sensor-month, and the persistent cells."""
    share, hits, nmonths = {}, None, 0
    land_set = pd.Index(land)
    for x, ds in SENSORS.items():
        for y in years:
            df = inp.read(ds, y, ["cell", "ym", "n_fire"])
            if df is None:
                continue
            df = df[(df.ym // 100 == y) & (df.n_fire > 0)]
            on_land = land_set.get_indexer(df.cell) >= 0
            tot = df.groupby("ym").n_fire.sum()
            lnd = df[on_land].groupby("ym").n_fire.sum().reindex(tot.index, fill_value=0)
            for ym in tot.index:
                share[(x, int(ym))] = float(1 - lnd[ym] / tot[ym])
            if x == "s" and y in PERSIST_YEARS:
                c = df[on_land].groupby("cell").ym.nunique()
                hits = c if hits is None else hits.add(c, fill_value=0)
                nmonths += df.ym.nunique()
    persistent = set(hits[hits >= PERSIST_SHARE * nmonths].index) if hits is not None else set()
    return share, persistent


def year_table(inp: Inputs, y: int, land: np.ndarray, static: pd.DataFrame, share: dict, persistent: set) -> pd.DataFrame | None:
    months_present = set()
    fm = {}
    for x, ds in SENSORS.items():
        df = inp.read(ds, y)
        if df is None:
            continue
        df = df[df.ym // 100 == y]
        fm[x] = df
        months_present |= set(df.ym.unique().tolist())
    if not months_present:
        return None
    yms = np.array(sorted(months_present), dtype=np.int64)
    base = pd.DataFrame({"cell": np.repeat(land, yms.size), "ym": np.tile(yms, land.size)})
    out = base.merge(static, on="cell", how="left")
    out["year"] = np.int16(y)
    out["month"] = (out.ym % 100).astype(np.int8)
    for x, df in fm.items():
        sub = df[df.cell.isin(land)].rename(columns={k: f"{x}_{v}" for k, v in FM_COLS.items()})
        cols = [f"{x}_{v}" for v in FM_COLS.values()]
        out = out.merge(sub[["cell", "ym"] + cols], on=["cell", "ym"], how="left")
        have = out.ym.isin(df.ym.unique())
        for c in cols:
            out[c] = np.where(have, out[c].fillna(0), np.nan).astype(np.float32)
        out[f"{x}_month_ok"] = np.array([share.get((x, int(m)), 1.0) <= OCEAN_SHARE_MAX for m in out.ym])
        fpp = out[f"{x}_frp"] / out[f"{x}_fire"].where(out[f"{x}_fire"] > 0)
        out[f"{x}_frp_ok"] = ~(fpp > FRP_PER_FIRE_MAX)
    for x in SENSORS:
        if f"{x}_fire" not in out:
            for v in FM_COLS.values():
                out[f"{x}_{v}"] = np.float32(np.nan)
            out[f"{x}_month_ok"] = False
            out[f"{x}_frp_ok"] = True
    out.loc[out.ym < NOAA20_START_YM, "j_month_ok"] = False
    out["persistent"] = out.cell.isin(persistent)
    calib = (out.ym >= 201201) & (out.ym <= 202301)
    out["in_calib"] = calib
    out["backtest"] = out.year.isin(inp.cfg["cv"]["backtest_years"])

    # NDVI / EVI at t-1 month
    for x, ds in (("s", "ndvi_vnp13c2"), ("j", "ndvi_vj113c2")):
        parts = [d for d in (inp.read(ds, yy, ["cell", "ym", "ndvi", "evi"]) for yy in (y - 1, y)) if d is not None]
        if parts:
            v = pd.concat(parts)
            mm = v.ym % 100
            v["ym"] = np.where(mm == 12, (v.ym // 100 + 1) * 100 + 1, v.ym + 1)     # value at t-1 -> row t
            v = v.rename(columns={"ndvi": f"ndvi_{x}_lag1", "evi": f"evi_{x}_lag1"})
            out = out.merge(v, on=["cell", "ym"], how="left")
        else:
            out[f"ndvi_{x}_lag1"] = np.nan
            out[f"evi_{x}_lag1"] = np.nan
    out["ndvi_lag1"] = out.ndvi_s_lag1.fillna(out.ndvi_j_lag1)

    g = inp.read("gldas", y, ["cell", "ym"] + GLDAS)
    out = out.merge(g, on=["cell", "ym"], how="left") if g is not None else out.assign(**{c: np.nan for c in GLDAS})
    w = inp.read("gfwed", y, ["cell", "ym", "fwi_days_gt30", "fwi_ndays", "source"] + GFWED)
    if w is not None:
        w["fwi_frac_gt30"] = w.fwi_days_gt30 / w.fwi_ndays
        w["gfwed_src"] = w.pop("source").map(GFWED_SRC).astype("float32")
        out = out.merge(w.drop(columns=["fwi_days_gt30", "fwi_ndays"]), on=["cell", "ym"], how="left")
    else:
        out = out.assign(**{c: np.nan for c in GFWED + ["fwi_frac_gt30", "gfwed_src"]})

    lc = inp.read("landcover_mcd12c1", y - 1, ["cell"] + LC)
    out = out.merge(lc, on="cell", how="left") if lc is not None else out.assign(**{c: np.nan for c in LC})
    ph, py = None, None
    for yy in range(y - 1, 2011, -1):
        ph = inp.read("pheno_vnp22c2", yy, ["cell"] + PHENO)
        if ph is not None:
            py = yy
            break
    out = out.merge(ph, on="cell", how="left") if ph is not None else out.assign(**{c: np.nan for c in PHENO})
    out["pheno_year"] = np.int16(py or 0)

    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=parse_years, required=True)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    grid = make_grid(cfg)
    inp = Inputs(cfg, grid, Path(args.cache))
    t0 = time.time()
    land = land_cells(inp, cfg, grid)
    lat, lon = grid.center(land)
    blocks = spatial_block(grid, land, cfg)
    static = pd.DataFrame({"cell": land, "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
                           "area_km2": grid.area_km2(land).astype(np.float32), "block": blocks,
                           "fold": block_folds(blocks, cfg["cv"]["folds"], cfg["cv"]["seed"]).astype(np.int8)})
    qc_years = sorted(set(args.years) | set(PERSIST_YEARS))
    share, persistent = sensor_qc(inp, land, qc_years)
    bad = sorted((x, ym, round(s, 3)) for (x, ym), s in share.items() if s > OCEAN_SHARE_MAX)
    print(f"{land.size:,} land cells; {len(persistent):,} persistent; flagged sensor-months: {bad}", flush=True)
    for y in args.years:
        inp.used = []
        df = year_table(inp, y, land, static, share, persistent)
        if df is None:
            print(f"  {y}: no fire-mask data, skipped", flush=True)
            continue
        key = output_key(cfg, grid.name, "train_table", str(y))
        local = ROOT / "out" / key
        write_parquet(df, local, cfg, "train_table", inputs=sorted(set(inp.used)),
                      extra={"ocean_share_max": OCEAN_SHARE_MAX, "frp_per_fire_max": FRP_PER_FIRE_MAX,
                             "noaa20_start_ym": NOAA20_START_YM, "persist_share": PERSIST_SHARE,
                             "flagged_sensor_months": [b for b in bad if b[1] // 100 == y]})
        msg = f"  {y}: {len(df):,} rows x {df.shape[1]} cols, months {df.ym.nunique()} -> {local.relative_to(ROOT)}"
        if args.upload:
            io_r2.upload(cfg, local, key)
            msg += "  (uploaded)"
        print(msg, f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
