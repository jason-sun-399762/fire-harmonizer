"""Train and validate the VIIRS -> MODIS-Aqua-equivalent model, then predict every cell-month.

    python scripts/train.py --stage all [--sample-cells 0.1] [--rounds 600] [--upload]
    stages: cv, backtest, ablation, final, all

Target    y  = Aqua fire pixel-days (MYD14A1) per cell-month.
Offset    E  = S-NPP clear pixel-days (VNP14A1). An Aqua exposure cannot be the offset: it does
               not exist after Aqua, and every input must exist after MODIS ends.
Features  VIIRS signal (rate, FRP per fire, cloud share, observed days) + covariates.
Training rows: Aqua in-spec window (2012-01..2023-01), both sensors' month and FRP flags OK,
               and both sensors >= 90 % of their usual observed days that month (so a data gap
               on one side cannot pose as a sensor difference).
Validation: 5-fold spatial-block CV (10 degree blocks) for the model ladder, a 2019-2021 back-test,
            and leave-one-group-out ablations on the back-test. Model choice: simplest model
            within one standard error of the best CV deviance.
Prediction: every cell-month 2012-2026 from S-NPP, falling back to NOAA-20 (same features, j_
            columns) when S-NPP is missing or flagged. Exposure is scaled to a full month of
            observations (capped at x3) so data gaps do not read as less fire. Where both
            sensors exist, the NOAA-20-based prediction is kept too as a consistency check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firecal import io_r2  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.cv import simplest_within_1se, summarize  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.models import GlobalRatio, LightGBMModel, PoissonGLM, StratifiedRatio, score  # noqa: E402
from firecal.provenance import output_key, write_parquet  # noqa: E402

SENS = ["fire", "clear", "cloud", "valid", "frp", "obs"]
GROUPS = {
    "weather": ["tair_c", "precip_mm_day", "soilmoist_0_10_kgm2", "rootmoist_kgm2", "rh_pct", "vpd_kpa", "wind_ms"],
    "fire_weather": ["fwi_mean", "fwi_p90", "fwi_frac_gt30", "ffmc_mean", "dmc_mean", "dc_mean", "isi_mean", "bui_mean"],
    "vegetation": ["ndvi_lag1", "evi_lag1"],
    "landcover": ["lc_forest", "lc_shrub", "lc_savanna", "lc_grass", "lc_wetland", "lc_crop", "lc_urban", "lc_barren_snow", "land_frac"],
    "phenology": ["greenup_doy", "senescence_doy", "dormancy_doy", "season_length"],
    "location": ["lat", "lon"],
    "season": ["month"],
}
VIIRS_FEATS = ["v_rate", "v_fire", "v_frp_per_fire", "v_cloud_frac", "v_obs_frac", "persistent"]
FULL_FEATS = VIIRS_FEATS + [f for g in GROUPS.values() for f in g]
GLM_FEATS = ["v_rate", "v_frp_per_fire", "v_cloud_frac", "ndvi_lag1", "tair_c", "fwi_mean", "fwi_missing", "abs_lat"]
LC_CLASSES = GROUPS["landcover"][:-1]
TABLE_COLS = (["cell", "ym", "year", "month", "lat", "lon", "fold", "persistent", "in_calib",
               "a_fire", "a_clear", "a_valid", "a_obs", "a_month_ok", "a_frp_ok"]
              + [f"{x}_{v}" for x in "sj" for v in SENS] + [f"{x}_{k}" for x in "sj" for k in ("month_ok", "frp_ok")]
              + ["ndvi_s_lag1", "ndvi_j_lag1", "evi_s_lag1", "evi_j_lag1"]
              + [f for g, fs in GROUPS.items() if g not in ("vegetation", "location", "season") for f in fs])
MIN_COMPLETE = 0.9
GLM_MAX_ROWS = 3_000_000
EXPOSURE_SCALE_MAX = 3.0


def log(msg):
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} peak {rss:.1f} GB] {msg}", flush=True)


# ---------- data ----------

def load_years(cfg, grid, years, cache: Path, sample_cells: float | None, seed: int) -> pd.DataFrame:
    parts = []
    keep = None
    for y in years:
        key = output_key(cfg, grid.name, "train_table", str(y))
        p = cache / "train_table" / f"{y}.parquet"
        if not p.exists():
            io_r2.download(cfg, key, p)
        d = pd.read_parquet(p, columns=TABLE_COLS)
        if sample_cells:
            if keep is None:
                cells = np.unique(d.cell)
                keep = set(np.random.default_rng(seed).choice(cells, int(sample_cells * cells.size), replace=False))
            d = d[d.cell.isin(keep)]
        parts.append(d)
        log(f"  loaded {y}: {len(d):,} rows")
    return pd.concat(parts, ignore_index=True)


def usual_obs_per_day(df: pd.DataFrame, x: str) -> pd.Series:
    """Per-cell median observed tile-days per calendar day over the calibration window."""
    dim = pd.to_datetime((df.ym * 100 + 1).astype(str), format="%Y%m%d").dt.days_in_month
    r = (df[f"{x}_obs"] / dim)[df.in_calib & (df[f"{x}_obs"] > 0)]
    return r.groupby(df.cell[r.index]).median()


def add_features(df: pd.DataFrame, src: str, usual: pd.Series) -> pd.DataFrame:
    """Build v_* features from sensor `src` ('s' or 'j') plus shared covariates."""
    out = pd.DataFrame(index=df.index)
    for v in SENS:
        out[f"v_{v}"] = df[f"{src}_{v}"].astype(np.float32)
    dim = pd.to_datetime((df.ym * 100 + 1).astype(str), format="%Y%m%d").dt.days_in_month.to_numpy()
    expected = df.cell.map(usual).to_numpy() * dim
    out["v_rate"] = (1e4 * out.v_fire / out.v_clear.where(out.v_clear > 0)).astype(np.float32)
    out["v_frp_per_fire"] = (out.v_frp / out.v_fire.where(out.v_fire > 0)).astype(np.float32)
    out["v_cloud_frac"] = (out.v_cloud / out.v_valid.where(out.v_valid > 0)).astype(np.float32)
    out["v_obs_frac"] = (out.v_obs / expected).astype(np.float32)
    out["exposure"] = out.v_clear
    out["exposure_full"] = (out.v_clear * np.clip(1 / out.v_obs_frac.where(out.v_obs_frac > 0), 1, EXPOSURE_SCALE_MAX)).astype(np.float32)
    out["ndvi_lag1"] = df[f"ndvi_{src}_lag1"].fillna(df[f"ndvi_{'j' if src == 's' else 's'}_lag1"])
    out["evi_lag1"] = df[f"evi_{src}_lag1"].fillna(df[f"evi_{'j' if src == 's' else 's'}_lag1"])
    for f in FULL_FEATS:
        if f not in out and f in df:
            out[f] = df[f]
    out["persistent"] = df.persistent.astype(np.float32)
    out["month"] = df.month.astype(np.float32)
    out["fwi_missing"] = out.fwi_mean.isna().astype(np.float32)
    out["abs_lat"] = df.lat.abs()
    out["lc_major"] = np.asarray(LC_CLASSES)[np.nan_to_num(df[LC_CLASSES].to_numpy(), nan=-1).argmax(1)]
    return out


def glm_frame(X: pd.DataFrame, med: dict) -> pd.DataFrame:
    g = X.copy()
    for f in ("v_rate", "v_frp_per_fire"):
        g[f] = np.log1p(g[f].fillna(0))
    g["fwi_mean"] = g.fwi_mean.fillna(0)
    for f in ("v_cloud_frac", "ndvi_lag1", "tair_c"):
        g[f] = g[f].fillna(med[f])
    return g


def training_rows(df: pd.DataFrame, usual_a: pd.Series, usual_s: pd.Series) -> pd.Series:
    dim = pd.to_datetime((df.ym * 100 + 1).astype(str), format="%Y%m%d").dt.days_in_month.to_numpy()
    comp_a = df.a_obs / (df.cell.map(usual_a).to_numpy() * dim)
    comp_s = df.s_obs / (df.cell.map(usual_s).to_numpy() * dim)
    return (df.in_calib & df.a_month_ok & df.s_month_ok & df.a_frp_ok & df.s_frp_ok
            & (df.s_clear > 0) & (df.a_valid > 0) & (comp_a >= MIN_COMPLETE) & (comp_s >= MIN_COMPLETE))


# ---------- models ----------

class GLMWrap(PoissonGLM):
    def __init__(self, med):
        super().__init__(GLM_FEATS)
        self.med = med

    def fit(self, df):
        # 9 coefficients do not need 20M rows; statsmodels holds several copies of the design
        # matrix (OOM on EC2 with two runs at once), so fit on a fixed random subset.
        if len(df) > GLM_MAX_ROWS:
            df = df.sample(GLM_MAX_ROWS, random_state=0)
        return super().fit(glm_frame(df, self.med))

    def predict(self, df):
        return super().predict(glm_frame(df, self.med))


def ladder(rounds: int, med: dict) -> list:
    return [
        GlobalRatio(),
        StratifiedRatio(["lc_major", "month"]),
        GLMWrap(med),
        named(LightGBMModel(VIIRS_FEATS, num_boost_round=rounds), "4_lgbm_viirs_only"),
        named(LightGBMModel(FULL_FEATS, num_boost_round=rounds), "5_lgbm_full"),
    ]


def named(m, name):
    m.name = name
    return m


LGBM_EXTRA: dict = {}


def lgbm_params():
    return {"num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05, **LGBM_EXTRA}


def frame(X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    f = X.copy()
    f["viirs"] = f.v_fire
    if y is not None:
        f["y"] = y.to_numpy()
    return f


def block_month_score(te: pd.DataFrame, mu: np.ndarray, blocks: pd.Series) -> dict:
    g = pd.DataFrame({"b": blocks.to_numpy(), "ym": te.ym.to_numpy() if "ym" in te else 0, "y": te.y.to_numpy(), "mu": mu})
    a = g.groupby(["b", "ym"])[["y", "mu"]].sum()
    s = score(a.y, a.mu)
    return {"bm_deviance": s["deviance"], "bm_r2_log": s["r2_log"]}


def peak_month_agreement(te: pd.DataFrame, mu: np.ndarray, min_fires: float = 20) -> dict:
    g = pd.DataFrame({"cell": te.cell.to_numpy(), "m": te.month.to_numpy(), "y": te.y.to_numpy(), "mu": mu})
    c = g.groupby(["cell", "m"])[["y", "mu"]].sum().unstack("m", fill_value=0)
    tot = c["y"].sum(axis=1)
    c = c[tot >= min_fires]
    py, pm = c["y"].to_numpy().argmax(1), c["mu"].to_numpy().argmax(1)
    d = np.abs(py - pm)
    d = np.minimum(d, 12 - d)
    return {"peak_exact": float((d == 0).mean()), "peak_within1": float((d <= 1).mean()), "peak_cells": int(d.size)}


def run_cv(F, rounds, med, folds, outdir: Path | None = None) -> pd.DataFrame:
    import gc
    rows = []
    for f in sorted(np.unique(folds)):
        tr, te = F[folds != f], F[folds == f]
        for m in ladder(rounds, med):
            if isinstance(m, LightGBMModel):
                m.params.update(lgbm_params())
            t = time.time()
            m.fit(tr)
            mu = m.predict(te)
            rows.append({"model": m.name, "fold": int(f), **score(te.y, mu), "fit_s": round(time.time() - t, 1)})
            log(f"  cv fold {f} {m.name}: dev {rows[-1]['deviance']:.4f} bias {rows[-1]['rel_bias']:+.3f} ({rows[-1]['fit_s']} s)")
            del m, mu
            gc.collect()
        del tr, te
        gc.collect()
        if outdir is not None:
            pd.DataFrame(rows).to_csv(outdir / "cv_folds_partial.csv", index=False)   # survive a crash
    return pd.DataFrame(rows)


def run_backtest(F, rounds, med, years, models=None, blocks=None) -> pd.DataFrame:
    te_mask = F.year.isin(years)
    tr, te = F[~te_mask], F[te_mask]
    rows = []
    for m in (models or ladder(rounds, med)):
        if isinstance(m, LightGBMModel):
            m.params.update(lgbm_params())
        t = time.time()
        m.fit(tr)
        mu = m.predict(te)
        r = {"model": m.name, **score(te.y, mu), **block_month_score(te, mu, blocks[te_mask]), **peak_month_agreement(te, mu),
             "fit_s": round(time.time() - t, 1)}
        rows.append(r)
        log(f"  backtest {m.name}: dev {r['deviance']:.4f} bias {r['rel_bias']:+.3f} peak {r['peak_exact']:.2f}/{r['peak_within1']:.2f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="all, or a comma list of: cv, backtest, ablation, final")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM threads (0 = all cores)")
    ap.add_argument("--sample-cells", type=float, default=None, help="fraction of cells (for quick tests)")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    grid = make_grid(cfg)
    cache = Path(args.cache)
    outdir = ROOT / "out" / "train" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["cv"]["seed"]
    stages = ["cv", "backtest", "ablation", "final"] if args.stage == "all" else args.stage.split(",")
    LGBM_EXTRA["num_threads"] = args.threads

    log("loading calibration years 2012-2023")
    df = load_years(cfg, grid, range(2012, 2024), cache, args.sample_cells, seed)
    usual_a, usual_s = usual_obs_per_day(df, "a"), usual_obs_per_day(df, "s")
    tr_mask = training_rows(df, usual_a, usual_s)
    d = df[tr_mask].reset_index(drop=True)
    X = add_features(d, "s", usual_s)
    F = frame(X, d.a_fire)
    F["year"], F["cell"], F["ym"] = d.year.to_numpy(), d.cell.to_numpy(), d.ym.to_numpy()
    blocks = pd.Series(d.lat.floordiv(10).astype(int) * 100 + d.lon.floordiv(10).astype(int))
    med = {f: float(np.nanmedian(F[f])) for f in ("v_cloud_frac", "ndvi_lag1", "tair_c")}
    log(f"training rows {len(F):,} of {int(df.in_calib.sum()):,} calibration rows; "
        f"Aqua fires {F.y.sum():,.0f}, S-NPP fires {F.v_fire.sum():,.0f}")
    del df
    meta = {"tag": args.tag, "rows": int(len(F)), "sample_cells": args.sample_cells, "rounds": args.rounds,
            "features_full": FULL_FEATS, "features_viirs": VIIRS_FEATS, "glm": GLM_FEATS, "medians": med,
            "min_complete": MIN_COMPLETE}
    results = {}

    if "cv" in stages:
        log("stage cv: 5-fold spatial-block CV of the model ladder")
        cvr = run_cv(F, args.rounds, med, d.fold.to_numpy(), outdir)
        cvr.to_csv(outdir / "cv_folds.csv", index=False)
        summ = summarize(cvr)
        for mtr in ("rel_bias", "r2_log"):
            summ[mtr] = cvr.groupby("model")[mtr].mean()
        summ.to_csv(outdir / "cv_summary.csv")
        meta["chosen_by_1se"] = simplest_within_1se(summarize(cvr))
        log("cv summary:\n" + summ.round(4).to_string() + f"\nsimplest within 1 SE: {meta['chosen_by_1se']}")
        results["cv"] = summ

    years_bt = cfg["cv"]["backtest_years"]
    if "backtest" in stages:
        log(f"stage backtest: train without {years_bt}, predict them")
        bt = run_backtest(F, args.rounds, med, years_bt, blocks=blocks)
        bt.to_csv(outdir / "backtest.csv", index=False)
        log("backtest:\n" + bt.round(4).to_string(index=False))

    if "ablation" in stages:
        log("stage ablation: full LightGBM minus one feature group, on the back-test")
        models = [named(LightGBMModel(FULL_FEATS, num_boost_round=args.rounds), "full")]
        for g, fs in GROUPS.items():
            models.append(named(LightGBMModel([f for f in FULL_FEATS if f not in fs], num_boost_round=args.rounds), f"minus_{g}"))
        models.append(named(LightGBMModel([f for f in FULL_FEATS if f != "persistent"], num_boost_round=args.rounds), "minus_persistent_flag"))
        ab = run_backtest(F, args.rounds, med, years_bt, models=models, blocks=blocks)
        base = ab.loc[ab.model == "full", "deviance"].iloc[0]
        ab["delta_dev_pct"] = 100 * (ab.deviance - base) / base
        ab.to_csv(outdir / "ablation.csv", index=False)
        log("ablation:\n" + ab.round(4).to_string(index=False))

    if "final" in stages:
        choice = meta.get("chosen_by_1se") or "5_lgbm_full"
        log(f"stage final: fit on all training rows; product model = 5_lgbm_full (1-SE choice: {choice})")
        final = LightGBMModel(FULL_FEATS, num_boost_round=args.rounds)
        final.params.update(lgbm_params())
        final.fit(F)
        final.booster_.save_model(str(outdir / "lgbm_full.txt"))
        imp = pd.Series(final.booster_.feature_importance("gain"), index=FULL_FEATS).sort_values(ascending=False)
        imp.to_csv(outdir / "feature_importance_gain.csv")
        ratio = GlobalRatio().fit(F)
        meta["global_ratio"] = float(ratio.ratio_)
        log("top features (gain): " + ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(10).items()))
        del F, X, d
        for y in range(2012, 2027):
            t = load_years(cfg, grid, [y], cache, args.sample_cells, seed)
            s_ok = t.s_month_ok & t.s_frp_ok & t.s_clear.notna()
            j_ok = t.j_month_ok & t.j_frp_ok & t.j_clear.notna()
            out = pd.DataFrame({"cell": t.cell, "ym": t.ym, "a_fire": t.a_fire, "a_ok": t.a_month_ok & t.a_frp_ok})
            pred = {}
            s_loose = t.s_frp_ok & t.s_clear.notna()       # S-NPP in a flagged month: last resort, marked
            for src, ok, usual in (("s", s_loose, usual_s), ("j", j_ok, usual_s)):
                Xs = add_features(t, src, usual)
                mu = final.predict(Xs.assign(exposure=Xs.exposure_full))
                pred[src] = np.where(ok & (Xs.exposure_full > 0), mu, np.nan)
                out[f"{src}_fire"] = t[f"{src}_fire"]
            out["pred_from_s"] = np.where(s_ok, pred["s"], np.nan).astype(np.float32)
            out["pred_from_j"] = pred["j"].astype(np.float32)
            s_flag = np.where(~s_ok, pred["s"], np.nan)
            out["pred_aqua_equiv"] = out.pred_from_s.fillna(out.pred_from_j).fillna(pd.Series(s_flag, index=out.index)).astype(np.float32)
            out["pred_source"] = np.select([out.pred_from_s.notna(), out.pred_from_j.notna(), ~np.isnan(s_flag)],
                                           ["snpp", "noaa20", "snpp_flagged"], "none")
            key = output_key(cfg, grid.name, f"pred_aqua_{args.tag}", str(y))
            local = ROOT / "out" / key
            write_parquet(out, local, cfg, f"pred_aqua_{args.tag}", inputs=[output_key(cfg, grid.name, "train_table", str(y))],
                          extra={"model": "5_lgbm_full", "tag": args.tag, "sample_cells": args.sample_cells})
            if args.upload and not args.sample_cells:
                io_r2.upload(cfg, local, key)
            both = out.pred_from_s.notna() & out.pred_from_j.notna()
            msg = f"  predicted {y}: total {np.nansum(out.pred_aqua_equiv):,.0f} (obs Aqua {np.nansum(out.a_fire[out.a_ok]):,.0f})"
            if both.any():
                msg += f"; NOAA-20/S-NPP based ratio {out.pred_from_j[both].sum() / out.pred_from_s[both].sum():.3f}"
            log(msg)

    (outdir / f"meta_{args.stage.replace(',', '_')}.json").write_text(json.dumps(meta, indent=1, default=str))
    if args.upload and not args.sample_cells:
        for p in outdir.iterdir():
            io_r2.upload(cfg, p, f"{cfg['r2']['out_prefix']}/{cfg['version']}/{grid.name}/train_results/{args.tag}/{p.name}")
        log(f"uploaded results to features/v2/{grid.name}/train_results/{args.tag}/")
    log("done")


if __name__ == "__main__":
    main()
