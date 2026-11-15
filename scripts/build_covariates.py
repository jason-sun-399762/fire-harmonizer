"""Build covariate tables on the configured grid, straight from the R2 bucket.

    python scripts/build_covariates.py ndvi --product VNP13C2 --years 2012-2026
    python scripts/build_covariates.py ndvi --product VJ113C2 --years 2018-2026
    python scripts/build_covariates.py pheno --product VNP22C2 --years 2013-2025
    python scripts/build_covariates.py landcover --years 2001-2025
    python scripts/build_covariates.py gldas --years 2000-2026
    python scripts/build_covariates.py gfwed --years 2000-2026

Writes out/features/<version>/<grid>/<dataset>/<year>.parquet with provenance metadata;
add --upload to also put it at the same key in R2. Every file passes a conservation check
(area-weighted mean before vs after regridding) or the run stops.
Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the environment.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firecal import io_r2  # noqa: E402
from firecal import products as P  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.provenance import output_key, write_parquet  # noqa: E402
from firecal.qc import check_range, check_rel  # noqa: E402
from firecal.rasters import CMG_005, regridded_area_mean, regridder, source_area_mean  # noqa: E402

RANGES = {"ndvi": (-1, 1), "evi": (-1, 1), "tair_c": (-90, 60), "rh_pct": (0, 100), "vpd_kpa": (0, 10),
          "precip_mm_day": (0, 500), "wind_ms": (0, 60), "fwi_mean": (0, 500), "season_length": (0, 731)}


def parse_years(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def conserve(name, arr, spec, df, col, grid):
    """Stop if regridding changed the area-weighted mean by more than 1e-6 (relative)."""
    src = source_area_mean(arr, spec)
    out = regridded_area_mean(regridder(spec, grid, P.auto_supersample(spec, grid)), arr, df, col)
    if np.isfinite(src) and abs(src) > 1e-9:
        check_rel(f"{name}:{col} area mean", src, out, 1e-6)


def check_ranges(df):
    for col, (lo, hi) in RANGES.items():
        if col in df:
            check_range(col, df[col], lo, hi)


class Builder:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config()
        self.grid = make_grid(self.cfg)
        self.tmp = tempfile.TemporaryDirectory()
        self.skipped: list[tuple[str, str]] = []

    def read(self, key: str, fn, *a, **kw):
        """Download + read one input; an unreadable file (truncated upload, bad HDF) is logged and
        skipped instead of stopping the run. Returns None when skipped."""
        try:
            return fn(self.fetch(key), *a, **kw)
        except (OSError, ValueError, KeyError) as e:
            self.skipped.append((key, str(e)[:200]))
            print(f"  SKIPPED unreadable input {key}: {str(e)[:120]}", flush=True)
            return None

    def fetch(self, key: str, clear: bool = True) -> Path:
        """Download into a scratch dir; by default it holds one file at a time (global files are large).
        clear=False keeps earlier files, for products read as a month of daily files."""
        if clear:
            for old in Path(self.tmp.name).iterdir():
                old.unlink()
        return io_r2.download(self.cfg, key, Path(self.tmp.name) / Path(key).name)

    def done(self, dataset: str, year: int, df: pd.DataFrame, inputs: list[str]):
        check_ranges(df)
        key = output_key(self.cfg, self.grid.name, dataset, str(year))
        local = ROOT / "out" / key
        bad = [k for k, _ in self.skipped if k in set(inputs)]
        write_parquet(df, local, self.cfg, dataset, inputs=[k for k in inputs if k not in bad],
                      extra={"skipped_inputs": bad})
        msg = f"  {dataset} {year}: {len(df):,} rows, {df['cell'].nunique():,} cells -> {local.relative_to(ROOT)}"
        if self.args.upload:
            io_r2.upload(self.cfg, local, key)
            msg += f"  (uploaded {key})"
        print(msg, flush=True)

    def keys(self, prefix: str, suffix: tuple[str, ...]) -> list[str]:
        return sorted(k["key"] for k in io_r2.list_keys(self.cfg, prefix) if k["key"].endswith(suffix))

    # ---------- datasets ----------

    def ndvi(self):
        prod = self.args.product or "VNP13C2"
        by_year = defaultdict(list)
        for k in self.keys(f"raw/{prod}/", (".h5",)):
            y = P.product_year(k)
            if y in self.args.years:
                by_year[y].append(k)
        for y, ks in sorted(by_year.items()):
            parts = []
            for k in ks:
                vi = self.read(k, P.read_vi_cmg)
                if vi is None:
                    continue
                df = P.to_cells({"ndvi": vi["ndvi"], "evi": vi["evi"]}, CMG_005, self.grid)
                conserve(k, vi["ndvi"], CMG_005, df, "ndvi", self.grid)
                parts.append(df.assign(ym=P.product_ym(k), source=prod))
            if parts:
                self.done(f"ndvi_{prod.lower()}", y, pd.concat(parts, ignore_index=True), ks)

    def pheno(self):
        prod = self.args.product or "VNP22C2"
        for k in self.keys(f"raw/{prod}/", (".h5",)):
            y = P.product_year(k)
            if y not in self.args.years:
                continue
            ph = self.read(k, P.read_phenology_cmg)
            if ph is None:
                continue
            df = P.to_cells(ph, CMG_005, self.grid, circular=("greenup_doy", "senescence_doy", "dormancy_doy"))
            conserve(k, ph["season_length"], CMG_005, df, "season_length", self.grid)
            self.done(f"pheno_{prod.lower()}", y, df.assign(year=y, source=prod), [k])

    def landcover(self):
        for k in self.keys("raw/MCD12C1/", (".hdf",)):
            y = P.product_year(k)
            if y not in self.args.years:
                continue
            pct = self.read(k, P.read_landcover_cmg)
            if pct is None:
                continue
            df = P.landcover_to_cells(pct, self.grid, P.LC_GROUPS)
            frac_cols = [c for c in df.columns if c.startswith("lc_") and c != "lc_valid"]
            s = df[frac_cols].sum(axis=1) + (1 - df["land_frac"])
            if not np.allclose(s[df["lc_valid"] > 0], 1.0, atol=1e-4):
                raise RuntimeError(f"{k}: land-cover fractions do not sum to 1")
            del pct
            self.done("landcover_mcd12c1", y, df.assign(year=y, source="MCD12C1"), [k])

    def gldas(self):
        main = {P.product_ym(k): k for k in self.keys("raw/GLDAS_NOAH025_M/", (".nc4",))}
        early = {P.product_ym(k): k for k in self.keys("raw/GLDAS_NOAH025_M_EP/", (".nc4",))}
        months = {**{m: (k, "GLDAS_NOAH025_M_EP") for m, k in early.items()},
                  **{m: (k, "GLDAS_NOAH025_M") for m, k in main.items()}}  # main stream wins
        by_year = defaultdict(list)
        for m, v in months.items():
            if m // 100 in self.args.years:
                by_year[m // 100].append((m, *v))
        for y, items in sorted(by_year.items()):
            parts = []
            for m, k, src in sorted(items):
                got = self.read(k, P.read_gldas)
                if got is None:
                    continue
                fields, spec = got
                df = P.to_cells(fields, spec, self.grid)
                conserve(k, fields["tair_c"], spec, df, "tair_c", self.grid)
                parts.append(df.assign(ym=m, source=src))
            if parts:
                self.done("gldas", y, pd.concat(parts, ignore_index=True), [i[1] for i in items])

    def gfwed(self):
        """Per month, the first complete source wins: MERRA-2 daily, GEOS-5 daily analysis,
        GEOS-5 day-1 forecasts, then a monthly-mean file. The source is recorded so the GEOS-5
        months can be calibrated to MERRA-2 on their 2025-2026 overlap later."""
        merra = defaultdict(list)
        geos_an = defaultdict(list)   # GEOS-5 daily analysis files (date only in the name)
        geos = defaultdict(list)      # GEOS-5 day-1 forecasts (init + valid date in the name)
        monthly = {}
        for k in self.keys("raw/GFWED/", (".nc",)):
            name = Path(k).name
            if ".Monthly." in name:
                m = int(re.search(r"\.(\d{6})\.nc$", name).group(1))
                if "MERRA2" in name:
                    monthly[m] = k            # prefer the reanalysis
                else:
                    monthly.setdefault(m, k)
                continue
            if "MERRA2" in name:
                merra[P.product_ym(name)].append(k)
            elif re.search(r"\.Default\.\d{8}\.nc$", name):
                geos_an[P.product_ym(name)].append(k)
            else:
                init, valid = re.search(r"\.(\d{10})\.(\d{8})\.nc$", name).groups()
                lead = (np.datetime64(f"{valid[:4]}-{valid[4:6]}-{valid[6:]}") - np.datetime64(f"{init[:4]}-{init[4:6]}-{init[6:8]}")).astype(int)
                if lead == 1:
                    geos[int(valid[:6])].append(k)
        months = sorted(set(merra) | set(geos_an) | set(geos) | set(monthly))
        by_year = defaultdict(list)
        for m in months:
            if m // 100 in self.args.years:
                by_year[m // 100].append(m)
        for y, ms in sorted(by_year.items()):
            parts, inputs = [], []
            for m in ms:
                ndays = pd.Period(f"{m // 100}-{m % 100:02d}").days_in_month
                if len(merra[m]) >= ndays:
                    ks, src = merra[m], "MERRA2_daily"
                elif len(geos_an[m]) >= ndays - 2:
                    ks, src = geos_an[m], "GEOS5_analysis"
                elif len(geos[m]) >= ndays - 2:
                    ks, src = geos[m], "GEOS5_day1"
                elif m in monthly:
                    ks, src = [monthly[m]], "monthly_mean"
                else:
                    print(f"  gfwed {m}: no complete source, skipped", flush=True)
                    continue
                try:
                    paths = [self.fetch(k, clear=(i == 0)) for i, k in enumerate(ks)]
                    if src == "monthly_mean":
                        fields, spec = P.read_gfwed_monthly(paths[0])
                    else:
                        fields, spec = P.gfwed_month(paths)
                except (OSError, ValueError, KeyError) as e:
                    self.skipped += [(k, str(e)[:200]) for k in ks]
                    print(f"  SKIPPED gfwed {m} ({src}): {str(e)[:120]}", flush=True)
                    continue
                df = P.to_cells(fields, spec, self.grid)
                conserve(f"gfwed {m}", fields["fwi_mean"], spec, df, "fwi_mean", self.grid)
                parts.append(df.assign(ym=m, source=src))
                inputs += ks
                for p in paths:
                    p.unlink(missing_ok=True)
            if parts:
                self.done("gfwed", y, pd.concat(parts, ignore_index=True), inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ndvi", "pheno", "landcover", "gldas", "gfwed"])
    ap.add_argument("--product")
    ap.add_argument("--years", type=parse_years, required=True)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    t = time.time()
    b = Builder(args)
    getattr(b, args.dataset)()
    print(f"done in {time.time() - t:.0f} s; skipped inputs: {len(b.skipped)}")
    for k, e in b.skipped:
        print(f"  skipped {k}: {e}")


if __name__ == "__main__":
    main()
