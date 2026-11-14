"""Real-data smoke test against the R2 bucket. Needs the R2 token in the environment:

    export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
    python scripts/smoke_r2.py            # inventory + one FIRMS year
    python scripts/smoke_r2.py --year 2020

It lists what is in the bucket, reads one year of clean/firms, runs it through
standardize -> filters -> 0.25 deg aggregation with the conservation check, and
reports the problems the plan flagged: is Aqua separable from Terra, is the
`type` column present, and what area the points cover.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firecal import firms, io_r2  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.qc import check_exact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    args = ap.parse_args()
    cfg = load_config()

    print("== bucket inventory (top-level prefixes, first 5,000 keys)")
    keys = io_r2.list_keys(cfg, "", limit=5000)
    tops = Counter(k["key"].split("/")[0] + "/" + (k["key"].split("/")[1] if "/" in k["key"] else "") for k in keys)
    for p, n in sorted(tops.items()):
        print(f"  {p:<40s} {n:>6d} objects")
    new = [k["key"] for k in keys if k["key"].startswith(f"{cfg['r2']['out_prefix']}/{cfg['version']}/")]
    print(f"  outputs already under {cfg['r2']['out_prefix']}/{cfg['version']}/: {len(new)}")

    key = f"clean/firms/{args.year}.parquet"
    print(f"\n== {key}")
    raw = io_r2.read_parquet(cfg, key)
    if len(raw) > args.max_rows:
        raw = raw.sample(args.max_rows, random_state=0)
    print(f"  rows {len(raw):,}; columns {list(raw.columns)}")
    sat_col = "satellite" if "satellite" in raw else "sensor"
    print(f"  raw {sat_col} values: {raw[sat_col].value_counts().to_dict()}")

    d = firms.standardize(raw, cfg)
    print(f"  normalized satellites: {d['satellite'].value_counts().to_dict()}")
    if (d["satellite"] == "modis_unknown").any():
        print("  !! MODIS rows are not split into Aqua/Terra: the target cannot be built from this file")
    print(f"  type column present: {'type' in d}  (absent = volcano/static/offshore filter cannot run)")
    print(f"  lat range {d['lat'].min():.2f}..{d['lat'].max():.2f}, lon range {d['lon'].min():.2f}..{d['lon'].max():.2f}")

    kept, rep = firms.apply_filters(d, cfg)
    print(f"  filters: {rep}")
    grid = make_grid(cfg)
    agg = firms.aggregate(kept, grid)
    check_exact("detections before/after gridding", len(kept), int(agg["n_det"].sum()))
    print(f"  aggregated to {grid.name}: {len(agg):,} rows, {agg['cell'].nunique():,} cells; conservation OK")
    print(f"  pixel-days / detections = {agg['n_pixel_days'].sum() / max(agg['n_det'].sum(), 1):.3f}")


if __name__ == "__main__":
    main()
