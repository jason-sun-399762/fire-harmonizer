"""Daily FireMask tiles -> monthly per-cell counts, one product-year per run (easy to shard).

    python scripts/build_firemask.py MYD14A1 --years 2012 --threads 16 --upload
    python scripts/build_firemask.py VNP14A1 --years 2012-2014
    python scripts/build_firemask.py VJ114A1 --years 2018 --limit-days 3     # quick test

Output per (cell, ym): pixel-day sums of valid / clear / land-clear / cloud / fire pixels, FRP of fire
pixels, and obs_days = days with any valid observation (so an outage day reads as "not observed",
not "no fire"). Unreadable files are skipped and listed in the file metadata.
Lat/lon grids only (dense accumulation); needs the R2 token in the environment.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firecal import io_r2  # noqa: E402
from firecal.config import load_config  # noqa: E402
from firecal.firemask import METRICS, TileCellIndex, class_table, count_fast, read_firemask_frp  # noqa: E402
from firecal.grid import make_grid  # noqa: E402
from firecal.provenance import output_key, write_parquet  # noqa: E402
from firecal.sinusoidal import parse_date, parse_tile  # noqa: E402

VERSION = {"MYD14A1": "061", "MOD14A1": "061", "VNP14A1": "002", "VJ114A1": "002", "VJ214A1": "002"}


def parse_years(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=sorted(VERSION))
    ap.add_argument("--years", type=parse_years, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit-days", type=int, default=0, help="only the first N file dates (testing)")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    grid = make_grid(cfg)
    if grid.kind != "latlon":
        raise SystemExit("build_firemask.py accumulates on a dense lat/lon grid; set grid.kind: latlon")
    table = class_table(cfg)
    index = TileCellIndex(grid)
    prod = args.product
    ext = ".hdf" if prod.startswith("M") else ".h5"

    for year in args.years:
        t0 = time.time()
        keys = [k["key"] for k in io_r2.list_keys(cfg, f"raw/{prod}/{VERSION[prod]}/{prod}.A{year}") if k["key"].endswith(ext)]
        if args.limit_days:
            dates = sorted({Path(k).name.split(".")[1] for k in keys})[: args.limit_days]
            keys = [k for k in keys if Path(k).name.split(".")[1] in dates]
        # group by tile so each tile's pixel->cell lookup is built once and then dropped
        keys.sort(key=lambda k: (parse_tile(Path(k).name), Path(k).name.split(".")[1]))
        if not keys:
            print(f"{prod} {year}: no files", flush=True)
            continue
        acc: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((len(METRICS) + 1, grid.n_cells)))
        skipped = []
        tmp = tempfile.TemporaryDirectory()

        def load(key):
            try:
                p = io_r2.download(cfg, key, Path(tmp.name) / Path(key).name)
                m, f, d = read_firemask_frp(p)
                p.unlink()
                return key, m, f, d
            except Exception as e:  # truncated upload, bad HDF, network
                return key, None, None, str(e)[:200]

        def bounded(ex, items, window):
            """Like ex.map, but never more than `window` files downloaded ahead of the counting."""
            from collections import deque

            q = deque()
            it = iter(items)
            for x in it:
                q.append(ex.submit(load, x))
                if len(q) >= window:
                    break
            while q:
                yield q.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    q.append(ex.submit(load, nxt))

        done = 0
        with ThreadPoolExecutor(args.threads) as ex:
            for key, m, f, d in bounded(ex, keys, 2 * args.threads):
                if m is None:
                    skipped.append(key)
                    print(f"  SKIPPED {key}: {d}", flush=True)
                    continue
                h, v = parse_tile(Path(key).name)
                cells, cnt = count_fast(m, f, h, v, index, table)
                for k, day in enumerate(d):
                    s = str(day)
                    ym = int(s[:4]) * 100 + int(s[5:7])
                    a = acc[ym]
                    a[: len(METRICS), cells] += cnt[k]
                    a[len(METRICS), cells] += cnt[k, 0] > 0      # this tile observed the cell that day
                done += 1
                if done % 5000 == 0:
                    print(f"  {prod} {year}: {done:,}/{len(keys):,} files, {time.time() - t0:.0f} s", flush=True)

        frames = []
        for ym, a in sorted(acc.items()):
            nz = np.nonzero(a[0] > 0)[0]
            df = pd.DataFrame({m: a[i, nz] for i, m in enumerate(METRICS)})
            df.insert(0, "cell", nz.astype(np.int64))
            df.insert(1, "ym", ym)
            df["tile_obs_days"] = a[len(METRICS), nz]
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        for c in METRICS[:5]:
            out[c] = out[c].astype(np.int64)
        dataset = f"firemask_{prod.lower()}"
        key = output_key(cfg, grid.name, dataset, str(year))
        local = ROOT / "out" / key
        write_parquet(out, local, cfg, dataset, inputs=[k for k in keys if k not in set(skipped)],
                      extra={"skipped_inputs": skipped, "n_files": len(keys), "limit_days": args.limit_days})
        if args.upload:
            io_r2.upload(cfg, local, key)
        print(f"{prod} {year}: {len(keys):,} files ({len(skipped)} skipped) -> {len(out):,} cell-months "
              f"in {time.time() - t0:.0f} s{' (uploaded)' if args.upload else ''}", flush=True)


if __name__ == "__main__":
    main()
