# fire-harmonizer

Pipeline for the MODIS-Aqua-equivalent fire calendar (Space Apps 2026, "MODIS and VIIRS Hot Spots").
It follows the team's Data Fusion & Harmonization Protocol: every rule and threshold sits in
`config/default.yaml`, and every output Parquet records the config hash and git commit.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # pyhdf on Windows: conda install -c conda-forge pyhdf
python -m pytest -q                                      # 33 tests, ~5 s
python scripts/dry_run.py                                # synthetic end-to-end run, ~8 s
python scripts/bench.py                                  # throughput check
```

Real data (needs the R2 token; never commit it):

```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
python scripts/smoke_r2.py --year 2020
```

`smoke_r2.py` lists the bucket, pushes one year of `clean/firms` through the FIRMS steps and the
conservation check, and reports three things the plan flagged: whether Aqua and Terra are
separable, whether the FIRMS `type` column exists, and what area the points cover.

## Switches in `config/default.yaml`

| Setting | Default | Alternative |
| --- | --- | --- |
| `grid.kind` | `latlon` at `res_deg: 0.25` | `h3` at `h3_res: 5` |
| `target.definition` | `pixel_days_l3` (from MYD14A1) | `firms_detections` |
| `time.step` | `month` | `week` |

Outputs go to `features/<version>/<grid name>/<dataset>/`, e.g. `features/v2/ll0p25/firms/2020.parquet`,
so the existing `features/v1/h3r5/` files are never overwritten.

## Modules

| Module | Does |
| --- | --- |
| `grid.py` | 0.25 deg lat/lon and H3 grids behind one interface; cell areas; exact parents; CV blocks |
| `timeutil.py` | local solar time, month keys, 52-slot week, month lags |
| `sinusoidal.py` | MODIS/VIIRS tile pixel centres in closed form (no reprojection) |
| `firms.py` | FIRMS raw and bucket layouts -> one schema; satellite names; confidence classes; type / confidence / persistent-source filters; cell x month aggregation |
| `firemask.py` | MYD14A1 (HDF4, 8 days per file), VNP14A1 / VJ114A1 (HDF5) -> clear / cloud / fire pixel counts per cell-day |
| `aggregate.py` | raster -> cells by variable type: intensive, extensive, categorical, circular; area weights; supersampling for coarse grids |
| `qc.py` | conservation, range and fraction checks that stop the pipeline |
| `provenance.py` | Parquet with config hash, git commit and inputs in its metadata |
| `io_r2.py` | list / download / upload / read Parquet on R2 |
| `models.py` | model ladder: global ratio -> stratified ratio -> Poisson GLM -> LightGBM, with log(exposure) offset; metrics |
| `cv.py` | spatial-block folds, year back-test, "simplest within 1 SE" selection |
| `calendar.py` | 52-week grid, circular (von Mises) smoothing, season timing that handles seasons crossing New Year |
| `synthetic.py` | a world with a known answer, for tests and the dry run |

## Verified so far (synthetic data)

- 0.25 deg cell areas sum to the Earth's surface; 1 deg parents hold exactly 16 children; +180/-180 and the poles map correctly.
- Local solar dates: a 15:30 UTC night pass over eastern Australia is dated the next day.
- Sinusoidal pixel centres match known tile corners; off-Earth pixels are masked.
- Real-format HDF4 (8-day MYD14A1) and HDF5 (VJ114A1) files read and count correctly.
- FIRMS filters drop volcanoes and low-confidence rows; aggregation conserves detections exactly.
- The model ladder ranks as expected; spatial blocks never leak between train and test; doubling exposure doubles predictions.
- Season timing finds a mid-year peak and a peak that straddles New Year.

Throughput (2 vCPU): 2M FIRMS points standardized + filtered + gridded in ~7 s; one 8-day FireMask tile in ~0.3 s.

## Known limits

- `n_pixel_days` from FIRMS uses fixed ~1 km bins, so two detections of one pixel can straddle a bin edge. The target therefore uses MYD14A1's own pixel grid.
- H3 cell ids are computed point by point (~0.8M points/s). Fine for aggregates; for the full VIIRS archive prefer `latlon`, or vectorize with h3ronpy.
- Readers for GLDAS, GFWED, MCD12Q1, MOD44B, VNP13A3 and the other covariates are next; the aggregation rules they will call are already tested.
