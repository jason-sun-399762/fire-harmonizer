# fire-harmonizer

Pipeline for the MODIS-Aqua-equivalent fire calendar (Space Apps 2026, "MODIS and VIIRS Hot Spots").
It follows the team's Data Fusion & Harmonization Protocol: every rule and threshold sits in
`config/default.yaml`, and every output Parquet records the config hash and git commit.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # pyhdf on Windows: conda install -c conda-forge pyhdf
python -m pytest -q                                      # 43 tests, ~25 s
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

## Covariates from R2

```bash
python scripts/build_covariates.py ndvi --product VNP13C2 --years 2012-2026   # S-NPP NDVI/EVI, 0.05 deg monthly
python scripts/build_covariates.py ndvi --product VJ113C2 --years 2018-2026   # NOAA-20 NDVI/EVI
python scripts/build_covariates.py pheno --years 2013-2025                    # VNP22C2 green-up / senescence / dormancy
python scripts/build_covariates.py landcover --years 2001-2025                # MCD12C1 -> 8 groups + land_frac
python scripts/build_covariates.py gldas --years 2000-2026                    # temp, RH, VPD, wind, precip, soil moisture
python scripts/build_covariates.py gfwed --years 2000-2026                    # FWI mean / p90 / days > 30, FFMC, DMC, DC, ISI, BUI
```

Output: `out/features/v2/ll0p25/<dataset>/<year>.parquet`; add `--upload` to write the same key to R2.
Each file is regridded through a cached sparse overlap matrix. The script stops if the area-weighted
mean changes during regridding by more than 1e-6.

Tested on real bucket files (2 vCPU): NDVI 2020 in 60 s, land cover 2020 in 29 s, phenology 2020 in 17 s,
GLDAS 2000-2001 in 41 s, GFWED 2026 in 6 min (mostly downloading ~300 daily files). GFWED is the slow
one, so run it in parallel by year range: `--years 2000-2004`, `--years 2005-2009`, and so on.

GFWED source per month: MERRA-2 daily, then GEOS-5 daily analysis, then GEOS-5 day-1 forecast, then a
monthly-mean file; the `source` column records which. **GEOS-5 needs calibrating before use:** on the same
month (July 2025) and cells it correlates 0.95 with MERRA-2 but reads ~22% lower, more in the most
fire-prone cells. MERRA-2 daily ends April 2026, so May 2026 onward is GEOS-5.

## Known limits

- `n_pixel_days` from FIRMS uses fixed ~1 km bins, so two detections of one pixel can straddle a bin edge. The target therefore uses MYD14A1's own pixel grid.
- H3 cell ids are computed point by point (~0.8M points/s). Fine for aggregates; for the full VIIRS archive prefer `latlon`, or vectorize with h3ronpy.
- VIIRS VI pixel reliability is read but not filtered on yet (rank meanings still need checking in the user guide).
- Not yet read: VJ114A1 / VNP14A1 / MYD14A1 runs at scale (reader done, waiting on the global pulls), MOD44B, NASADEM/GMTED, GPW, gROADS, VNP46A4 / VJ146A4, GEDI, Global Fire Atlas.
