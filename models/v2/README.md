# Product model v2 (LightGBM, 600 trees)

Trained 2026-11-15 on all 27.7 M training cell-months (2012-01..2023-01), commit 66d4582,
`python scripts/train.py --stage cv,backtest,final --drop-groups fire_weather,phenology --tag v2`.
Results: [docs/RESULTS.md](../../docs/RESULTS.md).

- `lgbm_full.txt`: the model (LightGBM text format).
- `meta.json`: run settings, including `features_full` (the 27 inputs, in order) and the GLM medians.
- `feature_importance_gain.csv`: gain per input.

**The model predicts a rate, not a count**: expected MODIS-Aqua fire pixel-days per VIIRS clear
pixel-day in a 0.25 degree cell-month. Count = rate x exposure. Build the inputs and exposure with `add_features` in
`scripts/train.py` (its `final` stage shows the whole prediction step, including the NOAA-20 fallback):

```python
import lightgbm as lgb, json
booster = lgb.Booster(model_file="models/v2/lgbm_full.txt")
feats = json.load(open("models/v2/meta.json"))["features_full"]
# X = add_features(train_table_rows, "s", usual_obs_per_day)   # see scripts/train.py
# count = booster.predict(X[feats]) * X["exposure_full"]
```

Inputs: v_rate, v_fire, v_frp_per_fire, v_cloud_frac, v_obs_frac, persistent, tair_c, precip_mm_day, soilmoist_0_10_kgm2, rootmoist_kgm2, rh_pct, vpd_kpa, wind_ms, ndvi_lag1, evi_lag1, lc_forest, lc_shrub, lc_savanna, lc_grass, lc_wetland, lc_crop, lc_urban, lc_barren_snow, land_frac, lat, lon, month.
