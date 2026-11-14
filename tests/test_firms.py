import numpy as np
import pandas as pd
import pytest

from firecal import firms
from firecal.grid import LatLonGrid


def raw_modis():
    return pd.DataFrame(
        {
            "latitude": [-15.012, -15.014, 45.5, 45.5, 10.0],
            "longitude": [30.0, 30.001, -120.0, -120.0, 10.0],
            "acq_date": ["2020-08-01", "2020-08-01", "2020-08-01", "2020-08-02", "2020-08-01"],
            "acq_time": [1130, 1135, 2045, 2050, 1200],
            "satellite": ["Aqua", "Aqua", "Terra", "A", "Aqua"],
            "confidence": [85, 50, 10, 70, 90],
            "frp": [20.0, 5.0, 3.0, 8.0, 40.0],
            "daynight": ["D", "D", "D", "D", "D"],
            "type": [0, 0, 0, 0, 1],  # last row is a volcano
            "scan": [1.0] * 5,
            "track": [1.0] * 5,
        }
    )


def raw_viirs():
    return pd.DataFrame(
        {
            "latitude": [-15.0, -15.0],
            "longitude": [30.0, 30.0],
            "acq_date": ["2020-08-01", "2020-08-01"],
            "acq_time": ["1100", "1150"],
            "satellite": ["N", "N20"],
            "confidence": ["h", "l"],
            "frp": [4.0, 1.0],
            "daynight": ["D", "D"],
        }
    )


def test_satellite_spellings():
    s = firms.normalize_satellite(["Aqua", "A", "Terra", "N", "Suomi NPP", "N20", "NOAA-20", "1", "N21", "NOAA_21", "MODIS", "x"])
    assert list(s) == ["aqua", "aqua", "terra", "snpp", "snpp", "noaa20", "noaa20", "noaa20", "noaa21", "noaa21",
                       "modis_unknown", "unknown"]


def test_standardize_and_confidence(cfg):
    d = firms.standardize(raw_modis(), cfg)
    assert list(d["conf_class"]) == [2, 1, 0, 1, 2]
    assert set(d["instrument"]) == {"modis"}
    v = firms.standardize(raw_viirs(), cfg)
    assert list(v["satellite"]) == ["snpp", "noaa20"] and list(v["conf_class"]) == [2, 0]
    # local solar date at 120W, 20:45 UTC -> 12:45 local, same date
    assert d.loc[2, "local_time"] == pd.Timestamp("2020-08-01 12:45")


def test_bucket_layout_is_accepted(cfg):
    clean = pd.DataFrame(
        {
            "latitude": [40.0], "longitude": [-100.0], "sensor": ["VIIRS_NOAA20"],
            "utc_time": [pd.Timestamp("2021-07-01 19:00")], "is_day": [True],
            "conf_ok": [True], "conf_high": [False], "frp": [3.0],
        }
    )
    d = firms.standardize(clean, cfg)
    assert d.loc[0, "satellite"] == "noaa20" and d.loc[0, "conf_class"] == 1 and d.loc[0, "daynight"] == "D"


def test_merged_modis_label_warns(cfg):
    df = raw_modis().assign(satellite="MODIS")
    with pytest.warns(UserWarning, match="Aqua"):
        firms.standardize(df, cfg)


def test_filters_drop_volcano_and_low_conf(cfg):
    d = firms.standardize(raw_modis(), cfg)
    kept, rep = firms.apply_filters(d, cfg)
    assert rep == {"rows_in": 5, "dropped_type": 1, "dropped_confidence": 1, "rows_out": 3}
    assert set(kept["conf_class"]) <= {1, 2}


def test_aggregate_conserves_and_dedupes(cfg):
    d = firms.standardize(raw_modis(), cfg)
    kept, _ = firms.apply_filters(d, cfg)
    agg = firms.aggregate(kept, LatLonGrid(0.25))
    assert agg["n_det"].sum() == len(kept)                 # conservation
    a = agg[(agg["satellite"] == "aqua")]
    # two Aqua detections 100 m apart on the same day share one ~1 km pixel-day
    zambia = a[a["n_det"] == 2]
    assert len(zambia) == 1 and zambia["n_pixel_days"].iloc[0] == 1
    assert zambia["frp_sum"].iloc[0] == pytest.approx(25.0)


def test_persistent_sources_removed(cfg):
    cfg["filters"]["persistent"].update(min_days_per_year=5, min_years=2)
    days = pd.date_range("2019-01-01", periods=10).append(pd.date_range("2020-01-01", periods=10))
    flare = pd.DataFrame({"latitude": 30.0, "longitude": 48.0, "acq_date": days.strftime("%Y-%m-%d"),
                          "acq_time": 1030, "satellite": "N", "confidence": "n", "frp": 10.0, "daynight": "N"})
    fire = raw_viirs()
    d = firms.standardize(pd.concat([flare, fire], ignore_index=True), cfg)
    bins = firms.persistent_bins(d, cfg)
    assert len(bins) == 1
    out = firms.drop_persistent(d, bins, cfg)
    assert len(out) == len(fire)
