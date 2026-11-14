import numpy as np
import pandas as pd
import pytest

from firecal import aggregate as agg
from firecal.firemask import VIIRS_FIREMASK, TileCellIndex, count_file, count_tile, read_firemask
from firecal.grid import LatLonGrid
from firecal.provenance import read_meta, write_parquet
from firecal.qc import QCError, check_exact, check_fractions, check_range, check_rel


def fake_mask(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([3, 4, 5, 7, 8, 9, 0], size=(n, n), p=[0.05, 0.3, 0.55, 0.02, 0.03, 0.02, 0.03]).astype(np.uint8)


def test_count_tile_totals(cfg):
    m = np.stack([fake_mask(seed=1), fake_mask(seed=2)])
    dates = np.array(["2020-07-01", "2020-07-02"], dtype="datetime64[D]")
    idx = TileCellIndex(LatLonGrid(0.25))
    out = count_tile(m, dates, 8, 5, idx, cfg)
    for k in range(2):
        day = out[out["date"] == dates[k]]
        assert day["n_cloud"].sum() == (m[k] == 4).sum()
        assert day["n_fire"].sum() == np.isin(m[k], [7, 8, 9]).sum()
        assert day["n_valid"].sum() == (m[k] >= 3).sum()
    # coarse 1 km test pixels at 10 km spacing: each 0.25 deg cell gets a handful
    assert out["cell"].nunique() > 10


def test_hdf5_viirs_reader(tmp_path, cfg):
    import h5py

    p = tmp_path / "VJ114A1.A2026310.h08v05.002.2026311000000.h5"
    mask = fake_mask(1200, seed=3)
    with h5py.File(p, "w") as f:
        f.create_dataset(VIIRS_FIREMASK, data=mask)
    arr, dates = read_firemask(p)
    assert arr.shape == (1, 1200, 1200) and dates[0] == np.datetime64("2026-11-06")
    out = count_file(p, TileCellIndex(LatLonGrid(0.25)), cfg)
    assert out["n_fire"].sum() == np.isin(mask, [7, 8, 9]).sum()
    assert set(out["product"]) == {"VJ114A1"}


def test_hdf4_modis_reader_eight_days(tmp_path, cfg):
    from pyhdf.SD import SD, SDC

    p = tmp_path / "MYD14A1.A2020185.h08v05.061.2020194000000.hdf"
    data = np.stack([fake_mask(1200, seed=s) for s in range(8)])
    sd = SD(str(p), SDC.WRITE | SDC.CREATE)
    ds = sd.create("FireMask", SDC.UINT8, data.shape)
    ds[:] = data
    ds.endaccess()
    sd.end()
    arr, dates = read_firemask(p)
    assert arr.shape == (8, 1200, 1200)
    assert dates[0] == np.datetime64("2020-07-03") and dates[-1] == np.datetime64("2020-07-10")
    out = count_file(p, TileCellIndex(LatLonGrid(0.25)), cfg)
    assert out["date"].nunique() == 8
    assert out["n_cloud"].sum() == (data == 4).sum()


def test_intensive_mean_std_valid():
    cells = np.array([1, 1, 1, 2, 2])
    vals = np.array([1.0, 3.0, np.nan, 5.0, 5.0])
    r = agg.intensive(cells, vals).set_index("cell")
    assert r.loc[1, "mean"] == 2.0 and r.loc[1, "std"] == 1.0
    assert r.loc[1, "valid_frac"] == pytest.approx(2 / 3)
    assert r.loc[2, "mean"] == 5.0 and r.loc[2, "valid_frac"] == 1.0


def test_extensive_and_categorical_and_circular():
    cells = np.array([1, 1, 2, 2])
    e = agg.extensive(cells, [1.0, 2.0, 3.0, np.nan]).set_index("cell")
    assert e["sum"].sum() == 6.0 and e.loc[2, "valid_frac"] == 0.5
    c = agg.categorical(cells, [10, 12, 12, -1]).set_index("cell")
    check_fractions("lc", [c[col] for col in c.columns if col.startswith("frac_")])
    assert c.loc[1, "frac_10"] == 0.5 and c.loc[2, "frac_12"] == 1.0
    # day 360 and day 5 average to ~day 0, not mid-year
    circ = agg.circular([1, 1], [360.0, 5.0]).set_index("cell")
    m = circ.loc[1, "circ_mean"]
    assert min(m, 365.25 - m) < 3 and circ.loc[1, "resultant"] > 0.99


def test_supersampled_latlon_conserves_area_mean():
    # a 0.5 x 0.625 deg source (GFWED-like) onto 0.25 deg cells
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 50, (20, 16))
    lat_e = np.linspace(30, 40, 21)
    lon_e = np.linspace(-110, -100, 17)
    lat, lon, v, w = agg.latlon_raster_points(vals, lat_e, lon_e, supersample=5)
    g = LatLonGrid(0.25)
    r = agg.intensive(g.cell_of(lat, lon), v, w)
    area = g.area_km2(r["cell"])
    src_mean = np.average(v, weights=w)
    assert np.average(r["mean"], weights=area) == pytest.approx(src_mean, rel=2e-3)
    # weights reproduce the true area of the source box
    box_sr = np.radians(10.0) * (np.sin(np.radians(40.0)) - np.sin(np.radians(30.0)))
    assert w.sum() == pytest.approx(box_sr, rel=1e-9)


def test_qc_checks():
    check_exact("n", 10, 10)
    with pytest.raises(QCError):
        check_exact("n", 10, 9)
    check_rel("ba", 100.0, 100.4, 0.005)
    with pytest.raises(QCError):
        check_rel("ba", 100.0, 101.0, 0.005)
    with pytest.raises(QCError):
        check_range("ndvi", [0.2, 1.5], -1, 1)
    check_range("ndvi", [0.2, np.nan], -1, 1)


def test_parquet_provenance_roundtrip(tmp_path, cfg):
    p = tmp_path / "x.parquet"
    write_parquet(pd.DataFrame({"cell": [1, 2], "v": [0.1, 0.2]}), p, cfg, "demo", inputs=["a.hdf"])
    meta = read_meta(p)
    assert meta["dataset"] == "demo" and meta["grid"]["kind"] == "latlon" and len(meta["config_hash"]) == 16
    assert pd.read_parquet(p).shape == (2, 2)
