import numpy as np
import pandas as pd
import pytest

from firecal import products as P
from firecal.grid import H3Grid, LatLonGrid
from firecal.rasters import CMG_005, RasterSpec, Regridder, cells_area_mean, source_area_mean

G = LatLonGrid(0.25)


def test_spec_from_centers_handles_south_up():
    spec, flip = RasterSpec.from_centers(np.arange(-59.875, 90, 0.25), np.arange(-179.875, 180, 0.25))
    assert flip and spec.lat0 == pytest.approx(90.0) and spec.lon0 == pytest.approx(-180.0) and spec.ny == 600


def test_auto_supersample_is_exact_for_known_grids():
    gldas, _ = RasterSpec.from_centers(np.arange(-59.875, 90, 0.25), np.arange(-179.875, 180, 0.25))
    merra, _ = RasterSpec.from_centers(np.arange(-58, 75.01, 0.5), np.arange(-180, 179.9, 0.625))
    geos, _ = RasterSpec.from_centers(np.arange(-58, 75.01, 0.25), np.arange(-180, 179.9, 0.3125))
    assert P.auto_supersample(CMG_005, G) == 1
    assert P.auto_supersample(gldas, G) == 1
    assert P.auto_supersample(merra, G) == 10
    assert P.auto_supersample(geos, G) == 10


def test_aligned_block_mean_is_exact():
    # a 0.05 deg patch: every 0.25 deg cell holds 5 x 5 source pixels
    spec = RasterSpec(20, 20, 40.0, 10.0, 0.05, 0.05)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 20))
    x[0, 0] = np.nan
    rg = Regridder(spec, G, 1)
    out = rg.mean(x).set_index("cell")
    assert len(out) == 16
    cell = G.cell_of(39.99, 10.01)[()]                      # top-left 0.25 deg cell
    blk = x[:5, :5]
    lat_top = 40.0 - np.arange(5) * 0.05
    w = np.repeat((np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_top - 0.05)))[:, None], 5, 1)
    ok = np.isfinite(blk)
    assert out.loc[cell, "mean"] == pytest.approx(np.average(blk[ok], weights=w[ok]), rel=1e-12)
    assert out.loc[cell, "valid_frac"] == pytest.approx(1 - w[0, 0] / w.sum(), rel=1e-12)


def test_unaligned_source_conserves_area_mean():
    spec, _ = RasterSpec.from_centers(np.arange(30.25, 40, 0.5), np.arange(-109.6875, -100, 0.625))
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 50, (spec.ny, spec.nx))
    x[3, 4] = np.nan
    df = P.to_cells({"fwi": x}, spec, G)
    assert cells_area_mean(df, "fwi", "fwi_valid", G) == pytest.approx(source_area_mean(x, spec), rel=1e-9)


def test_partly_covered_cells_still_conserve():
    # GEOS-5-like grid whose top edge (75.125N) cuts a row of 0.25 deg cells in half
    from firecal.rasters import regridded_area_mean, regridder

    spec, _ = RasterSpec.from_centers(np.arange(70.0, 75.01, 0.25), np.arange(-10.0, 10.0, 0.3125))
    x = np.random.default_rng(2).uniform(0, 80, (spec.ny, spec.nx))
    df = P.to_cells({"fwi": x}, spec, G)
    rg = regridder(spec, G, P.auto_supersample(spec, G))
    assert regridded_area_mean(rg, x, df, "fwi") == pytest.approx(source_area_mean(x, spec), rel=1e-12)


def test_circular_field_and_h3_path():
    spec = RasterSpec(10, 10, 10.0, 0.0, 0.05, 0.05)
    doy = np.full((10, 10), 360.0)
    doy[:, 5:] = 6.0                                             # half Dec 26, half Jan 6
    df = P.to_cells({"greenup_doy": doy}, spec, G, circular=("greenup_doy",))
    d = df["greenup_doy"].to_numpy()
    assert np.all(np.minimum(d, 365.25 - d) < 8)                # around New Year, not mid-year
    h = P.to_cells({"v": np.ones((10, 10))}, spec, H3Grid(5))
    assert np.allclose(h["v"], 1.0)


def test_landcover_groups_sum_with_water_to_one():
    pct = np.zeros((3600, 7200, 17), dtype=np.uint8)
    pct[..., 0] = 100                                            # open water everywhere
    pct[:5, :5, 0] = 40                                          # top-left 0.25 deg cell: 40% water
    pct[:5, :5, 12] = 60                                         #                       60% cropland
    pct[5:10, :5, 0] = 255                                       # the cell below it: unmapped
    df = P.landcover_to_cells(pct, G, P.LC_GROUPS).set_index("cell")
    c = G.cell_of(89.99, -179.99)[()]
    assert df.loc[c, "lc_crop"] == pytest.approx(0.6) and df.loc[c, "land_frac"] == pytest.approx(0.6)
    below = G.cell_of(89.74, -179.99)[()]
    assert below not in df.index or df.loc[below, "lc_valid"] == 0


def test_gldas_derived_humidity(tmp_path):
    import xarray as xr

    lat = np.arange(-59.875, 90, 0.25)
    lon = np.arange(-179.875, 180, 0.25)
    shape = (1, lat.size, lon.size)
    ds = xr.Dataset(
        {
            "Tair_f_inst": (("time", "lat", "lon"), np.full(shape, 300.0, np.float32)),
            "Qair_f_inst": (("time", "lat", "lon"), np.full(shape, 0.01, np.float32)),
            "Psurf_f_inst": (("time", "lat", "lon"), np.full(shape, 100000.0, np.float32)),
            "Wind_f_inst": (("time", "lat", "lon"), np.full(shape, 3.0, np.float32)),
            "Rainf_f_tavg": (("time", "lat", "lon"), np.full(shape, 1e-5, np.float32)),
            "SoilMoi0_10cm_inst": (("time", "lat", "lon"), np.full(shape, 20.0, np.float32)),
            "RootMoist_inst": (("time", "lat", "lon"), np.full(shape, 200.0, np.float32)),
        },
        coords={"time": [np.datetime64("2020-07-01")], "lat": lat, "lon": lon},
    )
    p = tmp_path / "GLDAS_NOAH025_M.A202007.021.nc4"
    ds.to_netcdf(p)
    f, spec = P.read_gldas(p)
    assert f["tair_c"][0, 0] == pytest.approx(26.85, abs=1e-3)
    assert f["precip_mm_day"][0, 0] == pytest.approx(0.864, rel=1e-5)
    # q = 0.01 at 1000 hPa, 26.85 C -> RH about 45%, VPD about 1.9 kPa
    assert 40 < f["rh_pct"][0, 0] < 50 and 1.5 < f["vpd_kpa"][0, 0] < 2.3
    assert P.product_ym(p) == 202007


def test_product_dates():
    assert P.product_ym("VNP13C2.A2020061.002.x.h5") == 202003
    assert P.product_ym("FWI.MERRA2.Daily.Default.20200115.nc") == 202001
    assert P.product_year("VNP22C2.A2020001.002.x.h5") == 2020


def test_phenology_offset(tmp_path):
    import h5py

    p = tmp_path / "VNP22C2.A2020001.002.2024226130934.h5"
    with h5py.File(p, "w") as h:
        for f in P.PHENO_FIELDS.values():
            a = np.full((3600, 7200), P.PHENO_FILL, np.uint16)
            a[0, 0] = 7320 + 100 if "Length" not in f else 150
            h.create_dataset(P.PHENO_GROUP + f, data=a)
    ph = P.read_phenology_cmg(p)
    assert ph["greenup_doy"][0, 0] == 100 and ph["season_length"][0, 0] == 150 and np.isnan(ph["greenup_doy"][1, 1])
