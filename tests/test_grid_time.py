import numpy as np
import pandas as pd
import pytest

from firecal.grid import H3Grid, LatLonGrid, R_AUTHALIC_KM, make_grid, spatial_block
from firecal.sinusoidal import pixel_area_km2, pixel_centers, parse_date, parse_tile
from firecal.timeutil import firms_utc, local_solar, week_of_year, ym_shift


def test_latlon_cell_ids_and_edges():
    g = LatLonGrid(0.25)
    assert g.nx == 1440 and g.ny == 720
    assert g.cell_of(0.0, 0.0)[()] == 360 * 1440 + 720
    # +180 wraps to -180; lat +90 lands in the top row, -90 in the bottom row
    assert g.cell_of(10.0, 180.0)[()] == g.cell_of(10.0, -180.0)[()]
    assert g.cell_of(90.0, 0.0)[()] // g.nx == g.ny - 1
    assert g.cell_of(-90.0, 0.0)[()] // g.nx == 0


def test_latlon_areas_sum_to_earth():
    g = LatLonGrid(0.25)
    total = g.area_km2(np.arange(g.n_cells)).sum()
    assert total == pytest.approx(4 * np.pi * R_AUTHALIC_KM**2, rel=1e-9)
    # a 0.25 deg cell at 60N is about half the size of one at the equator
    eq, n60 = g.area_km2(g.cell_of([0.1, 60.1], [0.0, 0.0]))
    assert n60 / eq == pytest.approx(np.cos(np.radians(60.125)), rel=1e-3)


def test_latlon_center_roundtrip_and_exact_parent():
    g = LatLonGrid(0.25)
    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(-89, 89, 1000), rng.uniform(-179, 179, 1000)
    c = g.cell_of(lat, lon)
    clat, clon = g.center(c)
    assert np.all(np.abs(clat - lat) <= 0.125 + 1e-9) and np.all(np.abs(clon - lon) <= 0.125 + 1e-9)
    assert np.array_equal(g.cell_of(clat, clon), c)
    # every 1 deg parent holds exactly 16 children
    parents = g.parent(np.arange(g.n_cells), 1.0)
    assert np.all(np.bincount(parents) == 16)


def test_h3_grid_basic(cfg):
    g = H3Grid(5)
    c = g.cell_of([37.0, -33.9], [-122.0, 151.2])
    assert c.dtype == np.int64 and len(set(c)) == 2
    assert np.all((g.area_km2(c) > 150) & (g.area_km2(c) < 350))
    cfg["grid"]["kind"] = "h3"
    assert make_grid(cfg).name == "h3r5"
    assert spatial_block(g, c, cfg).shape == (2,)


def test_make_grid_default_is_quarter_degree(cfg):
    assert make_grid(cfg).name == "ll0p25"


def test_local_solar_time_and_dates():
    # US East Coast afternoon pass: 18:30 UTC at 75W is 13:30 local, same date
    lt = local_solar(pd.Series([pd.Timestamp("2020-07-01 18:30")]), [-75.0])
    assert lt.iloc[0] == pd.Timestamp("2020-07-01 13:30")
    # Eastern Australia night pass: 15:30 UTC at 150E is 01:30 local the NEXT day
    lt = local_solar(pd.Series([pd.Timestamp("2020-01-01 15:30")]), [150.0])
    assert lt.iloc[0] == pd.Timestamp("2020-01-02 01:30")


def test_firms_utc_parses_hhmm():
    t = firms_utc(["2021-08-01", "2021-08-01"], [5, "1835"])
    assert list(t) == [pd.Timestamp("2021-08-01 00:05"), pd.Timestamp("2021-08-01 18:35")]


def test_week_rule_has_52_slots():
    assert list(week_of_year([0, 6, 7, 357, 363, 364, 365])) == [0, 0, 1, 51, 51, 51, 51]


def test_ym_shift_crosses_years():
    assert list(ym_shift([202001, 202012, 202003], -1)) == [201912, 202011, 202002]
    assert list(ym_shift([201912], 13)) == [202101]


def test_sinusoidal_known_tiles():
    # h18v04: upper-left corner is (50N, 0E); first pixel centre is just inside
    lat, lon = pixel_centers(18, 4, 1200)
    assert lat[0, 0] == pytest.approx(50.0 - 10 / 1200 / 2, abs=1e-6)
    assert 0 < lon[0, 0] < 0.02
    # h08v05 (western US): centre near 35N, -116E
    lat, lon = pixel_centers(8, 5, 1200)
    assert lat[600, 600] == pytest.approx(35.0, abs=0.01)
    assert lon[600, 600] == pytest.approx(-116.0, abs=0.2)
    # a tile on the edge of the grid has pixels off the Earth
    lat, _ = pixel_centers(0, 8, 1200)
    assert np.isnan(lat).any() and np.isfinite(lat).any()
    assert pixel_area_km2(1200) == pytest.approx(0.8587, rel=1e-3)


def test_filename_parsing():
    name = "MYD14A1.A2020185.h08v05.061.2020194123456.hdf"
    assert parse_tile(name) == (8, 5)
    assert parse_date(name) == np.datetime64("2020-07-03")
