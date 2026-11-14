import numpy as np
import pandas as pd
import pytest

from firecal import calendar as cal
from firecal.cv import backtest, block_folds, evaluate, simplest_within_1se, summarize
from firecal.grid import LatLonGrid
from firecal.models import GlobalRatio, LightGBMModel, PoissonGLM, StratifiedRatio, poisson_deviance
from firecal.synthetic import make_world


@pytest.fixture(scope="module")
def world():
    return make_world(n_cells=300, years=range(2012, 2023), seed=1)


def ladder():
    return [
        GlobalRatio(),
        StratifiedRatio(["landcover"]),
        PoissonGLM(["viirs", "fwi"], log_features=["viirs"], offset=False),
        LightGBMModel(["viirs", "fwi", "landcover"], objective="poisson", offset=False, num_boost_round=200),
    ]


def test_stratified_ratio_recovers_true_shares(world):
    m = StratifiedRatio(["landcover"]).fit(world)
    truth = world.groupby("landcover").apply(lambda g: g["y"].sum() / g["viirs"].sum(), include_groups=False)
    assert np.allclose(m.table_.sort_index().to_numpy(), truth.sort_index().to_numpy(), rtol=0.05)


def test_exposure_offset_scales_predictions(world):
    m = PoissonGLM(["fwi"], offset=True).fit(world)
    a = world.head(50).copy()
    b = a.assign(exposure=a["exposure"] * 2)
    assert np.allclose(m.predict(b), 2 * m.predict(a))
    lg = LightGBMModel(["fwi", "landcover"], offset=True, num_boost_round=50).fit(world)
    assert np.allclose(lg.predict(b), 2 * lg.predict(a), rtol=1e-9)
    # no clear looks -> no prediction
    assert np.all(m.predict(a.assign(exposure=0.0)) == 0)


def test_ladder_ranks_sensibly_and_blocks_do_not_leak(world):
    g = LatLonGrid(0.25)
    cells = g.cell_of(world["lat"], world["lon"])
    blocks = g.parent(cells, 10.0)
    folds = block_folds(blocks, k=5, seed=0)
    for f in range(5):
        assert not set(blocks[folds == f]) & set(blocks[folds != f])
    res = evaluate(ladder, world, folds)
    s = summarize(res)
    assert s.loc["2_stratified_ratio", "mean"] < s.loc["1_global_ratio", "mean"]
    assert s.loc["5_lightgbm", "mean"] < s.loc["1_global_ratio", "mean"]
    assert simplest_within_1se(s) in s.index


def test_backtest_hides_years(world):
    r = backtest(ladder, world, [2019, 2020, 2021])
    assert set(r["model"]) == {"1_global_ratio", "2_stratified_ratio", "3_poisson_glm", "5_lightgbm"}
    assert (r["rel_bias"].abs() < 0.1).all()


def test_poisson_deviance_zero_for_perfect_fit():
    y = np.array([0, 1, 5, 10])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)


def test_circular_kde_and_season_timing():
    days = np.arange(365)
    prof = np.exp(-0.5 * ((days - 200) / 15) ** 2)
    k = cal.circular_kde(prof, bandwidth_days=5)
    assert k.sum() == pytest.approx(1.0) and abs(int(np.argmax(k)) - 200) <= 1
    t = cal.season_timing(prof)
    assert abs(t["q50_doy0"] - 200) <= 1 and t["q10_doy0"] < 200 < t["q90_doy0"]
    # a season centred on New Year is measured in one piece
    wrap = np.roll(prof, -200 + 3)  # peak at day 3
    tw = cal.season_timing(wrap)
    assert min(tw["q50_doy0"], 365 - tw["q50_doy0"]) <= 4
    assert (tw["q10_doy0"] > 300) and (tw["q90_doy0"] < 60)


def test_week_grid_and_daily_profile():
    df = pd.DataFrame({"year": [2020, 2020, 2021], "week": [0, 51, 10], "count": [1, 2, 3]})
    gtab = cal.week_grid(df)
    assert gtab.shape == (2, 52) and gtab.loc[2020, 51] == 2
    p = cal.daily_profile([0, 364, 365])
    assert p.size == 365 and p[364] == 2
