"""Model ladder, simplest first. Every model predicts expected MODIS-Aqua-equivalent counts.

Common input frame columns:
  y         MODIS-Aqua target count (training only)
  viirs     VIIRS signal in S-NPP-equivalent units (e.g. n_pixel_days)
  exposure  clear-look count used as offset (optional; 1 if absent)
  features  any extra columns listed per model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _exposure(df: pd.DataFrame) -> np.ndarray:
    return df["exposure"].to_numpy(float) if "exposure" in df else np.ones(len(df))


class GlobalRatio:
    """One worldwide MODIS/VIIRS ratio."""

    name = "1_global_ratio"

    def fit(self, df):
        self.ratio_ = df["y"].sum() / max(df["viirs"].sum(), EPS)
        return self

    def predict(self, df):
        return df["viirs"].to_numpy(float) * self.ratio_


class StratifiedRatio:
    """A ratio per stratum (e.g. land cover x month), shrunk toward the global ratio when data are thin."""

    name = "2_stratified_ratio"

    def __init__(self, strata: list[str], prior_strength: float = 50.0):
        self.strata = strata
        self.k = prior_strength

    def fit(self, df):
        g = df["y"].sum() / max(df["viirs"].sum(), EPS)
        s = df.groupby(self.strata, observed=True)[["y", "viirs"]].sum()
        self.global_ = g
        self.table_ = ((s["y"] + self.k * g) / (s["viirs"] + self.k)).rename("ratio")
        return self

    def predict(self, df):
        r = df[self.strata].merge(self.table_.reset_index(), how="left", on=self.strata)["ratio"]
        return df["viirs"].to_numpy(float) * r.fillna(self.global_).to_numpy()


class PoissonGLM:
    """Poisson regression with a log(exposure) offset and a handful of inputs."""

    name = "3_poisson_glm"

    def __init__(self, features: list[str], log_features: list[str] | None = None, offset: bool = True):
        self.features = features
        self.log_features = set(log_features or [])
        self.offset = offset

    def _X(self, df):
        cols = []
        for f in self.features:
            x = df[f].to_numpy(float)
            cols.append(np.log1p(x) if f in self.log_features else x)
        return np.column_stack([np.ones(len(df)), *cols])

    def fit(self, df):
        import statsmodels.api as sm

        e = _exposure(df)
        m = e > 0
        off = np.log(e[m]) if self.offset else None
        self.res_ = sm.GLM(df["y"].to_numpy(float)[m], self._X(df[m]), family=sm.families.Poisson(), offset=off).fit()
        return self

    def predict(self, df):
        e = _exposure(df)
        eta = self._X(df) @ self.res_.params
        mu = np.exp(np.clip(eta, -50, 50))
        return np.where(e > 0, mu * (e if self.offset else 1.0), 0.0)


class LightGBMModel:
    """Gradient boosting (Tweedie or Poisson) with log(exposure) as a fixed offset (init_score)."""

    name = "5_lightgbm"

    def __init__(self, features: list[str], objective: str = "tweedie", offset: bool = True, params: dict | None = None,
                 num_boost_round: int = 400, seed: int = 42):
        self.features = features
        self.objective = objective
        self.offset = offset
        self.params = {
            "objective": objective,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": seed,
            "verbose": -1,
            **({"tweedie_variance_power": 1.2} if objective == "tweedie" else {}),
            **(params or {}),
        }
        self.rounds = num_boost_round

    def fit(self, df):
        import lightgbm as lgb

        e = _exposure(df)
        m = e > 0
        init = np.log(e[m]) if self.offset else None
        ds = lgb.Dataset(df.loc[m, self.features], label=df["y"].to_numpy(float)[m], init_score=init)
        self.booster_ = lgb.train(self.params, ds, num_boost_round=self.rounds)
        return self

    def predict(self, df):
        e = _exposure(df)
        raw = self.booster_.predict(df[self.features], raw_score=True)
        mu = np.exp(np.clip(raw + (np.log(np.where(e > 0, e, 1.0)) if self.offset else 0.0), -50, 50))
        return np.where(e > 0, mu, 0.0)


# ---------- metrics ----------

def poisson_deviance(y, mu) -> float:
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def relative_bias(y, mu) -> float:
    return float((np.sum(mu) - np.sum(y)) / max(np.sum(y), EPS))


def r2_log(y, mu) -> float:
    a, b = np.log1p(np.asarray(y, float)), np.log1p(np.asarray(mu, float))
    ss = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / ss) if ss > 0 else float("nan")


def score(y, mu) -> dict:
    return {"deviance": poisson_deviance(y, mu), "rel_bias": relative_bias(y, mu), "r2_log": r2_log(y, mu)}
