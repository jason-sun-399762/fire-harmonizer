"""Validation: spatial-block folds, the year back-test, and 'simplest model within one SE' selection."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .models import score


def block_folds(blocks, k: int, seed: int = 42) -> np.ndarray:
    """Assign whole spatial blocks to k folds, so neighbouring cells never split across train/test."""
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    fold_of_block = dict(zip(uniq[perm], np.arange(uniq.size) % k))
    return np.array([fold_of_block[b] for b in blocks])


def evaluate(make_models: Callable[[], list], df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Fit every model on k-1 folds, score on the held-out fold. One row per model x fold."""
    rows = []
    for f in np.unique(folds):
        tr, te = df[folds != f], df[folds == f]
        for m in make_models():
            m.fit(tr)
            rows.append({"model": m.name, "fold": int(f), **score(te["y"], m.predict(te))})
    return pd.DataFrame(rows)


def backtest(make_models: Callable[[], list], df: pd.DataFrame, test_years: list[int], year_col: str = "year") -> pd.DataFrame:
    """Hide the test years entirely, train on the rest, predict the hidden years."""
    te_mask = df[year_col].isin(test_years)
    tr, te = df[~te_mask], df[te_mask]
    rows = []
    for m in make_models():
        m.fit(tr)
        rows.append({"model": m.name, **score(te["y"], m.predict(te))})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, metric: str = "deviance") -> pd.DataFrame:
    g = results.groupby("model")[metric]
    out = pd.DataFrame({"mean": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count())})
    return out.sort_index()


def simplest_within_1se(summary: pd.DataFrame) -> str:
    """Lowest-deviance model's mean + SE sets the bar; return the simplest model under it.
    Model names start with their rung number, so sorting by name = sorting by simplicity."""
    best = summary["mean"].idxmin()
    bar = summary.loc[best, "mean"] + summary.loc[best, "se"]
    ok = summary[summary["mean"] <= bar]
    return sorted(ok.index)[0]
