"""Automatic checks. A failed check raises, so the pipeline stops instead of writing bad data."""

from __future__ import annotations

import numpy as np


class QCError(RuntimeError):
    pass


def check_exact(name: str, before: float, after: float) -> None:
    if before != after:
        raise QCError(f"{name}: {before} before aggregation, {after} after (must match exactly)")


def check_rel(name: str, before: float, after: float, rel_tol: float) -> None:
    if before == 0 and after == 0:
        return
    rel = abs(after - before) / max(abs(before), 1e-12)
    if rel > rel_tol:
        raise QCError(f"{name}: relative difference {rel:.4%} exceeds {rel_tol:.4%}")


def check_range(name: str, values, lo: float, hi: float, allow_nan: bool = True) -> None:
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not allow_nan and not finite.all():
        raise QCError(f"{name}: {int((~finite).sum())} missing values")
    bad = finite & ((v < lo) | (v > hi))
    if bad.any():
        raise QCError(f"{name}: {int(bad.sum())} values outside [{lo}, {hi}]")


def check_fractions(name: str, frac_columns) -> None:
    """Class fractions in each cell must sum to 1 (within float error) where defined."""
    s = np.nansum(np.column_stack(frac_columns), axis=1)
    defined = np.any(np.isfinite(np.column_stack(frac_columns)), axis=1)
    if np.any(np.abs(s[defined] - 1.0) > 1e-6):
        raise QCError(f"{name}: class fractions do not sum to 1")
