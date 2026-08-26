"""Evaluation metrics for mutation-effect prediction.

All functions are pure numpy/scipy/pandas -- no torch dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation between predicted and true scores.

    Uses ``scipy.stats.spearmanr`` which applies the standard tie-corrected
    rank method.  Real binding data often has a censoring floor (many tied
    values at the detection limit), so ties are expected and handled correctly.

    Returns NaN if correlation is undefined (e.g. zero variance).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return float("nan")
    r, _ = stats.spearmanr(pred[mask], true[mask])
    return float(r)


def precision_at_k_fold(
    pred: np.ndarray,
    true: np.ndarray,
    wt_value: float,
    k: int = 10,
    fold: float = 5.0,
    log_scale: bool = True,
) -> float:
    """Fraction of top-k predicted variants that are true hits.

    A "hit" is a variant whose ``true`` value represents at least a ``fold``-fold
    improvement over the wild-type value ``wt_value``.

    Parameters
    ----------
    pred : array-like
        Predicted scores.
    true : array-like
        Measured scores.
    wt_value : float
        Wild-type (reference) score.
    k : int
        Number of top predictions to examine.
    fold : float
        Fold-improvement threshold (default 5.0).
    log_scale : bool
        If True (default), scores are ``-log10(Kd)``-like: a ``fold``-fold
        improvement corresponds to ``true >= wt_value + log10(fold)``
        (log base 10).  If False, scores are linear: a hit is
        ``true >= wt_value * fold``.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(pred))
    top_k_idx = np.argsort(pred)[::-1][:k]
    if log_scale:
        threshold = wt_value + np.log10(fold)
    else:
        threshold = wt_value * fold
    hits = np.sum(true[top_k_idx] >= threshold)
    return float(hits / k)


def precision_at_k(
    pred: np.ndarray, true: np.ndarray, k: int
) -> float:
    """Fraction of the top-k by ``pred`` that are also in the top-k by ``true``.

    Parameters
    ----------
    pred, true : array-like
        Score vectors of equal length.
    k : int
        Number of top entries to compare.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(pred))
    top_pred = set(np.argsort(pred)[::-1][:k])
    top_true = set(np.argsort(true)[::-1][:k])
    return float(len(top_pred & top_true) / k)


def stratify_by_n_mut(
    pred: np.ndarray,
    true: np.ndarray,
    n_mut: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = spearman,
) -> pd.DataFrame:
    """Compute a metric stratified by number of mutations.

    Parameters
    ----------
    pred, true : array-like
        Score vectors.
    n_mut : array-like of int
        Number of mutations per variant.
    metric : callable
        Function(pred, true) -> float.  Default is :func:`spearman`.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_mut``, ``count``, ``metric``.  One row per distinct
        ``n_mut`` value, sorted ascending.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n_mut = np.asarray(n_mut, dtype=int)
    rows = []
    for nm in sorted(set(n_mut)):
        mask = n_mut == nm
        rows.append(
            {"n_mut": nm, "count": int(mask.sum()), "metric": metric(pred[mask], true[mask])}
        )
    return pd.DataFrame(rows)


def summary(
    pred: np.ndarray,
    true: np.ndarray,
    n_mut: np.ndarray | None = None,
    wt_value: float | None = None,
) -> dict:
    """Bundle evaluation metrics into a flat dict.

    Parameters
    ----------
    pred, true : array-like
        Score vectors.
    n_mut : array-like of int, optional
        Number of mutations per variant (enables stratified metrics).
    wt_value : float, optional
        Wild-type score (enables precision_at_k_fold).

    Returns
    -------
    dict
        Keys include ``spearman``, ``precision_at_10``, and optionally
        ``precision_at_10_fold5`` and per-stratum spearman values.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    k = min(10, len(pred))
    result: dict = {
        "spearman": spearman(pred, true),
        "precision_at_10": precision_at_k(pred, true, k),
    }
    if wt_value is not None:
        result["precision_at_10_fold5"] = precision_at_k_fold(
            pred, true, wt_value, k=k, fold=5.0
        )
    if n_mut is not None:
        df = stratify_by_n_mut(pred, true, n_mut)
        for _, row in df.iterrows():
            result[f"spearman_nmut{int(row['n_mut'])}"] = row["metric"]
    return result
