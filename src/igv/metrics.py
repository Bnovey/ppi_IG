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


# ---------------------------------------------------------------------------
# AUROC / AUPRC
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    *labels* must be binary (0 or 1).  Returns NaN if either class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    *labels* must be binary (0 or 1).  Ties are broken conservatively
    (negatives ranked before positives at equal score).
    Returns NaN if there are no positives.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.lexsort((labels, -scores))
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    n = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    precision = tp / n
    recall = tp / n_pos
    d_recall = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * d_recall))


# ---------------------------------------------------------------------------
# Partial correlation and bootstrap
# ---------------------------------------------------------------------------


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    confounds: np.ndarray,
) -> float:
    """Spearman correlation between *x* and *y* after regressing out *confounds*.

    Confounds are removed via ordinary least squares (with intercept);
    the Spearman correlation of the residuals is returned.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Z = np.asarray(confounds, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z_aug = np.column_stack([np.ones(len(x)), Z])
    coef_x, _, _, _ = np.linalg.lstsq(Z_aug, x, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(Z_aug, y, rcond=None)
    resid_x = x - Z_aug @ coef_x
    resid_y = y - Z_aug @ coef_y
    return spearman(resid_x, resid_y)


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns ``(point_estimate, lo, hi)`` where *lo* and *hi* are the
    percentile bounds for the given confidence level.
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    point = fn(*arrays)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = fn(*(a[idx] for a in arrays))
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(boots, 100 * alpha))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha)))
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# Within-position metrics
# ---------------------------------------------------------------------------


def within_position_spearman(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    min_n: int = 5,
) -> np.ndarray:
    """Per-position Spearman correlation between two ``(n_positions, 20)`` matrices.

    For each row, computes the Spearman correlation across the substitutions
    where **both** matrices are non-NaN.  Returns ``np.nan`` for a row with
    fewer than *min_n* usable pairs or with zero variance on either side.

    The returned array has length ``n_positions`` -- NaN rows are kept in place
    so the result stays aligned with position labels.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n_pos = pred.shape[0]
    result = np.full(n_pos, np.nan)

    for i in range(n_pos):
        mask = np.isfinite(pred[i]) & np.isfinite(true[i])
        if mask.sum() < min_n:
            continue
        p = pred[i, mask]
        t = true[i, mask]
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        result[i] = spearman(p, t)

    return result


def aggregate_within_position(rhos: np.ndarray) -> dict:
    """Summary statistics for an array of per-position Spearman correlations.

    Resamples **positions** (the independent unit) for the bootstrap CI on
    the mean, via :func:`bootstrap_ci`.
    """
    rhos = np.asarray(rhos, dtype=np.float64)
    usable = rhos[np.isfinite(rhos)]
    n_usable = len(usable)

    if n_usable == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "n_usable": 0,
            "n_negative": 0,
            "min": float("nan"),
            "max": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }

    mean_val = float(np.mean(usable))
    _, ci_lo, ci_hi = bootstrap_ci(lambda a: float(np.mean(a)), usable)

    return {
        "mean": mean_val,
        "median": float(np.median(usable)),
        "n_usable": n_usable,
        "n_negative": int((usable < 0).sum()),
        "min": float(np.min(usable)),
        "max": float(np.max(usable)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }
