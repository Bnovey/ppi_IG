"""Tests for igv.metrics — pure numpy/scipy/pandas, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from scipy import stats

from igv.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    partial_spearman,
    precision_at_k,
    precision_at_k_fold,
    spearman,
    stratify_by_n_mut,
    summary,
)


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_monotonic(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert spearman(pred, true) == pytest.approx(1.0)

    def test_perfect_anti_monotonic(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert spearman(pred, true) == pytest.approx(-1.0)

    def test_ties_at_floor(self):
        rng = np.random.default_rng(42)
        n = 50
        true = np.concatenate([np.full(30, 1.0), rng.uniform(2.0, 10.0, 20)])
        pred = rng.standard_normal(n)
        result = spearman(pred, true)
        expected, _ = stats.spearmanr(pred, true)
        assert np.isfinite(result)
        assert result == pytest.approx(expected)

    def test_nan_safe(self):
        pred = np.array([1.0, np.nan, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        result = spearman(pred, true)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_overlap(self):
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        true = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert precision_at_k(pred, true, k=3) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert precision_at_k(pred, true, k=2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        pred = np.array([5.0, 4.0, 1.0, 2.0, 3.0])
        true = np.array([50.0, 10.0, 40.0, 20.0, 30.0])
        # top-3 by pred: indices {0, 1, 4}
        # top-3 by true: indices {0, 2, 4}
        # overlap: {0, 4} -> 2/3
        assert precision_at_k(pred, true, k=3) == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# precision_at_k_fold
# ---------------------------------------------------------------------------


class TestPrecisionAtKFold:
    def test_log_scale(self):
        wt = 6.0  # -log10(Kd) of wild type
        # 5-fold improvement threshold: 6.0 + log10(5) ~ 6.699
        true = np.array([7.0, 5.0, 8.0, 6.5, 6.0])  # hits: idx 0 and 2
        pred = np.array([10.0, 9.0, 8.0, 7.0, 6.0])  # top-3: 0, 1, 2
        # Among top-3 (idx 0,1,2): true[0]=7>=6.699 yes, true[1]=5 no, true[2]=8 yes
        result = precision_at_k_fold(pred, true, wt_value=wt, k=3, fold=5.0)
        assert result == pytest.approx(2.0 / 3.0)

    def test_linear_scale(self):
        wt = 10.0
        true = np.array([60.0, 40.0, 55.0])
        pred = np.array([3.0, 2.0, 1.0])  # top-2: idx 0, 1
        # linear threshold: 10*5 = 50; true[0]=60>=50 yes, true[1]=40 no
        result = precision_at_k_fold(
            pred, true, wt_value=wt, k=2, fold=5.0, log_scale=False,
        )
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# stratify_by_n_mut
# ---------------------------------------------------------------------------


class TestStratifyByNMut:
    def test_counts_and_rows(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        true = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        n_mut = np.array([1, 1, 1, 2, 2, 3])
        df = stratify_by_n_mut(pred, true, n_mut)
        assert len(df) == 3
        assert list(df["n_mut"]) == [1, 2, 3]
        assert list(df["count"]) == [3, 2, 1]

    def test_metric_values(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([10.0, 20.0, 30.0, 40.0])
        n_mut = np.array([1, 1, 2, 2])
        df = stratify_by_n_mut(pred, true, n_mut)
        assert df.loc[df["n_mut"] == 1, "metric"].values[0] == pytest.approx(1.0)
        assert df.loc[df["n_mut"] == 2, "metric"].values[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_basic_keys(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = summary(pred, true)
        assert "spearman" in result
        assert "precision_at_10" in result

    def test_with_wt_value(self):
        pred = np.arange(10, dtype=float)
        true = np.arange(10, dtype=float)
        result = summary(pred, true, wt_value=5.0)
        assert "precision_at_10_fold5" in result

    def test_with_n_mut(self):
        pred = np.arange(6, dtype=float)
        true = np.arange(6, dtype=float)
        n_mut = np.array([1, 1, 1, 2, 2, 2])
        result = summary(pred, true, n_mut=n_mut)
        assert "spearman_nmut1" in result
        assert "spearman_nmut2" in result


# ---------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------


class TestAuroc:
    def test_perfect_separation(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert auroc(scores, labels) == pytest.approx(1.0)

    def test_worst_separation(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert auroc(scores, labels) == pytest.approx(0.0)

    def test_tied_scores(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert auroc(scores, labels) == pytest.approx(0.5)

    def test_no_positives(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([0.0, 0.0])
        assert np.isnan(auroc(scores, labels))

    def test_no_negatives(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([1.0, 1.0])
        assert np.isnan(auroc(scores, labels))

    def test_known_value(self):
        # 3 positives, 2 negatives. Positives at ranks 5, 3, 2 (sorted asc).
        scores = np.array([0.9, 0.4, 0.7, 0.1, 0.5])
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        # Ranks: 0.1->1, 0.4->2, 0.5->3, 0.7->4, 0.9->5
        # Positive ranks: 5, 4, 3 -> sum=12
        # U = 12 - 3*4/2 = 12 - 6 = 6
        # AUROC = 6 / (3*2) = 1.0
        assert auroc(scores, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# auprc
# ---------------------------------------------------------------------------


class TestAuprc:
    def test_perfect_ranking(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert auprc(scores, labels) == pytest.approx(1.0)

    def test_known_value(self):
        scores = np.array([0.9, 0.4, 0.35, 0.8])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        # Sorted desc: [0.9, 0.8, 0.4, 0.35] -> labels [1, 0, 0, 1]
        # tp = [1, 1, 1, 2]; precision = [1.0, 0.5, 0.333, 0.5]
        # recall = [0.5, 0.5, 0.5, 1.0]; d_recall = [0.5, 0.0, 0.0, 0.5]
        # AP = 1.0*0.5 + 0.5*0.5 = 0.75
        assert auprc(scores, labels) == pytest.approx(0.75)

    def test_no_positives(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([0.0, 0.0])
        assert np.isnan(auprc(scores, labels))


# ---------------------------------------------------------------------------
# partial_spearman
# ---------------------------------------------------------------------------


class TestPartialSpearman:
    def test_removes_confound(self):
        rng = np.random.default_rng(42)
        n = 100
        z = rng.standard_normal(n)
        x = z + rng.standard_normal(n) * 0.3
        y = z + rng.standard_normal(n) * 0.3
        simple = spearman(x, y)
        assert simple > 0.7
        partial = partial_spearman(x, y, z)
        assert abs(partial) < abs(simple)

    def test_no_confound_effect(self):
        rng = np.random.default_rng(123)
        n = 50
        x = rng.standard_normal(n)
        y = 2 * x + rng.standard_normal(n) * 0.1
        z = rng.standard_normal(n)
        simple = spearman(x, y)
        partial = partial_spearman(x, y, z)
        assert abs(partial - simple) < 0.15

    def test_multiple_confounds(self):
        rng = np.random.default_rng(99)
        n = 60
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        x = z1 + z2 + rng.standard_normal(n) * 0.05
        y = z1 + z2 + rng.standard_normal(n) * 0.05
        confounds = np.column_stack([z1, z2])
        partial = partial_spearman(x, y, confounds)
        simple = spearman(x, y)
        assert abs(partial) < abs(simple)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCi:
    def test_perfect_correlation(self):
        x = np.arange(20, dtype=float)
        y = 2 * x
        point, lo, hi = bootstrap_ci(spearman, x, y)
        assert point == pytest.approx(1.0)
        assert lo > 0.9
        assert hi == pytest.approx(1.0)

    def test_contains_point(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(30)
        y = x + rng.standard_normal(30) * 0.5
        point, lo, hi = bootstrap_ci(spearman, x, y)
        assert lo <= point <= hi

    def test_wide_ci_for_noise(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(10)
        y = rng.standard_normal(10)
        _, lo, hi = bootstrap_ci(spearman, x, y, n_boot=1000)
        assert hi - lo > 0.3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
