"""Tests for igv.metrics — pure numpy/scipy/pandas, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from scipy import stats

from igv.metrics import (
    aggregate_within_position,
    auprc,
    auroc,
    bootstrap_ci,
    hotspot_precision_at_k,
    hotspot_precision_chance,
    partial_spearman,
    precision_at_k,
    precision_at_k_fold,
    spearman,
    step_convergence_spearman,
    stratify_by_n_mut,
    summary,
    topk_overlap_chance,
    within_position_spearman,
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


# ---------------------------------------------------------------------------
# within_position_spearman
# ---------------------------------------------------------------------------


class TestWithinPositionSpearman:
    """Tests for within_position_spearman(pred, true, *, min_n=5)."""

    def _make_matrices(self):
        """Build (4, 20) pred and true matrices with controlled rows.

        Row 0: both increasing -> Spearman = +1.0
        Row 1: pred increasing, true decreasing -> Spearman = -1.0
        Row 2: true is constant -> zero variance -> NaN
        Row 3: only 4 usable pairs (below min_n=5) -> NaN
        """
        pred = np.full((4, 20), np.nan)
        true = np.full((4, 20), np.nan)
        # Row 0: perfect positive
        for j in range(6):
            pred[0, j] = float(j + 1)
            true[0, j] = float(j + 1)
        # Row 1: perfect negative
        for j in range(6):
            pred[1, j] = float(j + 1)
            true[1, j] = float(6 - j)
        # Row 2: constant true (zero variance)
        for j in range(6):
            pred[2, j] = float(j + 1)
            true[2, j] = 5.0
        # Row 3: only 4 pairs (< min_n=5)
        for j in range(4):
            pred[3, j] = float(j)
            true[3, j] = float(j)
        return pred, true

    def test_monotone_row_gives_positive_one(self):
        """A perfectly monotone row must yield +1.0."""
        pred, true = self._make_matrices()
        result = within_position_spearman(pred, true)
        assert result[0] == pytest.approx(1.0, abs=1e-9)

    def test_reversed_row_gives_negative_one(self):
        """A perfectly reversed row must yield -1.0."""
        pred, true = self._make_matrices()
        result = within_position_spearman(pred, true)
        assert result[1] == pytest.approx(-1.0, abs=1e-9)

    def test_constant_true_row_is_nan(self):
        """Zero variance in true must produce NaN."""
        pred, true = self._make_matrices()
        result = within_position_spearman(pred, true)
        assert np.isnan(result[2])

    def test_fewer_than_min_n_pairs_is_nan(self):
        """A row with only 4 finite paired values must produce NaN (min_n=5)."""
        pred, true = self._make_matrices()
        result = within_position_spearman(pred, true)
        assert np.isnan(result[3])

    def test_output_length_equals_row_count(self):
        """Output must have length == n_positions, including NaN rows, to stay aligned."""
        pred, true = self._make_matrices()
        result = within_position_spearman(pred, true)
        assert len(result) == pred.shape[0]
        assert len(result) == 4

    def test_nan_in_pred_and_true_intersected_not_per_array_dropped(self):
        """NaNs in pred and true must be intersected, not dropped independently.

        Construct a row where:
          true[0]  = NaN, true[j]  = j  for j >= 1
          pred[1]  = NaN, pred[0]  = 99, pred[j] = j*2 for j >= 2

        Correct intersection is indices {2, ..., 19}.  At these indices both
        pred and true are monotonically increasing together, so Spearman = +1.0.

        Per-array dropna would compare [99, 4, 6, ...] with [1, 2, 3, ...],
        yielding a much lower (and incorrect) value.
        """
        pred_row = np.full(20, np.nan)
        true_row = np.full(20, np.nan)
        # true: NaN at 0, increasing from index 1
        for j in range(1, 20):
            true_row[j] = float(j)
        # pred: NaN at 1, large value at 0, increasing from index 2
        pred_row[0] = 99.0
        for j in range(2, 20):
            pred_row[j] = float(j) * 2

        pred = pred_row[np.newaxis, :]
        true = true_row[np.newaxis, :]

        result = within_position_spearman(pred, true)
        assert result[0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# aggregate_within_position
# ---------------------------------------------------------------------------


class TestAggregateWithinPosition:
    """Tests for aggregate_within_position(rhos)."""

    def test_all_nan_does_not_crash(self):
        """All-NaN input must not raise."""
        rhos = np.full(10, np.nan)
        result = aggregate_within_position(rhos)
        assert isinstance(result, dict)

    def test_all_nan_n_usable_is_zero(self):
        """All-NaN input must report n_usable == 0."""
        rhos = np.full(10, np.nan)
        result = aggregate_within_position(rhos)
        assert result["n_usable"] == 0

    def test_all_nan_statistics_are_nan(self):
        """mean, median, min, max must all be NaN when there is nothing usable."""
        rhos = np.full(5, np.nan)
        result = aggregate_within_position(rhos)
        assert np.isnan(result["mean"])
        assert np.isnan(result["median"])
        assert np.isnan(result["min"])
        assert np.isnan(result["max"])

    def test_nan_rows_excluded_from_n_usable(self):
        """NaN entries must not count toward n_usable."""
        rhos = np.array([0.5, np.nan, 0.3, np.nan, 0.1])
        result = aggregate_within_position(rhos)
        assert result["n_usable"] == 3

    def test_nan_rows_excluded_from_statistics(self):
        """Mean must equal the mean of finite values only."""
        rhos = np.array([0.4, np.nan, 0.6])
        result = aggregate_within_position(rhos)
        assert result["mean"] == pytest.approx(0.5)

    def test_ci_brackets_mean(self):
        """Bootstrap CI must satisfy ci_lo <= mean <= ci_hi."""
        rhos = np.array([0.1, 0.5, 0.3, -0.1, 0.7, 0.2, 0.4, 0.6, 0.8, 0.0])
        result = aggregate_within_position(rhos)
        assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]

    def test_ci_ordering(self):
        """ci_lo must be <= ci_hi."""
        rhos = np.array([0.1, 0.5, 0.3, -0.1, 0.7, 0.2, 0.4, 0.6, 0.8, 0.0])
        result = aggregate_within_position(rhos)
        assert result["ci_lo"] <= result["ci_hi"]

    def test_n_negative_counts_correctly(self):
        """n_negative must count entries strictly below zero (NaN excluded)."""
        rhos = np.array([0.5, -0.3, 0.1, -0.7, np.nan])
        result = aggregate_within_position(rhos)
        assert result["n_negative"] == 2

    def test_single_usable_value_mean_equals_value(self):
        """With one usable rho, mean must equal that value."""
        rhos = np.array([np.nan, 0.42, np.nan])
        result = aggregate_within_position(rhos)
        assert result["n_usable"] == 1
        assert result["mean"] == pytest.approx(0.42)

    def test_single_usable_value_min_max_equal(self):
        """With one usable rho, min and max must both equal that value."""
        rhos = np.array([np.nan, 0.42, np.nan])
        result = aggregate_within_position(rhos)
        assert result["min"] == pytest.approx(0.42)
        assert result["max"] == pytest.approx(0.42)

    def test_result_has_expected_keys(self):
        """Output dict must contain all documented keys."""
        rhos = np.array([0.3, 0.5, -0.1])
        result = aggregate_within_position(rhos)
        for key in ("mean", "median", "n_usable", "n_negative", "min", "max", "ci_lo", "ci_hi"):
            assert key in result


# ---------------------------------------------------------------------------
# hotspot_precision_at_k
# ---------------------------------------------------------------------------


class TestHotspotPrecisionAtK:
    def test_perfect_recovery(self):
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        assert hotspot_precision_at_k(pred, labels, k=2) == pytest.approx(1.0)

    def test_no_recovery(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        assert hotspot_precision_at_k(pred, labels, k=2) == pytest.approx(0.0)

    def test_partial_recovery(self):
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        # top-2 by pred: indices 0, 1 -> labels 1, 0 -> 1/2
        assert hotspot_precision_at_k(pred, labels, k=2) == pytest.approx(0.5)

    def test_k_greater_than_n(self):
        pred = np.array([3.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 1.0])
        # k clamped to 3, 2 positives out of 3
        assert hotspot_precision_at_k(pred, labels, k=10) == pytest.approx(2.0 / 3.0)

    def test_zero_hot_spots(self):
        pred = np.array([5.0, 4.0, 3.0])
        labels = np.array([0.0, 0.0, 0.0])
        assert hotspot_precision_at_k(pred, labels, k=2) == pytest.approx(0.0)

    def test_all_hot_spots(self):
        pred = np.array([5.0, 4.0, 3.0])
        labels = np.array([1.0, 1.0, 1.0])
        assert hotspot_precision_at_k(pred, labels, k=2) == pytest.approx(1.0)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            hotspot_precision_at_k(np.array([1.0]), np.array([1.0]), k=0)


# ---------------------------------------------------------------------------
# topk_overlap_chance
# ---------------------------------------------------------------------------


class TestTopkOverlapChance:
    def test_basic_arithmetic(self):
        assert topk_overlap_chance(5, 28) == pytest.approx(25.0 / 28.0)

    def test_k10_n28_capped(self):
        # k^2/n = 100/28 > 1.0, capped to 1.0
        assert topk_overlap_chance(10, 28) == pytest.approx(1.0)

    def test_k_equals_n(self):
        assert topk_overlap_chance(10, 10) == pytest.approx(1.0)

    def test_k_greater_than_n(self):
        assert topk_overlap_chance(20, 10) == pytest.approx(1.0)

    def test_capped_at_one(self):
        assert topk_overlap_chance(20, 28) <= 1.0

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be positive"):
            topk_overlap_chance(5, 0)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            topk_overlap_chance(0, 10)


# ---------------------------------------------------------------------------
# hotspot_precision_chance
# ---------------------------------------------------------------------------


class TestHotspotPrecisionChance:
    def test_prevalence(self):
        assert hotspot_precision_chance(7, 28) == pytest.approx(0.25)

    def test_zero_positives(self):
        assert hotspot_precision_chance(0, 28) == pytest.approx(0.0)

    def test_all_positives(self):
        assert hotspot_precision_chance(10, 10) == pytest.approx(1.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be positive"):
            hotspot_precision_chance(0, 0)


# ---------------------------------------------------------------------------
# step_convergence_spearman
# ---------------------------------------------------------------------------


class TestStepConvergenceSpearman:
    def test_identical_rankings(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert step_convergence_spearman(a, a) == pytest.approx(1.0)

    def test_reversed_rankings(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert step_convergence_spearman(a, b) == pytest.approx(-1.0)

    def test_returns_nan_on_constant(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([5.0, 5.0, 5.0])
        assert np.isnan(step_convergence_spearman(a, b))

    def test_noisy_but_correlated(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = a + rng.standard_normal(50) * 0.1
        assert step_convergence_spearman(a, b) > 0.9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
