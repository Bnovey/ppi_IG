"""Tests for igv.attrib — all CPU, analytic results from hand-built models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from igv.attrib import (
    AttribResult,
    completeness_error,
    integrated_gradient,
    plain_gradient,
    predict_mutants,
    score_deltas,
)

L, D = 4, 3


def _linear_fn(W: torch.Tensor):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return (W * x).sum()
    return fn


def _quadratic_fn():
    def fn(x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).sum()
    return fn


# ---------------------------------------------------------------------------
# 1. Analytic gradient for linear model
# ---------------------------------------------------------------------------


class TestAnalyticGradient:
    def test_gausslegendre(self):
        torch.manual_seed(0)
        W = torch.randn(1, L, D)
        embeddings = torch.randn(1, L, D)
        baseline = torch.randn(1, L, D)

        result = integrated_gradient(
            _linear_fn(W), embeddings, baseline=baseline, m_steps=15,
            quadrature="gausslegendre",
        )
        np.testing.assert_allclose(
            result.grad.squeeze(0).numpy(), W.squeeze(0).numpy(), atol=1e-5,
        )

    def test_uniform(self):
        torch.manual_seed(1)
        W = torch.randn(1, L, D)
        embeddings = torch.randn(1, L, D)
        baseline = torch.randn(1, L, D)

        result = integrated_gradient(
            _linear_fn(W), embeddings, baseline=baseline, m_steps=20,
            quadrature="uniform",
        )
        np.testing.assert_allclose(
            result.grad.squeeze(0).numpy(), W.squeeze(0).numpy(), atol=1e-4,
        )

    def test_plain_gradient(self):
        torch.manual_seed(2)
        W = torch.randn(1, L, D)
        embeddings = torch.randn(1, L, D)

        result = plain_gradient(_linear_fn(W), embeddings)
        np.testing.assert_allclose(
            result.grad.squeeze(0).numpy(), W.squeeze(0).numpy(), atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 2. Completeness
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_linear_completeness(self):
        torch.manual_seed(3)
        W = torch.randn(1, L, D)
        embeddings = torch.randn(1, L, D)
        baseline = torch.randn(1, L, D)
        fn = _linear_fn(W)

        result = integrated_gradient(fn, embeddings, baseline=baseline, m_steps=15)
        f_x = float(fn(embeddings))
        f_b = float(fn(baseline))
        err = completeness_error(result, f_x, f_b)
        assert err < 1e-4, f"completeness error {err}"


# ---------------------------------------------------------------------------
# 3. Path invariance for linear model
# ---------------------------------------------------------------------------


class TestPathInvariance:
    def test_m8_vs_m32_vs_plain(self):
        torch.manual_seed(4)
        W = torch.randn(1, L, D)
        embeddings = torch.randn(1, L, D)
        baseline = torch.randn(1, L, D)
        fn = _linear_fn(W)

        r8 = integrated_gradient(fn, embeddings, baseline=baseline, m_steps=8)
        r32 = integrated_gradient(fn, embeddings, baseline=baseline, m_steps=32)
        rp = plain_gradient(fn, embeddings)

        g8 = r8.grad.squeeze(0).numpy()
        g32 = r32.grad.squeeze(0).numpy()
        gp = rp.grad.squeeze(0).numpy()

        np.testing.assert_allclose(g8, g32, atol=1e-5)
        np.testing.assert_allclose(g8, gp, atol=1e-5)
        np.testing.assert_allclose(g32, gp, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. score_deltas hand-computed
# ---------------------------------------------------------------------------


class TestScoreDeltas:
    def test_hand_computed(self):
        grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        delta_emb = {
            (0, "A"): np.array([1.0, 0.0, 0.0]),
            (1, "G"): np.array([0.0, 1.0, 0.0]),
        }
        result = score_deltas(grad, delta_emb)
        assert result[(0, "A")] == pytest.approx(1.0)
        assert result[(1, "G")] == pytest.approx(5.0)

    def test_out_of_range_position(self):
        grad = np.zeros((2, 3))
        with pytest.raises(IndexError, match="out of range"):
            score_deltas(grad, {(5, "A"): np.zeros(3)})

    def test_dimension_mismatch(self):
        grad = np.zeros((2, 3))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            score_deltas(grad, {(0, "A"): np.zeros(4)})


# ---------------------------------------------------------------------------
# 5. predict_mutants additivity
# ---------------------------------------------------------------------------


class TestPredictMutants:
    def test_additivity(self):
        deltas = {(0, "A"): 1.5, (1, "G"): -0.5, (2, "V"): 3.0}
        subs = [
            (),                         # reference
            ((0, "A"),),                 # single
            ((1, "G"),),                 # single
            ((0, "A"), (1, "G")),        # double
            ((0, "A"), (1, "G"), (2, "V")),  # triple
        ]
        pred = predict_mutants(deltas, subs)
        assert pred[0] == 0.0  # reference is exactly zero
        assert pred[3] == pytest.approx(pred[1] + pred[2])  # double = sum of singles
        assert pred[4] == pytest.approx(pred[1] + pred[2] + deltas[(2, "V")])

    def test_empty_tuple_is_zero(self):
        deltas = {(0, "A"): 99.0}
        pred = predict_mutants(deltas, [()])
        assert pred[0] == 0.0


# ---------------------------------------------------------------------------
# 6. Missing-delta KeyError
# ---------------------------------------------------------------------------


class TestMissingDelta:
    def test_keyerror_names_substitution(self):
        deltas = {(0, "A"): 1.0}
        with pytest.raises(KeyError, match=r"\(1, 'G'\)"):
            predict_mutants(deltas, [((1, "G"),)])


# ---------------------------------------------------------------------------
# 7. Non-linear model: IG vs plain gradient differ
# ---------------------------------------------------------------------------


class TestNonLinearGap:
    def test_quadratic_gap(self):
        torch.manual_seed(10)
        embeddings = torch.randn(1, L, D) + 2.0
        baseline = torch.zeros(1, L, D)
        fn = _quadratic_fn()

        r_ig = integrated_gradient(fn, embeddings, baseline=baseline, m_steps=30)
        r_pg = plain_gradient(fn, embeddings, baseline=baseline)

        ig_grad = r_ig.grad.squeeze(0).numpy()
        pg_grad = r_pg.grad.squeeze(0).numpy()

        diff = np.abs(ig_grad - pg_grad).max()
        assert diff > 0.1, (
            f"IG and plain gradient should differ meaningfully for x^2, "
            f"but max diff was only {diff}"
        )

    def test_quadratic_completeness(self):
        torch.manual_seed(11)
        embeddings = torch.randn(1, L, D) + 2.0
        baseline = torch.zeros(1, L, D)
        fn = _quadratic_fn()

        result = integrated_gradient(fn, embeddings, baseline=baseline, m_steps=30)
        f_x = float(fn(embeddings))
        f_b = float(fn(baseline))
        err = completeness_error(result, f_x, f_b)
        assert err < 1e-3, f"completeness error {err} for quadratic model"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
