"""Tests for igv.attrib — all CPU, analytic results from hand-built models."""

from __future__ import annotations

import contextlib
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
    make_dead_target,
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


# ---------------------------------------------------------------------------
# offload_large_saved_tensors
# ---------------------------------------------------------------------------


def test_offload_is_numerically_exact_and_size_gated():
    """Offloading saved tensors must not change the gradient at all.

    Runs on CPU, where pack() short-circuits on ``not t.is_cuda``, so this
    pins the contract (identical grads, no crash, counter stays at zero for
    non-CUDA tensors) rather than the transfer itself.
    """
    from igv.attrib import offload_large_saved_tensors

    def run(ctx):
        x = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
        w = torch.randn(64, 64, dtype=torch.float64)
        with ctx:
            y = (x @ w).tanh().sum()
            y.backward()
        return x.grad.clone()

    torch.manual_seed(0)
    g_plain = run(contextlib.nullcontext())
    torch.manual_seed(0)
    with offload_large_saved_tensors(min_bytes=1) as stats:
        g_off = run(contextlib.nullcontext())
    torch.testing.assert_close(g_plain, g_off, rtol=0, atol=0)
    # CPU tensors are never offloaded regardless of how low the threshold is.
    assert stats["count"] == 0


def test_offload_threshold_default_is_above_a_500_token_pair_tensor():
    """The threshold should engage for 730 tokens but not ~500.

    z is (1, L, L, 128) fp32.  This is the property that makes the offload
    self-scaling, so it is worth locking down.
    """
    from igv.attrib import DEFAULT_OFFLOAD_MIN_BYTES

    def z_bytes(L):
        return L * L * 128 * 4

    assert z_bytes(730) > DEFAULT_OFFLOAD_MIN_BYTES
    assert z_bytes(500) < DEFAULT_OFFLOAD_MIN_BYTES

# ---------------------------------------------------------------------------
# make_dead_target
# ---------------------------------------------------------------------------


class TestMakeDeadTarget:
    """The dead-target check must be evidence, not a tautology.

    The point of the helper is that the *value* comes from the real forward
    while the *gradient* is exactly zero.  These tests pin both halves, plus
    the two implementation properties the value depends on: the forward is
    handed a detached tensor, and the objective is 0-dim.
    """

    @staticmethod
    def _x():
        torch.manual_seed(20)
        return torch.randn(1, L, D)

    def test_gradient_is_exactly_zero_plain(self):
        x = self._x()
        res = plain_gradient(make_dead_target(_quadratic_fn()), x)
        assert float(res.grad.abs().max()) == 0.0

    def test_gradient_is_exactly_zero_through_ig(self):
        x = self._x()
        baseline = torch.zeros_like(x)
        res = integrated_gradient(
            make_dead_target(_quadratic_fn()), x, baseline=baseline, m_steps=4,
        )
        assert float(res.grad.abs().max()) == 0.0
        assert float(res.ig.abs().max()) == 0.0

    def test_value_tracks_the_forward_fn(self):
        """Change what forward_fn returns; the objective's value must follow.

        This is the whole fix: today's ``(x * 0.0).sum() + 1.0`` records a
        constant 1.0 no matter what the model does.
        """
        x = self._x()
        obj = make_dead_target(_quadratic_fn())
        assert float(obj(x)) == pytest.approx(float((x ** 2).sum()))

        stub_value = {"v": 3.5}
        stub = make_dead_target(lambda t: torch.as_tensor(stub_value["v"]))
        first = float(stub(x))
        stub_value["v"] = -11.25
        second = float(stub(x))
        assert first == pytest.approx(3.5)
        assert second == pytest.approx(-11.25)
        assert first != second

    def test_value_is_model_dependent_not_constant_one(self):
        """Two different models must give two different recorded values."""
        x = self._x()
        W = torch.randn(1, L, D)
        v_quad = float(make_dead_target(_quadratic_fn())(x))
        v_lin = float(make_dead_target(_linear_fn(W))(x))
        assert v_quad != v_lin
        assert v_quad != 1.0

    def test_forward_fn_receives_a_detached_tensor(self):
        """A detached input under no_grad is what keeps the check cheap.

        ``confidence_forward`` also asserts ``scalar.requires_grad`` whenever
        grad is enabled, so the inner forward must not run with grad on.
        """
        seen = {}

        def stub(t):
            seen["requires_grad"] = t.requires_grad
            seen["grad_enabled"] = torch.is_grad_enabled()
            return (t ** 2).sum()

        plain_gradient(make_dead_target(stub), self._x())
        assert seen["requires_grad"] is False
        assert seen["grad_enabled"] is False

    def test_objective_is_zero_dim_even_for_shape_1_forward(self):
        x = self._x()
        obj = make_dead_target(lambda t: (t ** 2).sum().reshape(1))
        out = obj(x)
        assert out.dim() == 0
        # .backward() with no grad_output is the contract integrated_gradient
        # relies on; it only works for a 0-dim tensor.
        leaf = x.detach().requires_grad_(True)
        obj(leaf).backward()
        assert float(leaf.grad.abs().max()) == 0.0

    def test_objective_dtype_and_device_follow_the_input(self):
        x = torch.randn(1, L, D, dtype=torch.float64)
        obj = make_dead_target(lambda t: torch.as_tensor(2.0, dtype=torch.float32))
        out = obj(x)
        assert out.dtype == x.dtype
        assert out.device == x.device

    def test_non_scalar_forward_is_rejected(self):
        obj = make_dead_target(lambda t: t.sum(dim=-1))
        with pytest.raises(ValueError, match="scalar forward_fn"):
            obj(self._x())

    def test_weight_gives_an_analytic_positive_control(self):
        """weight != 0 makes the gradient analytically ``weight`` everywhere.

        A zero reading there indicts the attribution plumbing, not the model,
        so the same helper covers both directions of the check.
        """
        x = self._x()
        res = plain_gradient(make_dead_target(_quadratic_fn(), weight=0.25), x)
        torch.testing.assert_close(
            res.grad, torch.full_like(res.grad, 0.25), rtol=0, atol=0,
        )


# ---------------------------------------------------------------------------
# uniform-quadrature alphas dtype
# ---------------------------------------------------------------------------


def test_uniform_quadrature_respects_embedding_dtype():
    """``alphas`` must be built at ``embeddings.dtype``, like ``weights``.

    Without an explicit dtype, ``torch.linspace`` takes the *global* default,
    so the interpolation precision and the weighting precision could disagree.
    """
    W = torch.randn(1, L, D, dtype=torch.float64)
    embeddings = torch.randn(1, L, D, dtype=torch.float64)
    baseline = torch.zeros(1, L, D, dtype=torch.float64)

    res = integrated_gradient(
        _linear_fn(W), embeddings, baseline=baseline, m_steps=8,
        quadrature="uniform",
    )
    assert res.grad.dtype == torch.float64
    np.testing.assert_allclose(
        res.grad.squeeze(0).numpy(), W.squeeze(0).numpy(), atol=1e-12,
    )
