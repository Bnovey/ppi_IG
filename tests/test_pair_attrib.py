"""Tests for pair-layer IG plumbing — all CPU, no boltz, no model.

These tests exercise ONLY the boltz-free mechanics of ``pair_layer_ig``:
shape validation, interpolation grid matching, the dot-product contraction
over 128 channels (including sign preservation), and symmetrisation.

The real seam test — calling the actual confidence head on a real model and
checking that ``z.grad`` is non-None, finite, and completeness holds — must
run on the A100 in Stage 1. A synthetic mock like ``(z**2).sum()*0.01``
would be a tautology: IG on a quadratic satisfies completeness by
construction and exercises none of the actual risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from igv.attrib import (
    PairAttribResult,
    integrated_gradient,
    pair_completeness_error,
    pair_layer_ig,
)


# ---------------------------------------------------------------------------
# 1. Shape validation
# ---------------------------------------------------------------------------


class TestShapeValidation:
    def test_rejects_rank3_z_baseline(self):
        z = torch.randn(4, 4, 128)
        with pytest.raises(ValueError, match="rank 4"):
            pair_layer_ig(lambda z: z.sum(), z, torch.randn(1, 4, 4, 128))

    def test_rejects_rank3_z_x(self):
        with pytest.raises(ValueError, match="rank 4"):
            pair_layer_ig(lambda z: z.sum(), torch.randn(1, 4, 4, 128), torch.randn(4, 4, 128))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            pair_layer_ig(
                lambda z: z.sum(),
                torch.randn(1, 4, 4, 128),
                torch.randn(1, 5, 5, 128),
            )

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="128 channels"):
            pair_layer_ig(
                lambda z: z.sum(),
                torch.randn(1, 4, 4, 64),
                torch.randn(1, 4, 4, 64),
            )

    def test_rejects_mismatched_L(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            pair_layer_ig(
                lambda z: z.sum(),
                torch.randn(1, 3, 4, 128),
                torch.randn(1, 4, 4, 128),
            )


# ---------------------------------------------------------------------------
# 2. Interpolation grid matches integrated_gradient
# ---------------------------------------------------------------------------


class TestInterpolationGrid:
    """The quadrature nodes and weights must be identical to integrated_gradient's."""

    @staticmethod
    def _extract_alphas_weights_from_ig(m_steps, quadrature, dtype=torch.float64):
        """Run integrated_gradient with a probe that records the alphas it sees."""
        seen = []

        def probe(x):
            seen.append(x.clone())
            return x.sum()

        D = 3
        emb = torch.randn(1, 2, D, dtype=dtype)
        baseline = torch.zeros(1, 2, D, dtype=dtype)
        integrated_gradient(probe, emb, baseline=baseline, m_steps=m_steps, quadrature=quadrature)

        alphas = []
        delta = emb - baseline
        for interp in seen:
            # interp = baseline + alpha * delta; solve for alpha at first nonzero delta element
            flat_d = delta.flatten()
            flat_i = (interp - baseline).flatten()
            idx = flat_d.abs().argmax()
            alpha = flat_i[idx] / flat_d[idx]
            alphas.append(float(alpha))
        return alphas

    @staticmethod
    def _extract_alphas_from_pair_ig(m_steps, quadrature, dtype=torch.float64):
        seen = []

        def probe(z):
            seen.append(z.clone())
            return z.sum()

        L = 2
        z_x = torch.randn(1, L, L, 128, dtype=dtype)
        z_b = torch.zeros(1, L, L, 128, dtype=dtype)
        pair_layer_ig(probe, z_b, z_x, m_steps=m_steps, quadrature=quadrature)

        alphas = []
        delta = z_x - z_b
        for interp in seen:
            flat_d = delta.flatten()
            flat_i = (interp - z_b).flatten()
            idx = flat_d.abs().argmax()
            alpha = flat_i[idx] / flat_d[idx]
            alphas.append(float(alpha))
        return alphas

    def test_gausslegendre_alphas_match(self):
        for m in [5, 10, 15]:
            ig_alphas = self._extract_alphas_weights_from_ig(m, "gausslegendre")
            pair_alphas = self._extract_alphas_from_pair_ig(m, "gausslegendre")
            np.testing.assert_allclose(pair_alphas, ig_alphas, atol=1e-12)

    def test_uniform_alphas_match(self):
        for m in [5, 10, 20]:
            ig_alphas = self._extract_alphas_weights_from_ig(m, "uniform")
            pair_alphas = self._extract_alphas_from_pair_ig(m, "uniform")
            np.testing.assert_allclose(pair_alphas, ig_alphas, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Dot-product contraction: sign preservation and correctness
# ---------------------------------------------------------------------------


class TestDotProductContraction:
    """Hand-constructed tensors where the (L,L) answer is known analytically."""

    def test_linear_score_hand_computed(self):
        """f(z) = sum over all elements. Gradient is 1 everywhere.
        delta * grad summed over C channels = sum_c (z_x - z_b)[i,j,c] * 1.
        Before symmetrisation: A_unsym[i,j] = sum_c delta[i,j,c].
        After: A[i,j] = A_unsym[i,j] + A_unsym[j,i].
        """
        L, C = 2, 128
        torch.manual_seed(42)
        z_x = torch.randn(1, L, L, C, dtype=torch.float64)
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=10, quadrature="gausslegendre"
        )

        delta = (z_x - z_b).squeeze(0)
        unsym = delta.sum(dim=-1)
        expected = (unsym + unsym.T) / 2

        torch.testing.assert_close(
            result.interaction_map.double(), expected, atol=1e-10, rtol=1e-10
        )

    def test_sign_preserved_negative_entries(self):
        """Construct z_x - z_b so that one pair has all-negative channels.
        The dot product must be negative; an L2 norm would not be.
        """
        L, C = 3, 128
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x = torch.zeros(1, L, L, C, dtype=torch.float64)
        # Make pair (0,1) have all-negative delta
        z_x[0, 0, 1, :] = -1.0

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=10, quadrature="gausslegendre"
        )

        # unsym[0,1] = sum_c(-1) * 1 = -128, unsym[1,0] = 0
        # A[0,1] = (-128 + 0) / 2 = -64
        assert result.interaction_map[0, 1].item() < 0, (
            "Dot product must preserve sign; got non-negative for an all-negative delta"
        )
        assert result.interaction_map[0, 1].item() == pytest.approx(-64.0, abs=1e-6)

    def test_l2_norm_would_disagree(self):
        """Demonstrate that an L2 norm gives a different, always-nonneg answer."""
        L, C = 3, 128
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x[0, 0, 1, :] = -1.0
        z_x[0, 1, 0, :] = 0.5

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=10, quadrature="gausslegendre"
        )

        # The interaction_map from dot product:
        # unsym[0,1] = -128, unsym[1,0] = 64
        # A[0,1] = (-128 + 64) / 2 = -32  (negative)
        assert result.interaction_map[0, 1].item() < 0

        # An L2 norm would give: ||delta[0,1]|| + ||delta[1,0]|| = 128^0.5*... > 0
        delta = (z_x - z_b).squeeze(0)
        l2_unsym = delta.norm(dim=-1)
        l2_map = l2_unsym + l2_unsym.T
        assert (l2_map >= 0).all(), "L2 norm must be non-negative everywhere"
        assert l2_map[0, 1].item() > 0, "L2 norm would be positive where dot product is negative"

    def test_weighted_linear_channels(self):
        """f(z) = (W * z).sum() with known W; gradient is W everywhere.

        A_unsym[i,j] = sum_c delta[i,j,c] * W[c]
        """
        L, C = 2, 128
        torch.manual_seed(99)
        W = torch.randn(C, dtype=torch.float64)
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x = torch.randn(1, L, L, C, dtype=torch.float64)

        def score_fn(z):
            return (W * z).sum()

        result = pair_layer_ig(
            score_fn, z_b, z_x, m_steps=15, quadrature="gausslegendre"
        )

        delta = (z_x - z_b).squeeze(0)
        unsym = (delta * W).sum(dim=-1)
        expected = (unsym + unsym.T) / 2

        torch.testing.assert_close(
            result.interaction_map.double(), expected, atol=1e-8, rtol=1e-8
        )


# ---------------------------------------------------------------------------
# 4. Symmetrisation
# ---------------------------------------------------------------------------


class TestSymmetrisation:
    def test_output_is_symmetric(self):
        L, C = 4, 128
        torch.manual_seed(7)
        z_x = torch.randn(1, L, L, C, dtype=torch.float64)
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=5, quadrature="gausslegendre"
        )

        torch.testing.assert_close(
            result.interaction_map, result.interaction_map.T, atol=0, rtol=0
        )

    def test_asymmetric_delta_becomes_symmetric(self):
        """Even when z_x - z_b is not symmetric, the output map is."""
        L, C = 3, 128
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x = torch.zeros(1, L, L, C, dtype=torch.float64)
        z_x[0, 0, 2, :] = 1.0  # only (0,2) is nonzero, not (2,0)

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=5, quadrature="gausslegendre"
        )

        assert result.interaction_map[0, 2].item() == pytest.approx(
            result.interaction_map[2, 0].item()
        )
        assert result.interaction_map[0, 2].item() != 0.0


# ---------------------------------------------------------------------------
# 5. PairAttribResult and pair_completeness_error
# ---------------------------------------------------------------------------


class TestPairCompletenessError:
    def test_perfect_completeness(self):
        """For f(z) = z.sum(), completeness should be near-exact."""
        L, C = 2, 128
        torch.manual_seed(55)
        z_x = torch.randn(1, L, L, C, dtype=torch.float64)
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=10, quadrature="gausslegendre"
        )

        f_x = float(z_x.sum())
        f_b = float(z_b.sum())
        err = pair_completeness_error(result, f_x, f_b)
        assert err < 1e-6, f"completeness error {err}"

    def test_zero_diff_zero_sum(self):
        m = PairAttribResult(
            interaction_map=torch.zeros(3, 3),
            grad=torch.zeros(1, 3, 3, 128),
            n_steps=5,
            quadrature="gausslegendre",
        )
        assert pair_completeness_error(m, 1.0, 1.0) == 0.0

    def test_zero_diff_nonzero_sum(self):
        m = PairAttribResult(
            interaction_map=torch.ones(3, 3),
            grad=torch.zeros(1, 3, 3, 128),
            n_steps=5,
            quadrature="gausslegendre",
        )
        assert pair_completeness_error(m, 1.0, 1.0) == float("inf")


# ---------------------------------------------------------------------------
# 6. Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_grad_shape(self):
        L, C = 3, 128
        z_x = torch.randn(1, L, L, C, dtype=torch.float64)
        z_b = torch.zeros(1, L, L, C, dtype=torch.float64)

        result = pair_layer_ig(
            lambda z: z.sum(), z_b, z_x, m_steps=5, quadrature="gausslegendre"
        )

        assert result.grad.shape == (1, L, L, C)
        assert result.interaction_map.shape == (L, L)
        assert result.n_steps == 5
        assert result.quadrature == "gausslegendre"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
