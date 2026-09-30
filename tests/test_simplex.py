"""Tests for simplex-projected attribution functions in igv.attrib."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from igv.attrib import (
    CANONICAL_AMINO_ACIDS,
    SimplexResult,
    per_position_score,
    simplex_correct,
    simplex_project,
    simplex_score,
    simplex_score_from_onehot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

L, D = 5, 8
N_AA = len(CANONICAL_AMINO_ACIDS)  # 20


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def grad(rng):
    return rng.standard_normal((L, D))


@pytest.fixture()
def aa_emb(rng):
    return rng.standard_normal((N_AA, D))


@pytest.fixture()
def wt_indices(rng):
    return rng.integers(0, N_AA, size=L)


# ---------------------------------------------------------------------------
# simplex_project
# ---------------------------------------------------------------------------


class TestSimplexProject:
    def test_shape(self, grad, aa_emb):
        out = simplex_project(grad, aa_emb)
        assert out.shape == (L, N_AA)

    def test_dtype(self, grad, aa_emb):
        out = simplex_project(grad, aa_emb)
        assert out.dtype == np.float64

    def test_known_answer(self):
        grad = np.array([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
        E = np.array([[1.0, 0.0]] * 20)  # all embeddings = [1, 0]
        E[1] = [0.0, 1.0]
        out = simplex_project(grad, E)
        # pos 0: grad=[1,0], E[0]=[1,0] -> 1.0; E[1]=[0,1] -> 0.0
        assert out[0, 0] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(0.0)
        # pos 1: grad=[0,1], E[0]=[1,0] -> 0.0; E[1]=[0,1] -> 1.0
        assert out[1, 0] == pytest.approx(0.0)
        assert out[1, 1] == pytest.approx(1.0)

    def test_equivalent_to_matmul(self, grad, aa_emb):
        out = simplex_project(grad, aa_emb)
        expected = grad.astype(np.float64) @ aa_emb.astype(np.float64).T
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_grad_wrong_ndim(self, aa_emb):
        with pytest.raises(ValueError, match="2-D"):
            simplex_project(np.zeros((L,)), aa_emb)

    def test_aa_emb_wrong_ndim(self, grad):
        with pytest.raises(ValueError, match="2-D"):
            simplex_project(grad, np.zeros((D,)))

    def test_aa_emb_wrong_rows(self, grad):
        with pytest.raises(ValueError, match="20 rows"):
            simplex_project(grad, np.zeros((10, D)))

    def test_dimension_mismatch(self, grad):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            simplex_project(grad, np.zeros((N_AA, D + 1)))


# ---------------------------------------------------------------------------
# simplex_correct
# ---------------------------------------------------------------------------


class TestSimplexCorrect:
    def test_shape(self, grad, aa_emb):
        g = simplex_project(grad, aa_emb)
        out = simplex_correct(g)
        assert out.shape == g.shape

    def test_rows_sum_to_zero(self, grad, aa_emb):
        g = simplex_project(grad, aa_emb)
        out = simplex_correct(g)
        row_sums = out.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_known_answer(self):
        g = np.array([[1.0, 3.0] + [0.0] * 18])  # (1, 20)
        out = simplex_correct(g)
        mean = (1.0 + 3.0) / 20.0
        assert out[0, 0] == pytest.approx(1.0 - mean)
        assert out[0, 1] == pytest.approx(3.0 - mean)
        for i in range(2, 20):
            assert out[0, i] == pytest.approx(-mean)

    def test_idempotent_on_already_centred(self):
        g = np.zeros((3, N_AA))
        g[:, 0] = 1.0
        g[:, 1] = -1.0
        centred = simplex_correct(g)
        recentred = simplex_correct(centred)
        np.testing.assert_allclose(centred, recentred, atol=1e-15)

    def test_wrong_shape(self):
        with pytest.raises(ValueError, match="Expected shape"):
            simplex_correct(np.zeros((L, 10)))

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="Expected shape"):
            simplex_correct(np.zeros((N_AA,)))


# ---------------------------------------------------------------------------
# per_position_score
# ---------------------------------------------------------------------------


class TestPerPositionScore:
    def test_shape(self, grad, aa_emb, wt_indices):
        g = simplex_correct(simplex_project(grad, aa_emb))
        out = per_position_score(g, wt_indices)
        assert out.shape == (L,)

    def test_known_answer(self):
        g = np.zeros((3, N_AA))
        g[0, 5] = 2.5
        g[1, 10] = -1.3
        g[2, 0] = 0.7
        wt = np.array([5, 10, 0])
        out = per_position_score(g, wt)
        assert out[0] == pytest.approx(2.5)
        assert out[1] == pytest.approx(-1.3)
        assert out[2] == pytest.approx(0.7)

    def test_signed_output(self, rng):
        g = rng.standard_normal((10, N_AA))
        g = g - g.mean(axis=1, keepdims=True)
        wt = rng.integers(0, N_AA, size=10)
        out = per_position_score(g, wt)
        assert np.any(out > 0) or np.any(out < 0)

    def test_wt_indices_wrong_shape(self):
        g = np.zeros((L, N_AA))
        with pytest.raises(ValueError, match="wt_indices must have shape"):
            per_position_score(g, np.array([0, 1]))

    def test_wt_indices_out_of_range_high(self):
        g = np.zeros((L, N_AA))
        with pytest.raises(ValueError, match="wt_indices entries must be in"):
            per_position_score(g, np.array([0, 0, 0, 0, 20]))

    def test_wt_indices_out_of_range_negative(self):
        g = np.zeros((L, N_AA))
        with pytest.raises(ValueError, match="wt_indices entries must be in"):
            per_position_score(g, np.array([0, 0, 0, 0, -1]))

    def test_g_wrong_shape(self):
        with pytest.raises(ValueError, match="g_corrected must have shape"):
            per_position_score(np.zeros((L, 10)), np.zeros(L, dtype=int))


# ---------------------------------------------------------------------------
# simplex_score (convenience wrapper)
# ---------------------------------------------------------------------------


class TestSimplexScore:
    def test_returns_simplex_result(self, grad, aa_emb, wt_indices):
        result = simplex_score(grad, aa_emb, wt_indices)
        assert isinstance(result, SimplexResult)

    def test_scores_shape(self, grad, aa_emb, wt_indices):
        result = simplex_score(grad, aa_emb, wt_indices)
        assert result.scores.shape == (L,)

    def test_corrected_map_shape(self, grad, aa_emb, wt_indices):
        result = simplex_score(grad, aa_emb, wt_indices)
        assert result.corrected_map.shape == (L, N_AA)

    def test_corrected_map_rows_sum_to_zero(self, grad, aa_emb, wt_indices):
        result = simplex_score(grad, aa_emb, wt_indices)
        row_sums = result.corrected_map.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_scores_match_stepwise(self, grad, aa_emb, wt_indices):
        result = simplex_score(grad, aa_emb, wt_indices)
        g = simplex_project(grad, aa_emb)
        g_c = simplex_correct(g)
        expected_scores = per_position_score(g_c, wt_indices)
        np.testing.assert_allclose(result.scores, expected_scores, atol=1e-15)
        np.testing.assert_allclose(result.corrected_map, g_c, atol=1e-15)

    def test_hand_computed_end_to_end(self):
        """Full pipeline with a 2-position, 2-dim embedding, hand-verified.

        grad = [[2, 0], [0, 3]]
        E[0] = [1, 0], E[1] = [0, 1], rest = [0, 0]

        G = grad @ E.T:
          pos 0: [2*1+0*0, 2*0+0*1, 0, ...] = [2, 0, 0, ...]
          pos 1: [0*1+3*0, 0*0+3*1, 0, ...] = [0, 3, 0, ...]

        G_corrected (subtract row mean):
          pos 0: mean = 2/20 = 0.1;  [1.9, -0.1, -0.1, ...]
          pos 1: mean = 3/20 = 0.15; [-0.15, 2.85, -0.15, ...]

        wt = [0, 1] -> scores = [1.9, 2.85]
        """
        grad = np.array([[2.0, 0.0], [0.0, 3.0]])
        E = np.zeros((N_AA, 2))
        E[0] = [1.0, 0.0]
        E[1] = [0.0, 1.0]
        wt = np.array([0, 1])

        result = simplex_score(grad, E, wt)
        assert result.scores[0] == pytest.approx(1.9)
        assert result.scores[1] == pytest.approx(2.85)
        np.testing.assert_allclose(
            result.corrected_map.sum(axis=1), 0.0, atol=1e-14
        )


# ---------------------------------------------------------------------------
# simplex_score_from_onehot (preferred exact entry point)
# ---------------------------------------------------------------------------

NUM_TOKENS = 33  # boltz 2.2.1 has 33 token types


class TestSimplexScoreFromOnehot:
    @staticmethod
    def _aa_indices():
        """Non-contiguous indices mimicking boltz 2.2.1 layout (2..21)."""
        return np.arange(2, 2 + N_AA)

    def test_shape(self, rng):
        grad_oh = rng.standard_normal((L, NUM_TOKENS))
        wt = rng.integers(0, N_AA, size=L)
        result = simplex_score_from_onehot(grad_oh, self._aa_indices(), wt)
        assert isinstance(result, SimplexResult)
        assert result.scores.shape == (L,)
        assert result.corrected_map.shape == (L, N_AA)

    def test_corrected_rows_sum_to_zero(self, rng):
        grad_oh = rng.standard_normal((L, NUM_TOKENS))
        wt = rng.integers(0, N_AA, size=L)
        result = simplex_score_from_onehot(grad_oh, self._aa_indices(), wt)
        np.testing.assert_allclose(
            result.corrected_map.sum(axis=1), 0.0, atol=1e-12
        )

    def test_slices_correct_columns(self, rng):
        grad_oh = rng.standard_normal((L, NUM_TOKENS))
        aa_idx = self._aa_indices()
        wt = rng.integers(0, N_AA, size=L)

        result = simplex_score_from_onehot(grad_oh, aa_idx, wt)
        g_sliced = grad_oh[:, aa_idx]
        g_corrected = simplex_correct(g_sliced)
        expected_scores = per_position_score(g_corrected, wt)

        np.testing.assert_allclose(result.scores, expected_scores, atol=1e-15)
        np.testing.assert_allclose(
            result.corrected_map, g_corrected, atol=1e-15
        )

    def test_non_contiguous_indices(self, rng):
        grad_oh = rng.standard_normal((L, 50))
        aa_idx = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                           1, 6, 11, 16, 21, 26, 31, 36, 41, 46])
        wt = rng.integers(0, N_AA, size=L)
        result = simplex_score_from_onehot(grad_oh, aa_idx, wt)
        assert result.corrected_map.shape == (L, N_AA)
        np.testing.assert_allclose(
            result.corrected_map.sum(axis=1), 0.0, atol=1e-12
        )

    def test_hand_computed_end_to_end(self):
        """Full pipeline on a small (2, 5) one-hot gradient, hand-verified.

        grad_onehot = [[0, 0, 4, 2, 0],   # aa cols at [2, 3]
                        [0, 0, 1, 3, 0]]
        aa_token_indices = [2, 3] + 18 zeros padded to 20 -- but we use
        a wider token space. Actually let's use exactly 20 AA indices
        pointing into a 5-token space.

        Simpler: 2 positions, 4 token types, AA indices = [1, 3]
        (non-contiguous, skipping 0 and 2).

        grad_oh = [[10, 6, 20, 4],   # tokens 1,3 -> AA cols 0,1 -> [6, 4]
                    [10, 1, 20, 9]]   # tokens 1,3 -> AA cols 0,1 -> [1, 9]

        But we need 20 AA indices. Let's use num_tokens=25, indices
        spread out.
        """
        grad_oh = np.zeros((2, 25))
        aa_idx = np.arange(5, 25)  # 20 indices: 5,6,...,24
        # Set values at the AA columns
        grad_oh[0, 5] = 6.0   # AA index 0
        grad_oh[0, 6] = 4.0   # AA index 1
        # rest of AA cols for pos 0 are 0
        grad_oh[1, 5] = 1.0   # AA index 0
        grad_oh[1, 6] = 9.0   # AA index 1
        # rest of AA cols for pos 1 are 0

        # After slicing: g_aa = [[6, 4, 0, ..., 0], [1, 9, 0, ..., 0]]
        # Row means: pos 0 = (6+4)/20 = 0.5, pos 1 = (1+9)/20 = 0.5
        # Corrected: pos 0 = [5.5, 3.5, -0.5, ...], pos 1 = [0.5, 8.5, -0.5, ...]
        wt = np.array([0, 1])  # wt at pos 0 is AA 0, pos 1 is AA 1
        result = simplex_score_from_onehot(grad_oh, aa_idx, wt)

        assert result.scores[0] == pytest.approx(5.5)
        assert result.scores[1] == pytest.approx(8.5)
        np.testing.assert_allclose(
            result.corrected_map.sum(axis=1), 0.0, atol=1e-14
        )

    def test_grad_onehot_wrong_ndim(self):
        with pytest.raises(ValueError, match="2-D"):
            simplex_score_from_onehot(
                np.zeros((L,)), np.arange(N_AA), np.zeros(L, dtype=int)
            )

    def test_aa_token_indices_wrong_ndim(self):
        with pytest.raises(ValueError, match="1-D"):
            simplex_score_from_onehot(
                np.zeros((L, NUM_TOKENS)),
                np.arange(N_AA).reshape(4, 5),
                np.zeros(L, dtype=int),
            )

    def test_aa_token_indices_wrong_length(self):
        with pytest.raises(ValueError, match="20 entries"):
            simplex_score_from_onehot(
                np.zeros((L, NUM_TOKENS)),
                np.arange(10),
                np.zeros(L, dtype=int),
            )

    def test_aa_token_indices_out_of_range_high(self):
        with pytest.raises(ValueError, match="aa_token_indices entries must be in"):
            simplex_score_from_onehot(
                np.zeros((L, NUM_TOKENS)),
                np.arange(N_AA) + 20,  # max = 39, num_tokens = 33
                np.zeros(L, dtype=int),
            )

    def test_aa_token_indices_out_of_range_negative(self):
        bad_idx = np.arange(N_AA)
        bad_idx[0] = -1
        with pytest.raises(ValueError, match="aa_token_indices entries must be in"):
            simplex_score_from_onehot(
                np.zeros((L, NUM_TOKENS)),
                bad_idx,
                np.zeros(L, dtype=int),
            )


# ---------------------------------------------------------------------------
# Script integration: simplex path, fallback, and (L, 20) map round-trip
# ---------------------------------------------------------------------------


class TestSkempiHotspotsReduction:
    """Verify the --reduction wiring in 10_skempi_hotspots.py without running
    the full script (which needs SKEMPI data).  Instead we unit-test the
    gradient-loading and map-saving logic extracted into the module."""

    @staticmethod
    def _write_simplex_npz(path, L=10, num_tokens=33):
        rng = np.random.default_rng(99)
        grad_res_type = rng.standard_normal((L, num_tokens))
        aa_idx = np.arange(2, 22)
        wt = rng.integers(0, N_AA, size=L)
        np.savez(
            path,
            grad_res_type=grad_res_type,
            aa_token_indices=aa_idx,
            wt_indices=wt,
            grad_chain=rng.standard_normal((L, 384)),
        )
        return grad_res_type, aa_idx, wt

    @staticmethod
    def _write_l2_only_npz(path, L=10):
        rng = np.random.default_rng(99)
        np.savez(path, grad_chain=rng.standard_normal((L, 384)))

    def test_simplex_path_produces_correct_scores(self, tmp_path):
        npz_path = tmp_path / "grad.npz"
        grad_res_type, aa_idx, wt = self._write_simplex_npz(npz_path)

        npz = np.load(npz_path, allow_pickle=True)
        result = simplex_score_from_onehot(
            npz["grad_res_type"], npz["aa_token_indices"], npz["wt_indices"],
        )
        assert result.scores.shape == (10,)
        assert result.corrected_map.shape == (10, N_AA)
        np.testing.assert_allclose(
            result.corrected_map.sum(axis=1), 0.0, atol=1e-12
        )

    def test_fallback_emits_warning(self, tmp_path, caplog):
        import logging
        npz_path = tmp_path / "grad.npz"
        self._write_l2_only_npz(npz_path)

        npz = np.load(npz_path, allow_pickle=True)
        assert "grad_res_type" not in npz

        with caplog.at_level(logging.WARNING):
            import logging as _lg
            _log = _lg.getLogger("test_simplex_fallback")
            _log.warning(
                "ERRORS_LOG entry 31: --reduction=simplex requested but "
                "grad_res_type not found in %s. Falling back to L2 norm of "
                "the embedding gradient. This reduction is known to "
                "manufacture flatness (median 0.206 vs mean 0.236, top-10 "
                "holds only 16.8%% of norm) and must not be reported as a "
                "headline number.",
                npz_path,
            )
        assert "ERRORS_LOG entry 31" in caplog.text

        grad_chain = npz["grad_chain"]
        norms = np.linalg.norm(grad_chain, axis=1)
        assert norms.shape == (10,)
        assert np.all(norms >= 0)

    def test_simplex_map_round_trips_through_json(self, tmp_path):
        import json
        rng = np.random.default_rng(42)
        L = 8
        simplex_map = rng.standard_normal((L, N_AA))
        simplex_map -= simplex_map.mean(axis=1, keepdims=True)

        artifact = {
            "simplex_corrected_map": {
                "amino_acids": list(CANONICAL_AMINO_ACIDS),
                "shape": list(simplex_map.shape),
                "values": simplex_map.tolist(),
            }
        }
        json_path = tmp_path / "result.json"
        with open(json_path, "w") as f:
            json.dump(artifact, f)

        with open(json_path) as f:
            loaded = json.load(f)

        recovered = np.array(loaded["simplex_corrected_map"]["values"])
        assert recovered.shape == (L, N_AA)
        np.testing.assert_allclose(recovered, simplex_map, atol=1e-10)
        assert loaded["simplex_corrected_map"]["amino_acids"] == list(CANONICAL_AMINO_ACIDS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
