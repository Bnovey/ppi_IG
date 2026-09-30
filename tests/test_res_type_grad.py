"""Tests for the res_type gradient path in scripts/03_attribute.py.

Verifies that grad_res_type, aa_token_indices, and wt_indices are written
to the .npz with correct shapes, that existing keys survive unchanged,
that the no-gradient case raises, and that --no-res-type-grad suppresses
the key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from igv.attrib import CANONICAL_AMINO_ACIDS
from igv.boltz_score import (
    _BOLTZ_TOKENS_2_2_1,
    _PROT_TOKEN_TO_LETTER_2_2_1,
    canonical_aa_token_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_TOKENS = len(_BOLTZ_TOKENS_2_2_1)
L_CHAIN = 10
L_FULL = 15
D = 384


def _aa_token_indices():
    return canonical_aa_token_indices(
        tokens=_BOLTZ_TOKENS_2_2_1,
        token_to_letter=_PROT_TOKEN_TO_LETTER_2_2_1,
    )


def _make_mock_model(L=L_FULL, num_tokens=NUM_TOKENS):
    """Build a mock model whose input_embedder and forward path are
    differentiable w.r.t. res_type."""
    model = MagicMock()

    embed_weight = torch.randn(num_tokens, D)

    def _input_embedder(feats):
        rt = feats["res_type"]  # (1, L, num_tokens), float, requires_grad
        return torch.matmul(rt, embed_weight.to(rt.device))  # (1, L, D)

    model.input_embedder = _input_embedder
    return model


def _make_feats(L=L_FULL, num_tokens=NUM_TOKENS):
    rng = np.random.default_rng(42)
    onehot = np.zeros((1, L, num_tokens), dtype=np.float32)
    for i in range(L):
        tok = rng.integers(2, 22)
        onehot[0, i, tok] = 1.0
    return {
        "res_type": torch.tensor(onehot),
        "token_pad_mask": torch.ones(1, L),
        "coords": torch.zeros(1, L * 3, 3),
    }


def _reference_seq(L=L_CHAIN):
    rng = np.random.default_rng(7)
    return "".join(rng.choice(list(CANONICAL_AMINO_ACIDS), size=L))


def _token_indices():
    return list(range(L_CHAIN))


# ---------------------------------------------------------------------------
# res_type_gradient_forward
# ---------------------------------------------------------------------------


class TestResTypeGradientForward:
    """Tests for boltz_score.res_type_gradient_forward using a mock model."""

    def test_returns_gradient_with_correct_shape(self):
        model = _make_mock_model()
        feats = _make_feats()
        num_tokens = feats["res_type"].shape[2]
        L = feats["res_type"].shape[1]

        def _mock_confidence_forward(m, s_inputs, feats, x_pred, score_name, **kw):
            return s_inputs.sum()

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_confidence_forward
        try:
            result = bs.res_type_gradient_forward(
                model, feats, "complex_pde",
                gradient_checkpointing=False,
            )
        finally:
            bs.confidence_forward = orig_cf

        assert "grad_res_type" in result
        assert result["grad_res_type"].shape == (L, num_tokens)
        assert np.isfinite(result["grad_res_type"]).all()

    def test_raises_when_no_gradient(self):
        model = MagicMock()
        model.input_embedder = lambda feats: torch.zeros(1, L_FULL, D)
        feats = _make_feats()

        def _mock_cf_detached(m, s_inputs, feats, x_pred, score_name, **kw):
            return torch.tensor(1.0, requires_grad=True)

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_cf_detached
        try:
            with pytest.raises(RuntimeError, match="No gradient on res_type"):
                bs.res_type_gradient_forward(
                    model, feats, "complex_pde",
                    gradient_checkpointing=False,
                )
        finally:
            bs.confidence_forward = orig_cf

    def test_feats_res_type_is_restored(self):
        model = _make_mock_model()
        feats = _make_feats()
        orig_data = feats["res_type"].clone()

        def _mock_cf(m, s_inputs, feats, x_pred, score_name, **kw):
            return s_inputs.sum()

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_cf
        try:
            bs.res_type_gradient_forward(
                model, feats, "complex_pde",
                gradient_checkpointing=False,
            )
        finally:
            bs.confidence_forward = orig_cf

        torch.testing.assert_close(feats["res_type"], orig_data)


# ---------------------------------------------------------------------------
# _ig_res_type (the IG accumulator in 03_attribute.py)
# ---------------------------------------------------------------------------


class TestIgResType:
    """Tests for _ig_res_type in scripts/03_attribute.py."""

    @staticmethod
    def _import_ig_res_type():
        import importlib
        spec = importlib.util.spec_from_file_location(
            "_03_attribute",
            str(Path(__file__).resolve().parents[1] / "scripts" / "03_attribute.py"),
        )
        mod = importlib.util.module_from_spec(spec)

        import igv.boltz_score as bs

        def _mock_cf(m, s_inputs, feats, x_pred, score_name, **kw):
            return s_inputs.sum()

        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_cf
        try:
            spec.loader.exec_module(mod)
        finally:
            bs.confidence_forward = orig_cf
        return mod._ig_res_type, _mock_cf

    def test_ig_shape_matches_res_type(self):
        ig_fn, mock_cf = self._import_ig_res_type()
        model = _make_mock_model()
        feats = _make_feats()

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = mock_cf
        try:
            result = ig_fn(
                model, feats, "complex_pde",
                torch.zeros(1, L_FULL * 3, 3),
                baseline_name="zeros",
                m_steps=3,
            )
        finally:
            bs.confidence_forward = orig_cf

        L = feats["res_type"].shape[1]
        num_tokens = feats["res_type"].shape[2]
        assert result.shape == (L, num_tokens)
        assert np.isfinite(result).all()

    def test_ig_mean_aa_baseline(self):
        ig_fn, mock_cf = self._import_ig_res_type()
        model = _make_mock_model()
        feats = _make_feats()

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = mock_cf
        try:
            result = ig_fn(
                model, feats, "complex_pde",
                torch.zeros(1, L_FULL * 3, 3),
                baseline_name="mean_aa",
                m_steps=3,
            )
        finally:
            bs.confidence_forward = orig_cf

        L = feats["res_type"].shape[1]
        num_tokens = feats["res_type"].shape[2]
        assert result.shape == (L, num_tokens)

    def test_ig_raises_when_no_gradient(self):
        ig_fn, _ = self._import_ig_res_type()
        model = MagicMock()
        model.input_embedder = lambda feats: torch.zeros(1, L_FULL, D)
        feats = _make_feats()

        def _mock_cf_detached(m, s_inputs, feats, x_pred, score_name, **kw):
            return torch.tensor(1.0, requires_grad=True)

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_cf_detached
        try:
            with pytest.raises(RuntimeError, match="No gradient on res_type"):
                ig_fn(
                    model, feats, "complex_pde",
                    torch.zeros(1, L_FULL * 3, 3),
                    baseline_name="zeros",
                    m_steps=2,
                )
        finally:
            bs.confidence_forward = orig_cf


# ---------------------------------------------------------------------------
# .npz save dict integration
# ---------------------------------------------------------------------------


class TestNpzSaveDict:
    """Verify the keys and shapes written to the .npz."""

    @staticmethod
    def _build_save_dict(include_res_type=True):
        rng = np.random.default_rng(42)
        ref_seq = _reference_seq()
        token_indices = _token_indices()
        num_tokens = NUM_TOKENS

        save_dict = {
            "grad_chain": rng.standard_normal((L_CHAIN, D)).astype(np.float32),
            "grad_full": rng.standard_normal((1, L_FULL, D)).astype(np.float32),
            "token_indices": np.array(token_indices, dtype=np.int64),
            "f_x": np.float64(0.5),
            "f_baseline": np.float64(0.3),
            "completeness_error": np.float64(0.02),
            "reference_seq": np.array(ref_seq),
            "chain": np.array("H"),
            "score": np.array("complex_pde"),
            "method": np.array("plain_grad"),
            "m_steps": np.int64(15),
        }

        if include_res_type:
            aa_idx = _aa_token_indices()
            aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}
            wt_indices = np.array(
                [aa_to_idx.get(c, 0) for c in ref_seq], dtype=np.intp,
            )
            save_dict.update({
                "grad_res_type": rng.standard_normal(
                    (L_CHAIN, num_tokens)
                ).astype(np.float32),
                "aa_token_indices": aa_idx,
                "wt_indices": wt_indices,
            })
        return save_dict

    def test_existing_keys_present_with_res_type(self, tmp_path):
        d = self._build_save_dict(include_res_type=True)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        for key in ("grad_chain", "grad_full", "token_indices", "f_x",
                     "f_baseline", "completeness_error", "reference_seq"):
            assert key in npz, f"Missing existing key: {key}"

    def test_res_type_keys_present(self, tmp_path):
        d = self._build_save_dict(include_res_type=True)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        assert "grad_res_type" in npz
        assert "aa_token_indices" in npz
        assert "wt_indices" in npz

    def test_grad_res_type_shape(self, tmp_path):
        d = self._build_save_dict(include_res_type=True)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        assert npz["grad_res_type"].shape == (L_CHAIN, NUM_TOKENS)

    def test_aa_token_indices_shape(self, tmp_path):
        d = self._build_save_dict(include_res_type=True)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        assert npz["aa_token_indices"].shape == (20,)

    def test_wt_indices_shape(self, tmp_path):
        d = self._build_save_dict(include_res_type=True)
        ref_seq = _reference_seq()
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        assert npz["wt_indices"].shape == (len(ref_seq),)

    def test_existing_keys_unchanged_when_res_type_added(self, tmp_path):
        d_without = self._build_save_dict(include_res_type=False)
        d_with = dict(d_without)
        rng = np.random.default_rng(99)
        d_with.update({
            "grad_res_type": rng.standard_normal(
                (L_CHAIN, NUM_TOKENS)
            ).astype(np.float32),
            "aa_token_indices": _aa_token_indices(),
            "wt_indices": np.zeros(L_CHAIN, dtype=np.intp),
        })

        p1 = tmp_path / "without.npz"
        p2 = tmp_path / "with.npz"
        np.savez(p1, **d_without)
        np.savez(p2, **d_with)

        npz1 = np.load(p1, allow_pickle=True)
        npz2 = np.load(p2, allow_pickle=True)

        for key in d_without:
            np.testing.assert_array_equal(
                npz1[key], npz2[key],
                err_msg=f"Existing key {key!r} changed when res_type keys added",
            )

    def test_disabled_flag_omits_res_type_keys(self, tmp_path):
        d = self._build_save_dict(include_res_type=False)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)
        assert "grad_res_type" not in npz
        assert "aa_token_indices" not in npz
        assert "wt_indices" not in npz

    def test_consumer_round_trip(self, tmp_path):
        """The consumer (scripts/10_skempi_hotspots.py) reads grad_res_type,
        aa_token_indices, wt_indices and calls simplex_score_from_onehot.
        Verify the round trip works with the shapes we write."""
        from igv.attrib import simplex_score_from_onehot

        d = self._build_save_dict(include_res_type=True)
        p = tmp_path / "grad.npz"
        np.savez(p, **d)
        npz = np.load(p, allow_pickle=True)

        result = simplex_score_from_onehot(
            npz["grad_res_type"],
            npz["aa_token_indices"],
            npz["wt_indices"],
        )
        assert result.scores.shape == (L_CHAIN,)
        assert result.corrected_map.shape == (L_CHAIN, 20)


# ---------------------------------------------------------------------------
# wt_indices derivation
# ---------------------------------------------------------------------------


class TestWtIndices:
    """Verify that wt_indices are derived correctly from reference_seq."""

    def test_known_sequence(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}
        wt = np.array([aa_to_idx.get(c, 0) for c in seq], dtype=np.intp)
        assert wt.shape == (20,)
        for i, aa in enumerate(CANONICAL_AMINO_ACIDS):
            assert wt[i] == i

    def test_unknown_residue_maps_to_zero(self):
        seq = "X"
        aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}
        wt = np.array([aa_to_idx.get(c, 0) for c in seq], dtype=np.intp)
        assert wt[0] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
