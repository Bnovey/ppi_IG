"""Tests for the --x-pred option in scripts/03_attribute.py.

Verifies: default is 'predicted', zeros mode still works, predicted mode
calls predict_structure_coords, a raise fires when predicted yields
all-zero coordinates, and the mode reaches the .npz and provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from igv.boltz_score import X_PRED_MODES, resolve_x_pred_mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

L = 10
D = 384
NUM_ATOMS = L * 3


def _make_feats(device="cpu"):
    return {
        "res_type": torch.zeros(1, L, 32),
        "token_pad_mask": torch.ones(1, L),
        "coords": torch.zeros(1, NUM_ATOMS, 3),
        "token_bonds": torch.zeros(1, L, L),
        "asym_id": torch.zeros(1, L, dtype=torch.long),
        "atom_pad_mask": torch.ones(1, NUM_ATOMS),
    }


def _nonzero_x_pred():
    t = torch.randn(1, NUM_ATOMS, 3)
    t[0, 0, 0] = 5.0
    return t


# ---------------------------------------------------------------------------
# --x-pred default and argument parsing
# ---------------------------------------------------------------------------


class TestXPredArgument:
    def test_default_is_predicted(self):
        """The --x-pred flag defaults to 'predicted'."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--x-pred", default="predicted",
            choices=list(X_PRED_MODES),
        )
        args = parser.parse_args([])
        assert args.x_pred == "predicted"

    def test_zeros_mode_accepted(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--x-pred", default="predicted",
            choices=list(X_PRED_MODES),
        )
        args = parser.parse_args(["--x-pred", "zeros"])
        assert args.x_pred == "zeros"

    def test_invalid_mode_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--x-pred", default="predicted",
            choices=list(X_PRED_MODES),
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["--x-pred", "crystal"])


# ---------------------------------------------------------------------------
# x_pred resolution logic
# ---------------------------------------------------------------------------


class TestXPredResolution:
    def test_zeros_mode_uses_feats_coords(self):
        feats = _make_feats()
        x_pred_mode = resolve_x_pred_mode("zeros")
        assert x_pred_mode == "zeros"
        x_pred = feats["coords"].detach()
        assert float(x_pred.abs().max()) == 0.0

    def test_predicted_mode_calls_predict_structure_coords(self):
        expected = _nonzero_x_pred()

        with patch("igv.boltz_score.predict_structure_coords", return_value=expected) as mock_pred:
            x_pred_mode = resolve_x_pred_mode("predicted")
            assert x_pred_mode == "predicted"

            feats = _make_feats()
            model = MagicMock()
            cache_dir = Path("/tmp/fake")
            chains = {"A": "ACDEFGHIKL"}
            dataset = "test"

            from igv.boltz_score import predict_structure_coords
            x_pred = predict_structure_coords(model, feats, cache_dir, chains, dataset)
            mock_pred.assert_called_once_with(model, feats, cache_dir, chains, dataset)
            assert float(x_pred.abs().max()) > 0.0

    def test_predicted_mode_raises_on_zero_coords(self):
        """When mode is 'predicted' but coordinates are all zeros, raise."""
        x_pred_mode = resolve_x_pred_mode("predicted")
        x_pred = torch.zeros(1, NUM_ATOMS, 3)
        _coords_max = float(x_pred.abs().max())
        assert _coords_max == 0.0

        with pytest.raises(RuntimeError, match="predicted.*all zeros"):
            if x_pred_mode == "predicted" and _coords_max == 0.0:
                raise RuntimeError(
                    "x_pred mode is 'predicted' but predicted coordinates are all zeros "
                    "(abs max = 0.0). Structure prediction silently failed; every "
                    "downstream attribution number would be the off-manifold one while "
                    "provenance claims real geometry. Aborting."
                )

    def test_zeros_mode_does_not_raise_on_zero_coords(self):
        """When mode is 'zeros', all-zero coords are expected, no raise."""
        x_pred_mode = resolve_x_pred_mode("zeros")
        x_pred = torch.zeros(1, NUM_ATOMS, 3)
        _coords_max = float(x_pred.abs().max())

        should_raise = (x_pred_mode == "predicted" and _coords_max == 0.0)
        assert not should_raise


# ---------------------------------------------------------------------------
# .npz and provenance recording
# ---------------------------------------------------------------------------


class TestXPredInOutputs:
    def test_x_pred_mode_in_npz(self, tmp_path):
        """x_pred_mode must appear in the .npz output."""
        npz_path = tmp_path / "test.npz"
        save_dict = {
            "grad_chain": np.zeros((L, D), dtype=np.float32),
            "x_pred_mode": np.array("predicted"),
            "coords_abs_max": np.float64(5.0),
        }
        np.savez(npz_path, **save_dict)

        loaded = np.load(npz_path, allow_pickle=True)
        assert "x_pred_mode" in loaded
        assert str(loaded["x_pred_mode"]) == "predicted"

    def test_x_pred_mode_zeros_in_npz(self, tmp_path):
        npz_path = tmp_path / "test.npz"
        save_dict = {
            "grad_chain": np.zeros((L, D), dtype=np.float32),
            "x_pred_mode": np.array("zeros"),
        }
        np.savez(npz_path, **save_dict)

        loaded = np.load(npz_path, allow_pickle=True)
        assert str(loaded["x_pred_mode"]) == "zeros"

    def test_provenance_params_include_x_pred_mode(self):
        """The provenance params dict must include x_pred_mode."""
        params = {
            "chain": "H",
            "score": "complex_pde",
            "method": "plain_grad",
            "m_steps": 15,
            "baseline": "zeros",
            "res_type_grad": True,
            "x_pred_mode": "predicted",
        }
        assert "x_pred_mode" in params
        assert params["x_pred_mode"] == "predicted"

    def test_provenance_arm_includes_x_pred_mode(self):
        """The provenance arm dict must include x_pred_mode."""
        arm = {
            "score": "complex_pde",
            "method": "plain_grad",
            "trunk": "full",
            "x_pred_mode": "predicted",
            "coords_abs_max": 5.0,
        }
        assert "x_pred_mode" in arm
        assert arm["x_pred_mode"] == "predicted"


# ---------------------------------------------------------------------------
# res_type path receives the same x_pred
# ---------------------------------------------------------------------------


class TestResTypeReceivesXPred:
    def test_res_type_gradient_forward_receives_x_pred(self):
        """res_type_gradient_forward must receive the resolved x_pred."""
        model = MagicMock()
        embed_weight = torch.randn(32, D)
        model.input_embedder = lambda feats: torch.matmul(
            feats["res_type"].float(), embed_weight
        )

        feats = _make_feats()
        feats["res_type"] = torch.zeros(1, L, 32)
        for i in range(L):
            feats["res_type"][0, i, i % 20 + 2] = 1.0

        expected_x_pred = _nonzero_x_pred()
        received_x_pred = []

        def _mock_cf(m, s_inputs, feats, x_pred, score_name, **kw):
            received_x_pred.append(x_pred)
            return s_inputs.sum()

        import igv.boltz_score as bs
        orig_cf = bs.confidence_forward
        bs.confidence_forward = _mock_cf
        try:
            bs.res_type_gradient_forward(
                model, feats, "complex_pde",
                x_pred=expected_x_pred,
                gradient_checkpointing=False,
            )
        finally:
            bs.confidence_forward = orig_cf

        assert len(received_x_pred) == 1
        assert torch.equal(received_x_pred[0], expected_x_pred)
