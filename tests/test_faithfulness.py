"""Tests for Stage 24 -- faithfulness analysis.

CPU only. Tests the extraction of A_grad from synthetic npz + scan CSV,
the permutation null, per-position aggregates, and the correlation metrics
on data with known correlation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "24_faithfulness.py"
_spec = importlib.util.spec_from_file_location("faithfulness_24", str(_SCRIPT))
faithfulness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(faithfulness)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    tmp_path: Path,
    n_positions: int = 5,
    n_vocab: int = 33,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Create a synthetic attribution npz and scan CSV with known linear relationship.

    A_grad = grad[pos, mut_col] - grad[pos, wt_col]
    B = scale * A_grad + noise

    So Spearman(A, B) should be close to 1.0 for low noise.
    """
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list(AA_LIST)) for _ in range(n_positions))

    aa_token_indices = np.arange(20, dtype=np.int64)
    wt_indices = np.array([AA_TO_IDX[c] for c in seq], dtype=np.intp)

    grad_res_type = rng.standard_normal((n_positions, n_vocab)).astype(np.float32)

    npz_path = tmp_path / "attr.npz"
    np.savez(
        npz_path,
        grad_res_type=grad_res_type,
        aa_token_indices=aa_token_indices,
        wt_indices=wt_indices,
        reference_seq=np.array(seq),
        method=np.array("plain_grad"),
    )

    scan_rows = []
    row_idx = 0

    scan_rows.append({
        "mutation_id": "WT",
        "row_index": -1,
        "sequence": seq,
        "substitutions": "",
        "n_mut": 0,
        "binding_score": float("nan"),
        "model_score": 1.0,
        "wt_score": 1.0,
        "delta": 0.0,
        "mode": "resswap",
    })

    for pos in range(n_positions):
        wt_aa = seq[pos]
        wt_col = int(aa_token_indices[wt_indices[pos]])
        for aa in AA_LIST:
            if aa == wt_aa:
                continue
            mut_col = int(aa_token_indices[AA_TO_IDX[aa]])
            a_val = float(grad_res_type[pos, mut_col] - grad_res_type[pos, wt_col])
            b_val = 2.0 * a_val + rng.normal(0, 0.01)
            scan_rows.append({
                "mutation_id": f"{pos}{aa}",
                "row_index": row_idx,
                "sequence": seq[:pos] + aa + seq[pos + 1:],
                "substitutions": f"{pos}{aa}",
                "n_mut": 1,
                "binding_score": float("nan"),
                "model_score": 1.0 + b_val,
                "wt_score": 1.0,
                "delta": b_val,
                "mode": "resswap",
            })
            row_idx += 1

    scan_path = tmp_path / "scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_path, index=False)

    return npz_path, scan_path


# ---------------------------------------------------------------------------
# 1. Extraction
# ---------------------------------------------------------------------------


class TestExtractAttribution:
    def test_extracts_correct_count(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=3)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        expected_n = 3 * 19  # 3 positions, 19 alternatives each
        assert len(result["A_grad"]) == expected_n
        assert len(result["B"]) == expected_n

    def test_skips_wt_row(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=2)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        assert len(result["A_grad"]) == 2 * 19

    def test_method_is_recorded(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        assert result["method"] == "plain_grad"


# ---------------------------------------------------------------------------
# 2. Correlation metrics on known data
# ---------------------------------------------------------------------------


class TestCorrelationMetrics:
    def test_perfect_linear_gives_spearman_near_one(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=5, seed=0)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        A = result["A_grad"]
        B = result["B"]
        from igv.metrics import spearman
        rho = spearman(A, B)
        assert rho > 0.99, f"Expected near-perfect correlation, got {rho}"

    def test_pearson_near_one(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=5, seed=0)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        rho = faithfulness.pearson(result["A_grad"], result["B"])
        assert rho > 0.99

    def test_scale_ratio_is_approximately_half(self, tmp_path):
        """B = 2*A + noise, so median|A| / median|B| ~ 0.5."""
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=10, seed=0)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        A = result["A_grad"]
        B = result["B"]
        ratio = float(np.median(np.abs(A))) / float(np.median(np.abs(B)))
        assert 0.3 < ratio < 0.7, f"Expected ~0.5, got {ratio}"


# ---------------------------------------------------------------------------
# 3. Permutation null
# ---------------------------------------------------------------------------


class TestPermutationNull:
    def test_correlated_data_gives_low_p_value(self, tmp_path):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=5, seed=0)
        scan_df = pd.read_csv(scan_path)
        result = faithfulness._extract_attribution(npz_path, scan_df)
        perm = faithfulness._permutation_null(result["A_grad"], result["B"])
        assert perm["p_value_two_sided"] < 0.01

    def test_shuffled_data_gives_high_p_value(self):
        rng = np.random.default_rng(123)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        perm = faithfulness._permutation_null(a, b, n_perms=1000)
        assert perm["p_value_two_sided"] > 0.01

    def test_returns_required_keys(self):
        a = np.arange(20, dtype=float)
        b = np.arange(20, dtype=float)
        perm = faithfulness._permutation_null(a, b, n_perms=100)
        assert "observed" in perm
        assert "p5" in perm
        assert "p95" in perm
        assert "p_value_two_sided" in perm


# ---------------------------------------------------------------------------
# 4. Per-position aggregates
# ---------------------------------------------------------------------------


class TestPerPositionAggregates:
    def test_correct_n_positions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.1, 2.1, 3.1, 4.1])
        pos = np.array([0, 0, 1, 1])
        agg = faithfulness._per_position_aggregates(a, b, pos)
        assert agg["n_positions"] == 2

    def test_monotone_gives_positive_spearman(self):
        rng = np.random.default_rng(42)
        n_pos = 10
        a_list = []
        b_list = []
        pos_list = []
        for p in range(n_pos):
            for _ in range(19):
                val = rng.standard_normal()
                a_list.append(val)
                b_list.append(val * 2 + rng.normal(0, 0.01))
                pos_list.append(p)
        a = np.array(a_list)
        b = np.array(b_list)
        pos = np.array(pos_list)
        agg = faithfulness._per_position_aggregates(a, b, pos)
        assert agg["sum_spearman"] > 0.5
        assert agg["max_spearman"] > 0.5


# ---------------------------------------------------------------------------
# 5. End-to-end (main writes JSON)
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    def test_writes_json(self, tmp_path, monkeypatch):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=3)
        out_path = tmp_path / "faith.json"

        import igv.dms as _dms_mod
        monkeypatch.setattr(
            _dms_mod, "resolve_pdb_complex",
            lambda *a, **kw: (_ for _ in ()).throw(KeyError("mock")),
        )

        old_argv = sys.argv
        sys.argv = [
            "24_faithfulness.py",
            "--attribution", str(npz_path),
            "--scan", str(scan_path),
            "--dataset", "test",
            "--out", str(out_path),
        ]
        try:
            faithfulness.main()
        finally:
            sys.argv = old_argv

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "spearman" in data
        assert "pearson" in data
        assert "permutation_null" in data
        assert "per_position" in data
        assert data["n"] == 3 * 19

    def test_prov_sidecar_written(self, tmp_path, monkeypatch):
        npz_path, scan_path = _make_synthetic_data(tmp_path, n_positions=3)
        out_path = tmp_path / "faith.json"

        import igv.dms as _dms_mod
        monkeypatch.setattr(
            _dms_mod, "resolve_pdb_complex",
            lambda *a, **kw: (_ for _ in ()).throw(KeyError("mock")),
        )

        old_argv = sys.argv
        sys.argv = [
            "24_faithfulness.py",
            "--attribution", str(npz_path),
            "--scan", str(scan_path),
            "--dataset", "test",
            "--out", str(out_path),
        ]
        try:
            faithfulness.main()
        finally:
            sys.argv = old_argv

        prov_path = Path(str(out_path) + ".prov.json")
        assert prov_path.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
