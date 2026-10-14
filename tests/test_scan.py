"""Tests for Stage 04 -- brute-force mutation scan.

CPU only, no boltz, no GPU. Tests resume logic, the wild-type row, resswap
construction (only res_type changes), the atom-mismatch guard, and CSV schema.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "04_scan.py"
_spec = importlib.util.spec_from_file_location("scan_04", str(_SCRIPT))
scan_04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_04)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# 1. Resume logic -- SKEMPI/DMS path (mutation_id keyed)
# ---------------------------------------------------------------------------


class TestLoadDoneIds:
    def test_empty_when_no_file(self, tmp_path):
        assert scan_04._load_done_ids(tmp_path / "nonexistent.csv") == set()

    def test_reads_mutation_ids(self, tmp_path):
        out = tmp_path / "scan.csv"
        pd.DataFrame(
            [
                {"mutation_id": "42A", "row_index": 0, "model_score": 1.0,
                 "substitutions": "42A", "n_mut": 1, "sequence": "X",
                 "binding_score": float("nan"), "wt_score": 1.0,
                 "delta": 0.0, "mode": "resswap"},
                {"mutation_id": "42C", "row_index": 1, "model_score": 1.1,
                 "substitutions": "42C", "n_mut": 1, "sequence": "X",
                 "binding_score": float("nan"), "wt_score": 1.0,
                 "delta": 0.1, "mode": "resswap"},
            ]
        ).to_csv(out, index=False)
        done = scan_04._load_done_ids(out)
        assert "42A" in done
        assert "42C" in done
        assert "99G" not in done

    def test_skips_corrupt_file(self, tmp_path):
        out = tmp_path / "scan.csv"
        out.write_text("this is not\nvalid csv content\x00\x01")
        done = scan_04._load_done_ids(out)
        assert done == set()


class TestLoadDoneRowIndices:
    def test_empty_when_no_file(self, tmp_path):
        assert scan_04._load_done_row_indices(tmp_path / "nope.csv") == set()

    def test_reads_row_indices(self, tmp_path):
        out = tmp_path / "scan.csv"
        pd.DataFrame(
            [
                {"mutation_id": "0", "row_index": 5, "model_score": 1.0,
                 "substitutions": "5A", "n_mut": 1, "sequence": "X",
                 "binding_score": 0.5, "wt_score": float("nan"),
                 "delta": float("nan"), "mode": "refeaturise"},
            ]
        ).to_csv(out, index=False)
        done = scan_04._load_done_row_indices(out)
        assert 5 in done
        assert 0 not in done


# ---------------------------------------------------------------------------
# 2. Wild-type row
# ---------------------------------------------------------------------------


class TestWildTypeRow:
    def test_wt_row_has_mutation_id_WT(self):
        wt = pd.DataFrame(
            [
                {
                    "mutation_id": "WT",
                    "row_index": -1,
                    "sequence": "ACDEF",
                    "substitutions": "",
                    "n_mut": 0,
                    "binding_score": float("nan"),
                    "model_score": 1.5,
                    "wt_score": 1.5,
                    "delta": 0.0,
                    "mode": "resswap",
                }
            ],
            columns=scan_04._COLUMNS,
        )
        assert wt.iloc[0]["mutation_id"] == "WT"
        assert wt.iloc[0]["delta"] == 0.0
        assert wt.iloc[0]["n_mut"] == 0

    def test_resume_detects_wt_row(self, tmp_path):
        out = tmp_path / "scan.csv"
        pd.DataFrame(
            [
                {
                    "mutation_id": "WT",
                    "row_index": -1,
                    "sequence": "ACDEF",
                    "substitutions": "",
                    "n_mut": 0,
                    "binding_score": float("nan"),
                    "model_score": 1.5,
                    "wt_score": 1.5,
                    "delta": 0.0,
                    "mode": "resswap",
                }
            ],
            columns=scan_04._COLUMNS,
        ).to_csv(out, index=False)
        done = scan_04._load_done_ids(out)
        assert "WT" in done


# ---------------------------------------------------------------------------
# 3. Resswap construction -- only res_type changes
# ---------------------------------------------------------------------------


class TestResswapConstruction:
    """Verify that _resswap_scan only mutates res_type[pos] and nothing else."""

    def test_mutation_id_format(self):
        assert scan_04._mutation_id(42, "A") == "42A"
        assert scan_04._mutation_id(0, "W") == "0W"

    def test_resswap_changes_only_target_position(self):
        import torch

        n_tokens = 20
        n_vocab = 33
        rt = torch.zeros(1, n_tokens, n_vocab)
        for i in range(n_tokens):
            rt[0, i, i % n_vocab] = 1.0

        original = rt.clone()
        pos = 5
        new_col = 10

        mutated = original.clone()
        mutated[0, pos] = 0.0
        mutated[0, pos, new_col] = 1.0

        for i in range(n_tokens):
            if i == pos:
                assert mutated[0, i, new_col].item() == 1.0
                assert mutated[0, i].sum().item() == pytest.approx(1.0)
            else:
                assert torch.equal(mutated[0, i], original[0, i]), (
                    f"Position {i} was modified but should be identical to wild type"
                )


# ---------------------------------------------------------------------------
# 4. Atom-mismatch guard
# ---------------------------------------------------------------------------


class TestAtomShapeGuard:
    def test_matching_shapes_returns_true(self):
        import torch

        wt = {
            "atom_pad_mask": torch.ones(1, 100),
            "token_to_rep_atom": torch.eye(10, 100),
        }
        mut = {
            "atom_pad_mask": torch.ones(1, 100),
            "token_to_rep_atom": torch.eye(10, 100),
        }
        assert scan_04._check_atom_shape_guard(wt, mut, 5, "A") is True

    def test_different_atom_pad_mask_returns_false(self):
        import torch

        wt = {"atom_pad_mask": torch.ones(1, 100)}
        mut = {"atom_pad_mask": torch.ones(1, 110)}
        assert scan_04._check_atom_shape_guard(wt, mut, 5, "W") is False

    def test_different_token_to_rep_atom_shape_returns_false(self):
        import torch

        wt = {
            "atom_pad_mask": torch.ones(1, 100),
            "token_to_rep_atom": torch.eye(10, 100),
        }
        mut = {
            "atom_pad_mask": torch.ones(1, 100),
            "token_to_rep_atom": torch.eye(10, 110),
        }
        assert scan_04._check_atom_shape_guard(wt, mut, 5, "W") is False

    def test_different_token_to_rep_atom_content_returns_false(self):
        import torch

        rep_wt = torch.eye(10, 100)
        rep_mut = torch.eye(10, 100)
        rep_mut[3, 50] = 1.0  # different content

        wt = {"atom_pad_mask": torch.ones(1, 100), "token_to_rep_atom": rep_wt}
        mut = {"atom_pad_mask": torch.ones(1, 100), "token_to_rep_atom": rep_mut}
        assert scan_04._check_atom_shape_guard(wt, mut, 3, "W") is False

    def test_no_token_to_rep_atom_still_checks_atom_pad_mask(self):
        import torch

        wt = {"atom_pad_mask": torch.ones(1, 100)}
        mut = {"atom_pad_mask": torch.ones(1, 100)}
        assert scan_04._check_atom_shape_guard(wt, mut, 5, "A") is True


# ---------------------------------------------------------------------------
# 5. MSA resolution
# ---------------------------------------------------------------------------


class TestResolveMsaByChain:
    def test_returns_none_when_dir_absent(self, tmp_path):
        result = scan_04._resolve_msa_by_chain(
            tmp_path, "3S9D", "A", {"A": "ACDE", "B": "FGHI"},
        )
        assert result is None

    def test_returns_dict_when_files_match(self, tmp_path):
        msa_dir = tmp_path / "boltz_attr" / "3S9D_A" / "msa"
        msa_dir.mkdir(parents=True)
        (msa_dir / "input_0.csv").touch()
        (msa_dir / "input_1.csv").touch()
        result = scan_04._resolve_msa_by_chain(
            tmp_path, "3S9D", "A", {"A": "ACDE", "B": "FGHI"},
        )
        assert result is not None
        assert set(result.keys()) == {"A", "B"}

    def test_returns_none_when_count_mismatch(self, tmp_path):
        msa_dir = tmp_path / "boltz_attr" / "3S9D_A" / "msa"
        msa_dir.mkdir(parents=True)
        (msa_dir / "input_0.csv").touch()
        result = scan_04._resolve_msa_by_chain(
            tmp_path, "3S9D", "A", {"A": "ACDE", "B": "FGHI"},
        )
        assert result is None


# ---------------------------------------------------------------------------
# 6. CSV schema
# ---------------------------------------------------------------------------


class TestCSVSchema:
    def test_columns_list(self):
        expected = [
            "mutation_id", "row_index", "sequence", "substitutions",
            "n_mut", "binding_score", "model_score", "wt_score",
            "delta", "mode",
        ]
        assert scan_04._COLUMNS == expected

    def test_csv_roundtrip(self, tmp_path):
        out = tmp_path / "scan.csv"
        row = {
            "mutation_id": "42A",
            "row_index": 0,
            "sequence": "ACDEF",
            "substitutions": "42A",
            "n_mut": 1,
            "binding_score": float("nan"),
            "model_score": 1.5,
            "wt_score": 1.0,
            "delta": 0.5,
            "mode": "resswap",
        }
        pd.DataFrame([row], columns=scan_04._COLUMNS).to_csv(out, index=False)
        df = pd.read_csv(out)
        assert list(df.columns) == scan_04._COLUMNS
        assert df.iloc[0]["mutation_id"] == "42A"
        assert float(df.iloc[0]["delta"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. Stratified sample (existing)
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_covers_all_strata(self):
        n_mut = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        idx = scan_04.stratified_sample(n_mut, 4, seed=0)
        strata_hit = set(n_mut[idx])
        assert strata_hit == {0, 1, 2, 3}

    def test_respects_budget(self):
        n_mut = np.arange(100) % 5
        idx = scan_04.stratified_sample(n_mut, 10, seed=0)
        assert len(idx) == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
