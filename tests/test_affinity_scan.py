"""Tests for Stage 20 -- brute-force affinity scan.

CPU only, no boltz, no GPU. Tests argument parsing, --dry-run, the res_type
mutation helper (round-trip, wild-type assertion), vocab-index lookup,
crop-exclusion reporting, resumability, and the CSV schema.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_SCRIPT = Path(__file__).parent.parent / "scripts" / "20_affinity_scan.py"
_spec = importlib.util.spec_from_file_location("affinity_scan", str(_SCRIPT))
affinity_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(affinity_scan)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402
from igv.boltz_score import canonical_aa_token_indices  # noqa: E402

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_res_type(n_tokens: int = 10, n_vocab: int = 33):
    """Build a fake one-hot res_type tensor with known AA at each position."""
    aa_idx = canonical_aa_token_indices()
    rt = torch.zeros(1, n_tokens, n_vocab)
    for i in range(n_tokens):
        col = int(aa_idx[i % 20])
        rt[0, i, col] = 1.0
    return rt, aa_idx


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        args = affinity_scan.build_parser().parse_args([])
        assert args.score == "affinity_pred_value"
        assert args.tki == "imatinib"
        assert args.dry_run is False
        assert args.resnum_offset == 0
        assert args.pdb is None
        assert args.protein_sequence is None

    def test_custom_args(self):
        args = affinity_scan.build_parser().parse_args([
            "--protein-sequence", "MVLSPAD",
            "--ligand-smiles", "CC(=O)O",
            "--pdb", "/tmp/test.pdb",
            "--resnum-offset", "235",
            "--tki", "all",
            "--score", "affinity_logits_binary",
            "--dry-run",
        ])
        assert args.protein_sequence == "MVLSPAD"
        assert args.ligand_smiles == "CC(=O)O"
        assert args.pdb == "/tmp/test.pdb"
        assert args.resnum_offset == 235
        assert args.tki == "all"
        assert args.score == "affinity_logits_binary"
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# 2. TKI resolution
# ---------------------------------------------------------------------------


class TestResolveTkis:
    def test_single_tki(self):
        assert affinity_scan._resolve_tkis("imatinib") == ["imatinib"]

    def test_all_tkis(self):
        from igv.abl import ABL_TKIS
        result = affinity_scan._resolve_tkis("all")
        assert set(result) == set(ABL_TKIS)

    def test_unknown_tki_raises(self):
        with pytest.raises(SystemExit, match="Unknown TKI"):
            affinity_scan._resolve_tkis("aspirin")


# ---------------------------------------------------------------------------
# 3. Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def _make_hauser_csv(self, tmp_path):
        rows = [
            {
                "TKI": "imatinib",
                "mutation": "T315I",
                "IC50": 10000.0,
                "DDG_exp": 2.32,
                "DDG_prime": 0.0,
                "DDG_fep1": 0.0, "dDDG_fep1": 0.0,
                "DDG_fep2": 0.0, "dDDG_fep2": 0.0,
                "DDG_fep3": 0.0, "dDDG_fep3": 0.0,
            },
            {
                "TKI": "imatinib",
                "mutation": "Y253F",
                "IC50": 5000.0,
                "DDG_exp": 1.50,
                "DDG_prime": 0.0,
                "DDG_fep1": 0.0, "dDDG_fep1": 0.0,
                "DDG_fep2": 0.0, "dDDG_fep2": 0.0,
                "DDG_fep3": 0.0, "dDDG_fep3": 0.0,
            },
        ]
        csv_path = tmp_path / "hauser.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_exits_zero(self, tmp_path):
        csv_path = self._make_hauser_csv(tmp_path)
        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--dry-run",
                "--hauser-csv", str(csv_path),
            ],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    def test_prints_plan(self, tmp_path):
        csv_path = self._make_hauser_csv(tmp_path)
        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--dry-run",
                "--hauser-csv", str(csv_path),
            ],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        out = result.stdout
        assert "Affinity scan" in out
        assert "imatinib" in out
        assert "T315I" in out
        assert "Forward passes" in out


# ---------------------------------------------------------------------------
# 4. mutate_res_type -- round trip and assertion
# ---------------------------------------------------------------------------


class TestMutateResType:
    def test_round_trip(self):
        rt, aa_idx = _make_res_type()
        wt_aa = AA_LIST[0]
        mut_aa = AA_LIST[1]
        mutated = affinity_scan.mutate_res_type(rt, 0, wt_aa, mut_aa, aa_idx)

        assert mutated.shape == rt.shape
        mut_col = int(aa_idx[AA_TO_IDX[mut_aa]])
        assert mutated[0, 0, mut_col].item() == 1.0
        assert mutated[0, 0].sum().item() == pytest.approx(1.0)

        # Other positions unchanged
        assert torch.equal(mutated[0, 1:], rt[0, 1:])

    def test_wt_assertion_fires_on_mismatch(self):
        rt, aa_idx = _make_res_type()
        wrong_wt = AA_LIST[5]
        actual_wt = AA_LIST[0]
        assert wrong_wt != actual_wt
        with pytest.raises(AssertionError, match="Wild-type assertion failed"):
            affinity_scan.mutate_res_type(rt, 0, wrong_wt, AA_LIST[1], aa_idx)

    def test_mutation_zeroes_wt_column(self):
        rt, aa_idx = _make_res_type()
        wt_aa = AA_LIST[0]
        mut_aa = AA_LIST[3]
        mutated = affinity_scan.mutate_res_type(rt, 0, wt_aa, mut_aa, aa_idx)
        wt_col = int(aa_idx[AA_TO_IDX[wt_aa]])
        assert mutated[0, 0, wt_col].item() == 0.0

    def test_does_not_modify_original(self):
        rt, aa_idx = _make_res_type()
        original = rt.clone()
        affinity_scan.mutate_res_type(rt, 0, AA_LIST[0], AA_LIST[1], aa_idx)
        assert torch.equal(rt, original)


# ---------------------------------------------------------------------------
# 5. Vocab-index lookup
# ---------------------------------------------------------------------------


class TestVocabIndex:
    def test_canonical_aa_indices_are_20(self):
        aa_idx = canonical_aa_token_indices()
        assert len(aa_idx) == 20

    def test_all_indices_unique(self):
        aa_idx = canonical_aa_token_indices()
        assert len(set(aa_idx.tolist())) == 20

    def test_all_indices_in_range(self):
        aa_idx = canonical_aa_token_indices()
        for idx in aa_idx:
            assert 0 <= idx < 33


# ---------------------------------------------------------------------------
# 6. Crop-exclusion reporting
# ---------------------------------------------------------------------------


class TestCropExclusion:
    def test_cropped_position_skipped(self, tmp_path):
        from igv.boltz_score import check_crop_coverage
        token_map = {("0", 315): 5, ("0", 253): 10}
        coverage = check_crop_coverage(token_map, chain_id="0", sites=[315, 253, 459])
        assert 459 in coverage["cropped"]
        assert 315 in coverage["surviving"]
        assert 253 in coverage["surviving"]
        assert coverage["n_cropped"] == 1


# ---------------------------------------------------------------------------
# 7. Resumability
# ---------------------------------------------------------------------------


class TestResumability:
    def test_load_done_keys_empty(self, tmp_path):
        out = tmp_path / "scan.csv"
        assert affinity_scan._load_done_keys(out) == set()

    def test_load_done_keys_from_csv(self, tmp_path):
        out = tmp_path / "scan.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=affinity_scan.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "tki": "imatinib",
                "mutation": "T315I",
                "resnum": 315,
                "wt_aa": "T",
                "mut_aa": "I",
                "token_index": 5,
                "wt_score": 2.5,
                "mut_score": 3.0,
                "delta_score": 0.5,
                "crop_survived": True,
                "DDG_exp": 2.32,
            })
        done = affinity_scan._load_done_keys(out)
        assert "imatinib|T315I" in done
        assert len(done) == 1

    def test_resume_skips_existing(self, tmp_path):
        out = tmp_path / "scan.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=affinity_scan.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "tki": "imatinib",
                "mutation": "T315I",
                "resnum": 315,
                "wt_aa": "T",
                "mut_aa": "I",
                "token_index": 5,
                "wt_score": 2.5,
                "mut_score": 3.0,
                "delta_score": 0.5,
                "crop_survived": True,
                "DDG_exp": 2.32,
            })
        done = affinity_scan._load_done_keys(out)
        assert "imatinib|T315I" in done
        assert "imatinib|Y253F" not in done


# ---------------------------------------------------------------------------
# 8. CSV schema
# ---------------------------------------------------------------------------


class TestCSVSchema:
    def test_columns_list(self):
        expected = [
            "tki", "mutation", "resnum", "wt_aa", "mut_aa",
            "token_index", "wt_score", "mut_score", "delta_score",
            "crop_survived", "DDG_exp",
        ]
        assert affinity_scan.CSV_COLUMNS == expected

    def test_csv_roundtrip(self, tmp_path):
        out = tmp_path / "scan.csv"
        row = {
            "tki": "imatinib",
            "mutation": "T315I",
            "resnum": 315,
            "wt_aa": "T",
            "mut_aa": "I",
            "token_index": 5,
            "wt_score": 2.62,
            "mut_score": 3.10,
            "delta_score": 0.48,
            "crop_survived": True,
            "DDG_exp": 2.32,
        }
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=affinity_scan.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

        df = pd.read_csv(out)
        assert list(df.columns) == affinity_scan.CSV_COLUMNS
        assert len(df) == 1
        assert df.iloc[0]["tki"] == "imatinib"
        assert df.iloc[0]["mutation"] == "T315I"
        assert df.iloc[0]["resnum"] == 315
        assert float(df.iloc[0]["delta_score"]) == pytest.approx(0.48)


# ---------------------------------------------------------------------------
# 9. AA_LIST consistency
# ---------------------------------------------------------------------------


class TestAAListConsistency:
    def test_matches_canonical(self):
        assert affinity_scan.AA_LIST == list(CANONICAL_AMINO_ACIDS)

    def test_length_20(self):
        assert len(affinity_scan.AA_LIST) == 20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
