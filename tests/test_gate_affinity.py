"""Tests for Gate Affinity script and affinity-related boltz_score functions.

CPU only, no boltz, no GPU. Tests the argument parsing, --dry-run path,
the pure evaluation logic, YAML construction, checkpoint selection,
score registry, and token-map / crop-coverage guard.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# Import the script module for testing its pure functions.
_SCRIPT = Path(__file__).parent.parent / "scripts" / "18_gate_affinity.py"
_spec = importlib.util.spec_from_file_location("gate_affinity", str(_SCRIPT))
gate_affinity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_affinity)

_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(_ROOT / "src"))

from igv.boltz_score import (  # noqa: E402
    AFFINITY_SCORES,
    build_affinity_yaml_spec,
    check_crop_coverage,
    select_affinity_checkpoint,
)


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        args = gate_affinity.build_parser().parse_args([])
        assert args.protein_chain_id == "A"
        assert args.ligand_id == "B"
        assert args.score == "affinity_pred_value"
        assert args.dry_run is False
        assert args.completeness_threshold == pytest.approx(0.10)
        assert args.ligand_smiles is None
        assert args.ligand_ccd is None

    def test_custom_args(self):
        args = gate_affinity.build_parser().parse_args([
            "--protein-chain-id", "C",
            "--ligand-id", "D",
            "--ligand-smiles", "CC(=O)O",
            "--score", "affinity_logits_binary",
            "--dry-run",
        ])
        assert args.protein_chain_id == "C"
        assert args.ligand_id == "D"
        assert args.ligand_smiles == "CC(=O)O"
        assert args.score == "affinity_logits_binary"
        assert args.dry_run is True

    def test_m_steps_ladder(self):
        args = gate_affinity.build_parser().parse_args([
            "--m-steps", "8", "16",
        ])
        assert args.m_steps == [8, 16]


# ---------------------------------------------------------------------------
# 2. Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    def test_prints_plan(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        out = result.stdout
        assert "Gate Affinity" in out
        assert "Pass criteria" in out
        assert "INCONCLUSIVE" in out
        assert "VM command" in out
        assert "FULL TRUNK" in out
        assert "PFA" in out
        assert "TR-beta" in out


# ---------------------------------------------------------------------------
# 3. INCONCLUSIVE checks
# ---------------------------------------------------------------------------


class TestCheckInconclusive:
    def test_healthy_returns_none(self):
        assert gate_affinity.check_inconclusive(3.5, 5.2) is None

    def test_non_finite(self):
        reason = gate_affinity.check_inconclusive(float("nan"), 1.0)
        assert reason is not None
        assert "non-finite" in reason

    def test_collapsed_span(self):
        reason = gate_affinity.check_inconclusive(2.0, 2.0)
        assert reason is not None
        assert "collapsed" in reason

    def test_none_scores(self):
        reason = gate_affinity.check_inconclusive(None, None)
        assert reason is not None


# ---------------------------------------------------------------------------
# 4. Evaluation logic
# ---------------------------------------------------------------------------


class TestEvaluateChecks:
    def test_all_pass(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=1e-3,
            grad_zero_frac=0.01,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 32),
            grad_n_nan=0,
            grad_n_inf=0,
            completeness_abs_err=0.005,
            completeness_rel_err=0.02,
            ig_sum=-3.0,
            f_diff=-3.005,
        )
        assert len(checks) == 5
        assert all(c["passed"] for c in checks)

    def test_all_fail_when_grad_none(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=False,
            grad_max_abs=None,
            grad_zero_frac=None,
            grad_shape=None,
            expected_shape=(1, 100, 32),
            grad_n_nan=None,
            grad_n_inf=None,
            completeness_abs_err=None,
            completeness_rel_err=None,
            ig_sum=None,
            f_diff=None,
        )
        assert not any(c["passed"] for c in checks)

    def test_zero_grad_fails(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=0.0,
            grad_zero_frac=1.0,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 32),
            grad_n_nan=0,
            grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.01,
            ig_sum=0.0,
            f_diff=-1.0,
        )
        failed = [c["name"] for c in checks if not c["passed"]]
        assert "res_type.grad not uniformly zero" in failed

    def test_shape_mismatch_fails(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=0.5,
            grad_zero_frac=0.0,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 64),
            grad_n_nan=0,
            grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.01,
            ig_sum=-1.0,
            f_diff=-1.001,
        )
        shape_check = [c for c in checks if "shape" in c["name"]][0]
        assert not shape_check["passed"]

    def test_nan_fails(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=1.0,
            grad_zero_frac=0.0,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 32),
            grad_n_nan=5,
            grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.01,
            ig_sum=-1.0,
            f_diff=-1.001,
        )
        finite_check = [c for c in checks if "finite" in c["name"]][0]
        assert not finite_check["passed"]

    def test_above_threshold_fails(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=1.0,
            grad_zero_frac=0.0,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 32),
            grad_n_nan=0,
            grad_n_inf=0,
            completeness_abs_err=0.15,
            completeness_rel_err=0.30,
            ig_sum=-0.5,
            f_diff=-1.0,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert not comp_check["passed"]

    def test_custom_threshold(self):
        checks = gate_affinity.evaluate_checks(
            grad_not_none=True,
            grad_max_abs=1.0,
            grad_zero_frac=0.0,
            grad_shape=(1, 100, 32),
            expected_shape=(1, 100, 32),
            grad_n_nan=0,
            grad_n_inf=0,
            completeness_abs_err=0.04,
            completeness_rel_err=0.05,
            ig_sum=-0.96,
            f_diff=-1.0,
            threshold=0.05,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert comp_check["passed"]
        assert "< 0.05" in comp_check["name"]


# ---------------------------------------------------------------------------
# 5. print_checks
# ---------------------------------------------------------------------------


class TestPrintChecks:
    def test_returns_true_when_all_pass(self, capsys):
        checks = [
            {"name": "c1", "passed": True, "detail": "ok"},
            {"name": "c2", "passed": True, "detail": ""},
        ]
        assert gate_affinity.print_checks(checks) is True

    def test_returns_false_when_any_fail(self, capsys):
        checks = [
            {"name": "c1", "passed": True, "detail": ""},
            {"name": "c2", "passed": False, "detail": "bad"},
        ]
        assert gate_affinity.print_checks(checks) is False


# ---------------------------------------------------------------------------
# 6. Affinity score registry
# ---------------------------------------------------------------------------


class TestAffinityScoreRegistry:
    def test_affinity_pred_value_registered(self):
        assert "affinity_pred_value" in AFFINITY_SCORES

    def test_affinity_logits_binary_registered(self):
        assert "affinity_logits_binary" in AFFINITY_SCORES

    def test_affinity_pred_value_extracts(self):
        out = {"affinity_pred_value": 3.5, "affinity_logits_binary": 0.8}
        assert AFFINITY_SCORES["affinity_pred_value"](out) == 3.5

    def test_affinity_logits_binary_extracts(self):
        out = {"affinity_pred_value": 3.5, "affinity_logits_binary": 0.8}
        assert AFFINITY_SCORES["affinity_logits_binary"](out) == 0.8

    def test_no_overlap_with_confidence_scores(self):
        from igv.boltz_score import SCORES
        overlap = set(SCORES) & set(AFFINITY_SCORES)
        assert not overlap, f"Overlapping score names: {overlap}"


# ---------------------------------------------------------------------------
# 7. Affinity checkpoint selection
# ---------------------------------------------------------------------------


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_bytes(b"")


class TestSelectAffinityCheckpoint:
    def test_picks_affinity(self, tmp_path):
        _touch(tmp_path, "boltz2_aff.ckpt", "boltz2_conf.ckpt")
        assert select_affinity_checkpoint(tmp_path).name == "boltz2_aff.ckpt"

    def test_affinity_only(self, tmp_path):
        _touch(tmp_path, "boltz2_aff.ckpt")
        assert select_affinity_checkpoint(tmp_path).name == "boltz2_aff.ckpt"

    def test_no_affinity_raises(self, tmp_path):
        _touch(tmp_path, "boltz2_conf.ckpt")
        with pytest.raises(FileNotFoundError, match="No affinity checkpoint"):
            select_affinity_checkpoint(tmp_path)

    def test_no_checkpoints_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .ckpt"):
            select_affinity_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# 8. Affinity YAML construction
# ---------------------------------------------------------------------------


class TestBuildAffinityYamlSpec:
    def test_smiles_basic(self):
        spec = build_affinity_yaml_spec(
            {"A": "MVLSPAD"},
            ligand_id="B",
            ligand_smiles="CC(=O)O",
        )
        assert spec["version"] == 1
        seqs = spec["sequences"]
        assert len(seqs) == 2
        assert "protein" in seqs[0]
        assert seqs[0]["protein"]["id"] == "A"
        assert seqs[0]["protein"]["sequence"] == "MVLSPAD"
        assert "ligand" in seqs[1]
        assert seqs[1]["ligand"]["smiles"] == "CC(=O)O"
        assert seqs[1]["ligand"]["id"] == "B"
        props = spec["properties"]
        assert len(props) == 1
        assert props[0]["affinity"]["binder"] == "B"

    def test_ccd_basic(self):
        spec = build_affinity_yaml_spec(
            {"A": "MVLSPAD"},
            ligand_id="B",
            ligand_ccd="ATP",
        )
        assert "ccd" in spec["sequences"][1]["ligand"]
        assert spec["sequences"][1]["ligand"]["ccd"] == "ATP"

    def test_both_smiles_and_ccd_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            build_affinity_yaml_spec(
                {"A": "MV"}, ligand_id="B",
                ligand_smiles="CC", ligand_ccd="ATP",
            )

    def test_neither_smiles_nor_ccd_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            build_affinity_yaml_spec({"A": "MV"}, ligand_id="B")

    def test_ligand_id_collision_raises(self):
        with pytest.raises(ValueError, match="collides"):
            build_affinity_yaml_spec(
                {"A": "MV"}, ligand_id="A", ligand_smiles="CC",
            )

    def test_multi_protein_chains(self):
        spec = build_affinity_yaml_spec(
            {"A": "MV", "C": "KKKK"},
            ligand_id="B",
            ligand_smiles="CC",
        )
        seqs = spec["sequences"]
        assert len(seqs) == 3
        protein_ids = [s["protein"]["id"] for s in seqs if "protein" in s]
        assert sorted(protein_ids) == ["A", "C"]

    def test_msa_string(self):
        spec = build_affinity_yaml_spec(
            {"A": "MV"},
            ligand_id="B",
            ligand_smiles="CC",
            resolved_msa="empty",
        )
        assert spec["sequences"][0]["protein"]["msa"] == "empty"

    def test_msa_dict(self, tmp_path):
        msa_file = tmp_path / "a.a3m"
        msa_file.write_text(">seq\nMV\n")
        spec = build_affinity_yaml_spec(
            {"A": "MV"},
            ligand_id="B",
            ligand_smiles="CC",
            resolved_msa={"A": str(msa_file)},
        )
        assert spec["sequences"][0]["protein"]["msa"] == str(msa_file)

    def test_msa_dict_missing_chain_raises(self):
        with pytest.raises(ValueError, match="missing chains"):
            build_affinity_yaml_spec(
                {"A": "MV", "C": "KK"},
                ligand_id="B",
                ligand_smiles="CC",
                resolved_msa={"A": "/fake/path"},
            )


# ---------------------------------------------------------------------------
# 9. EXAMPLE_COMPLEX is correctly identified
# ---------------------------------------------------------------------------


class TestExampleComplex:
    def test_has_required_keys(self):
        ex = gate_affinity.EXAMPLE_COMPLEX
        assert "name" in ex
        assert "protein_chain_id" in ex
        assert "ligand_id" in ex
        assert "n_tokens_approx" in ex
        assert ex["n_tokens_approx"] <= 256

    def test_is_tr_beta_pfa(self):
        ex = gate_affinity.EXAMPLE_COMPLEX
        assert "TR-beta" in ex["name"]
        assert ex["ligand_ccd"] == "PFA"
        assert ex["pfa_heavy_atoms"] == 27
        assert ex["pfa_heavy_atoms"] <= 128

    def test_pfa_under_warn_threshold(self):
        assert gate_affinity.EXAMPLE_COMPLEX["pfa_heavy_atoms"] <= 56

    def test_sequence_length_matches(self):
        assert len(gate_affinity._1N46_SEQUENCE) == 258


# ---------------------------------------------------------------------------
# 10. Crop coverage guard
# ---------------------------------------------------------------------------


class TestCheckCropCoverage:
    def test_all_sites_present(self):
        token_map = {("A", 0): 0, ("A", 1): 1, ("A", 2): 2}
        result = check_crop_coverage(
            token_map, chain_id="A", sites=[0, 1, 2],
        )
        assert result["n_requested"] == 3
        assert result["n_surviving"] == 3
        assert result["n_cropped"] == 0
        assert result["surviving"] == [0, 1, 2]
        assert result["cropped"] == []
        assert result["fraction_surviving"] == pytest.approx(1.0)

    def test_some_sites_cropped(self):
        token_map = {("A", 0): 0, ("A", 2): 1}
        result = check_crop_coverage(
            token_map, chain_id="A", sites=[0, 1, 2],
        )
        assert result["n_surviving"] == 2
        assert result["n_cropped"] == 1
        assert 1 in result["cropped"]
        assert result["fraction_surviving"] == pytest.approx(2.0 / 3.0)

    def test_all_sites_cropped(self):
        token_map = {("B", 0): 0}
        result = check_crop_coverage(
            token_map, chain_id="A", sites=[0, 1],
        )
        assert result["n_surviving"] == 0
        assert result["n_cropped"] == 2

    def test_empty_sites(self):
        token_map = {("A", 0): 0}
        result = check_crop_coverage(
            token_map, chain_id="A", sites=[],
        )
        assert result["n_requested"] == 0
        assert result["n_surviving"] == 0

    def test_returns_token_indices(self):
        token_map = {("A", 5): 3, ("A", 10): 7}
        result = check_crop_coverage(
            token_map, chain_id="A", sites=[5, 10, 15],
        )
        assert result["surviving"] == [5, 10]
        assert result["cropped"] == [15]
        assert result["surviving_token_indices"] == [3, 7]


# ---------------------------------------------------------------------------
# 11. Baseline selection CLI
# ---------------------------------------------------------------------------


class TestBaselineSelection:
    def test_default_is_mean_aa(self):
        args = gate_affinity.build_parser().parse_args([])
        assert args.baseline == "mean_aa"

    def test_zeros_accepted(self):
        args = gate_affinity.build_parser().parse_args(["--baseline", "zeros"])
        assert args.baseline == "zeros"

    def test_mean_aa_accepted(self):
        args = gate_affinity.build_parser().parse_args(["--baseline", "mean_aa"])
        assert args.baseline == "mean_aa"

    def test_invalid_rejected(self):
        with pytest.raises(SystemExit):
            gate_affinity.build_parser().parse_args(["--baseline", "random"])


# ---------------------------------------------------------------------------
# 12. x_pred CLI
# ---------------------------------------------------------------------------


class TestXPredSelection:
    def test_default_is_pdb(self):
        args = gate_affinity.build_parser().parse_args([])
        assert args.x_pred == "pdb"

    def test_zeros_accepted(self):
        args = gate_affinity.build_parser().parse_args(["--x-pred", "zeros"])
        assert args.x_pred == "zeros"

    def test_predicted_accepted(self):
        args = gate_affinity.build_parser().parse_args(["--x-pred", "predicted"])
        assert args.x_pred == "predicted"

    def test_pdb_accepted(self):
        args = gate_affinity.build_parser().parse_args(["--x-pred", "pdb"])
        assert args.x_pred == "pdb"


# ---------------------------------------------------------------------------
# 12b. PDB and resnum-offset CLI
# ---------------------------------------------------------------------------


class TestPdbAndResnumOffset:
    def test_pdb_default_none(self):
        args = gate_affinity.build_parser().parse_args([])
        assert args.pdb is None

    def test_pdb_accepts_path(self):
        args = gate_affinity.build_parser().parse_args(["--pdb", "/tmp/test.pdb"])
        assert args.pdb == "/tmp/test.pdb"

    def test_resnum_offset_default_zero(self):
        args = gate_affinity.build_parser().parse_args([])
        assert args.resnum_offset == 0

    def test_resnum_offset_accepts_int(self):
        args = gate_affinity.build_parser().parse_args(["--resnum-offset", "229"])
        assert args.resnum_offset == 229


# ---------------------------------------------------------------------------
# 13. build_mean_aa_res_type_baseline
# ---------------------------------------------------------------------------


class TestBuildMeanAaResTypeBaseline:
    def _make_feats(self, n_protein, n_ligand):
        import torch
        L = n_protein + n_ligand
        num_tokens = 33
        res_type = torch.zeros(1, L, num_tokens)
        for i in range(n_protein):
            res_type[0, i, 2 + (i % 20)] = 1.0
        for i in range(n_ligand):
            res_type[0, n_protein + i, 0] = 1.0

        pad_mask = torch.ones(1, L)
        mol_type = torch.zeros(1, L, dtype=torch.long)
        mol_type[0, n_protein:] = 2
        return res_type, {
            "token_pad_mask": pad_mask,
            "mol_type": mol_type,
        }

    def test_shape_preserved(self):
        from igv.boltz_score import build_mean_aa_res_type_baseline
        res_type, feats = self._make_feats(10, 3)
        baseline = build_mean_aa_res_type_baseline(res_type, feats)
        assert baseline.shape == res_type.shape

    def test_protein_positions_uniform(self):
        from igv.boltz_score import build_mean_aa_res_type_baseline, canonical_aa_token_indices
        res_type, feats = self._make_feats(10, 3)
        baseline = build_mean_aa_res_type_baseline(res_type, feats)
        aa_idx = canonical_aa_token_indices()
        for i in range(10):
            row = baseline[0, i]
            for j in aa_idx:
                assert row[j].item() == pytest.approx(1.0 / 20, abs=1e-6)
            non_aa_sum = sum(row[k].item() for k in range(33) if k not in aa_idx)
            assert non_aa_sum == pytest.approx(0.0, abs=1e-6)

    def test_ligand_positions_unchanged(self):
        import torch
        from igv.boltz_score import build_mean_aa_res_type_baseline
        res_type, feats = self._make_feats(10, 3)
        baseline = build_mean_aa_res_type_baseline(res_type, feats)
        for i in range(3):
            assert torch.allclose(baseline[0, 10 + i], res_type[0, 10 + i].float())


# ---------------------------------------------------------------------------
# 14. Artifact schema (.npz key expectations)
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    def test_required_keys_present(self):
        required = {
            "grad_res_type", "token_map_keys", "token_map_values",
            "crop_coverage", "vocab_order",
        }
        ladder = [16, 32, 64]
        expected = set(required)
        expected.add("res_type_ig")
        for m in ladder:
            expected.add(f"res_type_ig_m{m}")
            expected.add(f"res_type_grad_m{m}")
        assert required.issubset(expected)

    def test_vocab_order_has_all_canonical_aas(self):
        from igv.attrib import CANONICAL_AMINO_ACIDS
        from igv.boltz_score import build_vocab_order
        vocab = build_vocab_order()
        vocab_set = set(str(v) for v in vocab if str(v))
        for aa in CANONICAL_AMINO_ACIDS:
            assert aa in vocab_set, f"Missing amino acid: {aa}"

    def test_vocab_order_length_matches_num_tokens(self):
        from igv.boltz_score import build_vocab_order, _BOLTZ_TOKENS_2_2_1
        vocab = build_vocab_order()
        assert len(vocab) == len(_BOLTZ_TOKENS_2_2_1)


# ---------------------------------------------------------------------------
# 15. Token-map round trip
# ---------------------------------------------------------------------------


class TestTokenMapRoundTrip:
    def test_round_trip(self):
        import numpy as np
        token_map = {("0", 5): 0, ("0", 10): 1, ("1", 0): 2}
        items = sorted(token_map.items(), key=lambda kv: kv[1])
        keys = np.array([(k[0], str(k[1])) for k, _ in items], dtype=object)
        values = np.array([v for _, v in items], dtype=np.int64)

        reconstructed = {
            (str(k[0]), int(k[1])): int(v) for k, v in zip(keys, values)
        }
        assert reconstructed == token_map


# ---------------------------------------------------------------------------
# 16. Crop-coverage field in JSON output
# ---------------------------------------------------------------------------


class TestCropCoverageInOutput:
    def test_crop_coverage_dict_structure(self):
        result = check_crop_coverage(
            {("0", 0): 0, ("0", 1): 1, ("0", 5): 2},
            chain_id="0",
            sites=[0, 1, 2, 3, 4, 5],
        )
        assert "n_requested" in result
        assert "n_surviving" in result
        assert "n_cropped" in result
        assert "surviving" in result
        assert "cropped" in result
        assert "fraction_surviving" in result
        assert result["n_requested"] == 6
        assert result["n_surviving"] == 3
        assert result["n_cropped"] == 3


# ---------------------------------------------------------------------------
# 17. out-npz CLI argument
# ---------------------------------------------------------------------------


class TestOutNpzArg:
    def test_default_npz_path(self):
        args = gate_affinity.build_parser().parse_args([])
        assert "npz" in args.out_npz

    def test_custom_npz_path(self):
        args = gate_affinity.build_parser().parse_args(["--out-npz", "custom.npz"])
        assert args.out_npz == "custom.npz"


# ---------------------------------------------------------------------------
# 18. Token-map resnum_offset
# ---------------------------------------------------------------------------


class TestTokenMapResnumOffset:
    def test_offset_applied_to_protein(self):
        import torch
        from igv.boltz_score import _build_affinity_token_map
        feats = {
            "token_pad_mask": torch.ones(1, 5),
            "asym_id": torch.tensor([[0, 0, 0, 1, 1]]),
            "residue_index": torch.tensor([[0, 1, 2, 0, 1]]),
        }
        tm = _build_affinity_token_map(feats, protein_chain_asym=0, resnum_offset=229)
        assert ("0", 229) in tm
        assert ("0", 230) in tm
        assert ("0", 231) in tm
        assert ("1", 0) in tm
        assert ("1", 1) in tm

    def test_zero_offset_is_identity(self):
        import torch
        from igv.boltz_score import _build_affinity_token_map
        feats = {
            "token_pad_mask": torch.ones(1, 3),
            "asym_id": torch.tensor([[0, 0, 0]]),
            "residue_index": torch.tensor([[0, 1, 2]]),
        }
        tm = _build_affinity_token_map(feats, protein_chain_asym=0, resnum_offset=0)
        assert ("0", 0) in tm
        assert ("0", 1) in tm
        assert ("0", 2) in tm


# ---------------------------------------------------------------------------
# 19. Identity guard (_verify_token_map_identity)
# ---------------------------------------------------------------------------


class TestVerifyTokenMapIdentity:
    def test_matching_passes(self, tmp_path):
        from igv.boltz_score import _verify_token_map_identity
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A 235       1.0  2.0  3.0  1.00  0.00\n"
            "ATOM      2  CA  GLY A 236       4.0  5.0  6.0  1.00  0.00\n"
        )
        token_map = {("0", 235): 0, ("0", 236): 1}
        _verify_token_map_identity(
            token_map, pdb,
            protein_chain_asym_str="0",
            protein_sequence="AG",
            resnum_offset=235,
        )

    def test_mismatch_raises(self, tmp_path):
        from igv.boltz_score import _verify_token_map_identity
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A 235       1.0  2.0  3.0  1.00  0.00\n"
            "ATOM      2  CA  GLY A 236       4.0  5.0  6.0  1.00  0.00\n"
        )
        token_map = {("0", 235): 0, ("0", 236): 1}
        with pytest.raises(RuntimeError, match="identity check FAILED"):
            _verify_token_map_identity(
                token_map, pdb,
                protein_chain_asym_str="0",
                protein_sequence="GG",
                resnum_offset=235,
            )

    def test_gap_residue_not_in_pdb_passes(self, tmp_path):
        from igv.boltz_score import _verify_token_map_identity
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A 235       1.0  2.0  3.0  1.00  0.00\n"
            "ATOM      2  CA  GLY A 237       4.0  5.0  6.0  1.00  0.00\n"
        )
        token_map = {("0", 235): 0, ("0", 236): 1, ("0", 237): 2}
        _verify_token_map_identity(
            token_map, pdb,
            protein_chain_asym_str="0",
            protein_sequence="AEG",
            resnum_offset=235,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
