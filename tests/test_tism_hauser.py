"""Tests for Stage 19 — TISM attribution vs Hauser 2018.

CPU only, no boltz, no GPU. Tests argument parsing, --dry-run, the pure
analysis functions, artifact validation, crop exclusion, sign convention,
DOF guard, and an end-to-end synthetic gradient fixture with a planted
correlation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "19_tism_hauser.py"
_spec = importlib.util.spec_from_file_location("tism_hauser", str(_SCRIPT))
tism_hauser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tism_hauser)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from igv.abl import RT_LN10_300K, load_hauser2018_from_csv  # noqa: E402
from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# Helpers for synthetic artifact construction
# ---------------------------------------------------------------------------


def _make_synthetic_artifact(
    tmp_path: Path,
    *,
    n_positions: int = 50,
    vocab: list[str] | None = None,
    seed: int = 123,
    planted_correlation: float = 0.8,
) -> tuple[Path, pd.DataFrame]:
    """Build a synthetic gradient .npz and matching Hauser-format CSV.

    Plants a known correlation between the TISM predictions and DDG_exp
    so that the full pipeline can be verified end to end.
    """
    if vocab is None:
        vocab = AA_LIST

    rng = np.random.default_rng(seed)
    L = n_positions
    V = len(vocab)

    grad = rng.standard_normal((L, V))

    tkis = ["imatinib", "dasatinib", "nilotinib", "ponatinib", "bosutinib", "axitinib"]

    rows = []
    for tki in tkis:
        for pos_idx in range(min(25, L)):
            wt_aa = vocab[pos_idx % len(vocab)]
            mut_candidates = [aa for aa in vocab if aa != wt_aa]
            if not mut_candidates:
                continue
            mut_aa = mut_candidates[pos_idx % len(mut_candidates)]

            wt_col = vocab.index(wt_aa)
            mut_col = vocab.index(mut_aa)
            tism_pred = float(grad[pos_idx, mut_col] - grad[pos_idx, wt_col])

            noise = rng.normal(0, 0.5)
            ddg_target = tism_pred * planted_correlation + noise

            ddg_exp = ddg_target
            dlog10 = ddg_exp / RT_LN10_300K
            ic50_wt = 200.0
            ic50_mut = ic50_wt * 10 ** dlog10

            resnum = 244 + pos_idx
            mut_str = f"{wt_aa}{resnum}{mut_aa}"

            rows.append({
                "TKI": tki,
                "mutation": mut_str,
                "IC50": ic50_mut,
                "DDG_exp": ddg_exp,
                "DDG_prime": 0.0,
                "DDG_fep1": 0.0, "dDDG_fep1": 0.0,
                "DDG_fep2": 0.0, "dDDG_fep2": 0.0,
                "DDG_fep3": 0.0, "dDDG_fep3": 0.0,
            })

    df = pd.DataFrame(rows)

    chain_id = "0"
    token_map_keys = []
    token_map_values = []
    surviving = []
    for i in range(L):
        resnum = 244 + i
        token_map_keys.append([chain_id, resnum])
        token_map_values.append(i)
        surviving.append(resnum)

    crop_coverage = {
        "n_requested": L,
        "n_surviving": L,
        "n_cropped": 0,
        "surviving": surviving,
        "cropped": [],
    }

    npz_path = tmp_path / "test_gradient.npz"
    csv_path = tmp_path / "test_hauser.csv"

    np.savez(
        str(npz_path),
        grad_res_type=grad,
        token_map_keys=np.array(token_map_keys, dtype=object),
        token_map_values=np.array(token_map_values),
        crop_coverage=json.dumps(crop_coverage),
        vocab_order=np.array(vocab),
    )

    df.to_csv(csv_path, index=False)

    return npz_path, csv_path, df


def _make_artifact_with_crop(tmp_path: Path) -> tuple[Path, Path]:
    """Build an artifact where some positions are cropped out."""
    vocab = AA_LIST
    rng = np.random.default_rng(42)
    L = 30
    V = len(vocab)
    grad = rng.standard_normal((L, V))

    chain_id = "0"
    kept_positions = list(range(244, 244 + 20))
    cropped_positions = list(range(264, 264 + 10))

    token_map_keys = []
    token_map_values = []
    for i, resnum in enumerate(kept_positions):
        token_map_keys.append([chain_id, resnum])
        token_map_values.append(i)

    crop_coverage = {
        "n_requested": 30,
        "n_surviving": 20,
        "n_cropped": 10,
        "surviving": kept_positions,
        "cropped": cropped_positions,
    }

    npz_path = tmp_path / "crop_gradient.npz"
    np.savez(
        str(npz_path),
        grad_res_type=grad,
        token_map_keys=np.array(token_map_keys, dtype=object),
        token_map_values=np.array(token_map_values),
        crop_coverage=json.dumps(crop_coverage),
        vocab_order=np.array(vocab),
    )

    rows = []
    for tki in ["imatinib", "dasatinib"]:
        for resnum in kept_positions[:5] + cropped_positions[:3]:
            wt_idx = (resnum - 244) % V
            wt_aa = vocab[wt_idx]
            mut_aa = vocab[(wt_idx + 1) % V]
            rows.append({
                "TKI": tki,
                "mutation": f"{wt_aa}{resnum}{mut_aa}",
                "IC50": 100.0,
                "DDG_exp": 0.5,
                "DDG_prime": 0.0,
                "DDG_fep1": 0.0, "dDDG_fep1": 0.0,
                "DDG_fep2": 0.0, "dDDG_fep2": 0.0,
                "DDG_fep3": 0.0, "dDDG_fep3": 0.0,
            })

    csv_path = tmp_path / "crop_hauser.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return npz_path, csv_path


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        args = tism_hauser.build_parser().parse_args(["--gradient", "test.npz"])
        assert args.gradient == "test.npz"
        assert args.dry_run is False
        assert args.n_permutations == 1000
        assert args.seed == 42
        assert args.protein_chain_id == "0"

    def test_dry_run(self):
        args = tism_hauser.build_parser().parse_args([
            "--gradient", "test.npz", "--dry-run",
        ])
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# 2. Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_exits_zero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", "fake.npz", "--dry-run",
             "--out", str(tmp_path / "dry.json")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    def test_prints_plan(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", "fake.npz", "--dry-run",
             "--out", str(tmp_path / "dry.json")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        out = result.stdout
        assert "TISM" in out
        assert "Hauser" in out
        assert "grad_res_type" in out
        assert "T315I" in out

    def test_writes_provenance(self, tmp_path):
        out_json = tmp_path / "dry.json"
        subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", "fake.npz", "--dry-run",
             "--out", str(out_json)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        prov_path = Path(str(out_json) + ".prov.json")
        assert prov_path.exists()


# ---------------------------------------------------------------------------
# 3. Artifact validation
# ---------------------------------------------------------------------------


class TestValidateArtifact:
    def test_valid_passes(self):
        npz = {k: np.array([1]) for k in tism_hauser.REQUIRED_NPZ_KEYS}
        tism_hauser.validate_artifact(npz)

    def test_missing_key_raises(self):
        npz = {k: np.array([1]) for k in tism_hauser.REQUIRED_NPZ_KEYS}
        del npz["grad_res_type"]
        with pytest.raises(ValueError, match="grad_res_type"):
            tism_hauser.validate_artifact(npz)

    def test_extra_keys_ok(self):
        npz = {k: np.array([1]) for k in tism_hauser.REQUIRED_NPZ_KEYS}
        npz["bonus"] = np.array([42])
        tism_hauser.validate_artifact(npz)


# ---------------------------------------------------------------------------
# 4. TISM prediction
# ---------------------------------------------------------------------------


class TestTismPredict:
    def test_difference_form(self):
        grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vocab_lookup = {"A": 0, "C": 1, "D": 2}
        result = tism_hauser.tism_predict(grad, 0, "A", "D", vocab_lookup)
        assert result == pytest.approx(3.0 - 1.0)

    def test_same_aa_is_zero(self):
        grad = np.array([[1.0, 2.0, 3.0]])
        vocab_lookup = {"A": 0, "C": 1, "D": 2}
        result = tism_hauser.tism_predict(grad, 0, "A", "A", vocab_lookup)
        assert result == pytest.approx(0.0)

    def test_antisymmetric(self):
        grad = np.array([[1.0, 5.0]])
        vocab_lookup = {"A": 0, "C": 1}
        fwd = tism_hauser.tism_predict(grad, 0, "A", "C", vocab_lookup)
        rev = tism_hauser.tism_predict(grad, 0, "C", "A", vocab_lookup)
        assert fwd == pytest.approx(-rev)


# ---------------------------------------------------------------------------
# 5. Sign convention
# ---------------------------------------------------------------------------


class TestSignConvention:
    def test_positive_ddg_means_resistance_no_negation(self):
        """Positive DDG_exp = resistance = weaker binding.
        affinity_pred_value = log10(IC50 uM), so increase = weaker binding.
        Both on the same convention; no negation needed.

        This test plants a trivially positive-correlated gradient and DDG
        to verify no sign flip occurs in the pipeline.
        """
        grad = np.zeros((5, 20))
        vocab = AA_LIST
        vocab_lookup = {aa: i for i, aa in enumerate(vocab)}

        for i in range(5):
            grad[i, vocab_lookup["A"]] = 0.0
            grad[i, vocab_lookup["C"]] = float(i + 1)

        preds = []
        true_vals = []
        for i in range(5):
            pred = tism_hauser.tism_predict(grad, i, "A", "C", vocab_lookup)
            preds.append(pred)
            true_vals.append(float(i + 1))

        from igv.metrics import spearman
        rho = spearman(np.array(preds), np.array(true_vals))
        assert rho > 0.9, (
            f"Sign flip detected: rho={rho}. Positive DDG and positive TISM "
            f"pred should correlate positively without negation."
        )


# ---------------------------------------------------------------------------
# 6. Crop exclusion
# ---------------------------------------------------------------------------


class TestCropExclusion:
    def test_cropped_positions_get_nan_pred(self, tmp_path):
        npz_path, csv_path = _make_artifact_with_crop(tmp_path)
        npz = dict(np.load(str(npz_path), allow_pickle=True))

        token_map = tism_hauser.reconstruct_token_map(
            npz["token_map_keys"], npz["token_map_values"],
        )
        vocab_lookup = tism_hauser.build_vocab_lookup(npz["vocab_order"])
        crop_coverage = json.loads(str(npz["crop_coverage"]))
        crop_surviving = set(crop_coverage["surviving"])

        hauser_df = load_hauser2018_from_csv(csv_path)
        from igv.abl import measured, xtal_only
        xtal_df = xtal_only(measured(hauser_df))

        mapped = tism_hauser.map_mutations_to_predictions(
            xtal_df, npz["grad_res_type"], token_map, vocab_lookup,
            "0", crop_surviving,
        )

        excluded = mapped[mapped["crop_excluded"]]
        included = mapped[~mapped["crop_excluded"]]

        assert len(excluded) > 0, "Expected some crop-excluded mutations"
        assert all(np.isnan(excluded["tism_pred"])), "Excluded should have NaN pred"
        assert all(~np.isnan(included["tism_pred"])), "Included should have real pred"

    def test_excluded_not_in_correlation(self, tmp_path):
        npz_path, csv_path = _make_artifact_with_crop(tmp_path)
        npz = dict(np.load(str(npz_path), allow_pickle=True))

        token_map = tism_hauser.reconstruct_token_map(
            npz["token_map_keys"], npz["token_map_values"],
        )
        vocab_lookup = tism_hauser.build_vocab_lookup(npz["vocab_order"])
        crop_coverage = json.loads(str(npz["crop_coverage"]))
        crop_surviving = set(crop_coverage["surviving"])

        hauser_df = load_hauser2018_from_csv(csv_path)
        from igv.abl import measured, xtal_only
        xtal_df = xtal_only(measured(hauser_df))

        mapped = tism_hauser.map_mutations_to_predictions(
            xtal_df, npz["grad_res_type"], token_map, vocab_lookup,
            "0", crop_surviving,
        )

        per_inh = tism_hauser.per_inhibitor_analysis(mapped, n_perm=10, seed=42)
        for tki, res in per_inh.items():
            n = res["n"]
            included_for_tki = mapped[
                (mapped["tki"] == tki) & ~mapped["crop_excluded"]
            ]
            assert n == len(included_for_tki), \
                "Per-inhibitor n should count only included, not excluded"


# ---------------------------------------------------------------------------
# 7. DOF guard
# ---------------------------------------------------------------------------


class TestDofGuard:
    def test_refuses_when_too_few_dof(self):
        result = tism_hauser.dof_guard(n_positions=10, n_confounds=10)
        assert not result["partial_correlation_safe"]
        assert "REFUSED" in result["note"]

    def test_allows_when_enough_dof(self):
        result = tism_hauser.dof_guard(n_positions=50, n_confounds=10)
        assert result["partial_correlation_safe"]

    def test_hauser_20_positions_is_unsafe(self):
        result = tism_hauser.dof_guard(n_positions=20, n_confounds=10)
        assert not result["partial_correlation_safe"]
        assert result["residual_dof"] == 9


# ---------------------------------------------------------------------------
# 8. Permutation null
# ---------------------------------------------------------------------------


class TestPermutationNull:
    def test_correlated_data_low_p(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = x + rng.normal(0, 0.3, 50)
        result = tism_hauser.permutation_null(x, y, n_perm=500, seed=42)
        assert result["p_value_two_sided"] < 0.05

    def test_uncorrelated_data_high_p(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        result = tism_hauser.permutation_null(x, y, n_perm=500, seed=42)
        assert result["p_value_two_sided"] > 0.05

    def test_returns_required_keys(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = tism_hauser.permutation_null(x, y, n_perm=100)
        assert "n_permutations" in result
        assert "null_mean" in result
        assert "p_value_two_sided" in result


# ---------------------------------------------------------------------------
# 9. End-to-end synthetic gradient fixture
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_planted_correlation_recovers(self, tmp_path):
        """Build a synthetic artifact with a planted correlation and verify
        the whole pipeline recovers it."""
        npz_path, csv_path, _ = _make_synthetic_artifact(
            tmp_path, planted_correlation=1.5, seed=99,
        )

        result = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", str(npz_path),
             "--hauser", str(csv_path),
             "--out", str(tmp_path / "result.json"),
             "--n-permutations", "100",
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

        out = json.loads((tmp_path / "result.json").read_text())

        assert out["n_usable"] > 0
        assert out["n_docked_excluded"] == 0
        assert "per_inhibitor" in out
        assert "pooled_within_inhibitor_ranks" in out
        assert "within_mutation_across_inhibitor" in out
        assert "t315i_sanity_check" in out
        assert "dof_guard" in out

        pooled_rho = out["pooled_within_inhibitor_ranks"]["spearman"]
        assert pooled_rho > 0.3, (
            f"Planted correlation should yield positive pooled rho, got {pooled_rho}"
        )

    def test_output_has_provenance(self, tmp_path):
        npz_path, csv_path, _ = _make_synthetic_artifact(tmp_path)
        out_json = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", str(npz_path),
             "--hauser", str(csv_path),
             "--out", str(out_json),
             "--n-permutations", "10"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        prov_path = Path(str(out_json) + ".prov.json")
        assert prov_path.exists()
        prov = json.loads(prov_path.read_text())
        assert prov["stage"] == "19_tism_hauser"

    def test_sign_preserved_in_output(self, tmp_path):
        """Verify that the sign convention note is in the output."""
        npz_path, csv_path, _ = _make_synthetic_artifact(tmp_path)
        out_json = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--gradient", str(npz_path),
             "--hauser", str(csv_path),
             "--out", str(out_json),
             "--n-permutations", "10"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        out = json.loads(out_json.read_text())
        assert "no negation" in out["sign_convention"].lower()


# ---------------------------------------------------------------------------
# 10. Vocab lookup
# ---------------------------------------------------------------------------


class TestVocabLookup:
    def test_builds_from_array(self):
        vocab = np.array(["A", "C", "D"])
        lookup = tism_hauser.build_vocab_lookup(vocab)
        assert lookup["A"] == 0
        assert lookup["C"] == 1
        assert lookup["D"] == 2

    def test_all_canonical_present(self):
        vocab = np.array(AA_LIST)
        lookup = tism_hauser.build_vocab_lookup(vocab)
        for aa in CANONICAL_AMINO_ACIDS:
            assert aa in lookup


# ---------------------------------------------------------------------------
# 11. Token map reconstruction
# ---------------------------------------------------------------------------


class TestReconstructTokenMap:
    def test_round_trips(self):
        keys = np.array([["0", 10], ["0", 20], ["1", 5]], dtype=object)
        values = np.array([0, 1, 2])
        tm = tism_hauser.reconstruct_token_map(keys, values)
        assert tm[("0", 10)] == 0
        assert tm[("0", 20)] == 1
        assert tm[("1", 5)] == 2


# ---------------------------------------------------------------------------
# 12. Within-mutation across-inhibitor
# ---------------------------------------------------------------------------


class TestWithinMutationAcrossInhibitor:
    def test_returns_per_mutation_dict(self, tmp_path):
        npz_path, csv_path, _ = _make_synthetic_artifact(
            tmp_path, planted_correlation=1.0, seed=77,
        )
        npz = dict(np.load(str(npz_path), allow_pickle=True))
        token_map = tism_hauser.reconstruct_token_map(
            npz["token_map_keys"], npz["token_map_values"],
        )
        vocab_lookup = tism_hauser.build_vocab_lookup(npz["vocab_order"])
        crop_coverage = json.loads(str(npz["crop_coverage"]))
        crop_surviving = set(crop_coverage["surviving"])

        hauser_df = load_hauser2018_from_csv(csv_path)
        from igv.abl import measured, xtal_only
        xtal_df = xtal_only(measured(hauser_df))

        mapped = tism_hauser.map_mutations_to_predictions(
            xtal_df, npz["grad_res_type"], token_map, vocab_lookup,
            "0", crop_surviving,
        )
        result = tism_hauser.within_mutation_across_inhibitor(mapped, n_perm=10)
        assert "per_mutation" in result
        assert result["n_mutations"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
