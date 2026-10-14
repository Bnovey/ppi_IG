"""Tests for Stage 26 -- model vs lab and shortcut vs lab.

CPU only, no network: the SKEMPI loader is exercised against a synthetic
semicolon-delimited CSV and a two-chain PDB written into ``tmp_path``.

The sign convention is the thing most worth protecting here.  ``complex_pde``
is lower-is-better, so a positive scan delta must line up with a positive
SKEMPI DDG without any negation; getting it backwards inverts every
conclusion in the section (see commit f1a0a0c).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "26_vs_experiment.py"
_spec = importlib.util.spec_from_file_location("vs_experiment_26", str(_SCRIPT))
vs_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs_experiment)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ---------------------------------------------------------------------------
# Fixtures: a minimal 1BRS-shaped complex (chain A vs chain D) + SKEMPI rows
# ---------------------------------------------------------------------------

_CHAIN_A_SEQ = "ARNDCQEGHI"
_CHAIN_D_SEQ = "LKMFPSTWYV"


def _atom_line(serial: int, resname: str, chain: str, resseq: int, xyz) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
    )


def _write_pdb(path: Path) -> None:
    """Chain A along x at y=0; chain D along x at y=8, so distances vary."""
    lines = []
    serial = 1
    for i, aa in enumerate(_CHAIN_A_SEQ):
        lines.append(
            _atom_line(serial, _ONE_TO_THREE[aa], "A", i + 1, (3.8 * i, 0.0, 0.0))
        )
        serial += 1
    for i, aa in enumerate(_CHAIN_D_SEQ):
        lines.append(
            _atom_line(serial, _ONE_TO_THREE[aa], "D", i + 1, (3.8 * i, 8.0, 0.0))
        )
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_skempi(path: Path, rows: list[dict]) -> None:
    """Write the handful of SKEMPI 2.0 columns the loader actually reads."""
    frame = pd.DataFrame(
        [
            {
                "#Pdb": "1BRS_A_D",
                "Mutation(s)_PDB": r["mut"],
                "Mutation(s)_cleaned": r["mut"],
                "Affinity_mut_parsed": r["kd_mut"],
                "Affinity_wt_parsed": r["kd_wt"],
                "Temperature": "298",
                "Hold_out_type": "Pr/PI",
            }
            for r in rows
        ]
    )
    frame.to_csv(path, sep=";", index=False)


def _make_cache(tmp_path: Path, rows: list[dict]) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    _write_pdb(cache / "1brs.pdb")
    _write_skempi(cache / "skempi_v2.csv", rows)
    return cache


_DEFAULT_SKEMPI_ROWS = [
    # chain A, residue 1 (A) -> G, destabilising 100x
    {"mut": "AA1G", "kd_mut": 1e-7, "kd_wt": 1e-9},
    # same mutation measured again, slightly different Kd
    {"mut": "AA1G", "kd_mut": 2e-7, "kd_wt": 1e-9},
    # residue 3 (N) -> A, mildly destabilising
    {"mut": "NA3A", "kd_mut": 1e-8, "kd_wt": 1e-9},
    # residue 5 (C) -> A, neutral
    {"mut": "CA5A", "kd_mut": 1e-9, "kd_wt": 1e-9},
    # residue 7 (E) -> K, stabilising
    {"mut": "EA7K", "kd_mut": 1e-10, "kd_wt": 1e-9},
    # on the partner chain -- must be ignored for chain A
    {"mut": "LD1A", "kd_mut": 1e-7, "kd_wt": 1e-9},
]


def _write_scan(path: Path, pairs: dict[tuple[int, str], float], mode="resswap"):
    rows = [
        {
            "mutation_id": "WT", "row_index": -1, "sequence": _CHAIN_A_SEQ,
            "substitutions": "", "n_mut": 0, "binding_score": float("nan"),
            "model_score": 1.0, "wt_score": 1.0, "delta": 0.0, "mode": mode,
        }
    ]
    for i, ((pos, aa), delta) in enumerate(pairs.items()):
        rows.append(
            {
                "mutation_id": f"{pos}{aa}", "row_index": i,
                "sequence": _CHAIN_A_SEQ, "substitutions": f"{pos}{aa}",
                "n_mut": 1, "binding_score": float("nan"),
                "model_score": 1.0 + delta, "wt_score": 1.0,
                "delta": delta, "mode": mode,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_npz(path: Path, seq: str, grad: np.ndarray, method="ig") -> None:
    np.savez(
        path,
        grad_res_type=grad,
        aa_token_indices=np.arange(20, dtype=np.int64),
        wt_indices=np.array([AA_TO_IDX[c] for c in seq], dtype=np.intp),
        reference_seq=np.array(seq),
        method=np.array(method),
    )


# ---------------------------------------------------------------------------
# 1. Sign convention
# ---------------------------------------------------------------------------


class TestOrientation:
    _V = np.array([-1.0, 0.0, 2.0])

    def test_complex_pde_is_not_negated(self):
        # Predicted distance error: lower is a better complex, so a rising
        # score already means "destabilising", same as a positive SKEMPI DDG.
        out = vs_experiment.orient_for_experiment(self._V, "complex_pde")
        np.testing.assert_array_equal(out, self._V)

    def test_affinity_pred_value_is_not_negated(self):
        out = vs_experiment.orient_for_experiment(self._V, "affinity_pred_value")
        np.testing.assert_array_equal(out, self._V)

    @pytest.mark.parametrize("score", ["iptm", "ptm", "complex_iplddt", "protein_iptm"])
    def test_higher_is_better_scores_are_negated(self, score):
        out = vs_experiment.orient_for_experiment(self._V, score)
        np.testing.assert_array_equal(out, -self._V)

    def test_lower_is_better_set_matches_stage_12(self):
        spec = importlib.util.spec_from_file_location(
            "compare_12_for_26", _ROOT / "scripts" / "12_compare.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert vs_experiment.LOWER_IS_BETTER == mod.LOWER_IS_BETTER

    def test_input_is_not_mutated(self):
        v = np.array([1.0, 2.0])
        vs_experiment.orient_for_experiment(v, "iptm")
        np.testing.assert_array_equal(v, np.array([1.0, 2.0]))


class TestOrientationAnchor:
    def test_detects_correct_orientation(self):
        ddg = np.array([3.0, 2.5, 0.1, 0.0, -0.2])
        delta = np.array([0.05, 0.04, 0.001, 0.0, -0.001])
        out = vs_experiment.orientation_anchor(delta, ddg)
        assert out["sign_as_expected"] is True
        assert out["n_hot"] == 2
        assert out["n_neutral"] == 3
        assert out["separation"] > 0

    def test_detects_inverted_orientation(self):
        ddg = np.array([3.0, 2.5, 0.1, 0.0, -0.2])
        delta = np.array([-0.05, -0.04, 0.001, 0.0, 0.002])
        out = vs_experiment.orientation_anchor(delta, ddg)
        assert out["sign_as_expected"] is False
        assert out["separation"] < 0

    def test_no_hot_spots_reports_none(self):
        out = vs_experiment.orientation_anchor(
            np.array([0.1, 0.2]), np.array([0.0, 0.1])
        )
        assert out["sign_as_expected"] is None


# ---------------------------------------------------------------------------
# 2. Loading and joining
# ---------------------------------------------------------------------------


class TestLoadExperimentalDdg:
    def test_keeps_only_the_requested_chain(self, tmp_path):
        cache = _make_cache(tmp_path, _DEFAULT_SKEMPI_ROWS)
        agg, raw, diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        assert set(agg["mut_aa"]) == {"G", "A", "K"}
        assert diag["n_rows_other_chain"] == 1
        assert diag["n_unique_mutations"] == 4

    def test_maps_author_numbering_to_zero_based_index(self, tmp_path):
        cache = _make_cache(tmp_path, _DEFAULT_SKEMPI_ROWS)
        agg, _raw, _diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        row = agg[agg["mut_aa"] == "G"].iloc[0]
        assert row["pos"] == 0  # author residue 1 -> index 0
        assert row["wt_aa"] == "A"
        assert row["residue_id"] == "1"

    def test_ddg_sign_is_positive_for_weaker_binding(self, tmp_path):
        cache = _make_cache(tmp_path, _DEFAULT_SKEMPI_ROWS)
        agg, _raw, _diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        weaker = agg[agg["mut_aa"] == "G"].iloc[0]["ddg"]
        tighter = agg[agg["mut_aa"] == "K"].iloc[0]["ddg"]
        assert weaker > 0
        assert tighter < 0

    def test_duplicate_measurements_collapse_to_median(self, tmp_path):
        cache = _make_cache(tmp_path, _DEFAULT_SKEMPI_ROWS)
        agg, raw, diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        row = agg[agg["mut_aa"] == "G"].iloc[0]
        assert row["n_measurements"] == 2
        assert row["ddg_min"] < row["ddg"] < row["ddg_max"]
        assert diag["n_replicated_mutations"] == 1
        assert len(raw) == 5  # chain-A measurements, before collapsing

    def test_wild_type_mismatch_is_skipped_by_default(self, tmp_path):
        rows = _DEFAULT_SKEMPI_ROWS + [{"mut": "WA3A", "kd_mut": 1e-7, "kd_wt": 1e-9}]
        cache = _make_cache(tmp_path, rows)
        _agg, _raw, diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        assert diag["n_wild_type_mismatch"] == 1

    def test_wild_type_mismatch_can_be_fatal(self, tmp_path):
        rows = _DEFAULT_SKEMPI_ROWS + [{"mut": "WA3A", "kd_mut": 1e-7, "kd_wt": 1e-9}]
        cache = _make_cache(tmp_path, rows)
        with pytest.raises(SystemExit):
            vs_experiment.load_experimental_ddg(
                "1BRS", "A", cache, allow_mismatch=False
            )

    def test_tie_diagnostics_count_flat_ddg(self, tmp_path):
        rows = [
            {"mut": "AA1G", "kd_mut": 1e-9, "kd_wt": 1e-9},
            {"mut": "NA3A", "kd_mut": 1e-9, "kd_wt": 1e-9},
            {"mut": "CA5A", "kd_mut": 1e-7, "kd_wt": 1e-9},
        ]
        cache = _make_cache(tmp_path, rows)
        _agg, _raw, diag = vs_experiment.load_experimental_ddg("1BRS", "A", cache)
        assert diag["n_exactly_zero_ddg"] == 2
        assert diag["largest_tie_group"] == 2
        assert diag["n_distinct_ddg_values"] == 2


class TestLoadScanDeltas:
    def test_drops_wt_row_and_parses_keys(self, tmp_path):
        scan = tmp_path / "scan.csv"
        _write_scan(scan, {(0, "G"): 0.03, (2, "A"): 0.01})
        df, info = vs_experiment.load_scan_deltas(scan)
        assert len(df) == 2
        assert set(df["pos"]) == {0, 2}
        assert info["n_scan_mutants"] == 2
        assert info["scan_modes"] == ["resswap"]

    def test_multi_digit_positions_parse(self, tmp_path):
        scan = tmp_path / "scan.csv"
        _write_scan(scan, {(149, "A"): 0.02})
        df, _info = vs_experiment.load_scan_deltas(scan)
        assert df.iloc[0]["pos"] == 149
        assert df.iloc[0]["mut_aa"] == "A"

    def test_non_finite_deltas_are_dropped(self, tmp_path):
        scan = tmp_path / "scan.csv"
        _write_scan(scan, {(0, "G"): float("nan"), (2, "A"): 0.01})
        df, info = vs_experiment.load_scan_deltas(scan)
        assert info["n_scan_mutants"] == 2
        assert info["n_scan_scored"] == 1
        assert df.iloc[0]["pos"] == 2

    def test_missing_column_exits(self, tmp_path):
        scan = tmp_path / "bad.csv"
        pd.DataFrame({"mutation_id": ["0A"]}).to_csv(scan, index=False)
        with pytest.raises(SystemExit):
            vs_experiment.load_scan_deltas(scan)


class TestAttributionDeltas:
    def test_computes_mut_minus_wt(self, tmp_path):
        grad = np.zeros((len(_CHAIN_A_SEQ), 20), dtype=np.float32)
        grad[0, AA_TO_IDX["G"]] = 5.0
        grad[0, AA_TO_IDX[_CHAIN_A_SEQ[0]]] = 2.0
        npz = tmp_path / "a.npz"
        _write_npz(npz, _CHAIN_A_SEQ, grad)
        vals, method = vs_experiment.attribution_deltas(npz, [0], ["G"])
        assert vals[0] == pytest.approx(3.0)
        assert method == "ig"

    def test_out_of_range_position_is_nan(self, tmp_path):
        grad = np.zeros((len(_CHAIN_A_SEQ), 20), dtype=np.float32)
        npz = tmp_path / "a.npz"
        _write_npz(npz, _CHAIN_A_SEQ, grad)
        vals, _ = vs_experiment.attribution_deltas(npz, [999], ["G"])
        assert np.isnan(vals[0])

    def test_non_canonical_residue_is_nan(self, tmp_path):
        grad = np.zeros((len(_CHAIN_A_SEQ), 20), dtype=np.float32)
        npz = tmp_path / "a.npz"
        _write_npz(npz, _CHAIN_A_SEQ, grad)
        vals, _ = vs_experiment.attribution_deltas(npz, [0], ["X"])
        assert np.isnan(vals[0])

    def test_missing_key_exits(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(npz, method=np.array("ig"))
        with pytest.raises(SystemExit):
            vs_experiment.attribution_deltas(npz, [0], ["G"])


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------


class TestCorrelationBlock:
    def test_perfect_monotone_gives_spearman_one(self):
        x = np.arange(20.0)
        block = vs_experiment.correlation_block(x, x**2, n_perms=200, n_boot=100)
        assert block["spearman"] == pytest.approx(1.0)
        assert block["n"] == 20
        assert block["permutation_null"]["p_value_two_sided"] < 0.05

    def test_anticorrelation_is_negative(self):
        x = np.arange(20.0)
        block = vs_experiment.correlation_block(x, -x, n_perms=200, n_boot=100)
        assert block["spearman"] == pytest.approx(-1.0)

    def test_distance_partial_is_reported(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=40)
        x = d + 0.01 * rng.normal(size=40)
        y = d + 0.01 * rng.normal(size=40)
        block = vs_experiment.correlation_block(x, y, d, n_perms=200, n_boot=100)
        assert abs(block["partial_spearman_distance"]) < abs(block["spearman"])
        assert len(block["partial_spearman_distance_ci"]) == 2

    def test_too_few_points_is_flagged_underpowered(self):
        block = vs_experiment.correlation_block(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
        assert block["underpowered"] is True
        assert block["n"] == 2

    def test_non_finite_pairs_are_dropped(self):
        x = np.array([1.0, 2.0, 3.0, np.nan])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        block = vs_experiment.correlation_block(x, y, n_perms=100, n_boot=50)
        assert block["n"] == 3


class TestPermutationNull:
    def test_strong_signal_gives_small_p(self):
        x = np.arange(30.0)
        out = vs_experiment.permutation_null(x, x, n_perms=500)
        assert out["p_value_two_sided"] < 0.01
        assert out["p5"] < 0 < out["p95"]

    def test_noise_gives_large_p(self):
        rng = np.random.default_rng(1)
        out = vs_experiment.permutation_null(
            rng.normal(size=40), rng.normal(size=40), n_perms=500
        )
        assert out["p_value_two_sided"] > 0.05


class TestExperimentalReliability:
    def test_identical_replicates_give_ceiling_of_one(self):
        raw = pd.DataFrame(
            {
                "pos": [0, 0, 1, 1, 2, 2, 3, 3],
                "mut_aa": list("AABBCCDD"),
                "ddg_raw": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            }
        )
        out = vs_experiment.experimental_reliability(raw, n_draws=20)
        assert out["test_retest_spearman_median"] == pytest.approx(1.0)
        assert out["n_replicated_mutations"] == 4

    def test_noisy_replicates_lower_the_ceiling(self):
        rng = np.random.default_rng(3)
        rows = []
        for p in range(10):
            true = float(p)
            for _ in range(2):
                rows.append(
                    {"pos": p, "mut_aa": "A", "ddg_raw": true + rng.normal(0, 8.0)}
                )
        raw = pd.DataFrame(rows)
        out = vs_experiment.experimental_reliability(raw, n_draws=50)
        assert out["test_retest_spearman_median"] < 0.9

    def test_no_replicates_returns_none(self):
        raw = pd.DataFrame(
            {"pos": [0, 1], "mut_aa": ["A", "B"], "ddg_raw": [1.0, 2.0]}
        )
        out = vs_experiment.experimental_reliability(raw)
        assert out["test_retest_spearman_median"] is None


# ---------------------------------------------------------------------------
# 4. End to end
# ---------------------------------------------------------------------------


def _run_main(tmp_path, scan_pairs, grad, out_name="out.json", extra=()):
    cache = _make_cache(tmp_path, _DEFAULT_SKEMPI_ROWS)
    scan = tmp_path / "scan.csv"
    _write_scan(scan, scan_pairs)
    npz = tmp_path / "attr.npz"
    _write_npz(npz, _CHAIN_A_SEQ, grad)
    out_path = tmp_path / out_name
    argv = [
        "--dataset", "1BRS",
        "--chain", "A",
        "--scan", str(scan),
        "--attribution", f"ig_test={npz}",
        "--cache-dir", str(cache),
        "--n-perms", "200",
        "--n-boot", "100",
        "--out", str(out_path),
        *extra,
    ]
    vs_experiment.main(argv)
    with open(out_path) as f:
        return json.load(f), out_path


class TestMainEndToEnd:
    def _grad(self):
        grad = np.zeros((len(_CHAIN_A_SEQ), 20), dtype=np.float32)
        grad[0, AA_TO_IDX["G"]] = 3.0
        grad[2, AA_TO_IDX["A"]] = 1.0
        grad[6, AA_TO_IDX["K"]] = -1.0
        return grad

    def _pairs(self):
        # Deltas deliberately ordered like the DDGs: A1G worst, E7K stabilising.
        return {(0, "G"): 0.05, (2, "A"): 0.02, (4, "A"): 0.0, (6, "K"): -0.01}

    def test_writes_json_with_both_comparisons(self, tmp_path):
        data, _ = _run_main(tmp_path, self._pairs(), self._grad())
        assert data["n_joined"] == 4
        assert data["model_vs_lab"]["signed"]["spearman"] == pytest.approx(1.0)
        assert data["shortcut_vs_lab"]["ig_test"]["signed"]["spearman"] > 0.5
        assert "magnitude" in data["model_vs_lab"]
        assert data["orientation"]["negated_for_experiment"] is False

    def test_orientation_anchor_passes_on_agreeing_data(self, tmp_path):
        data, _ = _run_main(tmp_path, self._pairs(), self._grad())
        assert data["orientation_anchor"]["sign_as_expected"] is True

    def test_inverted_scan_is_caught_by_the_anchor(self, tmp_path):
        flipped = {k: -v for k, v in self._pairs().items()}
        data, _ = _run_main(tmp_path, flipped, self._grad())
        assert data["orientation_anchor"]["sign_as_expected"] is False
        assert data["model_vs_lab"]["signed"]["spearman"] < 0

    def test_bound_check_records_whether_a_exceeds_b(self, tmp_path):
        data, _ = _run_main(tmp_path, self._pairs(), self._grad())
        bound = data["bound_check"]["ig_test"]
        assert bound["a_vs_c_spearman"] <= bound["b_vs_c_spearman"] + 1e-12
        assert bound["a_exceeds_b"] is False

    def test_distance_partial_present_for_both_comparisons(self, tmp_path):
        data, _ = _run_main(tmp_path, self._pairs(), self._grad())
        assert "partial_spearman_distance" in data["model_vs_lab"]["signed"]
        assert (
            "partial_spearman_distance"
            in data["shortcut_vs_lab"]["ig_test"]["signed"]
        )

    def test_empty_join_is_reported_not_crashed(self, tmp_path):
        data, _ = _run_main(tmp_path, {(9, "W"): 0.1}, self._grad())
        assert data["n_joined"] == 0
        assert data["underpowered"] is True

    def test_prov_sidecar_is_written(self, tmp_path):
        _data, out_path = _run_main(tmp_path, self._pairs(), self._grad())
        prov = Path(str(out_path) + ".prov.json")
        assert prov.exists()
        with open(prov) as f:
            rec = json.load(f)
        assert rec["stage"] == "26_vs_experiment"
        assert rec["arm"]["dataset"] == "1BRS"

    def test_per_mutation_rows_carry_provenance_of_the_measurement(self, tmp_path):
        data, _ = _run_main(tmp_path, self._pairs(), self._grad())
        row = [r for r in data["per_mutation"] if r["mut_aa"] == "G"][0]
        assert row["n_measurements"] == 2
        assert row["ddg_spread"] > 0
        assert row["residue_id"] == "1"
        assert row["distance_to_partner"] > 0
