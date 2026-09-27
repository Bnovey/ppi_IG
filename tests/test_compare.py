"""Tests for scripts/12_compare.py.

Covers sign conventions, join logic, rank-correlation invariance,
DMS sign handling, provenance reading, and end-to-end smoke.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from igv.metrics import spearman  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "12_compare.py"


def _load_stage12():
    """Import 12_compare.py by path — the module name starts with a digit."""
    spec = importlib.util.spec_from_file_location("igv_stage12_compare", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compare = _load_stage12()


# ---------------------------------------------------------------------------
# 1. _orient_for_experiment — sign convention
# ---------------------------------------------------------------------------


class TestOrientForExperiment:
    """Each of the six score names is tested explicitly — not via a loop over LOWER_IS_BETTER.

    Looping over the module's own LOWER_IS_BETTER set would make the test tautological:
    if the set were wrong, the test would still pass.  Hardcoding the expectation per
    score name ensures we'd catch a mis-classified score.
    """

    _V = np.array([0.5, 1.0, 2.0])

    def test_complex_pde_lower_is_better_passes_through_unchanged(self):
        # Rising PDE = worse binding = positive DDG: no sign flip needed.
        result = compare._orient_for_experiment(self._V, "complex_pde")
        np.testing.assert_array_equal(result, self._V)

    def test_complex_iplddt_higher_is_better_is_negated(self):
        result = compare._orient_for_experiment(self._V, "complex_iplddt")
        np.testing.assert_array_equal(result, -self._V)

    def test_complex_plddt_higher_is_better_is_negated(self):
        result = compare._orient_for_experiment(self._V, "complex_plddt")
        np.testing.assert_array_equal(result, -self._V)

    def test_iptm_higher_is_better_is_negated(self):
        result = compare._orient_for_experiment(self._V, "iptm")
        np.testing.assert_array_equal(result, -self._V)

    def test_ptm_higher_is_better_is_negated(self):
        result = compare._orient_for_experiment(self._V, "ptm")
        np.testing.assert_array_equal(result, -self._V)

    def test_protein_iptm_higher_is_better_is_negated(self):
        result = compare._orient_for_experiment(self._V, "protein_iptm")
        np.testing.assert_array_equal(result, -self._V)

    def test_negation_does_not_mutate_input_array(self):
        v = np.array([1.0, 2.0, 3.0])
        original = v.copy()
        compare._orient_for_experiment(v, "iptm")
        np.testing.assert_array_equal(v, original)

    def test_complex_pde_with_negative_values_passes_through(self):
        # Negative values must also pass through unchanged for complex_pde.
        v = np.array([-1.0, 0.0, 1.0])
        result = compare._orient_for_experiment(v, "complex_pde")
        np.testing.assert_array_equal(result, v)


# ---------------------------------------------------------------------------
# 2. _scan_key / _pred_key / _merge_scan_pred — join logic
# ---------------------------------------------------------------------------


def _scan_series(sub: str, score: float = 1.0) -> pd.Series:
    return pd.Series({"substitutions": sub, "model_score": score})


def _pred_series(pos: int, aa: str, delta: float = 1.0) -> pd.Series:
    return pd.Series({"position": pos, "mut_aa": aa, "score_delta": delta})


class TestScanKey:
    def test_two_digit_position_intact(self):
        assert compare._scan_key(_scan_series("84A")) == "84A"

    def test_three_digit_position_intact(self):
        assert compare._scan_key(_scan_series("104W")) == "104W"

    def test_leading_and_trailing_whitespace_stripped(self):
        assert compare._scan_key(_scan_series("  84A  ")) == "84A"


class TestPredKey:
    def test_two_digit_position(self):
        assert compare._pred_key(_pred_series(84, "A")) == "84A"

    def test_three_digit_position(self):
        # position=104 must produce "104", not "10" (fixed-width would fail).
        assert compare._pred_key(_pred_series(104, "W")) == "104W"

    def test_zero_position(self):
        assert compare._pred_key(_pred_series(0, "G")) == "0G"


class TestMergeScanPred:
    """Tests for _merge_scan_pred inner join and asymmetry counts."""

    @staticmethod
    def _scan(subs, scores=None):
        n = len(subs)
        return pd.DataFrame(
            {
                "substitutions": subs,
                "model_score": scores if scores is not None else [1.0] * n,
            }
        )

    @staticmethod
    def _pred(positions, aas, deltas=None):
        n = len(positions)
        return pd.DataFrame(
            {
                "position": positions,
                "mut_aa": aas,
                "score_delta": deltas if deltas is not None else [1.0] * n,
            }
        )

    def test_correct_pairing_simple(self):
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(
            self._scan(["84A"]), self._pred([84], ["A"])
        )
        assert len(merged) == 1
        assert n_pred_only == 0
        assert n_scan_only == 0

    def test_three_digit_position_is_correctly_joined(self):
        """'104W' in scan must join with position=104, mut_aa='W' in pred."""
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(
            self._scan(["104W"]), self._pred([104], ["W"])
        )
        assert len(merged) == 1
        assert n_pred_only == 0
        assert n_scan_only == 0

    def test_position_10_does_not_match_position_104(self):
        """'10W' and '104W' are different keys and must not match."""
        merged, _, _ = compare._merge_scan_pred(
            self._scan(["10W"]), self._pred([104], ["W"])
        )
        assert len(merged) == 0

    def test_rows_in_scan_only_are_excluded(self):
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(
            self._scan(["84A", "85G"]), self._pred([84], ["A"])
        )
        assert len(merged) == 1
        assert n_scan_only == 1
        assert n_pred_only == 0

    def test_rows_in_pred_only_are_excluded(self):
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(
            self._scan(["84A"]), self._pred([84, 85], ["A", "G"])
        )
        assert len(merged) == 1
        assert n_pred_only == 1
        assert n_scan_only == 0

    def test_overlap_asymmetry_counts_match_deliberately_mismatched_pair(self):
        """3 scan entries, 2 pred entries, only 1 overlapping — counts must be exact."""
        scan = self._scan(["84A", "85G", "86T"])
        pred = self._pred([84, 87], ["A", "K"])
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(scan, pred)
        assert len(merged) == 1
        assert n_pred_only == 1   # 87K in pred only
        assert n_scan_only == 2   # 85G and 86T in scan only

    def test_merged_contains_both_score_columns(self):
        scan = self._scan(["84A"], [2.5])
        pred = self._pred([84], ["A"], [0.7])
        merged, _, _ = compare._merge_scan_pred(scan, pred)
        assert "model_score" in merged.columns
        assert "score_delta" in merged.columns
        assert merged.iloc[0]["model_score"] == pytest.approx(2.5)
        assert merged.iloc[0]["score_delta"] == pytest.approx(0.7)

    def test_empty_scan_gives_empty_merge(self):
        merged, n_pred_only, n_scan_only = compare._merge_scan_pred(
            self._scan([]), self._pred([84], ["A"])
        )
        assert len(merged) == 0
        assert n_pred_only == 1
        assert n_scan_only == 0


# ---------------------------------------------------------------------------
# 3. Rank-correlation invariance
# ---------------------------------------------------------------------------


class TestRankCorrelationInvariance:
    """Spearman is rank-based; additive shifts and monotone transforms must not change it."""

    _DELTA = np.array([0.1, -0.3, 0.5, -0.1, 0.2, -0.4, 0.8])
    _MODEL = np.array([1.2, 0.8, 1.5, 0.9, 1.1, 0.6, 1.8])

    def test_perfectly_agreeing_pair_gives_plus_one(self):
        x = np.arange(1, 6, dtype=float)
        assert spearman(x, x) == pytest.approx(1.0)

    def test_inverted_pair_gives_minus_one(self):
        x = np.arange(1, 6, dtype=float)
        assert spearman(x, -x) == pytest.approx(-1.0)

    def test_additive_shift_on_model_score_leaves_correlation_unchanged(self):
        rho_original = spearman(self._DELTA, self._MODEL)
        rho_shifted = spearman(self._DELTA, self._MODEL + 1000.0)
        assert rho_original == pytest.approx(rho_shifted)

    def test_additive_shift_on_score_delta_leaves_correlation_unchanged(self):
        rho_original = spearman(self._DELTA, self._MODEL)
        rho_shifted = spearman(self._DELTA + 1000.0, self._MODEL)
        assert rho_original == pytest.approx(rho_shifted)

    def test_strictly_monotone_increasing_transform_leaves_correlation_unchanged(self):
        # exp() is strictly monotone increasing; Spearman must be invariant.
        rho_original = spearman(self._DELTA, self._MODEL)
        rho_transformed = spearman(np.exp(self._DELTA), self._MODEL)
        assert rho_original == pytest.approx(rho_transformed)

    def test_both_sides_shifted_simultaneously_unchanged(self):
        rho_original = spearman(self._DELTA, self._MODEL)
        rho_shifted = spearman(self._DELTA + 5.0, self._MODEL + 5.0)
        assert rho_original == pytest.approx(rho_shifted)


# ---------------------------------------------------------------------------
# 4. _load_experiment_dms — sign handling
# ---------------------------------------------------------------------------


def _make_dms_monkeypatches(monkeypatch, tmp_path, bind_avg: float):
    """Wire up the four inner imports so _load_experiment_dms runs without I/O."""
    import igv.dms as dms_mod
    import igv.data as data_mod
    import igv.skempi as skempi_mod
    from igv.dms import DmsComplex

    fake_cx = DmsComplex(pdb_id="6M0J", mutated_chain="E", partner_chains=("A",))
    synthetic = pd.DataFrame(
        {
            "site_SARS2": [417],
            "mutant": ["A"],
            "wildtype": ["K"],
            "bind_avg": [bind_avg],
        }
    )

    monkeypatch.setattr(dms_mod, "get_complex", lambda key: fake_cx)
    monkeypatch.setattr(
        data_mod, "download_rcsb", lambda pdb_id, cache_dir: tmp_path / "fake.pdb"
    )
    monkeypatch.setattr(
        skempi_mod,
        "read_pdb_residue_ids",
        lambda path: ({"E": ["417"]}, {"E": "K"}),
    )
    monkeypatch.setattr(dms_mod, "load_starr2020", lambda cache_dir: synthetic)
    monkeypatch.setattr(dms_mod, "singles", lambda df: df)
    monkeypatch.setattr(
        dms_mod,
        "map_sites_to_indices",
        lambda df, chain_ids, chain_seq, allow_mismatch: ({417: 0}, []),
    )


class TestLoadExperimentDmsSign:
    """Sign convention: Starr bind_avg and SKEMPI DDG must map onto the same axis."""

    def test_positive_bind_avg_gives_negative_ddg(self, monkeypatch, tmp_path):
        """bind_avg=+2 (tighter binding) → ddg < 0 (stabilising on SKEMPI convention)."""
        _make_dms_monkeypatches(monkeypatch, tmp_path, bind_avg=2.0)
        result = compare._load_experiment_dms("spike_rbd", tmp_path)
        assert result is not None
        ddg = float(result.iloc[0]["ddg"])
        assert ddg < 0, (
            f"positive bind_avg=2.0 (tighter binding) must give negative ddg; got {ddg}"
        )

    def test_negative_bind_avg_gives_positive_ddg(self, monkeypatch, tmp_path):
        """bind_avg=-3 (weaker binding) → ddg > 0 (destabilising on SKEMPI convention)."""
        _make_dms_monkeypatches(monkeypatch, tmp_path, bind_avg=-3.0)
        result = compare._load_experiment_dms("spike_rbd", tmp_path)
        assert result is not None
        ddg = float(result.iloc[0]["ddg"])
        assert ddg > 0, (
            f"negative bind_avg=-3.0 (weaker binding) must give positive ddg; got {ddg}"
        )

    def test_result_columns_are_position_mut_aa_ddg(self, monkeypatch, tmp_path):
        _make_dms_monkeypatches(monkeypatch, tmp_path, bind_avg=1.0)
        result = compare._load_experiment_dms("spike_rbd", tmp_path)
        assert result is not None
        assert {"position", "mut_aa", "ddg"} <= set(result.columns)

    def test_position_is_zero_based_index_not_residue_id(self, monkeypatch, tmp_path):
        """map_sites_to_indices maps site 417 → index 0; result must carry 0, not 417."""
        _make_dms_monkeypatches(monkeypatch, tmp_path, bind_avg=1.0)
        result = compare._load_experiment_dms("spike_rbd", tmp_path)
        assert result is not None
        assert int(result.iloc[0]["position"]) == 0

    def test_unknown_dataset_returns_none(self, monkeypatch, tmp_path):
        """Dataset not in DMS registry must return None without raising."""
        import igv.dms as dms_mod

        def _raise(key):
            raise KeyError(key)

        monkeypatch.setattr(dms_mod, "get_complex", _raise)
        result = compare._load_experiment_dms("not_a_real_dataset", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# 5. _get_backward_passes
# ---------------------------------------------------------------------------


class TestGetBackwardPasses:
    def test_returns_m_steps_from_provenance(self, tmp_path):
        from igv.provenance import write as prov_write

        pred_path = tmp_path / "pred.csv"
        pred_path.touch()
        prov_write(pred_path, stage="05_predict", arm={"m_steps": 16})
        assert compare._get_backward_passes(pred_path) == 16

    def test_returns_correct_integer_for_m_steps_1(self, tmp_path):
        from igv.provenance import write as prov_write

        pred_path = tmp_path / "pred.csv"
        pred_path.touch()
        prov_write(pred_path, stage="05_predict", arm={"m_steps": 1})
        assert compare._get_backward_passes(pred_path) == 1

    def test_returns_none_when_provenance_sidecar_absent(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        pred_path.touch()
        # No .prov.json written; FileNotFoundError must be caught internally.
        result = compare._get_backward_passes(pred_path)
        assert result is None

    def test_returns_none_when_m_steps_key_absent_from_arm(self, tmp_path):
        from igv.provenance import write as prov_write

        pred_path = tmp_path / "pred.csv"
        pred_path.touch()
        prov_write(pred_path, stage="05_predict", arm={"score": "complex_pde"})
        assert compare._get_backward_passes(pred_path) is None

    def test_none_does_not_crash_the_cost_section_in_main(self, tmp_path, monkeypatch, capsys):
        """When provenance is absent, main must print the fallback message, not raise."""
        scan_path = tmp_path / "scan.csv"
        pred_path = tmp_path / "pred.csv"
        out_path = tmp_path / "compare.csv"
        _write_minimal_scan(scan_path)
        _write_minimal_pred(pred_path)
        # No provenance sidecar → _get_backward_passes returns None.

        monkeypatch.setattr(compare, "_load_experiment_skempi", lambda *a, **kw: None)
        monkeypatch.setattr(compare, "_load_experiment_dms", lambda *a, **kw: None)
        monkeypatch.setattr(
            compare, "bootstrap_ci", lambda fn, *args, **kwargs: (fn(*args), -0.1, 0.1)
        )

        _call_main(
            ["--pred", str(pred_path), "--scan", str(scan_path),
             "--dataset", "test_dataset", "--out", str(out_path)]
        )
        captured = capsys.readouterr().out
        assert "unknown (no provenance)" in captured


# ---------------------------------------------------------------------------
# Helpers shared by the smoke tests
# ---------------------------------------------------------------------------

_N = 8
_POSITIONS = list(range(80, 80 + _N))
_AAS = list("ACDEFGHI")  # 8 characters


def _write_minimal_scan(path: Path, n: int = _N) -> None:
    positions = _POSITIONS[:n]
    aas = _AAS[:n]
    pd.DataFrame(
        {
            "row_index": range(n),
            "sequence": ["ACDEFACDEF"] * n,
            "substitutions": [f"{p}{a}" for p, a in zip(positions, aas)],
            "n_mut": [1] * n,
            "binding_score": [float("nan")] * n,
            "model_score": np.linspace(1.0, 2.0, n),
        }
    ).to_csv(path, index=False)


def _write_minimal_pred(path: Path, n: int = _N) -> None:
    positions = _POSITIONS[:n]
    aas = _AAS[:n]
    pd.DataFrame(
        {
            "position": positions,
            "mut_aa": aas,
            "score_delta": np.linspace(-1.0, 1.0, n),
        }
    ).to_csv(path, index=False)


def _call_main(extra_args: list[str]) -> None:
    old_argv = sys.argv
    sys.argv = ["12_compare.py"] + extra_args
    try:
        compare.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# 6. main — smoke test and error handling
# ---------------------------------------------------------------------------


class TestMain:
    def _setup(self, tmp_path, monkeypatch, n: int = _N):
        scan_path = tmp_path / "scan.csv"
        pred_path = tmp_path / "pred.csv"
        out_path = tmp_path / "compare.csv"
        _write_minimal_scan(scan_path, n=n)
        _write_minimal_pred(pred_path, n=n)

        monkeypatch.setattr(compare, "_load_experiment_skempi", lambda *a, **kw: None)
        monkeypatch.setattr(compare, "_load_experiment_dms", lambda *a, **kw: None)
        # Replace bootstrap_ci with a stub to keep each test sub-second.
        monkeypatch.setattr(
            compare, "bootstrap_ci", lambda fn, *args, **kwargs: (fn(*args), -0.1, 0.1)
        )
        return scan_path, pred_path, out_path

    def _run(self, tmp_path, monkeypatch, n: int = _N):
        scan_path, pred_path, out_path = self._setup(tmp_path, monkeypatch, n=n)
        _call_main(
            [
                "--pred", str(pred_path),
                "--scan", str(scan_path),
                "--dataset", "test_dataset",
                "--score", "complex_pde",
                "--out", str(out_path),
            ]
        )
        return out_path

    def test_smoke_writes_output_csv(self, tmp_path, monkeypatch):
        out_path = self._run(tmp_path, monkeypatch)
        assert out_path.exists()

    def test_output_has_required_columns(self, tmp_path, monkeypatch):
        out_path = self._run(tmp_path, monkeypatch)
        df = pd.read_csv(out_path)
        assert {"comparison", "scope", "statistic", "value"} <= set(df.columns)

    def test_output_contains_attribution_vs_scan_spearman_row(self, tmp_path, monkeypatch):
        out_path = self._run(tmp_path, monkeypatch)
        df = pd.read_csv(out_path)
        mask = (df["comparison"] == "attribution_vs_scan") & (df["statistic"] == "spearman")
        assert mask.sum() == 1, "expected exactly one attribution_vs_scan / spearman row"

    def test_output_has_no_duplicate_comparison_scope_statistic_tuples(
        self, tmp_path, monkeypatch
    ):
        """Each (comparison, scope, statistic) tuple must appear exactly once."""
        out_path = self._run(tmp_path, monkeypatch)
        df = pd.read_csv(out_path)
        dupes = df[["comparison", "scope", "statistic"]].duplicated()
        assert not dupes.any(), f"Duplicate rows found:\n{df[dupes]}"

    def test_n_overlap_recorded_correctly(self, tmp_path, monkeypatch):
        out_path = self._run(tmp_path, monkeypatch)
        df = pd.read_csv(out_path)
        row = df[(df["comparison"] == "attribution_vs_scan") & (df["statistic"] == "n")]
        assert len(row) == 1
        assert int(row.iloc[0]["value"]) == _N

    def test_fails_clearly_when_scan_missing_model_score(self, tmp_path):
        scan_path = tmp_path / "scan.csv"
        pred_path = tmp_path / "pred.csv"
        # scan CSV is missing model_score
        pd.DataFrame(
            {"row_index": [0], "substitutions": ["84A"], "n_mut": [1]}
        ).to_csv(scan_path, index=False)
        pd.DataFrame(
            {"position": [84], "mut_aa": ["A"], "score_delta": [0.1]}
        ).to_csv(pred_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            _call_main(
                [
                    "--pred", str(pred_path),
                    "--scan", str(scan_path),
                    "--dataset", "test_dataset",
                    "--out", str(tmp_path / "out.csv"),
                ]
            )
        assert "ERROR" in str(exc_info.value)

    def test_fails_clearly_when_pred_missing_score_delta(self, tmp_path):
        scan_path = tmp_path / "scan.csv"
        pred_path = tmp_path / "pred.csv"
        _write_minimal_scan(scan_path, n=4)
        # pred CSV is missing score_delta
        pd.DataFrame({"position": [80, 81], "mut_aa": ["A", "C"]}).to_csv(
            pred_path, index=False
        )

        with pytest.raises(SystemExit) as exc_info:
            _call_main(
                [
                    "--pred", str(pred_path),
                    "--scan", str(scan_path),
                    "--dataset", "test_dataset",
                    "--out", str(tmp_path / "out.csv"),
                ]
            )
        assert "ERROR" in str(exc_info.value)

    def test_fails_clearly_when_scan_missing_substitutions(self, tmp_path):
        scan_path = tmp_path / "scan.csv"
        pred_path = tmp_path / "pred.csv"
        # scan CSV is missing substitutions (the other required column)
        pd.DataFrame(
            {"row_index": [0], "model_score": [1.2], "n_mut": [1]}
        ).to_csv(scan_path, index=False)
        _write_minimal_pred(pred_path, n=2)

        with pytest.raises(SystemExit) as exc_info:
            _call_main(
                [
                    "--pred", str(pred_path),
                    "--scan", str(scan_path),
                    "--dataset", "test_dataset",
                    "--out", str(tmp_path / "out.csv"),
                ]
            )
        assert "ERROR" in str(exc_info.value)
