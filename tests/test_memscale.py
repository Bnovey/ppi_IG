"""Tests for scripts/08_memscale.py -- all CPU, no torch-CUDA, no boltz.

The script itself cannot run without a GPU, but the three things that decide
whether its output means anything are pure Python:

  * the ladder plan (whole-chain subsets, never truncated chains),
  * the completed-vs-truncated partition that feeds the fit,
  * the log-log fit itself.

The regression these guard is specific: a sweep that quietly folds OOM'd rows
into the fit produces a *lower* exponent and a *smaller* extrapolated
requirement -- i.e. it makes the model look affordable in exactly the direction
that gets a run scheduled and then killed.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_memscale.py"


def _load():
    """Import 08_memscale.py by path -- '08_memscale' is not a valid module name."""
    spec = importlib.util.spec_from_file_location("igv_memscale_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m8 = _load()


def _row(n_tokens, subset, peak, completed=True, oom=False, profile="large", error=None):
    row = {c: None for c in m8._COLUMNS}
    row.update(
        n_tokens=n_tokens,
        n_chains=len(subset),
        subset=subset,
        completed=completed,
        oom=oom,
        status="completed" if completed else ("oom" if oom else "error"),
        truncated=not completed,
        peak_allocated_gib=peak,
        chunk_profile=profile,
        error=error,
    )
    return row


# ---------------------------------------------------------------------------
# 1. Import safety -- this is the whole point of the laptop constraint
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_imports_without_torch_or_boltz(self):
        """The module must not pull in torch or boltz at import time.

        Asserted in a FRESH interpreter, because ``boltz`` raising ImportError
        in *this* process proves nothing about the module under test -- it only
        says boltz is absent from the laptop that ran the suite. On the GPU host
        boltz and torch are both installed, and the old form of this test
        (``pytest.raises(ImportError): __import__("boltz")``) failed there for
        that reason alone, i.e. it was red on the only machine where import
        weight actually costs anything.
        """
        code = (
            "import sys, importlib.util as u\n"
            f"sys.path.insert(0, {str(_SCRIPT.parents[1] / 'src')!r})\n"
            f"spec = u.spec_from_file_location('m8', {str(_SCRIPT)!r})\n"
            "mod = u.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert hasattr(mod, 'main')\n"
            "print('boltz' in sys.modules, 'torch' in sys.modules)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        pulled_boltz, pulled_torch = out.stdout.split()
        assert pulled_boltz == "False", "08_memscale imported boltz at module level"
        assert pulled_torch == "False", "08_memscale imported torch at module level"
        assert hasattr(m8, "main")

    def test_parser_builds_and_dry_run_is_a_flag(self):
        args = m8.build_parser().parse_args(["--dry-run"])
        assert args.dry_run is True
        # Defaults that must not drift: the sweep is only valid with one forced
        # profile, and recycling_steps must be pinned rather than inherited.
        assert args.chunk_profile == "large"
        assert args.recycling_steps == 1
        assert args.gradient_checkpointing is True
        assert args.msa_spec == "server"


# ---------------------------------------------------------------------------
# 2. The ladder
# ---------------------------------------------------------------------------


class TestLadder:
    def test_recorded_4fqi_lengths_sum_to_730_not_753(self):
        """ERRORS_LOG entry 12's chain lengths sum to 753; the real total is 730.

        read_pdb_chains on data/raw/4fqi_hlab.pdb gives A=324 B=176 H=121 L=109.
        """
        assert sum(m8.FQI_CHAIN_LENGTHS.values()) == 730
        assert m8.TARGET_TOKENS == 730

    def test_plan_is_ascending_whole_chain_subsets(self):
        points = m8.chain_subset_plan(m8.FQI_CHAIN_LENGTHS)
        sizes = [p["n_tokens"] for p in points]
        assert sizes == sorted(sizes)
        assert sizes == [230, 406, 500, 554, 730]
        # Every point is a union of whole chains, never a truncation: its token
        # count is exactly the sum of the chosen chains' full lengths.
        for p in points:
            assert p["n_tokens"] == sum(m8.FQI_CHAIN_LENGTHS[c] for c in p["chains"])
            assert p["n_chains"] == len(p["chains"])

    def test_ladder_has_three_points_above_the_chunk_threshold(self):
        """The fit needs a lever arm inside the large profile, not one point."""
        sizes = [p["n_tokens"] for p in m8.chain_subset_plan(m8.FQI_CHAIN_LENGTHS)]
        above = [s for s in sizes if s > m8.CHUNK_SIZE_THRESHOLD]
        assert len(above) >= 3, above

    def test_every_point_has_at_least_two_chains(self):
        """A single-chain point loses paired MSAs and the interface entirely."""
        for p in m8.chain_subset_plan(m8.FQI_CHAIN_LENGTHS):
            assert p["n_chains"] >= 2

    def test_missing_chain_is_skipped_not_mis_sized(self):
        points = m8.chain_subset_plan({"H": 121, "L": 109})
        assert [p["n_tokens"] for p in points] == [230]

    def test_select_sizes_rejects_unknown_size(self):
        points = m8.chain_subset_plan(m8.FQI_CHAIN_LENGTHS)
        with pytest.raises(SystemExit, match="not available in this ladder"):
            m8.select_sizes(points, [230, 999])

    def test_select_sizes_filters_and_sorts(self):
        points = m8.chain_subset_plan(m8.FQI_CHAIN_LENGTHS)
        got = m8.select_sizes(points, [554, 230])
        assert [p["n_tokens"] for p in got] == [230, 554]

    def test_cross_system_totals_are_recorded_without_a_guessed_mapping(self):
        """Only the locally verified structure->tokens pair may be asserted."""
        assert m8.VERIFIED_TOKEN_COUNTS == {"4fqi_hlab": 730}
        assert 730 in m8.CROSS_SYSTEM_MEASURED_TOTALS
        assert len(m8.CROSS_SYSTEM_MEASURED_TOTALS) == 13


# ---------------------------------------------------------------------------
# 3. The completed / truncated partition -- the load-bearing part
# ---------------------------------------------------------------------------


class TestPartition:
    def test_oom_rows_are_excluded_with_a_reason(self):
        rows = [
            _row(230, "HL", 2.0),
            _row(406, "HLB", 11.0),
            _row(730, "ABHL", 78.4, completed=False, oom=True),
        ]
        part = m8.partition_rows(rows)
        assert [r["n_tokens"] for r in part["usable"]] == [230, 406]
        assert len(part["excluded"]) == 1
        assert "lower bound" in part["excluded"][0]["exclusion_reason"]

    def test_non_oom_failures_are_excluded_too(self):
        rows = [_row(500, "AB", None, completed=False, oom=False, error="RuntimeError(boom)")]
        part = m8.partition_rows(rows)
        assert not part["usable"]
        assert "boom" in part["excluded"][0]["exclusion_reason"]

    def test_completed_row_without_a_peak_is_not_trusted(self):
        """No CUDA -> PeakMemory reports peak_gib=None; that is not a datum."""
        part = m8.partition_rows([_row(230, "HL", None)])
        assert not part["usable"]
        assert "no usable peak" in part["excluded"][0]["exclusion_reason"]

    def test_nan_peak_is_excluded(self):
        part = m8.partition_rows([_row(230, "HL", float("nan"))])
        assert not part["usable"]

    def test_mixed_profiles_refuse_to_fit(self):
        rows = [
            _row(230, "HL", 2.0, profile="small"),
            _row(406, "HLB", 11.0, profile="large"),
        ]
        with pytest.raises(ValueError, match="mixed chunk profiles"):
            m8.assert_single_profile(rows)

    def test_single_profile_is_accepted(self):
        rows = [_row(230, "HL", 2.0), _row(406, "HLB", 11.0)]
        assert m8.assert_single_profile(rows) == "large"


# ---------------------------------------------------------------------------
# 4. The fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_recovers_a_known_exponent(self):
        """peak = c * L^2.7 must come back as b = 2.7."""
        sizes = [230, 406, 500, 554]
        peaks = [1e-6 * L ** 2.7 for L in sizes]
        fit = m8.fit_loglog(sizes, peaks)
        assert fit["exponent_b"] == pytest.approx(2.7, abs=1e-6)
        assert fit["intercept_a"] == pytest.approx(math.log(1e-6), abs=1e-6)
        assert fit["r2"] == pytest.approx(1.0)
        assert fit["predicted_peak_gib_at_target"] == pytest.approx(
            1e-6 * 730 ** 2.7, rel=1e-9
        )
        assert fit["token_range"] == [230, 554]

    def test_r2_is_below_one_with_scatter(self):
        rng = np.random.default_rng(0)
        sizes = [230, 406, 500, 554, 648]
        peaks = [1e-6 * L ** 2.7 * math.exp(rng.normal(0, 0.05)) for L in sizes]
        fit = m8.fit_loglog(sizes, peaks)
        assert 0.5 < fit["r2"] < 1.0
        assert fit["residual_max_abs_log"] > 0

    def test_refuses_fewer_than_three_points(self):
        with pytest.raises(ValueError, match="need >= 3 points"):
            m8.fit_loglog([230, 406], [2.0, 11.0])

    def test_refuses_a_single_distinct_token_count(self):
        """Three rows at one L give an unidentifiable slope, not R^2 = 1."""
        with pytest.raises(ValueError, match="distinct token counts"):
            m8.fit_loglog([730, 730, 730], [70.0, 71.0, 72.0])

    def test_including_oom_rows_would_understate_the_requirement(self):
        """The exact failure this script exists to prevent, made concrete.

        With c = 2.5e-6 the true cost at 730 is ~135 GiB, so the recorded
        78.4 GiB OOM is a badly truncated lower bound. Folding it into the fit
        flattens the exponent and shrinks the extrapolation -- the dangerous
        direction, because it is the direction that gets a run scheduled.
        """
        sizes = [230, 406, 500, 554]
        peaks = [2.5e-6 * L ** 2.7 for L in sizes]
        honest = m8.fit_loglog(sizes, peaks)
        contaminated = m8.fit_loglog(sizes + [730], peaks + [78.4])
        assert honest["predicted_peak_gib_at_target"] > 100  # true requirement
        assert contaminated["exponent_b"] < honest["exponent_b"]
        assert (
            contaminated["predicted_peak_gib_at_target"]
            < honest["predicted_peak_gib_at_target"]
        )

    def test_partition_then_fit_is_what_the_script_does(self):
        rows = [_row(L, "x" * 2, 1e-6 * L ** 2.7) for L in (230, 406, 500, 554)]
        rows.append(_row(730, "ABHL", 78.4, completed=False, oom=True))
        usable = m8.partition_rows(rows)["usable"]
        fit = m8.fit_loglog(
            [r["n_tokens"] for r in usable], [r["peak_allocated_gib"] for r in usable]
        )
        assert fit["n_points"] == 4
        assert fit["exponent_b"] == pytest.approx(2.7, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Notes and schema -- the artifact has to say what it is
# ---------------------------------------------------------------------------


class TestArtifactHonesty:
    def test_notes_state_that_oom_rows_are_lower_bounds(self):
        notes = m8.build_notes("server", "chain-subset")
        assert "LOWER BOUND" in notes.upper()
        assert "excluded from the fit" in notes

    def test_notes_state_the_uncounted_cuda_context_overhead(self):
        notes = m8.build_notes("server", "chain-subset")
        assert "UNDERSTATE" in notes
        assert str(m8.CUDA_CONTEXT_OVERHEAD_GIB) in notes

    def test_notes_flag_the_msa_comparability_hazard(self):
        assert "NOT comparable" in m8.build_notes("empty", "chain-subset")
        assert "NOT comparable" not in m8.build_notes("server", "chain-subset")

    def test_notes_flag_the_chain_subset_confound(self):
        assert "paired MSAs" in m8.build_notes("server", "chain-subset")

    def test_notes_do_not_assert_the_literature_exponent_as_fact(self):
        notes = m8.build_notes("server", "chain-subset")
        assert "UNCONFIRMED" in notes

    def test_row_schema_carries_the_load_bearing_fields(self):
        for field in (
            "completed", "oom", "status", "truncated",
            "peak_allocated_gib", "peak_reserved_gib", "chunk_profile",
            "recycling_steps", "msa_depth", "msa_spec", "git_commit",
        ):
            assert field in m8._COLUMNS


# ---------------------------------------------------------------------------
# 6. Profile forcing
# ---------------------------------------------------------------------------


class TestProfileForcing:
    def test_sets_the_env_var(self, monkeypatch):
        monkeypatch.delenv("IGV_CHUNK_PROFILE", raising=False)
        m8.force_chunk_profile("large", [230, 730], strict=False)
        import os

        assert os.environ["IGV_CHUNK_PROFILE"] == "large"

    def test_forced_profile_is_l_invariant(self, monkeypatch):
        """The property the fit depends on: same config at 230 and at 730."""
        monkeypatch.setenv("IGV_CHUNK_PROFILE", "large")
        try:
            from igv.boltz_score import chunk_profile
        except ImportError:
            pytest.skip("chunk_profile not available in this tree")
        assert chunk_profile(230) == chunk_profile(730)
        # And that it really is the >threshold branch, not the small one.
        assert chunk_profile(230) == chunk_profile(230, profile="large")
        assert chunk_profile(230) != chunk_profile(230, profile="small")

    def test_unforced_profile_differs_across_the_threshold(self, monkeypatch):
        """Why forcing is mandatory: two algorithms either side of 384."""
        monkeypatch.delenv("IGV_CHUNK_PROFILE", raising=False)
        try:
            from igv.boltz_score import chunk_profile
        except ImportError:
            pytest.skip("chunk_profile not available in this tree")
        assert chunk_profile(230) != chunk_profile(730)


# ---------------------------------------------------------------------------
# 7. End-to-end --dry-run, which is the only executable path on a laptop
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_zero_and_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        out = tmp_path / "memscale_{dataset}_{score}.csv"
        rc = m8.main(["--dry-run", "--out", str(out)])
        assert rc == 0
        assert not list(tmp_path.iterdir())

    def test_dry_run_does_not_rewrite_an_existing_artifact(self, tmp_path, monkeypatch):
        """A dry run must never touch results, even when the sweep looks done."""
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        csv = tmp_path / "memscale_4fqi_h1_complex_pde.csv"
        import pandas as pd

        rows = [
            _row(L, s, 1e-6 * L ** 2.7)
            for L, s in ((230, "HL"), (406, "HLB"), (500, "AB"), (554, "HLA"))
        ]
        rows.append(_row(730, "ABHL", 78.4, completed=False, oom=True))
        pd.DataFrame(rows, columns=m8._COLUMNS).to_csv(csv, index=False)
        before = sorted(p.name for p in tmp_path.iterdir())
        rc = m8.main(["--dry-run", "--out", str(tmp_path / "memscale_{dataset}_{score}.csv")])
        assert rc == 0
        assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# 8. CSV round-trip hazards
# ---------------------------------------------------------------------------


class TestBooleanRoundTrip:
    def test_string_false_is_not_true(self):
        """bool("False") is True -- the misread that would turn every OOM row
        into a finished measurement and report the card's size as the model's
        requirement."""
        assert bool("False") is True  # the trap itself
        assert m8.as_bool("False") is False
        assert m8.as_bool("false") is False
        assert m8.as_bool("True") is True

    @pytest.mark.parametrize(
        "value,expected",
        [(True, True), (False, False), (np.bool_(True), True), (1, True), (0, False),
         (None, False), (float("nan"), False), ("", False), ("0", False), ("1", True)],
    )
    def test_recognised_forms(self, value, expected):
        assert m8.as_bool(value) is expected

    @pytest.mark.parametrize("value", ["yes", "maybe", 2, "completed"])
    def test_unrecognised_forms_raise_rather_than_guess(self, value):
        with pytest.raises(ValueError, match="boolean"):
            m8.as_bool(value)

    def test_partition_survives_a_string_valued_csv(self, tmp_path):
        """A CSV whose booleans came back as strings must still exclude the OOM."""
        import pandas as pd

        rows = [_row(230, "HL", 2.0), _row(730, "ABHL", 78.4, completed=False, oom=True)]
        csv = tmp_path / "rows.csv"
        frame = pd.DataFrame(rows, columns=m8._COLUMNS)
        frame["completed"] = frame["completed"].astype(str)
        frame["oom"] = frame["oom"].astype(str)
        frame.to_csv(csv, index=False)
        back = pd.read_csv(csv, dtype={"completed": str, "oom": str}).to_dict("records")
        part = m8.partition_rows(back)
        assert [r["n_tokens"] for r in part["usable"]] == [230]
        assert "lower bound" in part["excluded"][0]["exclusion_reason"]


class TestJsonSafe:
    def test_nan_becomes_null_not_a_number(self):
        """json.dumps emits bare NaN, which is not valid JSON; a missing peak
        must read as null rather than as a value."""
        import json

        text = json.dumps(m8.json_safe({"peak": float("nan")}), allow_nan=False)
        assert json.loads(text) == {"peak": None}

    def test_inf_becomes_null(self):
        assert m8.json_safe({"x": float("inf")}) == {"x": None}

    def test_numpy_scalars_are_converted(self):
        out = m8.json_safe({"i": np.int64(3), "f": np.float64(1.5), "b": np.bool_(False)})
        assert out == {"i": 3, "f": 1.5, "b": False}
        assert isinstance(out["i"], int) and isinstance(out["b"], bool)

    def test_nested_structures(self):
        out = m8.json_safe({"rows": [{"peak": np.float64("nan")}], "t": (1, 2)})
        assert out == {"rows": [{"peak": None}], "t": [1, 2]}


class TestPointLabel:
    def test_chain_subset_label_is_the_chain_ids(self):
        assert m8.point_label({"chains": ("H", "L"), "n_tokens": 230}) == "HL"

    def test_cross_system_label_is_the_structure_name(self):
        """Two structures can share a token count AND a chain-id set, so the
        resume key must be the structure, not the chains."""
        point = {"chains": ("A", "B", "C"), "n_tokens": 245, "structure": "1mlc_bae"}
        assert m8.point_label(point) == "1mlc_bae"

    def test_label_matches_the_resume_key_written_into_the_row(self):
        """The cache dir, the row's subset field and the resume key all derive
        from one function; if they diverge, resume either loops or skips."""
        point = {"chains": ("H", "L", "A"), "n_tokens": 554}
        label = m8.point_label(point)
        assert m8._row_key({"n_tokens": 554, "subset": label}) == (554, label)


class TestDeviceHelpers:
    @pytest.mark.parametrize(
        "device,index", [("cuda", 0), ("cuda:0", 0), ("cuda:3", 3), ("cpu", 0), ("cuda:x", 0)]
    )
    def test_cuda_index(self, device, index):
        assert m8._cuda_index(device) == index

    def test_allocated_gib_returns_none_without_cuda(self):
        """IGV_SKIP_VRAM_CHECK=1 can reach the loop on a laptop; annotating a
        row must never be what crashes the sweep."""

        class _NoCuda:
            class cuda:
                @staticmethod
                def is_available():
                    return False

        assert m8._allocated_gib(_NoCuda, 0) is None

    def test_allocated_gib_swallows_a_broken_allocator(self):
        class _Broken:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def memory_allocated(_index):
                    raise RuntimeError("CUDA context is toast after the OOM")

        assert m8._allocated_gib(_Broken, 0) is None

    def test_allocated_gib_converts_bytes_to_gib(self):
        class _Fake:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def memory_allocated(_index):
                    return 2 ** 30

        assert m8._allocated_gib(_Fake, 0) == pytest.approx(1.0)
