"""Tests for Gate A de-risking script -- CPU only, no boltz, no GPU.

Tests the argument parsing, --dry-run path, the pure evaluation logic
that decides PASS/FAIL from measured values, and the INCONCLUSIVE pre-checks.
The GPU path (the actual gate) cannot be tested here -- it runs on the VM.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# Import the script module so we can test its pure functions directly.
_SCRIPT = Path(__file__).parent.parent / "scripts" / "14_gate_a.py"
_spec = importlib.util.spec_from_file_location("gate_a", str(_SCRIPT))
gate_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_a)

_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        args = gate_a.build_parser().parse_args([])
        assert args.dataset == "1VFB"
        assert args.chain == "A"
        assert args.chain_subset == "A,B,C"
        assert args.score == "complex_pde"
        assert args.m_steps == 5
        assert args.recycling_steps == 1
        assert args.dry_run is False
        assert args.completeness_threshold == pytest.approx(0.10)
        assert args.z_baseline == "mean_aa"
        assert args.msa is None

    def test_custom_args(self):
        args = gate_a.build_parser().parse_args([
            "--dataset", "1JTG", "--chain", "B", "--chain-subset", "A,B",
            "--score", "iptm", "--m-steps", "10", "--dry-run",
            "--completeness-threshold", "0.15",
            "--z-baseline", "zeros",
            "--msa", "empty",
        ])
        assert args.dataset == "1JTG"
        assert args.chain == "B"
        assert args.chain_subset == "A,B"
        assert args.score == "iptm"
        assert args.m_steps == 10
        assert args.dry_run is True
        assert args.completeness_threshold == pytest.approx(0.15)
        assert args.z_baseline == "zeros"
        assert args.msa == "empty"


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
        assert "Gate A" in out
        assert "L (tokens)" in out
        assert "Pass criteria" in out
        assert "VM command" in out
        assert "INCONCLUSIVE" in out

    def test_prints_l_value(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert "352" in result.stdout

    def test_prints_baseline_and_msa(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert "mean_aa" in result.stdout
        assert "msa" in result.stdout.lower()

    def test_custom_dataset(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run",
             "--dataset", "1JTG", "--chain", "B", "--chain-subset", "A,B"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "427" in result.stdout

    def test_single_chain_warning(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run",
             "--chain-subset", "A"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0
        assert "WARNING" in result.stdout
        assert "degenerate" in result.stdout


# ---------------------------------------------------------------------------
# 3. L resolution
# ---------------------------------------------------------------------------


class TestResolveL:
    def test_1vfb_full_complex(self):
        args = gate_a.build_parser().parse_args([])
        chains, n_tokens, label = gate_a.resolve_l(args)
        assert chains is not None
        assert n_tokens == 352

    def test_1vfb_chain_a_only(self):
        args = gate_a.build_parser().parse_args(["--chain-subset", "A"])
        chains, n_tokens, label = gate_a.resolve_l(args)
        assert chains is not None
        assert n_tokens == 107
        assert "A" in label

    def test_1jtg_chain_ab(self):
        args = gate_a.build_parser().parse_args([
            "--dataset", "1JTG", "--chain", "B", "--chain-subset", "A,B",
        ])
        chains, n_tokens, label = gate_a.resolve_l(args)
        assert chains is not None
        assert n_tokens == 427

    def test_missing_pdb(self):
        args = gate_a.build_parser().parse_args([
            "--dataset", "NONEXISTENT", "--structure", "nonexistent",
        ])
        chains, n_tokens, label = gate_a.resolve_l(args)
        assert chains is None
        assert n_tokens is None


# ---------------------------------------------------------------------------
# 4. INCONCLUSIVE checks (new)
# ---------------------------------------------------------------------------


class TestCheckInconclusive:
    def test_healthy_returns_none(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is None

    def test_non_finite_f_x(self):
        reason = gate_a.check_inconclusive(
            f_x=float("nan"), f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "non-finite" in reason

    def test_non_finite_f_baseline(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=float("inf"),
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "non-finite" in reason

    def test_f_x_exactly_zero(self):
        reason = gate_a.check_inconclusive(
            f_x=0.0, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "exactly zero" in reason

    def test_f_baseline_exactly_zero(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=0.0,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "exactly zero" in reason

    def test_collapsed_span(self):
        reason = gate_a.check_inconclusive(
            f_x=2.620000, f_baseline=2.620000,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "collapsed span" in reason

    def test_z_x_constant(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=True, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "constant" in reason

    def test_path_constant(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=True,
            msa_mode="server",
        )
        assert reason is not None
        assert "constant" in reason

    def test_none_scores(self):
        reason = gate_a.check_inconclusive(
            f_x=None, f_baseline=None,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "could not be computed" in reason

    def test_nan_is_not_finite(self):
        reason = gate_a.check_inconclusive(
            f_x=float("nan"), f_baseline=1.0,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="empty",
        )
        assert reason is not None

    def test_negative_inf_is_not_finite(self):
        reason = gate_a.check_inconclusive(
            f_x=float("-inf"), f_baseline=1.0,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is not None
        assert "non-finite" in reason


# ---------------------------------------------------------------------------
# 5. Evaluation logic (the part that decides PASS/FAIL from synthetic inputs)
# ---------------------------------------------------------------------------


class TestEvaluateChecksAllPass:
    def test_all_pass(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.234e-3,
            z_grad_zero_frac=0.01,
            z_grad_shape=(1, 50, 50, 128),
            expected_shape=(1, 50, 50, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.05,
            ig_sum=-2.345,
            f_diff=-2.346,
        )
        assert len(checks) == 5
        assert all(c["passed"] for c in checks), [
            c["name"] for c in checks if not c["passed"]
        ]


class TestEvaluateChecksGradNone:
    def test_all_fail_when_grad_none(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=False,
            z_grad_max_abs=None,
            z_grad_zero_frac=None,
            z_grad_shape=None,
            expected_shape=(1, 50, 50, 128),
            z_grad_n_nan=None,
            z_grad_n_inf=None,
            completeness_abs_err=None,
            completeness_rel_err=None,
            ig_sum=None,
            f_diff=None,
        )
        assert not any(c["passed"] for c in checks)


class TestEvaluateChecksUniformlyZero:
    def test_zero_grad_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=0.0,
            z_grad_zero_frac=1.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.05,
            ig_sum=0.0,
            f_diff=-1.0,
        )
        passed_names = [c["name"] for c in checks if c["passed"]]
        failed_names = [c["name"] for c in checks if not c["passed"]]
        assert "z.grad not uniformly zero" in failed_names
        assert "z.grad is not None" in passed_names


class TestEvaluateChecksWrongShape:
    def test_shape_mismatch_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=0.5,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 64),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.05,
            ig_sum=-1.0,
            f_diff=-1.05,
        )
        shape_check = [c for c in checks if "shape" in c["name"]][0]
        assert not shape_check["passed"]


class TestEvaluateChecksNanInf:
    def test_nan_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=3,
            z_grad_n_inf=0,
            completeness_abs_err=0.001,
            completeness_rel_err=0.05,
            ig_sum=-1.0,
            f_diff=-1.05,
        )
        finite_check = [c for c in checks if "finite" in c["name"]][0]
        assert not finite_check["passed"]
        assert "NaN=3" in finite_check["detail"]

    def test_inf_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=7,
            completeness_abs_err=0.001,
            completeness_rel_err=0.05,
            ig_sum=-1.0,
            f_diff=-1.05,
        )
        finite_check = [c for c in checks if "finite" in c["name"]][0]
        assert not finite_check["passed"]
        assert "Inf=7" in finite_check["detail"]


class TestEvaluateChecksCompleteness:
    def test_above_threshold_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.15,
            completeness_rel_err=0.25,
            ig_sum=-1.5,
            f_diff=-2.0,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert not comp_check["passed"]
        assert "abs_err=0.150000" in comp_check["detail"]

    def test_at_threshold_fails(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.10,
            completeness_rel_err=0.20,
            ig_sum=-0.8,
            f_diff=-1.0,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert not comp_check["passed"]

    def test_just_below_threshold_passes(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.099,
            completeness_rel_err=0.199,
            ig_sum=-0.901,
            f_diff=-1.0,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert comp_check["passed"]

    def test_custom_threshold(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.04,
            completeness_rel_err=0.09,
            ig_sum=-0.91,
            f_diff=-1.0,
            threshold=0.05,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert comp_check["passed"]
        assert "< 0.05" in comp_check["name"]

    def test_completeness_reports_span(self):
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.05,
            completeness_rel_err=0.10,
            ig_sum=-0.95,
            f_diff=-1.0,
        )
        comp_check = [c for c in checks if "completeness" in c["name"]][0]
        assert "span=" in comp_check["detail"]


# ---------------------------------------------------------------------------
# 6. print_checks
# ---------------------------------------------------------------------------


class TestPrintChecks:
    def test_returns_true_when_all_pass(self, capsys):
        checks = [
            {"name": "check1", "passed": True, "detail": "ok"},
            {"name": "check2", "passed": True, "detail": ""},
        ]
        assert gate_a.print_checks(checks) is True
        out = capsys.readouterr().out
        assert "[PASS]" in out
        assert "[FAIL]" not in out

    def test_returns_false_when_any_fail(self, capsys):
        checks = [
            {"name": "check1", "passed": True, "detail": ""},
            {"name": "check2", "passed": False, "detail": "bad"},
        ]
        assert gate_a.print_checks(checks) is False
        out = capsys.readouterr().out
        assert "[PASS]" in out
        assert "[FAIL]" in out


# ---------------------------------------------------------------------------
# 7. INCONCLUSIVE does NOT trigger on genuine FAIL conditions
# ---------------------------------------------------------------------------


class TestInconclusiveDoesNotMaskFail:
    """Genuine grad failures (None, zero, NaN) should NOT be INCONCLUSIVE.

    INCONCLUSIVE is about the *setup* being degenerate. A healthy setup
    producing z.grad=None is a real FAIL.
    """

    def test_healthy_setup_with_none_grad_is_not_inconclusive(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is None

    def test_healthy_setup_allows_fail_on_zero_grad(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is None
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=0.0,
            z_grad_zero_frac=1.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=0,
            z_grad_n_inf=0,
            completeness_abs_err=0.0,
            completeness_rel_err=0.0,
            ig_sum=0.0,
            f_diff=-0.63,
        )
        assert not all(c["passed"] for c in checks)

    def test_healthy_setup_allows_fail_on_nan_grad(self):
        reason = gate_a.check_inconclusive(
            f_x=2.62, f_baseline=3.25,
            z_x_is_constant=False, path_is_constant=False,
            msa_mode="server",
        )
        assert reason is None
        checks = gate_a.evaluate_checks(
            z_grad_not_none=True,
            z_grad_max_abs=1.0,
            z_grad_zero_frac=0.0,
            z_grad_shape=(1, 10, 10, 128),
            expected_shape=(1, 10, 10, 128),
            z_grad_n_nan=5,
            z_grad_n_inf=0,
            completeness_abs_err=0.05,
            completeness_rel_err=0.10,
            ig_sum=-0.95,
            f_diff=-1.0,
        )
        finite_check = [c for c in checks if "finite" in c["name"]][0]
        assert not finite_check["passed"]


# ---------------------------------------------------------------------------
# 8. _is_single_chain helper
# ---------------------------------------------------------------------------


class TestIsSingleChain:
    def test_single(self):
        assert gate_a._is_single_chain("A") is True

    def test_multi(self):
        assert gate_a._is_single_chain("A,B") is False

    def test_multi_three(self):
        assert gate_a._is_single_chain("A,B,C") is False


# ---------------------------------------------------------------------------
# 9. --compare-x-pred argument parsing
# ---------------------------------------------------------------------------


class TestCompareXPredArg:
    def test_default_off(self):
        args = gate_a.build_parser().parse_args([])
        assert args.compare_x_pred is False

    def test_flag_on(self):
        args = gate_a.build_parser().parse_args(["--compare-x-pred"])
        assert args.compare_x_pred is True

    def test_dry_run_with_compare(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run", "--compare-x-pred"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "compare-x-pred" in result.stdout or "structure prediction" in result.stdout


# ---------------------------------------------------------------------------
# 10. format_x_pred_comparison
# ---------------------------------------------------------------------------


class TestFormatXPredComparison:
    def test_near_identical(self):
        block = gate_a.format_x_pred_comparison("complex_pde", 2.620000, 2.620050)
        assert "complex_pde" in block
        assert "zeros" in block
        assert "predicted" in block
        assert "nearly identical" in block

    def test_meaningful_difference(self):
        block = gate_a.format_x_pred_comparison("complex_pde", 2.62, 3.15)
        assert "changes meaningfully" in block
        assert "difference" in block

    def test_difference_values(self):
        block = gate_a.format_x_pred_comparison("complex_pde", 2.0, 3.0)
        assert "+1.000000" in block
        assert "+50.0000%" in block

    def test_zero_score_does_not_crash(self):
        block = gate_a.format_x_pred_comparison("complex_pde", 0.0, 1.0)
        assert "inf" in block.lower() or "complex_pde" in block

    def test_negative_difference(self):
        block = gate_a.format_x_pred_comparison("complex_pde", 3.0, 2.5)
        assert "-0.500000" in block


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
