"""Tests for scripts/09_path_profile.py -- all CPU, no torch-CUDA, no boltz.

The script itself needs a GPU, but the arithmetic that decides whether its
output means anything is pure Python: the alpha grid, the finite-difference
formula, the one-sided fallback, ratio/relerr, and the trapezoid estimate.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_path_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location("igv_path_profile", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load()


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_imports_without_torch_or_boltz(self):
        code = (
            "import sys, importlib.util as u\n"
            f"sys.path.insert(0, {str(_SCRIPT.parents[1] / 'src')!r})\n"
            f"spec = u.spec_from_file_location('pp', {str(_SCRIPT)!r})\n"
            "mod = u.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert hasattr(mod, 'main')\n"
            "print('boltz' in sys.modules, 'torch' in sys.modules)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        pulled_boltz, pulled_torch = out.stdout.split()
        assert pulled_boltz == "False", "09_path_profile imported boltz at module level"
        assert pulled_torch == "False", "09_path_profile imported torch at module level"

    def test_parser_builds_and_defaults(self):
        args = pp.build_parser().parse_args(["--dataset", "4fqi_h1", "--dry-run"])
        assert args.dry_run is True
        assert args.profile_steps == 21
        assert args.fd_step == 0.1
        assert args.grad_alphas == "0.1,0.5,0.9"
        assert args.score == "complex_pde"


# ---------------------------------------------------------------------------
# 2. alpha_grid
# ---------------------------------------------------------------------------


class TestAlphaGrid:
    def test_endpoints_included(self):
        g = pp.alpha_grid(5)
        assert g[0] == 0.0
        assert g[-1] == 1.0
        assert len(g) == 5

    def test_n_points(self):
        for n in (2, 3, 11, 21):
            g = pp.alpha_grid(n)
            assert len(g) == n
            assert g[0] == 0.0
            assert g[-1] == pytest.approx(1.0)

    def test_uniform_spacing(self):
        g = pp.alpha_grid(5)
        diffs = [g[i + 1] - g[i] for i in range(len(g) - 1)]
        for d in diffs:
            assert d == pytest.approx(0.25)

    def test_n_zero_disables(self):
        assert pp.alpha_grid(0) == []

    def test_n_one_gives_zero(self):
        assert pp.alpha_grid(1) == [0.0]


# ---------------------------------------------------------------------------
# 3. parse_grad_alphas
# ---------------------------------------------------------------------------


class TestParseGradAlphas:
    def test_default_spec(self):
        assert pp.parse_grad_alphas("0.1,0.5,0.9") == [0.1, 0.5, 0.9]

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="outside"):
            pp.parse_grad_alphas("0.5,1.5")
        with pytest.raises(ValueError, match="outside"):
            pp.parse_grad_alphas("-0.1")

    def test_boundary_values_accepted(self):
        assert pp.parse_grad_alphas("0.0,1.0") == [0.0, 1.0]


# ---------------------------------------------------------------------------
# 4. Finite difference: central vs one-sided
# ---------------------------------------------------------------------------


class TestFiniteDifference:
    def test_central_against_known_analytic(self):
        # f(x) = x^3, f'(0.5) = 3*(0.5)^2 = 0.75
        # Central FD error is O(h^2); with h=0.01 the error is ~3e-4.
        h = 0.01
        f_lo = (0.5 - h) ** 3
        f_hi = (0.5 + h) ** 3
        fd = pp.finite_difference(f_lo, f_hi, 0.5 - h, 0.5 + h)
        assert fd == pytest.approx(0.75, abs=1e-3)

    def test_fd_kind_central(self):
        assert pp.fd_kind_for(0.5, 0.1) == "central"

    def test_fd_kind_forward_near_zero(self):
        assert pp.fd_kind_for(0.05, 0.1) == "forward"

    def test_fd_kind_backward_near_one(self):
        assert pp.fd_kind_for(0.95, 0.1) == "backward"

    def test_fd_kind_rejects_h_le_zero(self):
        with pytest.raises(ValueError, match="h must be > 0"):
            pp.fd_kind_for(0.5, 0.0)
        with pytest.raises(ValueError, match="h must be > 0"):
            pp.fd_kind_for(0.5, -0.1)

    def test_fd_points_central(self):
        lo, hi, kind = pp.fd_points(0.5, 0.1)
        assert kind == "central"
        assert lo == pytest.approx(0.4)
        assert hi == pytest.approx(0.6)

    def test_fd_points_forward(self):
        lo, hi, kind = pp.fd_points(0.05, 0.1)
        assert kind == "forward"
        assert lo == pytest.approx(0.05)
        assert hi == pytest.approx(0.25)

    def test_fd_points_backward(self):
        lo, hi, kind = pp.fd_points(0.95, 0.1)
        assert kind == "backward"
        assert lo == pytest.approx(0.75)
        assert hi == pytest.approx(0.95)

    def test_one_sided_fd_against_analytic(self):
        # f(x) = x^2, forward FD at alpha=0.05 with h=0.1 uses [0.05, 0.25].
        # The FD approximates f' at the midpoint (0.15), so expect 2*0.15=0.30.
        lo, hi, kind = pp.fd_points(0.05, 0.1)
        assert kind == "forward"
        f_lo = lo ** 2
        f_hi = hi ** 2
        fd = pp.finite_difference(f_lo, f_hi, lo, hi)
        assert fd == pytest.approx(0.30, abs=1e-10)

    def test_impossible_fd_raises(self):
        with pytest.raises(ValueError, match="cannot fit"):
            pp.fd_kind_for(0.5, 0.6)


# ---------------------------------------------------------------------------
# 5. Ratio and relative error
# ---------------------------------------------------------------------------


class TestRatioRelerr:
    def test_perfect_agreement(self):
        r, re = pp.ratio_and_relerr(5.0, 5.0)
        assert r == pytest.approx(1.0)
        assert re == pytest.approx(0.0)

    def test_4x_overshoot(self):
        r, re = pp.ratio_and_relerr(4.64, 1.0)
        assert r == pytest.approx(4.64)
        assert re == pytest.approx(3.64)

    def test_fd_zero_gives_nan(self):
        r, re = pp.ratio_and_relerr(1.0, 0.0)
        assert math.isnan(r)
        assert math.isnan(re)

    def test_f1_minus_f0_zero_guard(self):
        r, re = pp.ratio_and_relerr(0.0, 0.0)
        assert math.isnan(r)


# ---------------------------------------------------------------------------
# 6. Trapezoid estimate
# ---------------------------------------------------------------------------


class TestTrapezoid:
    def test_against_hand_computed(self):
        # integral of f(x)=x from 0 to 1 = 0.5
        alphas = [0.0, 0.5, 1.0]
        values = [0.0, 0.5, 1.0]
        assert pp.trapezoid_estimate(alphas, values) == pytest.approx(0.5)

    def test_constant_function(self):
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        assert pp.trapezoid_estimate(alphas, values) == pytest.approx(3.0)

    def test_unordered_alphas_sorted(self):
        # Should sort internally
        alphas = [1.0, 0.0, 0.5]
        values = [1.0, 0.0, 0.5]
        assert pp.trapezoid_estimate(alphas, values) == pytest.approx(0.5)

    def test_too_few_points(self):
        assert math.isnan(pp.trapezoid_estimate([0.5], [1.0]))
        assert math.isnan(pp.trapezoid_estimate([], []))

    def test_mismatched_lengths(self):
        assert math.isnan(pp.trapezoid_estimate([0.0, 1.0], [1.0]))

    def test_three_default_alphas(self):
        # integral of f(x)=-8.6 (constant) from 0.1 to 0.9 = -8.6 * 0.8 = -6.88
        alphas = [0.1, 0.5, 0.9]
        values = [-8.6, -8.6, -8.6]
        assert pp.trapezoid_estimate(alphas, values) == pytest.approx(-6.88)


# ---------------------------------------------------------------------------
# 7. CSV row schema
# ---------------------------------------------------------------------------


class TestCSVSchema:
    def test_profile_columns_are_subset_of_all(self):
        all_cols = list(dict.fromkeys(pp._PROFILE_COLUMNS + pp._GRADCHECK_COLUMNS))
        for c in pp._PROFILE_COLUMNS:
            assert c in all_cols

    def test_gradcheck_columns_present(self):
        required = {"kind", "alpha", "D_analytic", "D_fd", "ratio", "relerr", "fd_kind"}
        assert required <= set(pp._GRADCHECK_COLUMNS)


# ---------------------------------------------------------------------------
# 8. Wall-time estimate
# ---------------------------------------------------------------------------


class TestWallTimeEstimate:
    def test_defaults(self):
        # 21 profile + 3 gradcheck: 21 + 3*2 = 27 fwd, 3 bwd
        # at 1s each: 27 + 3 = 30
        t = pp.estimate_wall_time(21, 3, 1.0, 1.0)
        assert t == pytest.approx(30.0)

    def test_zero_gradcheck(self):
        t = pp.estimate_wall_time(10, 0, 2.0, 5.0)
        assert t == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 9. dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_zero_and_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        out = tmp_path / "pp_{dataset}_{score}.csv"
        rc = pp.main([
            "--dataset", "4fqi_h1",
            "--chain-subset", "H,L,A",
            "--dry-run",
            "--out", str(out),
        ])
        assert rc == 0
        assert not list(tmp_path.iterdir())

    def test_dry_run_reports_cost(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        out = tmp_path / "pp_{dataset}_{score}.csv"
        pp.main([
            "--dataset", "4fqi_h1",
            "--chain-subset", "H,L,A",
            "--dry-run",
            "--out", str(out),
        ])
        captured = capsys.readouterr().out
        assert "L=554" in captured
        assert "forward-equivalents" in captured
        assert "bf16" in captured
        assert "fp32" in captured


# ---------------------------------------------------------------------------
# 10. --skip-fd
# ---------------------------------------------------------------------------


class TestSkipFd:
    def test_parser_default_is_false(self):
        args = pp.build_parser().parse_args(["--dataset", "4fqi_h1"])
        assert args.skip_fd is False

    def test_parser_accepts_flag(self):
        args = pp.build_parser().parse_args(["--dataset", "4fqi_h1", "--skip-fd"])
        assert args.skip_fd is True

    def test_dry_run_cost_with_skip_fd(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        out = tmp_path / "pp_{dataset}_{score}.csv"
        pp.main([
            "--dataset", "4fqi_h1",
            "--chain-subset", "H,L,A",
            "--skip-fd",
            "--dry-run",
            "--out", str(out),
        ])
        captured = capsys.readouterr().out
        assert "no FD" in captured
        assert "skip_fd:        True" in captured
        # 21 profile + 3 gradcheck (1 bwd each, no 2 fwd) = 24 forward-equivalents
        assert "24 forward-equivalents" in captured

    def test_dry_run_cost_without_skip_fd(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        out = tmp_path / "pp_{dataset}_{score}.csv"
        pp.main([
            "--dataset", "4fqi_h1",
            "--chain-subset", "H,L,A",
            "--dry-run",
            "--out", str(out),
        ])
        captured = capsys.readouterr().out
        # 21 profile + 3*(2 fwd + 1 bwd) = 30 forward-equivalents
        assert "30 forward-equivalents" in captured
        assert "skip_fd:        False" in captured

    def test_estimate_wall_time_skip_fd(self):
        # 21 profile, 3 gradcheck, skip_fd: 21 fwd + 0 fwd + 3 bwd = 21+3=24
        t = pp.estimate_wall_time(21, 3, 1.0, 1.0, skip_fd=True)
        assert t == pytest.approx(24.0)

    def test_estimate_wall_time_no_skip_fd(self):
        t = pp.estimate_wall_time(21, 3, 1.0, 1.0, skip_fd=False)
        assert t == pytest.approx(30.0)

    def test_gradcheck_columns_has_fd_kind(self):
        assert "fd_kind" in pp._GRADCHECK_COLUMNS
