"""Tests for scripts/23_affinity_path_profile.py -- CPU only, no torch-CUDA.

Tests the pure helpers (total_variation, count_sign_changes,
nearest_profile_value, noise helpers), CLI parsing, dry-run path, and
noise-adjusted sign counting on synthetic data.  The 09_path_profile
helpers imported by the script are already tested in test_path_profile.py.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "23_affinity_path_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location("affinity_path_profile", str(_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


app = _load()


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_imports_without_torch_or_boltz(self):
        code = (
            "import sys, importlib.util as u\n"
            f"sys.path.insert(0, {str(_SCRIPT.parents[1] / 'src')!r})\n"
            f"spec = u.spec_from_file_location('app', {str(_SCRIPT)!r})\n"
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
        assert pulled_boltz == "False"
        assert pulled_torch == "False"


# ---------------------------------------------------------------------------
# 2. total_variation
# ---------------------------------------------------------------------------


class TestTotalVariation:
    def test_monotone_equals_span(self):
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        tv = app.total_variation(alphas, values)
        assert tv == pytest.approx(4.0)

    def test_oscillating_exceeds_span(self):
        alphas = [0.0, 0.5, 1.0]
        values = [0.0, 10.0, 0.0]
        tv = app.total_variation(alphas, values)
        assert tv == pytest.approx(20.0)
        span = abs(values[-1] - values[0])
        assert tv > span

    def test_constant_is_zero(self):
        alphas = [0.0, 0.5, 1.0]
        values = [3.0, 3.0, 3.0]
        assert app.total_variation(alphas, values) == pytest.approx(0.0)

    def test_too_few_points(self):
        assert math.isnan(app.total_variation([0.5], [1.0]))
        assert math.isnan(app.total_variation([], []))

    def test_mismatched_lengths(self):
        assert math.isnan(app.total_variation([0.0, 1.0], [1.0]))

    def test_unsorted_input(self):
        alphas = [1.0, 0.0, 0.5]
        values = [5.0, 1.0, 3.0]
        tv = app.total_variation(alphas, values)
        assert tv == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. count_sign_changes
# ---------------------------------------------------------------------------


class TestCountSignChanges:
    def test_no_changes(self):
        assert app.count_sign_changes([1.0, 2.0, 3.0]) == 0

    def test_one_change(self):
        assert app.count_sign_changes([1.0, -1.0, -2.0]) == 1

    def test_multiple_changes(self):
        assert app.count_sign_changes([1.0, -1.0, 1.0, -1.0]) == 3

    def test_skips_at_threshold(self):
        assert app.count_sign_changes([1.0, 0.0, -1.0], threshold=0.0) == 1

    def test_all_zero(self):
        assert app.count_sign_changes([0.0, 0.0, 0.0]) == 0

    def test_empty(self):
        assert app.count_sign_changes([]) == 0

    def test_single_value(self):
        assert app.count_sign_changes([1.0]) == 0

    def test_threshold_filters_noise(self):
        noisy = [1.0, -0.01, 0.02, -0.005, -1.0]
        assert app.count_sign_changes(noisy, threshold=0.0) == 3
        assert app.count_sign_changes(noisy, threshold=0.05) == 1

    def test_threshold_zero_same_as_raw(self):
        vals = [1.0, -2.0, 3.0]
        assert app.count_sign_changes(vals, threshold=0.0) == 2

    def test_large_threshold_suppresses_all(self):
        vals = [0.1, -0.1, 0.1, -0.1]
        assert app.count_sign_changes(vals, threshold=1.0) == 0


# ---------------------------------------------------------------------------
# 3b. Noise-adjusted sign counting on synthetic smooth + noisy data
# ---------------------------------------------------------------------------


class TestNoiseAdjustedSignChanges:
    """Verify that the threshold-based sign counting correctly separates
    real sign changes from noise in a known-smooth function."""

    def test_smooth_sine_no_noise(self):
        import math as m
        n = 101
        alphas = [i / (n - 1) for i in range(n)]
        values = [m.sin(2 * m.pi * a) for a in alphas]
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        raw = app.count_sign_changes(diffs, threshold=0.0)
        assert raw == 2

    def test_smooth_with_noise_raw_overcounts(self):
        import math as m
        import random
        rng = random.Random(42)
        n = 201
        noise_sd = 0.5
        alphas = [i / (n - 1) for i in range(n)]
        values = [m.sin(2 * m.pi * a) + rng.gauss(0, noise_sd) for a in alphas]
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        raw = app.count_sign_changes(diffs, threshold=0.0)
        assert raw > 10

        threshold = app.noise_adjusted_threshold(noise_sd, span=2.0)
        adjusted = app.count_sign_changes(diffs, threshold=threshold)
        assert adjusted <= 5

    def test_no_noise_sd_falls_back(self):
        threshold = app.noise_adjusted_threshold(None, span=1.0)
        assert threshold == pytest.approx(max(1e-4, 1e-8))

    def test_zero_noise_sd_falls_back(self):
        threshold = app.noise_adjusted_threshold(0.0, span=0.5)
        assert threshold == pytest.approx(max(0.5e-4, 1e-8))


# ---------------------------------------------------------------------------
# 3c. expected_noise_tv
# ---------------------------------------------------------------------------


class TestExpectedNoiseTv:
    def test_zero_noise(self):
        assert app.expected_noise_tv(100, 0.0) == 0.0

    def test_single_point(self):
        assert app.expected_noise_tv(1, 0.5) == 0.0

    def test_positive(self):
        tv = app.expected_noise_tv(257, 0.01)
        assert tv > 0
        expected = 256 * 0.01 * 2.0 / math.sqrt(math.pi)
        assert tv == pytest.approx(expected)

    def test_scales_linearly_with_sd(self):
        tv1 = app.expected_noise_tv(100, 0.01)
        tv2 = app.expected_noise_tv(100, 0.02)
        assert tv2 == pytest.approx(2 * tv1)


# ---------------------------------------------------------------------------
# 4. nearest_profile_value
# ---------------------------------------------------------------------------


class TestNearestProfileValue:
    def test_exact_match(self):
        profile = {0.0: 1.0, 0.5: 2.0, 1.0: 3.0}
        alpha, val = app.nearest_profile_value(profile, 0.5)
        assert alpha == 0.5
        assert val == 2.0

    def test_nearest_lower(self):
        profile = {0.0: 1.0, 0.5: 2.0, 1.0: 3.0}
        alpha, val = app.nearest_profile_value(profile, 0.24)
        assert alpha == 0.0
        assert val == 1.0

    def test_nearest_upper(self):
        profile = {0.0: 1.0, 0.5: 2.0, 1.0: 3.0}
        alpha, val = app.nearest_profile_value(profile, 0.76)
        assert alpha == 1.0
        assert val == 3.0

    def test_single_point(self):
        profile = {0.5: 7.0}
        alpha, val = app.nearest_profile_value(profile, 0.0)
        assert alpha == 0.5
        assert val == 7.0


# ---------------------------------------------------------------------------
# 5. CLI defaults
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        args = app.build_parser().parse_args([])
        assert args.baseline == "mean_aa"
        assert args.profile_steps == 257
        assert args.n_grad_alphas == 65
        assert args.fd_step == pytest.approx(0.02)
        assert args.skip_fd is False
        assert args.dry_run is False
        assert args.deterministic is False
        assert args.protein_chain_id == "A"
        assert args.ligand_id == "B"

    def test_baseline_choices(self):
        for bl in ("mean_aa", "unk", "shuffled"):
            args = app.build_parser().parse_args(["--baseline", bl])
            assert args.baseline == bl

    def test_invalid_baseline_rejected(self):
        with pytest.raises(SystemExit):
            app.build_parser().parse_args(["--baseline", "zeros"])

    def test_grad_alphas_overrides_n(self):
        args = app.build_parser().parse_args(["--grad-alphas", "0.1,0.5,0.9"])
        assert args.grad_alphas == "0.1,0.5,0.9"

    def test_custom_profile_steps(self):
        args = app.build_parser().parse_args(["--profile-steps", "129"])
        assert args.profile_steps == 129

    def test_deterministic_flag(self):
        args = app.build_parser().parse_args(["--deterministic"])
        assert args.deterministic is True


# ---------------------------------------------------------------------------
# 6. Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True,
            cwd=str(_SCRIPT.parents[1]),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    def test_prints_plan(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dry-run"],
            capture_output=True, text=True,
            cwd=str(_SCRIPT.parents[1]),
        )
        out = result.stdout
        assert "Affinity path profile" in out
        assert "profile steps" in out
        assert "gradcheck" in out
        assert "noise floor" in out
        assert "forward-equivalents" in out
        assert "VM command" in out
        assert "deterministic" in out


# ---------------------------------------------------------------------------
# 7. CSV schema
# ---------------------------------------------------------------------------


class TestCSVSchema:
    def test_has_required_columns(self):
        assert "head" in app._CSV_COLUMNS
        assert "kind" in app._CSV_COLUMNS
        assert "alpha" in app._CSV_COLUMNS
        assert "F_alpha" in app._CSV_COLUMNS
        assert "F_alpha_grad_pass" in app._CSV_COLUMNS
        assert "D_analytic" in app._CSV_COLUMNS
        assert "baseline" in app._CSV_COLUMNS
        assert "fd_snr_warning" in app._CSV_COLUMNS


# ---------------------------------------------------------------------------
# 8. Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_heads_tuple(self):
        assert app.HEADS == ("affinity", "confidence")

    def test_affinity_score(self):
        assert app._AFFINITY_SCORE == "affinity_pred_value"

    def test_confidence_score(self):
        assert app._CONFIDENCE_SCORE == "complex_pde"

    def test_baseline_choices(self):
        assert "mean_aa" in app.BASELINE_CHOICES
        assert "unk" in app.BASELINE_CHOICES
        assert "shuffled" in app.BASELINE_CHOICES

    def test_noise_alphas(self):
        assert app._NOISE_ALPHAS == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_noise_repeats(self):
        assert app._NOISE_REPEATS == 3


# ---------------------------------------------------------------------------
# 9. Imported helpers available
# ---------------------------------------------------------------------------


class TestImportedHelpers:
    def test_alpha_grid(self):
        g = app.alpha_grid(5)
        assert g[0] == 0.0
        assert g[-1] == 1.0
        assert len(g) == 5

    def test_parse_grad_alphas(self):
        assert app.parse_grad_alphas("0.1,0.5") == [0.1, 0.5]

    def test_trapezoid_estimate(self):
        val = app.trapezoid_estimate([0.0, 1.0], [1.0, 1.0])
        assert val == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. noise_adjusted_threshold
# ---------------------------------------------------------------------------


class TestNoiseAdjustedThreshold:
    def test_with_measured_noise(self):
        th = app.noise_adjusted_threshold(0.01, span=1.0, multiplier=3.0)
        expected = 3.0 * 0.01 * math.sqrt(2)
        assert th == pytest.approx(expected)

    def test_fallback_without_noise(self):
        th = app.noise_adjusted_threshold(None, span=10.0)
        assert th == pytest.approx(10.0 * 1e-4)

    def test_fallback_small_span(self):
        th = app.noise_adjusted_threshold(None, span=1e-6)
        assert th == pytest.approx(1e-8)

    def test_custom_multiplier(self):
        th = app.noise_adjusted_threshold(0.05, span=1.0, multiplier=5.0)
        expected = 5.0 * 0.05 * math.sqrt(2)
        assert th == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 11. Profile / gradient path agreement (mock-based)
# ---------------------------------------------------------------------------


class TestProfileGradientAgreement:
    """A deterministic quadratic function f(x)=x^2 evaluated through a thin
    mock verifies that when the same code path is used, profile and gradient
    evaluations agree exactly."""

    def test_same_function_gives_zero_diff(self):
        def f(alpha):
            return (3.0 * alpha - 1.5) ** 2

        alphas = [i / 100 for i in range(101)]
        profile = {a: f(a) for a in alphas}

        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            f_profile = profile[a]
            f_grad = f(a)
            assert abs(f_profile - f_grad) < 1e-12

    def test_fd_matches_analytic_derivative(self):
        def f(alpha):
            return (3.0 * alpha - 1.5) ** 2

        h = 0.01
        alpha = 0.5
        analytic = 2 * (3.0 * alpha - 1.5) * 3.0

        f_lo = f(alpha - h)
        f_hi = f(alpha + h)
        fd = app.finite_difference(f_lo, f_hi, alpha - h, alpha + h)
        assert fd == pytest.approx(analytic, abs=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
