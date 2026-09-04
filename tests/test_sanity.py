"""Tests for the Tier-0 gate in scripts/07_sanity.py -- all CPU, no boltz.

The gate is the one stage whose *own* bugs are indistinguishable from
scientific findings, so the three fixed bugs each get a test that fails on the
old code:

* ``randomise_`` must not zero LayerNorm scales (that annihilated the layer,
  made the gradient constant, made Spearman NaN and fired the gate's
  project-killing message about a network the check itself broke);
* an informational check must stay informational when it RAISES;
* ``check_dead_target`` must actually run the model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

from igv.metrics import spearman  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_sanity.py"


def _load_stage07():
    """Import 07_sanity.py by path -- the module name starts with a digit."""
    spec = importlib.util.spec_from_file_location("igv_stage07_sanity", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sanity = _load_stage07()

L, D = 6, 4


class _TinyNormNet(torch.nn.Module):
    """Linear -> LayerNorm: the minimal model with a 1-D norm *scale*."""

    def __init__(self, d=D):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.lin(x))


def _zero_every_1d_(module, seed=0):
    """The pre-fix initialisation (scripts/07_sanity.py:132-138), for contrast."""
    torch.manual_seed(seed)
    for p in module.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)


def _token_grad_norms(model, x):
    x = x.detach().requires_grad_(True)
    model(x).sum().backward()
    return x.grad.squeeze(0).norm(dim=-1).detach().numpy()


# ---------------------------------------------------------------------------
# 1. randomise_ : biases zeroed, norm scales left at unity
# ---------------------------------------------------------------------------


class TestRandomiseInit:
    def test_norm_scale_is_ones_and_biases_are_zero(self):
        m = _TinyNormNet()
        n, n2d, nbias, nnorm = sanity.randomise_(m, seed=0)

        assert (n, n2d, nbias, nnorm) == (4, 1, 2, 1)
        # The whole point: the LayerNorm scale must be the norm's identity.
        assert torch.equal(m.norm.weight, torch.ones(D))
        assert torch.equal(m.norm.bias, torch.zeros(D))
        assert torch.equal(m.lin.bias, torch.zeros(D))
        # ...while the matrix really was re-randomised.
        assert m.lin.weight.abs().max() > 0

    def test_output_is_not_annihilated(self):
        torch.manual_seed(7)
        x = torch.randn(1, L, D)

        good = _TinyNormNet()
        sanity.randomise_(good, seed=0)
        bad = _TinyNormNet()
        _zero_every_1d_(bad, seed=0)

        assert float(good(x).std()) > 0.0
        # Old behaviour, kept as the regression witness: a zeroed norm scale
        # makes the layer emit exactly zero for every input.
        assert torch.equal(bad(x), torch.zeros(1, L, D))

    # scipy warns on the constant-input arm; that is the behaviour under test.
    @pytest.mark.filterwarnings("ignore:An input array is constant")
    def test_gradient_variance_survives_so_spearman_is_finite(self):
        """The NaN chain the fix breaks: zero scale -> constant grad -> NaN rho."""
        torch.manual_seed(11)
        x = torch.randn(1, L, D)

        trained = _TinyNormNet()
        tm = _token_grad_norms(trained, x)

        good = _TinyNormNet()
        sanity.randomise_(good, seed=0)
        assert np.isfinite(spearman(tm, _token_grad_norms(good, x)))

        bad = _TinyNormNet()
        _zero_every_1d_(bad, seed=0)
        bad_norms = _token_grad_norms(bad, x)
        assert float(bad_norms.std()) == 0.0
        assert not np.isfinite(spearman(tm, bad_norms))

    def test_seed_is_reproducible(self):
        a, b = _TinyNormNet(), _TinyNormNet()
        sanity.randomise_(a, seed=3)
        sanity.randomise_(b, seed=3)
        assert torch.equal(a.lin.weight, b.lin.weight)


# ---------------------------------------------------------------------------
# 2. check_random_weights : a non-finite Spearman is an ERROR, not a verdict
# ---------------------------------------------------------------------------


class TestCheckRandomWeights:
    @staticmethod
    def _make_forward_fn_factory(flat: bool):
        """flat=True gives every token the same gradient magnitude -> rho NaN."""
        torch.manual_seed(0)
        w = torch.ones(1, L, D) if flat else torch.randn(1, L, D)

        def make_forward_fn(m, **_kw):
            del m  # the tiny stand-in model is irrelevant to the gradient here

            def fn(s):
                return (w * s).sum()

            return fn

        return make_forward_fn

    @pytest.mark.filterwarnings("ignore:An input array is constant")
    def test_non_finite_rho_never_prints_the_project_killing_text(self):
        x = torch.randn(1, L, D)
        r = sanity.check_random_weights(
            self._make_forward_fn_factory(flat=True), _TinyNormNet(), x,
            torch.zeros_like(x),
        )
        assert r["passed"] is False
        assert not np.isfinite(r["value"])
        assert "KILLS THE PROJECT" not in r["detail"]
        assert "not finite" in r["detail"]
        assert "degenerate-model artifact" in r["detail"]

    def test_detail_reports_the_init_split(self):
        x = torch.randn(1, L, D)
        r = sanity.check_random_weights(
            self._make_forward_fn_factory(flat=False), _TinyNormNet(), x,
            torch.zeros_like(x),
        )
        assert "4 parameter tensors" in r["detail"]
        assert "1 xavier" in r["detail"]
        assert "2 zeroed biases" in r["detail"]
        assert "1 unit norms" in r["detail"]


# ---------------------------------------------------------------------------
# 3. informational checks stay informational, including on the exception path
# ---------------------------------------------------------------------------


class TestInformationalFlag:
    def test_frozen_vs_full_is_the_only_informational_check(self):
        assert sanity.INFORMATIONAL == frozenset({"frozen_vs_full"})
        assert sanity.INFORMATIONAL <= set(sanity.CHECKS)

    def test_flag_is_set_on_the_failure_path_too(self):
        """The recorded regression: an OOM in frozen_vs_full blocked the gate."""
        crashed = sanity._result(
            "frozen_vs_full", False, None, "raised OutOfMemoryError: CUDA OOM"
        )
        assert crashed["informational"] is True
        # This is main()'s blocking filter (scripts/07_sanity.py).
        blocking = [
            r for r in [crashed] if not r["passed"] and not r.get("informational")
        ]
        assert blocking == []

    def test_real_checks_are_not_informational(self):
        for name in sanity.CHECKS:
            if name == "frozen_vs_full":
                continue
            assert sanity._result(name, False, None, "boom")["informational"] is False

    def test_every_check_has_a_threshold(self):
        assert set(sanity.CHECKS) == set(sanity.THRESHOLDS)


# ---------------------------------------------------------------------------
# 4. dead_target : gradient exactly zero, value model-dependent, one forward
# ---------------------------------------------------------------------------


class TestCheckDeadTarget:
    @staticmethod
    def _counting_forward_fn(scale: float):
        calls = []

        def fn(s):
            calls.append(1)
            return (s * scale).sum()

        return fn, calls

    def test_gradient_vanishes_and_check_passes(self):
        torch.manual_seed(2)
        x = torch.randn(1, L, D)
        fn, calls = self._counting_forward_fn(3.0)
        r = sanity.check_dead_target(fn, x, torch.zeros_like(x))

        assert r["passed"] is True
        assert r["value"] == 0.0
        assert len(calls) == 1, "must stay at ONE forward -- it runs on a GPU at the limit"

    def test_recorded_value_is_the_real_model_score(self):
        """The old objective was (x*0).sum()+1: value 1.0 for every model."""
        torch.manual_seed(2)
        x = torch.randn(1, L, D)
        fn_a, _ = self._counting_forward_fn(3.0)
        fn_b, _ = self._counting_forward_fn(-5.0)

        r_a = sanity.check_dead_target(fn_a, x, torch.zeros_like(x))
        r_b = sanity.check_dead_target(fn_b, x, torch.zeros_like(x))

        expected_a = float((x * 3.0).sum())
        assert f"f(x) = {expected_a:.6f}" in r_a["detail"]
        assert r_a["detail"] != r_b["detail"], "value must depend on the model"
        assert "f(x) = 1.000000" not in r_a["detail"]

    def test_detail_reports_the_gradient_too(self):
        x = torch.randn(1, L, D)
        fn, _ = self._counting_forward_fn(1.0)
        assert "max|grad| = 0.000e+00" in sanity.check_dead_target(
            fn, x, torch.zeros_like(x)
        )["detail"]


# ---------------------------------------------------------------------------
# 5. the VRAM gate comes from igv.gpu and is not re-implemented here
# ---------------------------------------------------------------------------


def test_uses_the_shared_vram_gate():
    """The local 15-line _require_vram copy is gone (one of four; see igv.gpu).

    Identity against ``igv.gpu.require_vram`` is deliberately NOT asserted:
    tests/test_gpu.py calls ``importlib.reload`` on the module, which rebinds
    the function object and would make this order-dependent.
    """
    assert sanity.require_vram.__module__ == "igv.gpu"
    assert sanity.require_vram.__name__ == "require_vram"
    assert not hasattr(sanity, "_require_vram")


def test_dry_run_lists_every_check(capsys):
    """--dry-run must return before the VRAM gate (no GPU on the laptop)."""
    argv = sys.argv
    sys.argv = ["07_sanity.py", "--dataset", "4fqi_h1", "--dry-run"]
    try:
        sanity.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert f"Would run {len(sanity.CHECKS)} check(s)" in out
    for name in sanity.CHECKS:
        assert name in out
