"""Tests for the hoist of the per-script _require_vram copies into igv.gpu.

These are deliberately torch-free and boltz-free: the stage scripts cannot be
imported on a laptop (they pull in boltz at module scope via igv.boltz_score),
so the invariants are checked against the parsed source instead. That is enough
to catch the two failure modes this change exists to prevent:

  1. A local copy of the VRAM gate reappearing and drifting from the shared one
     -- the previous four copies were NOT verbatim, and the stage-02 copy read
     a nonexistent ``.total_mem`` attribute, so the gate it looked like it was
     enforcing was actually an unhandled AttributeError on the first GPU run.
  2. The 78 GiB threshold silently changing at a call site during the hoist.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# scripts/07_sanity.py is hoisted by the same change and is included here on
# purpose: the invariant is about all the GPU stages, not one file.
STAGE_SCRIPTS = [
    "scripts/02_embed_deltas.py",
    "scripts/03_attribute.py",
    "scripts/04_scan.py",
    "scripts/07_sanity.py",
]

EXPECTED_MIN_GIB = 78


def _tree(rel: str) -> ast.Module:
    return ast.parse((REPO / rel).read_text(), rel)


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


@pytest.mark.parametrize("rel", STAGE_SCRIPTS)
def test_no_local_require_vram_definition(rel):
    """The per-script copies are gone; there is one implementation."""
    tree = _tree(rel)
    defs = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "require_vram" in n.name
    ]
    assert defs == [], f"{rel} still defines a local VRAM gate: {defs}"


@pytest.mark.parametrize("rel", STAGE_SCRIPTS)
def test_imports_require_vram_from_igv_gpu(rel):
    """Routed through the shared helper, imported at module scope."""
    tree = _tree(rel)
    imported = [
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "igv.gpu"
        for alias in n.names
    ]
    assert "require_vram" in imported, f"{rel} does not import igv.gpu.require_vram"


@pytest.mark.parametrize("rel", STAGE_SCRIPTS)
def test_calls_require_vram_exactly_once(rel):
    tree = _tree(rel)
    assert len(_calls(tree, "require_vram")) == 1, rel
    # The old private name must not linger anywhere, including in a stale call.
    assert not _calls(tree, "_require_vram"), rel


@pytest.mark.parametrize("rel", STAGE_SCRIPTS)
def test_call_site_threshold_is_unchanged(rel):
    """Any explicit min_gib is still 78; anything else must be intentional.

    The four pre-hoist copies all defaulted to 78 and the one explicit call site
    (stage 03) passed 78, so a bare call or ``min_gib=78`` both preserve today's
    behaviour. A different literal here would quietly move the gate.
    """
    (call,) = _calls(_tree(rel), "require_vram")
    assert call.args == [], f"{rel}: pass min_gib by keyword so it stays greppable"
    for kw in call.keywords:
        if kw.arg == "min_gib":
            assert isinstance(kw.value, ast.Constant), rel
            assert kw.value.value == EXPECTED_MIN_GIB, (
                f"{rel} moved the VRAM gate to {kw.value.value} GiB"
            )


def test_total_mem_typo_is_gone_everywhere():
    """The live bug: ``.total_mem`` does not exist; the attribute is ``.total_memory``.

    Walked as an AST over the whole tree because the defect is a silent
    AttributeError at runtime, not something any import-time check would catch.
    Attribute nodes only, so prose about the bug does not trip the guard.
    """
    offenders = []
    for d in ("scripts", "src"):
        for p in sorted((REPO / d).rglob("*.py")):
            tree = ast.parse(p.read_text(), str(p))
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr == "total_mem":
                    offenders.append(f"{p.relative_to(REPO)}:{n.lineno}")
    assert offenders == [], "reads .total_mem (the attribute is .total_memory): " + ", ".join(offenders)


def test_scan_dry_run_returns_before_the_gate():
    """--dry-run must stay usable with no GPU, so it must precede the gate."""
    src = (REPO / "scripts/04_scan.py").read_text().splitlines()
    dry = next(i for i, ln in enumerate(src) if "if args.dry_run:" in ln)
    gate = next(i for i, ln in enumerate(src) if ln.strip() == "require_vram()")
    assert dry < gate


def test_attribute_casts_grad_to_fp32_before_numpy():
    """numpy has no bfloat16: ``.numpy()`` on a bf16 grad raises TypeError.

    Stage 03 writes grad_full straight into the .npz, so the cast has to happen
    before the host copy. Today the leaf is fp32 and ``.float()`` is a no-op, but
    IGV_AUTOCAST could change the score dtype under it.
    """
    src = (REPO / "scripts/03_attribute.py").read_text()
    line = next(ln for ln in src.splitlines() if ln.strip().startswith("grad_full ="))
    assert ".float()" in line
    assert line.index(".float()") < line.index(".numpy()")


def test_require_vram_still_honours_the_skip_env(monkeypatch):
    """The one behaviour every call site depended on: IGV_SKIP_VRAM_CHECK=1 does not raise.

    Skipped rather than failed if igv.gpu is not present yet, so this file is
    useful on its own; the shared helper's own semantics are covered by
    tests/test_gpu.py.
    """
    gpu = pytest.importorskip("igv.gpu")
    monkeypatch.setenv("IGV_SKIP_VRAM_CHECK", "1")
    result = gpu.require_vram()
    assert isinstance(result, float)
