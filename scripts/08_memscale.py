#!/usr/bin/env python3
"""Stage 08 (DIAGNOSTIC) -- measure the true VRAM requirement as a function of L.

Not part of the pipeline. Deliberately absent from ``make all`` and
``scripts/run_all.sh``: it is expected to OOM at its largest points, and a
deliberately-OOMing sweep in the default path would break every full run.

WHY THIS EXISTS
---------------
Every "peak VRAM" number this project has recorded so far comes from a run that
died. The nine OOM-truncated rows from early profiling (77.64 - 78.64 GiB) and the
79.21 GiB / 78.43 GiB pair in ``results/sanity_4fqi_h1_complex_pde.json`` are
the allocator hitting the wall, i.e. TRUNCATED LOWER BOUNDS on the requirement,
not the requirement. A run that got further would have reported *more*, not
less. So the true cost of a full-trunk backward at L=730 is, as of today,
unknown -- and no go/no-go on bf16 autocast or finer checkpointing can be taken
against an unknown.

This script measures it the only way that is valid: sweep L over complexes
small enough to FINISH, run the real forward+backward TO COMPLETION, record the
peak with an explicit completed/OOM flag, and fit log(peak) ~ a + b*log(L) on
the COMPLETED rows only in order to extrapolate to 730.

Literature expectation, to be confirmed rather than assumed: MegaFold reports
AF3 EvoAttention activations of 3.75 GB at L=96 rising to 24.61 GB at L=192,
which is an exponent of log(24.61/3.75)/log(2) ~ 2.7. If this sweep's fitted
``b`` lands far from that, distrust the sweep before distrusting the paper.

FOUR THINGS THAT MAKE OR BREAK THE MEASUREMENT
----------------------------------------------
1. ``torch.cuda.max_memory_allocated`` is process-global and MONOTONIC. Without
   an explicit ``reset_peak_memory_stats`` per point, every row reports the same
   largest-so-far number and the CSV looks entirely plausible. ``PeakMemory``
   (``igv.gpu``) does the reset on ``__enter__``; we therefore open a FRESH
   context per sweep point and never reuse one. ``free_cuda_memory()``
   (``igv.attrib``) does NOT reset peak stats -- it only gc/synchronises/empties
   the cache -- so it is no substitute.

2. ONE chunk profile across the whole sweep. boltz's
   ``const.chunk_size_threshold`` is 384, and below it a *different algorithm*
   runs: pairformer triangle-attention chunk 512 instead of 128, and all four
   MSA chunk knobs off. Fitting across that cliff mixes two algorithms into one
   curve and the extrapolation to 730 is meaningless. We force
   ``IGV_CHUNK_PROFILE=large`` and hard-refuse to run if the forced profile is
   not actually L-invariant.

3. A unique boltz ``cache_dir`` per point. ``process_inputs`` skips any input
   whose YAML stem is already processed, and this repo always writes the stem
   "input" (``boltz_score.py``), so a shared directory silently returns the
   FIRST point's features -- i.e. a flat memory curve that looks like a
   discovery. Mirrors the correct per-point pattern in ``scripts/04_scan.py``.

4. Only ``completed=True`` rows enter the fit, and the number excluded plus the
   reason is logged and written into the artifact. Silent truncation reading as
   "we covered everything" is the exact failure class this script corrects.

VARYING L NEEDS NO NEW MACHINERY: a complex is just ``dict[chain_id -> seq]``
and L is the sum of chain lengths (asserted inside ``build_complex_feats``). We
use WHOLE-CHAIN SUBSETS of one real complex, never truncated chains --
``_build_token_map`` raises when two chains share a length and neither sorted
nor insertion order matches the observed asym_id runs, which is precisely what
uniform truncation produces.

Example
-------
    python scripts/08_memscale.py --dry-run
    python scripts/08_memscale.py --dataset 4fqi_h1 --score complex_pde
    python scripts/08_memscale.py --fit-only --out results/memscale_4fqi_h1_complex_pde.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import read_pdb_chains  # noqa: E402
from igv.provenance import _git_commit, environment  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402

log = logging.getLogger("memscale")

# 4fqi_h1 and 4fqi_h3 share one deposited complex (same table as 04_scan.py).
_STRUCTURE_FOR = {"4fqi_h1": "4fqi_hlab", "4fqi_h3": "4fqi_hlab"}

# boltz-2.2.1 src/boltz/data/const.py: chunk_size_threshold = 384. Kept here
# only for reporting which ladder points sit on which side of the cliff; the
# sweep forces one profile so the cliff is not crossed.
CHUNK_SIZE_THRESHOLD = 384

# The extrapolation target: the production complex.
TARGET_TOKENS = 730

# Chain lengths of data/raw/4fqi_hlab.pdb, MEASURED with igv.data.read_pdb_chains
# in this repo: A=324, B=176, H=121, L=109, total 730 -- matching the 730 tokens
# the pipeline actually reports. An earlier debug log recorded "A:336 + B:185,
# Fab H:123 + L:109" (summing to 753) from a misparsed chain split; the correct
# counts above supersede it. Used ONLY as a fallback so --dry-run can print a plan on a laptop with
# no PDB downloaded; a real run re-parses the PDB.
FQI_CHAIN_LENGTHS: dict[str, int] = {"A": 324, "B": 176, "H": 121, "L": 109}

# Whole-chain subsets of 4fqi_hlab, ascending in token count:
#   H+L = 230, H+L+B = 406, A+B = 500, H+L+A = 554, A+B+H+L = 730.
# Same system, real sequences, no truncation, one PDB to download. The confound
# to keep in view (recorded in every row as n_chains, and in the artifact notes):
# chain count and MSA pairing change with the subset -- boltz fetches paired MSAs
# only when there is more than one protein entity -- so part of any residual
# scatter is "which chains", not "how many tokens".
CHAIN_SUBSET_LADDER: tuple[tuple[str, ...], ...] = (
    ("H", "L"),
    ("H", "L", "B"),
    ("A", "B"),
    ("H", "L", "A"),
    ("A", "B", "H", "L"),
)

# Cross-system alternative. Reader 5 measured these thirteen totals with
# read_pdb_chains over igv.data._STRUCTURE_NAMES:
#   239, 245, 245, 248, 320, 355, 409, 523, 648, 730, 731, 811, 1030
# The structure -> total MAPPING was not recorded with them, so it is NOT
# reproduced here as a lookup table: guessing it would be exactly the kind of
# plausible-looking fabrication this repo keeps getting burned by. Only the one
# value verified locally is asserted. `--ladder cross-system` therefore resolves
# sizes by parsing each PDB with read_pdb_chains at run time, and --dry-run
# prints sizes for whichever PDBs are already in --cache-dir.
CROSS_SYSTEM_MEASURED_TOTALS: tuple[int, ...] = (
    239, 245, 245, 248, 320, 355, 409, 523, 648, 730, 731, 811, 1030,
)
VERIFIED_TOKEN_COUNTS: dict[str, int] = {"4fqi_hlab": 730}

# Non-PyTorch CUDA context overhead, measured from the recorded OOM in
# results/sanity_4fqi_h1_complex_pde.json: 79.21 GiB in use by the process vs
# 78.43 GiB allocated by PyTorch. max_memory_allocated does not see it, so every
# peak in this artifact -- and hence the fit -- UNDERSTATES the true requirement
# by roughly this much, plus fragmentation.
CUDA_CONTEXT_OVERHEAD_GIB = 0.78

_COLUMNS = [
    "n_tokens",
    "n_chains",
    "subset",
    "chains",
    "completed",
    "oom",
    "status",
    "truncated",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "resident_gib_before",
    "current_allocated_gib_at_failure",
    "grad_abs_max",
    "wall_s",
    "error",
    "pf_chunk",
    "pf_group",
    "chunk_profile",
    "chunk_profile_wired",
    "recycling_steps",
    "gradient_checkpointing",
    "autocast",
    "tri_attn_ckpt",
    "use_kernels",
    "msa_depth",
    "msa_spec",
    "score",
    "gpu_total_gib",
    "torch_version",
    "boltz_version",
    "git_commit",
]


# ---------------------------------------------------------------------------
# Planning (pure, CPU-testable)
# ---------------------------------------------------------------------------


def chain_subset_plan(
    chain_lengths: dict[str, int],
    subsets: tuple[tuple[str, ...], ...] = CHAIN_SUBSET_LADDER,
) -> list[dict]:
    """Turn whole-chain subsets into sweep points, ascending in token count.

    Returns one dict per point with ``chains`` (tuple of ids), ``n_tokens`` and
    ``n_chains``. Subsets naming a chain absent from ``chain_lengths`` are
    dropped with a warning rather than silently mis-sized.
    """
    points: list[dict] = []
    for subset in subsets:
        missing = [c for c in subset if c not in chain_lengths]
        if missing:
            log.warning(
                "Skipping subset %s: chain(s) %s not in the structure (have %s)",
                "".join(subset), missing, sorted(chain_lengths),
            )
            continue
        points.append(
            {
                "chains": tuple(subset),
                "n_tokens": sum(chain_lengths[c] for c in subset),
                "n_chains": len(subset),
            }
        )
    points.sort(key=lambda p: p["n_tokens"])
    return points


def select_sizes(points: list[dict], sizes: list[int] | None) -> list[dict]:
    """Filter a plan down to the requested token counts, in ascending order."""
    if not sizes:
        return points
    by_size: dict[int, dict] = {}
    for p in points:
        if p["n_tokens"] in by_size:
            # The cross-system ladder has genuine ties (two structures both
            # measured at 245 tokens). Keep the first so --sizes stays a
            # one-to-one selector instead of silently picking the last.
            log.warning(
                "Two ladder points share %d tokens; --sizes will select the "
                "first (%s), not %s.",
                p["n_tokens"], by_size[p["n_tokens"]].get("structure", by_size[p["n_tokens"]]["chains"]),
                p.get("structure", p["chains"]),
            )
            continue
        by_size[p["n_tokens"]] = p
    unknown = [s for s in sizes if s not in by_size]
    if unknown:
        raise SystemExit(
            f"--sizes {unknown} not available in this ladder. "
            f"Available token counts: {sorted(by_size)}"
        )
    return [by_size[s] for s in sorted(set(sizes))]


# ---------------------------------------------------------------------------
# The fit (pure, CPU-testable)
# ---------------------------------------------------------------------------


def fit_loglog(n_tokens, peaks, target: int = TARGET_TOKENS) -> dict:
    """Least-squares fit of log(peak) ~ a + b*log(L); extrapolate to ``target``.

    ``b`` is the scaling exponent. Requires at least 3 points and at least 2
    distinct token counts -- with one distinct L the slope is unidentifiable and
    R^2 is undefined, which is reported as NaN rather than as 1.0.
    """
    x = np.log(np.asarray(n_tokens, dtype=float))
    y = np.log(np.asarray(peaks, dtype=float))
    if x.size < 3:
        raise ValueError(f"need >= 3 points to fit, got {x.size}")
    if np.unique(x).size < 2:
        raise ValueError(
            f"need >= 2 distinct token counts to fit a slope, got {np.unique(x).size}"
        )
    design = np.vstack([np.ones_like(x), x]).T
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - (intercept + slope * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "exponent_b": float(slope),
        "intercept_a": float(intercept),
        "r2": float(r2),
        "n_points": int(x.size),
        "n_distinct_tokens": int(np.unique(x).size),
        "token_range": [int(min(n_tokens)), int(max(n_tokens))],
        "residual_max_abs_log": float(np.abs(resid).max()),
        "target_tokens": int(target),
        "predicted_peak_gib_at_target": float(
            math.exp(intercept + slope * math.log(target))
        ),
    }


def as_bool(value) -> bool:
    """Strict truthiness for a boolean that has round-tripped through CSV.

    ``bool("False")`` is ``True``. If the ``completed`` column ever comes back
    as strings -- a hand-edited CSV, a locale, a different pandas -- the naive
    read reclassifies every OOM row as a finished measurement, which is the
    single most dangerous misread this artifact allows: the sweep would then
    report the card's size as the model's requirement and look right doing it.
    So anything not recognisably boolean raises instead of guessing.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0", ""):
            return False
    raise ValueError(
        f"Cannot read {value!r} as a boolean. The completed/oom columns decide "
        "which rows are real measurements and which are truncated lower bounds; "
        "refusing to guess."
    )


def json_safe(obj):
    """Recursively make an object JSON-clean: no NaN/Inf, no numpy scalars.

    ``json.dumps`` emits bare ``NaN``, which is not valid JSON and which some
    readers turn into a number. A missing peak must read as ``null``, not as a
    value.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def partition_rows(rows: list[dict]) -> dict:
    """Split rows into fit-eligible and excluded, with the reason for each.

    A row is fit-eligible ONLY if it completed and carries a real peak. Anything
    else is a lower bound on the requirement, and a lower bound averaged into a
    power-law fit drags the exponent toward zero -- i.e. makes the model look
    cheaper than it is, in exactly the direction that gets a run scheduled and
    then killed.
    """
    usable: list[dict] = []
    excluded: list[dict] = []
    for r in rows:
        if not as_bool(r.get("completed")):
            reason = (
                "oom (peak is a truncated lower bound)"
                if as_bool(r.get("oom"))
                else f"failed: {r.get('error')}"
            )
            excluded.append({**r, "exclusion_reason": reason})
            continue
        peak = r.get("peak_allocated_gib")
        if peak is None or not np.isfinite(float(peak)) or float(peak) <= 0:
            excluded.append(
                {**r, "exclusion_reason": f"no usable peak recorded ({peak!r})"}
            )
            continue
        usable.append(r)
    return {"usable": usable, "excluded": excluded}


def assert_single_profile(rows: list[dict]) -> str:
    """Refuse to fit across mixed chunk profiles. Returns the single profile.

    Raises ``ValueError`` rather than exiting so the caller can still write the
    rows out -- the measurements are valuable even when the fit is refused.
    """
    profiles = sorted({str(r.get("chunk_profile")) for r in rows})
    if len(profiles) != 1:
        raise ValueError(
            f"Refusing to fit across mixed chunk profiles {profiles}. Below "
            f"boltz's chunk_size_threshold={CHUNK_SIZE_THRESHOLD} a different "
            "algorithm runs (pairformer chunk 512 not 128, all four MSA chunk "
            "knobs off), so a fit spanning both is two algorithms in one curve. "
            "Re-run the sweep with a single --chunk-profile."
        )
    return profiles[0]


# ---------------------------------------------------------------------------
# Chunk-profile forcing
# ---------------------------------------------------------------------------


def force_chunk_profile(profile: str, token_counts: list[int], strict: bool = True):
    """Force one chunk profile for the whole sweep and PROVE it took effect.

    Sets ``IGV_CHUNK_PROFILE`` and then checks that ``chunk_profile(L)`` returns
    the *same* configuration for the smallest and largest L in the plan. That is
    the property the fit depends on; asserting it directly beats trusting the
    env var to have been honoured.

    Returns ``(config_dict_or_None, wired_flag_or_None)``. With ``strict``,
    a missing knob or an L-dependent config is fatal.
    """
    os.environ["IGV_CHUNK_PROFILE"] = profile

    try:
        from igv.boltz_score import chunk_profile
    except ImportError:
        msg = (
            "igv.boltz_score.chunk_profile is not available, so IGV_CHUNK_PROFILE "
            "cannot be verified to take effect. Without a forced profile every "
            f"point below {CHUNK_SIZE_THRESHOLD} tokens runs a different "
            "algorithm and the extrapolation to 730 is invalid."
        )
        if strict:
            raise SystemExit(msg)
        log.warning("%s (tolerated for --dry-run)", msg)
        return None, None

    lo, hi = min(token_counts), max(token_counts)
    cfg_lo, cfg_hi = chunk_profile(lo), chunk_profile(hi)
    if cfg_lo != cfg_hi:
        msg = (
            f"IGV_CHUNK_PROFILE={profile!r} did NOT force a single profile: "
            f"chunk_profile({lo}) = {cfg_lo} but chunk_profile({hi}) = {cfg_hi}. "
            "The sweep would mix two algorithms into one curve."
        )
        if strict:
            raise SystemExit(msg)
        log.warning("%s (tolerated for --dry-run)", msg)
    else:
        log.info(
            "Chunk profile FORCED to %r and verified L-invariant over %d..%d: %s",
            profile, lo, hi, cfg_lo,
        )

    # Wiring check, recorded rather than assumed: chunk_profile() honouring the
    # env var is worthless if confidence_forward never calls it. Source
    # inspection is coarse (a later refactor into a helper would read False), so
    # this is reported in every row, not enforced.
    wired = None
    try:
        import inspect

        from igv.boltz_score import confidence_forward

        wired = "chunk_profile" in inspect.getsource(confidence_forward)
    except Exception as exc:  # pragma: no cover - diagnostics only
        log.warning("Could not inspect confidence_forward for wiring: %r", exc)
    log.info("chunk_profile is wired into confidence_forward: %s", wired)
    if wired is False:
        log.warning(
            "confidence_forward's source does not mention chunk_profile. If the "
            "knob is genuinely unwired, IGV_CHUNK_PROFILE is a no-op and this "
            "sweep is invalid. Recorded as chunk_profile_wired=False in every "
            "row -- verify before interpreting the fit."
        )
    return cfg_lo, wired


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _allocated_gib(torch_module, index: int) -> float | None:
    """Currently-allocated GiB, or None when there is no CUDA to ask.

    Guarded because ``IGV_SKIP_VRAM_CHECK=1`` lets the sweep reach this point on
    a machine with no GPU, and a crash here would destroy the row it is trying
    to annotate.
    """
    try:
        if not torch_module.cuda.is_available():
            return None
        return torch_module.cuda.memory_allocated(index) / 2**30
    except Exception:  # pragma: no cover - diagnostics must never crash a row
        return None


def _cuda_index(device: str) -> int:
    if ":" in str(device):
        try:
            return int(str(device).split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _feats_kwargs(msa_spec: str) -> dict:
    """Build the MSA kwargs for build_complex_feats, adapting to its signature.

    ``msa_spec='empty'`` needs the ``msa_spec`` parameter that change
    ``feats-sequence-assert`` adds. Without it, ``use_msa_server=False`` and no
    ``msa`` key makes boltz RAISE, so the sweep is not offline as the code
    stands -- say so loudly instead of failing five minutes in.
    """
    import inspect

    from igv.boltz_score import build_complex_feats

    params = inspect.signature(build_complex_feats).parameters
    if "msa_spec" in params:
        return {"msa_spec": msa_spec, "use_msa_server": msa_spec == "server"}
    if msa_spec != "server":
        log.warning(
            "build_complex_feats has no msa_spec parameter, so --msa-spec=%r "
            "falls back to use_msa_server=False. boltz raises when a sequence "
            "has no msa key and the server is off; expect every point to fail "
            "with that error until change feats-sequence-assert lands.",
            msa_spec,
        )
    return {"use_msa_server": msa_spec == "server"}


def point_label(point: dict) -> str:
    """Stable identifier for one sweep point.

    The structure name on the cross-system ladder (where two structures can
    share both a token count and a chain-id set), the chain subset within one
    complex. Used for the row's ``subset`` field, the per-point cache dir and
    the resume key, so those three can never disagree -- a resume key that
    disagrees with the row it wrote re-measures points forever or, worse, skips
    ones that were never done.
    """
    return str(point.get("structure") or "".join(point["chains"]))


def _row_key(row) -> tuple[int, str]:
    return int(row["n_tokens"]), str(row["subset"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_notes(msa_spec: str, ladder: str) -> str:
    lines = [
        "DIAGNOSTIC sweep, not a pipeline stage.",
        "peak_allocated_gib is torch.cuda.max_memory_allocated with peak stats "
        "reset per point.",
        "Rows with completed=False are TRUNCATED LOWER BOUNDS -- the allocator "
        "hit the wall, so the requirement is at least that and unknown above. "
        "They are excluded from the fit and counted in fit.excluded_rows.",
        f"max_memory_allocated does not see the ~{CUDA_CONTEXT_OVERHEAD_GIB} GiB "
        "of non-PyTorch CUDA context overhead (the recorded OOM shows 79.21 GiB "
        "in use by the process vs 78.43 GiB allocated by PyTorch), so both the "
        "measured peaks and the extrapolation UNDERSTATE the true requirement by "
        "roughly that much plus fragmentation.",
        "The nine OOM-truncated peaks from early profiling are lower bounds "
        "from runs that never finished and must not be compared with these as "
        "if they were requirements.",
        "Literature expectation for the exponent is ~2.7 (MegaFold: AF3 "
        "EvoAttention activations 3.75 GB at L=96 -> 24.61 GB at L=192). "
        "UNCONFIRMED for this code path -- confirm against fit.exponent_b, do "
        "not assume it.",
    ]
    if ladder == "chain-subset":
        lines.append(
            "Confound: chain count and MSA pairing change with the chain subset "
            "(boltz fetches paired MSAs only when there is more than one protein "
            "entity), so n_chains is a second variable alongside n_tokens. "
            "Recorded per row."
        )
    else:
        lines.append(
            "Confound: cross-system points differ in sequence, chain count and "
            "MSA depth as well as length, so the residual scatter is not purely "
            "length-driven."
        )
    lines.append(
        "msa_depth is recorded per point because MSA depth is unpadded and "
        "data-dependent (pad_to_max_seqs=False), making it a second uncontrolled "
        "variable that directly drives the checkpointed MSA term (3.25 GiB at "
        "L=730 as measured in the checkpointed MSA forward)."
    )
    if msa_spec != "server":
        lines.append(
            f"WARNING: msa_spec={msa_spec!r}. These peaks are NOT comparable with "
            "the 730-token production numbers, which were measured WITH MSAs."
        )
    return " ".join(lines)


def log_plan(args, points: list[dict], out_csv: Path, profile_cfg, resolved_env: dict) -> None:
    log.info("Ladder: %s (%d points)", args.ladder, len(points))
    for p in points:
        side = "large" if p["n_tokens"] > CHUNK_SIZE_THRESHOLD else "small"
        label = point_label(p)
        log.info(
            "  L=%-5d point=%-11s n_chains=%d natural_profile=%s cache_dir=%s",
            p["n_tokens"], label, p["n_chains"], side,
            Path(args.cache_dir) / f"boltz_mem/{p['n_tokens']}_{label}",
        )
    n_above = sum(1 for p in points if p["n_tokens"] > CHUNK_SIZE_THRESHOLD)
    log.info(
        "%d/%d points sit above boltz's chunk_size_threshold=%d; the forced "
        "profile %r applies to all of them either way",
        n_above, len(points), CHUNK_SIZE_THRESHOLD, args.chunk_profile,
    )
    log.info("Pinned knobs (None = module default, inherited unchanged): %s", resolved_env)
    log.info("Forced chunk profile config: %s", profile_cfg)
    log.info(
        "Per point: build_complex_feats -> embedder_only -> plain_gradient "
        "(one forward + one backward, gradient_checkpointing=%s, "
        "recycling_steps=%d, score=%s, msa_spec=%s)",
        args.gradient_checkpointing, args.recycling_steps, args.score, args.msa_spec,
    )
    log.info("Rows appended to %s; artifact + provenance to %s", out_csv, out_csv.with_suffix(".json"))


def finalise(out_csv: Path, out_json: Path, args, resolved_env: dict, profile_cfg) -> int:
    """Fit the completed rows and write the JSON artifact. Returns an exit code."""
    if not out_csv.exists():
        raise SystemExit(f"No rows in {out_csv}; nothing to fit.")
    frame = pd.read_csv(out_csv)
    rows = frame.to_dict("records")
    part = partition_rows(rows)
    usable, excluded = part["usable"], part["excluded"]

    log.info(
        "Fit input: %d/%d rows completed; %d excluded",
        len(usable), len(rows), len(excluded),
    )
    for e in excluded:
        log.warning(
            "EXCLUDED from fit: L=%s chains=%s -- %s (peak %.2f GiB is a LOWER "
            "BOUND, not a requirement)",
            e.get("n_tokens"), e.get("subset"), e.get("exclusion_reason"),
            float(e.get("peak_allocated_gib") or float("nan")),
        )

    fit: dict | None = None
    fit_error: str | None = None
    if len(usable) < 3:
        fit_error = (
            f"Only {len(usable)} COMPLETED row(s); need >= 3 to fit log(peak) vs "
            f"log(L). {len(excluded)} row(s) were excluded as truncated lower "
            "bounds. Extend the sweep downward (smaller chain subsets) until at "
            "least three points finish -- do NOT fit the OOM peaks; they are the "
            "card's size, not the model's cost."
        )
        log.error("%s", fit_error)
    else:
        try:
            profile = assert_single_profile(usable)
            fit = fit_loglog(
                [int(r["n_tokens"]) for r in usable],
                [float(r["peak_allocated_gib"]) for r in usable],
                target=args.target_tokens,
            )
        except ValueError as exc:
            # Refusing to fit is a result. Write the rows anyway; a refused fit
            # with the measurements preserved beats a lost sweep.
            fit, fit_error = None, str(exc)
            log.error("Refusing to fit: %s", exc)
    if fit is not None:
        fit["chunk_profile"] = profile
        fit["excluded_rows"] = len(excluded)
        fit["excluded_reasons"] = [
            {
                "n_tokens": e.get("n_tokens"),
                "subset": e.get("subset"),
                "reason": e.get("exclusion_reason"),
                "lower_bound_peak_gib": e.get("peak_allocated_gib"),
            }
            for e in excluded
        ]
        fit["predicted_peak_gib_at_target_plus_context"] = (
            fit["predicted_peak_gib_at_target"] + CUDA_CONTEXT_OVERHEAD_GIB
        )
        log.info(
            "FIT (completed rows only, n=%d, profile=%s): exponent b=%.3f, "
            "intercept a=%.3f, R^2=%.4f",
            fit["n_points"], profile, fit["exponent_b"], fit["intercept_a"], fit["r2"],
        )
        log.info(
            "Extrapolated requirement at L=%d: %.1f GiB allocated (%.1f GiB "
            "including the ~%.2f GiB CUDA context overhead PyTorch does not "
            "count). Literature expects b ~ 2.7; this run measured %.3f.",
            fit["target_tokens"], fit["predicted_peak_gib_at_target"],
            fit["predicted_peak_gib_at_target_plus_context"],
            CUDA_CONTEXT_OVERHEAD_GIB, fit["exponent_b"],
        )

    artifact = {
        "stage": "08_memscale",
        "ladder": args.ladder,
        "score": args.score,
        "dataset": args.dataset,
        "target_tokens": args.target_tokens,
        "chunk_profile_requested": args.chunk_profile,
        "chunk_profile_config": profile_cfg,
        "pinned_env": resolved_env,
        "recycling_steps": args.recycling_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "msa_spec": args.msa_spec,
        "n_rows": len(rows),
        "n_completed": len(usable),
        "n_excluded": len(excluded),
        "rows": rows,
        "fit": fit,
        "fit_error": fit_error,
        "cuda_context_overhead_gib": CUDA_CONTEXT_OVERHEAD_GIB,
        "notes": build_notes(args.msa_spec, args.ladder),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    prov_write(
        out_json,
        stage="08_memscale",
        inputs={"dataset": args.dataset, "rows_csv": str(out_csv)},
        params={
            "ladder": args.ladder,
            "sizes": [int(r["n_tokens"]) for r in rows],
            "recycling_steps": args.recycling_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "msa_spec": args.msa_spec,
            "pinned_env": resolved_env,
            "target_tokens": args.target_tokens,
        },
        arm={
            "score": args.score,
            "method": "plain_grad",
            "trunk": "full",
            "measurement": "peak_vram",
            "chunk_profile": args.chunk_profile,
            "dataset": args.dataset,
        },
        notes=artifact["notes"],
    )
    log.info("Wrote %s (+ provenance sidecar)", out_json)
    return 1 if fit is None else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="4fqi_h1")
    p.add_argument("--score", default="complex_pde")
    p.add_argument(
        "--ladder",
        default="chain-subset",
        choices=["chain-subset", "cross-system"],
        help="chain-subset: whole-chain subsets of one complex (preferred -- "
             "same system, real sequences, one PDB). cross-system: other "
             "structures from igv.data._STRUCTURE_NAMES.",
    )
    p.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated token counts to restrict the ladder to, "
             "e.g. 230,406,500. Default: every point in the ladder.",
    )
    p.add_argument("--structure", default=None, help="Structure stem override")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--out", default="results/memscale_{dataset}_{score}.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    p.add_argument(
        "--msa-spec",
        default="server",
        choices=["server", "empty"],
        help="server: query ColabFold (comparable with production numbers). "
             "empty: offline single-sequence MSA -- peaks are NOT comparable.",
    )
    # --- pinned knobs. Every point in a sweep must see identical settings or
    # the extrapolation is only comparable to the one historical row that
    # happened to match. `None` means "do not set the env var, inherit and
    # record whatever the module default resolves to" -- so this script never
    # silently changes behaviour that a plain run would have had.
    p.add_argument(
        "--chunk-profile",
        default="large",
        choices=["large", "small"],
        help="Forced for the whole sweep. Defaults to 'large' FOR THIS STAGE "
             "specifically: below boltz's chunk_size_threshold=384 a different "
             "algorithm runs, and a fit across that cliff is invalid.",
    )
    p.add_argument(
        "--pf-chunk",
        default="128",
        help="IGV_PF_CHUNK, pinned across the sweep. 128 is exactly the value "
             "the large profile selects, so this is a no-op reinforcement of "
             "--chunk-profile large; it also pins IGV_PF_CHUNK's second reader, "
             "the MSA triangle chunk. Pass '' to inherit instead.",
    )
    p.add_argument("--pf-group-size", default=None, help="IGV_PF_GROUP_SIZE (default: inherit)")
    p.add_argument("--autocast", default=None, help="IGV_AUTOCAST (default: inherit)")
    p.add_argument("--tri-attn-ckpt", default=None, help="IGV_TRI_ATTN_CKPT (default: inherit)")
    p.add_argument("--use-kernels", default=None, help="IGV_USE_KERNELS (default: inherit)")
    p.add_argument(
        "--recycling-steps",
        type=int,
        default=1,
        help="confidence_forward kwarg; the trunk runs recycling_steps+1 "
             "iterations. Passed explicitly so it is pinned and recorded rather "
             "than left to the callee default.",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Measure the uncheckpointed peak. Expect an immediate OOM at any "
             "interesting L; recorded as such.",
    )
    p.set_defaults(gradient_checkpointing=True)
    p.add_argument(
        "--fit-only",
        action="store_true",
        help="Re-fit an existing rows CSV and rewrite the artifact. No GPU, no "
             "boltz -- use it to analyse a sweep brought back from the VM.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the pinned knobs, then stop. No GPU, no boltz.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    out_csv = Path(args.out.format(dataset=args.dataset, score=args.score))
    out_json = out_csv.with_suffix(".json")
    cache_dir = Path(args.cache_dir)
    sizes = (
        [int(s) for s in str(args.sizes).replace(",", " ").split()]
        if args.sizes
        else None
    )

    # --- Pin the knobs BEFORE anything imports/reads them. -----------------
    requested_env = {
        "IGV_PF_CHUNK": args.pf_chunk,
        "IGV_PF_GROUP_SIZE": args.pf_group_size,
        "IGV_AUTOCAST": args.autocast,
        "IGV_TRI_ATTN_CKPT": args.tri_attn_ckpt,
        "IGV_USE_KERNELS": args.use_kernels,
    }
    for key, value in requested_env.items():
        if value is not None and str(value) != "":
            os.environ[key] = str(value)
    resolved_env = {
        "IGV_CHUNK_PROFILE": args.chunk_profile,
        **{k: os.environ.get(k) for k in requested_env},
    }

    # --- Resolve the ladder. ----------------------------------------------
    stem = args.structure or _STRUCTURE_FOR.get(args.dataset)
    pdb = cache_dir / f"{stem}.pdb" if stem else None

    if args.ladder == "chain-subset":
        if stem is None:
            raise SystemExit(
                f"No structure known for {args.dataset}; pass --structure."
            )
        if pdb.exists():
            chain_lengths = {c: len(s) for c, s in read_pdb_chains(pdb).items()}
            log.info("Structure %s chains (parsed): %s", stem, chain_lengths)
            total = sum(chain_lengths.values())
            if stem in VERIFIED_TOKEN_COUNTS and total != VERIFIED_TOKEN_COUNTS[stem]:
                log.warning(
                    "%s parses to %d tokens but %d was measured previously; the "
                    "ladder is still built from the parsed lengths.",
                    stem, total, VERIFIED_TOKEN_COUNTS[stem],
                )
        elif stem == "4fqi_hlab":
            chain_lengths = dict(FQI_CHAIN_LENGTHS)
            log.warning(
                "%s not in %s; planning from the recorded lengths %s. Run "
                "scripts/00_fetch_data.py before a real sweep.",
                pdb, cache_dir, chain_lengths,
            )
        else:
            raise SystemExit(
                f"Missing {pdb} and no recorded chain lengths for {stem!r}. "
                "Run scripts/00_fetch_data.py first."
            )
        points = select_sizes(chain_subset_plan(chain_lengths), sizes)
    else:
        from igv.data import _STRUCTURE_NAMES

        log.info(
            "cross-system ladder. Reader 5 measured these thirteen totals with "
            "read_pdb_chains: %s -- the structure->total mapping was not "
            "recorded, so sizes are resolved by parsing each PDB here.",
            list(CROSS_SYSTEM_MEASURED_TOTALS),
        )
        points = []
        for name in _STRUCTURE_NAMES:
            path = cache_dir / f"{name}.pdb"
            if not path.exists():
                log.warning("  %s: not downloaded, cannot size it here", name)
                continue
            lengths = {c: len(s) for c, s in read_pdb_chains(path).items()}
            points.append(
                {
                    "chains": tuple(sorted(lengths)),
                    "n_tokens": sum(lengths.values()),
                    "n_chains": len(lengths),
                    "structure": name,
                    "chain_lengths": lengths,
                }
            )
            log.info("  %s: %d tokens, chains %s", name, points[-1]["n_tokens"], lengths)
        points.sort(key=lambda p: p["n_tokens"])
        points = select_sizes(points, sizes)

    if not points:
        raise SystemExit(
            "Empty ladder. For --ladder cross-system, download the structures "
            "with scripts/00_fetch_data.py first."
        )

    profile_cfg, wired = force_chunk_profile(
        args.chunk_profile,
        [p["n_tokens"] for p in points],
        strict=not (args.dry_run or args.fit_only),
    )

    if args.fit_only:
        return finalise(out_csv, out_json, args, resolved_env, profile_cfg)

    log_plan(args, points, out_csv, profile_cfg, resolved_env)

    # Resume: one row is appended per point, so an OOM or a kill mid-sweep
    # loses nothing.
    done: set[tuple[int, str]] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {_row_key(r) for _, r in prev.iterrows()}
        log.info("Resuming: %d point(s) already measured in %s", len(done), out_csv)
    todo = [p for p in points if (p["n_tokens"], point_label(p)) not in done]

    if args.dry_run:
        # Returns BEFORE require_vram() and before any artifact is written, so
        # this path works on a laptop with no torch-CUDA and no boltz.
        log.info(
            "--dry-run: would measure %d of %d point(s), at L=%s, then fit and "
            "write the artifact. Stopping before require_vram(); no GPU touched, "
            "nothing written.",
            len(todo), len(points), [p["n_tokens"] for p in todo],
        )
        return 0

    if not todo:
        log.info("All %d ladder points already measured; re-fitting.", len(points))
        return finalise(out_csv, out_json, args, resolved_env, profile_cfg)

    # ---------------------------------------------------------------------
    # Everything below needs a GPU and boltz.
    # ---------------------------------------------------------------------
    from igv.gpu import PeakMemory, require_vram

    gpu_total_gib = require_vram()

    import torch

    from igv.attrib import free_cuda_memory, plain_gradient
    from igv.boltz_score import (
        SCORES,
        build_complex_feats,
        confidence_forward,
        embedder_only,
        load_model,
    )

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    # Record the EFFECTIVE group size, not None. IGV_PF_GROUP_SIZE unset means
    # boltz_score's own default, and a row that just says "None" for the
    # dominant checkpointing knob cannot be compared with anything later.
    import igv.boltz_score as _bs

    pf_group_effective = os.environ.get("IGV_PF_GROUP_SIZE")
    if pf_group_effective is None:
        pf_group_effective = getattr(_bs, "_PF_GROUP_SIZE", None)
    resolved_env["IGV_PF_GROUP_SIZE"] = pf_group_effective
    log.info("Effective pairformer checkpoint group size: %s", pf_group_effective)

    env = environment()
    commit = _git_commit(Path(__file__).resolve().parents[1])
    feats_kwargs = _feats_kwargs(args.msa_spec)
    cuda_idx = _cuda_index(args.device)

    model, boltz_version = load_model(args.checkpoint_dir, args.device)

    if args.ladder == "chain-subset":
        if not pdb.exists():
            raise SystemExit(
                f"Missing {pdb}. The plan above was built from recorded chain "
                "lengths, which is fine for --dry-run but not for a measurement. "
                "Run scripts/00_fetch_data.py first."
            )
        full_chains = read_pdb_chains(pdb)

        def sequences_for(point):
            return {c: full_chains[c] for c in point["chains"]}
    else:
        def sequences_for(point):
            return read_pdb_chains(cache_dir / f"{point['structure']}.pdb")

    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for n, point in enumerate(todo, 1):
        subset = point_label(point)
        chains = sequences_for(point)
        # Own cache_dir per point. A shared one would silently hand back the
        # FIRST point's features (boltz skips already-processed YAML stems and
        # this repo always writes the stem "input"), producing a flat curve.
        point_cache = cache_dir / f"boltz_mem/{point['n_tokens']}_{subset}"
        point_pdb = (
            pdb if args.ladder == "chain-subset"
            else cache_dir / f"{point['structure']}.pdb"
        )

        row = {
            "n_tokens": point["n_tokens"],
            "n_chains": point["n_chains"],
            "subset": subset,
            "chains": json.dumps({c: len(s) for c, s in chains.items()}, sort_keys=True),
            "completed": False,
            "oom": False,
            "status": "error",
            "truncated": True,
            "peak_allocated_gib": None,
            "peak_reserved_gib": None,
            "resident_gib_before": None,
            "current_allocated_gib_at_failure": None,
            "grad_abs_max": None,
            "wall_s": None,
            "error": None,
            "pf_chunk": (profile_cfg or {}).get("pf_chunk"),
            "pf_group": pf_group_effective,
            "chunk_profile": args.chunk_profile,
            "chunk_profile_wired": wired,
            "recycling_steps": args.recycling_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "autocast": resolved_env.get("IGV_AUTOCAST"),
            "tri_attn_ckpt": resolved_env.get("IGV_TRI_ATTN_CKPT"),
            "use_kernels": resolved_env.get("IGV_USE_KERNELS"),
            "msa_depth": None,
            "msa_spec": args.msa_spec,
            "score": args.score,
            "gpu_total_gib": gpu_total_gib,
            "torch_version": env.get("torch"),
            "boltz_version": boltz_version or env.get("boltz"),
            "git_commit": commit,
        }

        point_t0 = time.time()
        result = None
        try:
            feats, _ = build_complex_feats(
                chains, point_pdb, point_cache, args.device, **feats_kwargs
            )
            # MSA depth is unpadded and data-dependent, so it varies with the
            # subset independently of L and drives the checkpointed MSA term.
            row["msa_depth"] = json.dumps(list(tuple(feats["msa"].shape)))
            x_pred = feats["coords"].detach()
            s_inputs = embedder_only(model, feats)

            def forward_fn(s, _feats=feats, _x=x_pred):
                return confidence_forward(
                    model, s, _feats, _x, args.score,
                    gradient_checkpointing=args.gradient_checkpointing,
                    recycling_steps=args.recycling_steps,
                )

            # A FRESH PeakMemory per point: __enter__ calls
            # reset_peak_memory_stats, without which max_memory_allocated stays
            # at the largest value seen so far in the process and every row
            # reports the same plausible-looking number. The reset rebases the
            # peak on what is currently resident (weights, feats, s_inputs), so
            # the recorded peak is the whole-process requirement, not just the
            # backward's transient -- resident_gib_before makes the split
            # visible.
            row["resident_gib_before"] = _allocated_gib(torch, cuda_idx)
            with PeakMemory(
                device=cuda_idx, reset=True, suppress_oom=True,
                label=f"L={point['n_tokens']}:{subset}",
            ) as mem:
                result = plain_gradient(forward_fn, s_inputs)

            row["peak_allocated_gib"] = mem.peak_gib
            row["peak_reserved_gib"] = mem.reserved_gib
            row["completed"] = bool(mem.completed)
            row["oom"] = bool(mem.oom)
            row["error"] = mem.error
            row["truncated"] = not bool(mem.completed)
            row["status"] = "completed" if mem.completed else ("oom" if mem.oom else "error")
            if not mem.completed:
                row["current_allocated_gib_at_failure"] = _allocated_gib(torch, cuda_idx)
            if result is not None:
                # Evidence the backward really produced a gradient. A completed
                # run with an all-zero gradient is not a measurement of the
                # attribution path's cost.
                row["grad_abs_max"] = float(result.grad.detach().abs().max().item())
        except Exception as exc:  # noqa: BLE001 -- one bad point must not end the sweep
            row["error"] = repr(exc)
            row["status"] = "error"
            row["completed"] = False
            row["truncated"] = True
            log.exception("L=%d (%s) failed outside the measured region", point["n_tokens"], subset)
        finally:
            row["wall_s"] = time.time() - point_t0
            # Drop this point's tensors BEFORE the next point allocates, so one
            # point's features are never resident during the next point's peak.
            # forward_fn holds feats/x_pred in default arguments, so it has to
            # go too. Plain assignment (not del) because an early failure may
            # have left some of these unbound.
            result = feats = x_pred = s_inputs = forward_fn = None
            # free_cuda_memory does NOT reset peak stats -- that is PeakMemory's
            # job -- it only releases the cache so the next point starts clean.
            free_cuda_memory()

        pd.DataFrame([row], columns=_COLUMNS).to_csv(
            out_csv, mode="a", header=write_header, index=False
        )
        write_header = False

        rate = (time.time() - t0) / n
        log.info(
            "[%d/%d] L=%d (%s) status=%s peak=%s GiB reserved=%s GiB (%.1fs/point, "
            "eta %.0f min)",
            n, len(todo), point["n_tokens"], subset, row["status"],
            f"{row['peak_allocated_gib']:.2f}" if row["peak_allocated_gib"] is not None else "n/a",
            f"{row['peak_reserved_gib']:.2f}" if row["peak_reserved_gib"] is not None else "n/a",
            rate, rate * (len(todo) - n) / 60,
        )
        if row["status"] != "completed":
            log.warning(
                "L=%d did not complete (%s). Its peak is a LOWER BOUND and will "
                "be excluded from the fit.", point["n_tokens"], row["status"],
            )

    log.info("Sweep finished in %.1f min", (time.time() - t0) / 60)
    return finalise(out_csv, out_json, args, resolved_env, profile_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
