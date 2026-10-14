#!/usr/bin/env python3
"""Memory-ladder diagnostic for Protenix attribution.

Measures peak VRAM at different token counts and recycle depths.
Run on GPU only; skipped when CUDA is unavailable.

Usage:
    python scripts/08_memscale_protenix.py --sizes 300,450 --cycles 1,2,4
    python scripts/08_memscale_protenix.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

log = logging.getLogger(__name__)


def _measure_one(
    model, input_feature_dict, x_pred, score_name, n_cycle, device,
):
    """Run one forward+backward and return peak VRAM in GiB."""
    import torch
    from igv.protenix_score import confidence_forward

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    restype = input_feature_dict["restype"].detach().float().requires_grad_(True)

    t0 = time.time()
    scalar = confidence_forward(
        model, restype, input_feature_dict, x_pred, score_name,
        n_cycle=n_cycle, gradient_checkpointing=True,
    )
    scalar.backward()
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gib = peak_bytes / (1024 ** 3)

    grad_ok = restype.grad is not None and restype.grad.abs().sum() > 0

    del restype, scalar
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "peak_gib": round(peak_gib, 2),
        "elapsed_s": round(elapsed, 1),
        "grad_ok": grad_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protenix memory ladder")
    parser.add_argument(
        "--sizes", default="300,450",
        help="Comma-separated token counts to test",
    )
    parser.add_argument(
        "--cycles", default="1,2,4,10",
        help="Comma-separated recycle depths",
    )
    parser.add_argument("--score", default="complex_pde")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    sizes = [int(s) for s in args.sizes.split(",")]
    cycles = [int(c) for c in args.cycles.split(",")]

    if args.dry_run:
        log.info("DRY RUN: would test sizes=%s cycles=%s score=%s", sizes, cycles, args.score)
        for s in sizes:
            for c in cycles:
                log.info("  L=%d, N_cycle=%d", s, c)
        return

    import torch

    if not torch.cuda.is_available():
        log.error("CUDA not available; memory ladder requires a GPU.")
        sys.exit(1)

    from igv.protenix_score import load_model

    log.info("Loading model...")
    model, configs = load_model(
        args.device, weights_dir=args.weights_dir, n_cycle=max(cycles),
    )

    results = []
    for L in sizes:
        seq_a = "A" * (L // 2)
        seq_b = "G" * (L - L // 2)

        log.info("Building features for L=%d...", L)
        from igv.protenix_score import build_features

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ifd, _ = build_features(
                {"A": seq_a, "B": seq_b},
                Path(td), args.device, configs, model=model,
            )

        n_atoms = ifd["ref_pos"].shape[-2]
        x_pred = torch.zeros(1, 1, n_atoms, 3, device=args.device)

        for nc in cycles:
            log.info("Measuring L=%d, N_cycle=%d ...", L, nc)
            try:
                r = _measure_one(model, ifd, x_pred, args.score, nc, args.device)
                r["L"] = L
                r["n_cycle"] = nc
                results.append(r)
                log.info(
                    "  L=%d N_cycle=%d: %.2f GiB, %.1f s, grad_ok=%s",
                    L, nc, r["peak_gib"], r["elapsed_s"], r["grad_ok"],
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("  L=%d N_cycle=%d: OOM", L, nc)
                results.append({"L": L, "n_cycle": nc, "peak_gib": None, "elapsed_s": None,
                                "grad_ok": False})
                gc.collect()
                torch.cuda.empty_cache()

    log.info("--- Results ---")
    for r in results:
        log.info("  %s", r)


if __name__ == "__main__":
    main()
