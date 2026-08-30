#!/usr/bin/env python3
"""Gradient attribution of a Boltz-2 confidence score w.r.t. the input token embedding."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.boltz_score import (
    SCORES,
    build_complex_feats,
    confidence_forward,
    embedder_only,
    load_model,
    record_iptm_argmax,
)
from igv.attrib import (
    completeness_error,
    free_cuda_memory,
    integrated_gradient,
    plain_gradient,
)
from igv.data import build_library, download, read_pdb_chains
from igv.provenance import write as prov_write

log = logging.getLogger(__name__)

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}

IPTM_SCORES = {"iptm", "ptm", "protein_iptm"}


def _require_vram(min_gib: float = 80) -> None:
    if os.environ.get("IGV_SKIP_VRAM_CHECK") == "1":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available. Attribution requires >= {min_gib} GiB VRAM. "
            "Set IGV_SKIP_VRAM_CHECK=1 to override."
        )
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    if total_gib < min_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.1f} GiB VRAM, need >= {min_gib} GiB. "
            "Set IGV_SKIP_VRAM_CHECK=1 to override."
        )


def _confidence_forward_out_dict(model, s_inputs, feats, x_pred, score_name):
    """No-grad trunk + confidence pass returning the raw out_dict."""
    import torch
    from boltz.data import const as _boltz_const

    device = s_inputs.device
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    rel_pos = model.rel_pos(feats)
    token_bonds_z = model.token_bonds(feats["token_bonds"].float())
    contact_z = model.contact_conditioning(feats)

    s_init = model.s_init(s_inputs)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None, :]
        + model.z_init_2(s_inputs)[:, None, :, :]
        + rel_pos + token_bonds_z + contact_z
    )

    s_ = torch.zeros_like(s_init)
    z_ = torch.zeros_like(z_init)

    for _ in range(2):  # recycling_steps=1 → 2 iterations
        s_ = s_init + model.s_recycle(model.s_norm(s_))
        z_ = z_init + model.z_recycle(model.z_norm(z_))
        z_ = z_ + model.msa_module(z_, s_inputs, feats, use_kernels=False)
        s_, z_ = model.pairformer_module(
            s_, z_, mask=mask, pair_mask=pair_mask, use_kernels=False,
        )

    pdistogram = model.distogram_module(z_)
    pred_distogram_logits = pdistogram[:, :, :, 0]

    out_dict = model.confidence_module(
        s_inputs=s_inputs,
        s=s_,
        z=z_,
        x_pred=x_pred,
        feats=feats,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=1,
        run_sequentially=False,
        use_kernels=False,
    )
    return out_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient attribution of Boltz-2 confidence score w.r.t. token embedding"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. 4fqi_h1)")
    parser.add_argument("--chain", default="H", help="Varying chain id")
    parser.add_argument(
        "--score",
        required=True,
        choices=sorted(SCORES.keys()),
        help="Confidence score to differentiate",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["plain_grad", "ig"],
        help="Attribution method",
    )
    parser.add_argument("--m-steps", type=int, default=15, help="IG quadrature steps")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument(
        "--out",
        default="results/{dataset}_{score}_{method}_grad.npz",
        help="Output .npz path",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Boltz checkpoint dir (default: $BOLTZ_CACHE or ~/.boltz)",
    )
    parser.add_argument("--structure", default=None, help="Override structure name")
    parser.add_argument(
        "--no-msa-server", action="store_true", help="Disable MSA server queries"
    )
    parser.add_argument(
        "--baseline",
        default="zeros",
        choices=["zeros", "none"],
        help="Baseline for IG (zeros or none)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz"))
    checkpoint_dir = Path(checkpoint_dir)

    cache_dir = Path(args.cache_dir)
    dataset = args.dataset
    chain = args.chain
    score = args.score
    method = args.method
    out_path = Path(args.out.format(
        dataset=dataset, score=score, method=method,
    ))

    _require_vram(min_gib=80)

    # --- 1. Load library and structure ---
    log.info("Building library for %s (chain=%s)", dataset, chain)
    lib = build_library(dataset, cache_dir, chain=chain)
    reference_seq = lib.reference_seq

    struct_name = args.structure or STRUCTURE_FOR_DATASET.get(dataset)
    if struct_name is None:
        raise ValueError(
            f"No default structure for dataset {dataset!r}. "
            "Provide --structure explicitly."
        )
    pdb_path = download(struct_name, "structure", cache_dir)
    pdb_chains = read_pdb_chains(pdb_path)

    chains: dict[str, str] = {}
    for ch_id, seq in pdb_chains.items():
        if ch_id == chain:
            chains[ch_id] = reference_seq
        else:
            chains[ch_id] = seq
    log.info(
        "Complex chains: %s",
        {c: len(s) for c, s in chains.items()},
    )

    # --- 2. Load model and featurise ---
    log.info("Loading model from %s", checkpoint_dir)
    model, boltz_version = load_model(checkpoint_dir, args.device)

    log.info("Building complex features (use_msa_server=%s)", not args.no_msa_server)
    feats, token_map = build_complex_feats(
        chains, pdb_path, cache_dir, args.device,
        use_msa_server=not args.no_msa_server,
    )

    s_inputs = embedder_only(model, feats)
    log.info("s_inputs shape: %s", list(s_inputs.shape))

    # --- 3. Obtain x_pred from feats (fixed geometry) ---
    # Boltz featurisation places the wild-type atom coordinates into
    # feats["coords"] (shape [B, N_atoms, 3]). We detach to ensure geometry
    # is fixed and not differentiated through.
    x_pred = feats["coords"].detach()
    log.info(
        "GEOMETRY IS FIXED: x_pred detached from feats['coords'] (shape %s). "
        "Both attribution and the brute-force scan measure sensitivity at "
        "identical wild-type geometry.",
        list(x_pred.shape),
    )

    # --- 4. Build forward_fn ---
    def forward_fn(s):
        return confidence_forward(
            model, s, feats, x_pred, score,
            gradient_checkpointing=True,
        )

    # --- 5. Baseline ---
    if args.baseline == "zeros":
        baseline = torch.zeros_like(s_inputs)
    else:
        baseline = None

    # --- 6. Dispatch attribution ---
    log.info("Running %s attribution (score=%s)", method, score)
    t0 = time.time()

    if method == "plain_grad":
        result = plain_gradient(forward_fn, s_inputs, baseline=baseline)
    elif method == "ig":
        result = integrated_gradient(
            forward_fn, s_inputs,
            baseline=baseline,
            m_steps=args.m_steps,
            log_progress=True,
            clear_cache_each_step=True,
        )

    elapsed = time.time() - t0
    log.info("Attribution completed in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # --- 7. f_x and completeness ---
    with torch.no_grad():
        f_x = float(forward_fn(s_inputs))
    log.info("f(x) = %.6f", f_x)

    f_baseline_val = float("nan")
    comp_err = float("nan")
    if method == "ig":
        if baseline is None:
            baseline_eval = torch.zeros_like(s_inputs)
        else:
            baseline_eval = baseline
        with torch.no_grad():
            f_baseline_val = float(forward_fn(baseline_eval))
        log.info("f(baseline) = %.6f", f_baseline_val)
        comp_err = completeness_error(result, f_x, f_baseline_val)
        log.info(
            "Completeness error: %.4f (%.2f%%) — %s 5%% criterion",
            comp_err,
            comp_err * 100,
            "PASSES" if comp_err < 0.05 else "FAILS",
        )

    # --- 8. Slice gradient to varying chain ---
    token_indices = [token_map[(chain, i)] for i in range(len(reference_seq))]
    grad_full = result.grad.detach().cpu().numpy()
    grad_chain = grad_full[0, token_indices, :]
    log.info("grad_chain shape: (%d, %d)", *grad_chain.shape)

    # --- 9. ipTM argmax ---
    argmax_record: dict = {}
    if score in IPTM_SCORES:
        token_to_chain: dict[int, str] = {}
        for (ch_id, resi), tok_idx in token_map.items():
            token_to_chain[tok_idx] = ch_id

        log.info("Running no-grad trunk pass to obtain out_dict for ipTM argmax")
        with torch.no_grad():
            out_dict = _confidence_forward_out_dict(
                model, s_inputs, feats, x_pred, score,
            )
        argmax_record = record_iptm_argmax(out_dict, token_to_chain)
        log.info(
            "ipTM argmax: token=%s chain=%s resi=%s — %s",
            argmax_record.get("argmax_token"),
            argmax_record.get("argmax_chain"),
            argmax_record.get("argmax_resi"),
            "argmax frame is IN the attributed chain"
            if argmax_record.get("argmax_chain") == chain
            else "argmax frame is OUTSIDE the attributed chain",
        )
        del out_dict

    # --- 10. Save .npz ---
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "grad_chain": grad_chain,
        "grad_full": grad_full,
        "token_indices": np.array(token_indices, dtype=np.int64),
        "f_x": np.float64(f_x),
        "f_baseline": np.float64(f_baseline_val),
        "completeness_error": np.float64(comp_err),
        "reference_seq": np.array(reference_seq),
        "chain": np.array(chain),
        "score": np.array(score),
        "method": np.array(method),
        "m_steps": np.int64(args.m_steps),
    }
    if argmax_record:
        for k, v in argmax_record.items():
            save_dict[f"argmax_{k}" if not k.startswith("argmax_") else k] = (
                np.array(v) if v is not None else np.array(None)
            )

    np.savez(out_path, **save_dict)
    log.info("Wrote %s", out_path)

    # --- 11. Provenance ---
    prov_write(
        out_path,
        stage="03_attribute",
        inputs={
            "dataset": dataset,
            "structure": struct_name,
            "cache_dir": str(cache_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
        params={
            "chain": chain,
            "score": score,
            "method": method,
            "m_steps": args.m_steps,
            "baseline": args.baseline,
        },
        arm={
            "score": score,
            "method": method,
            "trunk": "full",
            "m_steps": args.m_steps if method == "ig" else 1,
            "dataset": dataset,
            "chain": chain,
        },
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
