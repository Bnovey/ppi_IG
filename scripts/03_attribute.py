#!/usr/bin/env python3
"""Gradient attribution of a Boltz-2 confidence score w.r.t. the input token embedding."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.boltz_score import (
    SCORES,
    build_complex_feats,
    canonical_aa_token_indices,
    compute_homopolymer_embeddings,
    confidence_forward,
    embedder_only,
    load_model,
    numerics_arm,
    record_iptm_argmax,
    res_type_gradient_forward,
)
from igv.attrib import (
    CANONICAL_AMINO_ACIDS,
    build_mean_aa_baseline,
    completeness_error,
    integrated_gradient,
    plain_gradient,
)
from igv.data import build_library, download, read_pdb_chains
from igv.dms import resolve_pdb_complex
from igv.gpu import require_vram
from igv.provenance import write as prov_write

log = logging.getLogger(__name__)

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}

IPTM_SCORES = {"iptm", "ptm", "protein_iptm"}


def _ig_res_type(
    model, feats, score_name, x_pred,
    baseline_name: str,
    m_steps: int,
    quadrature: str = "gausslegendre",
) -> np.ndarray:
    """Integrated gradient of the score w.r.t. feats["res_type"].

    Interpolates the one-hot res_type tensor along the same quadrature path
    that :func:`igv.attrib.integrated_gradient` uses for the embedding, so the
    two IG results are consistent (same alpha schedule, same number of steps).

    The baseline in res_type space is zeros when the embedding baseline is
    "zeros" or "none", and a uniform 1/20 over the 20 canonical AA columns
    when the embedding baseline is "mean_aa".
    """
    import torch
    from igv.boltz_score import confidence_forward as _cf

    res_type_orig = feats["res_type"]
    L_full = res_type_orig.shape[1]
    num_tokens = res_type_orig.shape[2]

    if baseline_name == "mean_aa":
        from igv.boltz_score import canonical_aa_token_indices as _caa
        aa_idx = _caa()
        rt_baseline = torch.zeros_like(res_type_orig).float()
        rt_baseline[0, :, aa_idx] = 1.0 / len(aa_idx)
    else:
        rt_baseline = torch.zeros_like(res_type_orig).float()

    rt_x = res_type_orig.detach().float()

    if quadrature == "gausslegendre":
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
        alphas_np = (gl_nodes + 1.0) / 2.0
        weights_np = gl_weights / 2.0
    else:
        alphas_np = np.linspace(0.0, 1.0, m_steps + 1)
        weights_np = np.ones(len(alphas_np)) / len(alphas_np)

    accumulated = np.zeros((L_full, num_tokens), dtype=np.float64)

    for step_i, (alpha, weight) in enumerate(zip(alphas_np, weights_np)):
        log.info("res_type IG step %d/%d", step_i + 1, len(alphas_np))

        rt_interp = (rt_baseline + alpha * (rt_x - rt_baseline)).detach().requires_grad_(True)
        feats["res_type"] = rt_interp

        try:
            s_inputs_interp = model.input_embedder(feats)
            scalar = _cf(
                model, s_inputs_interp, feats, x_pred, score_name,
                gradient_checkpointing=True,
            )
            scalar.backward()
        finally:
            feats["res_type"] = res_type_orig

        grad = rt_interp.grad
        if grad is None:
            raise RuntimeError(
                "No gradient on res_type at IG step %d/%d. An intermediate "
                "detach() or torch.no_grad() inside input_embedder severed "
                "the computation graph." % (step_i + 1, len(alphas_np))
            )
        accumulated += weight * grad[0].detach().cpu().float().numpy().astype(np.float64)
        del rt_interp, s_inputs_interp, scalar

    feats["res_type"] = res_type_orig
    return accumulated


def _confidence_forward_out_dict(model, s_inputs, feats, x_pred, score_name):
    """No-grad trunk + confidence pass returning the raw out_dict."""
    import torch

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
        choices=["zeros", "none", "mean_aa"],
        help="Baseline for IG (zeros, none, or mean_aa)",
    )
    parser.add_argument(
        "--no-res-type-grad",
        action="store_true",
        help="Disable the res_type (one-hot) gradient computation",
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

    # min_gib=78 preserved verbatim from the local copy this replaced. Only the
    # skip path differs: IGV_SKIP_VRAM_CHECK=1 now logs a WARNING (it was silent).
    require_vram(min_gib=78)

    # --- 1. Load library and structure ---
    try:
        resolved = resolve_pdb_complex(
            dataset, chain, cache_dir, structure_override=args.structure,
        )
    except KeyError:
        resolved = None

    if resolved is not None:
        data_source = resolved.data_source
        log.info("%s complex %s (chain=%s)", data_source.upper(), resolved.struct_name, chain)
        pdb_path = resolved.pdb_path
        reference_seq = resolved.reference_seq
        struct_name = resolved.struct_name
        chains = resolved.chains
        n_tokens = resolved.n_tokens
        log.info("Chain subset: %s  L=%d", sorted(chains), n_tokens)
    else:
        data_source = "abbibench"
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

        chains = {}
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

    # Per-dataset cache dir. boltz's process_inputs SKIPS any input whose YAML
    # stem is already in <cache_dir>/processed/records, and this repo always
    # writes the stem "input" -- so a shared cache_dir silently returns the
    # FIRST complex ever featurised there. Caught in production on 1JTG, which
    # came back with 4fqi_hlab's asym_id runs [121, 324, 176, 109].
    feat_cache = cache_dir / "boltz_attr" / f"{dataset}_{chain}"
    log.info("Building complex features (use_msa_server=%s, cache=%s)",
             not args.no_msa_server, feat_cache)
    feats, token_map = build_complex_feats(
        chains, pdb_path, feat_cache, args.device,
        use_msa_server=not args.no_msa_server,
    )

    s_inputs = embedder_only(model, feats)
    log.info("s_inputs shape: %s", list(s_inputs.shape))

    # --- 3. Obtain x_pred from feats (fixed geometry) ---
    # Boltz featurisation places the wild-type atom coordinates into
    # feats["coords"] (shape [B, N_atoms, 3]). We detach to ensure geometry
    # is fixed and not differentiated through.
    x_pred = feats["coords"].detach()
    _coords_max = float(x_pred.abs().max())
    log.info(
        "x_pred detached from feats['coords'] (shape %s), coords.abs().max()=%g. "
        "Geometry is FIXED (identical for attribution and the brute-force scan) "
        "but it is NOT necessarily the deposited structure: measured 0.0 on "
        "igv-gpu 2026-09-04, i.e. the sequence-only YAML places every atom at "
        "the origin. structure_pdb is accepted by build_complex_feats and never "
        "read. Do not describe this as wild-type geometry unless this number is "
        "non-zero.",
        list(x_pred.shape),
        _coords_max,
    )
    if _coords_max == 0.0:
        log.warning(
            "coords.abs().max() == 0: attribution is being taken at a collapsed "
            "all-zeros geometry, not at the 4fqi structure. The gradient is "
            "still sequence-dependent (the confidence head sees s and z), but "
            "no claim about structural context is supported."
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
    elif args.baseline == "mean_aa":
        log.info("Computing mean-AA baseline (20 homopolymer embeddings)")
        token_indices_arr = np.array(
            [token_map[(chain, i)] for i in range(len(reference_seq))],
            dtype=np.int64,
        )
        per_aa_embs = compute_homopolymer_embeddings(
            model,
            chains,
            chain,
            pdb_path,
            cache_dir,
            args.device,
            use_msa_server=not args.no_msa_server,
        )
        baseline = build_mean_aa_baseline(s_inputs, token_indices_arr, per_aa_embs)
        log.info("mean_aa baseline built, shape %s", list(baseline.shape))
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

    # --- 6b. Res-type (one-hot) gradient ---
    token_indices = [token_map[(chain, i)] for i in range(len(reference_seq))]
    res_type_grad_enabled = not args.no_res_type_grad
    res_type_extras: dict = {}
    if res_type_grad_enabled:
        log.info("Computing res_type gradient (method=%s)", method)
        t0_rt = time.time()

        aa_token_indices = canonical_aa_token_indices()

        aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}
        wt_indices = np.array(
            [aa_to_idx.get(c, 0) for c in reference_seq], dtype=np.intp,
        )

        if method == "plain_grad":
            rt_result = res_type_gradient_forward(
                model, feats, score,
                gradient_checkpointing=True,
            )
            grad_res_type_full = rt_result["grad_res_type"]  # (L_full, num_tokens)
        elif method == "ig":
            grad_res_type_full = _ig_res_type(
                model, feats, score, x_pred,
                baseline_name=args.baseline,
                m_steps=args.m_steps,
            )

        grad_res_type_chain = grad_res_type_full[token_indices, :]
        if grad_res_type_chain.shape[0] != len(reference_seq):
            raise RuntimeError(
                f"grad_res_type sliced to {grad_res_type_chain.shape[0]} rows "
                f"but reference_seq has {len(reference_seq)} residues"
            )

        res_type_extras = {
            "grad_res_type": grad_res_type_chain.astype(np.float32),
            "aa_token_indices": aa_token_indices,
            "wt_indices": wt_indices,
        }

        elapsed_rt = time.time() - t0_rt
        log.info(
            "res_type gradient: shape %s, computed in %.1f s",
            grad_res_type_chain.shape, elapsed_rt,
        )

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
    # .float() is load-bearing: numpy has no bfloat16, so .numpy() on a non-fp32
    # grad raises "TypeError: Got unsupported ScalarType BFloat16". Today the leaf
    # is fp32 (src/igv/attrib.py builds it that way) so this is a no-op copy-free
    # cast, but it keeps the writer safe if IGV_AUTOCAST ever lets a bf16 grad out.
    grad_full = result.grad.detach().float().cpu().numpy()
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
    save_dict.update(res_type_extras)

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
            "data_source": data_source,
        },
        params={
            "chain": chain,
            "score": score,
            "method": method,
            "m_steps": args.m_steps,
            "baseline": args.baseline,
            "res_type_grad": res_type_grad_enabled,
        },
        arm={
            "score": score,
            "method": method,
            "trunk": "full",
            "m_steps": args.m_steps if method == "ig" else 1,
            "dataset": dataset,
            "chain": chain,
            # The knobs that change the numbers. Two runs differing only in
            # IGV_AUTOCAST give different scores from identical inputs, so an
            # artifact that does not record them cannot be safely compared with
            # any other artifact.
            **numerics_arm(),
            # MEASURED, not claimed. The geometry here is whatever
            # featurisation produced; on a sequence-only YAML that is all
            # zeros, which is not the deposited structure. Recording the number
            # means no reader has to trust a label.
            "coords_abs_max": float(x_pred.abs().max()),
        },
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
