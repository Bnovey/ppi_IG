"""Boltz-2 confidence-score surface for gradient attribution.

Wraps the Boltz-2 confidence head so that a single scalar score is
differentiable w.r.t. the token-level input embedding ``s_inputs``.
The trunk is run with per-block gradient checkpointing to fit within
80 GB VRAM during backward (the same strategy as the predecessor repo).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score registry
# ---------------------------------------------------------------------------

# Pairformer blocks per gradient checkpoint. 4 keeps the retained boundary
# tensors to 16 per recycling iteration (~4.1 GiB at 730 tokens, down from
# ~16.3 GiB at one block per checkpoint) while recomputing only 4 blocks at a
# time. Override with IGV_PF_GROUP_SIZE.
_PF_GROUP_SIZE = 4

SCORES: dict[str, Callable[[dict], Any]] = {}


def _register_score(name: str, fn: Callable) -> None:
    SCORES[name] = fn


def _complex_pde(out_dict):
    return out_dict["complex_pde"]


def _complex_iplddt(out_dict):
    return out_dict["complex_iplddt"]


def _complex_plddt(out_dict):
    return out_dict["complex_plddt"]


def _iptm(out_dict):
    return out_dict["iptm"]


def _ptm(out_dict):
    return out_dict["ptm"]


def _protein_iptm(out_dict):
    return out_dict["protein_iptm"]


_register_score("complex_pde", _complex_pde)
_register_score("complex_iplddt", _complex_iplddt)
_register_score("complex_plddt", _complex_plddt)
_register_score("iptm", _iptm)
_register_score("ptm", _ptm)
_register_score("protein_iptm", _protein_iptm)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def select_checkpoint(checkpoint_dir):
    """Return the Boltz-2 *confidence* checkpoint inside *checkpoint_dir*.

    A full ``download_boltz2`` leaves two checkpoints in the cache::

        boltz2_aff.ckpt    affinity head   -- sorts FIRST alphabetically
        boltz2_conf.ckpt   confidence head -- what this module needs

    Every score registered here (iptm, ptm, complex_pde, ...) reads the
    confidence head, so a naive ``sorted(...)[0]`` picks the affinity model and
    the pipeline would attribute gradients of the wrong network without ever
    raising. Kept torch-free so it is testable on CPU.
    """
    from pathlib import Path as _P

    ckpt_dir = _P(checkpoint_dir)
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    preferred = [c for c in candidates if "aff" not in c.name.lower()]
    if not preferred:
        raise FileNotFoundError(
            f"Only affinity checkpoints found in {ckpt_dir} "
            f"({[c.name for c in candidates]}). The confidence checkpoint "
            f"(boltz2_conf.ckpt) is required; re-run scripts/cloud/fetch_weights.sh."
        )
    conf = [c for c in preferred if "conf" in c.name.lower()]
    chosen = conf[0] if conf else preferred[0]
    if len(candidates) > 1:
        log.info(
            "Checkpoints present: %s -- selected %s",
            [c.name for c in candidates], chosen.name,
        )
    return chosen


def load_model(checkpoint_dir, device):
    """Load Boltz-2 in eval mode from *checkpoint_dir*.

    Returns ``(model, boltz_version)`` where *boltz_version* is
    the installed ``boltz`` package version string.
    """
    import importlib.metadata as md

    import torch
    from boltz.main import BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    from dataclasses import asdict

    boltz_version = md.version("boltz")

    ckpt_path = select_checkpoint(checkpoint_dir)
    log.info("Loading Boltz-2 from %s (boltz %s)", ckpt_path, boltz_version)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" in ckpt:
        _scrub_hparams(ckpt["hyper_parameters"], {"mse_rotational_alignment"})
        _ensure_pairformer_v2(ckpt["hyper_parameters"])
        _ensure_confidence_prediction(ckpt["hyper_parameters"])

    import os
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    try:
        torch.save(ckpt, tmp_path)
        model = Boltz2.load_from_checkpoint(
            tmp_path, map_location=device, strict=False, ema=False,
        )
    finally:
        os.unlink(tmp_path)

    model.confidence_prediction = True
    if model.steering_args is None:
        steering = BoltzSteeringParams()
        steering.fk_steering = False
        steering.physical_guidance_update = False
        steering.contact_guidance_update = False
        model.steering_args = asdict(steering)
    model.eval()
    model.to(device)
    log.info("Boltz-2 loaded, confidence_module present: %s",
             hasattr(model, "confidence_module") and model.confidence_module is not None)
    return model, boltz_version


def _scrub_hparams(obj, remove_keys):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        for k in remove_keys:
            try:
                obj.pop(k, None)
            except Exception:
                pass
        try:
            vals = list(obj.values())
        except Exception:
            vals = []
        for v in vals:
            _scrub_hparams(v, remove_keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scrub_hparams(v, remove_keys)


def _ensure_pairformer_v2(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        if "pairformer_args" in obj:
            pa = obj["pairformer_args"]
            if isinstance(pa, Mapping) or hasattr(pa, "items"):
                try:
                    pa["v2"] = True
                except Exception:
                    pass
        try:
            for v in obj.values():
                _ensure_pairformer_v2(v)
        except Exception:
            pass
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _ensure_pairformer_v2(v)


def _ensure_confidence_prediction(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        try:
            obj["confidence_prediction"] = True
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------


def build_complex_feats(
    chains: dict[str, str],
    structure_pdb: Path,
    cache_dir: Path,
    device,
    use_msa_server: bool = True,
) -> tuple[dict, dict[tuple[str, int], int]]:
    """Featurise a multi-chain complex for Boltz-2.

    Parameters
    ----------
    chains : dict[str, str]
        Maps chain id to amino-acid sequence, e.g.
        ``{"H": "EVQL...", "L": "DIQM...", "A": "MKYL..."}``.
    structure_pdb : Path
        Wild-type complex PDB used as the fixed structure reference.
    cache_dir : Path
        Scratch directory for Boltz intermediate files.
    device : torch.device
        Target device.
    use_msa_server : bool
        Whether to query ColabFold for MSAs.

    Returns
    -------
    feats : dict[str, Tensor]
        Collated feature dict for ``model.forward()`` / ``confidence_module``.
    token_map : dict[(chain_id, residue_index), token_index]
        Maps ``(chain_id, 0-based residue index)`` to the global token index
        in the feats tensors.
    """
    import yaml
    import torch
    from boltz.main import process_inputs

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    for chain_id, seq in chains.items():
        sequences.append({"protein": {"id": chain_id, "sequence": seq}})

    spec = {"version": 1, "sequences": sequences}
    spec_path = cache_dir / "input.yaml"
    spec_path.write_text(yaml.dump(spec))

    proc_kwargs, cache_root = _boltz_process_inputs_kwargs(use_msa_server)
    process_inputs(data=[spec_path], out_dir=cache_dir, **proc_kwargs)

    mol_dir = proc_kwargs.get("mol_dir", cache_root / "mols")
    feats = _featurize_processed(cache_dir, mol_dir, device)

    token_map = _build_token_map(chains, feats)

    expected_tokens = sum(len(s) for s in chains.values())
    n_pad = int(feats["token_pad_mask"][0].sum().item())
    assert n_pad == expected_tokens, (
        f"Token count mismatch: expected {expected_tokens} from chain "
        f"sequences, got {n_pad} non-padding tokens in feats"
    )

    return feats, token_map


def _boltz_process_inputs_kwargs(use_msa_server: bool) -> tuple[dict, "Path"]:
    import inspect
    import urllib.request
    from boltz.main import get_cache_path, process_inputs

    cache_root = Path(get_cache_path())
    proc_kwargs: dict = {"use_msa_server": use_msa_server}
    sig = inspect.signature(process_inputs)
    if "msa_server_url" in sig.parameters:
        proc_kwargs["msa_server_url"] = "https://api.colabfold.com"
    if "msa_pairing_strategy" in sig.parameters:
        proc_kwargs["msa_pairing_strategy"] = "greedy"
    if "ccd_path" in sig.parameters:
        ccd_path = cache_root / "ccd.pkl"
        if not ccd_path.exists():
            urllib.request.urlretrieve(
                "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl",
                ccd_path,
            )
        proc_kwargs["ccd_path"] = ccd_path
    if "mol_dir" in sig.parameters:
        proc_kwargs["mol_dir"] = cache_root / "mols"
    if "boltz2" in sig.parameters:
        proc_kwargs["boltz2"] = True
    return proc_kwargs, cache_root


_FEATS_SKIP_DEVICE = frozenset({
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
})


def _featurize_processed(proc_out_dir: Path, mol_dir: Path, device) -> dict:
    import torch
    from boltz.data.types import Manifest

    manifest_path = proc_out_dir / "processed" / "manifest.json"
    manifest = Manifest.load(manifest_path)
    processed = proc_out_dir / "processed"
    constraints_dir = processed / "constraints"
    templates_dir = processed / "templates"
    extra_mols_dir = processed / "mols"

    try:
        from boltz.data.module.inferencev2 import PredictionDataset, collate
        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=mol_dir,
            constraints_dir=constraints_dir if constraints_dir.exists() else None,
            template_dir=templates_dir if templates_dir.exists() else None,
            extra_mols_dir=extra_mols_dir if extra_mols_dir.exists() else None,
            affinity=False,
        )
    except ImportError:
        from boltz.data.module.inference import PredictionDataset, collate
        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            constraints_dir=constraints_dir if constraints_dir.exists() else None,
        )

    sample = dataset[0]
    if sample is None:
        raise RuntimeError("Boltz PredictionDataset returned None")
    sample.pop("record", None)
    feats = collate([sample])
    for key, val in feats.items():
        if key in _FEATS_SKIP_DEVICE:
            continue
        if isinstance(val, torch.Tensor):
            feats[key] = val.to(device)
    return feats


def _build_token_map(
    chains: dict[str, str], feats: dict
) -> dict[tuple[str, int], int]:
    """Build ``(chain_id, residue_index) -> global token index``.

    The chain-to-token ordering is *derived* from ``feats["asym_id"]`` rather
    than assumed.  Boltz assigns each token an ``asym_id``; tokens of one chain
    are contiguous, so the run lengths of ``asym_id`` give the actual layout.

    This matters because the obvious assumption -- that token order follows the
    alphabetically sorted chain ids -- is not verifiable from the YAML we write,
    and the token *count* assertion passes under any ordering.  A mismatch would
    therefore misalign every per-residue attribution silently, with no error.
    For the 4fqi complex the two candidate orderings differ (sorted gives
    A=324, B=176, H=121, L=109; insertion order gives H, L, A, B), so comparing
    run lengths against the expected chain lengths detects the wrong guess.

    Raises
    ------
    RuntimeError
        If the number of chain runs does not match the number of chains, or if
        no ordering of the supplied chains reproduces the observed run lengths.
    """
    pad = feats["token_pad_mask"][0].bool().cpu()
    asym = feats["asym_id"][0].cpu()[pad]

    # Contiguous runs of equal asym_id == one chain each.
    runs: list[tuple[int, int, int]] = []  # (asym_value, start, length)
    run_start = 0
    for i in range(1, len(asym) + 1):
        if i == len(asym) or int(asym[i]) != int(asym[run_start]):
            runs.append((int(asym[run_start]), run_start, i - run_start))
            run_start = i

    if len(runs) != len(chains):
        raise RuntimeError(
            f"Found {len(runs)} contiguous asym_id runs but {len(chains)} chains "
            f"were supplied: run lengths {[r[2] for r in runs]}, chain lengths "
            f"{ {c: len(s) for c, s in chains.items()} }. Cannot map tokens to "
            "chains; attribution slicing would be wrong."
        )

    observed = [r[2] for r in runs]

    # Try the two plausible orderings, then fall back to a length-multiset match.
    candidates = [sorted(chains.keys()), list(chains.keys())]
    order: list[str] | None = None
    for cand in candidates:
        if [len(chains[c]) for c in cand] == observed:
            order = cand
            break
    if order is None:
        # Unique-length chains can still be matched unambiguously by length.
        by_len: dict[int, list[str]] = {}
        for c, s in chains.items():
            by_len.setdefault(len(s), []).append(c)
        if all(len(by_len.get(n, [])) == 1 for n in observed):
            order = [by_len[n][0] for n in observed]
        else:
            raise RuntimeError(
                "Could not determine chain order from asym_id runs. Observed run "
                f"lengths {observed}; chain lengths "
                f"{ {c: len(s) for c, s in chains.items()} }. Two chains share a "
                "length and neither sorted nor insertion order matches, so the "
                "mapping is ambiguous."
            )

    token_map: dict[tuple[str, int], int] = {}
    for chain_id, (_asym_val, offset, length) in zip(order, runs):
        seq = chains[chain_id]
        if length != len(seq):
            raise RuntimeError(
                f"Chain {chain_id} has {len(seq)} residues but its token run is "
                f"{length} long."
            )
        for resi in range(length):
            token_map[(chain_id, resi)] = offset + resi
    return token_map


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------


def embedder_only(model, feats):
    """Run only the input embedder under no_grad. Returns ``s_inputs`` (1, N, D)."""
    import torch
    with torch.no_grad():
        return model.input_embedder(feats)


def enable_confidence_checkpointing(model) -> int:
    """Per-block gradient checkpointing inside the CONFIDENCE pairformer stack.

    The trunk (``pairformer_module`` / ``msa_module``) is checkpointed per block
    by :func:`confidence_forward`, but ``confidence_module.pairformer_stack`` was
    not, so during the outer checkpoint's backward recompute every one of its
    layers' L x L activations materialised at once. On a 753-token complex
    (4fqi: HA 336+185 + Fab 123+109) that overruns an 80 GiB A100, OOMing at
    ``transition.py: silu(fc1(x)) * fc2(x)`` while trying to allocate 1.02 GiB
    with 78.43 GiB already held.

    This is the same failure the predecessor repo hit and fixed in the trunk --
    "checkpointed the whole pairformer_module as ONE unit, so backward recompute
    still materialized all 64 blocks' activations at once" -- just relocated to
    the one module that still fit at their smaller L~540.

    Boltz's PairformerModule already implements exactly this, but gates it on
    ``self.activation_checkpointing and self.training``. Calling ``.train()`` to
    unlock it is not an option: it enables dropout, which makes the score
    stochastic and its gradient meaningless, and it sets
    ``chunk_size_tri_attn = None``, removing the triangle-attention chunking
    that eval mode gives us. So we rebind ``forward`` instead, keeping the
    module in eval and replicating upstream's eval-mode chunk selection.

    Idempotent. Returns the number of layers now checkpointed.
    """
    import types

    from torch.utils.checkpoint import checkpoint as _ckpt

    confidence = getattr(model, "confidence_module", None)
    stack = getattr(confidence, "pairformer_stack", None)
    if stack is None:
        log.warning("No confidence_module.pairformer_stack; nothing to checkpoint.")
        return 0
    if getattr(stack, "_igv_checkpointed", False):
        return len(stack.layers)

    try:
        from boltz.data import const as _c
        _threshold = _c.chunk_size_threshold
    except Exception:
        _threshold = 384

    def _checkpointed_forward(self, s, z, mask, pair_mask, use_kernels: bool = False):
        # Mirrors PairformerModule.forward's eval-mode branch.
        chunk_size_tri_attn = 128 if z.shape[1] > _threshold else 512
        for layer in self.layers:
            s, z = _ckpt(
                layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels,
                use_reentrant=False,
            )
        return s, z

    stack.forward = types.MethodType(_checkpointed_forward, stack)
    stack._igv_checkpointed = True
    log.info(
        "Confidence pairformer stack: per-block checkpointing enabled (%d layers)",
        len(stack.layers),
    )
    return len(stack.layers)


def confidence_forward(
    model,
    s_inputs,
    feats,
    x_pred,
    score_name: str,
    gradient_checkpointing: bool = True,
    recycling_steps: int = 1,
):
    """Run the full trunk from ``s_inputs``, then the confidence head.

    Returns the requested scalar score as a differentiable tensor.

    The trunk is run with per-block gradient checkpointing to fit within
    80 GB: each pairformer and MSA block is individually checkpointed so
    that backward recompute never materialises all 64 blocks' activations
    simultaneously. When ``gradient_checkpointing`` is set, the confidence
    module's own pairformer stack gets the same treatment -- see
    :func:`enable_confidence_checkpointing`.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as _checkpoint
    from boltz.data import const as _boltz_const

    if score_name not in SCORES:
        raise ValueError(
            f"Unknown score {score_name!r}; available: {sorted(SCORES)}"
        )

    if gradient_checkpointing:
        # The confidence stack needs the same per-block treatment as the trunk;
        # without it the backward recompute OOMs at 753 tokens. See
        # enable_confidence_checkpointing for why .train() is not the answer.
        enable_confidence_checkpointing(model)

    device = s_inputs.device
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

    _n_tokens = int(mask.shape[1])
    try:
        _threshold = _boltz_const.chunk_size_threshold
    except Exception:
        _threshold = 384
    # Triangle-attention chunk size. This is the dominant VRAM term at scale:
    # the attention weights are (chunk, heads, L, L), so at L=730 with chunk
    # 128 and 4 heads each chunk is ~1.09 GiB, and a peak profile attributed
    # 25.22 GiB to 26 live ones (primitives.py:170 softmax_no_cast). Halving
    # the chunk halves that term and costs only extra sequential chunks.
    _pf_chunk = int(os.environ.get("IGV_PF_CHUNK", 128 if _n_tokens > _threshold else 512))

    # How many pairformer blocks share one checkpoint. Env-overridable so the
    # memory/recompute tradeoff can be tuned per complex size without a code
    # change; see the loop below for what it buys.
    _pf_group = max(1, int(os.environ.get("IGV_PF_GROUP_SIZE", _PF_GROUP_SIZE)))

    _n_pf_blocks = len(model.pairformer_module.layers)
    _n_msa_blocks = model.msa_module.msa_blocks

    if _n_tokens > _threshold:
        _msa_chunk_heads_pwa = True
        _msa_chunk_trans_z = 64
        _msa_chunk_trans_msa = 32
        _msa_chunk_outer = 4
        _msa_chunk_tri = int(os.environ.get("IGV_PF_CHUNK", 128))
    else:
        _msa_chunk_heads_pwa = False
        _msa_chunk_trans_z = None
        _msa_chunk_trans_msa = None
        _msa_chunk_outer = None
        _msa_chunk_tri = 512

    # recycling_steps is a caller parameter (default 1, matching the
    # predecessor repo's "1 for speed, 3 for quality"). It is also the single
    # biggest VRAM lever: the trunk runs recycling_steps+1 iterations and each
    # one saves a checkpoint boundary tensor per block, so at 730 tokens
    # 64 blocks x 2 iterations x 0.254 GiB = 32.5 GiB of saved z alone.

    def _pairformer_block_fn(s_, z_, mask_, pair_mask_, layer_idx_dummy):
        idx = int(layer_idx_dummy.item())
        return model.pairformer_module.layers[idx](
            s_, z_, mask_, pair_mask_, _pf_chunk, use_kernels=False
        )

    def _pairformer_group_fn(s_, z_, mask_, pair_mask_, span):
        """Run pairformer layers [span[0], span[1]) as ONE checkpoint unit."""
        start, stop = int(span[0].item()), int(span[1].item())
        for i in range(start, stop):
            s_, z_ = model.pairformer_module.layers[i](
                s_, z_, mask_, pair_mask_, _pf_chunk, use_kernels=False
            )
        return s_, z_

    def _msa_block_fn(z_, m_, token_mask_, msa_mask_, layer_idx_dummy):
        idx = int(layer_idx_dummy.item())
        return model.msa_module.layers[idx](
            z_, m_, token_mask_, msa_mask_,
            _msa_chunk_heads_pwa,
            _msa_chunk_trans_z,
            _msa_chunk_trans_msa,
            _msa_chunk_outer,
            _msa_chunk_tri,
            False,
        )

    def _msa_forward_checkpointed(z_, s_interp_):
        try:
            from boltz.data import const as _c
            _num_tokens = _c.num_tokens
        except Exception:
            _num_tokens = 33

        msa = feats["msa"]
        msa_oh = torch.nn.functional.one_hot(msa, num_classes=_num_tokens)
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        msa_mask_ = feats["msa_mask"]
        token_mask_ = feats["token_pad_mask"].float()
        token_mask_ = token_mask_[:, :, None] * token_mask_[:, None, :]

        if model.msa_module.use_paired_feature:
            is_paired = feats["msa_paired"].unsqueeze(-1)
            m = torch.cat([msa_oh, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa_oh, has_deletion, deletion_value], dim=-1)

        m = model.msa_module.msa_proj(m)
        m = m + model.msa_module.s_proj(s_interp_).unsqueeze(1)

        for blk_i in range(_n_msa_blocks):
            idx_t = torch.tensor(blk_i, device=device)
            z_, m = _checkpoint(
                _msa_block_fn, z_, m, token_mask_, msa_mask_, idx_t,
                use_reentrant=False,
            )
        return z_

    def _full_trunk_and_confidence(s_interp):
        s_init = model.s_init(s_interp)
        z_init = (
            model.z_init_1(s_interp)[:, :, None, :]
            + model.z_init_2(s_interp)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )

        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)

        for _ in range(recycling_steps + 1):
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))

            if gradient_checkpointing:
                z_ = z_ + _msa_forward_checkpointed(z_, s_interp)
                # Checkpoint GROUPS of blocks, not single blocks. Each
                # checkpoint retains its input z (1, L, L, 128); at L=730 that
                # is 254 MiB apiece, and a profile of the peak attributed
                # 34.30 GiB across 135 such tensors -- the largest single term
                # by far. Grouping trades those boundaries against a bigger
                # transient during recompute (the classic sqrt(N) tradeoff):
                # group_size=1 -> 64 boundaries/iteration, minimum recompute;
                # group_size=8 ->  8 boundaries/iteration, 8 blocks of
                # activations live at once. See _PF_GROUP_SIZE.
                for start in range(0, _n_pf_blocks, _pf_group):
                    stop = min(start + _pf_group, _n_pf_blocks)
                    span = torch.tensor([start, stop], device=device)
                    s_, z_ = _checkpoint(
                        _pairformer_group_fn, s_, z_, mask, pair_mask, span,
                        use_reentrant=False,
                    )
            else:
                z_ = z_ + model.msa_module(z_, s_interp, feats, use_kernels=False)
                s_, z_ = model.pairformer_module(
                    s_, z_, mask=mask, pair_mask=pair_mask, use_kernels=False
                )

        pdistogram = model.distogram_module(z_)
        pred_distogram_logits = pdistogram[:, :, :, 0]

        out_dict = model.confidence_module(
            s_inputs=s_interp,
            s=s_,
            z=z_,
            x_pred=x_pred,
            feats=feats,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=1,
            run_sequentially=False,
            use_kernels=False,
        )

        scalar = SCORES[score_name](out_dict)
        if scalar.dim() > 0:
            scalar = scalar.squeeze()

        # Guards deliberately live OUTSIDE this function, after the checkpoint
        # boundary -- see below. requires_grad is not meaningful in here.
        return scalar

    if gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint as _ckpt
        # use_reentrant=True is load-bearing for memory, and is why the guards
        # below sit out here rather than inside _full_trunk_and_confidence.
        #
        # Reentrant checkpointing runs the wrapped function under
        # torch.no_grad(), so NO autograd graph is built during the forward at
        # all; the trunk is recomputed with grad during backward, where the
        # per-block non-reentrant checkpoints bound peak memory. That two-level
        # scheme is what makes this complex fit in 80 GB. Switching the outer
        # call to use_reentrant=False builds the full forward graph and OOMs
        # (measured: 79.25 GiB capacity, ~32 MiB free).
        #
        # The cost is that requires_grad is False for every tensor inside the
        # function, so the guards cannot live there. They run here instead,
        # where the value is real.
        scalar = _ckpt(
            _full_trunk_and_confidence, s_inputs, use_reentrant=True,
        )
    else:
        scalar = _full_trunk_and_confidence(s_inputs)

    # Only demand a gradient when the caller actually asked for one. Callers
    # that just want a score -- the signal_control sanity check scores 30
    # mutants under torch.no_grad() -- are legitimate and must not trip this.
    if torch.is_grad_enabled():
        assert scalar.requires_grad, (
            f"Score {score_name!r} does not require grad. This usually means "
            f"compute_ptms silently failed (check stdout for 'Error in "
            f"compute_ptms') and returned a zero tensor without grad."
        )
    assert torch.isfinite(scalar).all(), (
        f"Score {score_name!r} is not finite: {scalar.item()}"
    )
    # boltz wraps compute_ptms in a bare `except` that assigns
    # torch.zeros_like(complex_plddt) and only prints. A zero score therefore
    # looks perfectly well-formed, and IG would silently integrate a constant.
    assert scalar.abs().item() > 1e-12, (
        f"Score {score_name!r} is exactly zero. compute_ptms almost "
        "certainly failed silently -- check stdout for 'Error in "
        "compute_ptms'. Do not interpret this run."
    )

    return scalar


# ---------------------------------------------------------------------------
# ipTM argmax recording
# ---------------------------------------------------------------------------


def record_iptm_argmax(out_dict, token_to_chain):
    """Record which token holds the ipTM argmax frame index.

    Parameters
    ----------
    out_dict : dict
        Confidence module output dict (must contain ``iptm``-related tensors
        from ``compute_ptms``).
    token_to_chain : dict[int, str]
        Maps token index to chain id.

    Returns
    -------
    dict with ``argmax_token``, ``argmax_chain``, ``argmax_resi``.
    """
    import torch

    pae_logits = out_dict.get("pae_logits")
    if pae_logits is None:
        return {"argmax_token": None, "argmax_chain": None, "argmax_resi": None}

    from boltz.model.modules.confidence_utils import compute_aggregated_metric

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = torch.arange(
        start=0.5 * bin_width, end=32.0, step=bin_width, device=pae_logits.device
    ).unsqueeze(0)
    probs = torch.nn.functional.softmax(pae_logits, dim=-1)

    N_res = torch.tensor([pae_logits.shape[1]], device=pae_logits.device, dtype=pae_logits.dtype)
    d0 = 1.24 * (torch.clip(N_res, min=19) - 15) ** (1.0 / 3.0) - 1.8
    tm_value = 1.0 / (1.0 + (pae_value / d0) ** 2)
    tm_value = tm_value.unsqueeze(1).unsqueeze(2)
    tm_expected = (probs * tm_value).sum(dim=-1)

    per_frame = tm_expected.mean(dim=-1)
    argmax_token = int(per_frame[0].argmax().item())

    chain_id = token_to_chain.get(argmax_token, "?")

    chain_offsets = {}
    for tok_idx, cid in sorted(token_to_chain.items()):
        if cid not in chain_offsets:
            chain_offsets[cid] = tok_idx
    resi = argmax_token - chain_offsets.get(chain_id, 0)

    return {
        "argmax_token": argmax_token,
        "argmax_chain": chain_id,
        "argmax_resi": resi,
    }
