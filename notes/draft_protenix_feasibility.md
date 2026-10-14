# Protenix as the second model: feasibility (DRAFT, 2026-10-12)

Source read: `github.com/bytedance/Protenix` cloned at `main` @ `85767b8` (2026-09-21).
The latest tagged release is **v2.0.0** (tag commit `2475421`, 2026-04-08; PyPI `protenix==2.0.0`).
The files cited below are the same at v2.0.0 and `main` unless noted. `main` adds only
docs/test/template fixes. All `file:line` refs are into that clone. Refs into this repo are
prefixed `IG/`, and refs into the predecessor repo are prefixed `IG_Agros/`.

Tags: **[V]** means I read it in source. **[UNVERIFIED]** means plausible but not checked by
reading code or running it. **[EST]** means arithmetic estimate, not measured.

---

## Verdict

**The confidence path is feasible with four small patches. The affinity arm is not
feasible, because Protenix has no affinity head.**

- **Confidence path: feasible with patches P1-P4 (below), all done by monkeypatching.** No
  fork of Protenix is needed. Residue identity enters as a float one-hot `restype`
  (N_token, 32). It is concatenated straight into `s_inputs`, which is the same structure
  as Boltz-2's `res_type`. Making it a float leaf takes one line. Four things block the
  gradient today:
  1. a `set_grad_enabled(self.training …)` gate around the recycling loop
  2. an in-place slice update in `MSAStack` (this is the root cause of IG_Agros's unsolved
     "CopySlices [1, N, 64]" error)
  3. `ConfidenceHead.stop_gradient=True`, which detaches s_inputs, s_trunk and z_trunk
  4. a `@torch.no_grad()` on the summary-score function
- **Hard max: confirmed.** Protenix's ipTM and pTM both use `.max()` over tokens. pLDDT,
  PAE, PDE, and a `complex_pde`-style mean are dense softmax-expectations. So the
  `complex_pde` result transfers one-to-one, and ipTM carries the same caveat as on Boltz-2.
- **Affinity: no head exists** in any public Protenix release, model variant or branch.
  Ligand complexes can only be attributed through the confidence scores.
- **Memory: should fit on 1x A100-80GB at about 450 tokens [EST],** provided recycling is
  limited to about 2-4 cycles with gradient. The full default of 10 cycles with gradient
  through all of them is borderline in fp32.
- **Environment: Protenix needs its own image.** Its pins conflict hard with `boltz==2.2.1`
  (numpy 2.4.1 against <2.0, among others).
- **Chai-1 is not easier.** Its trunk ships as exported/compiled modules that can't be
  checkpointed per block, and it needs `torch<2.7`. **Recommendation: use Protenix.**

---

## 1. Where residue identity enters, and how to make it differentiable

**[V] Featurizer.**
- `protenix/data/core/featurizer.py:89-104` builds `restype_onehot_encoded` with
  `F.one_hot(...).float()` (`:86`) over `STD_RESIDUES_WITH_GAP`, which has 32 classes.
- It is stored at `featurizer.py:342` as `token_features["restype"]`.
- Class order is in `protenix/data/constants.py:270-315`: ALA=0 … VAL=19, UNK=20, RNA 21-25,
  DNA 26-30, gap=31. **The 20 canonical amino acids are indices 0-19.** In Boltz they are
  2-21, so `canonical_aa_token_indices` needs a Protenix variant.

**[V] Model entry, which works the same way as Boltz-2.**
- `protenix/model/modules/embedders.py:69` sets
  `self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}`.
- `embedders.py:102-110` concatenates the atom-encoder output `a` (384) with `restype`,
  `profile` and `deletion_mean`, giving `s_inputs` (N_token, 449). There is no embedding
  lookup, so `restype` is already a float tensor used as-is.
- `restype` is read **nowhere else** in `protenix/model` (grep). It reaches the rest of the
  model only through `s_inputs`:
  - `s_init` and `z_init` (`protenix.py:206-213`)
  - the MSA module (`pairformer.py` `MSAModule.forward`, `linear_no_bias_s(s_inputs)`)
  - the confidence head (`confidence.py:197-199`, `z_init` from `s_inputs`)

**What to do:** set
`input_feature_dict["restype"] = restype.detach().float().requires_grad_(True)` before
calling `model.input_embedder(...)`. This is exactly the `IG/scripts/03_attribute.py:100-106`
pattern, and the gradient lands on the leaf.

**Parity caveats.** These are the same ones Boltz already has, and they should be stated the
same way in the paper:
- **Atom features stay wild-type** (`ref_pos`, `ref_element`, `ref_atom_name_chars`,
  `ref_space_uid` going into the AtomAttentionEncoder). So does the MSA query row: `msa` is
  integer and one-hot'd inside `MSAModule` (`pairformer.py:870-881`), so it is not
  differentiable. This matches Boltz, where `feats["msa"]` is fixed in
  `IG/src/igv/boltz_score.py` `_msa_forward_checkpointed`.
- **`profile` is a hidden second copy of identity when there is no MSA.** In single-sequence
  mode `protenix/data/utils.py:918` sets `profile = restype.clone()`. If no MSA is used,
  `profile` is a fixed wild-type copy of residue identity inside `s_inputs`. Either always
  supply the pinned wild-type MSA (as in Boltz, `IG/notes/logging.md` §16), or tie `profile`
  to the interpolated `restype` and document it. **[V]** for the clone; the choice is a
  design decision.
- **ESM variants add a fixed ESM embedding** (`embedders.py:112-119`). `protenix_mini_esm`
  and `protenix_mini_ism` add an ESM2-3B embedding that would not move with `restype`.
  **Use the non-ESM base model.**

## 2. Confidence outputs, and the "hard max" question (PLAN.md §10)

### Head and logits [V]

`ConfidenceHead.forward`, `protenix/model/modules/confidence.py:144-256`:

| Step | Code | Gradient status |
|---|---|---|
| `:178-181` | `if self.stop_gradient: s_inputs/s_trunk/z_trunk = .detach()`. The default is `stop_gradient: True` (`configs/configs_base.py:353`). | **Blocks the gradient. Patch P3.** |
| `:183` | `torch.clamp(s_trunk, -512, 512)` | Zero gradient only if saturated. Unlikely, but log `s_trunk.abs().max()`. |
| `:279-306` | Distance `cdist` → **hard bins** via `one_hot(lower, upper)` (`protenix/model/utils.py:177-193`, comparisons) **plus** an un-binned linear term `linear_no_bias_d_wo_onehot(distance)` (`:290`, `:302`) | Coordinates are fixed in our design, so this is moot. Unlike Boltz, Protenix does have a smooth coordinate path if it is ever wanted. |
| `:310-318` | 4-block pairformer | Differentiable. Checkpointed per block when grad is enabled (`utils.py:865`, `:818-819` non-reentrant). |
| `:328-341` | `pae`, `pde` (64 bins, 0-32 Å) and `plddt` (50 bins) logits are linear readouts; `resolved` likewise | Differentiable. |

### Scalar scores [V]

All in `protenix/model/sample_confidence.py`:

| Score | Definition | Hard max / discretisation | Use for IG? |
|---|---|---|---|
| pLDDT | `logits_to_score`: softmax @ bin centres (`:300-330`), mean over atoms ×100 (`:117`) | none | **yes (dense)** |
| PAE (token pair) | softmax expectation (`:106`) | none | yes, aggregate as a mean |
| PDE (token pair) | softmax expectation (`:100`) | none | **yes. The `complex_pde` analogue is the mean over valid token pairs, as in Boltz `confidencev2.py:454`** |
| gPDE | PDE weighted by `contact_probs`, which comes from the distogram head (softmax, `:241-270`); `:118-120` | none (the threshold is fixed bin indices) | yes; optional second dense score |
| pTM | `token_token_ptm.mean(-1)[..., has_frame].max(dim=-1)` (`:470`) | **hard max over aligned token i** | caveat |
| ipTM | per-row interchain mean, then `iptm[..., has_frame].max(dim=-1)` (`:871`) | **hard max over aligned token i** | caveat |
| chain_iptm etc. | `calculate_chain_based_ptm` (`:474-600`) calls the same ptm/iptm functions with token masks | same max | caveat |
| ranking_score | `0.8·iptm + 0.2·ptm + 0.5·disorder − 100·has_clash` (`:171-176`). `has_clash` is a thresholded coordinate test, constant under fixed coordinates. | inherits the max | no |

- **The summary wrapper must not be called.**
  `compute_full_data_and_summary` is decorated `@torch.no_grad()` (`:904`). **Patch P4:** call
  `logits_to_score` / `calculate_iptm` directly on live logits, or reimplement them (about
  30 lines), as `IG_Agros/src/ig/protenix.py:641-674` already does for ipTM.
- **Answer to PLAN.md §10, "Confirm Chai-1 and Protenix really do apply the hard max": yes
  for Protenix [V].**
  - It is a max over the aligned residue i of an interchain mean over j. So at the logit
    level, only one row of PAE logits receives gradient. The 4 confidence pairformer blocks
    and the trunk then make that dense with respect to `restype`.
  - The selected i can also switch along the IG path, which makes f piecewise-smooth.
    Expect worse completeness on ipTM than on `complex_pde`.
  - Boltz-2 has the same structure: `boltz/model/layers/confidence_utils.py:164-169`,
    `torch.max` for ptm/iptm. So the BindEnergyCraft prediction can be tested identically
    on both models.
  - Chai-1: not checked [UNVERIFIED].

## 3. Affinity head: none

- **[V]** `git grep -i affinity` across `main`, `origin/dev_design` and
  `origin/constraint_esm`: no hits in any `.py` or `.md`.
- **[V]** Model list (`docs/supported_models.md`, `configs/configs_model_type.py`): base
  v0.5/v1.0/20250630, constraint, mini, tiny, mini_esm, mini_ism, and protenix-v2. **All are
  structure + confidence only.** The heads are `DistogramHead` and `ConfidenceHead`
  (`protenix.py:136-137`).
- **[V]** The technical reports `docs/PTX_V1_Technical_Report_*.pdf` and `docs/PX2.pdf`
  mention affinity only in citations (Boltz-2) and in wet-lab binder measurements.
- A web search turned up no Protenix affinity release. Forks were not exhaustively surveyed
  **[UNVERIFIED]**.

**Consequence:** on ligand complexes the Protenix arm can only attribute confidence scores.
For a ligand chain, use chain-pair ipTM (hard max) or a ligand-restricted mean PDE/PAE
(dense). Note that Boltz-2's affinity arm was already closed as non-converging
(`IG/notes/logging.md` §22), so losing it here costs nothing new.

## 4. Diffusion, coordinates, and injecting fixed geometry

**How this repo handles it for Boltz** (`IG/notes/logging.md` §13 "Two blockers" item 1, and
§14 "Real geometry"):
- `x_pred` is held fixed across IG steps.
- It originally came from `feats["coords"]`, which was all zeros for sequence-only inputs.
- It now comes from one real structure prediction (`--x-pred predicted`, the default in
  `03_attribute.py`), cached per dataset. That tripled the f(x) − f(baseline) span.
- The gradient never goes through the coordinates.

**Protenix equivalent [V]:**
- The confidence head takes coordinates as an **explicit argument**,
  `x_pred_coords` [..., N_sample, N_atom, 3] (`confidence.py:150`, `:191-193`). In the
  inference loop it is fed from `sample_diffusion` (`protenix.py:550-563`, `:592-603`).
  **So injection is trivial:** run once under `no_grad` (get_pairformer_output →
  `model.sample_diffusion`, already wrapped as `IG_Agros/src/ig/protenix.py:804-855`), take
  one sample, `.detach()` it, cache it, and pass the same tensor in every IG step.
- Each sample is processed separately in the head (`confidence.py:205-239`). Pass a single
  sample (N_sample=1) to keep memory down.
- For parity with Boltz, choose the sample the same way, e.g. the best of 5 by ranking score
  or a fixed seed, and record it in provenance.
- **Crystal coordinates:** they must be mapped into Protenix's atom ordering
  (`atom_to_token_idx`, `distogram_rep_atom_mask`). The training pipeline's
  `protenix/data/pipeline/data_pipeline.py:50` `get_data_from_mmcif` produces features and
  label coordinates in matching order, and could be reused **[UNVERIFIED for this use]**.
  "predicted" mode is the path that is known to work.
- **The geometry-confound control still applies:** coordinates are wild-type and fixed while
  `restype` moves.

## 5. Memory: Protenix base against Boltz-2

| | Boltz-2 (`boltz==2.2.1`) | Protenix base v1.0.0 |
|---|---|---|
| Params | not checked | 368.5 M (README); protenix-v2 is 464 M with c_z=256 |
| Trunk pairformer blocks | 64 (`boltz/main.py:83`) | **48** (`configs_base.py:118`) |
| c_s / c_z | 384 / 128 | 384 / 128 (`configs_base.py:112-113`) |
| MSA blocks | 4 | 4 (`configs_base.py:274`), c_m=64 |
| Template blocks | (template module) | 2 (`configs_model_type.py:92-94`) |
| Confidence pairformer | 8 (`IG/notes/logging.md` §12) | **4** (`configs_base.py:345`) |
| Default recycles | this repo uses `recycling_steps=1` (2 passes) | `N_cycle=10` |
| Per-block checkpoint | added by this repo | **built in**, non-reentrant, `blocks_per_ckpt=1`, active whenever grad is enabled (`utils.py:818-876`) |

**[EST] at L=450, fp32:**
- One z is 450²·128·4 B ≈ 0.097 GiB.
- Checkpoint boundaries per recycle ≈ (48 + 4 + 2) z copies ≈ 5.2 GiB.
- With gradient through every cycle: N_cycle=2 → ~10 GiB, 4 → ~21 GiB, 10 → ~52 GiB.
- Add the transient for one block recompute. Torch triangle-attention softmax is L³·H·4 B
  ≈ 1.5 GiB per attention.
- Add the MSA tensor: N_msa×L×64×4 B, i.e. 1.9 GiB at the default 16384-row cap
  (`configs_data.py:270-275`). **Cap it** to whatever depth Boltz used.
- Add the confidence head (4 blocks).

For comparison, this repo measured Boltz-2 (64 blocks, 2 passes) at **79.6 GiB for 730
tokens fp32 + ckpt** (`IG/docs/MEMSCALE_RESULTS.md:65`). Scaling by (450/730)² × 48/64
suggests about 23 GiB for Protenix with 2 cycles.

**Conclusion: N_cycle 2-4 fits comfortably. N_cycle=10 with full gradient is about 55-65 GiB
fp32 [EST].** It might fit, and bf16 or the repo's outer reentrant-checkpoint trick
(`IG/src/igv/boltz_score.py:2875-2891`) would make it safe. Measure at the start.

Other memory notes:
- **Truncated backprop is an option, not the default.** Running cycles 1..N−1 under
  `no_grad` and only the last with grad is what Protenix training does (`protenix.py:235-239`).
  It changes the attributed function, so it is not the default for parity.
- **IG_Agros's earlier full-trunk attempts on H100 hit an autograd error, not OOM**
  (`IG_Agros/ERRORS_LOG.md` 2026-06-28 01:28, 2026-07-12). Full-trunk Protenix memory has
  never been measured.

## 6. Licence, weights, install, and environment conflicts

**Licence and weights [V]:**
- Code and weights for v0.5 / v1.0 models are **Apache 2.0** (`README.md:188-190`, PyPI
  metadata).
- **protenix-v2 weights are proprietary.** README "Latest Updates" says they "are not
  released under any open-source license, and may not be reproduced, distributed …". There
  is a download URL (`protenix/web_service/dependency_url.py:16`), but **do not use v2 in a
  paper**.
- **Use `protenix_base_default_v1.0.0`** (2021-09-30 cutoff, the same as AF3, which is better
  for SKEMPI/DMS leakage arguments). Alternatively use `protenix_base_20250630_v1.0.0`.
- Weights are a direct download from
  `https://protenix.tos-cn-beijing.volces.com/checkpoint/<name>.pt`. The CCD files
  (`components.cif`, the rdkit pkl, etc.) are under `/common/` (`dependency_url.py:15-35`).
  Download speed from a GCP us-central VM is [UNVERIFIED]; pre-stage the files to a bucket.

**Install:**
- `pip install protenix==2.0.0` (Python ≥3.11). Or clone the repo, which you need anyway for
  root-level `configs/` and `runner/` (`IG_Agros/src/ig/protenix.py:156-172`). There is also
  a repo `Dockerfile` and `docs/docker_installation.md`.

**Why README:186 says "do not install protenix" [V]:**
- `requirements.txt` hard-pins `cuequivariance-ops-torch-cu12==0.8.0` and
  `cuequivariance-torch==0.8.0`.
- `IG_Agros/ERRORS_LOG.md` 2026-07-18/19 records that those wheels pulled newer cuBLAS,
  broke the cu124/cu126 image, and that cuequivariance segfaults on A100 sm_80. Mixed with
  boltzgen it also clobbered torch to 2.12+cu130.

**Remaining hard conflicts with `boltz==2.2.1` [V]:**

| Package | protenix 2.0.0 | boltz 2.2.1 |
|---|---|---|
| numpy | ==2.4.1 | <2.0 |
| biopython | ==1.85 | ==1.84 |
| gemmi | ==0.6.7 | ==0.6.5 |
| scikit-learn | ==1.7.1 | ==1.6.1 |
| modelcif | ==1.4 | ==1.2 |
| wandb | ==0.21.1 | ==0.18.7 |

`torch==2.7.1` itself matches.

**Recommendation: a separate image or venv.** Use the same `torch==2.7.1+cu126` base, then
`pip install --no-deps protenix==2.0.0` plus its pure-Python deps, **excluding
cuequivariance\* and deepspeed**. Both are imported lazily, only when that kernel is
selected (`protenix/model/triangular/triangular.py:42`, `triangular/layers.py:496`, `:594`).
- Set `triangle_multiplicative="torch"` and `triangle_attention="torch"`; the defaults are
  `"cuequivariance"` (`configs_base.py:129-130`).
- Set the env var `LAYERNORM_TYPE` to anything other than `fast_layernorm`. That selects
  pure-torch `OpenFoldLayerNorm` and avoids the nvcc JIT build (`triangular/layers.py:33`,
  `:246-256`). This replaces IG_Agros's sys.modules stub hack.
- The two models never need to share a process. The comparison happens on saved `.npz`
  artifacts.

## 7. Chai-1 compared with Protenix

- **Trunk is opaque.** Chai-1 ships the trunk, embedders and confidence head as
  exported/compiled modules (`load_exported("trunk.pt")` etc. in
  `IG_Agros/src/ig/chai1.py:595-599`). You cannot insert per-block checkpointing or patch
  internals.
- **Full-trunk backward already failed.** It OOM'd even on an H200 (141 GB) in a triangle
  einsum (`IG_Agros/ERRORS_LOG.md` 2026-06-28 00:26). That is why IG_Agros ran Chai only
  frozen-trunk.
- **Residue identity sits inside the exported feature embedding.** It is wrapped by Chai's
  featurizer and `feature_embedding.pt`. Exposing a one-hot leaf is possible in principle,
  but less direct than Protenix's plain `torch.cat` [UNVERIFIED].
- **Its pins conflict too.** PyPI `chai_lab 0.6.1` requires `torch<2.7`, `numpy~=1.21` and
  `rdkit~=2024.9.5`.
- **Licence:** weights licence terms [UNVERIFIED]. Re-check before publishing.
- **Recommendation: Protenix.** Its source is plain PyTorch, mirrors the Boltz structure
  (one-hot restype concatenated into s_inputs, the same AF3 Algorithm 31 confidence head),
  has built-in non-reentrant per-block checkpointing, Apache-2.0 weights, and a smaller
  trunk (48 vs 64 blocks). Keep Chai-1 at most as a frozen-trunk ipTM sanity check, labelled
  as a different attribution scope.

---

## Implementation sketch for the builder

Start from `IG_Agros/src/ig/protenix.py`, which already has config/weight loading,
JSON→features, `_prepare_protenix_features`, `_run_diffusion` and differentiable `_compute_iptm`.
Port it into `IG/src/igv/protenix_score.py`. Do **not** port its frozen-trunk IG; we want full
trunk, matching the Boltz `confidence_forward`.

**Patches, all as runtime monkeypatches in our module:**

- **P1 — own recycling loop.** Replace `Protenix.get_pairformer_output`, whose
  `set_grad_enabled(self.training …)` at `protenix.py:234-240` kills grad in eval. Port
  `IG_Agros/src/ig/protenix.py:504-571` `_ig_safe_pairformer` and add:
  - `restype` leaf → `model.input_embedder(ifd, inplace_safe=False)` → `s_inputs`
  - `relp` and `token_bonds` computed once under `no_grad`
  - `triangle_* = "torch"`, `inplace_safe=False`, `chunk_size` from
    `_get_dynamic_chunk_size`
  - an `N_cycle` parameter (default 2, matching Boltz recycling_steps=1)
- **P2 — fix the in-place MSA update.** `MSAStack.inference_forward` (`pairformer.py:547-572`)
  does `m[start:end] += …` in place whenever `not self.training` (`:471`, `:493`).
  - This is the root cause of IG_Agros's
    `[torch.cuda.FloatTensor [1, N, 64]] … CopySlices … version` error: c_m=64, single-row
    MSA.
  - Replace it with an out-of-place version: `chunks.append(mc + f(mc)); …; torch.cat`.
  - Do **not** switch to `.train()`. The training branch pads the MSA to 16384 rows
    (`:476-487`) and turns on dropout.
- **P3 — let gradient into the confidence head.** Set `model.confidence_head.stop_gradient = False`
  (`confidence.py:178-181`).
- **P4 — compute scores from live logits.** Never call
  `sample_confidence.compute_full_data_and_summary` (`@torch.no_grad`, `:904`). Register these
  scores from the head's logits (use 1 sample, squeeze):
  - `complex_pde` = mean over has-token pairs of `softmax(pde)·centres(0, 32, 64)`
  - `complex_plddt` = mean atom pLDDT
  - `iptm` / `ptm` via `calculate_iptm` / `calculate_ptm` (these are differentiable; only
    the wrapper is no_grad)
  - optionally `gpde`, with `contact_probs` from `model.distogram_head(z)`
  - Keep the Boltz guards: `requires_grad`, finite, not exactly zero.

**Wrapping and memory:**
- Wrap the trunk + confidence call in the same outer `checkpoint(..., use_reentrant=True)`
  plus inner non-reentrant scheme as `IG/src/igv/boltz_score.py:2875-2891`. Protenix's own
  `checkpoint_blocks` supplies the inner per-block checkpoints automatically when grad is on.
- Call `model.run_confidence_head` or `model.confidence_head` with
  `x_pred_coords` = the cached `(1, N_atom, 3)` detached coordinates.

**Model and inputs:**
- Load `protenix_base_default_v1.0.0` in `eval()` with fp32 first. Add bf16 later behind
  the same `IGV_AUTOCAST` knob; the provenance rule applies.
- Use the same pinned wild-type MSA as the Boltz run. Templates off, or wild-type and fixed.
  Record both.
- Baselines in 32-class space:
  - `mean_aa` = 1/20 over indices 0-19 at mutable protein tokens. Leave ligand, nucleic and
    other chains untouched, mirroring `IG/src/igv/boltz_score.py:1371-1405`.
  - `unk` = index 20.
  - `shuffled`.
- Output is the `(L, 32)` gradient; slice to `[:, 0:20]` before
  `simplex_score_from_onehot`.
- **The token map differs from Boltz.** Protenix tokenises modified residues and ligands
  per-atom, so build an explicit residue→token map and assert the decoded sequence, as
  `_build_token_map` does for Boltz.

**Tests the builder should add before any GPU spend:**
1. Gradient is non-None and nonzero on `restype` for each score.
2. Finite-difference check: perturb one `restype` row by ε and compare with grad·ε.
3. With P2 removed the backward raises, and with P2 applied it passes (a regression test).
4. A completeness gate on `complex_pde` for one complex, m=32 and m=64.
5. A peak-memory ladder at L=300/450/730 × N_cycle {2, 4, 10}.

**Effort [EST]:**
- About 3-4 builder-days for the module, patches and tests.
- About 1 day to build and validate the separate image.
- About 0.5 GPU-day (~$15-25 on A100) for the memory ladder and a first `complex_pde` IG on
  1JTG with real geometry.
- Rerunning the full Boltz confidence-path suite on Protenix should then cost about the
  same GPU time as Boltz did or less (48 vs 64 blocks), multiplied by the chosen N_cycle.

Sources: [Protenix GitHub](https://github.com/bytedance/Protenix) (cloned), PyPI JSON for
`protenix` and `chai_lab`, [Benchling Protenix page](https://www.benchling.com/ai/models/protenix),
[ByteDance Seed Protenix-v2 page](https://seed.bytedance.com/zh/public_papers/protenix-v2-broadening-the-reach-of-structure-prediction-and-biomolecular-design).
