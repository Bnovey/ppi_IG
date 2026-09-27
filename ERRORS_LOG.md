# Errors log

Format follows the predecessor repo (`IG_Agros/ERRORS_LOG.md`): one entry per
failure, with the symptom as observed, the root cause as *verified* (not
guessed), the fix, and a status. Entries are append-only — do not rewrite
history, add a follow-up entry instead.

---

# 2026-09-02 — First GPU session on `igv-gpu` (A100-80GB, us-central1-a)

Goal: get the pipeline running end-to-end in GCP and reach the stage-0 go/no-go
on `4fqi_h1`. Twelve distinct failures, all resolved except the last, which is
architectural.

## Summary

**Environment is proven.** The risk `HANDOFF.md` flagged as "the real unknown" —
whether `torch==2.7.1+cu126` and `boltz==2.2.1` coexist — is resolved. Full
verification table passes: torch 2.7.1+cu126, CUDA available, A100-SXM4-80GB,
boltz 2.2.1, all five forbidden packages absent.

**`HANDOFF.md` §2/§4 were stale.** It states the Docker image "never built
successfully" and that the four local fixes were "NOT yet on the VM and NOT
tested". In fact bootstrap had already run to completion at 20:43 the same day
with the fixes applied, and every verification check passed except the VRAM
gate. Read `~/bootstrap.log` on the VM before trusting a handoff's claim about
what has run.

**The science gate that matters passed.** `signal_control` — the load-bearing
check — passes. Everything still blocked is blocked on VRAM, not on biology.

**One unresolved issue.** The full-trunk backward does not fit in 80 GiB at 730
tokens. Nine software configurations were measured; all peak at ~78 GiB. See
entry 12.

---

## 1. Bootstrap VRAM gate was unreachable on the target card

- **Stage:** bootstrap / verification
- **Command:** `bash scripts/cloud/bootstrap.sh`
- **Symptom:** Every check passed except `gpu_0_vram_gib 79.2 FAIL`, so
  bootstrap exited non-zero and the pipeline never started.
- **Root cause:** `MIN_VRAM_GIB=80`. An A100-SXM4-80GB reports 81920 MiB
  (= 80 GiB) to `nvidia-smi`, but `torch.cuda.get_device_properties(0)
  .total_memory` returns the usable framebuffer after the ECC/reserve
  carve-out: **79.2 GiB**. `gib >= 80` can never be true on the exact hardware
  the project targets.
- **Fix:** `MIN_VRAM_GIB=78` (commit `fddbfaa`). Still rejects a 40GB A100
  (~39.5 GiB).
- **Status:** fixed

## 2. Same unreachable gate in all four GPU stages

- **Stage:** 02, 03, 04, 07
- **Symptom:** Would not have surfaced until each stage was reached.
- **Root cause:** `_require_vram(min_gib=80)` is duplicated verbatim in
  `02_embed_deltas.py`, `03_attribute.py`, `04_scan.py`, `07_sanity.py`. Every
  GPU stage in the pipeline would have refused to run on the target card.
- **Fix:** all four lowered to 78 (commit `8ef0fee`).
- **Status:** fixed
- **Note:** `_require_vram` is still copy-pasted four times. Worth hoisting into
  `src/igv/`.

## 3. No parquet engine

- **Stage:** 01_build_library
- **Symptom:** Stage fetched and validated `4fqi_h1`, logged "All 4fqi_h1
  assertions passed", then died on the final write:
  `ImportError: Unable to find a usable engine; tried using: 'pyarrow',
  'fastparquet'.`
- **Root cause:** `01_build_library.py` writes with `to_parquet` and
  `05_predict.py` reads it back, but no parquet engine was declared in
  `pyproject.toml`, `constraints.txt`, or the Dockerfile.
- **Fix:** added `pyarrow` to `pyproject.toml` dependencies (commit `4a0432d`).
- **Status:** fixed

## 4. Provenance recorded `git_commit: null` on every artifact

- **Stage:** all (any stage that writes a sidecar)
- **Symptom:** The stage-01 sidecar contained `"git_commit": null,
  "git_dirty": null` despite a clean checkout at a known commit on the host.
  **No error, no warning.**
- **Root cause:** `provenance._git_commit` shells out to `git rev-parse HEAD`,
  and the Docker image never installed `git`. The subprocess raised
  `FileNotFoundError`, which `_git_commit` swallows into `None`. Every artifact
  would have been unreproducible.
- **Fix:** `apt-get install git` in the image, as its own layer *after*
  torch/boltz so it does not invalidate those multi-GB cached layers; plus
  `git config --global --add safe.directory /app`, because the pipeline
  bind-mounts the host repo over `/app` where it is owned by the host user
  while the container runs as root, and git's dubious-ownership check would
  produce the same null (commit `6768ca3`).
- **Status:** fixed
- **Lesson:** this is the failure class `HANDOFF.md` §6 warns about — found only
  by *auditing the artifact*, not by watching the exit code. The run "succeeded".

## 5. Boltz-2 weights had no fetch path, and no persistent cache

- **Stage:** 07_sanity (first stage to load the model)
- **Symptom:** `FileNotFoundError: No .ckpt files in /root/.boltz`
- **Root cause:** Nothing in the repo ever downloaded Boltz-2's checkpoints.
  `boltz` caches to `~/.boltz` on first use, but the pipeline runs under
  `docker run --rm`, so that directory dies with the container — even a
  successful download would be discarded and re-fetched (~5.5 GB) per stage.
- **Fix:** added `scripts/cloud/fetch_weights.sh`, which downloads via
  `boltz.main.download_boltz2` into a persistent host cache
  (`~/boltz_cache`, override with `BOLTZ_CACHE`) on the boot disk, which
  survives VM stop/start. Idempotent. Every documented `docker run` now mounts
  `-v $HOME/boltz_cache:/root/.boltz` — `docs/CLOUD.md` (quick-start +
  full command + a troubleshooting row), `bootstrap.sh`, `gcp_launch.sh`
  (commit `502f207`; `aws_launch.sh` was also updated but has since been
  removed — the project runs on GCP).
- **Status:** fixed
- **Cache contents:** `boltz2_conf.ckpt` 2.2 GB, `boltz2_aff.ckpt` 2.0 GB,
  `mols/` + `mols.tar` ~3.5 GB. Total ~5.5 GB. One-time.

## 6. `load_model` selected the AFFINITY checkpoint, not the confidence one

- **Stage:** all GPU stages
- **Symptom:** None. That is the problem.
- **Root cause:**
  ```python
  candidates = sorted(ckpt_dir.glob("*.ckpt"))
  ckpt_path = candidates[0]
  ```
  A full `download_boltz2` leaves two checkpoints, and `boltz2_aff.ckpt` sorts
  before `boltz2_conf.ckpt`. Every score in `boltz_score` (`iptm`, `ptm`,
  `complex_pde`, …) reads the **confidence** head, so the pipeline would have
  taken gradients of the **affinity** network and reported them as the result —
  no exception, no warning, just wrong numbers. Unreachable while the cache was
  empty (it raised `FileNotFoundError` instead); downloading the weights in
  entry 5 made it live.
- **Fix:** extracted `select_checkpoint()`, torch-free so it is testable without
  the GPU container, preferring the confidence head and raising if only an
  affinity checkpoint is present. Added `tests/test_boltz_score.py` with 5 cases
  (commit `be48758`).
- **Status:** fixed
- **Lesson:** second instance of the silent-wrong-answer class in one session.
  Fixing one bug (weights missing) exposed another that had been latent.

## 7. `load_model`'s tuple not unpacked in stages 04 and 07

- **Stage:** 04_scan, 07_sanity
- **Symptom:** `AttributeError: 'tuple' object has no attribute
  'input_embedder'` in `embedder_only`.
- **Root cause:** `load_model` returns `(model, boltz_version)`; 04 and 07
  assigned the whole tuple to `model`. Stages 02 and 03 unpacked correctly,
  which is why it survived review. The failure only appears *after* Boltz-2
  loads and runs two full 600-step predictions (~2 min of A100 time), so it was
  invisible until the weights existed.
- **Fix:** unpack into `_boltz_version` and discard — provenance records `boltz`
  in its env block independently (commit `a9fd33d`).
- **Status:** fixed

## 8. `assert scalar.requires_grad` fired on every call, blaming the wrong thing

- **Stage:** 07_sanity — 5 of 7 checks
- **Symptom:** `AssertionError: Score 'complex_pde' does not require grad. This
  usually means compute_ptms silently failed (check stdout for 'Error in
  compute_ptms') and returned a zero tensor without grad.`
- **Root cause:** `compute_ptms` was innocent — that string never appeared in
  the log. The guard sat *inside* `_full_trunk_and_confidence`, which is wrapped
  in `checkpoint(..., use_reentrant=True)`. Reentrant checkpointing executes the
  wrapped function under `torch.no_grad()` on the forward pass. Verified against
  the pinned torch 2.7.1+cu126:

  | | inside | outside |
  |---|---|---|
  | `use_reentrant=True` | `requires_grad=False` | `True` |
  | `use_reentrant=False` | `True` | `True` |

  So the assertion could never pass, and its message sent you hunting in
  entirely the wrong module.
- **Fix (first attempt, wrong):** switched the outer checkpoint to
  `use_reentrant=False` (commit `0b9bd8a`). The guard then worked — and 4 checks
  immediately began OOMing instead, because reentrant mode was load-bearing for
  memory (see entry 9).
- **Fix (correct):** moved the guards *after* the checkpoint boundary, where
  `requires_grad` is meaningful, and restored `use_reentrant=True`
  (commit `079b110`).
- **Status:** fixed

## 9. Non-reentrant outer checkpoint OOMs — reentrant mode is load-bearing

- **Stage:** 07_sanity
- **Symptom:** `CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a
  total capacity of 79.25 GiB of which 31.81 MiB is free.`
- **Root cause:** the fix in entry 8's first attempt. Reentrant checkpointing
  runs the forward under `no_grad`, so **no autograd graph is built during the
  forward at all**; the trunk is recomputed with grad during backward, where the
  per-block checkpoints cap peak memory. Switching to non-reentrant builds the
  full forward graph. The predecessor's code comment states the same design:
  *"Outer checkpoint (use_reentrant=True): the forward pass runs under
  torch.no_grad, so peak VRAM equals the structure pass."*
- **Fix:** restored `use_reentrant=True` (commit `079b110`).
- **Status:** fixed
- **Do not** "clean up" the outer checkpoint to non-reentrant. It is not a style
  choice, and the deprecation warning upstream is a trap here.

## 10. `signal_control` failed an assertion written for the attribution path

- **Stage:** 07_sanity
- **Symptom:** `signal_control` raised the same `does not require grad`
  assertion as entry 8, but kept failing after entry 8 was fixed.
- **Root cause:** `signal_control` scores 30 mutants under `torch.no_grad()` —
  it wants score *values*, not gradients. Demanding `requires_grad`
  unconditionally is simply wrong for that caller.
- **Fix:** guard is now conditional on `torch.is_grad_enabled()`
  (commit `079b110`).
- **Status:** fixed

## 11. Confidence pairformer stack was not checkpointed

- **Stage:** 03_attribute / 07_sanity (backward pass)
- **Symptom:** OOM at
  `confidencev2.py:205 pairformer_stack → pairformer.py:102 z = z +
  self.transition_z(z) → transition.py:64 silu(fc1(x)) * fc2(x)`,
  allocating 1.02 GiB with 78.43 GiB already held (282 MiB
  reserved-unallocated, so **not** fragmentation).
- **Root cause:** the trunk (`pairformer_module`, `msa_module`) was checkpointed
  per block but `confidence_module.pairformer_stack` was not, so all 8 of its
  layers' L×L activations materialised at once during the backward recompute.
  Same failure the predecessor fixed in the trunk — *"checkpointed the whole
  pairformer_module as ONE unit"* — relocated to the module that still fit at
  their smaller L≈540.
- **Fix:** `enable_confidence_checkpointing()` rebinds the stack's `forward` to
  checkpoint per layer (commit `7eca3db`). Boltz implements this already but
  gates it on `self.activation_checkpointing and self.training`; calling
  `.train()` to unlock it is **not** an option — it enables dropout (stochastic
  score, meaningless gradient) and sets `chunk_size_tri_attn=None`, removing the
  eval-mode chunking. So `forward` is rebound while staying in eval.
- **Status:** fixed (reduced peak 78.43 → 77.64 GiB; necessary, not sufficient)

## 12. Full-trunk backward does not fit in 80 GiB at 730 tokens — UNRESOLVED

- **Stage:** 03_attribute / 07_sanity
- **Symptom:** `completeness`, `m_sweep`, `random_weights`, `frozen_vs_full` all
  fail with `OutOfMemoryError`. `GATE FAILED`.
- **Complex size:** `4fqi_hlab` = HA antigen A:336 + B:185, Fab H:123 + L:109 →
  **730 tokens** (`n_tokens` as featurised). The predecessor's full-trunk runs
  were at L≈540, making our L×L pair tensors ~1.8× larger. Model geometry: 64
  trunk pairformer blocks, 4 MSA blocks, 8 confidence pairformer layers;
  z is (1, 730, 730, 128) fp32 = **0.254 GiB**.

### Nine configurations measured, all OOM

| group | chunk | recycling | peak |
|---|---|---|---|
| 1 | 128 | 1 | 77.64 GiB |
| 2 | 128 | 1 | 78.35 GiB |
| 4 | 128 | 1 | 78.20 GiB |
| 8 | 128 | 1 | 78.40 GiB |
| 8 | 32 | 1 | 78.62 GiB |
| 8 | 16 | 1 | 78.64 GiB |
| 4 | 32 | 1 | 78.52 GiB |
| 1 | 32 | 1 | 77.64 GiB |
| 1 | 128 | **0** | 78.36 GiB |

### The knobs work; they only relocate the memory

Allocation-trace replay (live bytes attributed by call site at reconstructed
peak):

| site | group=1 | group=8 |
|---|---|---|
| `pairformer.py:102` (checkpoint boundary z) | **34.30 GiB** / 135 blocks | **4.32 GiB** / 17 |
| `primitives.py:170 softmax_no_cast` | 11.59 GiB / 12 | **25.22 GiB** / 26 |
| `transition.py:64` | 1.02 GiB / 1 | 6.12 GiB / 12 |
| `trunkv2.py:749` | 9.76 GiB / 6 | 9.76 GiB / 6 |
| `_msa_forward_checkpointed` | 3.25 GiB / 2 | 3.25 GiB / 2 |

Grouping does exactly what it should — boundary tensors drop 8× — and the
transient it trades against grows to match. This is the classic √N
checkpointing tradeoff landing at break-even. **The requirement is broadly
distributed, not concentrated in any one tunable term.**

`recycling_steps=0` is not a free knob either: it does not reduce peak (the
allocator reuses the freed space) and it *does* change the score
(3.925568 → 4.309255).

### Why CPU offload is not the answer

Two independent reasons, both verified:

1. **Blanket `save_on_cpu` is a known disaster here.** The predecessor tried it
   first: a single IG step ran >18 min on PCIe paging — *"Not an error, not OOM
   — pathological slowness"* — and their eventual fix was explicitly
   *"per-block gradient checkpointing (**not** CPU offload)"*. `attrib.py`'s
   docstring warning against it is hard-won, not an oversight.
2. **A size-thresholded offload cannot reach the tensors that matter.**
   Implemented `offload_large_saved_tensors` (≥200 MiB → pinned host RAM) and
   measured peak 77.64 → **76.50 GiB**. `torch.utils.checkpoint`'s
   *non-reentrant* path holds its inputs in a Python **closure**, not via
   `save_for_backward`, so `saved_tensors_hooks` never sees them. Switching the
   inner block checkpoints to reentrant *does* route them through the hooks but
   **breaks the gradient outright** — `UserWarning: None of the inputs have
   requires_grad=True. Gradients will be None`, because the outer checkpoint
   runs its forward under `no_grad` so block inputs carry no `requires_grad`.
   That attempt also drove host RSS to 165 GB and was killed by the kernel OOM
   killer.

   Kept in the tree, **defaulted off**, with the finding documented so it is not
   re-derived.

- **Root cause:** 80 GiB lacks the headroom for a full-trunk backward at 730
  tokens. Matches the predecessor's conclusion one size down: *"full-trunk IG at
  L≈540 tokens exceeds 80 GB on H100 (prior no-OOM analyses were on H200
  141 GB — H100 has far less headroom)"*, and their migration plan's remedy,
  *"pick H200 / B200 / GH200 to attack the L×L pair-tensor OOM directly with
  more VRAM"*.
- **Fix:** none on this hardware. **Decision taken: switch to frozen-trunk
  (confidence-only) attribution**, which backprops only through the confidence
  head and fits comfortably. Not yet implemented.
- **Status:** open — superseded by the frozen-trunk decision

---

## Sanity gate result (`results/sanity_4fqi_h1_complex_pde.json`)

Run at commit `079b110`, dataset `4fqi_h1`, score `complex_pde`.

| check | result | detail |
|---|---|---|
| completeness | FAIL | OOM |
| m_sweep | FAIL | OOM |
| random_weights | FAIL | OOM |
| dead_target | PASS | `max|grad| = 0.000e+00` |
| **signal_control** | **PASS** | `n=30 std=1.492e-02 range=[3.878765, 3.945837] spearman_vs_measured=-0.1595` |
| frozen_vs_full | FAIL | OOM |
| arm_assertion | PASS | no artifacts present yet (stages 02/03 not run) |

**`signal_control` passing is the headline.** Boltz-2's `complex_pde` genuinely
moves under mutation, so its gradient can carry information and a null
attribution result would be *interpretable* rather than vacuous. `HANDOFF.md` §6
correctly identifies this as the load-bearing check; it is now green.

Two caveats on the passes:

- **`dead_target` is passing vacuously.** It reports `max|grad| = 0.000e+00` and
  has done so in every run this session, including runs where the gradient path
  was completely severed. It is not currently evidence of anything.
- **`arm_assertion` is trivially true** until stages 02/03 produce artifacts.
- `signal_control`'s Spearman vs measured affinity is **−0.1595**. Only its
  being non-zero is required here, but the sign is worth checking against the
  `complex_pde` convention (lower predicted distance error = better) before it
  is read as a T2-style result.

---

## Environment facts worth not rediscovering

- **Verification table (passing):** python 3.11.15, torch 2.7.1+cu126,
  cuda_available True, gpu_0_vram_gib 79.2, NVIDIA A100-SXM4-80GB, gpu_count 1,
  boltz 2.2.1, and `boltzgen` / `protenix` / `chai_lab` / `gnina` /
  `cuequivariance` all "not found".
- **The VM's `~/IG` was not a git repo** — it had been `scp`'d. It is now a real
  checkout, synced by shipping a tarball that includes `.git` (only 488 KB) and
  running `git reset --hard`, so provenance records a genuine commit. `.claude/`
  is gitignored so it cannot flip `git_dirty`.
- **Container runs as root**, so artifacts under `results/` and `data/` come out
  root-owned on the host. `sudo chown` before `gcloud compute scp`.
- **Timings on A100:** model load ~45 s; one Boltz-2 structure prediction
  (600 steps) ~45 s; one scored mutant in `signal_control` ~40 s; a full
  `07_sanity` run ~25 min; Docker rebuild ~30 s when only the `pip install -e`
  layer is invalidated.
- **Rebuild cheaply:** put new `apt`/`pip` layers *after* the torch and boltz
  installs, or you re-download multiple GB.

## Diagnostics left on the VM

`~/IG/probe_mem.py` (VRAM at each stage + OOM traceback) and
`~/IG/probe_snap2.py` (allocation-trace replay attributing live bytes at peak by
call site). Both untracked. `probe_snap2.py` is the one that turned this from
guesswork into measurement — worth keeping.

---

# 2026-09-09 — Second GPU session: the completeness failure, resolved

`docs/MEMSCALE_RESULTS.md` §6a left `completeness` failing by 4.64x at L=730
with three unseparated causes. All three are now settled. The short version:
**bf16 was innocent, the quadrature was guilty, and m=16 was never enough.**

## 13. Completeness failed because m=16 under-resolves the path integral

`07_sanity --checks completeness`, chain subset H+L+A (L=554), single-tenant
A100-80GB, `IGV_TRI_ATTN_CKPT=1`, commit `a8e65b4`:

| arm | f(x) | f(baseline) | expected | sum(ig) | rel err | |
|---|---|---|---|---|---|---|
| bf16 m=8 | 3.648877 | 12.028320 | −8.379443 | **+21.602146** | 2.5781 | FAIL |
| bf16 m=16 | 3.649815 | 12.028320 | −8.378505 | −18.189129 | 1.1710 | FAIL |
| **bf16 m=32** | 3.656438 | 12.028320 | −8.371882 | **−8.741840** | **0.0442** | **PASS** |
| fp32 m=16 | 3.651461 | 11.749341 | −8.097880 | −22.029915 | 1.7205 | FAIL |

For reference, the previously recorded L=730 bf16 m=16 run: rel err 3.6387.

### bf16 is exonerated

This was the experiment §6a called decisive, and it needed L=554 because fp32
does not fit at 730 (79.57 GiB required against 79.20 usable). **fp32 does not
merely also fail — it fails worse than bf16 at the same L and m**, 1.7205
against 1.1710. Half precision cannot be what breaks completeness.

Two corollaries worth stating, because both were live worries:

- `IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16` stands. The memory fix did not buy
  55.22 GiB by corrupting the gradient.
- The score itself is barely dtype-sensitive here: f(x) is 3.649815 in bf16
  against 3.651461 in fp32, a 0.05% gap, far inside the ~1.6% run-to-run noise
  floor. bf16's fidelity on the forward pass is not in question either.

### The integral converges, non-monotonically, and late

The trajectory 2.5781 → 1.1710 → 0.0442 is real convergence, not luck: sum(ig)
goes 21.60 → −18.19 → −8.74 against an expected −8.37. But note **the sign flips
between m=8 and m=16**. That is not what a smooth integrand does under
Gauss-Legendre, which converges exponentially on analytic functions, and reading
it as proof of a divergent integral is a mistake this log records so it is not
made twice. It is a sharp feature being badly under-resolved and then resolved
once there are enough nodes.

Why the feature is there, most likely: the baseline is `torch.zeros_like(
s_inputs)`, an all-zeros embedding on no data manifold (f=12.03 there against
3.65 at the real input), and Boltz-2's trunk normalises its input. Normalisation
is homogeneous of degree zero, so `LN(alpha*x) = LN(x)` for alpha > 0 and the
composed function is near-flat along most of the path with everything happening
close to the origin. Gauss-Legendre's smallest node sits at alpha≈0.0198 for
m=8, 0.0052 for m=16 and 0.0013 for m=32, so each doubling reaches into that
region and the estimate only settles once it is resolved. **This is a hypothesis
with a measurement attached, not a conclusion** — `scripts/09_path_profile.py`
tests it by mapping F(alpha) and D_analytic(alpha) across three decades.

### The error scales steeply with L

rel err 1.1710 at L=554 against 3.6387 at L=730, while both endpoints barely
move (f(x) 3.650 against 3.855, f(baseline) 12.028 against 12.449). The error
tracks size, not the score. Two points fit an exponent near 4.1 in L, which is
far too steep to trust from two points, but the direction is unambiguous:
**a run at L=730 should be expected to need m>=64**, and the m=16 default in
`03_attribute` is wrong for the IG arm at full complex size.

Caveat to keep attached to that number: these ladder points differ in which
chains, not only in how many tokens, and MSA depth varies per subset
independently of L.

- **Root cause:** `check_completeness`'s m_steps=16 default, and
  `03_attribute --m-steps` defaulting to 15, are both below what this integrand
  needs at these sizes.
- **Fix:** not a code fix. Run the IG arm at m>=32, verified per L.
- **Status:** resolved at L=554. Unverified at 730, where m>=64 is the estimate.

## 14. Completeness was gating the wrong thing

Worth recording because it cost most of a session to notice. `completeness`
constrains `.ig = (x - baseline) * grad`. The pipeline's per-substitution
predictions use `.grad`, and `attrib.py:402-404` says so explicitly, listing the
`(x - baseline)` factor as belonging to the completeness identity alone.
`HANDOFF.md` §6 lists using `.ig` for predictions as a trap.

Further, the Makefile default is `METHOD ?= plain_grad` — a single gradient at
the real input, with no path, no baseline and no quadrature, for which
completeness is not merely satisfied but undefined. The efficiency claim that
motivates the project (~16 backward passes against ~615 GPU-hours) is one pass
per dataset, i.e. plain_grad; IG at m=16 would be sixteen times that and at the
m>=32 this entry establishes, thirty-two.

So a failing `completeness` gate blocks interpretation of the IG arm, and within
that arm it constrains a tensor no downstream stage reads. It is a real check and
worth keeping — it is what caught that m=16 is too coarse — but it is not
evidence about plain_grad, and `07_sanity`'s "GATE FAILED: downstream T1/T2/T3
numbers are NOT interpretable" text overstates its reach when the method under
test is plain_grad.

- **Status:** open as a documentation/gate-scoping issue. The gate text should
  name the arm it applies to.

## Environment notes from this session

- Syncing the VM by shipping a `.git` tarball and `git reset --hard` works, but
  build it with `COPYFILE_DISABLE=1` on macOS or it carries `._*` AppleDouble
  files that land untracked and flip `git_dirty` — as do the untracked
  `probe_*.py` diagnostics. Both are now gitignored, so this session's artifacts
  are the first recording `git_dirty: false`.
- Extract the tarball *over* the existing `.git` rather than deleting it, and
  check `git merge-base --is-ancestor` before resetting: the VM had commits the
  laptop did not on a previous occasion, and nothing warns you.
- Measured timings at L=554, single-tenant: bf16 forward+backward 55 s, fp32
  97 s — both within a second of the figures in `MEMSCALE_RESULTS.md` §6.

## 15. `random_weights` passed — the project's central risk is retired

First successful run in the project's history; it OOM'd on every prior attempt.
L=554, bf16, `IGV_TRI_ATTN_CKPT=1`, commit `e57e217`, artifact
`results/w6_randweights_L554.json`.

```
random_weights  PASS  spearman = 0.0767   (threshold |rho| < 0.3)
                reinitialised 5035 parameter tensors
                (3058 xavier, 1079 zeroed biases, 898 unit norms)
dead_target     PASS  max|grad| = 0.000e+00, f(x) = 3.657238
```

Randomising the weights destroys the attribution, so the gradient depends on
what Boltz-2 learned rather than on input geometry. This is the check whose
failure text reads "THIS KILLS THE PROJECT AS FRAMED", and it is the direct
answer to the off-simplex gradient-contamination concern (Majdandzic, Genome
Biology 2023) that `HANDOFF.md` §8 lists as a paper to engage with. It tests
`plain_gradient`, i.e. the headline method, not the IG arm.

Caveats: run at L=554, not the full 730. `dead_target` still returns exactly
zero every time and discriminates little, though it now carries a
model-dependent `f(x)` alongside.

## 16. One backward pass at the full complex: 104 seconds

`03_attribute --method plain_grad` at L=730 (all four chains), bf16, produced
`results/4fqi_h1_complex_pde_plain_grad_grad.npz` in **104.2 s**:

```
grad_chain (121, 384)   zero rows 0/121   abs max 0.0710918
per-residue norm: min 0.006782  median 0.01723  max 0.2056
top-8 residues by norm: [0, 102, 53, 54, 27, 24, 74, 101]
library variable positions: [28,29,30,51,56,57,58,70,73,74,75,76,83,86,94,105]
```

The 30x spread across residues rules out uniform noise, and several of the
highest-gradient residues (27, 53, 54, 74, 101, 102) sit in or beside the CDR
positions the library mutates. **Residue 0 topping the list is a flag** —
N-terminal artifacts are common — and should be remembered when reading T3.

This is the efficiency claim in concrete form: 104 s of gradient against the
~615 GPU-hours of brute-force scanning it is meant to replace.

## 17. Two bugs the repo's own guards caught in production

Both found by running, not by review, and both in stages that had never
executed successfully before.

1. **`02_embed_deltas` reused one `cache_dir` for every mutant.** boltz's
   `process_inputs` skips any input whose YAML stem is already in
   `<cache_dir>/processed/records`, and this repo always writes the stem
   `"input"`, so every mutant received the WILD-TYPE features and every delta
   would have been silently zero. `_check_featurised_sequences` stopped it:
   *"requested 'S', featurised 'F'"* at chain H index 28. `04_scan` already
   keyed its cache per row; this stage did not.

2. **`09_path_profile` called `torch.autograd.grad` under a reentrant
   checkpoint.** torch refuses outright: *"When use_reentrant=True,
   torch.utils.checkpoint is incompatible with .grad() or passing an `inputs`
   parameter to .backward()"*. Entry 9 records that the non-reentrant outer
   checkpoint OOMs, so reentrant is load-bearing and the CALL had to change.
   `attrib.integrated_gradient` already uses `backward()`; matching it also
   keeps the diagnostic on the same code path as the thing it diagnoses.

## 18. Featurisation is nondeterministic, and the cause is `ref_pos`

Chasing a failed assertion produced the most consequential finding of the
session. Stage 02 featurises the reference twice — once from the MSA server,
once re-read from the written files — and asserts the embeddings agree. It
failed:

```
Max abs difference between server-MSA and file-loaded-MSA reference
embeddings: 1.069308e+00   (tolerance 1e-4)
```

Three probes (`probe_msa.py`, `probe_msa2.py`, `probe_msa3.py`, committed in
`525433c` and since removed -- findings are recorded below) took it apart:

| comparison | max abs | rel |
|---|---|---|
| server MSA, run 1 vs run 2 | 1.527 | 3.4% |
| file MSA, run 1 vs run 2 | 1.136 | 2.8% |
| server vs file | 1.136 | 2.7% |

**MSA reuse is not lossy.** The server-vs-file gap is no larger than the gap
between two runs of the same path. The 1e-4 tolerance was asserting a
determinism the pipeline never had, against the wrong pair.

The embedder forward is **bit-deterministic** given fixed features — 0.0
difference across repeated calls, seeded or not, and the same for different
seeds — so this is not RNG in the model. `msa`, `res_type` and `token_index` are
identical across featurisations, and MSA depth is 9348 every time. Comparing
every feature tensor found exactly one culprit:

```
ref_pos   DIFFERS  max_abs = 10.7316   shape (1, 5728, 3)   float32
```

**Reference atom positions, varying by up to 10.7 Angstrom between two
featurisations of the identical input** — boltz's reference-conformer
generation (RDKit conformer embedding) is stochastic and unseeded.

Why it matters beyond stage 02: every embedding delta is `s_mut - s_ref`, and if
mutant and reference are featurised separately they carry independent `ref_pos`
draws, so the delta mixes the substitution with a conformer resample. Measured
at the mutated token, a real substitution delta is `|d| = 5.78`, against a
whole-tensor noise scale of ~1.1-1.5 — so the signal survives, but not by the
margin one would want, and it is contaminated for no reason.

- **Root cause — CORRECTED.** The original entry blamed "unseeded RDKit
  conformer generation". That is wrong, or at best secondary. Reading the boltz
  2.0.3 source directly:

  **Primary:** `boltz/data/feature/featurizerv2.py:1467-1473` loops over
  `ref_space_uid` groups — one per (chain, residue) — and calls
  `center_random_augmentation` on each with the default `augmentation=True`.
  That applies a random **rotation and translation**
  (`boltz/model/modules/utils.py:67-101`; `randomly_rotate` at `:93` ->
  `random_rotations` at `:60` -> `random_quaternions` at `:282`, drawing
  `torch.randn((n, 4))`). The inference dataset seeds **numpy only** —
  `boltz/data/module/inferencev2.py:270`, `seed = 42;
  random = np.random.default_rng(seed)` — and never calls
  `torch.manual_seed`. So the torch global RNG is unseeded and every residue's
  reference conformer gets a random orientation.

  A random rotation explains the 10.7 Angstrom magnitude far better than
  conformer resampling would; the latter perturbs locally, a rotation displaces
  the whole group.

  **Secondary:** `boltz/data/parse/schema.py:219-227` calls
  `AllChem.EmbedMolecule(mol, options)` with `ETKDGv3()`, whose `randomSeed`
  defaults to -1. This fires only for ligands and non-standard residues —
  canonical amino acids load pre-computed conformers from pickle via
  `load_canonicals` / `load_molecules`. So for our protein-only complexes it is
  not the operative path.

- **The near-miss.** The obvious worry is that seeding fixes run-to-run
  reproducibility without fixing the mutant-vs-reference delta: if RNG
  consumption scaled with atom count, a substitution (ALA 5 heavy atoms, TRP 14)
  would desynchronise every residue after the mutation site. Checked, and it
  does not — each call draws exactly **7 values regardless of atom count**: 4
  for the quaternion, 3 for the translation, because the translation is
  `torch.randn_like(atom_coords[:, 0:1, :])` and takes only the *first* atom's
  shape. Verified empirically across atom counts 3, 5, 7, 10, 14, 20, 50.
  Seeding alone would therefore have been sufficient.

- **Fix, implemented:** `src/igv/deterministic.py` provides a
  `deterministic_featurisation(seed)` context manager, wired into
  `build_complex_feats`. It patches `center_random_augmentation` to pass
  `augmentation=False` — keeping the centring, dropping the roto-translation —
  rather than relying on the 7-value invariant, which is undocumented and could
  change with any boltz release. Centring depends only on a residue's own atoms,
  so shared residues produce identical `ref_pos` by construction. Torch, numpy
  and Python RNGs are seeded and the RDKit entry points patched as
  defence-in-depth. Seed via `IGV_FEAT_SEED`, default 42.

  The random roto-translation is training-time data augmentation. There is no
  reason for it to run at inference at all.

- **Status:** implemented, **not yet proven**. `boltz` is not installed on the
  laptop, so the patch cannot be exercised locally and falls back to seeding
  alone with a debug log. `scripts/verify_deterministic_feats.py` is the proof:
  run it on the VM, where it featurises one input twice and asserts every tensor
  is byte-identical, then featurises a reference and a point mutant and checks
  `ref_pos` agrees while `res_type` differs at exactly one token. **It must exit
  0 before any `ref_pos`-dependent result is believed.**

- **Still open:** `03_attribute`'s reference is server-featurised (bare
  `cache_dir`) while stage 02's is file-featurised, so the gradient is taken at a
  slightly different point than the deltas expand around. The two stages should
  be put on one canonical featurisation.

Incidental: this is very likely the same root cause as the ~1.6% run-to-run
score noise recorded in `docs/MEMSCALE_RESULTS.md`, which was attributed to
kernel nondeterminism. It is not — the forward is deterministic; the *input* was
changing.

---

## 19. Burial alone predicts binding energy at rho ~0.5 — the bar before any gradient

Built the confound panel before running anything on GPU, and it changes how the
eventual result has to be read.

On **1JTG chain B** — 28 measured positions, 11 of them hot spots at
ddG >= 2.0 — computed from the PDB alone, no model involved:

| Confound | Spearman vs ddG (max per position) |
|---|---|
| **burial** (heavy atoms within 10 A) | **+0.54** |
| **residue volume** | **+0.48** |
| distance to binding partner | -0.32 |
| hydrophobicity | -0.14 |
| normalised position in chain | +0.11 |

**Counting how many heavy atoms sit near a residue predicts its contribution to
binding energy at rho ~0.5.** No gradient, no GPU, no Boltz-2. That is the bar.
An attribution scoring 0.5 against ddG would have told us nothing a distance
calculation would not.

This is precisely the failure mode of arXiv:2606.22181, where attribution on
allergenicity classifiers faithfully reported that the models were leaning on
*"physicochemical and compositional sequence features"* rather than biology. The
attribution was honest; the signal was a shortcut. **The partial correlation —
does the gradient predict ddG after burial and residue size are regressed out? —
is the load-bearing number, not the raw Spearman.**

- **Sensitive to aggregation.** Burial vs ddG is +0.54 taking the max signed ddG
  per position, +0.59 taking max |ddG|, +0.47 taking the mean. Report the
  aggregation with the number; the stage-10 script emits both max and mean for
  this reason.
- **The bar itself is poorly determined.** Bootstrap 95% CI on burial is
  **[+0.21, +0.78]** at n=28. Wide enough that pooling 3HFM, 1VFB, 1JRH and 2JEL
  moves from "nice to have" to close to mandatory before any claim is made.
- **Distance to partner is weak for a boring reason.** SKEMPI only measured
  interface residues, so they are all close to the partner and the variable
  cannot discriminate. Range restriction, not absence of an effect — do not read
  -0.32 as "the interface does not matter".
- Random-gradient control behaves: rho = +0.04, AUROC 0.56, AUPRC 0.49, so the
  shuffled null sits where it should.

- **Status:** measured and recorded. `scripts/10_skempi_hotspots.py` reports the
  confound panel, partial correlation, AUROC/AUPRC, a shuffled-ranking null and
  bootstrap CIs alongside the headline number.

---

# 2026-09-24 — First end-to-end numbers, on 1JTG chain B (A100-80GB, us-central1-a)

Determinism proven, the pipeline produced its first real correlations, and the
cause of the early nulls was found. ~4.7 h of A100, ~$24.

## 20. `ref_pos` determinism fix VERIFIED on the VM — entry 18 is closed

`scripts/verify_deterministic_feats.py` exits 0 on 1JTG:

- **Part 1:** all 78 feature tensors byte-identical across two featurisations of
  one input. `ref_pos` max_abs_diff **0.0000**, against the 10.7 A of entry 18.
- **Part 2:** reference vs a D131A point mutant — `ref_pos` changes are
  **confined to token 131**; all 421 other tokens byte-identical. Real atoms
  3242 -> 3239, exactly the three heavy atoms Asp loses becoming Ala.

Every embedding delta is now clean. This was the blocker for every number the
project wants.

- **Status:** entry 18 RESOLVED and verified. The 2.0 tolerance that
  `02_embed_deltas.py` carries as a stand-in can now be tightened.

### Three iterations of the *test* were wrong before the fix was confirmed

Worth recording, because each looked like the fix failing:

1. **Crash** — `max()` on a zero-size tensor, killing the comparison before it
   reached `ref_pos`. Boltz emits empty tensors for `chiral_*` / `connected_*` /
   `contact_pair_index` on a protein-only complex. Same crash `probe_msa3.py`
   hit; the guard had not been carried over.
2. **290 "leaked" tokens** — an element-wise `ref_pos` diff between reference and
   mutant. A substitution changes the heavy-atom count, so every atom after the
   mutation site shifts index while both tensors stay padded to 3264. The
   comparison was measuring neighbours against each other. Fixed by gathering
   each token's atoms through that featurisation's **own** `atom_to_token`.
3. **Token 0 mismatch** — padding atoms carry an all-zero `atom_to_token` row,
   and `argmax` on all zeros returns 0, so every pad atom was attributed to
   token 0. The mutant has three more pad atoms. Fixed by masking on
   `atom_pad_mask`.

**Lesson:** a global `max_abs_diff` cannot answer "did anything leak". The
original script printed one and asserted the difference was "expected ONLY at
the mutated residue's atoms" without checking. Localise, or do not claim it.

## 21. The all-zeros IG baseline was the cause of the early null results

Three runs on 1JTG chain B, `complex_pde`, fp32 + `IGV_TRI_ATTN_CKPT=1`,
427 tokens, against SKEMPI's 96 measured mutations at 28 positions:

| run | Spearman | null p95 | above null | partial | AUROC | null p95 |
|---|---|---|---|---|---|---|
| plain_grad (zeros) | 0.306 | 0.318 | no | 0.180 | 0.685 | 0.695 |
| IG m=32 (zeros) | 0.238 | 0.336 | no | 0.010 | 0.583 | 0.690 |
| **IG m=32 (mean_aa)** | **0.357** | 0.313 | **YES** | 0.211 | **0.695** | 0.679 |

**Why the first two failed.** `complex_pde` is a predicted distance error, lower
being better. The zeros baseline is not a protein and scores **11.4310**; the
real complex scores **2.6200**. The path integral was dominated by the model
reacting to an off-manifold input, which is also the earlier alpha-profile
observation (53% of the score change in the first 5% of the path) seen from a
different angle.

| baseline | f(baseline) | span | rel err | **abs err** |
|---|---|---|---|---|
| zeros | 11.4310 | -8.8110 | 17.28% | 1.5222 |
| mean_aa | 3.2491 | -0.6290 | 8.01% | **0.0504** |

- **The 5% completeness criterion is miscalibrated for a good baseline.** It is a
  *relative* threshold. Going to `mean_aa` improved the absolute error **30x**
  (1.52 -> 0.05) while the span shrank 14x, so the relative figure only halved
  and still "FAILS". Do not read 8.01% as worse than it is, and consider gating
  on absolute error, or on a relative error scaled to the score's own noise.
- **Path integration is not a rounding error on the plain gradient.** Spearman
  between the two per-residue norm rankings is +0.58, cosine 0.55, only **3 of
  the top 10 residues agree**, and IG norms are 3.6x larger. So `plain_grad` is
  not a cheap substitute for `ig` — the earlier open question is answered.
- **Do not oversell the positive.** Its 95% CI is **[-0.002, 0.656]**, excluding
  zero by two thousandths. The partial correlation is 0.211 with CI
  [-0.34, +0.56], against burial's 0.436. And gradient-vs-hydrophobicity is
  **-0.527, CI [-0.75, -0.18]** — excludes zero, while gradient-vs-ddG barely
  does. The clearest thing this attribution tracks is still a compositional
  property.
- **Status:** `--baseline mean_aa` implemented and used. `zeros` retained
  bit-identical for reproducing prior artifacts.

## 22. The alanine-scan size confound is MECHANISTIC, not statistical

The confound panel (entry 19) cannot simply be regressed away, and the ground
truth shows why. 1JTG chain B's 11 hot spots are W150, K74, H41, F36, H148, Y53,
W112, R160, F142, D49, W162 — Trp, His, Phe, Tyr, Arg, Lys, Asp. Textbook
hot-spot composition.

SKEMPI is ~90% X->Alanine. **X->A deletes a large side chain**, so ddG partly
measures how much volume was removed. Residue volume correlating with ddG at
+0.44 is therefore not a spurious confound to control for — it is what the assay
measures. Any method that learns "large aromatic at an interface matters" scores
respectably without representing binding at all.

- **Consequence:** an alanine scan can validate *which positions matter*. It
  cannot validate *how each of the 20 amino acids performs*, which is the
  project's actual claim, and it structurally favours the confound.
- **The fix is saturation data**, where all 20 substitutions are measured at a
  position. That permits a **within-position** comparison in which burial,
  exposure and distance-to-partner are constant and cancel out entirely.

## 23. SKEMPI's mutation count is replicates, not breadth

1JTG chain B reports 96 mutations, but they are **28 distinct mutations measured
repeatedly** from different literature sources. Position W150 carries five
separate W->A measurements: +4.81, +4.66, +4.34, +4.25, +3.50.

- **Experimental noise is ~±0.6 kcal/mol**, read straight off that spread. A
  perfect predictor could not correlate at 1.0 against this.
- **Effective n is 28, not 96.** Every confidence interval in entries 19 and 21
  reflects that, and it is why they are so wide. Pooling 3HFM, 1VFB, 1JRH and
  2JEL to ~128 positions is the cheapest available fix.
- **Max-aggregation hides sign reversals.** Position Y50's four measurements are
  -0.41, -2.05, -2.11, -2.26 — mutating it *improves* binding — yet it enters as
  -0.41 under max. Same for D163. Report mean as primary where the sign is
  consistent; `10_skempi_hotspots.py` emits both.

## 24. The boltz cache trap, third occurrence — now in stage 03

- **Symptom:** the first real 1JTG run died in `_build_token_map`:
  `Found 4 contiguous asym_id runs but 2 chains were supplied: run lengths
  [121, 324, 176, 109], chain lengths {'A': 262, 'B': 165}`, with the log line
  "All inputs are already processed. Processing 0 inputs with 0 threads."
- **Root cause:** `[121, 324, 176, 109]` is `4fqi_hlab` (H, A, B, L). boltz's
  `process_inputs` skips any input whose YAML stem already exists under
  `<cache_dir>/processed/records`, and this repo always writes the stem
  `"input"`. `03_attribute.py` passed the bare `data/raw`, populated by the 4fqi
  sessions, so a 1JTG request returned **4FQI's features**.
- **Without the guard we would have computed attributions on the wrong protein
  and labelled them 1JTG.** Stage 02 already had per-substitution dirs from
  entry 17; stage 03 did not.
- **Fix:** `cache_dir / "boltz_attr" / f"{dataset}_{chain}"`. `07_sanity.py` keyed
  on chain subset only, accidentally safe while 4fqi was the sole complex; now
  keyed on dataset too.
- **Status:** fixed. Any new call to `build_complex_feats` needs its own cache
  dir — treat a shared one as a bug on sight.

---

## 25. The within-position design removes the volume confound — verified on GB1

- **Symptom (not a crash):** every 1JTG result was uninterpretable because
  residue volume predicts `|ddG|` at **+0.44** on pooled SKEMPI data (entry 22),
  so "the gradient ranks hot spots" and "the gradient ranks big side chains"
  make the same prediction. No amount of partial correlation at n=28 separates
  them.
- **Verified fix, on real saturation data** (Olson 2014 GB1/IgG-Fc, 1045 singles
  = 55 positions x 19, every position saturated):

  | Design | Volume vs binding effect |
  |---|---|
  | Pooled | +0.44 |
  | Within-position | **mean +0.135, median +0.197** |

  Range across positions -0.92..+0.71, with **20 of 54 positions negative** — so
  the residual cancels on pooling rather than accumulating. Hydrophobicity
  behaves the same way (mean +0.181, median +0.117).
- **Why it works:** holding the position fixed holds burial, solvent exposure and
  distance-to-partner fixed, because they are properties of the site and not of
  the substitution. The entire confound panel is constant within a row and
  divides out. This is an experimental-design fix, not a statistical one, which
  is why it succeeds where regressing the confounds away could not.
- **Status:** adopted. All saturation-arm analysis is within-position by default;
  pooled correlations are reported only for continuity with the SKEMPI runs.

---

## 26. GB1 rejected as the saturation target — censored at exactly the hot spots

- **Symptom:** GB1 produced the entry-25 result and is cheap (56 residues), so it
  was the obvious primary target. Inspecting it closely killed it.
- **What the data actually shows:** the floor is exactly `ln(0.01) = -4.60517`
  with 87 values pinned there, and the censoring is *not* spread evenly —
  **position 27 has std 0.000, all 19 substitutions at the floor**, and position
  43 (Trp) has 17/19. 13 of 55 positions are flat (std < 0.3), 6 have more than
  3 values at the floor. Within-position ranking is undefined at precisely the
  residues that matter most.
- **Three further defects, each independently disqualifying:**
  - `W = f_N * W_N` by the authors' own definition, so folding and binding are
    coupled. For a *structure* model the likeliest null story is "the gradient
    tracks foldability", and GB1 cannot rule it out without merging in a
    separate paper (Nisthal 2019).
  - 1FCC is protein G **C2** (P19909), not **B1** (P06654) — the wrong
    paralogue. 3 of 56 positions differ, 57 wild-type mismatches, 988/1045
    mutants map cleanly, numbering offset exactly -226. No better PDB entry.
  - `lnW` is not ddG, so nothing pools or plots with the SKEMPI results.
- **Trap recorded for anyone who revisits it:** 1FCC chains A and B are an
  **obligate Fc homodimer, 2.34 A apart** — not redundant copies like 1JTG's
  C/D. Dropping B to save tokens models a half-molecule that does not exist in
  solution. Use A+B+C = 468 tokens; drop only D.
- **Status:** rejected in favour of Starr 2020 / 6M0J, which has real affinities
  and a matched expression control in the same file. See ROADMAP section 10.

---

## 27. Boltz-2's AbBiBench score is anti-correlated with per-amino-acid structure

- **Symptom:** shopping for a saturation dataset inside AbBiBench, the datasets
  where Boltz-2 already scores well are the ones that cannot test our claim.
- **Measured:** `2fjg` is a genuine complete saturation scan — verified 2223 rows
  = 117 positions x 19, 2223 unique heavy sequences, all singles, consensus ==
  PDB wild type. Boltz-2's per-dataset score there is **0.08**. Meanwhile
  `3gbn_h1` scores **0.71** with 11 positions x 1 alternative, **zero** true
  single mutants, and an 11% floor; `4fqi_h3` is **89% floor-censored**.
- **Consequence:** a high AbBiBench number is evidence about dataset shape, not
  about Boltz-2's per-amino-acid resolution. Recorded so the 0.71 is never cited
  as encouragement for this project.
- **Correction to an earlier characterisation of `4fqi_h1`:** computing variant
  identity against the dataset *consensus* gives 16 singles and a modal 8
  mutations. The consensus differs from the 4FQI structural wild type at **9 of
  16 varying positions**, so against the structure we actually model there are
  **0 true single mutants and a modal 10 mutations**. Always diff against the
  PDB sequence, never the dataset consensus.
- **Status:** closed. AbBiBench remains useful for structures and sequences, not
  as a leaderboard.

---

## 28. The folding confound on 6M0J is real but lives outside the interface

- **Why this was the open question:** for a *structure* model like Boltz-2 the
  most likely null story is that the gradient tracks foldability, not binding.
  GB1 could not rule it out (entry 26). Starr 2020 reports a matched
  **expression** readout for the same variants in the same assay, so it can.
- **The global number says the confound is fatal.** Pooled over 3669 usable
  singles, binding and expression correlate at **+0.64** (r2 0.40); the
  within-position median is **+0.75**, with only 8 of 201 positions negative.
  Taken alone this would mean predicting stability gets you binding for free.
- **Split by distance to ACE2, the picture inverts:**

  | | n positions | bind~expr median | bind std | expr std |
  |---|---|---|---|---|
  | Interface (<=5 A) | 21 | **+0.38** | **0.695** | 0.275 |
  | Non-interface | 173 | +0.752 | 0.338 | 0.481 |

  Pooled over interface mutants only, the correlation is **+0.074**, and
  residualising binding on expression keeps **100%** of the binding variance
  (std 1.224 -> 1.221). At the interface binding varies most and expression
  varies least; away from it the reverse. The entanglement is entirely a
  non-interface phenomenon — destabilise the fold and both readouts die
  together, which says nothing about recognition.
- **Consequence:** the primary analysis is **interface positions only, within
  position**, with expression reported as a control rather than regressed out.
  Reporting the pooled +0.64 as if it applied to the claim would be wrong in
  the pessimistic direction, and quoting an interface result without the
  non-interface contrast would be wrong in the optimistic direction. Report
  both.
- **Independent check that the cutoff is not arbitrary:** all 17 literature
  ACE2 contact residues fall inside the 5 A set, including the
  variant-of-concern positions K417, E484 and N501.
- **Status:** resolved in favour of the dataset. This is the control GB1 lacked,
  and it came out well.
