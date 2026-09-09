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
  full command + a troubleshooting row), `bootstrap.sh`, `gcp_launch.sh`,
  `aws_launch.sh` (commit `502f207`).
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
