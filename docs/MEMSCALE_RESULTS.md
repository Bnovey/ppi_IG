# Measured: the full-trunk backward at L=730 fits, in bf16, on one A100-80GB

Run 2026-09-04 on `igv-gpu` (1x A100-SXM4-80GB, 79.2 GiB usable, us-central1-a),
at commit `808dd23` with the working tree dirty (every artifact records
`git_commit: 808dd23`, `git_dirty: true`).

This is the measurement record for the levers `docs/MEMORY.md` describes. Read
that document for the diagnosis and the literature; read `ERRORS_LOG.md` entry
12 for the history. **This file corrects the conclusions of both in four
places and closes two more as verified**, all listed in section 5.

---

## 0. Headline

`IGV_TRI_ATTN_CKPT=1` + `IGV_AUTOCAST=bf16` completes a full-trunk
forward+backward at **L=730 tokens in 55.22 GiB**, with 24 GiB of headroom, in
108 s. No stage had ever completed at 730 before. The frozen-trunk fallback
decision in entry 12 is no longer forced, and no larger card is needed.

**fp32 does not fit at any setting tested** -- the fitted requirement is
79.57 GiB against 79.20 GiB usable, and five checkpoint group sizes all OOM.
The 0.37 GiB shortfall is in the *retained* term, unreachable by chunking or
grouping. bf16 is load-bearing, not a substitute for tuning.

**But the memory result is not a correctness result.** Run under this config,
`07_sanity`'s `completeness` check produced a number for the first time in the
project's history and **failed by 4.64x** (section 6a). The cause is not
isolated and may well be dtype-independent -- i.e. a flaw the OOMs have been
masking all along. Do not read "55.22 GiB, completed" as "the attribution is
valid".

---

## 1. The three ladders

Whole-chain subsets of `4fqi_hlab` (A=324, B=176, H=121, L=109). Forced chunk
profile `large`, `recycling_steps=1`, `gradient_checkpointing=True`,
`plain_grad`, `complex_pde`, MSA from server, `IGV_PF_GROUP_SIZE=4` (default),
`IGV_PF_CHUNK=128`.

| L | subset | baseline | +`TRI_ATTN_CKPT` | +`TRI_ATTN_CKPT` +bf16 |
|---|---|---|---|---|
| 230 | H+L | 22.20 | 21.64 | 15.57 |
| 406 | H+L+B | 42.05 | 39.43 | 27.43 |
| 500 | A+B | 59.12 | 54.45 | 37.70 |
| 554 | H+L+A | 66.81 | 56.21 | 39.06 |
| **730** | A+B+H+L | **OOM** 78.20 | **OOM** 78.41 | **55.22 COMPLETED** |

Fits on `completed=True` rows only, extrapolated to 730, plus the 0.78 GiB CUDA
context:

| arm | exponent b | R2 | requirement at 730 |
|---|---|---|---|
| baseline | 1.2542 | 0.9941 | **93.66 GiB** |
| +ckpt | 1.1234 | 0.9915 | **79.57 GiB** |
| +ckpt +bf16 | 1.0966 | 0.9941 | **55.28 GiB** |

The bf16 fit includes its own 730 row, so it is a consistency check rather than
a prediction. The genuine out-of-sample test: fitting only the bf16 arm's four
sub-730 rows predicts 53.8 GiB, and the measured value is 55.22 -- **2.6% out
of sample**, which is the strongest evidence that the ladder-and-fit method
works.

---

## 2. Why the previous nine measurements all read ~78 GiB

`torch.cuda.max_memory_allocated()` on a run that OOMs reports how far the run
got, not what it needed -- it reports the size of the *card*. The two fp32 arms
make this unmissable:

| arm | OOM peak "measured" | true requirement | difference |
|---|---|---|---|
| baseline | 78.20 GiB | 93.66 GiB | -- |
| +ckpt | 78.41 GiB | 79.57 GiB | **14.09 GiB apart** |

Two configurations whose real costs differ by 14 GiB reported peaks 0.2 GiB
apart. `docs/MEMORY.md` section 1 argued this from the allocation traces; it is
now demonstrated directly. **A peak from a run that did not complete carries no
information about the requirement.** `08_memscale.py` enforces that
structurally and earned its existence here.

The corollary matters for planning: the measured growth exponent is **1.10-1.25,
not the ~2.7** extrapolated from MegaFold. Chunking and per-block checkpointing
had already harvested the cubic terms, leaving a large L-independent constant.
Baseline was short by 18%, not by 250-400%.

---

## 3. The group-size knob: 4 is already right

Single-point probes at L=730, `IGV_TRI_ATTN_CKPT=1` throughout.

| dtype | group | chunk | result |
|---|---|---|---|
| fp32 | 1 | 128 | OOM 78.28 |
| fp32 | 2 | 128 | OOM 77.87 |
| fp32 | 4 | 128 | OOM 78.41 |
| fp32 | 8 | 128 | OOM 78.37 |
| fp32 | 8 | 16 | OOM 77.74 |
| fp32 | 16 | 128 | OOM 77.73 |
| bf16 | 16 | 16 | **OOM 78.22** |
| bf16 | 1 | 128 | 67.47 completed |
| **bf16** | **4** | **128** | **55.22 completed** |

Two findings:

1. **The knob is U-shaped and the existing default wins.** bf16 at group 16
   OOMs; bf16 at group 1 costs 12.25 GiB more than group 4. `_PF_GROUP_SIZE = 4`
   in `boltz_score.py` is near-optimal and should not be changed.

2. **A hypothesis worth recording as refuted.** Entry 12 found group size to be
   a break-even because a larger group shrank retained boundary z
   (34.30 -> 4.32 GiB) while inflating the retained softmax (11.59 -> 25.22).
   Since `IGV_TRI_ATTN_CKPT` deletes that softmax, a larger group *should* have
   become nearly free. It did not: going from group 4 to 16 in bf16 costs more
   than 23 GiB. **The dominant term in the group knob is the recompute
   transient** -- a bigger group means more blocks recomputed simultaneously in
   the backward -- not the retained boundary tensors. Entry 12's boundary-z
   column points one way and its total points the other; the total is what
   matters.

Also refuted: shrinking `IGV_PF_CHUNK` 128 -> 16 reduces the failing transient
from 1.0164 to 0.1271 GiB and moves the wall by nothing (77.74 vs 78.37). The
allocator simply fails on the next allocation. The fp32 gap is in the retained
term.

---

## 4. What bf16 costs numerically

`grad_abs_max` per arm, and the relative difference against baseline:

| L | baseline | +ckpt | +ckpt +bf16 | ckpt vs base | bf16 vs base |
|---|---|---|---|---|---|
| 230 | 0.103599 | 0.105340 | 0.102984 | +1.68% | -0.59% |
| 406 | 0.370585 | 0.364636 | 0.357257 | -1.61% | -3.60% |
| 500 | 0.380748 | 0.374850 | 0.374578 | -1.55% | -1.62% |
| 554 | 0.451731 | 0.450485 | 0.455825 | -0.28% | +0.91% |

**Read the `ckpt` column first: it is the control.** Per-chunk checkpointing is
mathematically transparent -- it recomputes the identical softmax -- yet moves
`grad_abs_max` by up to 1.68%. So **~1.7% is this pipeline's nondeterminism
floor** (reduction order, chunk paths), and bf16's <=3.6% is roughly 2x that
floor, not orders of magnitude. For scale, entry 12 recorded
`recycling_steps=0` moving `complex_pde` 3.925568 -> 4.309255 (~9.8%) and
treated it as disqualifying.

Group size barely touches the gradient: at 730 in bf16, group 1 gives 0.478486
and group 4 gives 0.476665, a 0.38% difference.

**This is necessary, not sufficient.** `grad_abs_max` is one scalar -- not the
score, not the gradient direction. The load-bearing validation is
`07_sanity`'s `completeness` and `frozen_vs_full` under the new dtype.

**Any score recorded under bf16 must be re-derived and never compared against
an fp32 one.** `numerics_arm()` now stamps the dtype into every stage's
provenance `arm` dict so the two cannot be silently mixed.

---

## 5. Corrections to the existing documents

1. **`docs/MEMORY.md` section 3 -- "short by 2.5-4x", requirement "180-320 GB".**
   Wrong. Measured requirement at L=730 is 93.66 GiB in fp32 baseline. The
   literature exponent of ~2.7 does not describe this configuration; the
   measured exponent is 1.10-1.25.

2. **`docs/MEMORY.md` section 6 -- "model parameters are never frozen", implying
   a full fp32 gradient copy of the weights.** Measured: 25,026,048 of
   506,724,992 parameters have `requires_grad` (4.9%), all in
   `confidence_module`. The gradient copy is **0.09 GiB**. boltz already freezes
   the rest. Closed as negligible.

3. **`ERRORS_LOG.md` entry 12 -- "no fix on this hardware", and the frozen-trunk
   decision.** A fix exists on this hardware. Entry 12 is append-only and stays
   as written; this is the follow-up.

4. **The "GEOMETRY IS FIXED ... identical wild-type geometry" claim** in
   `03_attribute.py` and the `"geometry": "fixed_wt"` provenance label in
   `04_scan.py`. Measured `feats['coords'].abs().max() = 0` at every ladder
   point and in the sanity run: the sequence-only YAML places **every atom at
   the origin**. Geometry is fixed, but it is not the deposited structure, and
   `data/raw/4fqi_hlab.pdb` contributes only chain lengths. `03_attribute.py`
   now logs and records the measured value and warns when it is zero;
   `04_scan.py`'s label is now `fixed_from_featurisation`. **The scientific
   consequence is not resolved and is the largest open question in the repo.**

5. **`docs/MEMORY.md` section 6 -- `PairformerLayer`'s trailing positional arg
   order "assumed", to be confirmed before trusting `IGV_TRI_ATTN_KERNEL`.**
   Confirmed correct by signature inspection against pinned boltz 2.2.1:
   `forward(self, s, z, mask, pair_mask, chunk_size_tri_attn=None,
   use_kernels=False, use_cuequiv_mul=False, use_cuequiv_attn=False)`. Likewise
   `TriangleAttention._chunk(self, x, tri_bias, mask_bias, mask, chunk_size,
   use_kernels=False)` matches the `IGV_TRI_ATTN_CKPT` wrapper exactly. Both
   closed.

6. **`docs/MEMORY.md` section 5.1 -- `IGV_ASSERT_FEAT_SEQ`'s `res_type` one-hot
   decode "has never met real boltz features".** It has now: on by default
   through 15 ladder points and two sanity runs at L=230-730, and it never
   raised. The decode works and the stale-cache hazard did not fire. Closed.

---

## 6. Recommended configuration

```bash
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16   # group size and chunk: leave at defaults
```

Also faster than today's baseline. At L=554, warm MSA cache: baseline 97.5 s
versus 55.2 s. L=730 completes in 108 s.

Caveat on timings: the `ckpt` arm ran first with a **cold** MSA cache, so its
wall times (87.7-227.4 s) include ColabFold fetches and are not comparable.
Checkpointing's true compute overhead is not measured by this sweep.

---

## 6a. The sanity gate under the winning config: completeness FAILS

`07_sanity --checks completeness,frozen_vs_full,dead_target,arm_assertion`
under `IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16`, single-tenant GPU, 2026-09-04
21:09-21:40. Artifact `results/sanity_bf16_4fqi_h1_complex_pde.json`.

| check | result | value |
|---|---|---|
| completeness | **FAIL** | **3.6387** (threshold < 0.05) |
| frozen_vs_full | OOM (informational, non-blocking) | -- |
| dead_target | PASS | `max|grad| = 0.000e+00`, `f(x) = 3.855304` |
| arm_assertion | PASS (trivially, no artifacts yet) | -- |

```
f(x)        =   3.855304
f(baseline) =  12.449280   -> expected sum(ig) = -8.593976
sum(ig)     = -39.865089   -> overshoot 4.64x
```

**This is the first completeness number the project has ever obtained** -- the
check OOM'd in every previous run -- and the integrated gradient does not sum to
the score difference. 16 Gauss-Legendre steps ran to completion in 28.6 min
with **zero exceptions in the log**, so this is a threshold failure, not a
crash.

**Do not attribute this to bf16 without a control.** Three candidate causes,
none yet separated:

1. bf16 damaged the gradient. There is **no fp32 control at L=730** and there
   cannot be one -- fp32 does not fit (section 1). The check has never passed in
   any dtype.
2. m=16 quadrature is not converged. This is exactly what `check_m_sweep`
   (`ms=(8, 16, 32)`) tests, and it was skipped for cost.
3. The zeros baseline is far off-manifold: `f(baseline)=12.449` against
   `f(x)=3.855` is a different regime entirely, and IG completeness degrades
   when the path leaves the data manifold.

A 4.64x overshoot is large for pure quadrature error, which points at (3)
and (1).

**The decisive and cheap experiment: run `completeness` at L=554 in BOTH
dtypes.** fp32 fits at 554 (56.21 GiB measured), so a control is possible
there. If fp32 also fails at ~3.6, bf16 is exonerated and this is a
pre-existing flaw in the attribution setup that the OOMs were masking all
along. ~15 min per dtype.

Two fixes from the previous session are confirmed working in production:

- **`frozen_vs_full`'s OOM did not block the gate.** It is recorded `INFO` and
  `GATE FAILED` names only `completeness` -- `MEMORY.md` section 5.3 bug 3
  (the `informational` flag set only on the success path) is genuinely fixed.
  Note it OOMs *even in bf16*, single-tenant, at 79.12 GiB in use: it is
  heavier than plain IG.
- **`dead_target` is no longer a tautology.** It reports a model-dependent
  `f(x)` alongside an exactly-zero gradient.

### Run-to-run noise is ~1.6% on the score

The two bf16 runs of `dead_target` gave `f(x) = 3.916786` and `3.855304` --
a **1.6% spread for an identical configuration**. So a single-pair comparison
against entry 12's fp32 `3.925568` cannot establish a dtype shift; the noise is
the same size. Consistent with the ~1.7% floor measured on `grad_abs_max`
(section 4). **bf16's score shift is <=2% and not separable from noise at this
sample size.** Any real determination needs repeated runs, not one pair.

### Bottom line

Memory is solved. Correctness is **not** established. The memory wall was
concealing a completeness failure; removing it revealed the next question
rather than answering it.

---

## 7. Still open

- **Why completeness fails by 4.64x** (section 6a). Now the largest open item,
  jointly with the zero geometry. Next experiment: `completeness` at L=554 in
  fp32 and bf16, to get the dtype control that L=730 cannot provide.
- **The zero-geometry question** (section 5, item 4). Plausibly *related* to the
  completeness failure: the score is being evaluated at a collapsed all-zeros
  structure, and there is no reason to expect the path from a zeros embedding
  baseline to be well behaved in that regime.
- **`m_sweep` has never run.** At `ms=(8, 16, 32)` it is 56 backward passes
  (~98 min at 730). It is the direct test of candidate cause 2 for the
  completeness failure, so it is no longer merely a nice-to-have.
- **`frozen_vs_full` OOMs even in bf16** at 730 (79.12 GiB in use). It needs its
  own memory work, or to be run at a smaller L.
- **Every reference score needs re-deriving under bf16**, including
  `signal_control`'s 30 mutants and the `complex_pde` reference of 3.925568.
- **Nothing is committed.** Artifacts record `git_dirty: true`, so these
  numbers are honest but not reproducible from a commit alone.
- **A PyTorch bug that masks OOMs.** An OOM inside the checkpoint body unwinds
  through `torch/autograd/graph.py:314`, pops the saved-tensors hook stack and
  raises `RuntimeError: is_initialized && !tls.stack.empty() INTERNAL ASSERT
  FAILED at SavedTensorHooks.cpp:69`. The real cause is reported as a PyTorch
  internal assert. Seen once, at fp32 group 8.

## 7a. Operational trap: `tmux kill-session` does not stop a container

Recorded because it produced a completely convincing false result.

The first `07_sanity` run was launched as `tmux new -s sanity -d "sudo docker
run ... "`. Killing that tmux session ended the tmux window and the `docker run`
*client*, but **the container kept running and kept its 57 GiB of VRAM**. The
replacement run therefore had ~22 GiB to work with and failed with:

```
OutOfMemoryError: Tried to allocate 834.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 109.00 MiB is free.
this process has 22.15 GiB memory in use.
```

`completeness` and `frozen_vs_full` both "failed", and the gate printed
`GATE FAILED`. Read carelessly that says bf16 breaks the gradient. It says
nothing of the kind -- the card was shared.

**The tell is in the message and costs nothing to check: "this process has
22.15 GiB in use" while only 109 MiB of 79.25 GiB is free.** When those two
numbers do not reconcile, the GPU has another tenant. Confirm with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

Rules: run containers with `--name` and stop them with `docker kill <name>`,
never by killing the tmux session; check `nvidia-smi` shows 0 MiB used before
starting a measurement run. The invalid artifact is kept as
`results/INVALID_contended_gpu_sanity_bf16.json` rather than deleted, so the
filename carries the reason.

---

## 7b. Code changed while measuring

Five defects surfaced during this session. All are in the working tree,
uncommitted, with 236 tests passing and no new ruff findings.

1. **Nine tests asserted a property of the HOST, not of the code.**
   `tests/test_gpu.py` and `tests/test_memscale.py` contained
   `assert gpu.has_cuda() is False` and
   `pytest.raises(ImportError): __import__("boltz")`. Green on a laptop, **red
   on the GPU box** -- so `igv.gpu`'s CPU-degradation contract was untested
   precisely where a regression matters, and `MEMORY.md` section 5.4's "235
   tests pass" was a laptop-only claim. Fixed with a `cpu_only` fixture that
   swaps `sys.modules["torch"]` for a fake reporting `is_available() == False`
   (covering both the lazy `_cuda_torch()` seam and the direct `import torch`
   inside `require_vram`), and an import-safety test that checks in a
   subprocess that importing `08_memscale` pulls in neither torch nor boltz.
   Verified 236/236 in-container on igv-gpu.

2. **`IGV_TRI_ATTN_CKPT` warned "Expect no memory change" while saving 14 GiB.**
   The rebind is idempotent, so any caller reusing one model object -- every
   `08_memscale` ladder point after the first, every `07_sanity` check after
   the first -- wrapped 0 modules and triggered the warning while all 156
   remained instrumented. Added `count_tri_attn_chunk_ckpt(model)` so the
   question is asked of the model rather than of the last call; the warning now
   fires only when nothing is instrumented. Confirmed in the sanity log:
   `156 newly wrapped, 0 already instrumented` then
   `0 newly wrapped, 156 already instrumented ... still in effect`.

3. **Provenance could not distinguish a bf16 artifact from an fp32 one.**
   Stages 02/03/04/07 recorded no dtype in their `arm` dict -- the exact hazard
   `MEMORY.md` section 7 named, live as soon as bf16 became the working config.
   Added `numerics_arm()` (autocast, dtype, tri_attn_ckpt, pf_group_size,
   pf_chunk, chunk_profile, use_kernels) and wired it into all four stages.
   `assert_provenance` only iterates the keys a caller passes, so the extra keys
   cannot break `arm_assertion`. `08_memscale` was already honest via its own
   `pinned_env`.

4. **A false claim baked into every scan artifact.** `04_scan.py` stamped
   `"geometry": "fixed_wt"`; measured `coords.abs().max() == 0`. Now
   `"fixed_from_featurisation"`, and `03_attribute.py` records the *measured*
   `coords_abs_max` in provenance and warns when it is zero, so no reader has to
   trust a label.

5. **A PyTorch bug that disguises OOMs** (section 7). Not fixable here;
   recorded so the next person is not misled.

Also added: `probe_params.py` (CPU-only trainable-parameter count). The
single-point probe drivers (`run_wave2.sh`, `run_wave3.sh`) were removed after
their findings were recorded here and in `ERRORS_LOG.md`.

---

## 8. Artifacts

`results/memscale_{base,ckpt,ckpt_bf16}.{csv,json}` (five ladder points each),
`results/w2_*.csv` and `results/w3_*.csv` (single-point probes at 730), each
with a provenance sidecar. Every row carries
`completed`/`oom`/`truncated`, and every fit records which rows it excluded and
why.
