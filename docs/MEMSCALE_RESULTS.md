# Measured: the full-trunk backward at L=730 fits, in bf16, on one A100-80GB

Run 2026-09-04 on `igv-gpu` (1x A100-SXM4-80GB, 79.2 GiB usable, us-central1-a),
at commit `808dd23` with the working tree dirty (every artifact records
`git_commit: 808dd23`, `git_dirty: true`).

---

## 0. Headline

`IGV_TRI_ATTN_CKPT=1` + `IGV_AUTOCAST=bf16` completes a full-trunk
forward+backward at **L=730 tokens in 55.22 GiB**, with 24 GiB of headroom, in
108 s. No stage had ever completed at 730 before.

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

### Background: why the backward pass is so large

Boltz-2's trunk contains 64 pairformer blocks. At L=730, the pair representation
`z` is shape (1, 730, 730, 128) in fp32, or **0.254 GiB per copy**. Each block's
triangle attention retains a softmax tensor for backward; across the full trunk,
these retained activations dominate memory. Per-block gradient checkpointing
(`gradient_checkpointing=True` in the Boltz config) trades compute for memory by
recomputing each block's forward during the backward pass, but the retained
softmax from the chunked attention survives checkpointing because Boltz's
`chunk_layer` stores every chunk's softmax for backward. `IGV_TRI_ATTN_CKPT`
wraps each triangle-attention chunk call in its own `torch.utils.checkpoint`,
which discards that softmax and recomputes it on demand. `IGV_AUTOCAST=bf16`
halves the size of every activation tensor. Together, these two settings bring
the requirement from 93.7 GiB down to 55.2 GiB.

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
apart. **A peak from a run that did not complete carries no
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

2. **A larger group does not become free under checkpointing.** Removing the
   retained softmax (via `IGV_TRI_ATTN_CKPT`) should have made bigger groups
   nearly free, since the checkpoint boundary tensors drop 8x (34.30 to 4.32 GiB
   at group 1 vs 8). It did not: going from group 4 to 16 in bf16 costs more
   than 23 GiB. **The dominant term in the group knob is the recompute
   transient** -- a bigger group means more blocks recomputed simultaneously in
   the backward -- not the retained boundary tensors.

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
floor, not orders of magnitude.

Group size barely touches the gradient: at 730 in bf16, group 1 gives 0.478486
and group 4 gives 0.476665, a 0.38% difference.

**This is necessary, not sufficient.** `grad_abs_max` is one scalar -- not the
score, not the gradient direction. The load-bearing validation is
`07_sanity`'s `completeness` and `frozen_vs_full` under the new dtype.

**Any score recorded under bf16 must be re-derived and never compared against
an fp32 one.** `numerics_arm()` now stamps the dtype into every stage's
provenance `arm` dict so the two cannot be silently mixed.

---

## 5. What changed in the code during this measurement session

Five defects surfaced and were fixed:

1. **Nine tests asserted a property of the HOST, not of the code.**
   `tests/test_gpu.py` and `tests/test_memscale.py` contained host-specific
   assertions (`assert gpu.has_cuda() is False`, `pytest.raises(ImportError):
   __import__("boltz")`). Green on a laptop, red on the GPU box. Fixed with a
   `cpu_only` fixture.

2. **`IGV_TRI_ATTN_CKPT` warned "Expect no memory change" on idempotent
   rebinds.** Added `count_tri_attn_chunk_ckpt(model)` to detect already-wrapped
   modules.

3. **Provenance could not distinguish bf16 from fp32 artifacts.** Added
   `numerics_arm()` (autocast, dtype, tri_attn_ckpt, pf_group_size, pf_chunk,
   chunk_profile, use_kernels) to stages 02/03/04/07.

4. **`04_scan.py` stamped `"geometry": "fixed_wt"` while coords were all-zeros.**
   Now `"fixed_from_featurisation"`, and `03_attribute.py` records the measured
   `coords_abs_max` in provenance.

5. **A PyTorch bug that disguises OOMs.** An OOM inside the checkpoint body can
   unwind through the saved-tensors hook stack and raise a misleading
   `INTERNAL ASSERT FAILED at SavedTensorHooks.cpp:69`. Not fixable here;
   recorded for awareness.

---

## 6. Recommended configuration

```bash
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16   # group size and chunk: leave at defaults
```

Also faster than baseline. At L=554, warm MSA cache: baseline 97.5 s
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
  `GATE FAILED` names only `completeness` -- the `informational` flag set only
  on the success path bug is genuinely fixed.
  Note it OOMs *even in bf16*, single-tenant, at 79.12 GiB in use: it is
  heavier than plain IG.
- **`dead_target` is no longer a tautology.** It reports a model-dependent
  `f(x)` alongside an exactly-zero gradient.

### Run-to-run noise is ~1.6% on the score

The two bf16 runs of `dead_target` gave `f(x) = 3.916786` and `3.855304` --
a **1.6% spread for an identical configuration**. So a single-pair comparison
against an fp32 run cannot establish a dtype shift; the noise is
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
  `signal_control`'s 30 mutants and the `complex_pde` reference.
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
starting a measurement run.

---

## 8. Artifacts

`results/memscale_{base,ckpt,ckpt_bf16}.{csv,json}` (five ladder points each),
`results/w2_*.csv` and `results/w3_*.csv` (single-point probes at 730), each
with a provenance sidecar. Every row carries
`completed`/`oom`/`truncated`, and every fit records which rows it excluded and
why.
