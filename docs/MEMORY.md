# Memory: why full-trunk attribution does not fit, and what we changed

Written 2026-09-04, at commit `808dd23` (working tree dirty — this session's
changes are uncommitted). Companion to `ERRORS_LOG.md` entry 12, which this
document corrects in two places and completes in one.

Read `ERRORS_LOG.md` entry 12 first. It is the measurement record. This is the
diagnosis, the literature survey, and the change log.

---

## 0. Summary

`ERRORS_LOG.md` entry 12 concluded that nine checkpointing configurations all
peak at ~78 GiB, that "the knobs work; they only relocate the memory," and that
the requirement is "broadly distributed, not concentrated in any one tunable
term." Three corrections:

1. **~78 GiB is not the requirement.** It is the allocator hitting the wall on
   runs that never finished — a truncated lower bound. The true requirement at
   L=730 is still unknown, and the literature suggests it is 2.5–4× the card.
2. **The chunk knob was wired to nothing.** boltz's `chunk_layer` retains every
   chunk's softmax for backward, so retained memory is chunk-size *invariant*.
   This fully explains commit `b1c29cc`. It is not a √N tradeoff landing at
   break-even; it is a knob with no effect on the retained total.
3. **There is a lever nobody had tried**: checkpoint each triangle-attention
   chunk, which is what the chunk knob was believed to do. Implemented as
   `IGV_TRI_ATTN_CKPT`, default off.

Also corrected: `use_kernels=True` cannot be enabled under `constraints.txt`
(it dispatches solely to cuequivariance; `trifast` is dead code in boltz 2.2.1),
and bf16 is bounded well below 2× because boltz forces the sequence track to
fp32.

Five real bugs were found and fixed along the way, one of which would have
crashed stage 02 on its first real GPU run.

---

## 1. The measurement error

`torch.cuda.max_memory_allocated()` reports the peak *reached*. On a run that
OOMs, that is how far the run got before the next allocation failed — not what
the run needed. Entry 12's own table proves the readings are truncated:

| site | group=1 | group=8 | Δ |
|---|---|---|---|
| `pairformer.py:102` boundary z | 34.30 | 4.32 | **−30.0** |
| `primitives.py:170 softmax_no_cast` | 11.59 | 25.22 | +13.6 |
| `transition.py:64` | 1.02 | 6.12 | +5.1 |
| `trunkv2.py:749` | 9.76 | 9.76 | 0 |
| `_msa_forward_checkpointed` | 3.25 | 3.25 | 0 |
| **attributed total** | **59.92** | **48.67** | **−11.3** |
| **reported "peak"** | 77.64 | 78.40 | +0.76 |

Nine configurations spanning group 1→8, chunk 16→128 and recycling 0→1 all
report peaks inside a 1.0 GiB band immediately below the 79.2 GiB wall.
Configurations whose attributed live bytes differ by 11 GiB cannot genuinely
peak within 0.76 GiB of each other. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is already set (`docker/Dockerfile:6`), which rules out fragmentation as the
explanation.

Consequence for entry 12's conclusion: the data cannot distinguish "grouping
gave no reduction" from "grouping reduced peak demand and the run simply
travelled further into the backward before hitting the same wall." The second
reading is the more natural one given the −11.3 GiB column. This is cheaply
testable — log which block index the backward reaches before OOM.

**Rule going forward: a peak from a run that did not complete is not evidence
of a requirement.** `scripts/08_memscale.py` enforces this structurally; every
row carries `completed` / `oom` / `truncated` and the fit uses only completed
rows.

---

## 2. What actually holds the memory

### 2.1 The chunk knob is inert (verified against pinned source)

`boltz-2.2.1/src/boltz/model/layers/triangular_attention/utils.py:322-375`
(`chunk_layer`) is a plain Python `for _ in range(no_chunks)` loop calling
`layer(**chunks)`. A grep of that file for `no_grad|detach|checkpoint|recompute`
returns **only a comment**. No `no_grad`, no `detach`, no recompute — so every
chunk's softmax output is saved for backward and the retained total does not
depend on `chunk_size`. Chunking is marginally *worse*: it additionally
allocates the `out` buffer via `t.new_zeros`.

The arithmetic confirms it exactly. At L=730, H=4, fp32:

```
730 * 4 * 730 * 730 * 4 bytes = 5.7968 GiB   per triangle attention
                              x2 per PairformerLayer
                              = 11.5936 GiB
```

`ERRORS_LOG.md` entry 12 measured **11.59 GiB** at `primitives.py:170` for
group=1. The match to three digits only occurs if all chunks are retained.

`IGV_PF_CHUNK` remains useful as a *transient and fragmentation* control only —
1.0164 GiB per chunk at 128 versus 0.1271 GiB at 16, and 1.02 GiB is exactly the
failing allocation in every recorded OOM. The comment at
`src/igv/boltz_score.py:559-564` asserting that halving the chunk halves the
term was wrong and has been corrected in place.

### 2.2 bf16 is bounded, not 2×

- `boltz pairformer.py:104-110` opens `with torch.autocast("cuda", enabled=False):`
  and calls `s.float()`, `z.float()`, `mask.float()`. **The sequence track of
  all 64 trunk layers stays fp32, and each layer takes a full fp32 copy of `z`
  (0.254 GiB at L=730) as attention bias.**
- `primitives.py:159-172 softmax_no_cast` *does* help: `:168` is the bf16 branch
  (inside `autocast(enabled=False)`), `:170` the fp32 branch. Entry 12's top
  site is `:170`. Under bf16 the allocation moves to `:168` and halves,
  11.59 → ~5.80 GiB per layer at group=1.
- `pairformer.py:102` (`z = z + self.transition_z(z)`, entry 12's largest term)
  *should* halve — `transition.py:32-35` is LayerNorm plus three bias-free
  Linears, so the output is bf16 under autocast. **Confirm by printing
  `z_.dtype` at the group boundary rather than assuming it.**
- `primitives.py:189` does in-place `a += b` with an fp32 `mask_bias`. This
  silently downcasts rather than raising, and `-1e9` in bf16 is `-998244352.0`
  — still overwhelmingly negative, so masking holds.

### 2.3 `use_kernels=True` is blocked by policy, not by capability

`boltz/model/layers/triangular_attention/primitives.py:199-202`:

```python
@torch.compiler.disable
def kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    from cuequivariance_torch.primitives.triangle import triangle_attention
    return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)
```

Single dispatch target. `triangular_mult.py:22` has a second, independent
cuequivariance import.

**`trifast` is dead code in boltz 2.2.1.** `grep -rn trifast` over the pinned
tree returns exactly three hits: `docs/prediction.md:165` (a stale CLI doc
string), `primitives.py:369` (`def _trifast_attn`) and `primitives.py:401` (its
lazy import). No caller anywhere; it is in no dependency list.

The path **is differentiable** — `grep -n 'no_grad|detach|inference_mode'` over
`triangular_attention/{primitives,attention,utils}.py` returns only
`primitives.py:77,80`, both inside a `Linear.__init__` weight initialiser. So
the only blocker is `constraints.txt:7-10`, which forbids cuequivariance
because it silently replaces the pinned `torch==2.7.1+cu126` wheel. boltz
declares it as an unbounded optional extra (`pyproject.toml:43-47`, `>=0.5.0`,
no upper bound), so `pip install boltz[cuda]` resolves to latest.

Enabling kernels would also silence chunking entirely —
`attention.py:157` is `if chunk_size is not None and not use_kernels:`.

**Lifting the ban is a user policy decision.** It was not taken as part of this
work. If revisited, it needs re-verification on the VM and a pin bounded on
both ends.

---

## 3. How large is the gap?

Unknown, and that is the point. Three calibration points from the literature,
all pointing the same way:

- **MegaFold** — the best published AF3-style training system — reaches L=768 on
  an H200 (141 GB), and is "the only system capable of training on sequence
  lengths of 640 and 768," extending feasibility from 512. We are attempting
  L=730 on 79.2 GB: 56% of the memory, same problem size, without their kernels.
- **MegaFold's scaling data**: EvoAttention activations of 3.75 GB at L=96
  rising to 24.61 GB at L=192 — 6.56× for a 2× length, i.e. an exponent of
  **≈2.71**, between quadratic and cubic. At that slope, 192 → 730 is a **~37×**
  multiplier.
- **AF3 itself trains at a 384-token crop**, fine-tuning at 640 and 768. A full
  backward at 730 tokens is at the outer edge of what anyone does with a cluster.
- **The predecessor repo: L≈540 exceeded 80 GB on H100, ran on H200 141 GB.**
  Consistent with MegaFold's 512→768 boundary, and it brackets the L=540
  requirement between 80 and 141 GB. Scaling 540→730 at exponent 2.7 multiplies
  by ~2.3, putting L=730 somewhere near **180–320 GB**.

Working estimate: short by **2.5–4×**, not a rounding error. Percentage-level
tuning is therefore irrelevant; only multiplicative levers count.

Checkpointing changes the constant and the depth term, not the exponent in L.
So the exponent from the literature is the right one to extrapolate with, and
the constant must come from our own completed runs — which is exactly what
`scripts/08_memscale.py` produces.

---

## 4. Literature survey

### 4.1 Multiplicative levers (system-level, keeps full-trunk IG as specified)

| Lever | Multiplier | Notes |
|---|---|---|
| **DS4Sci_EvoformerAttention** | **13× peak** | CUTLASS fused kernels for the AF attention variants; deployed in OpenFold at that reduction without accuracy loss. fp16/bf16 only, CC 7.0+, JIT-compiled, needs CUTLASS. `pip install deepspeed` declares torch — install `--no-deps` and re-run the verification table. |
| **MegaFold EvoFlash-3D** | large | Purpose-built for *training* memory of AF3 3D attention; two-kernel backward vs trifast's three. |
| **Per-chunk triangle-attention checkpointing** | see §2.1 | No new dependency, no dtype change. **Implemented this session.** |
| bf16 | <2×, see §2.2 | Bounded by boltz's fp32 sequence track. |
| GACT / ActNN / COAT | 8× on reachable tensors | Reaches the recompute transients (they go through `save_for_backward`), not the closure-held boundaries. |
| Sequence parallelism over the pair axis (MegaFold EvoSP-3D, FastFold DAP) | ×N GPUs | The only lever with no accuracy question attached. Real implementation project, not a flag. |
| LMI4Boltz | +66.7% token limit | Boltz-2-specific. In-place pair updates are inference-only; the host-offload, functional-scope and chunking items are backward-compatible. Offloads rel-pos enc, `z_init`, pdistogram, MSA skips — plausibly entry 12's constant 9.76 GiB `trunkv2.py:749` term. |

Note DS4Sci's 13× is end-to-end against a baseline without aggressive chunking.
Our `chunk=128` has already harvested ~5.7× of the cubic term
(5.80 → 1.02 GiB per chunk); the optimizations overlap, so do not budget 13× on
top of what we have.

### 4.2 A different method: lower memory by construction

**Sketched Jacobian via forward-mode AD.** Forward-mode computes a JVP in one
forward pass with constant memory — no graph, no activations, **no dependence on
the 64 blocks**. The objection is one JVP per direction, and we have ~13k
mutations. The escape: if the mutation-effect Jacobian is approximately
low-rank (the same premise that makes DMS landscapes learnable from sparse
measurements), then `k` random tangents plus randomized SVD recover a rank-`k`
approximation of the whole input→score Jacobian. Cost: `k` forward passes at
forward-pass memory (~5–10 GiB) versus one backward at >>80 GiB.

For the paper this is stronger than a fallback: it preserves the full-trunk
claim (nothing is frozen), and the rank-`k` truncation error is measurable
against the existing T1 term. The `completeness` check is the natural
validation — a rank-`k` sketch has a quantifiable completeness gap.

**Randomized AD** (Oktay et al., ICLR 2021) gives *unbiased* gradients at
reduced memory by randomly collapsing the linearized graph — trading variance
for memory without extra forward compute, complementary to checkpointing. Real
caveat from the paper: for some graph structures variance grows exponentially
with depth. 64 blocks plus recycling is deep. Pilot at small L before
committing.

**Compressed-sensing ISM (Yuzu)** — O(n) → O(1) forward passes, 247× speedup,
>0.99 Pearson, no gradients at all. Does **not** transfer: its sparsity
assumption is convolutional receptive fields ("changes at one input position
can't affect outputs beyond that field"), and a pairformer's triangle updates
give every position global reach after one block. What transfers is the
strategy — probe with random superpositions of mutations, recover by sparse or
low-rank regression, forward passes only. That is the sketched Jacobian above
with a different sensing matrix.

### 4.3 Cheap in compute, not in memory — recorded so it is not re-derived

**AttnLRP** (ICML 2024) is implemented as ε-LRP in Input×Gradient form, "a
single chain of Jacobian-vector products (one backward pass)" — so peak memory
≈ current peak. Same for **DeepLIFT**, **AtP\***, **EAP-IG**. **Spectral IG**
reorders the integration path by SVD of the baseline→input difference for
cleaner maps; the coarse-to-fine claim is about activation *order*, not
compression, and it reports no memory benefit.

Step-count reductions (Gauss-Legendre, Guided IG, IG²) do not help at all:
`src/igv/attrib.py:235-253` is already a strictly sequential loop, one leaf and
one backward per node, so peak equals a single backward and `m_steps` buys only
wall-clock. Batching over alphas would *multiply* peak, not reduce it.

---

## 5. What was implemented

Nothing committed. `git diff --stat`: +1847 / −192 across 9 tracked files, plus
6 new untracked files. Test suite **51 → 235 passing** in 4.2 s.

### 5.1 New knobs — every default reproduces today's behaviour

| Env var | Default | Effect when set |
|---|---|---|
| `IGV_TRI_ATTN_CKPT` | `0` | Rebinds every `TriangleAttention._chunk` so each chunk's `mha` runs under a non-reentrant checkpoint — softmax recomputed in backward instead of retained. Requires `gradient_checkpointing=True`; warns and no-ops otherwise. |
| `IGV_AUTOCAST` | `off` | `bf16`/`bfloat16` wraps **only** the trunk+confidence dispatch. **Raises** on `fp16`/`float16`/`half` — there is no `GradScaler` anywhere in the repo and fp16's 5-bit exponent breaks the guards at `boltz_score.py:745-757`. |
| `IGV_CHUNK_PROFILE` | `auto` | `large`/`small` force one side of boltz's `chunk_size_threshold=384` algorithm switch. Illegal values raise. |
| `IGV_USE_KERNELS` | `0` | Single switch for all live `use_kernels` sites. Eagerly raises if `cuequivariance_torch` is unimportable. Buys zero memory today (§2.3); exists for hygiene and future use. |
| `IGV_TRI_ATTN_KERNEL` | `0` | Narrow variant: `use_cuequiv_attn` only, at the two direct-`PairformerLayer` call sites. |
| `IGV_MSA_SPEC` | unset | Emits a per-chain `msa` key in the YAML. `empty` = boltz single-sequence mode. Enables offline sweeps. |
| `IGV_ASSERT_FEAT_SEQ` | **`1` (on)** | Decodes `feats["res_type"]` and raises if featurised residues differ from the requested chains. `0` downgrades to `log.error`. |

`IGV_CHUNK_PROFILE=auto` was verified by differential test against the old
inline selection over `n_tokens` in {1,100,383,384,385,500,730,1030} ×
`IGV_PF_CHUNK` in {unset,64,256} — all six contract keys, **zero drift**.

**One default drifted during implementation and was caught in integration.**
`resolve_use_kernels` initially fell back to `getattr(model, "use_kernels", False)`,
read from the checkpoint's saved hyperparameters. Had that flag ever been
`True`, an operator who set nothing would have silently gotten the
cuequivariance path — a different algorithm, with chunking disabled — and then a
hard crash blaming an env var they never set. Fixed to a literal `False`, with a
`log.warning` so the model's own flag stays visible without being obeyed.

**`IGV_ASSERT_FEAT_SEQ` is the one place the preservation rule was deliberately
bent.** It is read-only, so a correct run stays numerically bit-identical; it
can only raise where today's run silently proceeds on stale features. But its
`res_type` one-hot decode has never met real boltz features. Run one wild-type
4fqi featurisation before trusting it; escape hatch is `IGV_ASSERT_FEAT_SEQ=0`.

### 5.2 New files

- **`src/igv/gpu.py`** — the repo's first peak-memory instrumentation.
  `require_vram()` (single source for the gate, previously copy-pasted four
  times), and `PeakMemory` / `measure_peak` recording `peak_gib`,
  `reserved_gib`, `completed`, `oom`, `error`. Synchronizes and resets the
  process-global peak counters before each body. Import-safe with no torch and
  no CUDA. 37 CPU-only tests.
- **`scripts/08_memscale.py`** — sweeps L, runs the backward to completion,
  records true peak, fits `log(peak)` vs `log(L)`. Writes `fit=null` +
  `fit_error` and exits 1 until it has ≥3 rows with `truncated=False`. Forces
  one chunk profile across the sweep (fitting across the L=384 cliff mixes two
  algorithms into one curve). Gives each sweep point its own boltz `cache_dir`.
  `--dry-run` works on a laptop with no GPU and no boltz. **Deliberately not in
  `Makefile`'s `all:` chain or `run_all.sh`** — it is a diagnostic, and a
  deliberately-OOMing sweep in the default path would break every full run.
- `tests/test_gpu.py`, `tests/test_memscale.py`, `tests/test_sanity.py`,
  `tests/test_scripts_vram_hoist.py`.

### 5.3 Bugs found and fixed

1. **`scripts/02_embed_deltas.py:38` read `.total_mem`, which does not exist**
   (the attribute is `.total_memory`), and the enclosing `try` caught only
   `ImportError`. Stage 02 would have died with an unhandled `AttributeError`
   the first time it ran on a real GPU — meaning **this stage has never once
   passed its own VRAM gate.** Fixed by the hoist to `igv.gpu.require_vram`.
   Note the consequence: stage 02 will now *enforce* 78 GiB for the first time.
   If the embedder-only footprint does not need that, pass a lower `min_gib` at
   the call site rather than reverting the hoist.
2. **`dead_target` was a tautology.** `scripts/07_sanity.py:167-169` attributed
   `lambda x: (x * 0.0).sum() + 1.0`, whose gradient w.r.t. `x` is analytically
   and unconditionally zero, and `check_dead_target(s_inputs, baseline)` had
   neither `model` nor `forward_fn` in scope — so no Boltz code path could
   influence the result. It PASSED with `value: 0.0` in the very run where all
   four real gradient checks OOM'd. Replaced with `igv.attrib.make_dead_target`,
   which runs the real forward under `no_grad` on a detached input and connects
   only a zero-weighted term: the value is model-dependent (real evidence) while
   the gradient is still exactly zero and the backward traverses only the tiny
   connector, so it costs one no-grad forward and cannot OOM.
3. **`frozen_vs_full`'s `informational` flag was set only on the success path.**
   The exception handler called `_result(...)`, which never sets it, and the
   blocking filter is `not r['passed'] and not r.get('informational')`. So an
   OOM in a check documented as non-blocking was recorded as **blocking** —
   which is what happened in `results/sanity_4fqi_h1_complex_pde.json`.
4. **`check_random_weights` zeroed LayerNorm scales.** It zeroed 1-D parameters
   wholesale, annihilating layer outputs, making the gradient constant, making
   `spearman` return NaN (`src/igv/metrics.py:22-30` returns NaN on zero
   variance), and making `abs(nan) < 0.3` False — so the gate printed its
   "THIS KILLS THE PROJECT AS FRAMED" message about a network the check itself
   broke. Now zeroes biases by name, sets norm scales to ones, and reports an
   ERROR rather than the project-killing text on non-finite rho.
5. **`scripts/03_attribute.py:289` had no `.float()`** before `.numpy()`, which
   would raise `TypeError: Got unsupported ScalarType BFloat16` the moment
   `IGV_AUTOCAST` was used. Guarded.

Also corrected: the false chunk comment (§2.1); docstrings claiming the 4fqi
complex is 753 tokens (the repo's own `read_pdb_chains` on
`data/raw/4fqi_hlab.pdb` gives A=324, B=176, H=121, L=109 = **730**, matching
`n_tokens` everywhere else); and `DEFAULT_OFFLOAD_MIN_BYTES` docstring figures
(true values 260.2 MiB at L=730 and 122.1 MiB at L=500, ~2.4% off as written).
`ERRORS_LOG.md` was left untouched — it is an append-only historical record.

### 5.4 Verification

- 235 tests pass. `python -c "import igv.gpu, igv.attrib, igv.boltz_score, ..."`
  clean with boltz absent. `py_compile` clean on all changed files.
  `08_memscale.py --dry-run`, `04_scan.py --dry-run`, `07_sanity.py --dry-run`
  all exit 0. ruff: 13 findings, byte-identical to the set at `HEAD` — zero lint
  regression.
- Three review lenses (default-preservation, correctness-and-honesty,
  spec-adherence) produced three high-severity findings; **all three were
  refuted** under adversarial verification. Zero confirmed defects.
- `IGV_TRI_ATTN_CKPT` was verified against the pinned boltz source directly
  rather than on the builder's word: the wrapper reproduces upstream
  `_chunk(self, x, tri_bias, mask_bias, mask, chunk_size, use_kernels=False)`
  and its exact five-key `mha_inputs` dict, and `Attention.forward` does take
  `use_kernels` sixth positionally.

---

## 6. Not established — do not assume

- **No memory requirement is established by any of this.** The ~78 GiB figures
  remain truncated lower bounds. Only an `08_memscale` row with
  `truncated=False` is evidence.
- **`PairformerLayer`'s trailing positional arg order** at
  `boltz_score.py:1107`/`:1405` is assumed to be
  `(use_kernels, use_cuequiv_mul, use_cuequiv_attn)`. Harmless at defaults (all
  `False`), but confirm by signature inspection before trusting
  `IGV_TRI_ATTN_KERNEL`.
- **Whether bf16 halves the boundary `z`** (§2.2). Print `z_.dtype` at the group
  checkpoint boundary.
- **Model parameters are never frozen.** No `zero_grad`, no
  `requires_grad_(False)` anywhere in `src/igv/` or `scripts/`, so a full fp32
  gradient copy of Boltz-2's weights is allocated on the first backward and
  survives `free_cuda_memory()` for the whole IG loop. A single
  `for p in model.parameters(): p.requires_grad_(False)` before attribution
  would eliminate it and shrink the graph, since `s_inputs` is the only tensor
  whose gradient is wanted. boltz's `boltz2.py:350-357` already freezes
  non-confidence params when `structure_prediction_training` is false, so this
  may be partly moot — **measure
  `sum(p.numel() for p in model.parameters() if p.requires_grad)` on the VM
  before implementing anything.** Not changed this session.
- **The "GEOMETRY IS FIXED" claim is unsupported.** `structure_pdb` is accepted
  by `build_complex_feats` and **never read** (`grep -n structure_pdb` returns
  only `:214` and `:226`); features come solely from a sequences-only YAML, and
  boltz's schema parser sets every parsed atom's coords to (0,0,0) for
  sequence-only input (`boltz/data/parse/schema.py:733, :880`). The log lines at
  `03_attribute.py:222-231` and `04_scan.py:12-18` assert otherwise. Instrument
  `feats['coords'].abs().max()` and settle it before writing anything that
  depends on it. Not changed this session.
- **The boltz stale-cache hazard is real but not proven to have fired.**
  `process_inputs` skips any input whose YAML stem is already processed
  (`boltz/main.py:724-742`) and this repo always writes the stem `input`
  (`boltz_score.py:255`). `07_sanity.py:336-339` reuses one directory for all 30
  `signal_control` mutants. Against that, the recorded run shows 30 *distinct*
  scores (std 1.49e-2), which identical features could not produce. Unresolved;
  `IGV_ASSERT_FEAT_SEQ` makes the code verify rather than assume.
- **`scripts/03_attribute.py:46-91` keeps its own duplicated
  `_confidence_forward_out_dict`** with hardcoded `use_kernels=False` and a
  hardcoded recycling loop. It is only reached at `:286` for the ipTM argmax
  under `no_grad`, gated on `score in IPTM_SCORES` — so it does **not** affect
  the attribution gradient path (`forward_fn` at `:214-218` goes through
  `confidence_forward`, and all new knobs apply there). Worth collapsing for
  consistency; `resolve_use_kernels` is importable for whoever does.

---

## 7. Recommended next steps on the VM

1. `make memscale` at defaults → baseline curve and a measured exponent.
2. Re-run with `IGV_TRI_ATTN_CKPT=1` → the largest dependency-free lever.
3. Add `IGV_AUTOCAST=bf16` → three measured curves and a real extrapolation to
   L=730 instead of a guess.

Before step 1: run one wild-type 4fqi featurisation to exercise
`IGV_ASSERT_FEAT_SEQ`, confirm the `PairformerLayer` arg order, and measure the
trainable-parameter count.

Any score recorded under `IGV_AUTOCAST` must be re-derived, never compared
against an fp32 score. Entry 12 records `recycling_steps=0` moving `complex_pde`
3.925568 → 4.309255 and treats that as disqualifying; a dtype change is at least
as large a perturbation. The `IGV_AUTOCAST` value belongs in the provenance
`arm` dict so no artifact can be mistaken for an fp32 one.

If steps 1–3 leave us short: **the H200/B200 quota was declined**, but quota is
a GCP problem, not a hardware-availability problem. Lambda, RunPod, Vast.ai and
CoreWeave rent H200/B200 by the hour with no quota process, and the pipeline is
containerized under `docker/`, so it is portable. The efficiency claim needs
~16 attribution runs total — hours, not a reservation. `2× A100-80GB`
(`a2-ultragpu-2g`) is often an easier ask than any H200 and 158 GB aggregate
exceeds one, but using it needs pair-axis sequence parallelism, which is an
implementation project.

---

## 8. Sources

Boltz / AF3 systems work:
- LMI4Boltz — https://www.biorxiv.org/content/10.1101/2025.10.29.684571v2.full · https://github.com/tlitfin/lmi4boltz
- MegaFold — https://arxiv.org/abs/2506.20686 · https://github.com/Supercomputing-System-AI-Lab/MegaFold/
- DS4Sci_EvoformerAttention — https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/ · https://arxiv.org/pdf/2310.04610
- ScaleFold — https://arxiv.org/pdf/2404.11068 · FastFold — https://arxiv.org/pdf/2203.00854
- Flash IPA — https://arxiv.org/html/2505.11580v1
- Triangle Multiplication is All You Need — https://arxiv.org/html/2510.18870v1
- Memory-efficient transformer training in AI for Science (survey) — https://arxiv.org/pdf/2501.11847
- AutoChunk — https://arxiv.org/pdf/2401.10652

Rematerialization / compression:
- PyTorch activation checkpointing techniques — https://pytorch.org/blog/activation-checkpointing-techniques/ · https://github.com/pytorch/pytorch/issues/149258
- Reducing Activation Recomputation in Large Transformers — https://arxiv.org/pdf/2205.05198
- ActNN — https://arxiv.org/pdf/2104.14129 · GACT — https://proceedings.mlr.press/v162/liu22v/liu22v.pdf · COAT — https://arxiv.org/html/2410.19313v2
- RevNet — https://arxiv.org/abs/1707.04585

Attribution methods:
- Randomized Automatic Differentiation — https://arxiv.org/abs/2007.10412
- Gradients without Backpropagation — https://arxiv.org/pdf/2202.08587 · Scalable Backprop-Free Gradient Estimation — https://arxiv.org/html/2511.03110
- Equivalent Linear Mappings of LLMs — https://arxiv.org/pdf/2505.24293
- Yuzu (compressed-sensing ISM) — https://academic.oup.com/bioinformatics/article/38/14/3557/6604724 · https://github.com/kundajelab/yuzu
- AttnLRP — https://proceedings.mlr.press/v235/achtibat24a.html · https://github.com/rachtibat/LRP-eXplains-Transformers
- AtP* — https://arxiv.org/pdf/2403.00745 · Spectral IG — https://arxiv.org/pdf/2605.19607
