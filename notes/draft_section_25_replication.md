# Draft section 25: replicating the faithfulness number on the three remaining protein-protein complexes

Run 2026-10-13/14 on the A100 (`igv-gpu`, a2-ultragpu-1g, us-central1-a).

Section 24 measured, on 3S9D chain A, a faithfulness Spearman of **0.5937**
between gradient attribution on `complex_pde` and the model's own `resswap`
scan, and said in its own first caveat that the number must not be quoted until
1JTG, 2JEL and 3HFM had run the same pipeline. This section runs them.

<!-- HEADLINE VERDICT: fill from results -->

## 0. The thing that stopped the last three attempts, identified

Section 24 recorded that "an external watchdog stopped the VM three times" on
2026-10-13 and left it unexplained. **It is not external and it is not a
watchdog.** It is a property of the instance itself:

```
scheduling:
  maxRunDuration: {seconds: '7200'}
  instanceTerminationAction: STOP
```

`gcloud compute operations list` shows the three stops as
`compute.instances.deferredStop`, issued by `system`, with the message
"Instance stopped by Compute Engine" -- and each one lands **exactly two hours
after its own start**, to within three seconds:

| start (UTC) | stop (UTC) | delta |
|---|---|---|
| 2026-10-13 18:10:05.945 | 2026-10-13 20:10:08.331 | 2 h 00 m 02.4 s |
| 2026-10-13 20:41:23.995 | 2026-10-13 22:41:25.663 | 2 h 00 m 01.7 s |
| 2026-10-13 22:43:57.079 | 2026-10-14 00:43:58.809 | 2 h 00 m 01.7 s |

(Section 24 recorded the three stops as "20:10, 22:42, ~23:25 UTC". The first
two match the operations log exactly; the third was actually 00:43 UTC on
2026-10-14. The operations log is authoritative and is what is quoted here.)

**It is not the budget.** The billing account carries one budget,
"agrosbio all-time 500" ($500, `notificationsRule: {}`), which has no Pub/Sub
topic attached and therefore cannot stop anything -- a GCP budget alert emails,
it does not act. There is no instance-schedule resource policy on the project
(`gcloud compute resource-policies list` returns zero items). The cap is
self-imposed on the VM and nothing else in the project enforces anything.

Two further facts worth recording: a second principal, `as22836@nyu.edu`, has
compute admin on `agrosbio` and issued `start`/`stop`/`setMetadata` calls on
this instance at 05:14-05:24 PDT on 2026-10-13, so **this is a shared instance**;
and `onHostMaintenance: TERMINATE` with `automaticRestart: false` means a
Compute Engine maintenance event -- one was announced on 2026-10-12 -- also
kills a run with no restart.

### The cap is load-bearing, not cosmetic, and it bounds this section

Raising `maxRunDuration` was attempted and **refused by the permission layer**
as a shared-resource modification. That refusal is respected here, and it is the
single reason parts of the brief below are reported as not-run rather than as
results. Section 7 states exactly what it cost.

## 1. Cost model, measured rather than assumed

The replication was costed from 3S9D and the estimate was wrong, in the
expensive direction. The driver of cost is token count, and **the two complexes
section 14 treated as cheap are the two largest**:

| complex | tokens | vs 3S9D | SKEMPI positions | mutants (19/pos) |
|---|---|---|---|---|
| 3S9D A127_B180 | 307 | 1.00x | 50 | 950 |
| 1JTG A262_B165 | 427 | 1.39x | 28 | 532 |
| 2JEL H218_L217_P85 | **520** | 1.69x | 32 | 608 |
| 3HFM H215_L214_Y129 | **558** | 1.82x | 15 | 285 |

Position count and cost are unrelated: 3HFM has the fewest positions (15) and
the most tokens (558).

Measured IG step time: **50.75 s/step on 1JTG** (427 tokens, timed over steps
1-5), against ~20 s/step on 3S9D (307 tokens). That is 2.54x the time for 1.39x
the tokens -- an exponent near 2.8, which is quadratic attention plus the
`pf_chunk` "large" profile that switches on above 384 tokens. An IG run at m
costs 2m steps (one integral through the input embedder, one through
`res_type`).

| job | 1JTG | 2JEL | 3HFM |
|---|---|---|---|
| IG m=32 (64 steps) | 54 min | ~95 min | ~115 min |
| IG m=64 (128 steps) | ~108 min | **~189 min** | **~230 min** |
| resswap scan | ~81 min | ~137 min | ~74 min |
| refeaturise | ~20 min | ~30 min | ~15 min |

**The usable window is ~113 min** and stage 03 is not resumable. So IG m=64 is
arithmetically impossible for 2JEL and 3HFM, and IG m=32 is impossible for
3HFM. The scans resume cleanly and can span windows, so they are unaffected.

<!-- sections 2-7 to be filled from results -->

## 1b. Peak VRAM: section 24's housekeeping conclusion is wrong

Section 24 measured 20.2 GiB on 3S9D and concluded "the 80 GB card is not needed
for this arm". Measured here:

| job | complex | tokens | peak VRAM | samples |
|---|---|---|---|---|
| IG m=32 | 1JTG | 427 | **43608 MiB (42.6 GiB)** | 332 |
| plain_grad | 2JEL | 520 | **62708 MiB (61.2 GiB)** | 25 |
| plain_grad | 3HFM | 558 | **71828 MiB (70.1 GiB)** | 28 |

**The 80 GB card is required.** 3HFM's single backward pass already sits at 70.1
GiB of 80, about 10 GiB of headroom, and 3HFM is the *cheapest* job on the
largest complex. A 24 GB card fits none of these three, and a 40 GB card fits
none of them either. Section 24's figure was not wrong, it was measured on the
smallest complex in the set (307 tokens) and does not generalise -- the same
mistake in kind as quoting 3S9D's 0.5937 as "the" faithfulness number.

**Two honest qualifications on these numbers.** They are `nvidia-smi
memory.used`, which reports memory *reserved* by the driver -- CUDA context plus
the PyTorch caching allocator's pool -- not live tensor allocation. True
allocation is lower by an unmeasured amount, and `torch.cuda.max_memory_allocated`
was not logged. So these are an upper bound on what is needed, and the right
reading is "an 80 GB card is sufficient and a 24 GB card is not", not "70 GiB of
tensors exist". Second, they are sampled at 10 s intervals, so the 2JEL and 3HFM
figures rest on only 25-28 samples across a ~4 min job and may *understate* a
brief true peak.

## 2. Completeness

### The m=32 rung for 2JEL and 3HFM is reused, and the reuse check passed

The condition set out above -- that `complex_pde` is unchanged between 0bf19ef
and 420a456 -- was tested by running `plain_grad` fresh at 420a456 and comparing
its `f(x)` against the `f_x` stored in the section-14 artifact:

| complex | stored f(x) @ 0bf19ef | fresh f(x) @ 420a456 | agree |
|---|---|---|---|
| 2JEL P | 1.4196462631225586 | 1.419646 | **yes, to all 6 printed digits** |
| 3HFM Y | 0.9989417791366577 | 0.998942 | **yes, to all 6 printed digits** |

**The score function did not move**, so the section-14 m=32 attributions are
measuring the same function the 420a456 scans measure, and they are used as the
m=32 rung. A third, independent instance of the same check: 1JTG's resswap scan
wrote its wild-type row as **1.4329572916030884**, against stage 03's
`f(x) = 1.432957` for 1JTG -- scan and attribution provably on the same
geometry and featurisation, exactly as section 24 verified on 3S9D.

<!-- completeness ladder table to be filled when m=64 lands -->

## 2c. Revised cost model, from measured rates

The §1 estimates were priors; these are measurements.

| measurement | value |
|---|---|
| 1JTG IG, 64 steps (m=32) | **3322 s** = 51.9 s/step |
| 1JTG plain_grad | 116 s |
| 2JEL plain_grad | **245 s** (2.11x 1JTG) |
| 3HFM plain_grad | **271 s** (2.34x 1JTG) |
| 1JTG resswap scan | **11.0 s/mutant** |

1JTG IG m=32 took 3322 s against a 1700 s estimate -- **1.95x over**. Scaling IG
step cost by the measured plain_grad ratios:

| job | 1JTG | 2JEL | 3HFM |
|---|---|---|---|
| IG m=32 | 55 min (done) | ~117 min (reused) | ~130 min (reused) |
| IG m=64 | **~111 min** | **~234 min** | **~259 min** |
| resswap scan | ~57 min (313 left) | ~164 min | ~88 min |
| refeaturise | ~20 min | ~35 min | ~18 min |

**The full brief is ~16.4 h of remaining GPU against 11.3 h in the window, so it
still does not all fit** -- the binding constraint moved from per-job (the 2 h
cap) to total queue size. The queue is therefore ordered headline-first:
all three resswap+refeaturise scans and their faithfulness runs (~6.4 h, the
four-complex table), then 1JTG IG m=64 (~1.9 h, the only m=64 that fits).
2JEL and 3HFM IG m=64 are queued last and will `DEFER` by design.

## 3. Faithfulness against the model's own scan (`resswap`) -- the headline

**1JTG is in, and it replicates.** 2JEL and 3HFM are still running; this section
is written with one of three landed and will be completed when they are.

| | 3S9D A (IG m=64) | **1JTG B (IG m=32)** | 3S9D A (plain_grad) | **1JTG B (plain_grad)** |
|---|---|---|---|---|
| n | 950 | **532** | 950 | **532** |
| **Spearman** | **0.5937** | **0.6326** | 0.5885 | **0.5449** |
| CI | [0.5404, 0.6399] | **[0.5692, 0.6911]** | [0.5350, 0.6379] | [0.4741, 0.6154] |
| permutation null p95 | 0.0534 | **0.0720** | 0.0527 | 0.0704 |
| **clears own null** | **yes** | **yes** | **yes** | **yes** |
| partial, distance to partner | 0.5722 | **0.6317** | 0.5653 | 0.5502 |
| partial, all 8 confounds | 0.5920 | **0.5916** | 0.5557 | 0.5755 |
| scale ratio | 0.7484 | 0.7211 | 0.4380 | 0.5430 |
| per-position, sum | 0.5546 | 0.4986 | 0.6642 | 0.5068 |

Three things are worth saying plainly.

**1JTG is higher than 3S9D, not lower.** 0.6326 against 0.5937. The replication
was expected to fall back toward noise -- section 14's between-complex sd of
0.33 was the prior, and the prediction registered before these runs said
explicitly to expect a mean of 0.4-0.5 with sd above 0.15. On the two complexes
measured with the *same* method (`plain_grad`, which is rung-free and so
perfectly comparable), the numbers are **0.5885 and 0.5449: mean 0.5667, sd
0.0308**. That is an order of magnitude tighter than section 14's hotspot arm.

**The geometry control is, if anything, cleaner than on 3S9D.** Controlling for
minimum heavy-atom distance to the partner chain moves 1JTG's IG number from
0.6326 to **0.6317** -- a drop of 0.0009, essentially nothing, against 3S9D's
drop of 0.02. Controlling for all eight confounds at once leaves it at 0.5916,
which is 3S9D's 0.5920 to three decimals. **This is not geometry**, and that now
holds on two complexes rather than one.

**Both clear their own permutation null by a wide margin**, 0.63 and 0.54
against nulls of ±0.07. Contrast section 14, where 1 of 8 position-level
measurements cleared its null -- the difference is that this is a mutation-level
test at n=532 rather than a position-level test at n=28.

**Caveat on the comparison**: 1JTG's IG column is m=32 and 3S9D's is m=64. On
3S9D the two rungs agreed at Spearman 0.9990, which is the licence for treating
them as interchangeable, but it is one complex's worth of evidence. The
`plain_grad` columns carry no such caveat and are the honest head-to-head.

## 3b. IG vs plain_grad, again

On 3S9D the two methods were within 0.005 of each other. On 1JTG they are not:
IG m=32 **0.6326** against plain_grad **0.5449**, a gap of 0.088, with the CIs
overlapping but not nested. IG also beats plain_grad on scale calibration on
both complexes (0.72 vs 0.54 here, 0.75 vs 0.44 on 3S9D). So the section-24
statement "the cheap single backward pass is as good a ranker as the 64-step
integral" is **true on 3S9D and not clearly true on 1JTG**. Two complexes is not
enough to call it; recorded so the third and fourth can settle it.

## 4. The real mutant (`refeaturise`)

### The null replicates

| | 3S9D (IG m=64) | **1JTG (IG m=32)** | 3S9D (plain_grad) | **1JTG (plain_grad)** |
|---|---|---|---|---|
| n | 103 | **32** | 103 | **32** |
| Spearman | -0.0123 | **0.2053** | -0.0783 | **-0.2496** |
| CI | [-0.2269, 0.1931] | [-0.1771, 0.5394] | [-0.2789, 0.1432] | [-0.5401, 0.0772] |
| null p95 | 0.1613 | 0.2984 | 0.1593 | 0.2955 |
| **clears own null** | **no** | **no** | **no** | **no** |

**Four of four null.** 1JTG's two point estimates straddle zero in opposite
directions (+0.21 and -0.25), neither clearing a null of ±0.30 at n=32. Note how
wide that null is -- 32 surviving mutations buys very little power, so this is
"no detectable signal", not "signal shown absent".

### The skip rate is consistent, and slightly worse

| | 3S9D | **1JTG** |
|---|---|---|
| mutants attempted | 950 | **532** |
| `TOKEN_TO_REP_ATOM CONTENT DIFFERS` | 847 (89.2%) | **498 (93.6%)** |
| `TOKEN_TO_REP_ATOM SHAPE MISMATCH` | 0 | **0** |
| `ATOM SHAPE MISMATCH` (`atom_pad_mask`) | 0 | **2 (0.4%)** |
| **scored** | 103 (10.8%) | **32 (6.0%)** |

Skip accounting closes exactly on both (500 CSV skip rows = 500 log reasons on
1JTG). The cause reproduces: **the representative-atom content check is
essentially the whole of it**, 93.6% against 89.2%. One small difference worth
recording rather than smoothing over -- section 24 reported *zero* skips on the
bare atom-shape guard, and 1JTG has **2**. So "none on shape" is a 3S9D fact,
not a universal one, though at 0.4% it changes nothing.

### The decorrelation: scale reproduces emphatically, rank is untestable here

| complex | n both scored | Spearman(swap, refeat) | sign agree | median abs swap | median abs refeat | ratio |
|---|---|---|---|---|---|---|
| 3S9D | 103 | **-0.0744** | 0.641 | 0.0118 | 0.4331 | **36.8x** |
| **1JTG** | **32** | **0.2632** | 0.531 | 0.0137 | **0.9845** | **71.8x** |

**The scale finding replicates and strengthens.** The re-featurised mutant moves
`complex_pde` by a median of **0.98** on 1JTG against the one-hot swap's
**0.0137** -- a **71.8x** gap, double 3S9D's 36.8x. The sequence-structure
consistency response that section 24 identified is not a 3S9D quirk; it is
larger on the bigger complex.

**The rank decorrelation is not confirmed and not refuted.** 1JTG gives +0.2632
where 3S9D gave -0.0744, but at n=32 the sampling CI on a Spearman is roughly
±0.35, so +0.26 is statistically indistinguishable from zero *and* from -0.07.
Sign agreement is 0.531, i.e. coin-flip. The honest statement is that 1JTG is
**consistent with** the decorrelation without independently establishing it, and
that the guard's 94% skip rate is what destroys the power to test it. 2JEL has
32 positions and 3HFM 15, so neither is likely to do much better.

## 5. Predictions registered before the runs, scored

<!-- fill -->

## 6. Caveats

Section 24's caveat list applies here unchanged and is not repeated in full:
the MSA is pinned to wild type throughout; SKEMPI positions are interface-
enriched and not a random sample of the chain; `grad_res_type` is the
path-averaged gradient rather than a true IG attribution; the mutation-level
CIs and permutation nulls assume independent mutations when the rows cluster
within positions, so both are optimistic; and the whole measurement is
model-vs-model and says nothing about measured ddG. The caveats *new* to this
section are below.

1. **The m=32 artifacts for 2JEL and 3HFM were not re-run; they are reused
   from section 14** (commit 0bf19ef, clean tree, identical command line).
   This is only sound because `f(x)` is unchanged between 0bf19ef and 420a456
   -- see §2, where that is checked rather than assumed. It is still weaker
   than a fresh run: ~3,500 lines of `boltz_score.py` changed in between, and
   the check verifies the *value* of the score, not that every intermediate is
   identical.
2. **The headline is carried by different quadrature rungs on different
   complexes.** 3S9D's number is IG m=64; 1JTG's is m=32 and m=64; 2JEL's and
   3HFM's are m=32 only, because m=64 does not fit the 2 h window. Section 24
   measured the m=32-vs-m=64 attribution agreement on 3S9D at Spearman 0.9990
   with a maximum absolute difference of 0.0031, which is the evidence that the
   rung does not matter for ranking -- but that was measured on one complex,
   and it is an assumption everywhere else. The `plaingrad` column is
   rung-free and is reported alongside for exactly this reason.
3. **3HFM is underpowered at the position level and its 8-confound partial is
   not identifiable.** The confounds are constant within a position, so however
   many mutations there are the confound matrix has only as many distinct rows
   as positions -- 15 for 3HFM, against 8 confounds. Section 14 measured the
   resulting partial CI as [-1.0, +1.0]. The mutation-level n=285 is fine for
   the raw Spearman and for the single-confound distance partial; the 8-confound
   partial for 3HFM is printed with an UNIDENTIFIABLE flag and **must not be
   quoted as a result**.
4. **The refeaturise arm is a biased subsample everywhere.** As in section 24,
   the survivors of the `token_to_rep_atom` guard are close to exactly the
   heavy-atom-count-preserving substitutions, i.e. the most conservative
   mutations on the grid. A null there is "no detectable signal on a
   conservative subsample", not "no signal".
5. **`git_dirty` is `true` on every artifact in this section**, at the exact
   commit 420a456. The cause is known and benign: `results/*.prov.json` are
   tracked files, and the section-14 sidecars were renamed to `*_sec14_backup`
   so this run would not overwrite them, which git reports as deletions; plus an
   untracked `logs/` directory. No tracked source file differs from 420a456.

## 7. What the 2 h cap cost, and what is still owed

### State of the run at hand-off, 2026-10-14 06:05 UTC

The cap was raised to 43200 s (12 h) part-way through, with the user's approval,
after the first window had already been spent. What exists:

| artifact | state |
|---|---|
| 1JTG plain_grad | **done**, f(x) = 1.432957 |
| 1JTG IG m=32 | **done**, 3322 s, peak 42.6 GiB |
| 1JTG resswap scan | **running**, resumed cleanly at 220/533, 11.0 s/mutant |
| 2JEL plain_grad | **done**, f(x) = 1.419646 |
| 3HFM plain_grad | **done**, f(x) = 0.998942 |
| 2JEL / 3HFM IG m=32 | **reused from section 14**, check passed (§2) |
| everything else | queued |

### An operational defect in the driver, found and fixed

The first window's driver stamped a `*.DONE` completion marker whenever a stage
returned 0 -- including when the stage had been *skipped* because it was not
selected for that window. 2JEL and 3HFM therefore both acquired zero-byte
`scan.csv.DONE` and `scan_refeaturise.csv.DONE` markers for scans **that had
never run**, which would have caused every later window to skip them silently
and report the complexes as complete with no scan data at all.

Caught by reading the artifact listing rather than the driver's own log, which
cheerfully said `COMPLEX 2JEL COMPLETE`. The markers were deleted and the driver
now returns distinct codes for ran-ok (0), already-done (7), not-selected (8)
and deferred (9), and stamps the marker only on 0. **This is the same class of
error as section 14's "the producer was never wired": a pipeline reporting
success for work it never did.**

### Resume was verified rather than assumed

The 2 h stop killed the 1JTG scan mid-flight at 219 mutants. On restart the
scan logged `Resuming: 220 mutations already scored`, took the wild-type score
from the existing rows rather than recomputing it, and began at `[10/313]` --
the remainder only. The CSV sanitiser reported `CSV_OK (220 data rows)`, i.e.
the forced stop did not truncate the final line. **No restart, no duplicate
rows.** This is worth recording because `_load_done_ids` swallows a CSV parse
failure and returns an empty set, which would silently restart the scan *and*
append duplicates -- the failure mode the sanitiser exists to prevent.

### Still owed

- IG m=64 on 2JEL (~234 min) and 3HFM (~259 min). These did not fit the 2 h cap
  per-job, and do not fit the 12 h window either once the headline scans are
  paid for. They are queued last and will defer.
- The completeness ladder is therefore **m=32 only** on 2JEL and 3HFM.

## 8. Artifacts

<!-- fill -->
