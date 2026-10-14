# Draft section 24: the shortcut agrees with the slow way — on the confidence head, and only for the function the gradient actually sees

Run 2026-10-13 on the A100 (`igv-gpu`, a2-ultragpu-1g, us-central1-a).

We asked whether gradient attribution on Boltz-2's `complex_pde` confidence
score agrees with what the same model does when you mutate the residue and run
it again — on 3S9D chain A, at the 50 SKEMPI-measured positions, all 19
substitutions at each.

**It does, at Spearman 0.59 over 950 mutations, and it is not geometry.** That
is the shortcut-vs-slow-way number the project has been missing since section 8.

**And it does not transfer to the real mutant.** Against a fully re-featurised
mutant the same attribution is flat null (-0.01, p=0.90). Both results are in
this note, and the second one bounds the first.

## 1. Completeness: IG converges on this head

The gate is Sundararajan et al.'s 5% relative criterion. m=32 is the historical
rung from section 14; m=64 was run today.

| m | f(x) | f(baseline) | span | rel err | abs err | 5% gate |
|---|---|---|---|---|---|---|
| 32 | 1.883068 | 3.525829 | **-1.642760** | 0.0851 (8.51%) | 0.139723 | FAIL |
| 64 | 1.883068 | 3.525829 | **-1.642760** | **0.0024 (0.24%)** | 0.003971 | **PASS** |
| 128 | — | — | — | — | — | **did not run** (see §7) |

**Doubling the step count cut the error 35x, monotonically, with no sign flip.**
That is quadrature convergence, and it is the exact opposite of the affinity
head, where section 22 measured absolute error ~1-3 that would not shrink and
`ig_sum` flipping sign between consecutive m under two different baselines.

The reason is in the span column. Here `f(x) - f(baseline)` is **-1.6428**; on
the affinity head it was **0.085** (`mean_aa`) or 0.429 (`unk`). The confidence
score genuinely moves when the sequence is replaced by a uniform amino-acid
distribution, so there is a real quantity for the integral to recover. Sections
21-22 argued the affinity failure was a degenerate-span problem rather than a
coarse-grid problem; **this run is the positive control for that argument**, on
the same model and the same trunk.

A second convergence check, not previously run: the path-averaged `res_type`
gradient — the object that actually feeds §2 — is already stable at m=32.
Between m=32 and m=64 its Spearman is **0.9990**, Pearson 0.9992, max absolute
difference 0.0031 against an absolute max of 0.072. So the attribution is
insensitive to the rung even at the rung that fails completeness. m=128 was
therefore confirmatory, not load-bearing.

## 2. Faithfulness against the model's own scan (`resswap`)

Ground truth B is `04_scan.py --mode resswap`: 950 mutants, 50 positions x 19
substitutions, every feature held at wild type except `res_type[pos]`. That is
the same function the gradient differentiates. The wild-type row was written
first and its score, **1.8830682039260864**, matches stage 03's `f(x)` to all
printed digits — scan and attribution are provably on the same geometry and the
same featurisation, which is what defect 1 in section 23 was about.

A is the TISM-style difference `grad[pos, mut] - grad[pos, wt]`.

| | IG m=64 | plain_grad |
|---|---|---|
| n | 950 | 950 |
| **Spearman** | **0.5937** [0.5404, 0.6399] | **0.5885** [0.5350, 0.6379] |
| Pearson | 0.5518 [0.4599, 0.6372] | 0.4885 [0.3810, 0.5858] |
| permutation null (p5, p95) | -0.0522, 0.0534 | -0.0535, 0.0527 |
| p (two-sided) | **< 1e-4** | **< 1e-4** |
| scale ratio median\|A\|/median\|B\| | **0.7484** | 0.4380 |
| median\|A\| | 0.0090 | 0.0053 |
| median\|B\| | 0.0120 | 0.0120 |
| **partial, controlling distance to partner** | **0.5722** | **0.5653** |
| partial, all 8 confounds | **0.5920** | 0.5557 |
| per-position, sum\|A\| vs sum\|B\| | 0.5546 [0.3129, 0.7305] | 0.6642 [0.4654, 0.7890] |
| per-position, max\|A\| vs max\|B\| | 0.5773 | 0.4017 |

**The partial correlation is the row that matters.** This project was fooled by
geometry once — section 14 measured `distance_to_partner` at -0.5769 against
|ddG| on this very complex, and `burial` at +0.6411 — so the raw number is not
reportable alone. Controlling for minimum heavy-atom distance to chain B moves
Spearman from 0.5937 to **0.5722**, a drop of 0.02. Controlling for all eight
confounds at once (burial, delta_rsa, distance_to_partner, hydrophobicity,
norm_position, residue_volume, rsa_complex, rsa_monomer) leaves it at **0.5920**
— essentially undiminished.

**This is not the geometry artifact that fooled Phase 5.** Unlike every earlier
positive in this project, this one does not evaporate when the confounds come
out.

The permutation null sits at ±0.05 against an observed 0.59, about eleven times
the null's width. Contrast section 8's 1JTG number (0.357 against a null p95 of
0.313) and section 14's position-level 3S9D number (CI lower bound +0.206).

**Scale.** IG's ratio 0.75 means the first-order prediction is the right order
of magnitude, undershooting the true delta by ~25% at the median. plain_grad
undershoots by 56%.

**IG barely beats plain_grad on ranking** — 0.5937 vs 0.5885, far inside each
other's CIs. The honest statement is that on this head the cheap single backward
pass is as good a *ranker* as the 64-step integral, and IG's advantage is in
magnitude calibration (0.748 vs 0.438) and Pearson (0.55 vs 0.49). This does not
contradict section 8's "plain_grad is not a cheap substitute for ig" on 1JTG:
that was measured as ranking overlap *between the two attributions*, not as
faithfulness to a scan. Different questions, both answered.

The per-position aggregate here is trustworthy because the design is balanced:
exactly 19 substitutions at each of the 50 positions, so the per-position count
is constant by construction and cannot drive the correlation.

## 3. The real mutant (`refeaturise`): flat null

`--mode refeaturise` rebuilds the full mutant features and reuses wild-type
coordinates. The atom-shape guard skipped most of the grid, as section 23
predicted.

| | count | share |
|---|---|---|
| mutants attempted | 950 | 100% |
| **skipped, `TOKEN_TO_REP_ATOM CONTENT DIFFERS`** | **847** | **89.2%** |
| skipped, `ATOM SHAPE MISMATCH` (`atom_pad_mask`) | **0** | 0% |
| skipped, `TOKEN_TO_REP_ATOM SHAPE MISMATCH` | **0** | 0% |
| scored | **103** | 10.8% |

Every single skip was the representative-atom *content* check, not a shape
check. That is the hazard section 23 confirmed from `featurizer.py:726` and
`loss/confidence.py:188-189`: the rep-atom map is built from per-token
`disto_idx` into the atom table, so any change in heavy-atom count at position p
shifts the atoms every later token points at. The guard is working, not failing.

The 103 survivors sit at 45 of the 50 positions, 1-4 substitutions each, and are
close to exactly the set of substitutions that **preserve heavy-atom count**
(Arg(11) -> Phe(11), etc.). This is a strongly biased subsample.

| | IG m=64 | plain_grad |
|---|---|---|
| n | 103 | 103 |
| Spearman | **-0.0123** [-0.2269, 0.1931] | **-0.0783** [-0.2789, 0.1432] |
| Pearson | -0.0231 [-0.2198, 0.1733] | -0.1220 [-0.3482, 0.1037] |
| permutation null (p5, p95) | -0.1629, 0.1613 | -0.1646, 0.1593 |
| p (two-sided) | **0.901** | 0.435 |
| partial, distance to partner | 0.0261 | -0.0498 |
| partial, all 8 confounds | 0.0537 | -0.0497 |
| scale ratio | 0.0195 (uninterpretable) | 0.0118 (uninterpretable) |

**Both null, at every level.**

### Why: the real mutant's response is dominated by something the gradient cannot see

| | resswap | refeaturise |
|---|---|---|
| median \|delta\| | **0.0120** | **0.4331** |
| spread of delta | — | range 0.380-0.462, sd 0.0163 |

**A 36x difference.** The refeaturise deltas cluster tightly around +0.43
*regardless of which position or which substitution*, with only sd 0.016 of
residual variation — and that residual is the only part a `res_type` gradient
could possibly predict.

That pattern looked like a featurisation artifact, so it was tested rather than
assumed. `04_scan.py` computes `wt_score` once from the `boltz_ref_{dataset}`
cache and reuses it as the baseline for *both* modes, while refeaturise mutants
are built by `build_complex_feats` into a *different* cache directory. A
constant offset between the two featurisation routes would contaminate every
refeaturise delta.

**Control (`results/3S9D_wt_refeat_control.json`), one forward pass: the offset
is exactly zero.** Re-featurising the *unchanged* wild-type sequence into a
scan-style cache dir returns **1.8830682039260864** — bit-identical to the
reference-cache score — and the atom-shape guard passes on the identical
sequence.

```
wt_ref   = 1.8830682039260864
wt_fresh = 1.8830682039260864
offset   = 0.0
```

So **the null is not a featurisation bug.** The +0.43 is a real model response:
when a residue's reference-conformer and atom-identity features say one thing
and the supplied wild-type coordinates say another, Boltz-2 raises its predicted
distance error by ~0.43 almost regardless of which residue changed. It is a
**sequence-structure consistency signal, 36x larger than the mutation-effect
signal we are trying to measure**, and the `res_type` one-hot gradient is
structurally blind to it.

### The two scans are uncorrelated with each other

On the 103 mutations both modes scored:

| quantity | value |
|---|---|
| Spearman(resswap delta, refeaturise delta) | **-0.0744** |
| Pearson | -0.0208 |
| sign agreement | 0.641 |
| median \|difference\| | 0.425219 |
| max \|difference\| | 0.496477 |

**The one-hot swap and the real mutant are uncorrelated.** This is the most
important negative in this note, and it bounds §2 precisely: the 0.59 is a
statement about *the function the gradient differentiates*, not about what the
model does to a genuinely mutated input.

### A trap in the refeaturise per-position aggregate — do not quote it

The refeaturise JSONs report an apparently healthy per-position result: IG
`sum_spearman` **0.5925** CI [0.3173, 0.8013] over 45 positions, plain_grad
0.4084 [0.1430, 0.6191], both clearing zero. **It is an artifact.**

`sum_spearman` correlates sum|A| against sum|B| per position. Because B is a
near-constant 0.43 per surviving mutation, sum|B| is essentially a count of how
many substitutions survived the guard at that position:

| check | value |
|---|---|
| Spearman(survivor count, sum\|B\|) | **0.9425** |
| Spearman(survivor count, mean\|B\|) | -0.0128 |
| Spearman(survivor count, sum\|A\|) | 0.5654 |
| Spearman(sum\|A\|, sum\|B\|) — the reported number | 0.5925 |
| **partial Spearman(sum\|A\|, sum\|B\|) controlling for count** | **0.1603** |

Survivor count explains almost all of sum|B| (0.94), and per-mutation effect
size is flat with respect to it (-0.01). Controlling for the count collapses the
correlation from 0.5925 to **0.1603**. The apparent per-position agreement is
three-quarters survivor-count bookkeeping, and only arises because the guard
skipped an unbalanced 89.2% of the grid. The resswap per-position aggregate has
no such problem (19 substitutions at every position, count constant).

## 4. The four predictions registered in section 23

Registered before the runs, scored here plainly.

1. **"Completeness will converge on `complex_pde` and fail on
   `affinity_pred_value` on the SAME complex and path."**
   **HALF-CONFIRMED.** The `complex_pde` half held decisively: 8.51% -> 0.24%
   from m=32 to m=64, monotone, no sign flip. The affinity half **was not
   tested** — that needs `23_affinity_path_profile.py` (queue item 2), which did
   not run. The affinity failure is on record from section 22 but on a
   *different complex* (ABL1/2HYY), so the same-complex, same-path comparison
   this prediction actually asks for is **still open**.
2. **"`total_variation_above_noise` will be large for the affinity head and ~0
   for the confidence head."**
   **NOT TESTED.** Requires `23_affinity_path_profile.py`. No GPU time reached it.
3. **"Faithfulness on 3S9D `resswap` will beat the section-22 affinity
   numbers."**
   **CONFIRMED, decisively.** Section 22 got Spearman -0.4964 (n=18, p=0.036)
   under `mean_aa` and +0.2487 (p=0.320) under `unk` — unstable in sign, neither
   robust. Here: **+0.5937 at n=950, p<1e-4**, surviving the geometry control
   (0.5722) and stable across attribution methods (IG 0.594, plain_grad 0.589)
   and across quadrature rungs (m=32 vs m=64 attribution Spearman 0.9990).
   Magnitude, stability and power all better.
4. **"Protenix `complex_pde` completeness will land near Boltz's (~0.04 at
   m=32)."**
   **NOT TESTED.** No Protenix image was built.

**Score: one confirmed decisively, one half-confirmed, two untested.** The two
untested ones are untested because the GPU session was cut short, not because
they were dropped.

A prediction *not* registered in section 23, and worth recording as a miss:
nothing in the plan anticipated that `resswap` and `refeaturise` would be
**uncorrelated** (-0.07). Section 23 framed refeaturise as "the real mutant as a
secondary measurement of model behaviour", implicitly expecting it to agree
roughly with the primary. It does not, at all. That is the finding with the most
consequence for how the paper can be worded.

## 5. Caveats, in order of how much they should worry you

1. **One complex, one score, one chain.** Section 14's own lesson, in bold
   there: between-complex variance (sd ~0.33) swamped the between-method
   difference on the hotspot arm, and 3S9D was the single favourable draw in
   four. This is a different measurement — mutation-level faithfulness at n=950,
   not position-level hotspot ranking at n=50 — but the warning transfers.
   **Do not quote 0.59 as "the" faithfulness number for Boltz-2 until 1JTG,
   2JEL and 3HFM have run the same scan.** That is the obvious next buy and it
   is cheap: ~75 min of A100 per complex.
2. **The refeaturise null rests on 103 mutations selected by a biased filter.**
   89.2% of the grid was skipped, and the survivors are near-exactly the
   heavy-atom-count-preserving substitutions — i.e. the *most conservative*
   mutations available. A null on the mildest 11% of the grid is weaker evidence
   than a null on all of it. The honest reading is "no detectable signal on a
   conservative subsample", not "no signal".
3. **The completeness number does not gate the object being correlated.**
   `completeness_error` in stage 03 is computed on the *embedding-space* IG
   (`s_inputs` interpolated to the `mean_aa` baseline). The A used in §2 comes
   from `_ig_res_type`, a *different* path that interpolates the `res_type`
   one-hot through the input embedder. Same alpha schedule, same trunk, same
   step count — different endpoints, and the res_type path's own completeness
   was never measured. The ladder is strong evidence this trunk integrates
   cleanly; it is not a direct audit of the attribution that produced 0.5937.
   Fixing it needs `_ig_res_type` to return the full-complex array (it is
   currently sliced to the chain before saving) plus one forward pass at the
   res_type baseline.
4. **`grad_res_type` is the path-averaged gradient, not the IG attribution.** It
   is never multiplied by `(x - baseline)`, so the "IG" column is
   baseline-dependent by construction, exactly as section 22 flagged. The
   reassurance here is empirical, not structural: it is 0.999-correlated between
   m=32 and m=64, and lands within 0.005 Spearman of plain_grad, which has no
   baseline at all. A baseline sweep (`unk`, `zeros`) would settle it and was
   not run.
5. **The CIs and the permutation null assume independent mutations.**
   `bootstrap_ci` resamples the 950 rows i.i.d. and the permutation shuffles
   them freely, but the rows cluster in 50 positions sharing local structure.
   Both the CI width and the p-value are therefore optimistic. The per-position
   aggregate (n=50, sum_rho 0.5546 [0.3129, 0.7305]) is the honest unit of
   analysis, and it still excludes zero comfortably — which is the reassuring
   part.
6. **SKEMPI positions are not a random sample of the chain.** They are where
   experimentalists looked, which is the interface. The 0.59 is measured on an
   interface-enriched set and may not hold at buried or distal positions.
7. **MSA pinned to wild type throughout** (section 16), in both attribution and
   scan. Correct for this comparison — IG holds `profile` fixed by construction,
   so letting the scan re-query would compare two different functions — but it
   means neither number reflects what the deployed model would do on a real
   mutant sequence. Section 16's unresolved `mode="all"` note also still applies.
8. **No experimental ground truth here.** This is model-vs-model: attribution
   against the model's own scan. It says nothing about whether either tracks
   measured ddG. That is a separate question and section 18's verdict on it
   still stands.

## 6. Cost

| stage | forward passes | wall | exit |
|---|---|---|---|
| 03 IG m=64 | 128 fwd+bwd (64 embedding + 64 res_type) + structure pred | **44 min 22 s** | 0 |
| 03 plain_grad | 1 fwd+bwd + structure pred | **1 min 53 s** | 0 |
| 04 resswap | 951 fwd | **74 min 37 s** (4.7 s/mutant) | 0 |
| 04 refeaturise | 103 fwd + 847 featurise-and-skip | **18 min 30 s** (10.2 s/scored) | 0 |
| 24 faithfulness x 4 | 0 (CPU) | **28-30 s** each | 0 |
| 03 IG m=128 | — | killed at ~step 100/128 of 256 | — |

Totals: roughly **2,160 forward passes** actually executed (128 + 1 + 951 + 103,
plus structure predictions and baseline evaluations), against the ~2,330
estimated in section 23 — the shortfall is the 847 refeaturise mutants that were
featurised and skipped before any forward pass, and the lost m=128.

**Peak VRAM 20.2 GiB** (20648 MiB) during the m=64 attribution. During the
no-grad scans a sampled reading was 5.3 GiB; that is a sample, not a measured
peak. Either way **the 80 GB card is not needed for this arm**, consistent with
section 20's 16.80 GiB and section 22's 23.87 GiB. A 24 GB L4 is marginal on the
attribution stage; 40 GB would be comfortable.

## 7. What went wrong operationally, and the m=128 rung

**The VM was stopped by an external watchdog three times** during this session —
at 20:10, 22:42 and ~23:25 UTC — each time mid-run. Stage 03 is not resumable,
so each stop destroyed an in-flight attribution:

- **Stop 1** killed the first m=128 attempt. The m=64 artifact had already
  landed on the retained disk and was recovered.
- **Stop 2** killed nothing in flight (phases 1 and 2 were complete and synced).
- **Stop 3** killed the second m=128 attempt at roughly step 100 of its first
  128-step integral, about a third of the way through the job.

Restarting hit `ZONE_RESOURCE_POOL_EXHAUSTED` (STOCKOUT) in us-central1-a on
several attempts and succeeded on retry; the external IP changed between
restarts, which broke a cached rsync target once.

**m=128 was not relaunched.** It is confirmatory only: m=64 already passes the
5% gate by a factor of 20 (0.24%), the ladder 8.51% -> 0.24% already establishes
convergence, and the attribution itself is 0.999-correlated between m=32 and
m=64, so a third rung would not change any number in §2. Another ~90 minutes of
A100 against a near-certain confirmation was not worth the risk of a fourth
forced stop.

**Process changes adopted mid-session, worth keeping:** artifacts are now synced
to local `results/` with their `.prov.json` as each stage lands rather than
batched at the end (this is why stop 2 and stop 3 cost nothing), and the queue
was reordered to put the headline — plain_grad, the resswap scan, and
faithfulness — ahead of the expensive confirmatory rung.

## 8. Artifacts

All in `results/`, each with a `.prov.json` sidecar except the hand-written
control:

```
3S9D_complex_pde_ig_grad.npz              IG m=64, mean_aa, x-pred predicted
3S9D_complex_pde_ig_grad_m32_backup.npz   the section-14 m=32 run, preserved
3S9D_complex_pde_plain_grad_grad.npz      plain gradient at x
3S9D_complex_pde_scan.csv                 resswap, 950 mutants + WT
3S9D_complex_pde_scan_refeaturise.csv     refeaturise, 103 scored + 847 skipped + WT
3S9D_faith_ig64_resswap.json              the headline
3S9D_faith_plaingrad_resswap.json
3S9D_faith_ig64_refeat.json
3S9D_faith_plaingrad_refeat.json
3S9D_wt_refeat_control.json               offset 0.0 (no sidecar — ad-hoc probe)
```

Nothing was committed. The VM is TERMINATED and no instances are running.
