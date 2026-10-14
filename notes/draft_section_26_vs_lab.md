# Draft section 26: does the model track the lab, and does the shortcut?

Local analysis, 2026-10-14, no GPU. Everything here is computed from artifacts
already on disk.

Section 25 established that gradient attribution reproduces Boltz-2's own
brute-force substitution scan at a mean Spearman of 0.518 across four
complexes. This section asks the two questions that were still missing: does
that scan agree with measured binding free energies (**model vs lab**, B vs C),
and does the attribution (**shortcut vs lab**, A vs C). Both are computed on
the exact mutations SKEMPI measured, per complex, with a permutation null, a
bootstrap CI and a partial on distance to the partner chain.

## Headline

**The model tracks the lab on two of four complexes and not on the other two,
and the four-complex mean is not distinguishable from zero.** Model vs lab is
+0.4205 (3S9D), +0.4376 (1JTG), -0.1693 (2JEL), +0.2530 (3HFM): mean +0.2354,
sd 0.2824, t=+1.67, **p=0.19**, 2 of 4 clearing their own shuffled null. The
same pattern, slightly stronger at the top and weaker at the bottom, holds for
IG attribution vs lab: +0.4678, +0.5591, +0.0385, -0.0049, mean +0.2651,
**p=0.17**, 2 of 4 clearing null.

For scale: the experimental column agrees with **itself** at a split-half
Spearman of 0.84-0.96 on the three complexes with replicate measurements. The
ceiling is not the problem. The model is reaching about half of it where it
works and none of it where it does not.

Set against section 25, the project's three comparisons now read:

| comparison | mean Spearman, 4 complexes | complexes clearing own null |
|---|---|---|
| shortcut vs slow way (A vs B, section 25) | **0.518** | 4 / 4 |
| model vs lab (B vs C, here) | +0.235, p=0.19 | 2 / 4 |
| shortcut vs lab (A vs C, here, IG) | +0.265, p=0.17 | 2 / 4 |

**The shortcut reproduces the model far better than the model reproduces
nature.** That ordering is the honest summary of the project.

## 1. Sign convention, derived and then verified

Derived from first principles before any correlation was read:

- `complex_pde` is a predicted **distance error**. A LOWER value is a better
  complex. It is in `12_compare.LOWER_IS_BETTER` and stage 26 asserts the two
  sets are identical in a test.
- The scan's `delta` is `f(mutant) - f(wild type)`, so **positive delta = the
  model thinks the mutant is worse**.
- SKEMPI DDG is `R*T*ln(Kd_mut/Kd_wt)`, so **positive DDG = the mutation
  weakens binding**.
- Both are therefore already on the same "positive = destabilising" axis.
  **No negation.** Agreement is a POSITIVE correlation.
- A is a first-order Taylor term in the units of the score
  (`grad[pos, mut] - grad[pos, wt]`), so it inherits the score's orientation.

Verified on the mutations experiment is most confident about, before any
correlation was quoted. Median oriented scan delta for measured hot spots
(DDG >= 2) against measured neutrals (DDG <= 0.5):

| complex | n hot | n neutral | median delta, hot | median delta, neutral | Mann-Whitney p | anchor |
|---|---|---|---|---|---|---|
| 3S9D A | 9 | 34 | +0.01970 | +0.00013 | **0.0004** | holds |
| 1JTG B | 11 | 12 | +0.01752 | +0.00579 | **0.039** | holds |
| 3HFM Y | 22 | 13 | +0.02568 | +0.01343 | 0.157 | holds, not significant |
| 2JEL P | 9 | 31 | -0.00322 | -0.00002 | 0.145 | **inverted, not significant** |

The anchor is named rather than assumed: on 1JTG the five largest measured
DDGs are W150A (+4.34), K74A (+3.69), H41A (+3.44), F36A (+3.20), Y53A (+3.18)
-- the canonical BLIP hot spots -- and all five carry a positive scan delta.
On 3HFM the two largest are K96M and K97D, the canonical HyHEL-10 epitope hot
spots, both positive. The convention is right. 2JEL's inversion is not a sign
error; it is 2JEL having no signal (section 5).

The script logs `ORIENTATION ANCHOR INVERTED` loudly when the hot-spot median
does not exceed the neutral median, and records the verdict in JSON rather
than raising -- an inverted anchor is a result, but it must be visible.

## 2. What joined, and every n

SKEMPI measures specific mutations, not 19 substitutions per position, so the
join is small. Row counts are after mapping author residue numbering
(`Mutation(s)_PDB`, never `Mutation(s)_cleaned`) onto the chain index the scan
and the attribution share, and after collapsing duplicate measurements of the
same mutation by median.

| complex | SKEMPI rows on chain | unique mutations | positions | **n joined (resswap)** | to-Ala | replicated | scan rows available |
|---|---|---|---|---|---|---|---|
| 3S9D A | 108 | 54 | 50 | **54** | 87% | 30 | 950 |
| 1JTG B | 96 | 30 | 28 | **30** | 80% | 21 | 532 |
| 2JEL P | 44 | 44 | 32 | **44** | 14% | 0 | 608 |
| 3HFM Y | 67 | 42 | 15 | **42** | 31% | 23 | 285 |

Zero residue-ID lookup failures and zero wild-type mismatches on all four, so
nothing was silently dropped. The scan covered every SKEMPI position by
construction (stage 04 took its positions from `skempi_positions`), so the join
loses only substitutions SKEMPI never measured.

**n is 30-54.** Every number below is quoted with it, and nothing under n=20 is
interpreted.

### The experimental ceiling, measured from inside the data

Split-half test-retest Spearman over the mutations with replicate
measurements, 1000 random splits:

| complex | replicated mutations | test-retest Spearman | median replicate spread | max spread |
|---|---|---|---|---|
| 3S9D A | 30 | **0.960** [0.943, 0.976] | 0.24 kcal/mol | 2.59 |
| 1JTG B | 21 | **0.881** [0.843, 0.917] | 1.33 kcal/mol | 1.85 |
| 3HFM Y | 23 | **0.844** [0.823, 0.891] | 0.32 kcal/mol | 0.61 |
| 2JEL P | 0 | not estimable | -- | -- |

1JTG's median replicate spread of 1.33 kcal/mol is large in absolute terms --
D49A alone has seven literature measurements spanning 1.67 to 3.13 -- but the
ranking survives it at 0.88 because the dynamic range is 6.4 kcal/mol. These
ceilings are estimated on replicated mutations only, which skew towards
well-studied hot spots, so treat them as approximate.

## 3. Model vs lab (B vs C), resswap

Signed scan delta against signed DDG. Null is 10,000 shuffles of the model
column; partial regresses out minimum heavy-atom distance to the partner chain.

| complex | n | Spearman | 95% CI | Pearson | null p95 | perm p | partial (distance) | clears null |
|---|---|---|---|---|---|---|---|---|
| 3S9D A | 54 | **+0.4205** | [+0.183, +0.626] | +0.4589 | +0.228 | **0.0015** | **+0.4199** | YES |
| 1JTG B | 30 | **+0.4376** | [+0.011, +0.708] | +0.5239 | +0.312 | **0.0151** | **+0.5368** | YES |
| 2JEL P | 44 | -0.1693 | [-0.453, +0.139] | -0.1753 | +0.252 | 0.268 | -0.2090 | no |
| 3HFM Y | 42 | +0.2530 | [-0.039, +0.503] | +0.3006 | +0.258 | 0.109 | +0.1921 | no |

Mean +0.2354, sd 0.2824, t=+1.67, **p=0.19**.

**The geometry control does not take the positives away.** On 3S9D the partial
costs 0.0006 (0.4205 -> 0.4199); on 1JTG it *rises* to +0.5368, because
distance barely correlates with the model's response there (rho=+0.047) while
correlating -0.298 with DDG, so removing it is a suppression effect, not a
confound. This is the control that killed Phase 5's hot-spot result and it does
not kill this one.

### Sensitivity: dropping the DDG = 0 rows

Many SKEMPI rows report `Kd_mut == Kd_wt` to one significant figure, giving an
exact zero that carries no rank information:

| complex | zeros | n without zeros | Spearman | perm p |
|---|---|---|---|---|
| 3S9D A | 8 | 46 | **+0.5521** | **0.0000** |
| 1JTG B | 1 | 29 | +0.4296 | 0.0217 |
| 2JEL P | **22** | 22 | -0.1077 | 0.631 |
| 3HFM Y | 2 | 40 | +0.2285 | 0.157 |

3S9D rises from +0.42 to +0.55 once the flat rows are removed. 2JEL stays flat
either way.

## 4. Shortcut vs lab (A vs C), resswap

`A = grad[pos, mut] - grad[pos, wt]`, same join, same statistics. IG rung is
m=64 on 3S9D and 1JTG, m=32 on 2JEL and 3HFM (section 25: m=64 did not fit the
window on those two).

| complex | n | method | Spearman | 95% CI | Pearson | null p95 | perm p | partial (distance) | clears null |
|---|---|---|---|---|---|---|---|---|---|
| 3S9D A | 54 | IG m=64 | **+0.4678** | [+0.226, +0.668] | +0.5311 | +0.226 | **0.0008** | +0.3964 | YES |
| 3S9D A | 54 | plain_grad | +0.1196 | [-0.136, +0.354] | +0.1018 | +0.228 | 0.386 | +0.0046 | no |
| 1JTG B | 30 | IG m=64 | **+0.5591** | [+0.236, +0.759] | +0.5121 | +0.307 | **0.0013** | **+0.5635** | YES |
| 1JTG B | 30 | IG m=32 | +0.5613 | [+0.260, +0.744] | +0.4879 | +0.306 | 0.0011 | +0.5604 | YES |
| 1JTG B | 30 | plain_grad | +0.2618 | [-0.130, +0.588] | +0.2687 | +0.308 | 0.165 | +0.2587 | no |
| 2JEL P | 44 | IG m=32 | +0.0385 | [-0.230, +0.327] | +0.0555 | +0.254 | 0.802 | +0.0754 | no |
| 2JEL P | 44 | plain_grad | +0.1482 | [-0.206, +0.477] | +0.3084 | +0.252 | 0.334 | +0.1356 | no |
| 3HFM Y | 42 | IG m=32 | -0.0049 | [-0.300, +0.289] | +0.0619 | +0.260 | 0.975 | -0.0748 | no |
| 3HFM Y | 42 | plain_grad | -0.0442 | [-0.352, +0.277] | +0.1260 | +0.261 | 0.781 | +0.0445 | no |

IG mean +0.2651, sd 0.2897, p=0.17. plain_grad mean +0.1214, sd 0.1263,
p=0.15, **0 of 4 clearing null**.

Two things worth keeping. First, **IG beats plain_grad against experiment on
three of four complexes**, by +0.35 on 3S9D and +0.30 on 1JTG -- a much larger
margin than the 0.005-0.22 IG advantage section 25 measured against the model's
own scan. Where there is any lab signal to find, the path integral finds more
of it. Second, the m=32 and m=64 rungs on 1JTG give +0.5613 and +0.5591, i.e.
the quadrature rung is irrelevant at this n; the completeness gate failures
recorded in section 25 do not drive these numbers.

## 5. Magnitude, and what it does to section 14

Section 14 worked in **|DDG| at the position level** with the simplex
reduction, so the like-for-like comparison is magnitude, not signed.

| complex | sec 14 simplex vs \|DDG\| (positions) | **\|A_IG\| vs \|DDG\| (mutations)** | \|B\| vs \|DDG\| | \|A_plain\| vs \|DDG\| |
|---|---|---|---|---|
| 3S9D A | +0.4683 | **+0.4774** (p=0.0002) | +0.2104 (p=0.126) | -0.1757 |
| 1JTG B | +0.1714 | **+0.5195** (p=0.0038) | +0.4256 (p=0.021) | +0.0892 |
| 2JEL P | **-0.3396** | -0.0315 (p=0.835) | +0.1670 (p=0.272) | +0.3386 (p=0.025) |
| 3HFM Y | +0.1573 | +0.0422 (p=0.794) | +0.2140 (p=0.176) | +0.0242 |
| **mean** | **+0.1143**, sd 0.3349, p=0.54 | **+0.2519**, sd 0.2868, p=0.18 | +0.2542, sd 0.1162, **p=0.022** | +0.0691, p=0.56 |

**This supersedes section 14 on 3S9D and 1JTG, and confirms its verdict
overall.** It supersedes rather than contradicts: the complex ordering is
preserved (Spearman +0.80 between the two columns of four), 3S9D stays the
best complex and 2JEL the worst, but 1JTG triples (+0.17 -> +0.52) and 2JEL's
*anti*-predictive -0.34 relaxes to a flat -0.03. The mean improves from +0.11
to +0.25 and the per-complex sd falls from 0.33 to 0.29, yet the
four-complex t-test still does not clear 0.05. **Section 14's headline -- "what
noise gives at eight looks" -- is not overturned.** What is overturned is its
specific claim that the corrected map is actively anti-predictive on 2JEL; at
the mutation level it is merely null.

The pipelines differ in three ways and all three favour this one:

1. **Estimand.** Section 14 collapsed each position to one scalar via the
   simplex reduction and compared it against that position's max |DDG|. Here
   each measured substitution keeps its own identity, so the comparison
   includes within-position ranking instead of discarding it.
2. **n.** 15-50 positions there, 30-54 mutations here, on three of four
   complexes more.
3. **Artifact.** Section 14's 3S9D and 1JTG numbers predate the m=64 rung.

One oddity, recorded and not explained: `|B|` vs `|DDG|` is the only row whose
four-complex mean clears p=0.05 (+0.2542, p=0.022), and it does so by being
*consistent* (sd 0.12) rather than large. The signed version of the same
quantity is not significant. With four points this is a difference in variance
estimated from three degrees of freedom; do not build on it.

## 6. Does A vs C respect the bound set by B vs C?

**Numerically, no -- it is exceeded on three of four complexes. Statistically,
yes -- every CI overlaps.**

| complex | B vs C | A vs C (IG) | excess | CIs overlap | A vs B on these same rows | A vs B, full scan (sec 25) |
|---|---|---|---|---|---|---|
| 3S9D A | +0.4205 | +0.4678 | **+0.047** | yes | 0.7611 | 0.5937 |
| 1JTG B | +0.4376 | +0.5591 | **+0.122** | yes | 0.6952 | 0.6678 |
| 2JEL P | -0.1693 | +0.0385 | **+0.208** | yes | **0.1574** | 0.4042 |
| 3HFM Y | +0.2530 | -0.0049 | -0.258 | yes | 0.6618 | 0.4054 |

The bound "A vs C is bounded above by B vs C" is an informal data-processing
argument, not a theorem about finite-sample rank correlations. It would hold if
A were a deterministic function of B. It is not: A is a first-order Taylor term
that agrees with B at 0.16-0.76 on these rows, so A and B make *different*
errors against C, and at n=30-54 the sampling noise on a Spearman is roughly
+/-0.2. An excess of +0.05 to +0.12 is inside that noise, and the overlapping
CIs say so. **No claim in this section depends on A beating B**, and none
should: read the two columns as "both land near +0.45 on 3S9D and 1JTG and near
zero on 2JEL and 3HFM", which is what they do.

What the table does show, and is worth keeping, is the last two columns.
**On the SKEMPI subset of 2JEL, faithfulness itself collapses** -- A vs B is
0.1574 on these 44 mutations against 0.4042 on the full 608-row scan. 2JEL's
measured panel is only 14% alanine and is the one panel of unusual
substitutions; the attribution tracks the model well on the scan as a whole and
poorly on exactly the rows experiment chose. That is the cleanest available
explanation of 2JEL's null against lab, and it is a property of the subset, not
of the method.

## 7. Geometry

The partial on minimum heavy-atom distance to the partner chain, on every
correlation reported above, is in the tables. Summary of what it costs:

| complex | B vs C raw -> partial | A vs C (IG) raw -> partial | distance vs DDG | distance range | within 6 A |
|---|---|---|---|---|---|
| 3S9D A | +0.4205 -> +0.4199 (-0.1%) | +0.4678 -> +0.3964 (-15%) | -0.510 | 2.7-21.0 A | 24/54 |
| 1JTG B | +0.4376 -> **+0.5368** (+23%) | +0.5591 -> +0.5635 (+1%) | -0.298 | 2.6-12.6 A | 27/30 |
| 2JEL P | -0.1693 -> -0.2090 | +0.0385 -> +0.0754 | -0.576 | 2.6-24.0 A | 25/44 |
| 3HFM Y | +0.2530 -> +0.1921 (-24%) | -0.0049 -> -0.0748 | -0.206 | **2.3-3.9 A** | 42/42 |

Nothing here is geometry in disguise: the largest loss on a positive result is
15% (3S9D, IG), and 1JTG's partial is larger than its raw correlation.

**3HFM cannot test the confound at all.** All 42 measured mutations sit between
2.3 and 3.9 A of the antibody -- the same structural degeneracy section 24
flagged on Mpro. Its partial is reported for completeness and means nothing.
Section 14 already flagged 3HFM as unidentifiable for partials at 15 positions;
that flag stands, for a second and independent reason.

## 8. Refeaturise: there is no experiment to compare against

| complex | scan rows scored | **n joined with SKEMPI** | usable |
|---|---|---|---|
| 3S9D A | 103 / 950 | **0** | no |
| 1JTG B | 32 / 532 | **0** | no |
| 2JEL P | 70 / 608 | 18 | no (see below) |
| 3HFM Y | 29 / 285 | 3 | no |

The `token_to_rep_atom` guard drops 89-94% of mutants, and on 3S9D and 1JTG the
survivors contain **not one** SKEMPI-measured mutation. The mode cannot be
tested against experiment on this dataset at all; that is a statement about
coverage, not about the mode.

The one tempting number is 2JEL refeaturise, plain_grad vs DDG at **+0.7234**
(n=18, perm p=0.0010). **It is geometry and it should not be quoted.** On those
18 rows distance-to-partner correlates -0.938 with DDG and -0.741 with the
plain gradient; the partial drops the correlation to +0.607 with CI
[-0.060, +0.939], i.e. straddling zero. And on the same 18 rows A vs B is
-0.0815 -- the attribution does not track the model there at all, so it cannot
be tracking the model's biology. An attribution that agrees with the lab while
disagreeing with the model it was computed from is measuring the interface's
shape, which was already known.

## 9. Verdict

**Does Boltz-2's `complex_pde` scan track experimental binding energetics?
Partly, on some interfaces, and not reliably enough to use.**

- On 3S9D and 1JTG it does: +0.42 and +0.44, both clearing their own null,
  both surviving the geometry partial, +0.55 on 3S9D once the flat
  one-significant-figure rows are dropped. That is real signal and it is the
  first time this project has measured it on a brute-force scan.
- On 2JEL and 3HFM it does not: -0.17 and +0.25, neither clearing its null.
- Across four complexes the mean is +0.235 with p=0.19. **One cannot claim the
  model knows binding energetics from this.** One can claim it knows something
  on two of four interfaces.
- The gap is not the lab's fault. The experimental column reproduces itself at
  0.84-0.96; the model reaches about half of that at best.
- The shortcut inherits exactly this: A vs C is +0.47/+0.56/+0.04/-0.00 for IG,
  the same two complexes positive. **Attribution cannot be more right about
  biology than the model it explains**, and nothing in these numbers suggests
  it is -- the three apparent exceptions all sit inside their CIs.
- The practitioner-facing answer -- "what do I get from one gradient pass?" --
  is: on a favourable interface, a rank correlation with measured DDG of about
  0.5, which is competitive with the model's own 2,000-run scan on the same
  rows and roughly a tenth of the compute; on an unfavourable one, nothing, and
  **you cannot tell which you are on without the lab data that would have made
  the prediction unnecessary.**

## 10. Caveats, in the order a referee will raise them

1. **n is 30-54 per complex and 4 complexes at the top level.** The headline
   p-values (0.19, 0.17) come from t-tests on four numbers. The per-complex
   CIs are 0.4-0.7 wide. Nothing here settles anything; it bounds it.
2. **SKEMPI is alanine-dominated on the two complexes that worked** (87% and
   80% to-Ala on 3S9D and 1JTG) and not on the two that failed (14%, 31%).
   The positive results are therefore mostly "which positions matter", the
   question section 9 said alanine scanning can answer -- not "which of the 20
   amino acids", which is the project's actual claim. The one complex with a
   genuinely varied substitution panel, 2JEL, is the one that fails. **This
   confound is not controlled and cannot be with this data.**
3. **2JEL's experimental column is barely a ranking.** All 44 rows come from a
   single 1991 CSPRIA study with Kd quoted to one significant figure: 12
   distinct DDG values, 22 of 44 exactly zero, largest tie group 22, no
   replicates and therefore no measurable ceiling. A rank correlation on that
   column is close to uninformative in either direction, and 2JEL's -0.17
   should not be read as the model being wrong there.
4. **3HFM has no geometry contrast** (all 42 mutations within 3.9 A) and only
   15 positions. Its partial is decorative.
5. **The magnitude and signed analyses are not independent**, and neither are
   IG and plain_grad on the same complex. The tables report many numbers;
   9 of 24 clear their own null, which is more than noise would give, but no
   multiplicity correction is applied and none of the per-cell p-values should
   be read as a family-wise claim.
6. **The test-retest ceilings are estimated on replicated mutations only**,
   which skew to well-studied hot spots with a wide dynamic range. The true
   ceiling over the whole panel is probably lower, which makes the model's gap
   smaller than stated -- this caveat cuts in the model's favour.
7. **The join is at the mutation level, the distance confound is at the
   position level.** Substitutions at one position share a distance value, so
   the partial removes between-position geometry only. Within-position
   geometry does not exist, but within-position chemistry is uncontrolled.
8. **Refeaturise is untestable against experiment here** (section 8) and the
   coordinate-consistent control section 25 called "the next thing that
   matters" is still not run. Nothing in this section speaks to it.

## 11. Reproducing

CPU only, ~15 s per complex.

```
python scripts/26_vs_experiment.py --dataset 3S9D --chain A \
  --scan results/3S9D_complex_pde_scan.csv --mode resswap \
  --attribution ig_m64=results/3S9D_complex_pde_ig_grad.npz \
  --attribution plain_grad=results/3S9D_complex_pde_plain_grad_grad.npz
```

Same for `1JTG B` (adding `ig_m32=results/1JTG_complex_pde_ig_grad_m32.npz`),
`2JEL P` and `3HFM Y` (both with `ig_m32=..._ig_grad_m32.npz`), and for each
with `--mode refeaturise --scan results/{ds}_complex_pde_scan_refeaturise.csv`.

Outputs: `results/{dataset}_complex_pde_vs_experiment_{mode}.json` plus
`.prov.json` sidecars, eight of each. Tests: `tests/test_vs_experiment.py`,
44 cases, including an inverted-scan case that asserts the orientation anchor
catches a flipped sign and a case asserting stage 26's `LOWER_IS_BETTER` is
identical to stage 12's.
