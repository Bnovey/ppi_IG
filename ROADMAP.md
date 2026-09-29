# Roadmap — gradient attribution on Boltz-2, measured against experiment

Two validation arms. **Alanine scanning** (SKEMPI, sections 1-8) asks which
*positions* matter and is confounded by side-chain size; **saturation**
(Starr 2020 on 6M0J, section 10) asks how each of the 20 *amino acids*
performs and is not. Section 9 is the evidence for moving to the second.

Supersedes the target selection in `HANDOFF.md`. `PLAN.md` is the original
proposal and must not be modified; this document records what changed and why.

---

## 1. The claim

> ~32 runs of Boltz-2 can predict how every possible mutation changes binding at
> a protein-protein interface — closely enough to replace the ~2,400 full runs it
> would otherwise take, and closely enough to match measured binding energies.

This is a validation project. A clean negative is a result.

### The three comparisons, in plain terms

The repo previously called these T1/T2/T3. Use the plain names.

| Name | Compares | Answers |
|---|---|---|
| **shortcut vs slow way** | attribution vs Boltz-2's own brute-force scan | Does the shortcut reproduce the model? |
| **model vs lab** | Boltz-2's scan vs measured ddG | Is the model right about biology? |
| **shortcut vs lab** | attribution vs measured ddG | What does a practitioner get? |

The first is the honest test of the shortcut: it holds whether or not Boltz-2
knows any biology. The third is bounded above by the second.

---

## 2. Why the target changed

### AbBiBench's `4fqi_h1` is the wrong shape

Characterised directly from `data/raw/4fqi_h1_benchmarking_data.csv`:

- 65,094 variants over exactly **16 positions** of 121, one alternative amino
  acid each
- modal variant carries **8 simultaneous mutations**; only 16 single mutants
  exist in the whole file
- scores are **left-censored**, 1,675 variants piled at the assay floor of 7.000

It cannot validate a per-amino-acid prediction, because it never varies the
amino acids. Spearman(mutation count, score) = **-0.22**, so Boltz-2's score
there is genuine per-position signal rather than a mutation-count artifact --
the dataset is sound, just narrow.

### A number in `HANDOFF.md` was wrong

`HANDOFF.md:252` claims Boltz-2 scores 0.71 on `4fqi_h1`. From the published
leaderboard (`github.com/MSBMI-SAFE/AbBiBench`, README):

| Dataset | Boltz-2 | ProteinMPNN | ESM-IF1 | FoldX |
|---|---|---|---|---|
| `3gbn_h1` | **0.71** | 0.59 | 0.59 | 0.59 |
| `4fqi_h1` | **0.40** | 0.61 | 0.65 | 0.64 |

The 0.71 belongs to `3gbn_h1`. On `4fqi_h1` Boltz-2 is beaten by every cheap
alternative. We had picked as go/no-go a dataset where the model under study
loses.

Also worth recording: AbBiBench scores every model by **log-likelihood of the
sequence given the structure**, and there is no Boltz-2 code in their public
repo (14 models have `get_model_log_likelihood.py`; Boltz-2 and AF3 do not). So
what quantity they extracted from Boltz-2 is not verifiable, and our
`complex_pde` may not be comparable to their 0.13. Do not claim comparability
without checking.

### SKEMPI 2.0 has what we need

7,085 entries over 348 complexes, **5,112 single-point**, **2,961 to alanine**,
with wild-type and mutant Kd. Gives per-position *and* per-amino-acid
resolution, continuous ddG in kcal/mol, and it is not blocked by the `ref_pos`
defect (see section 4).

---

## 3. Target: 1JTG, with 3HFM as the second complex

Ranked by usable depth, all SKEMPI single-point mutations with computable ddG:

| Complex | Positions | Mutations | Hot spots | aa/position | Partners |
|---|---|---|---|---|---|
| **1JTG** | **49** | 138 | **13** | 2.8 | TEM-1 beta-lactamase / BLIP |
| 3S9D | 75 | 156 | 10 | 2.1 | Interferon alpha-2 / receptor |
| 1A22 | 133 | 212 | 7 | 1.6 | Human growth hormone / receptor |
| 1AO7 | 60 | 136 | 11 | 2.3 | TCR / pMHC |
| 3HFM | 27 | 96 | 13 | 3.6 | HyHEL-10 / HEW lysozyme |

**Primary: `1JTG`.** Nearly double 3HFM's positions, the same 13 hot spots,
almost the same per-amino-acid depth, and *smaller* once the duplicate chains
are dropped.

**Secondary: `3HFM`**, plus `1VFB` / `1JRH` / `2JEL` if the pooled hot-spot arm
needs more positions (27 alone is thin).

### 1JTG setup facts, verified

- **Four chains: A(262) B(165) C(261) D(165), total 853 tokens.** A/B and C/D
  are two copies of the complex in the asymmetric unit. **Use A+B only = 427
  tokens**, below 3HFM's 558 and well below `4fqi_hlab`'s 730 (which needs
  55.2 GiB in bf16). fp32 may fit.
- Mutation split: chain A (beta-lactamase) 42 muts / 21 positions / 2 hot spots;
  **chain B (BLIP, the inhibitor) 96 muts / 28 positions / 11 hot spots**. Chain
  B is where the signal is.
- **138/138 mutations map to residue indices with zero wild-type mismatches**,
  using the correct numbering column.

### The numbering bug -- fix before anything else

`scripts/10_skempi_hotspots.py:97` reads `Mutation(s)_cleaned`. That is
*sequential* numbering. `Mutation(s)_PDB` is what matches a deposited structure.

| Complex | Rows where the two columns agree |
|---|---|
| 3HFM | 96/96 |
| 1VFB | 56/56 |
| 1JTG | 96/138 |
| 1A22 | 54/212 |
| **3S9D** | **0/156** |

On 1JTG, `cleaned` gives 41 wild-type mismatches; `PDB` gives zero. 3HFM passed
only because its two schemes coincide -- that clean 96/96 was luck, not
validation. The wild-type guard caught this, which is the whole reason it exists
(cf. `ERRORS_LOG.md` entry 17).

---

## 4. Blockers

### `ref_pos` nondeterminism -- RESOLVED and VERIFIED 2026-09-24

Was: boltz featurisation differed by up to **10.7 Angstrom** in `ref_pos`
between two runs of the identical input, contaminating every `s_mut - s_ref`
delta and the brute-force scan alike.

Root cause was **not** the RDKit conformer generation the original entry blamed.
It is `featurizerv2.py:1467-1473`, which applies a random rotation and
translation per residue via `torch.randn`, while `inferencev2.py:270` seeds numpy
only and never calls `torch.manual_seed`. `src/igv/deterministic.py` patches
`center_random_augmentation` to `augmentation=False` -- keeping the centring,
dropping the roto-translation, which is training-time augmentation with no place
at inference.

**Verified on the VM**, not inferred: 78/78 feature tensors byte-identical across
two runs, and reference-vs-mutant `ref_pos` changes confined to the mutated token
with all 421 others identical. `ERRORS_LOG.md` entries 20 and 18.

### Substitution enumeration -- RESOLVED

`--positions skempi|<list>` enumerates all 19 substitutions at chosen positions;
the `assert n_deltas < 500` is now `--max-substitutions` (default 5000). 1JTG
chain B gives 532 substitutions at 28 positions.

### Stages 02 and 03 featurisation -- PARTIALLY ADDRESSED

Both now key their boltz cache on dataset (entry 24), so neither can silently
receive another complex's features. But stage 03 still featurises with the MSA
server while stage 02's reference path differs, so confirm they expand around the
same point before trusting a delta-based prediction.

### The 2.0 tolerance in `02_embed_deltas.py` -- can now be tightened

It was set against a measured ~1.5 noise scale that no longer exists. With
featurisation deterministic, this should return to something strict.

---

## 5. Plan

### Phase 0 -- no GPU -- ALL DONE except item 7

1. ~~Switch to `Mutation(s)_PDB`~~ done, with a regression test.
2. ~~Seed conformer generation~~ done and verified on the VM (section 4).
3. ~~Enumerate all 19 substitutions~~ done, `--positions` + `--max-substitutions`.
4. ~~Chain subsetting for 1JTG~~ done, 427 tokens, C/D excluded.
5. ~~Confound controls~~ done, and they set a bar the attribution has not cleared
   (section 6).
6. ~~Fix `HANDOFF.md` / `README.md`~~ done.
7. Characterise the remaining 16 AbBiBench datasets -- **in progress**.

### Phase 1 -- precheck (~$1)

One `plain_grad` backward pass on 1JTG chain B. Cheap look for any signal before
committing to longer runs. Go/no-go.

### Phase 2 -- IG (~$10)

`m=32` with a convergence check at 16 and 64, so "32 was enough" is demonstrated
rather than asserted. Produces the per-amino-acid grid.

Completeness is **load-bearing here**: `integrated_gradient` returns
`grad = accumulated_grads`, the path-averaged gradient, and `score_deltas` uses
exactly that array (`src/igv/attrib.py:270-277`, `:393-404`). If the quadrature
under-resolves, every prediction is wrong. m=32 passed at L=554 (0.0442);
1JTG A+B is 427 tokens, so it should hold comfortably.

### Phase 3 -- brute force (~$13)

Scan **all 19 substitutions at the 28 measured positions = 532 mutants**. The
"shortcut vs slow way" comparison needs no lab data, only the model, so scan as
widely as affordable. A full saturation scan of chain B (165 x 19 = 3,135) is
the eventual ideal at roughly another 20 GPU-hours -- hold it until Phase 1
shows something.

Corrected 2026-09-27: this section previously said 49 positions and 931
mutants. `skempi_positions("1JTG", "B")` returns **28**, so the real figure is
532 and the phase is ~43% cheaper than first estimated. Measured, not
re-estimated -- `python scripts/04_scan.py --dataset 1JTG --chain B --positions
skempi --dry-run` prints it.

### Phase 4 -- compare -- IMPLEMENTED 2026-09-27

`scripts/12_compare.py`. Until today this phase had no implementation and
stage 04's scan CSV was written and read by nothing, so the most expensive GPU
stage in the pipeline produced an artifact that was never compared to anything.
See entry 29.

It reports the triple that is the actual argument -- attribution vs experiment,
scan vs experiment, attribution vs scan -- plus the within-position split on a
DMS dataset and the forward-versus-backward pass counts, which is the practical
case for replacement. Score orientation is derived from the score name, since
`complex_pde` is lower-is-better and the other five are not; verified by feeding
a scan that agrees perfectly with experiment under each score's own semantics
and confirming both report positive.

All on the same mutations: shortcut vs slow way; shortcut vs lab; model vs lab;
and IG vs the single-pass precheck. That last one is a finding either way -- if
they agree the expensive version is unnecessary, and if they diverge the
function is strongly non-linear between baseline and input, which the alpha
profile already suggests (53% of the score change in the first 5% of the path).

### Cost

Anchored on the one measured number, 104.2 s per backward at 730 tokens. 1JTG
A+B is 427 tokens, so passes are cheaper than the 3HFM estimates. Roughly 12
GPU-hours core, ~18 with realistic slack for the OOMs and reruns this project
has hit in every prior session. At $5.07/hr that is **$60-90**.

The saturation arm (section 10) is on top of that. 6M0J is ~791 tokens
against 1JTG's 427, so budget ~1.5-3 h per 32-step IG run rather than ~40
min, and size the position set to the budget rather than running all ~200.

---

## 6. Controls we owe

From Adebayo (*Sanity Checks for Saliency Maps*, NeurIPS 2018) and from
arXiv:2606.22181, whose failure mode is the one most likely to bite us.

**Already passing** (at L=554): `random_weights` rho=0.077 -- randomising 5,035
parameter tensors destroys the attribution, so the signal depends on learned
weights. `dead_target` 0.0.

**The confound panel -- built, and it has already set the bar.** The
allergenicity paper found their classifiers relied on *"physicochemical and
compositional sequence features rather than epitope-specific mechanisms."*
Measured on **1JTG chain B**, 28 positions, from the PDB alone with no model
involved (`ERRORS_LOG.md` entry 19):

| Confound | Spearman vs ddG |
|---|---|
| **burial** (heavy atoms within 10 A) | **+0.54** |
| **residue volume** | **+0.48** |
| distance to binding partner | -0.32 |
| hydrophobicity | -0.14 |
| normalised position in chain | +0.11 |

**Counting nearby heavy atoms predicts binding energy at rho ~0.5.** An
attribution scoring 0.5 against ddG tells us nothing a distance calculation
would not. So the **partial correlation** -- does the gradient predict ddG after
burial and residue size are regressed out? -- is the load-bearing number, not
the raw Spearman. Report both.

Two caveats on that bar. It is sensitive to aggregation (+0.54 max signed, +0.59
max |ddG|, +0.47 mean), so always state which. And its bootstrap 95% CI is
**[+0.21, +0.78]** at n=28 -- wide enough that pooling 3HFM, 1VFB, 1JRH and 2JEL
moves from optional to close to mandatory before any claim is made. Distance to
partner looks weak only because SKEMPI measured interface residues exclusively,
so the variable cannot discriminate; that is range restriction, not evidence the
interface is unimportant.

`scripts/10_skempi_hotspots.py` now reports the panel, the partial correlation,
AUROC / AUPRC alongside precision@k, a shuffled-ranking null, and bootstrap CIs.
Matching arXiv:2606.22181's metrics makes the comparison to them direct.

---

## 7. Positioning

Do **not** claim first-to-validate, and do not aim at the leaderboard. IG
approximates Boltz-2's own scan, so "shortcut vs lab" is bounded by Boltz-2's
0.13 average -- ProteinMPNN's 0.30 is out of reach by construction, and
ProteinMPNN is also already fast, so the speed argument does not beat it either.
What the shortcut is 77x faster than is brute-force scanning *with a co-folding
model*.

**arXiv:2606.22181** (ICML 2026 workshop) is commonly miscited, including in
this repo, as "gradients on proteins, negative". Read it properly: they
benchmarked attribution on protein *allergenicity* classifiers against annotated
epitopes, and found

> *"Integrated Gradients identified residues that were functionally important to
> the model, but not overlapping annotated epitopes"*

That is the attribution **working** and the model relying on compositional
shortcuts -- the "shortcut is honest, model is wrong" quadrant. It supports this
work rather than contradicting it. Their ground truth was binary epitope
annotations and in-silico mutagenesis; ours is measured ddG in kcal/mol. Frame
as: *attribution on sequence-only models recovered model-internal features but
not biology; does a model that explicitly builds the interface do better, judged
against binding energies rather than annotations?*

Others to engage: **TISM** (iScience 2024, the method, genomics only, never
compared to experiment, Pearson 0.70 is the number to beat), **ProtDBench**,
**BindEnergyCraft**, **GearBind**, **Majdandzic** (Genome Biology 2023,
off-simplex contamination), the **AF3/SKEMPI** paper (NeurIPS 2024).

Venue: MLSB, 5 pages excluding references, welcomes work in progress.

---

## 8. Status

Committed through `4962fbc`. **549 tests, ruff 11.** Working tree clean on
`main`. VM `igv-gpu` TERMINATED, 500 GB pd-ssd retained at ~$2.83/day.

**No GPU result is newer than 2026-09-24.** The 2026-09-27 session was entirely
local: it built the saturation arm, fixed four pipeline defects, and never ran
a model. The table below is unchanged and is still the only measured science in
the project.

### Measured on 1JTG chain B, 2026-09-24 (427 tokens, fp32, ~$24 of A100)

| run | Spearman | null p95 | above null | partial | AUROC |
|---|---|---|---|---|---|
| plain_grad (zeros) | 0.306 | 0.318 | no | 0.180 | 0.685 |
| IG m=32 (zeros) | 0.238 | 0.336 | no | 0.010 | 0.583 |
| **IG m=32 (mean_aa)** | **0.357** | 0.313 | **YES** | 0.211 | **0.695** |

**The baseline was the cause of the early nulls.** `complex_pde` is a predicted
distance error; the all-zeros baseline is not a protein and scores 11.4310 where
the real complex scores 2.6200, so the integral was dominated by off-manifold
behaviour. `--baseline mean_aa` drops f(baseline) to 3.2491 and cuts the
integral's **absolute** error 30x (1.5222 -> 0.0504). See `ERRORS_LOG.md`
entry 21.

**Do not oversell the positive.** CI [-0.002, 0.656], excluding zero by two
thousandths. Partial correlation 0.211 (CI [-0.34, +0.56]) against burial's
0.436. Gradient-vs-hydrophobicity is -0.527 with CI [-0.75, -0.18], which
excludes zero while gradient-vs-ddG barely does.

**Determinism is proven, not inferred.** `verify_deterministic_feats.py` exits 0
on the VM: 78/78 tensors byte-identical across two runs, and reference-vs-mutant
`ref_pos` changes confined to the mutated token with all 421 others identical.
Entry 18 is closed (entry 20).

**Also settled:** path integration is not a rounding error on the plain
gradient — Spearman +0.58 between the two per-residue rankings, only 3 of the
top 10 residues shared, IG norms 3.6x larger. `plain_grad` is not a cheap
substitute for `ig`.

**Still not done:** the brute-force scan has never run, so "shortcut vs slow
way" — the headline claim — has no number. Completeness fails the 5% *relative*
gate at 8.01%, though see entry 21 on why that gate is miscalibrated for a tight
baseline.

VM `igv-gpu` is TERMINATED; the 500 GB pd-ssd is retained at ~$2.83/day.

---

## 9. The data problem, and where it points

**Alanine scanning cannot validate the project's actual claim.** SKEMPI is ~90%
X->Alanine, so it tests *which positions matter*, not *how each of the 20 amino
acids performs*. Worse, the confound is mechanistic rather than statistical:
1JTG chain B's 11 hot spots are W, K, H, F, H, Y, W, R, F, D, W — all large
aromatic or charged residues — and X->A *deletes a large side chain*. Residue
volume correlating with ddG at +0.44 is therefore what the assay measures, not
noise to regress away. Any method that learns "large aromatic at an interface
matters" scores respectably without representing binding. See entry 22.

**SKEMPI's counts are replicates, not breadth.** 1JTG chain B's 96 mutations are
28 distinct mutations measured repeatedly; W150's five W->A values span
+3.50..+4.81, putting experimental noise at ~±0.6 kcal/mol. Effective n is 28,
which is why every CI above is so wide. Entry 23.

**Saturation data is the fix, and it is measured, not argued.** With all 20
substitutions at one position, the comparison becomes *within-position*, so
burial, exposure and distance-to-partner are constant across the row and cancel
identically. Verified on real saturation data (GB1, below):

| Design | Residue volume vs binding effect |
|---|---|
| Pooled (what we did on SKEMPI) | **+0.44** |
| Within-position | **+0.135** mean, **+0.197** median |

And the within-position correlation *changes sign* position to position — 20 of
54 positions negative, range -0.92..+0.71 — so on pooling it cancels rather
than accumulating. The confound panel is not something to regress away; it is
something the experimental design removes. This is the single most important
methodological result in the project so far.

### GB1/IgG-Fc — assessed and rejected

Olson et al. 2014 (Current Biology). Verified locally: 1045 singles = exactly
55 positions x 19 substitutions, every position saturated. It is cheap (56
residues) and it is what produced the within-position numbers above. Rejected
as the primary saturation target on four counts:

1. **The hot spots are censored out.** The floor is exactly `ln(0.01)`, with 87
   values pinned there. Position 27 has **std 0.000 — all 19 substitutions at
   the floor**; position 43 (Trp) has 17/19. 13 of 55 positions are flat
   (std < 0.3). Within-position ranking is impossible at precisely the residues
   alanine scanning calls hottest, which are the ones we most need to rank.
2. **Folding and binding are coupled.** The authors define `W = f_N * W_N`, so a
   destabilised variant reads as a non-binder. For a *structure* model like
   Boltz-2, "the gradient tracks foldability" is an a priori likely confound,
   which makes a stability control mandatory rather than optional. GB1 needs a
   cross-paper merge (Nisthal 2019 PNAS folding ddG) to approximate one.
3. **The structure is the wrong paralogue.** 1FCC is protein G **C2** domain
   (P19909), not B1 (P06654); 3 of 56 positions differ, giving 57 wild-type
   mismatches, and 988/1045 mutants map cleanly. No better structure exists in
   the PDB. Numbering offset is exactly -226.
4. **`lnW` is not ddG**, so GB1 results cannot be pooled with, or plotted on the
   same axis as, anything from SKEMPI.

Structural trap recorded in case GB1 is ever revisited: in 1FCC chains A and B
are an **obligate Fc homodimer 2.34 A apart**, not redundant copies like 1JTG's
C/D. Dropping B to save tokens models a half-molecule that does not exist in
solution. Use A+B+C = 468 tokens, drop only D.

### AbBiBench saturation scans — why not

`2fjg` and `g6_LC` *are* complete saturation scans — `2fjg` verified at 2223
rows = 117 positions x 19, all singles, consensus == PDB wild type. The problem
is the other side of the ledger: Boltz-2's own per-dataset score on `2fjg` is
**0.08**. Across AbBiBench, Boltz-2's score correlates *inversely* with how much
per-amino-acid structure a dataset has — 0.71 on `3gbn_h1`, which has 11
positions x 1 alternative, no true singles, and an 11% floor. The datasets
Boltz-2 looks good on are the ones that cannot test our claim. Recorded so the
0.71 is never cited as encouragement.

## 10. The saturation arm: SARS-CoV-2 RBD / ACE2

**Decision (2026-09-27): go straight to Starr et al. 2020 on 6M0J. Skip GB1
entirely.** The within-position machinery has to be written either way, and
writing it against the dataset that can actually carry the claim avoids
building it twice. GB1's cheapness does not offset a censored hot-spot set, a
wrong-paralogue structure, and a readout that cannot be put on the same axis as
our SKEMPI results.

`SPIKE_SARS2_Starr_2020_binding` — Tite-seq on the RBD, all 20 amino acids at
essentially every position.

| | GB1 | **SPIKE/6M0J** |
|---|---|---|
| Singles | 1045 | **~3802** (~99.6% saturation) |
| Readout | enrichment ratio (`lnW`) | **delta log10 KD — real affinity** |
| Convertible to ddG? | no | **yes**, ~1.36 kcal/mol per log10 |
| Stability control | cross-paper merge needed | **same paper, same assay, same variants** |
| Structure | wrong paralogue, 3.2 A | 6M0J, 2.45 A, single copy |
| Tokens | 468 | ~791 |

The matched **expression** readout is the decisive advantage. It is the
folding-vs-binding decomposition GB1 needs a separate publication to
approximate, available for the identical variants in the identical assay. Since
the likeliest failure mode for a structure model's gradient is that it tracks
foldability, having that control in-file converts a fatal ambiguity into a
measurable one.

Cost is the only objection: 791 tokens against 1JTG's 427, so roughly 1.5-3 h
per 32-step IG run rather than ~40 min.

### Setup facts — VERIFIED 2026-09-27, independently of the loader

- Dataset: 4221 rows, of which **3802 singles** over 201 sites (331-531). 198 of
  201 sites carry all 19 substitutions; the other three carry 15, 14 and 11.
- The `raw.githubusercontent.com` path returns a **Git LFS pointer, not the CSV**.
  Use the `media.githubusercontent.com/media/...` URL.
- 6M0J chains: **A = ACE2, 597 residues; E = RBD, 194 residues**, author numbering
  333-526 and contiguous. Total **791 tokens**. No other protein chains. HETATM
  entities our parser drops: Zn, Cl, NAG, HOH.
- **Zero wild-type mismatches** across all 194 mapped sites. Contrast 1JTG, where
  the wrong numbering column gave 41 silent mismatches (entry 20).
- Chain E stops at 526, so sites 331, 332 and 527-531 have no structure.
  **133 singles are unusable; n = 3669 over 194 positions.**
- Sign convention confirmed in code: `binding_ddg(-1.0) = +1.36` kcal/mol, i.e.
  weaker binding maps to positive ddG, matching SKEMPI.
- **Censoring is not a problem.** `bind_avg` has one row at its minimum and 146
  within 0.1 of it, spread over 40 positions; only site 497 has as many as 10 of
  19 near the floor. Nothing resembling GB1's position 27 (19/19 pinned).
  `expr_avg` has exactly one row within 0.1 of its minimum.

### The within-position design, measured on this dataset

| Confound, within-position | mean | median | negative |
|---|---|---|---|
| Residue volume vs binding | **-0.034** | -0.045 | 109/201 |
| Hydrophobicity vs binding | **-0.006** | -0.084 | 108/201 |

Pooled on this dataset, volume vs binding is **+0.027** and volume vs |binding|
is **-0.020**. So the +0.44 measured on SKEMPI is not a property of proteins; it
is a property of the *alanine-scan design*, which confirms entry 22 by
construction rather than by argument.

### Folding vs binding: the confound is real, and it is confined to where it does not matter

This is the reason to use this dataset, and the global number is misleading.
Pooled, binding and expression correlate at **+0.64** (r2 0.40), and
within-position the median is **+0.75** — a structure model whose gradient
tracks foldability would score well on binding for the wrong reason. But split
by distance to ACE2:

| | n positions | bind~expr median | bind std | expr std |
|---|---|---|---|---|
| **Interface (<=5 A)** | 21 | **+0.38** | **0.695** | 0.275 |
| Non-interface | 173 | +0.752 | 0.338 | 0.481 |

Pooled over interface mutants only, binding and expression correlate at
**+0.074**, and residualising binding on expression keeps **100%** of the
binding variance (1.224 -> 1.221). At the interface the two readouts are
effectively orthogonal: binding varies most and expression varies least, while
away from the interface the reverse holds. The folding confound is therefore
not something to fight — it lives almost entirely outside the region the claim
is about, and the expression column lets us demonstrate that rather than assume
it.

**Consequence: the primary analysis is interface positions only, within
position, with expression reported as a control.** Also note 80 of 194 positions
have binding std < 0.3 and carry no rankable signal; the interface median is
0.695.

### Two calibration numbers, measured through stage 11 on 2026-09-27

**The volume confound is only *mostly* cancelled on the 21 positions we will
actually analyse.** The headline -0.034 is the mean over all 194 positions. On
the interface subset alone it is **+0.092 mean, +0.245 median**, CI
[-0.082, +0.266]. The CI includes zero and n is 21, so this is not a finding —
but it is not the -0.034 either, and the interface number is the one that
belongs next to any interface result. Report it.

**A *perfect* binding predictor scores +0.355 against expression.** Feeding
stage 11 the observed `bind_avg` as its own prediction gives
`pred_vs_bind = +1.000` by construction and `pred_vs_expr = **+0.355**`, because
binding and expression are themselves correlated at +0.38 within interface
positions. So a gradient correlating ~0.35 with expression is **not** evidence
of a folding confound — it is exactly what a pure binding predictor does here.
The diagnostic is the *gap*: `pred_vs_bind` clearly above `pred_vs_expr` is the
good case; the two roughly equal is the bad case. Without this reference value
a perfectly clean result could be read as contaminated.

### Position sets, and why only brute force pays for them

| Cutoff to ACE2 | Positions | Mutants |
|---|---|---|
| <= 4 A | 17 | 323 |
| **<= 5 A** | **21** | **399** |
| <= 6 A | 30 | 570 |
| <= 8 A | 46 | 874 |
| <= 10 A | 60 | 1140 |

All 17 literature ACE2 contact residues fall inside the 5 A set, including the
variant-of-concern positions **K417, E484, N501**, which is independent evidence
the cutoff is picking out the real interface rather than an arbitrary shell.

One IG run yields attributions for **every** position at once, so the position
count is free on the gradient side — 791 tokens is the whole cost. Only the
brute-force arm scales with positions, at 19 forward passes each: **399 mutant
predictions for the 5 A set**. Start there; widen to 8 A only if 21 positions
proves too few to separate the methods.

### Next, in priority order

1. **Pool 3HFM, 1VFB, 1JRH, 2JEL** (~$12, ~2.5 h). All registered and verified.
   Takes n from 28 to ~128 and directly tests whether 0.357 survives
   replication. Do this **before** the scan: if the effect does not reproduce,
   0.357 was n=28 noise and we learn it cheaply.
2. **The brute-force scan on 1JTG** (~$22, ~4 h). Mandatory — the headline claim
   is unpublishable without it, and it separates "the method fails" from
   "Boltz-2 has no binding signal here". Only one of those is fixable.
3. **The saturation arm on 6M0J.** Dataset and structure are verified (above);
   `src/igv/dms.py` loads them. Remaining local work: wire the 21 interface
   positions into stages 02/03 and add the within-position scoring to
   `06_metrics.py`. Then one IG run at 791 tokens gives all 194 positions x 20,
   and the brute-force comparison is 399 forward passes.
4. **Re-gate completeness** on absolute error -- **DONE 2026-09-29**, see
   `ERRORS_LOG.md` entry 32. Gates on `COMPLETENESS_ABS_THRESHOLD = 0.10`; the
   8.01% relative "failure" was the gate punishing the better baseline, and it
   blocked nothing.

Not yet justified: a different attribution method. Item 2 is what establishes
whether the method is at fault.

**Superseded in part by section 11**, which revises this list against the
literature and against a re-analysis of the saved 1JTG artifact.

---

## 11. What the field already knows, and what our own data says

Surveyed 2026-09-29, no GPU. Two things came out of it: the bar is better
defined than we thought, and our own saved artifact says something we had not
looked for.

### The baselines we are actually measured against

**RSALOR** -- Tsishyn, Hermans, Rooman & Pucci, *Bioinformatics* 41(6):btaf322
(2025), "Residue conservation and solvent accessibility are (almost) all you
need". `(1 - RSA) x LOR`, two features, **zero trainable parameters**. Average
Spearman **0.473** across 217 ProteinGym DMS datasets, matching or beating 27
deep predictors including ESM-2, SaProt, EVE and GEMME. **RSA alone scores
0.356.** Our confound panel (`src/igv/skempi.py:519`) has burial, distance,
hydrophobicity, volume and normalised position -- but **no conservation term**,
so we currently hold only half of the baseline the field will hold us to.

**SKEMPI per-interface Spearman**, the protocol that matters. Pooled numbers run
~0.3 higher because between-complex variance dominates, so any comparison must
state its protocol:

| Method | Per-interface rho |
|---|---|
| FoldX | 0.37 (name-split) / 0.48 (cluster-bootstrapped) |
| Flex ddG | 0.42 |
| Best learned, leakage-controlled (ProSST+ProtBFF) | 0.48 |
| Best learned, name-split (BA-DDG) | 0.51 |
| B-factor baseline | 0.169 |
| ESM-1v | **-0.012** |

Two further floors: a trivial predictor returning the mean ddG **for the
mutation type alone**, no structure at all, reaches Pearson **0.46**; and the
experimental noise ceiling is Pearson **~0.89**, so published 0.91s are overfit
by construction. Sequence-only pLMs are at zero per-structure -- that is the
company we are *not* in, and worth saying explicitly.

### The negative result closest to ours

Yao, Song, Baerenfaller & Zhakparov, arXiv:2606.22181 -- already cited in
section 6 as the failure mode most likely to bite us, and it is. Six attribution
signals including IG against IEDB epitopes: **no model-derived attribution
exceeded its random baseline**; IG scored AUROC **0.476 against random 0.501**.
But IG *passed* their faithfulness test -- masking top-IG residues moved the
prediction, p<0.001. **Model faithfulness and biological faithfulness fully
dissociate.** Our `random_weights` control (entry 15) tests the first; SKEMPI
tests the second. Report them separately and name them as such.

Also relevant: King et al., arXiv:2512.06592 (MLSB 2025) fine-tuned Boltz-2 for
protein-protein affinity and found it "underperforms relative to sequence-based
alternatives in both small- and larger-scale data regimes". Combined with the
affinity head having been trained predominantly on protein-ligand data, whether
it carries PPI energetic signal at all is a prior question we should state up
front rather than have a reviewer raise it.

### What the niche looks like

Three independent searches found **no published work applying gradient
attribution to a co-folding model's affinity head** -- not Boltz-2, not
AlphaFold3, not Chai-1. ExplainableFold (KDD 2023) is the nearest neighbour and
is counterfactual, on structure rather than affinity. PairSAE (arXiv:2606.27440)
is on Boltz-2 but uses sparse autoencoders and no gradients. The genomics
community solved the discrete-input attribution problem years ago -- shuffled
references, the simplex correction, ISM validation -- and none of it has been
carried into structure prediction. **Carrying it over carefully is itself the
contribution.**

### Measured on our own artifact, 2026-09-29

From `results/1JTG_hotspots_ig_meanaa.json`, n=28 positions, all against
**|ddG|** unless stated. Recorded because two earlier readings of these numbers
were wrong in opposite directions.

- **Burial does not significantly beat IG.** Burial 0.436, IG 0.357. Paired
  bootstrap of the difference over 20,000 resamples: **+0.199, CI95
  [-0.193, +0.600], P(burial > IG) = 0.84.** The burial figure of +0.54 quoted
  in section 8 is against *signed* ddG; against |ddG| it is 0.436. Compare
  like with like.
- **They are not measuring the same thing.** Spearman(IG, burial) = **+0.315**.
- **Their partial correlations are near-identical:** IG controlling for burial
  **+0.245**; burial controlling for IG **+0.256**. Neither is redundant.
- **Combined they reach +0.508** (rank sum), which sits at FoldX's
  cluster-bootstrapped per-interface 0.48 and the best leakage-controlled
  learned model's 0.477. Gain from adding IG to burial: **+0.071, CI95
  [-0.182, +0.304]**, not significant at n=28.

**This reframes the question.** It was "can the attribution beat a ruler", and
the answer looked like no. The better question is "does the attribution carry
energetic information that geometry does not", and the partial correlations say
plausibly yes. It also matches the strongest result in the structure-model
literature: ProtBFF's win comes from *injecting* burial and interface features
into a learned model, not from either alone -- its ablation shows interface and
burial are the two largest single contributors.

**Every number in this subsection has a CI spanning zero or nearly so.** This is
a hypothesis worth testing, not a result. The test is n, not method.

### Revised priorities

1. **Pool 3HFM, 1VFB, 1JRH, 2JEL** (~$12, ~2.5 h). Unchanged as item 1, and
   now better motivated: it tests the complementarity finding above as well as
   whether 0.357 replicates.
2. **Add a conservation term to the confound panel.** CPU only, one MSA fetch,
   no A100. Without it we cannot state how we do against RSALOR, which is the
   comparison the field will make first.
3. **Wire the simplex path** (entry 31) on the VM -- it needs boltz installed to
   backpropagate to `res_type`. Until then stage 10 still uses the L2 norm and
   the fix is not live.
4. **The brute-force scan on 1JTG** (~$22, ~4 h) -- unchanged, still the only
   thing that answers the headline claim.
5. **The alpha-profile diagnostic** (`--baseline` now exists, entry 32): plot
   `F(x' + a(x - x'))` for both baselines and report `F(baseline)`. Turns "we
   tried two baselines and one worked" into a mechanistic explanation. Cheap,
   but it does need GPU -- the only saved profile data is 4fqi at L=554.

Still not justified: a different attribution *method*. Items 1 and 4 establish
whether the method is at fault, and the reduction fix in entry 31 has not yet
been measured.

**Superseded as the focus by section 12 (Phase 5), decided 2026-09-29.** The
list above is all position-level work validated at n=28, where nothing survives
its own CI. Phase 5 asks a pair-level question at n=76 on data already on disk,
and rides the same backward pass as item 1. Items 1 and 2 remain worth doing and
are unchanged in cost; they are no longer the front of the queue.

---

## 12. Phase 5 — attribute to the pair representation, and predict residue coupling

Proposed 2026-09-29. **This is the new focus.** Everything in sections 8 and 11
attributes one scalar per residue and validates it against single-mutant ddG at
n=28, where nothing survives its own confidence interval. This phase asks a
different question, on a sample nearly three times larger, using data already
on disk and a gradient that falls out of a backward pass we already run.

### The idea

Binding is not a property of residues. It is a property of residue **pairs**,
and Boltz-2's central object is the pair tensor `z`, shape `(1, L, L, 128)`.
Every attribution in this project so far -- and, as far as three independent
literature sweeps found, every attribution published on any protein model --
collapses to one number per position. The gradient with respect to `z` gives an
**L x L interaction map** instead: for each pair of residues, how much that
specific contact contributes to predicted affinity.

Alternatives cannot reach this. Burial and the rest of the confound panel give
one number per position. Brute-force ISM would need **double** mutants, an L^2
scan -- the exact combinatorial wall this project exists to avoid.

### The ground truth, verified locally 2026-09-29

A double-mutant cycle measures whether two residues interact:
`coupling = ddG(AB) - ddG(A) - ddG(B)`. Zero if they act independently.
Computed from `data/raw/skempi_v2.csv` on 1JTG, the complex we have **already
run and already hold gradients for**:

| | |
|---|---|
| Complete cycles (both constituent singles measured) | **82** |
| Distinct position pairs | **76** |
| Distinct positions involved | 31 |
| Cross-chain pairs (spanning the interface) | 66 |
| Cycles with abs(coupling) > 0.5 kcal/mol | **53 / 82** |
| Coupling range | -4.96 to +7.40 kcal/mol, std 1.84 |
| Replicate spread on the 6 repeated pairs | median **0.21** kcal/mol, max 2.35 |

**1JTG has 83 double mutants, the most of any complex in SKEMPI** (next is
3S9D at 59, then 1BRS at 45). Our target was chosen for other reasons and
happens to be the best-suited complex in the canonical dataset for this.

Two facts matter for inference. Unlike entry 23 -- where 96 mutations collapsed
to 28 positions, 3.4x redundancy -- **82 cycles are 76 distinct pairs, 1.08x**.
This is a near-independent sample. But the 76 pairs draw on only 31 positions,
so the same residue recurs across pairs: **cluster-bootstrap by position, not by
pair**, or significance will be overstated. The 0.21 kcal/mol replicate spread
is the experimental noise floor and therefore the ceiling any predictor can hit.

Physics sanity check, which passes: cross-chain pairs show mean abs(coupling)
**1.374** kcal/mol against **0.889** for same-chain. Residues coupling across
the interface -- which is what binding is -- are more strongly coupled than
pairs within one chain.

### Method

**Layer IG on `z`, not input IG.** `BoltzScorer._checkpointed_forward(self, s,
z, mask, pair_mask)` (`src/igv/boltz_score.py:1335`) already takes `z` as an
argument and returns `(s, z)`, so `z` can be interpolated directly between the
`mean_aa` baseline's pair tensor and the wild-type's, and the model re-entered
from there. This makes completeness well-defined at that layer and avoids
retaining grad on an intermediate inside reentrant checkpointing.

    A[i,j] = (z_x[i,j] - z_b[i,j]) . integral_0^1 dF/dz[i,j] dalpha

contracted over the 128 channels, then symmetrised as `A[i,j] + A[j,i]` since
the pair representation is not guaranteed symmetric.

**Contract with a dot product, not an L2 norm.** This is entry 31 arriving in
advance: an L2 norm over 128 channels is non-negative, discards sign, and by
concentration of measure flattens the map whatever the model does. The dot
product preserves sign and satisfies completeness. Do not repeat that mistake
one dimension up.

### Controls — the second one decides whether this is interesting

1. **Inter-residue distance.** Nearby pairs couple more, so distance will
   predict coupling on its own. Report partial correlation controlling for
   Cbeta-Cbeta distance. This is the burial lesson (section 11) applied before
   rather than after.
2. **The two single-residue attributions.** Does the off-diagonal carry
   information beyond the diagonal? Partial correlation of `A[i,j]` against
   coupling, controlling for `A[i]` and `A[j]`. **If the pair term adds nothing
   over the two single terms, the pair tensor is not telling us anything new
   and this phase ends here.** Exactly the IG-vs-burial test from section 11,
   one dimension up.
3. **Permutation null** over pair labels, matching the shuffled-null convention
   already used in stage 10.
4. **Cluster bootstrap by position** (31 clusters), not by pair.

### Cost

No GPU beyond a run already required. The pair gradient is ~93 MB fp32 at
1JTG's 427 tokens (427^2 x 128 x 4). Capture it during the Phase 2 IG run.

### What would kill it, stated in advance

- Control 2 fails: the pair map is just the outer product of the singles.
- The map tracks contact geometry rather than energetics -- plausible, since
  the pair representation is what the model uses to predict structure. Control
  1 is the test, and a strong distance correlation with a null partial is the
  failure signature.
- `z` interpolation turns out not to be re-enterable cleanly given the
  checkpoint arrangement. Entries 8-11 record how load-bearing the checkpoint
  mode is here; **verify re-entry on a small L before costing this.**

### Status of the novelty claim

Three literature sweeps found no gradient attribution on a co-folding model's
affinity head. Pairwise readouts from protein models exist but all target
**contacts and structure** on sequence-only models -- Rao et al. (ICLR 2021,
attention-head logistic regression), Vig et al. (ICLR 2021), Zhang et al.
(*PNAS* 2024, the categorical Jacobian, ~19L forward passes), Thorstenson
(arXiv:2606.21876). None targets binding energetics, and none is a gradient on
an internal pair tensor. **A confirmatory search was still running when this
was written -- treat the novelty claim as unverified until it is recorded
here.**
