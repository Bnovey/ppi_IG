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
hydrophobicity, volume and normalised position -- but **no conservation term**.

**Corrected 2026-09-29: we hold neither RSALOR factor, not "half".** The
panel's `burial` is not RSA. `compute_burial` (`src/igv/skempi.py:450-482`)
counts neighbouring heavy atoms within 10 A -- a spatial density, as its own
docstring says. RSALOR is `(1 - RSA) x LOR`, so a real comparison needs **two**
new terms: a genuine relative solvent accessibility (DSSP or FreeSASA) and the
log-odds-ratio conservation term. Costing this as "one MSA fetch" understates
it. Also note **no MSA is on disk** and the only implemented fetch route runs
inside boltz (`src/igv/boltz_score.py:845`, via `build_complex_feats(
use_msa_server=True)`), which is not installed locally -- so this is not a
local CPU-only change either. It needs a VM session or new fetch code.

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

### The ground truth -- CORRECTED 2026-09-29 (second pass). Read this before quoting any coupling number.

> **The figures first recorded in this section were wrong, and wrong in the
> direction that flattered the phase.** They came from an undocumented
> row-order dependence -- keeping the *last* duplicate measurement per single
> mutant -- combined with computing ddG at a fixed 298 K. Reference
> `17430899` is a Van't Hoff temperature series measuring 1JTG at **279, 281,
> 283, 285, 288, 293, 298, 301, 303 K** while every other reference sits at
> 296-299 K, and the last-row rule systematically selects those rows. Their
> ddG was then computed as if measured at room temperature. The apparent
> coupling signal was substantially temperature-mismatch error plus
> between-lab offset. The old recipe reproduces every original figure to the
> digit (std 1.84 at ddof=0, range -4.96 to +7.40, 53/82, cross 1.374, same
> 0.888), which is how it was identified; that exactness is a symptom, not a
> validation.

A double-mutant cycle measures whether two residues interact:
`coupling = ddG(AB) - ddG(A) - ddG(B)`. Zero if they act independently. It is
a **difference of differences**, so it only means anything when all three ddGs
share an experimental context -- that is the only way the lab's wild-type Kd,
buffer, method and temperature cancel. Mixing references injects between-lab
offset directly into the coupling term, which is the very quantity we predict.
**Match within `Reference`**, breaking ties on temperature. Not by averaging
across measurements (also wrong, if less so: std 1.30), and never by row order.

Computed from `data/raw/skempi_v2.csv` on 1JTG under within-reference matching:

| | corrected | as first recorded (wrong) |
|---|---|---|
| Complete cycles | **80** | 82 |
| Distinct position pairs | **74** | 76 |
| Distinct positions involved | **31** | 31 |
| Cross-chain cycles (spanning the interface) | **66 / 80** | 66 (of 82; a *cycle* count, not pairs) |
| Cycles with abs(coupling) > 0.5 kcal/mol | **33 / 80** | 53 / 82 |
| Coupling range | **-3.53 to +1.96** kcal/mol, std **0.92** | -4.96 to +7.40, std 1.84 |
| Replicate spread on the 6 repeated pairs | median **0.42** kcal/mol, max 2.15 | median 0.21, max 2.35 |

**The target's dynamic range roughly halved and the pairs clearly above noise
fell from 53 to 33.** Sample size survives; difficulty does not. Note the
`abs(c) > 0.5` counts throughout this document and section 13's table were
computed under the wrong rule and are all inflated.

**1JTG has 83 double mutants, the most of any complex in SKEMPI** (next is
3S9D at 59, then 1BRS at 45). Our target was chosen for other reasons and
happens to be the best-suited complex in the canonical dataset for this.

Two facts matter for inference. Unlike entry 23 -- where 96 mutations collapsed
to 28 positions, 3.4x redundancy -- **80 cycles are 74 distinct pairs, 1.08x**.
This is a near-independent sample. But the 74 pairs draw on only 31 positions,
so the same residue recurs across pairs: **cluster-bootstrap by position, not by
pair**, or significance will be overstated.

**The noise floor is not 0.21 kcal/mol.** That figure was the coupling-level
replicate spread over 6 repeated pairs under the broken rule, and it was quoted
as "the experimental noise floor and therefore the ceiling any predictor can
hit". Corrected, the coupling replicate spread is **median 0.42, max 2.15**.
More importantly, the spread of the *same single mutation* measured by
different labs is **median 1.16, 90th percentile 1.63, max 5.37 kcal/mol** on
1JTG (33 of 58 singles are measured more than once, across 10 references; the
worst case differs by 5.37 kcal/mol between two papers). Within-reference
matching is what keeps that from propagating into the coupling term -- it is
not a refinement, it is the thing that makes the target measurable at all.

**Physics sanity check -- it does NOT pass. This was the artifact, not
evidence.** As first recorded: cross-chain mean abs(coupling) 1.374 against
0.889 same-chain, read as confirmation that "residues coupling across the
interface -- which is what binding is -- are more strongly coupled". Under
within-reference matching the two are **0.639 (n=66) and 0.569 (n=14)**: a gap
of **0.070 kcal/mol at Mann-Whitney p=0.75, Welch p=0.65**, against a coupling
noise floor of 0.42. There is **no evidence** that cross-interface pairs couple
more strongly than within-chain pairs in this dataset.

This one matters beyond its own line. It was the only independent check that
the ground truth reflected interface biology rather than assay noise, and it
was produced by the same bug as the inflated coupling values. Phase 5 now rests
on Gate C alone, with no prior reassurance that the target is physical. That is
a materially weaker position than this section originally described, and it
should be stated to referees rather than discovered by them.

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
- ~~`z` interpolation turns out not to be re-enterable cleanly~~ --
  **ASSESSED 2026-09-29, GREEN.** See below.

### Feasibility of the z seam -- assessed 2026-09-29

**Verdict: green.** `z` enters `_checkpointed_forward` as a parameter and
leaves as a return value with no side effects and no other uses
(`src/igv/boltz_score.py:1335-1349`), so it can be supplied externally without
touching any existing path.

**Seam: `src/igv/boltz_score.py:1780`**, immediately before the existing
`model.confidence_module(...)` call. Nothing upstream -- `z_init`, the
recycling loop, the MSA blocks -- needs to change; `s`, `x_pred` and `feats`
are fixed across interpolation steps.

**The checkpointing fear does not apply here.** The inner per-layer checkpoints
are already `use_reentrant=False` (line 1347). The reentrant one that caused
entries 8-11 is the *outer* wrapper at line 1832, and calling the confidence
head directly bypasses it: grad is enabled throughout, `_ckpt_mha`
(lines 1428-1429) takes its checkpointed branch, and gradients accumulate on
`z.grad` normally. Entry 11's triangle-attention checkpointing already uses
this exact pattern.

**Recycling resolves itself.** Only the final `z_` after all recycling
iterations reaches the confidence head (line 1777 onward); intermediate values
never do. So "the pair tensor" is unambiguous -- attribute to the final one.
Per-iteration attribution would need new code and is not needed here.

**Memory: ~5.5-8 GiB against the 78 GiB that full-trunk backward needs**
(entry 12, all nine measured configurations). A 10-14x reduction, because the
confidence head is 8 checkpointed layers rather than 64 pairformer blocks plus
4 MSA blocks. **This is the same frozen-trunk escape entry 12 already chose**,
so Phase 5 does not need the full-trunk problem solved -- it rides the decision
we already made.

Carry the matching caveat: attributing through the confidence head with `s` and
the trunk fixed answers "how does the confidence head use `z`", not "how does
the whole model use `z`". That is the limitation frozen-trunk already accepts,
and it should be stated rather than glossed.

### The de-risking test -- and what it must NOT be

Run at L ~ 100, `m_steps=5`, a few minutes on the A100, before anything is
costed.

**The test must call the real confidence head.** The version first proposed
scored a synthetic random `z` through a mock `(z**2).sum()*0.01` and checked
completeness. That is a tautology -- IG on a quadratic satisfies completeness
by construction, and it exercises none of the risk. The whole question is
whether gradient flows through the **real checkpointed confidence pairformer**
to an externally supplied `z`. A mock cannot answer it and would give false
assurance, which is exactly the failure mode of the `dead_target` check
(section 4) that passes vacuously at 0.000e+00 every run.

Pass criteria: `z.grad` is not None; not uniformly zero; shape
`(1, L, L, 128)`; all finite; and completeness relative error under ~20% at
m=5 against a real `f(z_x) - f(z_baseline)` computed from the same head.

Three runtime unknowns, none blocking: whether model parameters are frozen
(`MEMORY.md` section 6 says they are not -- check
`sum(p.numel() for p in model.parameters() if p.requires_grad)`), the exact `s`
dimension, and whether `z.grad` stays fp32 under autocast (`MEMORY.md` section
2.2 says `z` is fp32 in the forward even under bf16 -- verify).

### Status of the novelty claim

Three literature sweeps found no gradient attribution on a co-folding model's
affinity head. Pairwise readouts from protein models exist but all target
**contacts and structure** on sequence-only models -- Rao et al. (ICLR 2021,
attention-head logistic regression), Vig et al. (ICLR 2021), Zhang et al.
(*PNAS* 2024, the categorical Jacobian, ~19L forward passes), Thorstenson
(arXiv:2606.21876). None targets binding energetics, and none is a gradient on
an internal pair tensor.

**Confirmed 2026-09-29 by a dedicated search: no one has produced a pairwise
attribution map for binding affinity or ddG.** The strongest evidence is the
incumbent naming the gap itself. BALM-PPI (Singh et al., bioRxiv
10.64898/2026.03.30.715237) is the current state of the art in *explainable*
PPI affinity -- >12,000 complexes, residue-level IG maps that recover known
hotspots -- and its own text concedes that its Integrated Gradients
attributions "reflect mean-pooled embeddings and do not yet provide pairwise
residue-residue contact predictions."

Every ingredient exists separately; nobody has assembled them:

| Nearest neighbour | L x L | Attribution | Target | Gap |
|---|---|---|---|---|
| Categorical Jacobian (Zhang, *PNAS* 2024) | yes | finite difference | MLM logits | wrong target |
| Top-K attention heads (Thorstenson) | yes | no, a readout | contacts | wrong target |
| Nambiar et al. 2025 pLM epistasis matrix | **yes** | finite difference | DMS fitness epistasis | closest on form, wrong target |
| DFIM (Greenside, *Bioinformatics* 2018) | **yes** | **yes** | TF binding | **right method, wrong molecule -- DNA** |
| Integrated Hessians (Janizek, Sturmfels & Lee, *JMLR* 22(104):1-54, 2021) | **yes** | **yes**, axiomatic | anything differentiable | **never applied to any protein model** |
| BALM-PPI (2026) | no | yes, IG | **binding affinity** | first-order only, and says so |

Note DFIM: genomics solved pairwise attribution in 2018 and it never crossed
into proteins. That is the same pattern as the Majdandzic simplex correction in
entry 31 -- **the genomics community's answers keep not being carried over**,
and carrying them is repeatedly where the contribution turns out to be.

### Calibration -- what counts as a good coupling correlation

**Do not benchmark against SKEMPI multi-point numbers.** IGMI reports Pearson
0.77 on M1707, DDMut-PPI 0.83 on SM1124, MutPPI+ 0.880. Those are correlations
against **total** multi-point ddG, which is dominated by the additive sum of
the singles. A model can score 0.8 there with **zero** signal on the
non-additive part. The search found no paper reporting a SKEMPI-wide
correlation on the coupling term alone -- which is both why this is open and
why the comparison must not be made.

The honest benchmarks for a coupling-term prediction:

| Source | Target | Number |
|---|---|---|
| Nambiar et al. 2025 (bioRxiv 10.1101/2025.09.14.676130), ESM-2 | TEM-1 epistasis, ~12k doubles | **r = 0.37** |
| same, YAP1-WW, ~8,670 pairs | fitness epistasis | **r = 0.34** |
| same, Pab1-RRM2, ~36,500 pairs | fitness epistasis | **r = 0.26** |
| FEP (Schrodinger), 45 staph nuclease doubles | **non-additivity** | **r = 0.79** -- UNVERIFIED, snippet-sourced, verify before citing |

So **r in the 0.26-0.37 band is what a pLM currently achieves on epistasis**,
and that is against fitness, not binding. Corroborating the difficulty:
Kolchina et al. (bioRxiv 10.64898/2026.02.17.706292) benchmark 95 zero-shot VEP
models and find they "perform well for single mutations and non-epistatic
combinations ... [but] fail to predict the effect of strongly epistatic
combinations."

**Added 2026-09-29 with the ground-truth correction: our own attainable ceiling
is lower than those numbers, for a reason that has nothing to do with method.**
The pLM epistasis figures come from 8,600-36,500 doubles per dataset with
correspondingly averaged-down noise. We have 74 pairs against a coupling
replicate spread of 0.42 kcal/mol on a target whose corrected std is 0.92. That
is a signal-to-noise ratio that caps achievable r well below what a clean
36,500-pair fitness dataset supports, before any question of whether the
gradient carries information. **Compute the attenuation bound explicitly from
the replicate spread and report r alongside it**, rather than comparing a
74-pair correlation to a 36,500-pair one as though they were commensurable. A
result of r = 0.3 here and r = 0.3 on TEM-1 are not the same claim.

### The cost argument is not the contribution

Thorstenson gets the *contact* signal in one forward pass, so "cheaper than the
categorical Jacobian's ~19L" is not by itself a result. **The contribution is
the target quantity (ddG coupling) and the attribution semantics, not the
FLOPs.** State it that way.

### Two things to close before claiming priority

1. **IGMI -- READ IN FULL 2026-09-29. It does not pre-empt us.** Wu, Xu, Xin,
   Zhang, Liu, Zhu, Wei, Zhao, Yu & Feng, *Bioinformatics* 42(4):btag150, 2026,
   PMC13070472. Its only pairwise object is Figure 8c, whose in-image title is
   "Attention Mechanism Head 1 Weight Matrix Heatmap": **one head, one layer,
   one complex, 128x128 rather than L x L** (the model only ever sees a fixed
   128-residue mutation-centred subgraph), non-negative, unitless, computed in
   the forward pass and **never connected to the predicted ddG by any
   computation**. The predicted ddG for that complex is not even reported. The
   word "attribution" appears nowhere in the paper; neither do gradient,
   Shapley, saliency, or completeness. The deciding sentence is the first line
   of their section 3.4: *"Since attention weights can reflect residue-level
   contributions in PPIs, we analyzed IGMI's attention distributions"* --
   attention weights, justified by citation rather than axiom, and described as
   **residue-level**. Its pairwise object is validated against nothing: one
   qualitative case study, no coupling energies, no contact AUC. It also never
   reports the coupling term, only total multi-point ddG on M1707.

   **But phrase our novelty carefully.** IGMI contains an exact
   completeness-satisfying *per-residue* decomposition it never uses
   interpretively -- Eq. 15/17 give `ddG = sum_i ddG_i` by construction. So
   claim **pairwise** attribution, not "first attribution of any kind for
   ddG"; a referee who knows Eq. 17, or the per-residue decomposition
   literature generally, would push back on the broader phrasing.

   Three framings that survive, in increasing strength: (i) **attribution vs
   attention** -- theirs is an architectural internal with no output
   dependence, ours is a gradient of the score with a completeness story;
   (ii) **sign** -- attention is non-negative and cannot say whether a pair
   helps or hurts binding, which for ddG is the entire point; (iii)
   **validation** -- they validate their pairwise object against nothing, so
   validating against double-mutant cycles makes us first to show one means
   something.
2. **MLSB and OpenReview -- SWEPT 2026-09-29. No pre-emption.** All 225 MLSB
   titles across 2023/2024/2025; OpenReview via `api2.openreview.net` with
   `source=forum` across ICLR 2024-26, ICML 2025-26, NeurIPS 2023-25 mains and
   the MLSB / GEM / GenBio / LMRL / AI4Science / MLGenX / Mech-Interp
   workshops, including under-review, withdrawn and rejected; arXiv and Europe
   PMC with a preprint filter. `"pairwise attribution" AND protein` returns
   zero on arXiv; `abs:"integrated gradients" AND abs:"binding affinity"`
   returns zero.

   **Three papers must be cited and explicitly distinguished:**

   - **PairSAE** (Migliorini et al., MLSB 2025, arXiv:2606.27440) is the
     closest thing that exists -- it overlaps on pair representation,
     co-folding model, and Boltz-2 affinity simultaneously. It does not
     pre-empt us for a specific reason worth stating in the paper: its stated
     contribution is summarising the pair tensor "into token-wise interaction
     roles" precisely to avoid "a quadratic blow-up of features". **The L x L
     object is what they engineer around; it is what we produce.** It is also
     an SAE rather than a gradient, affinity enters as a probe *target* rather
     than the quantity attributed, and it is protein-ligand on PLINDER.
   - **TopoScorer** (ICLR 2026 submission, OpenReview QNcrdCKNa5) states our
     motivating gap almost verbatim -- deep affinity models "generally lack
     interpretable attributions" -- but answers it by building a new
     interpretable architecture rather than attributing an existing model.
     **This is the paper most likely to be raised in review.** Pre-empt it by
     framing ours as post-hoc attribution of a frozen co-folding score.
   - **AF2BIND** (MLSB 2023) probes AF2's pair representation for binding
     sites. So do not claim "first interpretability on pair representations",
     and do not claim "first interpretability of Boltz affinity" either --
     PairSAE has a colourable claim to the latter.

### The claim, phrased to survive review

**CORRECTED 2026-09-30 -- the original wording was infeasible.** It read
"...of a co-folding model's *binding-affinity* score...". Boltz-2 cannot
produce a binding-affinity score for a protein-protein complex: the parser
raises `ValueError("Affinity is currently only supported for ligands.")` at
`schema.py:1066-1070`, and the docs restrict the affinity binder to a ligand
chain of at most 128 heavy atoms. See `ERRORS_LOG.md` entry 36. Do not restore
the old phrasing; there is no such score to attribute.

**"First gradient-based, pair-resolved attribution of a co-folding model's
structure-confidence score onto the L x L pair representation."** Every element is
load-bearing: PairSAE has pair-representation + co-folding + affinity but is
SAE-based and deliberately token-level; TopoScorer has affinity + attribution
but is a bespoke architecture; the Boltz probing cluster has pair
representation + causal intervention but targets structure; IGMI has an L x L
object and a ddG target but the object is attention, not attribution. Claim the
intersection, never the parts.

Supporting argument worth citing: Jedryszek et al. (arXiv:2608.11475) showed a
beta-strand direction in Boltz-1 that was highly linearly decodable (F1 0.82)
yet produced **no** structural change when steered. **Linear decodability does
not imply causal influence** -- which is a direct argument for gradient-based
attribution over the probing and SAE work that currently dominates this space.

### Residual risk, stated honestly

MLSB 2020-2022 was not scanned (predates co-folding affinity heads).
Non-archival workshop posters with no OpenReview record are invisible to every
index. ICLR/ICML 2026 submissions still under embargo would not surface. And
Europe PMC search is title/abstract-weighted, so this done as an unadvertised
side analysis inside a broader affinity paper would be missed. Confidence is
high for the specific claim above, moderate-high for any looser phrasing.

---

## 13. Execution plan for Phase 5

Written 2026-09-29. Five stages, three decision gates, and an explicit early
exit at each. Total exposure if it runs to completion is roughly **$60-75 of
A100**; total exposure if it dies at the first gate is **about $2**. The
ordering is chosen so the cheapest disconfirming evidence arrives first.

### Which complexes can replicate this -- not the ones we planned

The Phase 1 pooling set (3HFM, 1VFB, 1JRH, 2JEL) was chosen for **single**
mutants. Phase 5 needs **doubles with both singles measured**, and that is a
different set. Computed from `data/raw/skempi_v2.csv`, complexes with at least
8 complete cycles:

**This table was computed under the broken row-order rule and is superseded.**
The `distinct pairs` column below is corrected to within-reference matching;
the `abs(c) > 0.5` and `std` columns are NOT corrected per complex -- they were
inflated by the same mechanism, so treat them as unreliable and recompute
before using them to prioritise. Only 1JTG has been recomputed in full.

| Complex | pairs (within-ref) | pairs (as first recorded) | abs(c) > 0.5 -- INFLATED | std -- INFLATED |
|---|---|---|---|---|
| **1JTG_A_B** | **74** | 76 | 33 / 80 (corrected) | **0.92** (corrected) |
| 3S9D_A_B | 53 | 58 | 6 | 0.49 |
| **1BRS_A_D** | **33** | 37 | 25 | 1.35 |
| 4G0N_A_B | 32 | 32 | 18 | 0.73 |
| 1LFD_A_B | 26 | 26 | 22 | 0.94 |
| 1AO7_ABC_DE | 21 | 24 | 16 | 0.81 |
| 1DAN_HL_UT | **4** | 14 | 9 | 0.93 |
| 1VFB_AB_C | 14 | 14 | 5 | 1.05 |
| 1DQJ_AB_C | 13 | 13 | 9 | 1.68 |
| 1DVF_AB_CD | 13 | 13 | 11 | 1.00 |
| 3HFM_HL_Y | 11 | 11 | 14 | **2.12** |
| **Total** | **294** | **375** | | |

**`1DAN_HL_UT` collapses from 14 pairs to 4** -- its cycles were almost
entirely assembled across labs. Drop it from the replication set; at n=4 it
cannot contribute either way.

**Read the `abs(c) > 0.5` column, not the cycle count.** 3S9D has the second
most cycles and almost no coupling -- 6 of 59 above 0.5 kcal/mol, std 0.49. It
is a near-additive system and would dilute the signal while looking like a
large sample. Do not use it early. (This conclusion survives the correction:
3S9D's coupling was the *least* inflated of the set, so if anything it is more
clearly near-additive than the table suggested.)

**1BRS is barnase-barstar**, the system on which double-mutant cycles were
established as a method (Schreiber & Fersht). 33 pairs on the canonical
coupling system is the single most persuasive replication available, and it
should be the first complex after 1JTG. But its "25 strongly-coupled pairs" is
an uncorrected figure -- recompute it before leaning on that phrasing, because
1JTG's equivalent count fell by 38% under correction.

**And check 1BRS's cluster count before trusting a CI from it.** Measured
2026-09-29 once it was registered: 35 cycles, **33 distinct pairs on only 12
distinct positions** -- 2.75x redundancy, against 1JTG's 74 pairs on 31
positions (2.4x). Section 12 correctly requires cluster-bootstrapping by
position rather than pair, but **12 clusters is a small bootstrap**, and a CI
computed from it will be wide and unstable regardless of the effect. Barnase-
barstar is the right system for persuasiveness and the wrong one for a tight
interval; plan to report it as a directional replication, not as independent
confirmation with its own significance claim. All 33 of its pairs are
cross-chain, which is at least the geometry we want.

Ceiling if everything runs: **294 distinct pairs** (290 excluding 1DAN, which
falls below the 8-cycle threshold entirely, and before any decision on 3S9D),
against n=28 today. Still an order of magnitude more than the position-level
work, which remains the point.

**Two things found when the implementation was verified 2026-09-29. Both
change what Stage 4 has to do.**

**1. Quote the right denominator -- 294 and 310 are both correct.** 294 is the
within-reference total over the 11 complexes this table lists. Running
`extract_all_complexes(min_cycles=8)` over all of SKEMPI gives **310**, because
it surfaces four complexes never listed here: `5XCO_A_B` (9 pairs), `5M2O_A_B`
(7), `1KNE_A_P` (3), `3MZG_A_B` (1). 5XCO is worth adding to the queue. But
note **`1KNE_A_P` has 33 cycles collapsing to 3 distinct pairs -- 11x
redundancy**, which is entry 23's problem in a new place: it would look like a
large sample and contribute almost nothing independent. Screen candidates on
distinct pairs, never on cycle count.

**2. Seven of the ten Stage 4 complexes are not registered, including 1BRS.**
`SKEMPI_COMPLEXES` (`src/igv/skempi.py:62`) holds only 1JTG, 3HFM, 1VFB, 1JRH
and 2JEL -- the Phase 1 *single*-mutant set. Missing: **1BRS**, 4G0N, 1LFD,
1AO7, 1DQJ, 1DVF, 3S9D. `scripts/13_coupling.py` calls `get_complex` before
doing any work and exits on `KeyError`, so **Stage 4a as written fails on its
first line** -- on barnase-barstar, the complex this plan calls the single most
persuasive replication available. Cheap to fix, and it had to be found before
the VM was billing rather than after. The general point: the replication set
was chosen from the CSV, and nobody checked it against the structural registry
the pipeline actually resolves through.

---

### Stage 0 -- local, free, no GPU

Can all be done now. None of it needs the VM.
**Executed 2026-09-29. Status against each item below.**

0a. ~~**Resolve the uncommitted 2026-09-29 09:49 changeset**~~ -- **DONE**
    before this session; tree clean at `33c545d`, "Clean up the repo for
    outside readers".
0b. ~~**Implement `confidence_head_forward`**~~ -- **DONE**, commit `6f41295`.
    `confidence_head_forward` in `src/igv/boltz_score.py`, `pair_layer_ig` +
    `PairAttribResult` in `src/igv/attrib.py`, 17 boltz-free plumbing tests in
    `tests/test_pair_attrib.py`. Two deviations from this section, both
    deliberate: symmetrisation is **`(A + A^T)/2`**, since the literal
    `A[i,j] + A[j,i]` doubles the total and breaks completeness; and the
    distogram is recomputed from the interpolated `z` rather than frozen,
    because it is a direct function of `z`. Section 12's line numbers were
    verified accurate at `33c545d`.
0c. **Implement `src/igv/coupling.py`** -- **DONE, then corrected.** Note the
    spec above said "compute ddG from `Affinity_*_parsed` at RT = 0.001987 *
    298.15". **That instruction was wrong and is the origin of the bug in the
    ground truth.** 1JTG spans 279-303 K (reference `17430899` is a Van't Hoff
    series); a fixed temperature is not available. Use per-row temperature via
    the existing `add_ddg` (`src/igv/skempi.py:219`), and match cycles
    **within `Reference`**. See the corrected ground-truth block in section 12.
0d. **Add a conservation term to the confound panel** (section 11 item 2).
    **NOT DONE -- deferred, and mis-scoped here.** It is not cheap and not
    local: no MSA is on disk, the only implemented fetch runs inside boltz
    which is not installed locally, and `burial` is not RSA so RSALOR needs
    two new terms rather than one. See the correction in section 11. It is
    confirmed off Phase 5's critical path -- nothing in the pair-attribution
    plan or the four controls consumes a conservation feature -- so deferring
    costs Phase 5 nothing.

### Stage 1 -- sync and de-risk -- **DONE 2026-09-29. GATE A: PASS. ~$1.70, ~20 min.**

1a. ~~Start the VM, `git pull`.~~ Done, synced to `62889c6`. **`git pull` does
    not work on the VM** -- see the sync note in `HANDOFF.md`; it has no GitHub
    credentials. A `git bundle` was used instead.
1b. ~~Run the real, non-mocked z-gradient completeness test.~~ Done at
    **L = 352** (full 1VFB complex, chains A/B/C), not L ~ 100 -- a single
    chain scored with `complex_pde` risks a degenerate score and therefore a
    spurious FAIL, and the cost difference was pennies. Real confidence head,
    `mean_aa` z baseline, MSA server. Artifact:
    `results/gate_a_1VFB_complex_pde.json`.

> **GATE A: PASS**, all five criteria.
>
> | criterion | result |
> |---|---|
> | `z.grad` not None | yes |
> | not uniformly zero | max abs **4.652914e-04**, zero fraction **0.0000** |
> | shape `(1, L, L, 128)` | `(1, 352, 352, 128)` |
> | all finite | NaN 0, Inf 0 |
> | completeness | **abs err 0.003627**, rel err 0.0064 |
>
> Completeness is the result worth keeping: `ig_sum = -0.571934` against a true
> `f(z_x) - f(z_b) = -0.568307`, i.e. **0.64% relative error at m=5**. The seam
> is not marginally usable, it is clean, and the L x L map is a genuine
> decomposition of the score. **Wall time 5.0 s** for all five IG steps.

**Two things in this section's own pass criteria were wrong, and would have
mattered:**

1. **"completeness relative error under ~20%" is the wrong gate.** That is
   entry 32's bug restated -- relative error divides by the span, so a *good*
   baseline sits close to the input, the span collapses, and the ratio
   inflates. Combined with the `mean_aa` baseline this section correctly
   specifies, gating on relative error can fail the better setup. Gate A gates
   on **absolute** error against `07_sanity.py`'s `COMPLETENESS_ABS_THRESHOLD`
   and reports relative as information. Here both were comfortable, so the
   distinction did not bite -- but it was luck, not design.
2. **There was no INCONCLUSIVE outcome.** A single FAIL code conflates
   "gradient does not reach `z`" with "the setup was degenerate", and those
   have opposite consequences: the first ends Phase 5, the second means fix the
   harness and rerun. `14_gate_a.py` now exits 2 for a collapsed span, a
   non-finite or zero score, a constant path, or degenerate `z_x`, checked
   *before* the five criteria.

**The three runtime unknowns are now measured, not assumed** (all three were
listed in `MEMORY.md` section 6 as open; all three are now settled there):

| unknown | measured |
|---|---|
| (a) parameters frozen? | **25,026,048 trainable / 481,698,944 frozen** -- 95% frozen already. MEMORY.md's "never frozen" was wrong. |
| (b) `s` dimension | `s.shape = (1, 352, 384)` |
| (c) `z.grad` fp32 under autocast? | **yes**, `torch.float32` |

**Memory: peak 8.68 GiB** (8.55 GiB for the single backward). Section 12
predicted 5.5-8 GiB, so slightly over the top of the range but ~9x below the
78 GiB a full-trunk backward needs. The frozen-trunk escape holds.

**Settled bonus:** `score.backward()` parks only **18.1 MiB across 315
tensors** in parameter `.grad`. So switching `pair_layer_ig` to
`torch.autograd.grad` would save ~18 MiB against an 8.68 GiB peak and is not
worth splitting the convention with `integrated_gradient`. Recorded because
the pre-run guess was that this was a significant VRAM cost; it is not.

### Two blockers found by Gate A that must be fixed before Stage 2

Both are free, local, and would corrupt the Stage 2 capture rather than crash
it -- which is the dangerous kind.

1. **`x_pred` is zeros.** Gate A logged `feats['coords'].abs().max() = 0`,
   settling the "GEOMETRY IS FIXED" question in `MEMORY.md` section 6: the
   sequence-only YAML gives every atom (0,0,0), so the confidence head sees no
   structure. Harmless for a gradient-flow test. **Not harmless for Stage 2**,
   where the pair map would describe a structureless complex. Stage 2 must run
   a real structure prediction (~45 s) and feed genuine `x_pred`.
2. **The featurisation cache substitutes silently.** `ERRORS_LOG.md` entry 34:
   the homopolymer path returned 1JTG's 427-token features for a 352-token
   1VFB request, because `data/raw/boltz_homopolymer/<AA>` is not keyed on
   dataset. Entry 24 fixed this for the main path only. Caught solely by
   `_build_token_map`'s run-length guard; without it the `mean_aa` baseline
   would have come from the wrong protein and **Gate A would plausibly have
   passed anyway**. Key every featurisation directory on dataset and audit for
   other instances.

### Stage 2 -- capture on 1JTG (~$10, ~2 h)

2a. One IG run on 1JTG capturing **both** the pair gradient (L, L) and the
    (L, 20) simplex map from entry 31. Same backward passes; take both.
2b. Also capture `F(baseline)` and `F(input)` for the alpha-profile
    (section 11 item 5) -- `09_path_profile.py` already supports `--baseline`.

> **GATE B, sanity only.** Is the pair map non-degenerate -- not near-constant,
> not concentrated on one row, not tracking the diagonal alone? Report the same
> concentration statistics entry 31 used, because a flat map here means the
> contraction is wrong, not that biology is absent.

### Stage 3 -- analysis, local, free

3a. Join to the 82 cycles / 76 pairs. Run all four controls from section 12:
    distance partial, **singles partial**, permutation null, cluster bootstrap
    by position (31 clusters, not 76 pairs).

### STAGE 3 RESULT -- GATE C FAILED, 2026-09-30. Phase 5 is closed as a negative.

Run: `scripts/13_coupling.py --complex 1JTG --pair-map
results/1JTG_complex_pde_pair_ig.npz`. Full detail in `ERRORS_LOG.md` entry 35.

| control | result |
|---|---|
| **Gate C -- partial(A[i,j], coupling \| A[i], A[j])** | **+0.0019** |
| raw Spearman(A[i,j], coupling) | -0.0335 |
| permutation null, n=1000 | null mean +0.0029, **p = 0.78** |
| cluster bootstrap, 31 positions | 95% CI **[-0.156, +0.062]** |
| Spearman(\|A[i,j]\|, \|coupling\|) | +0.072, p = 0.53 |
| distance ↔ coupling | **+0.4005** |
| partial(A[i,j], coupling \| distance) | +0.1764 |
| **Spearman(\|A[i,j]\|, centroid distance)** | **-0.527, p ~ 0** |

**The failure signature is the one this section predicted.** It listed "the map
tracks contact geometry rather than energetics" as a kill condition and said
"a strong distance correlation with a null partial is the failure signature".
Measured: -0.527 against distance, +0.0019 partial. Exactly that.

**The near-miss worth remembering.** The 80 SKEMPI cycles sit at the 90.7th
percentile of the map by \|A\|, 10.4x the map's median, with 50% inside the top
decile. In isolation that reads as the attribution finding the known hot pairs.
It is geometry: experimentalists mutate interface residues, which are close,
and the map is high on close pairs. That claim would have looked good and been
wrong, and only Control 1 separates the readings.

**Why this negative is trustworthy:** completeness exact (abs err 0.000000 at
m=64), ranking converged (rho = +1.0000 across m=8/16/32/64), map
non-degenerate on every Gate B statistic, and the target demonstrably
predictable -- distance alone scores +0.40 on it. A predictor can score here;
ours cannot.

**Second, independent diagnosis: the target was never an affinity score.** All
six registered scores are structure-confidence metrics, and Boltz-2's affinity
head `boltz2_aff.ckpt` sits in the checkpoint directory deliberately unloaded
(`boltz_score.py:475-493`, `affinity=False` at :891). The claim below says
"binding-affinity score"; `complex_pde` is predicted distance error. Correct
the claim, or change the target -- but do not restate it as written.

**Cost: ~$4 across four A100 sessions**, against the ~$60-75 budgeted to reach
this point. Stage 4's 1BRS replication and the remaining four complexes are
correctly unspent. The ordering this plan chose -- cheapest disconfirming
evidence first, analysis written before any GPU time -- is what bought that.

**Do not read this as "gradient attribution on pair representations does not
work."** It is narrower: attributing a *structure-confidence* score yields a
map of *geometry*, which does not predict energetic coupling. Whether the
affinity head carries PPI energetic signal is an open and separate question,
and section 11 already flagged the prior -- it is trained predominantly on
protein-ligand data, and King et al. (arXiv:2512.06592) find Boltz-2 fine-tuned
for PPI affinity "underperforms relative to sequence-based alternatives". Settle
that cheaply before building anything on it.

---

> **GATE C -- the kill condition, stated before the data exists.** If
> `A[i,j]` adds nothing over `A[i]` and `A[j]`, the pair tensor carries nothing
> the diagonal did not, and **Phase 5 ends here.** Write the result up as a
> negative and return to section 11's queue. Benchmark for a positive:
> r in 0.26-0.37 is what pLM epistasis achieves (section 12); do **not**
> compare against the 0.77-0.88 multi-point numbers, which are dominated by
> additivity.

### Stage 4 -- replication (~$10 per complex)

Only if Gate C passes. In order:

4a. **1BRS** (~$10) -- barnase-barstar, 37 pairs, the canonical system. If the
    effect does not appear here it is not real.
4b. **4G0N, 1LFD, 1AO7, 3HFM** (~$40) -- takes the pooled total toward ~250
    pairs. 3HFM is small at 11 pairs but has the strongest coupling in the set
    (std 2.12), so it is a high-information cheap add.
4c. 3S9D only as a **negative control**: a near-additive system where the
    predictor should score near zero. If it scores well there, the signal is
    additive contamination.

### Stage 5 -- write-up

Claim exactly the sentence in section 12, cite and distinguish PairSAE,
TopoScorer, AF2BIND and IGMI, report model-faithfulness and
biological-faithfulness separately per Yao et al., and state the frozen-trunk
limitation rather than glossing it.

### What this plan deliberately defers

- **The brute-force scan** (~$22). Still the only thing that answers the
  original headline claim, and still unpublishable without it -- but it is a
  claim about the *position-level* method, and Phase 5 is now the lead.
- **The 6M0J saturation arm.** Built and tested, never run. The (L, 20) map
  from entry 31 is its natural input, so revisit after Gate C.
- **Pooling 3HFM/1VFB/1JRH/2JEL for single mutants.** Note 2JEL has **zero**
  doubles and 1JRH six, so that set does almost nothing for Phase 5; it stays
  in the queue for the position-level question only.

---

## 14. The simplex reduction, measured at last -- and it makes no difference

Run 2026-09-30 on the A100, first execution of the `(L, 20)` path. Entry 31
opened 2026-09-24 and had never been tested end to end; three defects sat
between it and a trustworthy number (`ERRORS_LOG.md` entries 39, 40, plus the
consumer/producer gap and the hardcoded zero geometry). All four are closed.

### What was actually fixed before any number was believed

1. **Real geometry.** `03_attribute.py` took `x_pred = feats["coords"]`
   unconditionally, and the sequence-only YAML puts every atom at the origin --
   confirmed live: `feats['coords'].abs().max() = 0`. Now `--x-pred predicted`
   by default. Measured effect on 1JTG: `coords.abs().max()` 0 -> 31.5654.
2. **The producer was never wired.** `10_skempi_hotspots.py` read a
   `grad_res_type` key that nothing wrote, so the "fixed" path silently took
   the L2 fallback. Only the consumer half of entry 31 had ever existed.
3. **The homopolymer cache was keyed on the chains it did not featurise**
   (entry 40). Caught by the guard, which aborted rather than building the
   baseline from 262 alanines standing in for the partner chain.
4. **The chance baseline was a count, not a fraction** (entry 37), so every
   top-k comparison in the project had read "never beats chance" by
   construction.

### VERDICT after all four complexes -- 2026-10-01. The fix does not help.

**Written after 3HFM landed. The 3S9D-only reading below was recorded while two
of four complexes were still running and is kept for provenance, not as a
result. Read this block first.**

| complex | positions | simplex | L2 | diff | cleared own null? |
|---|---|---|---|---|---|
| 3S9D A | 50 | **+0.4683** | +0.0356 | +0.4327 | simplex only |
| 1JTG B | 28 | +0.1714 | +0.2946 | -0.1232 | neither |
| 2JEL P | 32 | **-0.3396** | +0.0031 | -0.3427 | neither |
| 3HFM Y | 15 | +0.1573 | +0.4218 | -0.2645 | neither |

- **simplex** mean +0.1143, sd 0.3349, t=+0.68, **p=0.54**
- **L2** mean +0.1888, sd 0.2028, t=+1.86, **p=0.16**
- **paired difference** mean -0.0744, t=-0.43, **p=0.70**
- simplex beat L2 in **1 of 4** complexes
- **1 of 8** measurements cleared its own shuffled null -- what noise gives at
  eight looks

**Conclusion: the L2 reduction was never what held the method back.** Entry 31's
diagnosis of the reduction was theoretically correct and is still worth keeping
(sign-discarding, concentration-of-measure flat), but fixing it changes nothing
measurable. On 2JEL the corrected map is actively anti-predictive -- AUROC
0.365, precision@5 = 0.00 against a 0.25 base rate, Spearman CI
[-0.6273, -0.0019] excluding zero on the wrong side.

**Do not quote the 3S9D number alone.** It is one favourable draw in four, and
the sign flips across complexes. The between-complex variance (sd ~0.33)
swamps the between-method difference (-0.07).

**Two caveats that weaken even these numbers:**
1. **Completeness failed on 3 of 4 runs** -- 8.51% (3S9D), 11.14% (3HFM), 5.99%
   (1JTG) against the 5% criterion; only 2JEL passed at 2.37%. Real geometry
   roughly tripled the `f(x) - f(baseline)` span, so m=32 is now too few
   quadrature steps. The one run that passed is the strongest negative.
2. **3HFM was underpowered by construction** -- 15 positions against 8
   confounds gives a partial-correlation CI of [-1.0, +1.0], i.e.
   unidentifiable. Check degrees of freedom before buying GPU time; this cost
   ~100 minutes of A100 to learn nothing.

**Predictions registered before the data and falsified by it** -- recorded
because a prediction only counts if its failure is written down:
- "global correlation will stay at or below chance" -- false on 3S9D (+0.4683
  against a null of +0.2441).
- "base rate and positional coverage explain the spread, so 2JEL and 3HFM will
  land between 3S9D and 1JTG" -- false; 2JEL (25%, intermediate) produced the
  most extreme negative. There is no working account of the between-complex
  variation. Do not invent one.

---

### Superseded: the 3S9D-only reading, recorded 2026-09-30 before the tie-break

### 3S9D chain A -- 50 positions, 6 hot spots, 12% base rate

The best-powered position-level test this project has run: nearly twice 1JTG's
position count, at a quarter of its hot-spot prior, on a shorter sequence
(L=307 vs 427).

| metric | simplex (L,20) | L2 norm | shuffled null p95 |
|---|---|---|---|
| Spearman vs \|ddG\| | **+0.4683** CI [+0.2060, +0.6734] | +0.0356 | +0.2441 |
| partial, all confounds | **+0.3632** CI [-0.0109, +0.7036] | -0.1367 | -- |
| AUROC | **0.8674** CI [0.6792, 0.9956] | 0.5644 | 0.7197 |
| AUPRC | **0.6335** | -- | 0.3390 |
| hot-spot precision@5 | **0.60** (chance 0.12) | **0.00** | -- |

Spearman clears the null and its CI excludes zero. AUROC and AUPRC both clear
their nulls, and AUPRC 0.63 against a 0.12 base rate is the strongest
needle-finding number in the project. **The reduction is what produced it:**
0.0356 -> 0.4683, and 0.00 -> 0.60 at the top of the list. On this complex the
L2 norm was indistinguishable from noise, exactly as entry 31's
concentration-of-measure argument predicts.

### But 1JTG says the opposite, and that is unresolved

Same artifact, both reductions, so the comparison is clean:

| complex | positions | base rate | L2 | simplex |
|---|---|---|---|---|
| 3S9D A | 50 | 12% | +0.0356 | **+0.4683** |
| 1JTG B | 28 | 43% | +0.2946 | **+0.1714** |

The reduction helps enormously on one complex and hurts on the other. Do not
report either in isolation. Candidate explanations, none yet tested: 1JTG's 43%
hot-spot prior compresses the dynamic range; 3S9D has far denser positional
coverage; 1JTG is an enzyme-inhibitor pair and 3S9D a different interface
class. 2JEL (P, 32 positions, 25%) and 3HFM (Y, 15 positions, 33%) were queued
the same session to break the tie.

### Honest caveats, recorded before the tie-break landed

- **The partial's CI still grazes zero** (-0.0109). "Adds signal beyond
  geometry" is suggestive, not established.
- **Confounds still win head-to-head on 3S9D**: `burial` +0.6411 vs |ddG|
  against the attribution's +0.4683; `distance_to_partner` -0.5769;
  `rsa_complex` -0.5698; `delta_rsa` +0.5620.
- **Completeness is out of tolerance**: 8.51% relative on 3S9D, 5.99% on 1JTG.
  Real geometry roughly tripled the `f(x) - f(baseline)` span, so m=32 is now
  too few quadrature steps for the absolute 0.10 gate. Re-run the survivors at
  m=64 before quoting anything as final.
- **A prediction registered before the run failed.** It was predicted that the
  global correlation would stay at or below chance and that the gradient would
  track `residue_volume` most strongly. Both wrong on 3S9D: the correlation
  clears the null, and the strongest associations are `rsa_complex` (-0.3815)
  and `burial` (+0.3322). Recorded because a prediction only counts if its
  failure is written down too.
- **1JTG was the weakest available test and was chosen for cache convenience.**
  At a 43% base rate a perfect top-5 is a 1-in-125 event; on 3S9D it is
  1-in-300,000. Complex selection must be justified on positional coverage and
  base rate from here on, never on what is already featurised.
