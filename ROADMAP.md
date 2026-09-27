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

### Phase 3 -- brute force (~$22)

Scan **all 19 substitutions at the 49 measured positions = 931 mutants**. The
"shortcut vs slow way" comparison needs no lab data, only the model, so scan as
widely as affordable. A full saturation scan of chain B (165 x 19 = 3,135) is
the eventual ideal at roughly another 20 GPU-hours -- hold it until Phase 1
shows something.

### Phase 4 -- compare

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

Committed through `b3d3763`. 410 tests, ruff 11.

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

### Setup facts to verify before spending GPU

- Chain assignment (A = ACE2, E = RBD) and the RBD residue range.
- Dataset Spike-site numbering maps directly onto 6M0J chain E author numbering,
  with **zero** wild-type mismatches. The `Mutation(s)_cleaned` disaster on 1JTG
  (41 mismatches, silently wrong residues) is the precedent: never trust a
  numbering column, always check the wild-type letter against the structure.
- Sign convention. Starr `bind_avg` is *negative for weaker binding*; SKEMPI ddG
  is *positive for weaker binding*. The conversion must negate.
- Censoring floor, and specifically whether the strongest positions are pinned
  at it the way GB1's positions 27 and 43 are.
- 6M0J carries a Zn ion and NAG glycans; our parser keeps only standard `ATOM`
  residues, so confirm what is being dropped and that it is not at the interface.

### Next, in priority order

1. **Pool 3HFM, 1VFB, 1JRH, 2JEL** (~$12, ~2.5 h). All registered and verified.
   Takes n from 28 to ~128 and directly tests whether 0.357 survives
   replication. Do this **before** the scan: if the effect does not reproduce,
   0.357 was n=28 noise and we learn it cheaply.
2. **The brute-force scan on 1JTG** (~$22, ~4 h). Mandatory — the headline claim
   is unpublishable without it, and it separates "the method fails" from
   "Boltz-2 has no binding signal here". Only one of those is fixable.
3. **The saturation arm on 6M0J.** Loader and within-position analysis are
   local and free; build and verify them alongside 1 and 2, then spend GPU.
4. **Re-gate completeness** on absolute error, or on relative error scaled to
   the score's own noise. Do this instead of spending an hour on m=64.

Not yet justified: a different attribution method. Item 2 is what establishes
whether the method is at fault.
