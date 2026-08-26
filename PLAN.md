# Does gradient attribution work on co-folding models?

A validation paper. One question, one experiment, publishable either way.

Confidence markers: **[V]** verified from the primary source. **[S]** from a
search snippet, not yet confirmed. **[INF]** our inference.

---

## 1. The question

When you run integrated gradients on Boltz-2 (or Chai-1, or Protenix) and it
tells you "position 7 matters," is that true?

Nobody has checked. People are already using attribution on these models to
guide design. If the signal is noise, that matters. If it works, it replaces a
slow brute-force scan with a single backward pass.

## 2. What "true" means — two separate questions

These get conflated constantly. They have opposite fixes, so measure them apart.

**Question A: is IG faithful to the model?**
Does IG agree with what the model itself says when you actually mutate residues
and re-score? Ground truth = brute-force mutation scan on the same model.

**Question B: is the model right about biology?**
Do those sensitivities match measured binding energies? Ground truth = SKEMPI
experimental ddG values.

Why the split matters:

| A | B | Meaning | What you do |
|---|---|---|---|
| pass | pass | IG is a cheap, accurate mutational scan | Use it. Best case. |
| pass | fail | IG is honest, the model is wrong | Trust attribution, fix the scoring model |
| fail | — | IG is not reading the model correctly | Drop IG, brute-force instead |

**[INF]** The genomics precedent (TISM) only tested A. Adding B, on proteins, is
the new part.

---

## 3. Prior art

**The template we're copying.** Sasse, Chikina, Mostafavi, "Quick and effective
approximation of in silico saturation mutagenesis experiments with first-order
Taylor expansion," *iScience* 27(9), Aug 2024 (PMC11404212).
- **[V]** Approximates a full mutation scan from gradients:
  `ISM(s0,l,b1) ~ df(l,b1)/ds0 - df(l,b0)/ds0`
- **[V]** Mean Pearson vs the real scan **0.70**, median **0.73**, 87% of cases
  above 0.6. Speedup ~160x.
- **[V]** "TISM's correlations to ISM are consistently higher than those of the
  gradient itself."
- **[V]** Limits: first-order only, struggles with non-linear effects; ReLU
  models dropped to 0.61.
- **[V]** Genomics only. Never proteins. Never compared to experiments.
- **[INF]** 0.70 is the number to beat.

**The finding that predicts half our result.** Nori, Parsan, Uhler, Jin,
"BindEnergyCraft," arXiv:2505.21241, May 2025.
- **[V]** *"ipTM computes a maximum over target residue indices, resulting in
  sparse gradients that constrain optimization to only a small subset of
  interface residue pairs."*
- **[V]** *"The ipTM score applies a hard max_i across reference residues when
  aggregating alignment confidence. This operation zeroes out gradients from all
  but the single target residue that achieves the maximum binder alignment."*
- **[V]** Their fix: treat pAE logits as an energy function via LogSumExp
  (`pTMEnergy`) to get a dense signal.
- **[INF]** So attribution on ipTM should fail, and attribution on Boltz-2's
  dense affinity head should work. That is a real prediction, stated before we
  run anything.

**Closest interp work on the same model.** Migliorini et al., "PairSAE,"
arXiv:2606.27440, Jun 2026.
- **[V]** Sparse autoencoders on Boltz-2 pair representations, checked against
  UniProt annotations and affinity prediction.
- **[V]** No gradient attribution anywhere in it.
- **[INF]** Cite as concurrent and non-competing. It leaves our slot open.

**Controls we owe.** Adebayo et al., "Sanity Checks for Saliency Maps,"
NeurIPS 2018. **[S]** Randomize model weights; a real attribution method should
change. Several popular methods don't.

**The off-manifold problem.** Majdandzic et al., "Correcting gradient-based
interpretations of deep neural networks for genomics," *Genome Biology* 2023.
**[S]** Gradients point in directions that aren't valid sequences. Their
correction projects back onto the data manifold.

---

## 4. The implementation change that makes this possible

Right now the code produces one unsigned number per position — an L2 norm over a
384-dimensional embedding gradient. You cannot compare that to a ddG, because it
has no sign and no per-mutation resolution.

The fix is small. You already compute the gradient of the score with respect to
the embedding, `g_l` (384 numbers per position). To get the predicted effect of
swapping residue `a` for residue `b` at position `l`:

```
delta(l, a -> b) = g_l . (E[b] - E[a])
```

where `E` is the model's (20, 384) amino-acid embedding matrix. Dot the gradient
you already have with the embedding difference. That's it — about ten lines, and
it's exactly TISM's formula with a 20-letter alphabet instead of four.

Output becomes an `(L, 20)` signed matrix: predicted score change for every
substitution at every position. Directly comparable to a real mutation scan and
to experimental ddG.

Two versions to compare:
- **Plain gradient** (one backward pass): `g_l` at the real sequence.
- **Integrated gradients** (m backward passes): average `g_l` along the path
  from a baseline sequence to the real one, then dot as above.

**[INF]** TISM found the plain-gradient version beat raw gradients but they never
tested IG. If IG doesn't beat the single-pass gradient here, that itself is worth
reporting — it would mean the extra 10-50x compute buys nothing.

---

## 5. Experiments

### 5.0 Sanity checks first (a few hours, under $5)

Don't skip these. If any fail, nothing downstream means anything.

| Check | How | Pass |
|---|---|---|
| Completeness | Do the attributions sum to `F(x) - F(baseline)`? | within 5% |
| Step count | Sweep m = 8, 16, 32, 64, 128, 256. Rank-correlate neighbors. | stable by m=32 |
| Random weights | Re-run with a randomly initialized model | attribution should fall apart. If it looks the same, IG is reading input geometry, not the model |
| Dead score | Attribute a constant target | should go to zero |
| Frozen trunk | Boltz-2 full-backprop vs `stopgrad(trunk)` | measure the gap |

The random-weights check is cheapest and the most likely to kill the project.
Run it first.

### 5.1 Question A — IG vs a real mutation scan (1-2 days, ~$20)

**Complexes:** ~20 peptide-protein structures. Peptide chain 8-25 residues,
resolution under 2.5 A, spread across interface sizes and target families.
Source: PepBDB or Propedia.

**Ground truth:** for each complex, mutate every peptide position to all 19
other amino acids and re-score. That's `L x 19` predictions. For a 15-mer,
285 predictions at ~12 s each is under an hour. Batch them.

**Prediction:** the `(L, 20)` matrix from section 4.

**Compare:**
- Spearman over the whole flattened matrix, per complex. Report the
  distribution, not just a mean.
- Position-level agreement: sum absolute values per position, then correlate.
  This is what position-locking actually used.
- Precision@K for the top-K positions (K = 3, 5, 8).

**Run each scorer separately.** Boltz-2 affinity, Boltz-2 ipTM, Chai-1 ipTM,
Protenix ipTM. This is the core of the paper — the prediction from section 3 is
that the dense affinity head works and the ipTM heads don't.

### 5.2 Question B — do the sensitivities match experiments (~1 week, ~$50)

**Data:** SKEMPI 2.0 single-point mutations in complexes with solved structures.
**[S]** roughly 3,000 of them; hotspots defined at ddG >= 2 kcal/mol.

**Honest scoping problem:** SKEMPI is mostly protein-protein, not
protein-peptide. **[ ]** Check how many entries have a chain under 30 residues.
If it's thin, split the paper: peptides for Question A, general protein-protein
for Question B, and say so plainly.

**Metrics:**
- Spearman, predicted vs measured ddG.
- AUROC for hotspot classification at both 1 and 2 kcal/mol cutoffs.

**Baselines it has to beat.** This is where most attribution papers quietly
fail, so put them in the main table, not an appendix:
1. random
2. relative solvent accessibility alone
3. hydrophobicity alone
4. distance to the partner chain / buried surface area
5. the real mutation scan on the same model (the ceiling IG is approximating)

**[INF]** If IG only recovers "this residue is buried at the interface," it
learned geometry you already had from the structure for free. Beating baseline 2
is the actual bar.

### 5.3 Cheap extra — where does attribution land? (free, run with 5.1)

Attribute onto the *target* protein instead of the peptide. Measure the share of
attribution mass within 5 A of the partner ("interface attribution fraction").
Compare to the random expectation, `interface_size / L`.

Doubles as a practical filter: if the mass sits away from the pocket you
designed against, the model likes the design for the wrong reason.

---

## 6. Contamination check (do this before trusting anything)

PepBDB was last updated **March 2020** **[S]**. Boltz-2 was almost certainly
trained on those structures. Memorized complexes make the results look better
than they are.

**[ ]** Find Boltz-2's training cutoff date.
**[ ]** Assemble a held-out set of complexes deposited after it (~5-10 is enough
to show the trend holds).
**[ ]** Report both sets separately. If the held-out numbers are much worse, that
is the paper's most important figure.

---

## 7. What the paper looks like

Short. Workshop length, 4-6 pages, 3 figures.

- **Fig 1** — IG vs real mutation scan, one panel per scorer. The money figure.
  Prediction: affinity head correlates, ipTM heads don't.
- **Fig 2** — agreement with SKEMPI ddG, with all five baselines on the same
  axes.
- **Fig 3** — the sanity checks, especially random weights, plus the m-sweep.
- **Table 1** — cost: backward passes vs full scan, wall clock, dollars.

**Framing if it works:** attribution gives you a mutational scan ~100x cheaper,
and here's the validation nobody had done.

**Framing if it fails:** attribution on co-folding confidence scores doesn't
recover binding determinants, and here are the two mechanisms — the ipTM max
operator zeroing gradients, and off-manifold gradient directions. Useful, because
people are already building on this signal.

Either way the contribution is the same: the first validation of gradient
attribution on co-folding models against both the model's own behavior and
experimental data.

**[ ]** Pick a venue. ICLR/NeurIPS MLSB workshop is the natural home. Check
deadlines.

---

## 8. Cost and time

| Stage | Time | GPU cost |
|---|---|---|
| Sanity checks | hours | < $5 |
| Question A (20 complexes) | 1-2 days | ~$20 |
| Question B (SKEMPI) | ~1 week | ~$50 |
| Held-out contamination set | 1 day | ~$10 |
| **Total** | **~2 weeks** | **under $100** |

At roughly $0.40/GPU-hr on a Modal H100. There is no cost reason to skip any of
this.

---

## 9. Risks

- **The random-weights check fails.** IG was never reading the model. Kills the
  positive framing but is itself a publishable finding, and it's a few hours to
  find out.
- **All four scorers fail.** Weaker paper, still real, and the ipTM mechanism
  gives it an explanation rather than a shrug.
- **SKEMPI has too few peptide entries.** Scope Question B to protein-protein
  and state the limitation.
- **Contamination inflates everything.** Handled by the held-out set in
  section 6.
- **Someone publishes first.** The field is moving fast — PairSAE is from
  June 2026. This is ~2 weeks of work; don't sit on it.

---

## 10. Open items

- **[ ]** Boltz-2 training cutoff date.
- **[ ]** SKEMPI entries with a chain under 30 residues — how many?
- **[ ]** Does Boltz-2 expose the amino-acid embedding matrix `E` cleanly?
- **[ ]** Confirm Chai-1 and Protenix really do apply the hard max the way
  BindEnergyCraft describes — read their ipTM code, don't assume.
- **[ ]** Get Adebayo and Majdandzic full texts; upgrade **[S]** to **[V]**.
- **[ ]** Venue and deadline.

---

## 11. Where the code lives

Attribution implementation to port from: `../IG_Agros/src/ig/` — `core.py`,
`boltz2.py`, `chai1.py`, `protenix.py`. The `(L, 20)` change from section 4 goes
in whatever computes the final per-position norm.

Background on the campaign that motivated this: `../IG_Agros/PAPER_PLAN.md`.
