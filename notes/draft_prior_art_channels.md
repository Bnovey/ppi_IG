# Prior art: the sequence channel vs the structural channel — DRAFT, 2026-10-14

> **STATUS: EVERY CITATION BELOW IS [UNVERIFIED].** This note is a transcription
> of a literature search run in-session on 2026-10-14. The searching agent
> reported that it verified these sources; **that verification has not been
> independently confirmed, and several entries postdate the searching model's
> training cutoff.** Fabricated or mis-dated references are a known failure
> mode of LLM literature search, and a missed or invented citation is the one
> error that sinks a benchmark paper.
>
> **Before any of this reaches an abstract: resolve every DOI/PMID/arXiv id by
> hand and confirm each paper says what is claimed here.** Entries that fail to
> resolve should be deleted, not softened.

## The question asked

Boltz-2 receives residue identity through two partly redundant channels: the
categorical one-hot `res_type`, and atom-level features plus coordinates built
from that residue's atom table. Gradients can only flow through the first. Has
anyone published this decoupling, or benchmarked gradient attribution against a
brute-force mutation scan on a folding model?

## Reported findings

### 1. Channel decoupling — partially addressed
Prior work traces information flow between sequence and pair representations,
and ablates the MSA, but does not isolate the one-hot channel from the
atom-level features derived from it. Our `resswap` isolation (one-hot swapped,
geometry frozen, median |delta| 0.0120 on 3S9D) is reported as the novel part.

### 2. Structure-conditioned sequence dependence — reverse direction covered
The inverse is well established: ProteinMPNN recovers 52.4% of sequence from
coordinates alone (90-95% core), ESM-IF1 51% (72% buried) — already cited in
section 21. The forward direction — how much a structure-conditioned model's
output depends on the explicit sequence token once coordinates are pinned —
is reported as unquantified.

### 3. Sequence-structure inconsistency — framework exists, protein analogue absent
Majdandzic, Rajesh & Koo (Genome Biology 24:109, 2023) establish off-simplex
gradient contamination and state it covers proteins; already cited in section
21. **No protein-structure analogue found** for feeding a model a residue label
that disagrees with its coordinates. This bears directly on whether our
refeaturise number means what we want it to — see the open control below.

### 4. Attribution faithfulness on structure predictors — THE GAP
Genomics has TISM (Sasse, Chikina & Mostafavi, iScience 27(9) 2024,
PMC11404212): gradients vs true in-silico saturation mutagenesis, mean Pearson
0.70, median 0.73, ~160x speedup. **No protein-structure equivalent found.**
Nearest attempts validate attributions against hand-picked hot spots rather
than against a scan.

This is the gap sections 24-26 occupy, and TISM's 0.70 is the number this work
will be measured against.

### 5. Terminology — no canonical term
Candidates considered and rejected as inexact: "gradient saturation" (Miglani
et al., arXiv:2010.12697 — single-region, not dual-channel), "off-manifold
effects" (simplex violations), "shortcut learning" (spurious correlations).
Closest usable term: **"input channel redundancy"**, borrowed from multimodal
learning, used with an explicit definition.

## Entries flagged as highest fabrication risk

These were cited with identifiers that postdate the searching model's cutoff
and could not be cross-checked from prior knowledge. **Resolve these first:**

- Lu et al., "Two Stages of Folding", NeurIPS 2026, arXiv:2602.06020
- Jedryszek et al., arXiv:2608.11475
- Yao et al., arXiv:2606.22181 (IG on ESM-2)
- AlphaInterp, PMC13131572
- IGMI, PMC13070472
- Migliorini et al., PairSAE, MLSB 2025, arXiv:2606.27440
- Singh et al., BALM-PPI, bioRxiv 10.64898/2026.03.30.715237
- AF3 Structure-guided Ligand Discovery, bioRxiv 2025.12.04.692352

Lower risk (predate cutoff, consistent with prior knowledge, and already cited
elsewhere in this repo): TISM/PMC11404212, Majdandzic/Genome Biology 24:109,
ProteinMPNN/PMC9997061, ESM-IF1/bioRxiv 2022.04.10.487779,
Miglani/arXiv:2010.12697.

## The control this makes urgent

The claim "the structural channel carries most of a mutation's effect" rests on
a 36x median-|delta| gap (0.4331 refeaturise vs 0.0120 resswap, Spearman
between them -0.0744). **That comparison is confounded**: refeaturise rebuilds
the mutant's atom features but reuses wild-type coordinates, so the model sees
an atom list that disagrees with its geometry — an input no real protein
presents. Finding 3 above says there is no published characterisation of how a
protein model behaves off that manifold, so we cannot borrow an answer.

**Run the coordinate-consistent control** (re-predict geometry for each mutant)
before stating the channel claim. If the gap survives, the claim holds; if it
collapses, the measurement was of the model's response to a malformed input.
