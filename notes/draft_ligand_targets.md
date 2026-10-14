# Draft: additional protein-ligand targets for the affinity-head IG arm

Opened 2026-10-12. Research pass only -- nothing here has been run.

Purpose: section 17-20 has exactly one target (ABL1 kinase domain + imatinib,
2HYY, Hauser 2018). A paper claim about the model needs 3+. This note ranks
candidate systems, records what was verified and how, and specifies the
data-acquisition steps for a fetch script in the shape of `17_fetch_abl.py`.

## Headline, stated before the table

**Two systems clear 20 positions, and they fail in opposite directions.** The
brief asked for 3-5 targets with 10+ (ideally 20+) single point mutations at
distinct positions, with measured affinity for a fixed ligand. After
exhausting the Platinum database (the only systematic source of exactly this
data type) and the resistance-mutation literature:

* **EGFR + 18 TKIs** (Robichaux 2021): 27 singles at **20 positions**, the
  best near/far spread of any candidate (3.3-23.1 A), machine-readable, and
  with a **measured replicate noise floor of 0.08 log10** against a 5-log10
  response range. The readout is Ba/F3 cell viability -- Hauser's assay class
  -- but it is far more reproducible than Hauser's, and unlike Hauser the
  noise can be measured from inside the dataset.
* **SARS-CoV-2 Mpro + nirmatrelvir** (Hu 2023): **102 single mutants** with
  enzymatic **Ki**, but at only **12 positions**, all of them in the drug site.
  Its value is the opposite of a position count -- roughly **8.5 substitutions
  per position**, which is the only dataset found anywhere in this search that
  **restores within-position ranking**, the trick section 17 recorded as lost.
* ***Candida albicans* NMT** + a Roche inhibitor: **25 positions**, clean
  enzymatic **Ki** -- but it is an **alanine scan** and **92% of its positions
  sit within 6 A of the ligand**, so it has no far arm at all.
* 3 further systems clear 10 positions (13, 13, 10), all biochemical Kd.
* HIV-1 protease, the obvious guess, **fails**: the largest single-mutant
  panel for any one inhibitor is 8 positions. Checked twice, independently.
* **No kinase other than ABL1 has 10+ single mutants with a *biochemical*
  readout.** The largest biochemical kinase resource (Davis 2011) tops out at
  5 single mutants for EGFR, KIT and FLT3 each.

So the honest framing is that **no single target is strong on every axis, and
the three axes trade against each other**: position count, assay quality, and
spatial spread. The paper gets 3-4 targets, but each is strong on a different
axis and the write-up has to say which. The diversity requirement is
comfortably met -- Mpro, CaNMT and PimA are all non-kinases.

A fourth axis matters more than the brief implied and is worth stating
separately: **within-position substitution depth**. Section 17 found that
Hauser's within-position ranking failed because only 6 of 20 positions carried
more than one mutant amino acid, and that this killed the trick that made the
Starr arm clean. Mpro fixes it outright (~8.5 substitutions per position,
Q192 alone has 11). That is a stronger reason to run Mpro than its raw
mutation count, because within a position burial, RSA, conservation and
position index are all **constant**, so the confound panel cancels
structurally rather than statistically -- the same argument section 17 used to
justify the within-mutation across-inhibitor design, applied to the mutation
axis instead.

A second, separate finding that affects the baseline story is in
[Leakage and baseline contamination](#leakage-and-baseline-contamination):
**every Platinum-derived target is inside mCSM-lig's training set**, so
mCSM-lig cannot serve as an independent baseline on candidates 3-8. EGFR and
Mpro were checked against all 1008 Platinum rows and are external (Mpro
completely; EGFR but for one mutation), which is a further reason to run those
two first. All candidates, including these two, are structurally leaked into
Boltz-2's training set -- every WT structure predates the 2023-06-01 cutoff.

## Where the data came from

Two productive sources: the **Platinum database** for biochemical Kd/Ki
panels, and **Robichaux 2021** for the EGFR cell-based panel. Everything else
searched (HIV-1 protease/RT, Stanford HIVDB, BindingDB, Davis/Karaman
KINOMEscan, ALK, KIT, BRAF, FLT3) is recorded under
[Rejected, with the reason](#rejected-with-the-reason).

### Source A -- Platinum

The **Platinum database** (Pires, Blundell &
Ascher, *Nucleic Acids Research* 43:D387-D391, 2015,
[doi:10.1093/nar/gku966](https://doi.org/10.1093/nar/gku966),
[PMC4384026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4384026/)).

It is a manually curated set of affinity measurements for mutants of
*structurally defined* protein-ligand complexes, with the curation rule that
wild-type and mutant must be measured **by the same group under the same
conditions** -- which is the inter-lab-noise control that Hauser's pooled
IC50 set does not have.

Bulk download, verified live 2026-10-12:

```
https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv        # 687 KB, 1008 data rows
https://biosig.lab.uq.edu.au/platinum/static/platinum_processed_pdb_files.tar.gz
```

There is no `/platinum/download` route (404); the links live on
`https://biosig.lab.uq.edu.au/platinum/data`.

**The flat file already carries almost every field this project needs**, which
is why it beat every other route. Columns include `mutation`, `affin.lig_id`
(the PDB CCD code), `affin.k_wt`, `affin.k_mt`, `affin.fold_change`,
`affin.unit`, `affin.exptal_method`, `mut.wt_pdb`, `mut.is_single_point`,
`mut.distance_to_lig`, `mut.in_binding_site`, `mut.residue_rsa`,
`mut.doi`, `mut.pmid`, `prot.resolution`, and `lig.atom_count`.

Composition, computed from the downloaded file (not quoted from the paper):

| quantity | value |
|---|---|
| data rows | 1008 |
| single point mutations | 797 |
| multi-point (excluded) | 211 |
| affinity units | KD 560, Ki 413, Km 35 |
| assay types | all biochemical -- kinetic 355+60+14, ITC 181, fluorescence variants ~200, SPR 22 |

**Note the assay quality relative to Hauser.** Platinum is Kd/Ki from ITC,
enzyme kinetics and fluorescence. Hauser's 131 of 144 are *cell-viability*
IC50 with a measured ddG-vs-ddG RMSE of 0.81 kcal/mol (section 17). Any
Platinum target therefore has a **lower noise floor on B-vs-C** than the
target we already have. That is the strongest argument for this route and it
is independent of the mutation counts.

### Source B -- Robichaux 2021, the EGFR panel

Robichaux JP et al., "Structure-based classification predicts drug response in
EGFR-mutant NSCLC", *Nature* 597:732-737, 2021,
[doi:10.1038/s41586-021-03898-1](https://doi.org/10.1038/s41586-021-03898-1),
PMID 34526717, PMC8481125.

Supplementary Table 4, fetched and parsed 2026-10-12:

```
https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03898-1/MediaObjects/41586_2021_3898_MOESM4_ESM.xlsx
76 KB .xlsx, sheets: "Panel A", "Panel B", "Panel C"
"Panel A" = 77 mutation rows x 59 columns
```

(The PMC mirror of this file returns HTML; use the Springer URL above.)

Layout of Panel A, read directly from the workbook: columns 1-3 are exon
numbers, column 4 is a structure-function group label, column 5 is the EGFR
mutation, and columns 6-59 are **18 TKIs x 3 replicates**. The 18 are
erlotinib, gefitinib, AZD3759, sapatinib, afatinib, dacomitinib, neratinib,
poziotinib, Tarlox-TKI, CLN-081, AZ5104, mobocertinib, osimertinib,
nazartinib, olmutinib, rociletinib, naquotinib, lazertinib.

Counts, recomputed from the file rather than taken from the paper:

| quantity | value |
|---|---|
| mutation rows | 77 |
| **single missense substitutions** | **27** |
| **distinct positions** | **20** |
| **positions with >1 mutant amino acid** | **7** (709, 718, 719, 747, 776, 833, 861) |
| excluded | 50 rows: Ex19del, insertions, duplications, and doubles such as `E709K/L858R` |

The 27 singles: E709A, E709K, L718Q, L718V, G719S, G719A, S720P, G724S,
T725M, L747P, L747S, K754E, K757R, D761N, S768I, V769L, V774M, R776H, R776C,
S784F, T790M, S811F, L833F, L833V, L858R, L861R, L861Q.

**Three properties make this better than Hauser on the mutation axis**, which
is the axis section 17 found Hauser could not carry:

1. **7 of 20 positions carry >1 substitution**, against Hauser's 6 of 20. Not
   a large gain, but it moves the right way, and the pairs are chemically
   informative (R776 H vs C, L833 F vs V, L718 Q vs V).
2. **18 inhibitors, against Hauser's 6-8.** The within-mutation
   across-inhibitor design that section 17 identified as the clean one -- the
   regime where burial, RSA and residue identity cancel structurally -- gets
   18 points per mutation instead of 6. That is the single biggest
   improvement available anywhere in this note.
3. **3 replicates per mutation-drug pair.** Hauser's assay noise had to be
   imported from a separate n=27 cross-check (RMSE 0.81 kcal/mol). Here the
   noise floor is **measurable inside the dataset**, which turns the "is this
   result inside the error bars" question from an argument into a computation.

The readout is `log(IC50_mutant / IC50_WT)` from a **Ba/F3 viability** assay --
already a ratio, so no separate WT row is needed, and no RT conversion is
involved.

**Sign and base, settled from the file rather than assumed.** Positive =
resistance, matching the SKEMPI convention already in use, so **no sign flip
is needed anywhere**. Verified against four cases whose direction is known
independently:

| mutation | erlotinib | gefitinib | osimertinib | known biology |
|---|---|---|---|---|
| T790M | **+1.56** | **+1.26** | **-0.76** | gatekeeper: resists 1st-gen, *sensitive* to osimertinib, which was designed for it |
| L858R | -1.17 | -0.65 | -1.58 | the classic *sensitising* mutation |
| L718Q | +0.79 | +0.97 | **+1.03** | known osimertinib-resistance mutation |
| S811F | -1.35 | -1.07 | -2.46 | sensitising |

All four come out in the right direction, including the sign *reversal* of
T790M between first-generation inhibitors and osimertinib. That reversal is a
qualitative anchor of exactly the kind section 17 named for ABL1
("ponatinib retains potency against T315I while imatinib fails"), and it is a
stronger one because it is a sign change rather than a magnitude change. **A
model that cannot reproduce T790M's erlotinib-versus-osimertinib flip has no
business being attributed.** The base is almost certainly log10 (T790M at
+1.56 is ~36-fold, which is the right order for this mutation; ln would make
it ~5-fold, which is too small), but confirm it in the methods.

**The noise floor is far better than Hauser's, and this is the finding that
most changes the ranking.** Computed from the 3 replicates, over all
27 singles x 18 drugs = **486 cells, with zero missing values**:

| quantity | value (log10 units) |
|---|---|
| **median replicate SD** | **0.080** |
| mean replicate SD | 0.162 |
| 90th percentile | 0.413 |
| response range | -3.17 to +2.00 |
| SD of the response itself | 0.97 |
| **response SD / median replicate SD** | **12.2** |

I had initially written this off as "Hauser's assay class, with Hauser's noise
floor". **That was wrong and the data says so.** Hauser's figure was RMSE 0.81
kcal/mol (~0.6 log10) with inter-lab repeatability 0.32 kcal/mol (~0.23
log10); the EGFR panel's median replicate scatter is **0.08 log10**, against a
response that spans 5 log10 units.

**The caveat that keeps this honest:** 0.08 is *technical repeatability within
one study*, whereas Hauser's 0.81 was *validity* -- ddG-from-IC50 against
ddG-from-Kd, a different and harder quantity. A tight replicate SD says the
assay is reproducible; it does not say cell-viability IC50 tracks binding
affinity. The systematic IC50-vs-Kd gap is unmeasured here and is not bounded
by this number. So the right claim is narrow: **EGFR's random noise is small
and measurable, its systematic bias is unknown.** That is still strictly more
than is known about Hauser, where the random component could not be separated
at all.

### Source C -- Hu 2023, the SARS-CoV-2 Mpro panel

Hu Y, Lewandowski EM, Tan H, et al., "Naturally Occurring Mutations of
SARS-CoV-2 Main Protease Confer Drug Resistance to Nirmatrelvir", *ACS Central
Science* 2023,
[doi:10.1021/acscentsci.3c00538](https://doi.org/10.1021/acscentsci.3c00538),
PMID 37637734, PMC10451032. Open access, confirmed via the Europe PMC record.

Verbatim from the abstract, which I fetched rather than inferred:

> we systematically characterized 102 naturally occurring M^pro mutants
> located at 12 residues at the nirmatrelvir-binding site, among which 22
> mutations in 5 residues, including S144M/F/A/G/Y, M165T, E166 V/G/A,
> H172Q/F, and Q192T/S/L/A/I/P/H/V/W/C/F, showed comparable enzymatic activity
> to the wild-type (k_cat/K_m < 10-fold change) while being resistant to
> nirmatrelvir (K_i > 10-fold increase).

**Read "12 residues" carefully -- this is the number that decides how to use
the dataset.** A first pass over this paper reported the panel as probably
"60+ distinct positions"; it is not. It is **102 mutants at 12 positions, all
of them in the drug-binding site**, selected from GISAID. So:

| quantity | value |
|---|---|
| single mutants characterised | **102** |
| distinct positions | **12** |
| mean substitutions per position | **~8.5** |
| Q192 substitutions alone | **11** (T, S, L, A, I, P, H, V, W, C, F) |
| confirmed resistant | 22 mutations at 5 positions |
| enzymatically dead (no usable Ki) | 4: H41M, H41T, H41Y, H163W |
| assay | **FRET enzymatic Ki**, biochemical |

The substitution depth is the point. Hauser offered within-position ranking at
6 of 20 positions and EGFR at 7 of 20; Mpro offers it at essentially 12 of 12,
with up to 11 amino acids to rank at a single site.

**The data are in a PDF supplement and this is the one real cost.** The file
is `oc3c00538_si_001.pdf` (Table S1 holds the per-mutant enzymology and
inhibition; S2-S4 are secondary), confirmed to exist by filename on the PMC
article page. **Both PMC and ACS block scripted download** -- PMC returns a
"Preparing to download" interstitial and ACS returns a Cloudflare page, and
the Europe PMC `supplementaryFiles` endpoint returned a truncated archive on
two attempts. A human can download it in a browser in one click; a fetch
script cannot. Plan for a **manually cached PDF checked into `data/raw/`**
with a recorded checksum, plus table extraction, rather than a clean
re-fetchable download. This is the only candidate in this note with that
problem.

I did **not** obtain Table S1, so the following remain [UNVERIFIED]: the
identity of the 12 positions beyond the 5 named in the abstract; the wild-type
Ki value; the exact count of the 102 that have a usable Ki; and whether the
table reports Ki with uncertainties. **Confirm these from Table S1 before
committing GPU time** -- the case for Mpro rests on substitution depth, and
only 22 mutations at 5 positions are confirmed in text.

## Ranked candidates

All structural numbers below (residue counts, gaps, WT-residue agreement,
heavy-atom distances, ligand heavy-atom counts) were **recomputed locally**
from the RCSB coordinate files, in the same way section 17 validated 2HYY.
They are not taken from Platinum's own columns except where stated.

| # | target | organism | ligand (CCD) | heavy atoms | WT PDB | res (A) | chain len | #single muts | #distinct positions | positions w/ >1 mutant AA | assay | within 6 A of ligand | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **EGFR kinase domain** | *H. sapiens* | **AQ4** (erlotinib) / **IRE** (gefitinib) / **YY3** (osimertinib) | 29 / 31 / 37 | **1M17** / 4WKQ / 4ZAU | 2.6 / **1.85** / 2.8 | 312 / 297 / 283 | 27 (x18 TKIs) | **20** | **7 of 20** | Ba/F3 viability, log IC50 ratio, **3 replicates** | **3/20 = 15%** (3.3-21.3 A) | Robichaux 2021, [doi:10.1038/s41586-021-03898-1](https://doi.org/10.1038/s41586-021-03898-1), PMID 34526717 |
| 2 | **Main protease (Mpro/3CLpro)** | SARS-CoV-2 | **4WI** (nirmatrelvir) | 35 | **8DZ2** (dimer; *not* 7VH8 -- see below) | 2.13 | **302 (1-302, 0 gaps) x 2 chains** | **102** | 12 | **~12 of 12, up to 11 AAs (Q192)** [UNVERIFIED] | **FRET enzymatic Ki** | all 12 in drug site; the 5 named sit 2.8-3.7 A | Hu 2023, [doi:10.1021/acscentsci.3c00538](https://doi.org/10.1021/acscentsci.3c00538), PMID 37637734 |
| 3 | **N-myristoyltransferase (Nmt)** | *Candida albicans* | **MIM** | 43 | **1IYK** | 2.3 | 392 (60-451, **0 gaps**) | 26 | **25** | 1 of 25 (L451) | enzymatic **Ki** | **23/25 = 92%** (2.6-7.2 A) | Sogabe 2002, [doi:10.1016/S1074-5521(02)00240-5](https://doi.org/10.1016/S1074-5521(02)00240-5), PMID 12401496 |
| 4 | **PimA mannosyltransferase** | *Corynebacterium glutamicum* | **GDD** (and **GDP**) | 39 (GDP 28) | **3OKP** (GDP: 3OKC) | 2.0 (2.4) | 378 (4-381, **0 gaps**) | 16 per ligand, **32 total over 2 ligands** | **13** | 3 of 13 (D13, I21, E290) | **Kd**, Trp fluorescence | **5/13 = 38%** (2.7-16.7 A) | Batt 2010, [doi:10.1074/jbc.M110.165407](https://doi.org/10.1074/jbc.M110.165407), PMID 20843801 |
| 5 | **Glucoamylase A starch-binding domain** | *Rhizopus oryzae* | **GLC** (glucose) | 12 | **2V8L** | 1.8 | 106 | 13 | **13** | 0 of 13 | **Kd**, fluorescence + saturation binding | 5/13 = 38% (2.5-23.2 A) | Tung 2008, [doi:10.1042/BJ20080580](https://doi.org/10.1042/BJ20080580), PMID 18588504 |
| 6 | **Dihydroorotate dehydrogenase A** | *Lactococcus lactis* | **ORO** (orotate) | 11 | **2DOR** | 2.0 | 311 | 10 | **10** | 0 of 10 | **Kd**, competition | 5/10 = 50% (2.8-10.7 A) | Norager 2003, [doi:10.1074/jbc.M303767200](https://doi.org/10.1074/jbc.M303767200), PMID 12732650 |
| 7 | **Protein O-fucosyltransferase 2** | *C. elegans* | **GDP** | 28 | **3ZY2** | 1.54 | 362 | 9 | **9** | 0 of 9 | **Kd**, ITC | 7/9 = 78% (2.8-8.5 A) | Lira-Navarrete 2011, [doi:10.1371/journal.pone.0025365](https://doi.org/10.1371/journal.pone.0025365), PMID 21966509 |
| 8 | **Aldose reductase (AKR1B1)** | *H. sapiens* | 10 ligands, best **ZST** | 29 | 1MAR (also 1PWM, 1US0, 2FZD, 2IKI) | 1.8 | 315 | 13 over all ligands, 8 for ZST | **8** | **1 position with 5 mutant AAs (T113 A/C/S/V/Y)** | **Kd**, ITC | 6/8 = 75% (2.9-6.6 A) | Steuber 2008, [doi:10.1016/j.jmb.2008.03.063](https://doi.org/10.1016/j.jmb.2008.03.063), PMID 18495158 |
| -- | **HIV-1 protease + darunavir** | HIV-1 | 017 (darunavir) | 38 | 2IEN | **1.3** | 99 (homodimer) | 9 | 8 | 1 (I54 M/V) | Ki | 5/8 = 62% (3.0-8.0 A) | multiple; largest single panel found |

**Every PMID in this note** -- in the table above and in the rejected list
below -- **was confirmed against the NCBI eutils record**, with title,
journal, year and first author all matching the system it is attributed to.
No DOI or PMID here is reconstructed, guessed, or carried over from a
secondary report without that check.

Two cells in the table are inferences rather than measurements and are marked
as such: Mpro's "~12 of 12 positions with >1 amino acid" is extrapolated from
102 mutants across 12 positions plus the 22 named in the abstract, and is
**[UNVERIFIED] pending Table S1**; the exact identity of Mpro's 12 positions
is likewise unconfirmed beyond the 5 named (S144, M165, E166, H172, Q192),
all of which I did verify present at 2.8-3.6 A in both 7VH8 and 8DZ2.

### Positions and distances, per candidate

Each listed mutation was mapped onto the WT coordinate file and its wild-type
residue identity checked, as was done for 2HYY:

```
EGFR, three structures, all single-chain, mutation set = the 20 Robichaux positions:
  1M17 (erlotinib, AQ4, 29 heavy): 312 res, span 672-995, 12 gaps (one block, 965-976)
        WT residue matches 20/20   <-- full coverage, requires offset -24, see below
        distances 3.3-21.3 A; 3 of 20 within 6 A
  4WKQ (gefitinib,  IRE, 31 heavy): 297 res, span 694-1020, 30 gaps
        WT residue matches 19/20; L747 is unresolved
        distances 3.4-23.1 A; 3 of 19 within 6 A
        near: T790 3.4, L718 3.5, G719 5.0
        far:  S811 23.1, L861 16.2, K754 15.6, S784 15.3, L833 13.4
  4ZAU (osimertinib, YY3, 37 heavy): 283 res, span 698-1017, 37 gaps
        WT residue matches 18/20; L747 and K754 unresolved
        distances 3.7-20.2 A; 4 of 18 within 6 A

1IYK chain A: 392 residues resolved, span 60-451, 0 gaps
             25/25 mutated positions present, WT residue matches 25/25
             MIM = 43 heavy atoms; distances to MIM 2.6-7.2 A
             only 2 positions beyond 6 A: L350 (7.2 A), I352 (6.3 A)

3OKP chain A: 378 residues resolved, span 4-381, 0 gaps
             13/13 mutated positions present, WT residue matches 13/13
             GDD = 39 heavy atoms by CCD formula
             distances 2.7-16.7 A; 8 positions beyond 6 A
             (N12 16.7, G123 14.5, D13 13.4, Q22 10.6, H118 9.5, I21 8.5, G20 6.5, G291 6.5)
```

**Zero gaps in 1IYK and 3OKP.** This removes the failure mode that section 17
caught on 2HYY, where the missing residue 275 silently shifted 14 of 20
measured positions under an additive `--resnum-offset`. For 1IYK,
`index + 60 == residue number` holds exactly; for 3OKP, `index + 4`.

**EGFR has gaps, and worse, it has a numbering trap -- this is the 2HYY
lesson repeating and it will bite anyone who does not look.** 1M17 is numbered
in the **mature-protein** frame (signal peptide removed) while 4WKQ, 4ZAU and
all clinical nomenclature (T790M, L858R, the Robichaux table) use the
**precursor** frame. The offset is exactly **-24**:

```
1M17 offsets in [-40, +40] giving >= 15/20 wild-type agreement:  [(20, -24)]
```

Exactly one offset in the whole scanned range works, and at that offset all
20 positions agree -- the same uniqueness argument section 17 used to pin
2HYY's shift at 0. So `T790` in the Robichaux table is **residue 766 in
1M17**. Applying the clinical numbers to 1M17 directly produces 0 of 20
matches, which is loud; but applying a *wrong* offset such as -18 or -19
would produce a few spurious matches and a silent misalignment, which is not.
Assert the 20/20 agreement at build time.

Note also that **1M17 covers all 20 positions where the higher-resolution
4WKQ covers 19 and 4ZAU covers 18**, because the disordered 747-751 stretch
(the exon-19 deletion region) is resolved in 1M17 and not in the others. The
higher-resolution structure is the worse structure for this purpose. That is a
genuine trade-off: 1M17 is 2.6 A, 4WKQ is 1.85 A.

## What each candidate is actually good for

They are not interchangeable, and the two leading candidates are close to
complementary.

**#1 EGFR is the strongest overall dataset, and its weakness is
interpretational rather than metrological.** 20 positions spanning 3.3 to 21.3
A with only 15% inside 6 A is the near/far contrast the brief asked for.
(ABL1 turns out to have a comparable spread, 32% inside 6 A -- see the
correction in the recommendation section -- so this is a match for the
existing target rather than an improvement on it.) With 18 inhibitors, 486 complete
mutation-by-drug cells, a measured 0.08 log10 replicate noise floor and a
built-in sign-flip anchor, it beats Hauser on every quantitative axis.

The caveat is not that the numbers are noisy -- they are not -- but that it is
**not clear what they measure**, which is the subject of the next two
paragraphs.

**But the thing that makes it spatially attractive is also its deepest
confound, and this needs saying before anyone runs it.** The far positions are
far *because they act through activation*: EGFR's distal mutations (L858R at
10.1 A, L861 at 16.2 A, S811 at 23.1 A, L833 at 13.4 A) shift the kinase's
active/inactive conformational equilibrium rather than contacting the drug.
Section 17's invalidity flag #1 says exactly this class of mutation is
**unmodellable by construction** in this repo, because we score at fixed WT
geometry on a single diffusion sample, so a conformational-selection effect
cannot be represented.

The consequence is sharp: **on EGFR, "near versus far" is confounded with
"modellable versus unmodellable".** A result showing attribution works near
the pocket and fails far from it would be exactly what you would see if the
model were perfect and the far mutations were simply outside what fixed
geometry can express. EGFR therefore cannot cleanly answer the proximity
question on its own, even though it has a wide distance range -- and neither
can ABL1, for the same reason. PimA (#4) can, because its far sites are on a
non-allosteric glycosyltransferase and are far in a substrate-binding sense
rather than a conformational-equilibrium sense. **That makes PimA the only
system in the whole set that can separate proximity from mechanism, which is
why it should not be dropped even though it has the fewest positions of the
four recommended runs.**

A further EGFR caveat on the response variable: Ba/F3 viability with
`log(IC50_mut / IC50_WT)` measures drug sensitivity of cells **whose growth
depends on the mutant receptor's own activity**. Many of these mutations
(L858R, G719S) are activating, so the ratio mixes drug binding with oncogenic
signalling strength. Hauser has the same defect; it is not a reason to prefer
one over the other, but it is a reason not to call either one "binding
affinity" in the paper.

**#2 Mpro is the only candidate that restores within-position ranking, and its
weakness is that every position is in the pocket.** 12 positions is below the
10+ bar only in the sense that it barely clears it, and all 12 are drug-site
residues, so like CaNMT it has no far arm. Judged on the position axis it is a
weak candidate.

**Judged on the axis that actually failed, it is the strongest in this note.**
Section 17's verdict on Hauser was that `within_position_spearman`
(`metrics.py:408-440`) "has nothing to rank at 14 of 20 positions", and that
the within-position trick that made the Starr arm clean "does not transfer".
Mpro transfers it: ~8.5 substitutions per position, 11 at Q192 alone. And the
argument for why that matters is already in section 17 -- within a fixed
position, burial, RSA, residue volume, conservation and position index are all
constant, so every confound that absorbed the signal in entry 38 cancels
*structurally*. No partial correlation is needed, so the degrees-of-freedom
problem that sinks every other candidate here **does not arise at all** on
this one.

The structure is also among the best in the set, with no numbering offset to
get wrong and no gap to fall into. But **the obvious choice is the wrong one,
and this was worth checking rather than assuming.**

7VH8 is the tempting entry: 1.59 A, chain A spanning 1-306 with zero gaps, a
single chain, nirmatrelvir (`4WI`) at 35 heavy atoms, all five confirmed
resistant positions present at 2.8-3.7 A from the ligand. It matches the
"single protein chain preferred" guideline exactly.

**It should not be used, because Mpro's drug site is built from both
protomers.** Measuring from the dimeric entry 8DZ2, the minimum distance from
each confirmed resistant position in chain A to *the partner chain*:

| position | distance to partner chain | |
|---|---|---|
| **E166** | **2.6 A** | **direct interface contact** |
| **H172** | **3.2 A** | **direct interface contact** |
| S144 | 7.2 A | |
| M165 | 8.5 A | |
| Q192 | 12.8 A | |

**Two of the five confirmed resistant positions make direct contact with the
second protomer.** E166 is the well-known case -- it packs against the
partner's N-finger -- and it is also one of the strongest resistance positions
in the panel (E166V/G/A). Scoring Mpro as a monomer would delete a contact
that the measured Ki depends on, at precisely the residues the result would
hinge on. That is an invalidity, not a noise source.

**Use 8DZ2 instead:** 2.13 A, two chains, `4WI` bound in *both* active sites
(35 heavy atoms each), chain A spanning **1-302 with zero gaps**, and all five
positions present at 2.8-3.6 A from the ligand. The cost is a second chain and
a lower resolution. The cost is small in practice because the affinity path
crops to ~256 tokens around the ligand anyway, so the dimer is not paid for in
full -- but **`check_crop_coverage` must be run on the dimer**, since the crop
now has to choose between two protomers' worth of residues and it is no longer
obvious that the measured positions all survive.

Costs, both real: the ground truth needs **manual PDF extraction** (above),
and 4 of the 102 mutants are enzymatically dead and must be dropped rather
than scored.

**#3 CaNMT is the only one with both the position count and a clean assay, and
its weakness is the spatial one.** 25 positions at 92% within 6 A is a *binding-site hotspot
scan*, not a near-versus-far contrast. It tests whether attribution finds the
pocket, which is close to the question `10_skempi_hotspots.py` already asks on
the PPI side, and it cannot distinguish a model that has learned binding
energetics from one that has learned "close to the ligand". With 2 of 25
positions beyond 6 A there is no far arm at all.

Two further flags on #3:

* **It is an alanine scan.** 24 of 26 mutations are X->Ala; only L451 carries
  two substitutions (A and Q). So `within_position_spearman`
  (`metrics.py:408-440`) has nothing to rank at 24 of 25 positions -- the
  **same failure that section 17 recorded for Hauser** (6 of 20 positions).
  This route does not fix that problem, it inherits it.
* **5 of 26 Ki values are right-censored** at `>50000 nM` (F115A, F117A,
  Y225A, F240A, Y256A), against a WT Ki of 250 nM. Uncensored n is **21**.
  Those 5 are the largest effects, so dropping them truncates the top of the
  response range; treating them as ordinal is the honest handling. Fold
  changes for the remaining 21 span about -87x to +3x.

**#4 PimA is the only candidate that can answer the proximity question
cleanly.** Its near/far contrast runs 2.7 to 16.7 A with 8 of 13 positions
beyond 6 A -- and unlike EGFR's, those far positions are on a
glycosyltransferase with no activation-loop equilibrium to confound them, so
"far" means far from the ligand rather than "acting by a mechanism the fixed
geometry cannot express". That is why it earns a slot despite having only 13
positions.

It is also the only candidate with a **second ligand on the same mutation
set** (GDP, 3OKC, same 16 mutations, same paper, same assay), which reproduces
the within-mutation across-inhibitor design at 2 ligands rather than 6.

**#8 aldose reductase is the only candidate that restores within-position
ranking**: T113 was mutated to A, C, S, V and Y, all with ITC Kd. 5 substitutions
at one position is a genuine within-position rank, and the Starr arm's clean
result came from exactly that trick. It is only 8 positions, so it is a
supporting measurement, not a target.

### The confound-identifiability problem is not solved by any of these

Section 17's correction stands and must be carried forward: identifiability is
bounded by the number of **measured positions** against a panel of 8-10
covariates, and 20 positions against 8-10 covariates is the 3HFM trap. **25
positions does not escape it.** Adding targets 2-4 at 13, 13 and 10 positions
does not pool into one larger design either, because burial, RSA and residue
chemistry are not comparable across different proteins without a random-effect
model that there is no data to fit.

What 3-4 targets *does* buy is a **replication claim** -- the same measurement
repeated on independent systems -- which is a different and weaker claim than a
confound-controlled partial correlation on one system. The paper should say
which of the two it is making. Named honestly, the available claim is: *the
affinity-head attribution behaves this way on N independent protein-ligand
systems*, with per-target numbers reported separately and never averaged.

## Leakage and baseline contamination

**Structural leakage: yes, for all of them, and it is not close.** Boltz-2's
training cutoff is 2023-06-01. Deposition dates, from the RCSB entry records:

| PDB | deposited | | PDB | deposited |
|---|---|---|---|---|
| 1IYK | 2002-08-29 | | 3ZY2 | 2011-08-17 |
| 8DZ2 (Mpro) | 2022-08-06 | | 1M17 (EGFR) | 2002-06-17 |
| 4WKQ (EGFR) | 2014-10-03 | | 4ZAU (EGFR) | 2015-04-14 |
| 3OKP | 2010-08-25 | | 2IEN | 2006-09-19 |
| 3OKC | 2010-08-24 | | 1MAR | 1993-07-20 |
| 2V8L | 2007-08-09 | | 2DOR | 1998-05-11 |
| | | | 2HYY (existing) | 2006-08-08 |

Every candidate predates the cutoff, by under a year in the closest case
(8DZ2, August 2022) and by 12-30 years for the rest. All are almost certainly
in the training set. This is flagged, not fixed -- the same position section 17
is already in for 2HYY, so it does not change the relative standing of any
candidate. It does mean no target here can support a generalisation claim.

**The baseline leakage is the one that is actually new, and it bites.**
mCSM-lig is *trained on Platinum*
([Pires & Ascher, Sci Rep 6:29575, 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4935856/)),
which section 17 already recorded as the reason Platinum's n=586 mCSM-lig
correlation of 0.76 is not a legitimate comparison point.

**This splits the candidates into two groups, and the split matters for which
baseline is usable where:**

| group | targets | mCSM-lig / PremPLI status |
|---|---|---|
| Platinum-derived | **#3 CaNMT, #4 PimA, #5-#8** | **inside the training set** -- baseline unusable |
| not from Platinum | **#1 EGFR** (Robichaux 2021), **#2 Mpro** (Hu 2023), ABL1 (Hauser) | **external** -- verified by UniProt filter against all 1008 Platinum rows; see point 2 |

Consequences:

1. Do not quote an mCSM-lig number on candidates 3-8 as a baseline. It is
   measuring its own training data.
2. **EGFR and Mpro overlap with Platinum was checked, not assumed, and they
   are effectively clean.** Filtering Platinum's 1008 rows on UniProt:

   | target | UniProt | rows in Platinum | single-mutant rows | overlap with our mutation set |
   |---|---|---|---|---|
   | **Mpro** | P0DTD1 | **0** | 0 | **none** -- Platinum contains *zero* SARS/coronavirus entries of any kind |
   | **EGFR** | P00533 | 2 | **1** (G719S) | **1 of 27** |
   | ABL1 (existing) | P00519 | 3 | 1 (T315I, ligand `627`) | 1 of 31 |

   So Mpro is **fully external** to mCSM-lig's training set, and EGFR is
   external but for a single mutation. If an mCSM-lig baseline is ever
   quoted on EGFR, drop G719S from it and say so.

   A useful by-product: Platinum's lone EGFR row gives a **biochemical Kd for
   G719S against gefitinib** (14.2 -> 31.9 nM, 2.2-fold, PMID 22349823, PDB
   2ITY), while the Robichaux cell panel gives G719S/gefitinib +0.18 log10 =
   1.5-fold. Those two agree to within a factor of 1.5 across completely
   different assay types. That is one point, not a validation, but it is the
   only direct cell-versus-biochemical cross-check available anywhere in this
   note and it points the right way.
3. The pre-registered 0.43-0.49 external band (Brief Bioinform 2026,
   PMC12874888) was measured on an *external* TKI set of n=116. Carrying that
   band over to Platinum-derived targets is an assumption, not a measurement.
   State it as the bar anyway -- it is still the right order of magnitude for
   the ML field -- but label it as transferred.

**This is a further reason to prefer #1 and #2 as the first two runs**: they
are the only candidates on which an independent ML baseline could legitimately
be quoted at all.

## Rejected, with the reason

Recording these so the same ground is not searched twice.

* **HIV-1 protease.** Checked independently two ways and it fails the same
  way both times. In Platinum, pooled over all 23 inhibitors it has only 16
  positions / 29 single mutants, and the largest panel **for a fixed ligand**
  is darunavir at **8 positions / 9 mutations**. A separate literature pass
  found the best classical biochemical panel to be Gulnik 1995 (*Biochemistry*,
  [doi:10.1021/bi00029a002](https://doi.org/10.1021/bi00029a002), PMID
  7626598): 11 mutants of which only **7 are single**, at **5 positions**
  (8, 32, 46, 82, 84; position 82 has three substitutions). Ragland 2014
  (JACS, PMID 25091085) has 3 singles. Ohtaka 2002 (PMID 12142445) has 1
  single. The field's published mutants are overwhelmingly *multi*-mutants,
  because that is what clinical isolates are. Protease is also a **homodimer**,
  so a point mutation is applied to both chains -- a second deviation from the
  single-chain setup.
* **Stanford HIV drug resistance database.** Bulk tab-delimited downloads do
  exist and were confirmed to load
  (`https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt`, also
  `NRTI_`/`NNRTI_` and `.Full.txt` variants; columns are SeqID, per-drug fold
  change, P1..P99, CompMutList). But rows are **patient isolates carrying 4-5
  mutations each**; there is no single-mutant table, and the readout is
  **fold-change IC50 from PhenoSense, a cell-based recombinant-virus assay** --
  i.e. the same assay class as Hauser's, with the same noise floor, and worse
  because the genetic background varies per row. The count of extractable
  single-mutant rows was not determined [UNVERIFIED]; expect it to be small and
  dominated by common polymorphisms.
* **HIV-1 RT + NNRTIs.** Panels run 3-6 mutants per paper. The largest found
  was Barnard 2019 (*AAC*, [doi:10.1128/aac.00676-19](https://doi.org/10.1128/aac.00676-19),
  PMID 31160281): 4 mutants at **1 position** (Y181 C/I/V/F). RT is also ~560
  residues in p66, a crop problem on top.
* **BindingDB.** Holds mutational variants, but they are not structured as
  WT/mutant pairs with a shared assay, and the advertised coverage is "110
  protein targets including isoforms and mutational variants" -- not 110
  mutant panels. Extracting a usable panel would be a curation project with no
  guarantee at the end. Platinum already did that curation.
* **Angiotensin-converting enzyme** (3BKL, 9 positions, Ki, 89% within 6 A,
  Watermeyer 2008 PMID 18457420). Good data, but the chain is **591 residues**,
  which is a crop problem well past the ~400 guideline.
* **Davis 2011 / Karaman 2008 KINOMEscan Kd panels.** These were the most
  promising-looking lead for a *biochemical* kinase panel and they do contain
  mutants -- 54 mutant-flagged entries of 442 in Davis -- but the mutation
  axis is thin for every individual kinase. Single mutants per kinase:
  **ABL1 8** (which we already have, better, from Hauser), **EGFR 5**, **KIT
  5**, **FLT3 5**, RET 3, MET 2, BRAF 1. **No kinase reaches 10.** Karaman is
  a strict subset of Davis and its supplement is PDF-only. Davis is still
  worth knowing about because it is machine-readable
  ([Supp Table 4, MOESM5](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnbt.1990/MediaObjects/41587_2011_BFnbt1990_MOESM5_ESM.xls),
  442 x 72 Kd matrix, `.xls`, needs `xlrd`; mutant flag is in
  [MOESM3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnbt.1990/MediaObjects/41587_2011_BFnbt1990_MOESM3_ESM.xls)).
  Citation: Davis MI et al., *Nat Biotechnol* 2011,
  [doi:10.1038/nbt.1990](https://doi.org/10.1038/nbt.1990), PMID 22037378.
  **Trap flagged by the search:** PMC3230241 is a *different* paper
  (Anastassiadis 2011, PMID 22037377) -- do not pull its files by mistake.
  A secondary use does exist: Davis gives **biochemical Kd for 5 KIT singles
  against both imatinib and sunitinib**, which could serve as a small
  orthogonal check, not a target.
* **ALK.** Gainor 2016 (*Cancer Discov* 6:1118-1133,
  [doi:10.1158/2159-8290.CD-16-0596](https://doi.org/10.1158/2159-8290.CD-16-0596),
  PMID 27432227) is the known panel: **11 single substitutions at 9
  positions**, only 1 position with >1 amino acid, across 5 inhibitors. Below
  the 10-position bar, the readout is cellular phospho-IC50 rather than
  affinity, and critically **the numbers exist only inside a figure image** --
  the supplements are PDFs. Transcription by hand is a fabrication risk for no
  gain over EGFR.
* **KIT, BRAF, FLT3.** No panel reaches 10 single mutants with any readout.
  Best biochemical KIT is Gajiwala 2009 (*PNAS*,
  [doi:10.1073/pnas.0812413106](https://doi.org/10.1073/pnas.0812413106), PMID
  19164557): 4 singles at 4 positions, sunitinib Kd. BRAF has essentially
  only V600E.
* **MET deep mutational scan** (Estevam 2025, *eLife*,
  [doi:10.7554/eLife.101882](https://doi.org/10.7554/eLife.101882), PMID
  39960754): ~5764 single-site kinase-domain variants against 11 inhibitors,
  data on GitHub (`fraser-lab/MET_kinase_Inhibitor_DMS`). **The mutation axis
  is enormous and this is the only candidate that would actually solve the
  degrees-of-freedom problem.** It is excluded here for one reason: the
  readout is **pooled growth-competition enrichment, not an affinity
  measurement at all**, so loss-of-function is confounded with resistance --
  the identical unresolved objection section 17 raised against promoting the
  imatinib DMS. If that objection is ever resolved (it needs a
  folding-vs-binding control in the shape of the Starr arm's expression
  readout), **MET DMS is a better target than anything in the ranked table**
  and this decision should be revisited. Flagging it rather than burying it.

## Recommendation: which 3 to run first

**Run ABL1/imatinib (existing) + EGFR/erlotinib (#1) + Mpro/nirmatrelvir (#2),
and add PimA (#4) as the fourth because it is the only clean proximity
control.**

The reasoning is that these three are chosen to *differ* along the axes that
the result depends on, rather than to maximise n:

1. **ABL1 + imatinib** -- keep. Clinical resistance mutations, cell-viability
   IC50, kinase, 20 positions. It is the only candidate whose ground truth is
   a drug-resistance phenotype, which is the application the paper is about.

   **Correction to a claim I made earlier in this note:** I had written that
   ABL1 has only 2 positions beyond 6 A. That is wrong. Recomputed from 2HYY
   chain A against imatinib (`STI`, 37 heavy atoms), of the 19 measured
   positions resolved there, **6 are within 6 A and 13 are beyond**, spanning
   3.0 to 27.2 A (E255 7.0, Q252 7.1, L384 7.3, G250 8.5, M351 8.5, E292 8.7,
   E355 9.4, F486 11.6, E279 12.2, M343 15.2, H396 15.4, D276 17.1, E459
   27.2). So **ABL1 already has a wide near/far spread** -- 32% within 6 A,
   very close to EGFR's 15% and far better than CaNMT's 92%.

   This does not change the ordering, but it does change two arguments. EGFR's
   spatial spread is no longer a *unique* selling point over the target we
   already have. And the ABL1 far sites are the same mechanistic class as
   EGFR's -- M351, H396 and E255 are classic conformational-equilibrium
   resistance mutations -- so the "near/far is confounded with
   modellable/unmodellable" problem flagged for EGFR **applies to ABL1 too,
   and always did**. That strengthens rather than weakens the case for PimA as
   a fourth run, since it remains the only system whose far sites are not
   allosteric.
2. **EGFR + erlotinib (1M17, ligand AQ4)** -- run second. It is the closest
   thing to a direct replication of the ABL1 design: same assay class
   (cell-viability IC50 ratios), same clinical-resistance framing, comparable
   position count (20 vs 20), but an independent protein, an independent lab
   and an independent drug class -- and with 18 inhibitors, a measurable
   0.08 log10 noise floor, and the T790M sign-flip anchor built in. If the ABL1 number is a property of the
   model it should reappear here; if it is a property of the ABL1 dataset it
   should not. That is the cleanest replication test available.

   Use **1M17 over the higher-resolution 4WKQ** because it resolves all 20
   measured positions against 4WKQ's 19, and apply the **-24 offset**.
   Erlotinib and gefitinib are both columns in the Robichaux table, so the
   choice of structure does not restrict the choice of drug; if 4WKQ's 1.85 A
   is preferred, drop L747 and use gefitinib (`IRE`) instead.
3. **Mpro + nirmatrelvir (8DZ2, ligand 4WI, dimeric)** -- run third, and it
   does two jobs that nothing else does.

   First it **breaks the assay confound**. Targets 1 and 2 are both
   cell-viability IC50 -- a readout whose systematic agreement with binding
   affinity is unmeasured -- so if both give weak correlations it is
   unidentifiable whether the model or the assay class is at fault. Mpro is biochemical FRET Ki, so a third weak number would point at
   the model rather than the assay -- and a *strong* Mpro number against two
   weak cell-based ones would point the other way. Either outcome is
   informative, which is the property to select on.

   Second it is **the only target that can carry a within-position result**,
   and that result needs no confound panel to be interpretable (see #2). It is
   also a non-kinase, so the diversity requirement is met here rather than
   needing a fourth run.

   Order it third rather than second only because its ground truth needs
   manual PDF extraction, so it has a human step in front of it that EGFR does
   not. If that extraction is done early it can swap to second at no cost.

**Then PimA (#4) as the fourth, and it is the one not to skip.** It is the
only clean proximity control in the set (see #4 above), and proximity is the
objection most likely to be raised against any positive result here. **CaNMT
(#3) is the fifth**, and is the one to drop if only four runs are bought: its
25 positions are its only advantage over Mpro, and it is strictly worse than
Mpro on assay, structure, substitution depth and censoring.

**A proximity control should be pre-registered before any of these run.**
CaNMT at 92% within 6 A will produce a positive correlation from proximity
alone, and EGFR's far sites are confounded with unmodellability. The control
is the one section 17 already names: regress measured effect on minimum
heavy-atom distance to the ligand, and report the attribution correlation
partialled on it. State in advance that on CaNMT a raw correlation
unaccompanied by that partial is not evidence. The per-target distance
distributions needed for this are already tabulated above, so the control
costs nothing to pre-specify.

### Do not expect these to fix the degrees-of-freedom problem

Stated plainly so it is not rediscovered later: 20, 12, 25 and 13 positions
against an 8-10 covariate panel is the 3HFM trap four more times, **on the
position axis**. The one exception is the Mpro within-position analysis, which
sidesteps the panel entirely by holding position fixed (see #2) -- and that is
the specific reason it earns a slot despite having the fewest positions. These
targets buy replication, an assay-quality contrast and a near/far contrast.
They do not buy a confound-controlled partial correlation on any single
target. The only dataset found anywhere in this search that would is the MET
deep mutational scan (~5764 variants), and it is disqualified on readout --
see the rejected list. If the degrees-of-freedom problem is judged to be the
blocking one, the right move is to solve MET's loss-of-function confound, not
to add more 13-position targets.

## Data acquisition -- what a builder needs

The shape is `scripts/21_fetch_ligand_targets.py` (21 and 22 are free;
20, 23 and 24 are taken), following
`17_fetch_abl.py`: download to `data/raw/` (gitignored), re-fetchable, write a
manifest plus `igv.provenance.write`.

Three affinity sources are needed: a Springer XLSX (EGFR), a manually cached
PDF (Mpro), and the Platinum CSV (everything else).

### Step 1a -- the EGFR affinity table (candidate #1)

```
URL     https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03898-1/MediaObjects/41586_2021_3898_MOESM4_ESM.xlsx
size    76 KB
format  .xlsx, 3 sheets; use "Panel A" (77 rows x 59 cols)
cache   data/raw/robichaux2021_supp_table4.xlsx
needs   openpyxl (already a fetch-time dependency per 17_fetch_abl.py)
```

Parsing Panel A: row 1 is the header, column 5 (`EGFR Mutation`) is the
mutation label, columns 6-59 are 18 drugs x 3 replicates with the drug name
repeated across its three columns. Keep only rows whose label matches
`^[A-Z]\d+[A-Z]$` -- that yields exactly **27** rows. Everything containing
`/`, `del`, `ins` or `dup` is a multi-mutant or indel and must be dropped (50
rows).

Values are already `log(IC50_mut / IC50_WT)`, so **no wild-type row and no
unit conversion is involved**; average the 3 replicates per drug and keep
their spread as the per-point noise estimate.

### Step 1b -- the Mpro affinity table (candidate #2, manual step)

```
article   https://pubs.acs.org/doi/10.1021/acscentsci.3c00538
          https://pmc.ncbi.nlm.nih.gov/articles/PMC10451032/
supp file oc3c00538_si_001.pdf   (Table S1 = per-mutant enzymology + inhibition)
cache     data/raw/hu2023_mpro_si_001.pdf
```

**This one cannot be scripted.** PMC serves a "Preparing to download"
interstitial, ACS is behind Cloudflare, and the Europe PMC
`supplementaryFiles` endpoint returned a truncated archive on both attempts.
Download it once in a browser, commit it to `data/raw/`, and record a SHA-256
in the manifest so the provenance chain still closes. Then extract Table S1
(`pdftotext -layout` first; fall back to `camelot`/`tabula` if the columns do
not survive).

Extraction must be checked against the abstract's own figures before use --
102 mutants, 12 positions, and the 22 named resistant mutations
(S144M/F/A/G/Y, M165T, E166V/G/A, H172Q/F, Q192T/S/L/A/I/P/H/V/W/C/F) should
all appear. If they do not, the extraction is wrong. Drop the 4 enzymatically
dead mutants (H41M, H41T, H41Y, H163W).

### Step 1c -- the Platinum affinity table (candidates 3-8)

```
URL     https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv
size    687 KB, 1009 lines (1 header + 1008 data)
format  CSV, quoted fields, 52 columns
cache   data/raw/platinum_flat_file.csv
```

No authentication, no scraping, no XLSX parsing -- **this is simpler than the
Hauser fetch**, which needed `openpyxl` for a GitHub-hosted XLSX. It is also
the only one of the three sources that is cleanly re-fetchable.

One parsing hazard, found by hitting it: numeric columns use the literal
strings `NULL`, `NR`, and censored values like `>50000`. `affin.delta_k` in
particular contains `NULL`. Coerce defensively; do not call `float()` bare.

Filter to a target with:

```python
rows = [r for r in csv.DictReader(open(path))
        if r["mut.is_single_point"] == "YES"
        and r["mut.uniprot"] == uniprot
        and r["affin.lig_id"] == ccd]
```

The per-target keys, all verified present in the file:

| target | `mut.uniprot` | `affin.lig_id` | `mut.wt_pdb` | rows |
|---|---|---|---|---|
| CaNMT | `P30418` | `MIM` | `1IYK` | 26 |
| PimA (GDD) | `Q8NNK8` | `GDD` | `3OKP` | 16 |
| PimA (GDP) | `Q8NNK8` | `GDP` | `3OKC` | 16 |
| Glucoamylase SBD | `B7XC04` | `GLC` | `2V8L` | 25 |
| DHODH A | `A2RJT9` | `ORO` | `2DOR` | 10 |
| PoFUT2 | `Q18014` | `GDP` | `3ZY2` | 9 |
| Aldose reductase | `P15121` | (10 ligands) | 1MAR etc. | 13 |

### Step 2 -- the structures

Reuse `igv.data.download_rcsb` unchanged. Fetch `1M17` (EGFR/erlotinib),
`8DZ2` (Mpro/nirmatrelvir, dimeric -- *not* 7VH8), `1IYK`, `3OKP`, `3OKC`, and optionally `4WKQ`,
`4ZAU`, `2V8L`, `2DOR`, `3ZY2`, `1MAR`.

Platinum also ships pre-processed biological assemblies at
`https://biosig.lab.uq.edu.au/platinum/static/platinum_processed_pdb_files.tar.gz`.
**Prefer the plain RCSB files.** The processed archive's provenance is not
documented on the site, and section 17's whole crop-invariance guarantee rests
on knowing exactly which coordinates went in.

### Step 3 -- build the construct (this is where ABL1 drew blood)

For each target emit a **continuous** single-chain construct and assert
`index + offset == residue number`:

| target | chain | span | length | offset | gaps |
|---|---|---|---|---|---|
| **Mpro / 8DZ2** | A **and B** | **1-302** | **302** per chain | **1** | **none** |
| CaNMT / 1IYK | A | 60-451 | 392 | 60 | **none** |
| PimA / 3OKP | A | 4-381 | 378 | 4 | **none** |
| **EGFR / 1M17** | A | 672-995 | 312 | 672 | **12** (965-976, one block) |
| EGFR / 4WKQ | A | 694-1020 | 297 | 694 | 30 |
| EGFR / 4ZAU | A | 698-1017 | 283 | 698 | 37 |

**Mpro, CaNMT and PimA are gap-free**, so unlike 2HYY no UniProt gap-filling
step is needed. Assert it at build time anyway rather than trusting this note
-- that assertion is
cheap and the failure it catches is silent.

Chain selection: 1IYK contains **two copies** (A and B) of what the paper
states is a **monomeric** enzyme, so chain A alone is correct and the second
copy should be dropped. 3OKP is already a single chain.

### Step 4 -- the ligand, and a complication on 1IYK

Pass the ligand to Boltz-2 by CCD code. Heavy-atom counts, from the CCD
formulae (method cross-checked: it returns 37 for `STI`, matching the value
section 17 recorded for imatinib):

| CCD | formula | heavy atoms | vs Boltz-2 limits |
|---|---|---|---|
| **`4WI`** (nirmatrelvir) | C23 H34 F3 N5 O4 | **35** | under 56, no warning |
| **`AQ4`** (erlotinib) | -- (counted from 1M17) | **29** | fine |
| **`IRE`** (gefitinib) | -- (counted from 4WKQ) | **31** | fine |
| **`YY3`** (osimertinib) | -- (counted from 4ZAU) | **37** | fine |
| `MIM` | C33 H52 N6 O4 | **43** | under 56, no warning |
| `GDD` | C16 H25 N5 O16 P2 | **39** | under 56, no warning |
| `GDP` | C10 H15 N5 O11 P2 | 28 | fine |
| `GLC` | C6 H12 O6 | 12 | fine |
| `ORO` | C5 H4 N2 O4 | 11 | fine |
| `ZST` | C19 H12 F3 N3 O3 S | 29 | fine |
| `STI` (existing) | C29 H31 N7 O | 37 | fine |

All candidates sit under the 56-atom warning threshold and far under the 128
hard limit.

**The complication, and it needs a decision before the run.** 1IYK is a
**ternary** complex: chain A contains MIM (43 heavy atoms, the inhibitor) *and*
**MYA**, myristoyl-CoA, the acyl donor, at **63 heavy atoms**. The inhibitor is
competitive for the *peptide* substrate site, not the CoA site, so MYA is a
genuine part of the pocket rather than a crystallisation additive. Three
options, none free:

1. Score MIM alone and delete MYA. Simplest; changes the pocket the model
   sees, and the measured Ki was determined with myristoyl-CoA present.
2. Keep MYA as a second ligand. 43 + 63 = 106 heavy atoms -- under the 128
   hard limit but well past the 56 warning, and it is not what the affinity
   head was trained to score.
3. Keep MYA as a HETATM context without naming it the affinity ligand, if the
   Boltz-2 input schema permits. **Whether it does is [UNVERIFIED]** -- check
   against the pinned `boltz-2.2.1` source before planning around it.

Option 1 is the recommended default on the grounds that it is the only one
that keeps the affinity-head input in distribution, but **record it as a
deliberate deviation**, exactly as the 2HYY-instead-of-1OPJ substitution was
recorded. 3OKP has no equivalent problem: GDD is the only non-water heteroatom
in chain A.

One further note on GDD: in 3OKP it is modelled in **two alternate
conformations** (altloc A and B, 28 atoms each, 30 unique atom names), i.e.
part of the mannose is unresolved. Building from the CCD gives the full 39
atoms, so the model will be scoring atoms the crystal does not resolve. Pick
one altloc explicitly rather than letting a parser take both.

### Step 5 -- the response variable

Platinum gives `affin.k_wt` and `affin.k_mt` in `affin.unit` (Ki or KD, nM),
not ddG. Follow section 17's own recommendation and **work in log units with no
RT at all**:

```
response = log10(k_mt) - log10(k_wt)      # positive = affinity loss = resistance
```

This matches the sign convention already in use (positive = resistance, the
SKEMPI convention) and sidesteps the 300 K vs 298 K `RT` ambiguity that
section 17 found in Hauser. Do **not** use `affin.delta_k` or
`affin.fold_change`: `delta_k` is a raw difference in nM, so it is dominated by
whichever mutant has the largest absolute Ki, and `fold_change` is signed by a
convention the file does not document.

Censoring must be handled explicitly: for CaNMT, 5 of 26 `affin.k_mt` values
are the string `>50000`. Parse them, do not silently drop them, and record
them as right-censored so the choice of how to treat them is visible.

**The other two sources use different response conventions and must not be
pooled with Platinum or with each other without conversion:**

* **EGFR / Robichaux** is already a log ratio of mutant to wild-type IC50 per
  replicate. Average the 3 replicates; keep their spread. No WT row, no unit
  conversion, no RT, and **no sign flip** -- positive is already resistance,
  confirmed against T790M, L858R, L718Q and S811F (table above). The
  logarithm base is very likely log10 but is [UNVERIFIED] in the methods;
  check it, because an ln/log10 mix-up rescales every number by 2.3 without
  changing any rank, so it would leave Spearman untouched and silently corrupt
  any calibration slope.
* **Mpro / Hu** reports Ki for wild-type and each mutant, so convert the same
  way as Platinum: `log10(Ki_mut) - log10(Ki_wt)`. Whether Table S1 carries
  uncertainties is [UNVERIFIED].

All three then live in log10 units with positive = resistance, matching the
SKEMPI convention already in use. **Report per-target correlations separately
and never pool them into one number** -- the assay classes differ by an order
of magnitude in noise, so a pooled correlation would be dominated by whichever
target has the widest response range rather than by model skill.

## Open questions

1. Whether Boltz-2's input schema admits a non-scored cofactor alongside the
   affinity ligand (the 1IYK/MYA question above). Answerable for free by
   reading the pinned 2.2.1 source, the same way `AffinityCropper` was settled.
2. Whether the 0.43-0.49 external reference band transfers to Platinum-derived
   targets at all, given mCSM-lig trained on them. It probably should be
   widened or simply not used as a per-target bar.
3. Whether 2 ligands on PimA is enough for a within-mutation across-ligand
   comparison, or whether 2 points per mutation is too few to rank. Hauser had
   6. This is a design question, not a data question, and it should be settled
   before the PimA run rather than after.
4. **The contents of Mpro Table S1** -- the 12 positions, the wild-type Ki,
   and how many of the 102 have usable values. The case for running Mpro
   third rests on substitution depth, and only 22 mutations at 5 positions are
   confirmed from the abstract. This is the single highest-value open item and
   it needs a human with a browser, not a script.
5. ~~Whether Mpro's dimerisation matters.~~ **Resolved -- it does, and the
   recommended structure changed as a result.** See the dimer note under
   candidate #2; use **8DZ2**, not 7VH8.
6. Whether the **EGFR log base** is 10 or e (above), and whether the 1M17
   -24 offset assertion is wired into the construct builder rather than left
   in this note.

### What would change the recommendation

Two findings would reorder everything above, so they are worth naming:

* If **Mpro Table S1 turns out to cover fewer than ~8 substitutions per
  position**, its only distinctive advantage evaporates and it drops below
  CaNMT, which has twice the positions and an equally clean assay.
* If the **MET DMS loss-of-function confound** is solved (rejected list), MET
  supersedes every target here on the mutation axis by two orders of
  magnitude, and the right move becomes one deep target rather than four
  shallow ones.
