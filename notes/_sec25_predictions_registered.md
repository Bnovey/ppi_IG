# Predictions registered BEFORE any section-25 run
Written 2026-10-14 ~04:05 UTC, before the first GPU job of the replication.

Basis: section 14 measured m=32 completeness with real geometry as
1JTG 5.99%, 2JEL 2.37%, 3HFM 11.14% (3S9D 8.51%). Section 24 measured
3S9D m=32 8.51% -> m=64 0.24%, a 35x reduction, monotone, no sign flip.

## Completeness (Sundararajan 5% relative gate)
- 1JTG B: m=32 ~6% FAIL (marginal). m=64 PASS, expect <1%.
- 2JEL P: m=32 ~2.4% PASS (the only complex expected to pass at m=32).
  m=64 PASS, expect <0.5%.
- 3HFM Y: m=32 ~11% FAIL (worst of the four). m=64 PASS, expect <1%.
- General: span |f(x)-f(baseline)| will be O(1) on all three, as on 3S9D
  (-1.64), not degenerate like the affinity head (0.085).

## Faithfulness (resswap, the headline)
- Registered prediction: the 3S9D 0.5937 will NOT cleanly replicate at
  that magnitude on all three. Section 14's between-complex sd ~0.33 is
  the prior. Expect mean ~0.4-0.5 with sd >0.15, and expect at least one
  complex to land materially lower.
- All four should nonetheless clear their own permutation null, because
  n=285..950 at the mutation level is far better powered than the n=15..50
  position-level test that failed in section 14.
- The partial controlling distance-to-partner will drop each raw Spearman
  by <0.05, as on 3S9D (0.5937 -> 0.5722).

## Refeaturise
- Skip rate 85-92% on all three, essentially all on
  TOKEN_TO_REP_ATOM CONTENT DIFFERS, ~0 on shape.
- refeaturise faithfulness will be null on all three.
- resswap/refeaturise decorrelation WILL reproduce (|rho| < 0.15), and
  median|delta| refeaturise will be >10x median|delta| resswap.

## VRAM
- Peak <25 GiB on all three; 3HFM (shortest) lowest, 1JTG (L=427) highest.

---
## Measured cost model (recorded 04:05 UTC, after the first timing reading)
NOT a prediction -- an operational measurement that changes what is achievable.

Token counts from the predicted-coords cache (these drive everything):

| complex | tokens | vs 3S9D |
|---|---|---|
| 3S9D A127_B180 | 307 | 1.00x |
| 1JTG A262_B165 | 427 | 1.39x |
| 2JEL H218_L217_P85 | **520** | 1.69x |
| 3HFM H215_L214_Y129 | **558** | 1.82x |

**2JEL and 3HFM are the two LARGEST complexes, not the smallest.** Section 14
chose them for position count (32, 15), which is uncorrelated with cost.

Measured IG step time: 3S9D ~20 s/step (307 tok), 1JTG **50.75 s/step** (427
tok, measured over steps 1-5). That is 2.54x the time for 1.39x the tokens, an
exponent of ~2.8 -- quadratic attention plus the `pf_chunk` "large" profile that
switches on above 384 tokens. Extrapolated: 2JEL ~89 s/step, 3HFM ~108 s/step.

An IG run at m does 2m steps (embedding integral + res_type integral).

| job | 1JTG | 2JEL | 3HFM |
|---|---|---|---|
| IG m=32 (64 steps) | **54 min** | **95 min** | **115 min** |
| IG m=64 (128 steps) | **108 min** | **189 min** | **230 min** |
| resswap scan | ~81 min (532 muts) | ~137 min (608) | ~74 min (285) |
| refeaturise | ~20 min | ~30 min | ~15 min |

**The usable window is ~113 min** (120 min cap minus boot and safety margin).
Stage 03 is NOT resumable. Therefore, under the 2 h cap:
- IG m=64 is **impossible** for 2JEL (189 min) and 3HFM (230 min).
- IG m=32 is **impossible** for 3HFM (115 min) and marginal for 2JEL (95 min).
- Scans are resumable and can span windows, so they are always achievable.

---
## Reuse decision (04:10 UTC): the m=32 rung for 2JEL and 3HFM already exists

`results/{2JEL,3HFM}_complex_pde_ig_grad.npz` from section 14 (2026-09-30,
commit 0bf19ef, **clean tree**) were produced by exactly the command this
replication specifies:

    03_attribute.py --dataset {2JEL,3HFM} --chain {P,Y} --score complex_pde \
        --method ig --m-steps 32 --baseline mean_aa --x-pred predicted

They carry `grad_res_type` -- the object `24_faithfulness.py` consumes -- plus
`f_x`, `f_baseline` and `completeness_error`:

| complex | f(x) | f(baseline) | span | completeness | sec-14 value |
|---|---|---|---|---|---|
| 2JEL P | 1.4196462631 | 2.0330910683 | -0.6134 | 0.023726 (2.37%) | 2.37% OK |
| 3HFM Y | 0.9989417791 | 2.3140511513 | -1.3151 | 0.111378 (11.14%) | 11.14% OK |

Reusing them saves 95 + 115 = **210 min of A100** that the 2 h cap cannot
supply in one piece anyway.

**The reuse is conditional, and the condition is testable for free.**
`boltz_score.py` changed by ~3,500 lines between 0bf19ef and 420a456 (the
affinity seam). If `complex_pde` itself moved, a 0bf19ef attribution against a
420a456 scan would compare two different functions -- defect 1 of section 23,
the exact error this project already paid for once.

**Test:** the 420a456 scan writes a wild-type row first. Its `model_score` must
equal the stored `f_x` above to printed precision (this is the same check
section 24 made on 3S9D, where scan WT and stage-03 f(x) agreed to all digits).
The fresh `plain_grad` run at 420a456 prints `f(x)` and gives the same check
earlier and cheaper. **If either disagrees, the reused m=32 artifacts are
discarded and the m=32 rung is reported as not-run for that complex.**

1JTG's section-14 IG was run WITHOUT `--x-pred predicted` (zeroed geometry), so
it is not reusable and is being re-run at m=32 now.

## VRAM prediction FALSIFIED early (04:09 UTC)
Predicted "<25 GiB on all three". Measured on 1JTG IG m=32 at step 12:
**43608 MiB = 42.6 GiB**, already 2.1x 3S9D's 20.2 GiB peak.
A 24 GB card does NOT fit 1JTG, and 2JEL (520 tok) / 3HFM (558 tok) will be
higher still. The 80 GB card is NOT oversized for this arm, contrary to
section 24's conclusion drawn from 3S9D alone.
