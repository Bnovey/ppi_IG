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
