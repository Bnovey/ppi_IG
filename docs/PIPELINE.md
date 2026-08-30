# Pipeline Reference

## Artifact DAG

```
                       00_fetch_data
                            |
                    data/raw/ CSVs + PDBs
                            |
                       01_build_library
                            |
              +-------------+-------------+
              |                           |
  data/processed/<ds>_library.parquet   .json companion
              |
              +--------+--------+
              |        |        |
        02_embed    03_attr   04_scan
         deltas      ibute
              |        |        |
   <ds>_deltas.npz  <ds>_<sc>  <ds>_<sc>
              |    _<meth>     _scan.csv
              |    _grad.npz       |
              +--------+          |
                       |          |
                  05_predict      |
                       |          |
           <ds>_<sc>_<meth>       |
              _pred.csv           |
                       |          |
                       +----+-----+
                            |
                       06_metrics
                            |
                    results/metrics.csv
                            |
                     (notebooks 02-04)


                    07_sanity  (standalone gate)
                         |
              results/sanity_<ds>_<sc>.json
```

Abbreviations: `<ds>` = dataset (e.g. `4fqi_h1`), `<sc>` = score
(e.g. `complex_pde`), `<meth>` = method (e.g. `plain_grad`).

## Artifact table

| Path pattern | Producing stage | Consuming stage(s) | Contents |
|---|---|---|---|
| `data/raw/<name>_benchmarking_data.csv` | 00 | 01 | AbBiBench affinity CSV (binding scores, heavy/light chain seqs) |
| `data/raw/<name>.pdb` | 00 | 01, 02, 03, 04 | PDB structure for the antibody--antigen complex |
| `data/raw/fetch_manifest.json` | 00 | (audit) | List of all fetched files with sizes |
| `data/processed/<ds>_library.parquet` | 01 | 02, 05 | Parsed mutant library: sequences, binding scores, substitutions, n_mut |
| `data/processed/<ds>_library.json` | 01 | (audit) | Companion metadata: reference seq, variable positions, alphabet, alignment |
| `data/processed/<ds>_deltas.npz` | 02 | 05 | Embedding deltas `E_mutant[l] - E_reference[l]` per (position, aa) pair |
| `results/<ds>_<sc>_<meth>_grad.npz` | 03 | 05 | Path-averaged gradient tensor, shape (L, D) |
| `results/<ds>_<sc>_scan.csv` | 04 | 06 | Brute-force mutation scan scores: one row per mutant |
| `results/<ds>_<sc>_<meth>_pred.csv` | 05 | 06 | Predicted binding scores from gradient dot embedding delta |
| `results/metrics.csv` | 06 | notebooks | Tidy long-format table: dataset, method, term (T1/T2/T3), score, value |
| `results/sanity_<ds>_<sc>.json` | 07 | (gate) | Sanity-check results: completeness, step-count stability, random-weights, dead-score |

## Provenance convention

Every artifact has a `<output>.prov.json` sidecar written by
`src/igv/provenance.py`. The sidecar records:

- **inputs**: paths or identifiers of all upstream artifacts consumed
- **params**: the flags/arguments that controlled this stage
- **arm**: the experimental arm this artifact belongs to (e.g.
  `{"score": "complex_pde", "method": "ig", "m_steps": 15}`)
- **env**: Python version, platform, hostname, `torch` version, `boltz`
  version, CUDA availability, GPU name/count/memory
- **git_commit** and **git_dirty**: the repo state when the artifact was
  produced
- **argv**: the exact command line that was run
- **created_utc**: ISO 8601 timestamp

### Why the arm matters

Analysis code calls `provenance.assert_provenance(artifact_path, **expected)`
to verify the *recorded* arm rather than trusting the flags that were passed
to the current invocation. This catches a class of bug that cost the
predecessor project significant time:

1. A default argument silently routed all attribution to one model while
   the experiment was supposed to be comparing three. The 35-hour campaign
   ran to completion with no errors.
2. A misordered threshold made a filter inert. Every variant passed, and
   downstream metrics looked plausible.

Both bugs were found only by auditing provenance sidecars after the fact.
Asserting the recorded arm at analysis time is the cheapest defence: if the
sidecar says the artifact came from arm X but the analysis expects arm Y,
it raises `AssertionError` naming every mismatch.

### Sidecar format

Sidecars are written atomically (temp file + `os.replace`) so that a
truncated write never masquerades as valid provenance. The file is JSON
with sorted keys and 2-space indent.

## How to reproduce a number in the paper

Given a row in `results/metrics.csv`:

```
dataset    method       term  score     value
4fqi_h1    plain_grad   T3    spearman  0.1842
```

1. **Find the prediction file.** The `06_metrics` provenance sidecar
   (`results/metrics.csv.prov.json`) records
   `inputs.pred = "results/4fqi_h1_complex_pde_plain_grad_pred.csv"`.

2. **Find the gradient and deltas.** The `05_predict` sidecar for that
   prediction CSV records:
   - `inputs.grad = "results/4fqi_h1_complex_pde_plain_grad_grad.npz"`
   - `inputs.deltas = "data/processed/4fqi_h1_deltas.npz"`
   - `inputs.library = "data/processed/4fqi_h1_library.parquet"`

3. **Find the raw data.** The `01_build_library` sidecar for the parquet
   records `inputs.dataset = "4fqi_h1"` and `inputs.cache_dir = "data/raw"`.
   The `00_fetch_data` sidecar records the HuggingFace URLs.

4. **Rerun from raw data:**
   ```bash
   python3 scripts/00_fetch_data.py --datasets 4fqi_h1
   python3 scripts/01_build_library.py --dataset 4fqi_h1
   python3 scripts/02_embed_deltas.py --dataset 4fqi_h1          # GPU
   python3 scripts/03_attribute.py --dataset 4fqi_h1 \
       --score complex_pde --method plain_grad                   # GPU
   python3 scripts/05_predict.py \
       --library data/processed/4fqi_h1_library.parquet \
       --grad results/4fqi_h1_complex_pde_plain_grad_grad.npz \
       --deltas data/processed/4fqi_h1_deltas.npz \
       --out results/4fqi_h1_complex_pde_plain_grad_pred.csv
   python3 scripts/06_metrics.py \
       --pred results/4fqi_h1_complex_pde_plain_grad_pred.csv \
       --dataset 4fqi_h1 --method plain_grad
   ```

5. **Verify environment match.** Compare `env.torch`, `env.boltz`, and
   `env.gpu_name` in the original sidecar against your environment. CUDA
   non-determinism means exact floating-point reproducibility requires the
   same GPU architecture; rank-order metrics (Spearman) are stable across
   GPUs.
