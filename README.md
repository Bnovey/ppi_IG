# igv -- Integrated-gradient validation for co-folding models

## Research question

Can a single backward pass (integrated gradients on a co-folding affinity head)
replace a brute-force alanine scan for predicting binding-site residue importance?

## Ground truth

AbBiBench combinatorial antibody--antigen mutant libraries with measured binding
scores (flow-cytometry-derived KD proxies). 17 affinity datasets, 13 PDB structures.

## Setup

```bash
# CPU (local development)
pip install -e ".[dev]"

# GPU container
docker build -t igv docker/
docker run --gpus all --shm-size=32g --ipc=host igv
```

## Tests

```bash
# Unit tests (no network, no GPU)
python -m pytest tests/ -v -m "not network"

# Integration tests (downloads ~16 MB from HuggingFace)
python -m pytest tests/ -v -m network
```

## Data loading

```python
from igv.data import build_library

lib = build_library("4fqi_h1", cache_dir="data/")
print(lib.variable_positions)   # 16 positions
print(lib.frame["n_mut"].describe())
```
