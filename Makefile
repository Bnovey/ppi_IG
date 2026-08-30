# IG validation pipeline
# Run `make help` (or just `make`) to see all targets.

DATASET  ?= 4fqi_h1
SCORE    ?= complex_pde
METHOD   ?= plain_grad
PY       ?= python3
SCRIPTS  := scripts

LIBRARY  := data/processed/$(DATASET)_library.parquet
DELTAS   := data/processed/$(DATASET)_deltas.npz
GRAD     := results/$(DATASET)_$(SCORE)_$(METHOD)_grad.npz
SCAN     := results/$(DATASET)_$(SCORE)_scan.csv
PRED     := results/$(DATASET)_$(SCORE)_$(METHOD)_pred.csv
METRICS  := results/metrics.csv
SANITY   := results/sanity_$(DATASET)_$(SCORE).json

.DEFAULT_GOAL := help

.PHONY: help setup test lint fetch library deltas attribute scan predict metrics sanity stage0 all clean

help: ## Print this help
	@echo "IG validation pipeline"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  DATASET  = $(DATASET)"
	@echo "  SCORE    = $(SCORE)"
	@echo "  METHOD   = $(METHOD)"
	@echo "  PY       = $(PY)"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "GPU-requiring targets: deltas, attribute, scan, sanity"

setup: ## Install the package in dev mode (CPU only)
	$(PY) -m pip install -e ".[dev]"

test: ## Run the pytest suite (no GPU, no network)
	$(PY) -m pytest tests/ -v -m "not network"

lint: ## Run ruff and black checks
	$(PY) -m ruff check src/ scripts/ tests/
	$(PY) -m black --check src/ scripts/ tests/

fetch: ## Download raw data (CPU)
	$(PY) $(SCRIPTS)/00_fetch_data.py --datasets $(DATASET)

library: ## Build mutant library from raw data (CPU)
	$(PY) $(SCRIPTS)/01_build_library.py --dataset $(DATASET)

deltas: ## Compute embedding deltas [GPU]
	$(PY) $(SCRIPTS)/02_embed_deltas.py --dataset $(DATASET)

attribute: ## Run gradient attribution [GPU]
	$(PY) $(SCRIPTS)/03_attribute.py --dataset $(DATASET) --score $(SCORE) --method $(METHOD)

scan: ## Run brute-force mutation scan [GPU]
	$(PY) $(SCRIPTS)/04_scan.py --dataset $(DATASET) --score $(SCORE)

predict: ## Predict mutant scores from gradients (CPU)
	$(PY) $(SCRIPTS)/05_predict.py \
		--library $(LIBRARY) \
		--grad $(GRAD) \
		--deltas $(DELTAS) \
		--out $(PRED)

metrics: ## Compute T1/T2/T3 evaluation metrics (CPU)
	$(PY) $(SCRIPTS)/06_metrics.py \
		--pred $(PRED) \
		--scan $(SCAN) \
		--dataset $(DATASET) \
		--method $(METHOD) \
		--out $(METRICS)

sanity: ## Run sanity checks before expensive stages [GPU]
	$(PY) $(SCRIPTS)/07_sanity.py --dataset $(DATASET) --score $(SCORE)

stage0: ## Cheapest end-to-end go/no-go on 4fqi_h1 (CPU stages only)
	$(MAKE) fetch library DATASET=4fqi_h1
	@echo "stage0 passed -- CPU pipeline works for 4fqi_h1"

all: fetch library deltas sanity attribute scan predict metrics ## Run the full pipeline

clean: ## Remove all generated data and results
	rm -rf data/raw data/processed results/*.csv results/*.npz results/*.json
