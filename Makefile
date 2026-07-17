.PHONY: help install build train merge export benchmark report run test clean lint

# Default config - override with: make build CONFIG=configs/xxx.yml
CONFIG ?= configs/athletes-qa.yml
PYTHON ?= python3
FLATTUNE = $(PYTHON) -m flattune

help:
	@echo "Flattune Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make CONFIG=configs/xxx.yml build     - Extract data + generate dataset"
	@echo "  make CONFIG=configs/xxx.yml train     - Train model"
	@echo "  make CONFIG=configs/xxx.yml merge     - Merge LoRA weights"
	@echo "  make CONFIG=configs/xxx.yml export    - Export to GGUF/MLX"
	@echo "  make CONFIG=configs/xxx.yml benchmark - Run benchmarks"
	@echo "  make CONFIG=configs/xxx.yml run       - Full pipeline"
	@echo ""
	@echo "Example configs:"
	@echo "  configs/earthquake-qa.yml  - Flatseek index → dataset"
	@echo ""
	@echo "  make install    - Install dependencies (pip install -e .)"
	@echo "  make test      - Run pytest"
	@echo "  make lint      - Run ruff linter"
	@echo "  make clean     - Clean output directories"

install:
	$(PYTHON) -m pip install -e ".[dev]"

build:
	$(FLATTUNE) build $(CONFIG)

train:
	$(FLATTUNE) train $(CONFIG)

merge:
	$(FLATTUNE) merge $(CONFIG)

export:
	$(FLATTUNE) export $(CONFIG)

benchmark:
	$(FLATTUNE) benchmark $(CONFIG)

report:
	$(FLATTUNE) report $(CONFIG)

run:
	$(FLATTUNE) run $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m pip install ruff -q
	ruff check src/
	ruff format --check src/

clean:
	rm -rf outputs/
	rm -rf runs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
