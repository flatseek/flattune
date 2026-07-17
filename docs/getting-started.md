# Getting Started with Flattune

## What is Flattune?

Flattune is a **Knowledge Compiler** — it transforms documentation, APIs, databases, and Flatseek indexes into training datasets, fine-tuned models, and benchmark reports.

Unlike traditional fine-tuning frameworks that expect an existing dataset, Flattune starts with your knowledge and compiles it into a trainable format.

```
Knowledge → Flattune → Fine-tuned Model
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for Unsloth training)
- 8GB+ VRAM (for 4B models)
- 20GB+ disk space
- LM Studio or Ollama (for benchmarking)

## Installation

```bash
# Base installation
pip install -e .

# With GPU support (Unsloth)
pip install -e ".[unsloth]"
```

## Two Ways to Use Flattune

### 1. HuggingFace Datasets (Easiest)

Use public datasets directly from HuggingFace:

```bash
# Build from HuggingFace .fsk (Olympic athletes example)
flattune build configs/athletes-qa.yml --yes

# Train
flattune train configs/athletes-qa.yml

# Merge + Export
flattune merge configs/athletes-qa.yml
flattune export configs/athletes-qa.yml

# Full pipeline
flattune run configs/athletes-qa.yml
```

Available datasets:
- `flatseek/public-dataset/271k-athletes.fsk` - 271K Olympic athletes (1800-2000)

### 2. Teach Framework (No Flatseek Required)

Start from raw files — markdown, OpenAPI specs, SQL schemas, FAQs:

```bash
# Documents → dataset (no LLM required)
flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl

# OpenAPI → tool-calling dataset
flattune teach openapi api.json --distill -o api_dataset.jsonl

# SQL schema → NL-to-SQL dataset
flattune teach database schema.sql --distill -o sql_dataset.jsonl

# With LLM teacher for higher quality
flattune teach knowledge ./docs/*.md --teacher openai -o dataset.jsonl
```

### 2. Traditional Pipeline (Flatseek Index Required)

Point Flattune at your Flatseek index:

```bash
# One command for the full pipeline
flattune run configs/earthquake-qa.yml

# Or step by step
flattune build configs/earthquake-qa.yml   # Extract + generate dataset
flattune train configs/earthquake-qa.yml   # Fine-tune
flattune merge configs/earthquake-qa.yml   # Merge LoRA
flattune export configs/earthquake-qa.yml  # Export
flattune benchmark configs/earthquake-qa.yml  # Benchmark
```

## Build Pipeline: Intelligent Dataset Planning

The build pipeline analyzes your sources and recommends dataset types:

```bash
# Interactive mode — review plan before generating
flattune build configs/earthquake-qa.yml

# Non-interactive — auto-accept suggestions
flattune build configs/earthquake-qa.yml --yes

# Explicit types
flattune build configs/earthquake-qa.yml --types facts --types context_qa --types glossary
```

### Built-in Dataset Types (17 types across 8 categories)

| Category | Types |
|----------|-------|
| `knowledge` | facts, glossary, concept, definition |
| `qa` | context_qa, direct_qa, yes_no |
| `conversation` | chat, dialogue |
| `instruction` | procedure, howto |
| `classification` | topic_classify, sentiment, entity_classify |
| `summary` | executive_summary, concise_summary |
| `extraction` | key_value_extract |
| `rag` | citation_qa |

## Configuration

Create `configs/my_project.yml`:

```yaml
name: my_project

flatseek:
  mode: auto
  path: ./data.fsk

dataset:
  type: qa
  generators:
    - qa
    - summary

model:
  source: local
  path: ~/.lmstudio/models/my-model.gguf

train:
  backend: unsloth
  epochs: 3
  lr: 2e-4
  lora_rank: 16

export:
  format: gguf

benchmark:
  backend: lmstudio
```

## Using Make

```bash
make install    # Install dependencies
make build      # Extract + generate dataset
make train      # Train model
make merge      # Merge LoRA
make export     # Export model
make benchmark  # Run benchmarks
make run        # Full pipeline
make test       # Run tests
make clean      # Clean outputs
```

## Training Without GPU

The Transformers backend works on CPU and Apple MPS:

```yaml
train:
  backend: unsloth  # Automatically falls back to transformers if unsloth unavailable
```

## Next Steps

- See [Configuration Reference](configuration.md) for all options
- See [Dataset Types](dataset-types.md) for generator plugins
- See [Benchmarking](benchmarking.md) for benchmark categories
- See [Teach Framework](teach-framework.md) for knowledge compilation
- See [CLI Reference](cli-reference.md) for all commands
