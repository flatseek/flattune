# Flattune

#### Compile knowledge into language models.

<small>
Flattune transforms documentation, APIs, databases, and Flatseek indexes into training datasets, fine-tuned models, and benchmark reports.
</small>

---

## The problems

Most frameworks expect you to already have a training dataset.  But that's rarely what you actually have.

- You have documentation.
- You have API specifications.
- You have database schemas.
- You have internal wikis and years of accumulated knowledge.

Turning all of that into high-quality training data is usually the hardest part.

```
Knowledge
    │
    ▼
Export → Clean → Split → Reformat → Generate →
Train → Merge → Export → Benchmark
```

Every stage uses a different tool. 
Every project rebuilds the same pipeline. 
Flattune replaces that entire workflow with one configuration and one reproducible pipeline.

---

## A knowledge compiler

<small>

Flattune doesn't start with datasets. It starts with knowledge.

Think of Flattune as a compiler. Instead of compiling source code into binaries, it compiles knowledge into language models. 

The dataset isn't your starting point—it's an artifact generated from your knowledge.
</small>

---

## Build once, run anywhere

Every stage of the pipeline is independent. Run everything with one command:

```bash
flattune run project.yml
```

Or execute individual stages:

```bash
flattune build project.yml      # Knowledge → Dataset
flattune train project.yml      # Dataset → LoRA
flattune merge project.yml      # LoRA → Base Model
flattune export project.yml     # GGUF / MLX / HuggingFace
flattune benchmark project.yml  # Performance & Quality
```

Each stage produces reusable artifacts that can be resumed independently.

---
## Why Flatseek?

Flattune works with raw files, but Flatseek unlocks a different workflow. Without Flatseek, every model starts by scanning the entire knowledge base.

```
docs/
api/
wiki/
faq/
database/

        │

        ▼

Generate Dataset

        │

        ▼

Train Model
```

Want another model for a different domain?

Start over.

Scan everything again.

Generate another dataset.

Repeat.

---

Flatseek indexes your knowledge once.

After that, every model starts with a search query instead of a full scan.

```
Knowledge
      │
      ▼
Build Flatseek Index (once)
      │
      ▼
knowledge.fsk
      │
      ├── tags:python
      ├── tags:docker
      ├── tags:postgres
      ├── tags:kubernetes
      └── ...
```

Flattune simply asks Flatseek for the knowledge it needs.

The knowledge never moves.

Only the search query changes.

Build one index.

Train as many specialized models as you need.


---
## Teach
Instead of manually writing instruction datasets, Teach analyzes your knowledge and generates dataset types that match the source.

| Source | Generated Datasets |
|---------|--------------------|
| Markdown | Facts, Context QA, Procedures, Conversations |
| OpenAPI | Tool Calling, API QA, Conversations |
| SQL Schema | Text-to-SQL, Schema QA |
| FAQ | QA, Conversations, Summaries |
| Product Catalog | Attribute Extraction, Recommendations |
| Flatseek Index | Coding QA, Bug Fixes, Conversations, Code Completion |


Teach supports two generation modes.


### Distill Mode

No LLM required.

```
Knowledge
    │
    ▼
Rules + Templates
    │
    ▼
Training Dataset
```

Fast, deterministic, reproducible, and works completely offline.

```bash
flattune teach knowledge ./docs --distill
```

Perfect for documentation, APIs, SQL schemas, FAQs, and structured knowledge.


## Teacher Mode

Use an LLM to enrich the generated datasets.

```
Knowledge
      │
      ▼
LLM Teacher
      │
      ▼
Synthetic Examples
Multi-turn Dialogue
Reasoning
Tool Calling
      │
      ▼
Training Dataset
```

```bash
flattune teach knowledge ./docs \
    --teacher openai

flattune teach knowledge ./docs \
    --teacher ollama \
    --model llama3.2
```

Teacher Mode doesn't replace your knowledge.

It expands it—creating additional instructions, conversations, and examples that improve coverage before fine-tuning.

---

## One source, many datasets

**Markdown**

```
docs.md → Facts        {"instruction": "...", "response": "..."}
        → Context QA   {"question": "...", "answer": "..."}
        → Procedure    {"steps": [...]}
```

**OpenAPI**

```
api.json → Tool calling  {"tool": "create_invoice", "args": {...}}
         → API QA        {"question": "...", "answer": "..."}
```

**SQL Schema**

```
schema.sql → Text-to-SQL  {"question": "...", "sql": "SELECT ..."}
           → Schema QA    {"question": "...", "answer": "..."}
```

**FAQ**

```
faq.json → Conversational  {"question": "...", "answer": "..."}
         → Direct QA       {"question": "...", "answer": "..."}
```

**Product Catalog**

```
products.csv → Classification  {"input": "...", "label": "..."}
             → Recommendation  {"query": "...", "products": [...]}
```

**Flatseek Index**

```
knowledge.fsk → Coding QA           {"question": "...", "answer": "..."}
              → Bug Fix             {"error": "...", "fix": "..."}
              → Code Completion     {"prefix": "...", "completion": "..."}
              → Conversation        {"messages": [...]}
              → Instruction Following {"instruction": "...", "response": "..."}
```

---

## Pipeline

```bash
flattune run configs/my_project.yml
```

or stage by stage:

```bash
flattune build configs/my_project.yml     # detect → plan → generate
flattune train configs/my_project.yml     # fine-tune
flattune merge configs/my_project.yml     # merge LoRA
flattune export configs/my_project.yml    # export
flattune benchmark configs/my_project.yml # benchmark
```

---

## Intelligent Planner

Flattune analyzes your sources before generating anything.

```
$ flattune build docs/

Detected
✓ Markdown
✓ OpenAPI
✓ SQL

Recommended
✓ Facts
✓ Context QA
✓ Tool Calling
✓ NL→SQL

Estimated samples: 48,200

Continue? (Y/n)
```

No surprise dataset sizes. No wasted generation runs.

---

## Features

| | |
|---|---|
 Dataset types | 17 built-in (facts, QA, tool calls, conversations, RAG, ...) |
| Source detection | Markdown, JSON, OpenAPI, SQL, CSV, HTML, Flatseek |
| Training backends | Unsloth (GPU), Transformers (CPU/MPS) |
| Export formats | GGUF, MLX, HuggingFace, Safetensors |
| Import targets | LM Studio, Ollama |
| Reproducibility | seeds, hashes, config snapshots, git info |
| Benchmarking | latency, throughput, quality, hallucination, regression |

---

## Installation

```bash
pip install -e .
pip install -e ".[unsloth]"   # GPU support
```

Python 3.10+ · 8GB+ VRAM for 4B models · 20GB+ disk

No GPU? Transformers backend runs on CPU and Apple MPS.

---

## Quick start

### Option 1: Use HuggingFace datasets (recommended)

Flattune can directly use public datasets from HuggingFace as `.fsk` files:

```bash
# Build dataset from HuggingFace .fsk (Olympic athletes example)
flattune build configs/athletes-qa.yml --yes

# Train the model
flattune train configs/athletes-qa.yml

# Merge LoRA with base model
flattune merge configs/athletes-qa.yml

# Export to GGUF
flattune export configs/athletes-qa.yml

# Run full pipeline
flattune run configs/athletes-qa.yml
```

### Option 2: Teach from local documents

```bash
flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl
```

### Example config: HuggingFace .fsk dataset

```yaml
# configs/athletes-qa.yml
name: athletes-qa

flatseek:
  path: https://huggingface.co/datasets/flatseek/public-dataset/resolve/main/271k-athletes.fsk
  query: "*"

dataset:
  type: qa
  query: "country:(China OR Japan OR Korea OR India)"
  max_samples: 500
  generators: [qa, facts]

model:
  source: huggingface
  repo: Qwen/Qwen2.5-0.5B-Instruct

train:
  backend: transformers
  epochs: 1
  lora_rank: 8

benchmark:
  backend: lmstudio
  prompt_file: configs/athletes-prompts.json
```

### Available public datasets on HuggingFace

- `flatseek/public-dataset/271k-athletes.fsk` - 271K Olympic athletes (1800-2000)
- More datasets coming soon

See [Configuration Reference](docs/configuration.md) for all options.

---

## Ecosystem

| | |
|---|---|
| **Flatseek** | Search & index engine — stores knowledge |
| **Flatvec** | Semantic search — Vector Retreival, RAG |
| **Flattune** | Fine-tuning orchestrator — compiles knowledge into models |

Each works standalone. Together, Flatseek feeds Flattune.

---

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration Reference](docs/configuration.md)
- [Dataset Types](docs/dataset-types.md)
- [Teach Framework](docs/teach-framework.md)
- [CLI Reference](docs/cli-reference.md)
- [API Reference](docs/api-reference.md)
- [Benchmarking](docs/benchmarking.md)

---

## Roadmap

| In progress | Coming next |
|---|---|
| Source detectors (PDF, CSV) | Axolotl backend |
| Streaming generation | MLX-LM backend |
| `flattune build --plan` preview | LlamaFactory backend |
| Regression testing | Distributed / multi-GPU training |
| Semantic deduplication | Hyperparameter optimization |

See [TODO.md](TODO.md).

---

## License

Apache License 2.0