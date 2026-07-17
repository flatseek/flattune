# CLI Reference

## flattune

Main CLI entry point.

```bash
flattune [OPTIONS] COMMAND [ARGS]...
```

### Options

| Option | Description |
|--------|-------------|
| `--version` | Show version |
| `--help` | Show help |

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `build` | Extract data and generate dataset with intelligent planning |
| `train` | Train a model using the dataset |
| `merge` | Merge LoRA adapter with base model |
| `export` | Export model to target format (GGUF/MLX/HF) |
| `benchmark` | Benchmark the model |
| `report` | Generate training and benchmark report |
| `run` | Run the full pipeline |
| `teach` | Knowledge & Skill Compiler (teach framework) |

---

## build

Extract data and generate dataset with intelligent planning.

```bash
flattune build CONFIG_FILE [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-y`, `--yes` | Skip interactive approval, use suggested types |
| `--source TEXT` | Source file/directory (overrides config) |
| `-t`, `--types TEXT` | Dataset types to generate (can specify multiple) |

**Examples:**

```bash
# Non-interactive (auto-accept suggested types)
flattune build configs/earthquake-qa.yml --yes

# Explicit dataset types
flattune build configs/earthquake-qa.yml --types facts --types context_qa

# Interactive mode
flattune build configs/earthquake-qa.yml
```

---

## train

Train a model using the dataset.

```bash
flattune train CONFIG_FILE
```

**Example:**

```bash
flattune train configs/my_project.yml
```

---

## merge

Merge LoRA adapter with base model.

```bash
flattune merge CONFIG_FILE
```

**Example:**

```bash
flattune merge configs/my_project.yml
```

---

## export

Export model to target format (GGUF/MLX/HF).

```bash
flattune export CONFIG_FILE
```

**Example:**

```bash
flattune export configs/my_project.yml
```

---

## benchmark

Benchmark the model using LM Studio, Ollama, or Transformers.

```bash
flattune benchmark CONFIG_FILE
```

**Example:**

```bash
flattune benchmark configs/my_project.yml
```

---

## report

Generate training and benchmark report.

```bash
flattune report CONFIG_FILE
```

**Example:**

```bash
flattune report configs/my_project.yml
```

---

## run

Run the full pipeline: extract → generate → train → merge → export → benchmark → report.

```bash
flattune run CONFIG_FILE
```

**Example:**

```bash
flattune run configs/my_project.yml
```

---

## teach

The `teach` command group provides the **Knowledge & Skill Compiler** functionality.

```bash
flattune teach [OPTIONS] COMMAND [ARGS]...
```

### teach knowledge

Teach from documents (Markdown, TXT, PDF, HTML, CSV, JSON, FlatSeek).

```bash
flattune teach knowledge SOURCES... [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--distill` | Use template-based generation (no LLM) |
| `--teacher {openai,anthropic,ollama}` | LLM teacher for generation |
| `--model TEXT` | Teacher model name (default: gpt-4o-mini) |
| `--output, -o FILE` | Output file |
| `--sample-types, -s TEXT` | Sample types to generate |
| `--max-samples N` | Maximum samples to generate |

**Examples:**

```bash
# Distill mode (no LLM required)
flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl

# With OpenAI teacher
flattune teach knowledge ./docs/*.md --teacher openai -o dataset.jsonl

# With Ollama (local)
flattune teach knowledge ./docs/*.md --teacher ollama --model llama3.2 -o dataset.jsonl
```

---

### teach openapi

Teach from OpenAPI/Swagger specifications.

```bash
flattune teach openapi SPEC [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--distill` | Use template-based generation |
| `--teacher TEXT` | LLM teacher for generation |
| `--output, -o FILE` | Output file |

**Example:**

```bash
flattune teach openapi api.json --distill -o api_dataset.jsonl
```

---

### teach database

Teach from database schemas.

```bash
flattune teach database SCHEMA [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--distill` | Use template-based generation |
| `--teacher TEXT` | LLM teacher for generation |
| `--output, -o FILE` | Output file |

**Example:**

```bash
flattune teach database schema.sql --distill -o sql_dataset.jsonl
```

---

### teach software

Teach from software APIs and tools.

```bash
flattune teach software SOURCES... [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--distill` | Use template-based generation |
| `--teacher TEXT` | LLM teacher for generation |
| `--sample-types, -s TEXT` | Sample types to generate |

---

### teach mcp

Teach from MCP server definitions.

```bash
flattune teach mcp SERVER [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--distill` | Use template-based generation |
| `--teacher TEXT` | LLM teacher for generation |
| `--output, -o FILE` | Output file |

---

### teach list-parsers

List available parsers.

```bash
flattune teach list-parsers
```

---

### teach list-teachers

List available teacher models.

```bash
flattune teach list-teachers
```

---

### teach list-generators

List available generators.

```bash
flattune teach list-generators
```

---

## Makefile Commands

```bash
make help      # Show help
make install   # Install dependencies (pip install -e .)
make build     # Extract + generate dataset
make train     # Train model
make merge     # Merge LoRA
make export    # Export model
make benchmark # Run benchmarks
make report    # Generate report
make run       # Full pipeline
make test      # Run pytest
make lint     # Run ruff linter
make clean     # Clean outputs
```

---

## Examples

### Teach Framework (No Flatseek Required)

```bash
# Documents → dataset (no LLM)
flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl

# OpenAPI → tool-calling dataset
flattune teach openapi api.json --distill -o api_dataset.jsonl

# SQL schema → NL-to-SQL dataset
flattune teach database schema.sql --distill -o sql_dataset.jsonl

# With LLM teacher
flattune teach knowledge ./docs/*.md --teacher openai -o dataset.jsonl
```

### Build Pipeline (Intelligent Planning)

```bash
# Non-interactive with auto-detection
flattune build configs/earthquake-qa.yml --yes

# Explicit types
flattune build configs/earthquake-qa.yml --types facts --types context_qa

# Interactive planning
flattune build configs/earthquake-qa.yml
```

### Full Traditional Pipeline

```bash
# Run full pipeline
flattune run configs/my_project.yml

# Or step by step
flattune build configs/my_project.yml
flattune train configs/my_project.yml
flattune merge configs/my_project.yml
flattune export configs/my_project.yml
flattune benchmark configs/my_project.yml
flattune report configs/my_project.yml
```
