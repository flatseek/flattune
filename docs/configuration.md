# Configuration Reference

## FlattuneConfig

Top-level configuration for a Flattune run.

```yaml
name: my_project
description: My fine-tuning project
output_dir: outputs  # Output directory (default: outputs)
```

## FlatseekConfig

Configuration for Flatseek data source.

```yaml
flatseek:
  mode: auto        # auto | library | cli
  path: ./data.fsk  # Path to .fsk or directory
  query: "*"        # Optional query filter
  encryption_key: null  # Optional encryption key
```

### Modes

| Mode | Description |
|------|-------------|
| `auto` | Try library first, fall back to CLI |
| `library` | Use Flatseek Python API |
| `cli` | Use flatseek CLI executable |

## ModelConfig

Configuration for the base model.

```yaml
model:
  source: local     # local | huggingface | gguf | mlx
  path: ~/.lmstudio/models/prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf
  repo: Qwen/Qwen3-4B  # For huggingface source
  adapter_path: null    # Optional LoRA adapter
```

### Sources

| Source | Description |
|--------|-------------|
| `local` | Local GGUF/MLX file |
| `huggingface` | HuggingFace repo |
| `gguf` | Local GGUF file |
| `mlx` | Local MLX model |

## DatasetConfig

Configuration for dataset generation.

```yaml
dataset:
  type: qa          # instruction | chat | qa | classification | summary | extraction | rag
  query: "tag:python"  # Flatseek query to filter records (optional)
  tags:              # Alternative: filter by tags/categories
    - python
    - rust
  field_mapping:     # Map index fields to generator input
    title: instruction
    body: input
    answer: output
  generators:       # List of generator plugins (or dataset types)
    - qa
    - summary
  train_split: 0.8  # Training split (0.0-1.0)
  val_split: 0.1    # Validation split (0.0-1.0)
  max_samples: null # Max samples (null = unlimited)
  min_length: 10    # Min record length
  max_length: 2048  # Max record length
```

### Query Filtering Examples

Filter StackOverflow data for specific tags:

```yaml
# Python only
dataset:
  query: "tags:python"
  type: qa

# Python OR Rust
dataset:
  tags: ["python", "rust"]
  type: qa

# Complex query
dataset:
  query: "score:>=10 AND (tags:python OR tags:rust)"
  type: qa
```

## TrainConfig

Configuration for training.

```yaml
train:
  backend: unsloth   # unsloth | transformers | axolotl | mlx-lm
  epochs: 3          # Number of epochs
  lr: 2e-4          # Learning rate
  lora_rank: 16     # LoRA rank
  lora_alpha: 16    # LoRA alpha
  lora_dropout: 0.05
  batch_size: 4     # Batch size
  max_seq_length: 2048
  gradient_accumulation: 4
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42
  bf16: true        # Use BF16
  fp16: false       # Use FP16
  gradient_checkpointing: true
```

## ExportConfig

Configuration for model export.

```yaml
export:
  format: gguf      # gguf | mlx | hf-merged | hf-adapter | safetensors
  quantization: Q4_K_M  # Quantization level (for GGUF)
  output_dir: null  # Output directory
```

### Quantization Levels

| Level | Description | Size |
|-------|-------------|------|
| `Q2_K` | 2-bit quantization | Smallest |
| `Q4_K_M` | 4-bit medium | Balanced |
| `Q5_K_M` | 5-bit medium | Better quality |
| `Q8_0` | 8-bit | Largest, best quality |

## BenchmarkConfig

Configuration for benchmarking.

```yaml
benchmark:
  backend: lmstudio  # lmstudio | ollama | transformers
  prompt_file: null # Optional prompt file
  num_runs: 10     # Number of benchmark runs
  max_tokens: 512  # Max tokens to generate
  temperature: 0.7
  top_p: 0.9
```

## Build Pipeline Options (CLI-only)

The build pipeline supports additional CLI options that don't require config changes:

```bash
# Non-interactive mode (auto-accept suggested types)
flattune build config.yml --yes

# Explicit dataset types
flattune build config.yml --types facts --types context_qa

# Override source
flattune build config.yml --source ./docs/*.md
```

### Available Dataset Types

| Type | Category | Generator |
|------|----------|-----------|
| `facts` | knowledge | facts |
| `glossary` | knowledge | glossary |
| `concept` | knowledge | concept |
| `definition` | knowledge | definition |
| `context_qa` | qa | qa |
| `direct_qa` | qa | qa |
| `yes_no` | qa | qa |
| `chat` | conversation | conversation |
| `dialogue` | conversation | conversation |
| `procedure` | instruction | procedure |
| `howto` | instruction | procedure |
| `topic_classify` | classification | classification |
| `sentiment` | classification | classification |
| `entity_classify` | classification | classification |
| `executive_summary` | summary | summary |
| `concise_summary` | summary | summary |
| `key_value_extract` | extraction | extraction |
| `citation_qa` | rag | qa |

## Full Example

```yaml
name: earthquake
description: Earthquake knowledge fine-tuning

flatseek:
  mode: auto
  path: ./earthquakes.fsk
  query: "magnitude:>=5"

dataset:
  type: qa
  generators:
    - qa
    - summary
  train_split: 0.8
  val_split: 0.1
  max_samples: 10000

model:
  source: local
  path: ~/.lmstudio/models/prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf

train:
  backend: unsloth
  epochs: 3
  lr: 2e-4
  lora_rank: 16
  lora_alpha: 16
  batch_size: 4
  max_seq_length: 2048

export:
  format: gguf
  quantization: Q4_K_M

benchmark:
  backend: lmstudio
  num_runs: 10
```
