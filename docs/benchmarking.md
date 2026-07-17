# Benchmarking

Flattune provides comprehensive benchmarking for fine-tuned models.

## Benchmark Backends

### LM Studio

Benchmark using LM Studio local server.

```yaml
benchmark:
  backend: lmstudio
```

Requires LM Studio running with API enabled (default: localhost:1234).

### Ollama

Benchmark using Ollama local server.

```yaml
benchmark:
  backend: ollama
```

Requires Ollama running (default: localhost:11434).

### Transformers

Direct benchmarking using HuggingFace Transformers.

```yaml
benchmark:
  backend: transformers
```

## Benchmark Categories

### latency

Measures inference latency.

**Metrics:**
- Time per token (ms)
- Total inference time (ms)
- Time to first token (ms)

### tokens_per_sec

Measures generation throughput.

**Metrics:**
- Tokens per second
- Total tokens generated
- Total time

### prompt_quality

Measures output quality.

**Metrics:**
- Perplexity
- Cross-entropy loss

### instruction_following

Measures instruction adherence.

**Metrics:**
- Instruction adherence score (0-100)
- Task completion rate (%)

### domain_accuracy

Measures domain-specific accuracy.

**Metrics:**
- Accuracy (%)
- F1 score
- Precision/Recall

### hallucination

Measures hallucination levels.

**Metrics:**
- Hallucination rate (%)
- Factual accuracy (%)

### regression

Compares against baseline model.

**Metrics:**
- Improvement over baseline (%)
- Per-category comparison

## Benchmark Configuration

```yaml
benchmark:
  backend: lmstudio
  num_runs: 10      # Number of benchmark runs
  max_tokens: 512   # Max tokens to generate
  temperature: 0.7  # Generation temperature
  top_p: 0.9        # Top-p sampling
```

## Running Benchmarks

### Single Benchmark

```bash
flattune benchmark configs/my_project.yml
```

### With Make

```bash
make benchmark
```

## Benchmark Reports

Results are saved to `outputs/PROJECT_TIMESTAMP/benchmark/`:

```
benchmark/
├── results.json      # Raw benchmark results
├── metrics.json     # Parsed metrics
└── report.md        # Markdown report
```

### Example Report

```markdown
# Benchmark Report: earthquake

## Summary

| Metric | Value |
|--------|-------|
| Tokens/sec | 42.5 |
| Latency | 23.4ms |
| Accuracy | 87.3% |

## Details

### Latency
- Time per token: 23.4ms
- Time to first token: 1.2s

### Quality
- Instruction following: 89%
- Domain accuracy: 87%
```

## Before/After Comparison

Compare fine-tuned vs baseline:

```bash
flattune benchmark --compare configs/my_project.yml
```

Generates `comparison.md` with side-by-side metrics.
