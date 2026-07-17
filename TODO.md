# TODO - Flattune

## High Priority

### Build Pipeline Refactor
- [x] Registry architecture (SourceRegistry, CategoryRegistry, DatasetTypeRegistry)
- [x] Source detection (markdown, json, jsonl, flatseek, text)
- [x] Content analyzer with type scoring
- [x] Build planner with plan generation
- [x] Interactive approval flow
- [x] BuildPipeline orchestrator (7 stages)
- [x] CLI --yes, --types flags
- [x] 17 built-in dataset types across 8 categories
- [x] New generators (facts, glossary, concept, procedure)
- [x] Backward compatibility with existing configs
- [ ] `flattune build plan` subcommand for preview

### Teach Framework
- [x] Architecture design (Knowledge Graph, Pipeline, Registries)
- [x] Parser plugin system (Markdown, JSON, CSV, FlatSeek, OpenAPI)
- [x] Teacher plugin system (OpenAI, Anthropic, Ollama)
- [x] Generator plugin system (Q&A, Tool Call, SQL)
- [x] Quality pipeline (Deduplication, Scoring, Hallucination detection)
- [x] CLI commands: teach knowledge, teach software, teach database, teach openapi, teach mcp
- [x] Distill mode (no LLM required) - template-based generation
- [ ] CLI commands: teach workflow (SOP, BPMN)
- [ ] CLI commands: teach python (package introspection)
- [ ] CLI commands: teach cli (--help parsing)

### Training
- [x] TransformersTrainer as fallback (CPU/MPS without CUDA)
- [x] UnslothTrainer signature fix (flatseek_provider support)
- [ ] Implement actual training in UnslothTrainer
- [ ] Checkpoint resume support
- [ ] Eval during training

### Export
- [ ] GGUF export (llama-cpp-python fails to build on this Mac)
- [ ] MLX export properly
- [ ] Additional quantization options

## Medium Priority

### Build Pipeline
- [ ] Additional source detectors (PDF, HTML, CSV)
- [ ] Semantic deduplication in quality pipeline
- [ ] Dataset type documentation (docs/build/types/)

### Benchmark
- [x] Benchmark prompts
- [ ] Regression testing
- [x] Hallucination tests (in quality pipeline)
- [ ] Before/after comparison

### LM Studio / Ollama
- [ ] Test model import
- [x] Benchmark prompts
- [ ] Before/after comparison

## Low Priority

### Training Backends
- [ ] Axolotl backend
- [ ] MLX-LM backend
- [ ] LlamaFactory backend

### Features
- [ ] Distributed training
- [ ] Multi-GPU support
- [ ] Experiment tracking integration
- [ ] Hyperparameter optimization

### Documentation
- [x] Build pipeline documentation
- [ ] Troubleshooting guide
- [ ] Video tutorials

## Known Issues

- [ ] llama-cpp-python fails to build from source on Mac (linker errors)
  - Workaround: Use pre-built wheel or Transformers fallback trainer
- [ ] Unsloth requires GPU for optimal performance
  - Workaround: Use TransformersTrainer on CPU/MPS

## Testing

- [x] Build registry imports
- [x] Source detection
- [x] Content analyzer
- [x] Build planner
- [x] Teach framework with distill mode (no LLM)
- [x] Teach knowledge command with Markdown parser
- [x] Teach openapi command
- [ ] Integration tests with real data
- [ ] Unit tests for parsers
- [ ] Unit tests for generators
- [ ] Unit tests for analyzer
- [ ] End-to-end build pipeline test

## Code Quality

- [x] Type hints throughout
- [x] Docstrings
- [ ] Set up CI/CD
- [ ] Add pre-commit hooks
