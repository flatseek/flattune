# Changelog

All notable changes to Flattune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Build Pipeline Refactor (Intelligent Dataset Planning)

- **`src/flattune/build/`** - New build pipeline module
  - `SourceRegistry`, `CategoryRegistry`, `DatasetTypeRegistry` - Plugin-based registries
  - `SourceDetector` - Auto-detects source types (markdown, json, jsonl, flatseek, text)
  - `ContentAnalyzer` - Scores content against dataset types
  - `BuildPlanner` - Creates build plans with estimated samples
  - `InteractiveApproval` - User approval flow for type selection
  - `BuildPipeline` - 7-stage orchestrator (Detect → Analyze → Plan → Approve → Generate → Validate → Export)
  - `BuildExporter` - Structured output by category/type

- **17 Built-in Dataset Types** across 8 categories:
  - `knowledge`: facts, glossary, concept, definition
  - `qa`: context_qa, direct_qa, yes_no
  - `conversation`: chat, dialogue
  - `instruction`: procedure, howto
  - `classification`: topic_classify, sentiment, entity_classify
  - `summary`: executive_summary, concise_summary
  - `extraction`: key_value_extract
  - `rag`: citation_qa

- **New Generators** (`src/flattune/dataset/generators/`):
  - `FactsGenerator` - Extracts factual statements
  - `GlossaryGenerator` - Extracts glossary terms
  - `ConceptGenerator` - Generates concept explanations
  - `ProcedureGenerator` - Generates procedural instructions

- **CLI Enhancements**:
  - `flattune build --yes` - Non-interactive mode (auto-accept suggested types)
  - `flattune build --types <types>` - Explicit dataset types
  - `flattune build plan <source>` - Preview build plan without generating

- **Backward Compatibility**:
  - `LEGACY_TYPE_MAP` maps old `DatasetType` enum to new types
  - Existing configs work without modification
  - `get_generator()` supports both old and new registries

### Changed

- **`DatasetBuilder`** - Refactored to support new pipeline while maintaining backward compatibility
- **`BaseGenerator`** - Added `supported_types` field and `instruction` parameter
- **`QAGenerator`** - Added metadata types (context_qa, direct_qa, yes_no)
- **`TeachFramework`** - Unchanged, continues to work as before

### Planned

- `flattune build plan` subcommand for build plan preview
- Additional source detectors (PDF, HTML, CSV)
- Additional dataset types and generators
- Streaming dataset generation

## [0.1.0] - 2024-07-17

### Added

#### Flatseek Integration
- `FlatseekProvider` abstract interface
- `LibraryProvider` - Python API integration
- `CLIProvider` - CLI fallback integration
- `AutoProvider` - automatic backend detection
- Support for directory index, .fsk files, HTTP remote, encrypted .fsk

#### Dataset Generation
- `DatasetBuilder` - dataset generation from Flatseek
- `DatasetSplitter` - train/val/test splitting
- Generator plugins:
  - `qa` - Question-answer generation
  - `summary` - Summarization generation
  - `classification` - Classification generation
  - `comparison` - Comparison generation
  - `conversation` - Multi-turn conversation
  - `extraction` - Entity extraction
  - `safety` - Safety examples
  - `custom` - Custom generator template

#### Training
- `TrainerBase` abstract interface
- `TrainerFactory` - backend factory
- `UnslothTrainer` - Unsloth LoRA fine-tuning
- Checkpoint management
- LoRA merging
- Training reports with loss curves

#### Export
- `ModelExporter` - model export interface
- GGUF export (llama.cpp)
- MLX export (Apple Silicon)
- HuggingFace merged/adapter export
- Safetensors export

#### Inference & Benchmarking
- `LMStudioClient` - LM Studio integration
- `OllamaClient` - Ollama integration
- `BenchmarkRunner` - benchmark orchestration
- `BenchmarkReporter` - report generation
- Benchmark categories: latency, tokens_per_sec, quality, instruction_following, domain_accuracy, hallucination, regression

#### Utilities
- Logging setup
- File system utilities
- Reproducibility (seed, git info, timestamp)
- Hashing utilities

#### CLI
- `flattune build` - Extract + generate dataset
- `flattune train` - Train model
- `flattune merge` - Merge LoRA
- `flattune export` - Export model
- `flattune benchmark` - Run benchmarks
- `flattune report` - Generate report
- `flattune run` - Full pipeline

#### Configuration
- YAML-based configuration
- Example configs: earthquakes, legal, finance, customer_support

#### Documentation
- README.md
- docs/getting-started.md
- docs/configuration.md
- docs/dataset-types.md
- docs/benchmarking.md
- docs/api-reference.md
- docs/cli-reference.md
- docs/teach-framework.md

### Changed

- Default model: Bonsai-1.7B-Q1_0.gguf (local, ~248MB)

### Fixed

- Package structure fixed (src/flattune layout)
- Config field names standardized
