Prompt — Build FlatTune

You are building a new open-source project called FlatTune.

Vision

FlatTune is the missing bridge between Flatseek and modern LLM fine-tuning frameworks.

Instead of manually creating datasets, converting formats, downloading models, exporting GGUF, importing into LM Studio, etc., FlatTune automates the entire pipeline.

The workflow is:

Flatseek Index
      │
      ▼
Dataset Builder
      │
      ▼
Fine-tuning (Unsloth / future backends)
      │
      ▼
Merge
      │
      ▼
GGUF / MLX
      │
      ▼
LM Studio / Ollama
      │
      ▼
Benchmark

The goal is:

Any Flatseek index can become a fine-tuned local LLM in one command.

⸻

Positioning

Flatseek indexes structured knowledge.

FlatTune converts that knowledge into models.

Flatseek answers using retrieval.

FlatTune teaches models permanently.

⸻

Project Structure

flattune/
    flattune/
        dataset/
        models/
        trainer/
        exporter/
        benchmark/
        lmstudio/
        ollama/
        config/
        utils/
    examples/
    configs/
        earthquakes.yml
        legal.yml
        finance.yml
        customer_support.yml
    outputs/
        run-001/
        run-002/
        run-003/
    README.md

⸻

Main Pipeline

Pipeline should be modular.

Index
↓
Extract
↓
Generate Dataset
↓
Split
↓
Train
↓
Merge
↓
Export
↓
Import
↓
Benchmark
↓
Report

Every stage must be replaceable.

⸻

Data Sources

Support two Flatseek storage types.

Directory Index

Flatseek(directory)

Packed Archive

FlatseekFileStorageAdapter(.fsk)

Both should expose exactly the same interface.

DatasetSource
.search()
.aggregate()
.columns()
.stats()

The rest of the pipeline should not know whether data comes from:

* local directory

or

* packed FSK

or

* remote HTTP FSK

⸻

Dataset Builder

Dataset builder should support:

Instruction tuning

Chat

QA

Classification

Summarization

Extraction

RAG distillation

Synthetic QA generation

Conversation generation

Evaluation datasets

The generator system should be plugin-based.

Example:

generators/
    summary.py
    qa.py
    classification.py
    comparison.py
    safety.py
    custom.py

Each generator returns

instruction
input
output
metadata

⸻

Multiple Fine-tuning Jobs

FlatTune is NOT a single training script.

It is a training orchestrator.

Each fine-tuning job has its own folder.

Example

runs/
    earthquake/
        config.yml
        dataset/
        checkpoints/
        merged/
        benchmark/
    legal/
    finance/
    support/

Nothing is global.

Everything is isolated.

⸻

Configuration

Every training uses its own YAML.

Example

earthquake.yml
legal.yml
medical.yml
finance.yml

The CLI becomes

flattune train earthquake.yml
flattune train finance.yml
flattune benchmark finance.yml
flattune export finance.yml

⸻

Model Sources

The user should never be forced to download models manually.

Support:

Local GGUF

models/
    llama.gguf

⸻

Local MLX

mlx_model/

⸻

HuggingFace

model:
    source: huggingface
    repo: Qwen/Qwen3-4B

Automatically pull if missing.

⸻

GGUF

If user provides

gguf:
    path:

Automatically locate original HF model if needed.

If conversion is required,

perform it automatically.

⸻

MLX

If user provides

mlx:

Use MLX workflow.

⸻

Training Backends

Architecture should support multiple backends.

Initially implement

Unsloth

Future

Axolotl

MLX-LM LoRA

LlamaFactory

TRL

PEFT

The backend interface:

Trainer
.train()
.merge()
.export()
.evaluate()

Backends are plugins.

⸻

Output Formats

Support exporting

GGUF

MLX

HF merged

HF adapter

LoRA

Safetensors

Adapters only

⸻

Benchmark

FlatTune should benchmark automatically.

Support

LM Studio

Ollama

Native Transformers

Benchmark categories

Latency

Tokens/sec

Prompt quality

Instruction following

Domain accuracy

Hallucination tests

Regression tests

Before vs After

Produce markdown report.

⸻

LM Studio Integration

Support

Import GGUF

Reload model

Run prompts

Collect outputs

Save benchmark

No manual copying.

⸻

Ollama Integration

Automatically

create Modelfile

import model

run benchmark

remove temporary files

⸻

Training Reports

Every run generates

report.md
metrics.json
benchmark.json
training.json
loss.png
comparison.md

⸻

Dataset Metadata

Every dataset should include

dataset.json
metadata.json
statistics.json
categories.json
split.json

⸻

Reproducibility

Every run stores

seed

base model

dataset hash

Flatseek hash

training config

git commit

timestamp

environment

Everything should be reproducible.

⸻

CLI

Examples

flattune build earthquakes.yml
flattune train earthquakes.yml
flattune merge earthquakes.yml
flattune export earthquakes.yml
flattune benchmark earthquakes.yml
flattune report earthquakes.yml

Or

flattune run earthquakes.yml

which performs

extract
↓
generate dataset
↓
train
↓
merge
↓
export
↓
benchmark
↓
report

⸻

Multi-run Support

Support dozens of independent fine-tuning projects.

runs/
    finance/
    support/
    earthquake/
    chemistry/
    law/
    biology/

Each run has completely separate

dataset

checkpoint

config

merged model

GGUF

MLX

benchmark

report

No shared state.

⸻

Relationship with Flatseek

Flatseek remains completely independent.

FlatTune only consumes

Directory Index

or

Packed FSK

through the public Flatseek API.

There must be no dependency from Flatseek back into FlatTune.

The dependency direction is:

Flatseek
↓
FlatTune
↓
Unsloth
↓
LM Studio / Ollama

⸻

Long-term Vision

FlatTune should become the standard fine-tuning pipeline for Flatseek datasets.

The project should abstract away the complexity of:

* dataset generation
* prompt generation
* train/validation/test splitting
* model downloading
* backend selection
* LoRA training
* checkpoint management
* GGUF/MLX export
* LM Studio import
* Ollama import
* benchmarking
* reporting

so users only need to configure a single YAML file and run:

flattune run my_project.yml

Everything else should be automated, modular, backend-agnostic, reproducible, and extensible through plugins.

⸻

Flatseek Integration

FlatTune must support two integration modes with Flatseek.

Mode 1 — Embedded Library (Preferred)

FlatTune should use Flatseek directly as a Python library whenever it is installed.

Example:

from flatseek import Flatseek
from flatseek.flatseek_file import FlatseekFileStorageAdapter

This mode provides:

* maximum performance
* zero process spawning
* direct access to Flatseek APIs
* easier streaming
* lower memory usage

All dataset extraction should happen through the Flatseek Python API.

⸻

Mode 2 — Flatseek CLI

If Flatseek is not available as a Python package, FlatTune should automatically fall back to invoking the Flatseek CLI.

Examples:

flatseek search \
    --index earthquakes.fsk \
    --query "magnitude:>=5"
flatseek export \
    --index earthquakes.fsk \
    --query "*"


FlatTune should execute the CLI through subprocess and parse JSON output.

The CLI mode makes FlatTune language-agnostic and allows Flatseek to evolve independently.

⸻

Automatic Backend Detection

FlatTune should automatically determine which integration mode to use.

Priority:

1. Explicit configuration in YAML
2. Embedded Python library
3. Flatseek CLI
4. Error with clear installation guidance

Example:

flatseek:
  mode: auto

or

flatseek:
  mode: library

or

flatseek:
  mode: cli

⸻

Common Interface

The rest of FlatTune must never know which backend is being used.

Define a common interface:

class FlatseekProvider:
    def columns(self):
        ...
    def search(self, query):
        ...
    def aggregate(self, query, aggs):
        ...
    def export(self, query):
        ...
    def stats(self):
        ...

Implementations:

LibraryProvider
    └── uses Flatseek Python API
CLIProvider
    └── uses flatseek executable

All downstream components (dataset generation, statistics, sampling, fine-tuning) consume only the FlatseekProvider interface.

⸻

Supported Index Types

Both integration modes must transparently support:

* Flatseek directory index
* Flatseek packed .fsk
* Remote .fsk over HTTP
* Encrypted .fsk (via encryption key)

The dataset generation pipeline must not distinguish between these storage types.

⸻

Streaming Support

FlatTune should never require loading the entire index into memory.

Support iterator-based extraction:

for doc in provider.stream(query):
    ...

This enables training datasets to be generated from multi-million document Flatseek indexes.

⸻

Design Principle

FlatTune must never duplicate search, filtering, aggregation, or indexing logic already implemented by Flatseek.

Flatseek remains the search engine.

FlatTune is only responsible for:

* extracting records
* generating AI datasets
* orchestrating fine-tuning
* exporting trained models
* benchmarking results

This keeps responsibilities clean:

            Flatseek
      (Search & Index Engine)
               │
               ▼
            FlatTune
(Dataset + Fine-tuning Orchestrator)
               │
               ▼
     Unsloth / MLX-LM / Axolotl
               │
               ▼
     GGUF / MLX / LM Studio / Ollama

This architecture ensures FlatTune remains a thin orchestration layer while Flatseek continues to own all indexing and retrieval capabilities.

# Flattune Evolution: From Dataset Generator to Knowledge & Skill Compiler

## Vision

Transform Flattune from a simple dataset generator into a **Knowledge & Skill Compiler**.

Instead of only generating Q&A datasets, Flattune should be able to compile knowledge, software capabilities, APIs, workflows, databases, SDKs, documentation, and search indexes into high-quality fine-tuning datasets.

The objective is to teach smaller language models new skills and domain-specific knowledge rather than merely memorizing text.

---

# Core Principles

- Modular architecture
- Plugin-based parsers
- Pluggable teacher models
- Extensible generators
- Unified pipeline
- Support both local and remote teacher models
- Backward compatible with the current `generate` workflow

---

# Unified Pipeline

All teaching modes must share the same pipeline.

Source
    │
    ▼
Parser
    │
    ▼
Intermediate Knowledge Graph
    │
    ▼
Teacher Model
    │
    ▼
Conversation Generator
    │
    ▼
Quality Evaluation
    │
    ▼
Deduplication
    │
    ▼
Dataset Export

Responsibilities:

Parser
- Read and normalize data.
- Never generate conversations.

Knowledge Graph
- Build a structured intermediate representation.
- Independent from the dataset format.

Teacher
- Generate high-quality conversations.
- May use local or remote LLMs.

Generator
- Produce datasets from the knowledge graph.

Quality Evaluator
- Score generated samples.
- Reject hallucinations.
- Remove low-quality outputs.

Exporter
- Export into multiple dataset formats.

---

# New CLI

Current:

flattune generate

Introduce:

flattune teach

with specialized subcommands.

---

# Teach Knowledge

Command

flattune teach knowledge

Supported sources

- Markdown
- TXT
- PDF
- HTML
- CSV
- JSON
- FlatSeek Index
- Directories

Parser should ingest every document.

The generator should create diverse dataset types instead of one Q&A pair per document.

Supported sample types:

- Question Answering
- Summarization
- Long-form Answers
- Information Extraction
- Comparison
- Classification
- Rewrite
- Explanation
- Reasoning
- Chain-of-thought style reasoning (optional)
- Multi-turn conversation

Target:
Generate dozens or hundreds of training samples from each document whenever possible.

---

# Teach Software

Command

flattune teach software

Supported sources

- OpenAPI
- Swagger
- GraphQL Schema
- MCP Server
- CLI Documentation
- SDK Documentation
- REST API Documentation

Generate datasets for:

Intent Recognition

User:
Create an invoice

↓

Tool Call

create_invoice(...)

Parameter Extraction

User:
Transfer two million dollars to John

↓

transfer(
    amount=2000000,
    recipient="John"
)

Conversation

User
Assistant
Tool
Assistant

Workflow

User
↓
Tool A
↓
Tool B
↓
Assistant

Include variations:

- Formal
- Casual
- Typos
- Incomplete requests
- Ambiguous requests
- Follow-up questions
- Multilingual prompts

---

# Teach Workflow

Command

flattune teach workflow

Supported sources

- SOP
- BPMN
- Markdown
- Documentation

Generate multi-step task datasets.

Example

User

↓

Tool A

↓

Tool B

↓

Tool C

↓

Assistant

Teach procedural reasoning rather than isolated tool calls.

---

# Teach Database

Command

flattune teach database

Supported sources

- PostgreSQL schema
- MySQL schema
- SQLite schema
- SQLAlchemy models

Generate:

Natural Language

↓

SQL

or

Natural Language

↓

Tool Calls

Examples

Top customers this month

↓

SQL

Revenue by country

↓

SQL

Orders created yesterday

↓

SQL

Support schema-aware SQL generation and optional tool-based execution.

---

# Teach CLI

Command

flattune teach cli

Supported sources

- Executable
- --help
- man pages
- Markdown docs

Generate

Compress this video

↓

ffmpeg ...

Current branch push

↓

git push

Create datasets for command discovery, parameter selection, troubleshooting, and workflow automation.

---

# Teach Python Library

Command

flattune teach python

Supported sources

- Package
- Source Code
- Type Hints
- Docstrings
- README
- Examples

Generate

Instruction

↓

Python Code

Example

Read CSV

↓

pandas.read_csv(...)

Include multiple solution styles when applicable.

---

# Teach OpenAPI

This should become one of the flagship features.

Command

flattune teach openapi openapi.json

Parser should understand

- Endpoints
- Parameters
- Authentication
- Request Body
- Response Schema
- Examples
- Enums
- Error Codes

Generate datasets for

- Intent Detection
- Tool Calling
- Slot Filling
- Parameter Extraction
- Confirmation Dialogues
- Validation
- Retry Logic
- Error Recovery
- Follow-up Questions
- Context Awareness
- Multi-turn Conversations
- Negative Examples
- Hallucination Prevention

Each endpoint should produce many different conversations instead of one example.

---

# Teach MCP

Command

flattune teach mcp server.json

Parser should read every available MCP tool.

Generate

- Single Tool Calls
- Multi-tool Workflows
- Context-aware Conversations
- Parameter Extraction
- Recovery Scenarios

Teach how multiple MCP tools work together.

---

# Teacher Model Abstraction

Teacher models should be interchangeable.

Support local providers

- llama.cpp
- Ollama
- LM Studio
- vLLM

Support remote providers

- OpenAI
- Anthropic
- Gemini
- OpenRouter
- Any OpenAI-compatible API

Example interface

Teacher.generate(...)

Adding a new provider should never require changes elsewhere in the pipeline.

---

# Dataset Formats

Support exporting to

- ShareGPT
- OpenAI Chat
- Alpaca
- ChatML
- LlamaFactory
- Unsloth
- Custom JSONL

Exporters should be independent plugins.

---

# Dataset Quality Pipeline

Implement optional quality stages

- Exact duplicate removal
- Semantic duplicate removal
- Diversity scoring
- Difficulty balancing
- Length balancing
- Teacher self-review
- Teacher rewrite
- Automatic answer validation
- Hallucination detection
- Dataset statistics

Only high-quality samples should be exported.

---

# Parser Plugin System

Implement a parser registry.

Example

ParserRegistry.register(...)

Every parser should be isolated from the core engine.

Future plugins should include

- Confluence
- Notion
- GitHub
- GitLab
- Swagger URL
- OpenAPI URL
- Postman Collection
- MCP Registry
- FlatSeek Remote Index
- Generic REST Documentation

No core modifications should be required when adding new parsers.

---

# Generator Plugin System

Generators should also be pluggable.

Examples

- QA Generator
- Tool Calling Generator
- Workflow Generator
- SQL Generator
- Summarization Generator
- Code Generator
- Classification Generator
- RAG Distillation Generator
- Reasoning Generator

New generators should be registerable without changing existing code.

---

# Long-Term Goal

Flattune should evolve into a general-purpose **Knowledge & Skill Compiler**.

Instead of simply converting documents into Q&A datasets, it should compile structured and unstructured sources—including APIs, SDKs, databases, documentation, workflows, CLI tools, and FlatSeek indexes—into high-quality fine-tuning datasets that teach models new capabilities.

The architecture should emphasize:

- Clean Architecture
- Separation of Concerns
- Plugin-first Design
- Extensibility
- Scalability
- Reusability
- Maintainability

The existing `generate` command should remain fully functional for backward compatibility, while the new `teach` framework becomes the recommended workflow for future development.