# Teach Framework - Knowledge & Skill Compiler

The `teach` framework transforms Flattune into a **Knowledge & Skill Compiler**. It compiles knowledge, software capabilities, APIs, workflows, databases, SDKs, and documentation into high-quality fine-tuning datasets.

## Overview

The teach framework is separate from the **Build Pipeline** (which focuses on FlatSeek indexes with intelligent planning). The teach framework processes:

- Documents (Markdown, TXT, PDF, HTML, CSV, JSON)
- OpenAPI/Swagger specifications
- Database schemas
- MCP server definitions
- And more...

## Architecture

```
Source → Parser → Knowledge Graph → Teacher Model → Generator → Quality Filter → Dataset
```

- **Parsers**: Read and normalize data from various sources
- **Knowledge Graph**: Intermediate structured representation
- **Teachers**: Generate high-quality conversations using LLMs (optional)
- **Generators**: Create training samples from knowledge
- **Quality Filter**: Deduplication, scoring, hallucination detection

## Quick Start

### Distill Mode (No LLM Required)

Generate datasets using templates without an LLM teacher:

```bash
# Knowledge from documents
flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl

# OpenAPI specifications
flattune teach openapi api.json --distill -o api_dataset.jsonl

# Database schemas
flattune teach database schema.sql --distill -o sql_dataset.jsonl

# MCP servers
flattune teach mcp server.json --distill -o mcp_dataset.jsonl
```

### With LLM Teacher

For higher quality datasets, use an LLM teacher:

```bash
# With OpenAI
flattune teach knowledge ./docs/*.md --teacher openai -o dataset.jsonl

# With Anthropic
flattune teach openapi api.json --teacher anthropic -o api_dataset.jsonl

# With Ollama (local)
flattune teach knowledge ./docs/*.md --teacher ollama --model llama3.2 -o dataset.jsonl
```

## Commands

### flattune teach knowledge

Process documents and extract knowledge for training.

```bash
flattune teach knowledge [SOURCES...] [OPTIONS]

Options:
  --distill                      Use template-based generation (no LLM)
  --teacher {openai,anthropic,ollama}  LLM teacher for generation
  --model TEXT                   Teacher model name (default: gpt-4o-mini)
  --output, -o FILE             Output file (default: knowledge_dataset.jsonl)
  --sample-types, -s TEXT       Sample types to generate
  --max-samples N               Maximum samples to generate
```

**Sample Types**: qa, summarization, extraction, reasoning, comparison

### flattune teach openapi

Generate tool-calling datasets from OpenAPI/Swagger specs.

```bash
flattune teach openapi SPEC [OPTIONS]

Options:
  --distill                      Use template-based generation
  --teacher TEXT                LLM teacher for generation
  --output, -o FILE             Output file
```

**Sample Types**: tool_call, intent, slot_fill, validation, error_recovery, multi_turn

### flattune teach database

Generate NL-to-SQL datasets from database schemas.

```bash
flattune teach database SCHEMA [OPTIONS]

Options:
  --distill                      Use template-based generation
  --teacher TEXT                LLM teacher for generation
  --output, -o FILE             Output file
```

**Sample Types**: nl_to_sql, sql

### flattune teach software

Generate datasets from software APIs and tools.

```bash
flattune teach software [SOURCES...] [OPTIONS]

Options:
  --distill                      Use template-based generation
  --teacher TEXT                LLM teacher for generation
  --sample-types, -s TEXT       Sample types to generate
```

### flattune teach mcp

Generate datasets from MCP server definitions.

```bash
flattune teach mcp SERVER [OPTIONS]

Options:
  --distill                      Use template-based generation
  --teacher TEXT                 LLM teacher for generation
  --output, -o FILE             Output file
```

## Available Plugins

### Parsers

```bash
flattune teach list-parsers
```

| Parser | Extensions | Description |
|--------|------------|-------------|
| markdown | .md, .markdown | Markdown documents |
| json | .json | JSON files |
| jsonl | .jsonl | Newline-delimited JSON |
| text | .txt | Plain text |
| csv | .csv | CSV files |
| flatseek | .fsk | FlatSeek index files |
| openapi | .json, .yaml, .yml | OpenAPI/Swagger specs |

### Teachers

```bash
flattune teach list-teachers
```

| Teacher | Type | Description |
|---------|------|-------------|
| openai | remote | OpenAI API (gpt-4o-mini, gpt-4o) |
| anthropic | remote | Anthropic Claude API |
| ollama | local | Ollama local inference |

### Generators

```bash
flattune teach list-generators
```

| Generator | Sample Types | Description |
|-----------|--------------|-------------|
| qa | qa, question_answer | Question-answer pairs |
| tool_call | tool_call, intent, slot_fill | Tool calling examples |
| sql | nl_to_sql, sql | NL-to-SQL queries |

## Output Format

Generated datasets are in JSONL format:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "sample_type": "qa",
  "source": "README.md",
  "quality_score": 0.8,
  "metadata": {}
}
```

## Distill Mode

Distill mode uses template-based generation without an LLM teacher. This is useful for:

- Quick prototyping
- Large datasets
- When API keys are not available
- Initial data exploration

Quality may be lower than LLM-generated samples, but templates can still produce useful training data.

## Quality Pipeline

Generated samples pass through:

1. **Deduplication**: Exact and semantic deduplication
2. **Quality Scoring**: Length, diversity, completeness, format checks
3. **Hallucination Detection**: Flags uncertain or fabricated content

Samples below the minimum quality threshold are filtered out.

## Relationship to Build Pipeline

| Feature | Build Pipeline | Teach Framework |
|---------|---------------|-----------------|
| Source | FlatSeek index | Documents, APIs, DBs |
| Planning | Intelligent (auto-detect) | Template-based |
| Interactive | Yes (approval flow) | No |
| LLM Required | Optional | Optional (distill mode available) |
| Use Case | Structured data indexing | Knowledge compilation |

The **Build Pipeline** is designed for FlatSeek indexes with intelligent source detection and user-guided type selection. The **Teach Framework** is designed for compiling external knowledge sources (documents, APIs, databases) into training datasets, with optional LLM-powered generation.
