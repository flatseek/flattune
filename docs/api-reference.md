# API Reference

## Flattune Package

```python
from flattune import FlattuneConfig
from flattune import Flatseek, create_provider
from flattune import DatasetBuilder
from flattune import TrainerBase, TrainerFactory
from flattune import LMStudioClient, OllamaClient
from flattune import BenchmarkRunner, BenchmarkReporter
```

## Configuration

### FlattuneConfig

```python
config = FlattuneConfig.from_yaml("config.yml")
```

### Config Classes

- `FlatseekConfig` - Flatseek data source
- `ModelConfig` - Model configuration
- `DatasetConfig` - Dataset generation
- `TrainConfig` - Training parameters
- `ExportConfig` - Export settings
- `BenchmarkConfig` - Benchmark settings

## FlatseekProvider

### create_provider

```python
from flattune.flatseek import create_provider

provider = create_provider(config.flatseek)
```

### FlatseekProvider Methods

```python
# Get column names
columns = provider.columns()

# Search records
results = provider.search(query="magnitude:>=5", limit=100)

# Aggregate
aggs = provider.aggregate(query="*", aggs={"avg_magnitude": "avg(magnitude)"})

# Export records
for record in provider.export(query="*"):
    print(record)

# Get statistics
stats = provider.stats()

# Stream records (memory efficient)
for record in provider.stream(query="*"):
    process(record)
```

## DatasetBuilder

```python
from flattune.dataset import DatasetBuilder

builder = DatasetBuilder(
    flatseek_config=config.flatseek,
    dataset_config=config.dataset,
    output_dir="./dataset",
    interactive=True,
    yes_flag=False,
)
builder.build(output_dir="./dataset")
```

### DatasetBuilder Methods

```python
# Build with legacy mode
builder.build(documents=docs, dataset_name="dataset")

# Build with new pipeline (if source_path provided)
builder.build(
    documents=docs,
    dataset_name="dataset",
    source_path="./docs/*.md",
    user_requested_types=["facts", "context_qa"],
)
```

## Build Pipeline (NEW)

The build pipeline provides intelligent dataset planning with automatic source detection.

```python
from flattune.build import BuildPipeline, SourceDetector, ContentAnalyzer, BuildPlanner
```

### SourceDetector

```python
from flattune.build import SourceDetector, SourceDetectionResult

detector = SourceDetector()
result: SourceDetectionResult = detector.detect("./README.md")
print(result.source_type)  # SourceType.MARKDOWN
print(result.confidence)  # 1.0
```

### ContentAnalyzer

```python
from flattune.build import ContentAnalyzer

analyzer = ContentAnalyzer()
analysis = analyzer.analyze(detection_result, sample_content)
print(analysis.suggested_types)  # [("facts", 0.7), ("context_qa", 0.6)]
```

### BuildPlanner

```python
from flattune.build import BuildPlanner

planner = BuildPlanner()
plan = planner.create_plan(detection, analysis, document_count=10)
print(plan.output_structure)  # {"knowledge": ["facts", "glossary"], "qa": ["context_qa"]}
```

### BuildPipeline

```python
from flattune.build import BuildPipeline

pipeline = BuildPipeline(
    source="./docs/*.md",
    config=dataset_config,
    output_dir="./build",
    interactive=True,
    yes_flag=False,
)
output_paths = pipeline.run(documents=doc_iterator)
```

### Registry Classes

```python
from flattune.build.registry import (
    CategoryRegistry,
    DatasetTypeRegistry,
    SourceRegistry,
)

# List registered categories
categories = CategoryRegistry.list_all_categories()

# List registered dataset types
types = DatasetTypeRegistry.list_all_types()

# Get specific type
dt = DatasetTypeRegistry.get("facts")
print(dt.category)  # "knowledge"
print(dt.generator_name)  # "facts"
```

## Trainer

### TrainerFactory

```python
from flattune.trainer import TrainerFactory

trainer = TrainerFactory.create(config.train.backend, config)
```

### TrainerBase Methods

```python
# Train
result = trainer.train(dataset_path, config)
# Returns: {final_loss, checkpoint_path, training_time, loss_history}

# Merge
merged_path = trainer.merge(checkpoint_path)

# Export
export_path = trainer.export(model_path, export_config)

# Evaluate
eval_result = trainer.evaluate(model_path, test_data)
```

### UnslothTrainer

```python
from flattune.trainer import UnslothTrainer

trainer = UnslothTrainer(config)
trainer.train("./dataset", config.train)
trainer.merge("./checkpoints/final")
```

## Exporter

```python
from flattune.exporter import create_exporter, ExportFormat

exporter = create_exporter(ExportFormat.GGUF)
exporter.export(model_path, output_dir)
```

## Clients

### LMStudioClient

```python
from flattune.lmstudio import LMStudioClient

client = LMStudioClient()
client.connect("localhost", 1234)

# Import model
client.import_model("./model.gguf")

# Run prompt
response = client.run_prompt("What is 2+2?", system_prompt="You are a calculator")

# Benchmark
results = client.benchmark(categories=["latency", "tokens_per_sec"])
```

### OllamaClient

```python
from flattune.ollama import OllamaClient

client = OllamaClient()

# Create modelfile
client.create_modelfile("./model", "my-model")

# Import
client.import_model("./Modelfile")

# Run prompt
response = client.run_prompt("What is 2+2?")

# Benchmark
results = client.benchmark(categories=["latency", "accuracy"])
```

## Benchmark

### BenchmarkRunner

```python
from flattune.benchmark import BenchmarkRunner

runner = BenchmarkRunner(config.benchmark)
results = runner.run(model_path, backend="lmstudio")
```

### BenchmarkReporter

```python
from flattune.benchmark import BenchmarkReporter

reporter = BenchmarkReporter()
reporter.generate_report(results, output_dir="./benchmark")
```

## Utils

### Logging

```python
from flattune.utils import setup_logging, get_logger

setup_logging("train")
logger = get_logger("train")
logger.info("Starting training...")
```

### File System

```python
from flattune.utils import create_run_folder, get_run_folder

run_folder = create_run_folder(config)
# Returns: outputs/PROJECT_TIMESTAMP/
```

### Reproducibility

```python
from flattune.utils import set_seed, get_git_info, get_timestamp

set_seed(42)
git_info = get_git_info()
timestamp = get_timestamp()
```

### Hashing

```python
from flattune.utils import compute_hash

dataset_hash = compute_hash(dataset_path)
flatseek_hash = compute_hash(flatseek_path)
```
