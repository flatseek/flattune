"""CLI entry point for FlatTune."""

import json
from pathlib import Path

import click

from flattune.benchmark.reporter import BenchmarkReporter
from flattune.benchmark.runner import BenchmarkRunner
from flattune.config import ExportFormat, FlatTuneConfig
from flattune.dataset.builder import DatasetBuilder
from flattune.exporter.exporter import ModelExporter
from flattune.flatseek.auto import create_provider
from flattune.teach.cli import teach as teach_group
from flattune.trainer.factory import TrainerFactory
from flattune.utils.fs import create_run_folder, get_run_folder
from flattune.utils.hash import compute_dataset_hash
from flattune.utils.logging import get_logger, setup_logging
from flattune.utils.reproducibility import (
    ReproducibilityContext,
    get_environment_info,
    get_git_info,
    get_timestamp,
    set_seed,
)

logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FlatTune - Fine-tuning pipeline for Flatseek datasets."""
    pass


# Register teach group
cli.add_command(teach_group)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--yes", "-y", is_flag=True, help="Skip interactive approval and use suggested types")
@click.option("--source", type=str, help="Source file/directory (overrides config)")
@click.option("--types", "-t", multiple=True, help="Dataset types to generate (can specify multiple)")
def build(config_file: str, yes: bool = False, source: str | None = None, types: tuple = ()):
    """Extract data and generate dataset from Flatseek index.

    Now supports:
    - Source auto-detection
    - Automatic dataset type suggestion
    - Interactive approval for type selection
    - --yes flag for non-interactive mode
    - --types to explicitly specify dataset types
    """
    setup_logging("build")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = create_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[build] Run folder: {run_folder}")

    # Set seed for reproducibility
    set_seed(config.train.seed)

    # Determine source path
    source_path = source or config.flatseek.path or "."

    # Create FlatseekProvider (only if flatseek path is configured)
    flatseek_cfg = config.flatseek
    provider = None
    if flatseek_cfg.path:
        provider = create_provider(
            path=flatseek_cfg.path,
            mode=flatseek_cfg.mode,
            source=flatseek_cfg.source,
            query=flatseek_cfg.query,
            encryption_key=flatseek_cfg.encryption_key,
        )
        click.echo("[build] Extracting data from Flatseek index...")
        logger.info("Connected to FlatSeek provider")

        # Get index stats
        stats = provider.stats()
        logger.info(f"Index stats: {stats}")
        click.echo(f"[build] Index stats: {json.dumps(stats, indent=2)}")
    else:
        stats = {}

    # Determine types
    type_list = list(types) if types else None

    # Create dataset builder with new pipeline support
    dataset_builder = DatasetBuilder(
        flatseek_config=flatseek_cfg,
        dataset_config=config.dataset,
        output_dir=run_folder / "dataset",
        interactive=True,
        yes_flag=yes,
    )

    # Stream documents from FlatSeek (if provider available)
    if provider:
        query = flatseek_cfg.query or "*"
        documents = provider.stream(query)
    else:
        # For non-FlatSeek sources, we'll use an empty iterator
        # The build pipeline will handle source detection
        documents = iter([])

    click.echo("[build] Building dataset...")
    if type_list:
        click.echo(f"[build] Using specified types: {type_list}")
    elif config.dataset.generators:
        click.echo(f"[build] Using config generators: {config.dataset.generators}")
    logger.info(f"Building dataset from query: {query if provider else 'N/A'}")

    # Build dataset with new pipeline
    dataset_paths = dataset_builder.build(
        documents=documents,
        dataset_name=config.name,
        source_path=source_path,
        user_requested_types=type_list,
    )

    # Compute dataset hash for reproducibility
    dataset_hash = compute_dataset_hash(run_folder / "dataset")
    logger.info(f"Dataset hash: {dataset_hash}")

    # Save build metadata
    build_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "git": get_git_info(),
        "environment": get_environment_info(),
        "dataset_hash": dataset_hash,
        "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
        "flatseek_stats": stats,
        "dataset_types": type_list or config.dataset.generators,
    }

    metadata_path = run_folder / "build_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(build_metadata, f, indent=2)

    click.echo("[build] Dataset generation complete.")
    click.echo(f"[build] Dataset saved to: {run_folder / 'dataset'}")
    logger.info(f"Build metadata saved to {metadata_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file: str):
    """Train a model using the dataset."""
    setup_logging("train")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = get_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[train] Run folder: {run_folder}")

    # Set seed for reproducibility
    set_seed(config.train.seed)

    # Load dataset paths
    dataset_dir = run_folder / "dataset"
    train_dataset = dataset_dir / f"{config.name}_train.jsonl"
    dataset_dir / f"{config.name}_val.jsonl"

    if not train_dataset.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_dataset}")

    click.echo(f"[train] Starting training with {config.train.backend.value} backend...")
    click.echo(f"[train] Model: {config.model.path}")
    click.echo(f"[train] Dataset: {train_dataset}")
    click.echo(f"[train] Epochs: {config.train.epochs}, LR: {config.train.lr}")

    # Determine model path/repo based on source
    if config.model.source.value == "huggingface":
        model_path = config.model.repo
    else:
        model_path = config.model.path

    click.echo(f"[train] Model: {model_path}")

    # Create reproducibility context
    with ReproducibilityContext(seed=config.train.seed) as ctx:
        # Create trainer via factory
        trainer = TrainerFactory.create(
            backend=config.train.backend,
            model_path=model_path,
            dataset_path=str(train_dataset),
            output_dir=run_folder / "checkpoints",
            config=config.train,
        )

        # Run training
        logger.info("Starting training...")
        metrics = trainer.train()

        logger.info(f"Training metrics: {metrics}")

    # Save training metadata
    train_metadata = {
        "status": "completed",
        "timestamp": get_timestamp(),
        "backend": config.train.backend.value,
        "metrics": metrics,
        "config": config.train.to_dict(),
        "reproducibility": ctx.get_info(),
    }

    metadata_path = run_folder / "train_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(train_metadata, f, indent=2)

    # Save checkpoints info
    checkpoints = trainer.get_checkpoints()
    checkpoints_info = {
        "total_checkpoints": len(checkpoints),
        "checkpoints": [str(cp) for cp in checkpoints],
        "latest": str(trainer.get_latest_checkpoint()) if checkpoints else None,
    }

    checkpoints_path = run_folder / "checkpoints_info.json"
    with open(checkpoints_path, "w") as f:
        json.dump(checkpoints_info, f, indent=2)

    click.echo("[train] Training complete.")
    click.echo(f"[train] Metrics: {json.dumps(metrics, indent=2)}")
    click.echo(f"[train] Saved {len(checkpoints)} checkpoints")
    logger.info(f"Training metadata saved to {metadata_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def merge(config_file: str):
    """Merge LoRA adapter with base model."""
    setup_logging("merge")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = get_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[merge] Run folder: {run_folder}")

    # Load trainer to get checkpoint
    checkpoints_info_path = run_folder / "checkpoints_info.json"
    if checkpoints_info_path.exists():
        with open(checkpoints_info_path) as f:
            checkpoints_info = json.load(f)
        latest_checkpoint = checkpoints_info.get("latest")
    else:
        latest_checkpoint = None

    click.echo("[merge] Merging adapter with base model...")
    click.echo(f"[merge] Base model: {config.model.path}")
    if latest_checkpoint:
        click.echo(f"[merge] Adapter checkpoint: {latest_checkpoint}")

    # Create trainer to perform merge
    # Use repo for HuggingFace source, path for local source
    merge_model_path = config.model.repo or config.model.path
    trainer = TrainerFactory.create(
        backend=config.train.backend,
        model_path=merge_model_path,
        dataset_path=str(run_folder / "dataset" / f"{config.name}_train.jsonl"),
        output_dir=run_folder / "checkpoints",
        config=config.train,
    )

    # Perform merge
    merged_path = trainer.merge()

    logger.info(f"Merged model saved to: {merged_path}")

    # Save merge metadata
    merge_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "base_model": config.model.path,
        "adapter_checkpoint": latest_checkpoint,
        "merged_path": str(merged_path),
    }

    metadata_path = run_folder / "merge_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(merge_metadata, f, indent=2)

    click.echo("[merge] Merge complete.")
    click.echo(f"[merge] Merged model saved to: {merged_path}")
    logger.info(f"Merge metadata saved to {metadata_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def export(config_file: str):
    """Export model to target format (GGUF/MLX/HF)."""
    setup_logging("export")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = get_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[export] Run folder: {run_folder}")

    # Get merged model path
    merge_metadata_path = run_folder / "merge_metadata.json"
    if merge_metadata_path.exists():
        with open(merge_metadata_path) as f:
            merge_metadata = json.load(f)
        merged_model_path = merge_metadata.get("merged_path", str(run_folder / "merged"))
    else:
        merged_model_path = str(run_folder / "merged")

    click.echo(f"[export] Exporting to {config.export.format.value} format...")
    click.echo(f"[export] Quantization: {config.export.quantization}")
    click.echo(f"[export] Source model: {merged_model_path}")

    # Create exporter
    exporter = ModelExporter(
        model_path=Path(merged_model_path),
        output_dir=run_folder / "export",
        format=ExportFormat(config.export.format.value),
    )

    # Perform export
    export_path = exporter.export(quantization=config.export.quantization)

    logger.info(f"Exported model saved to: {export_path}")

    # Save export metadata
    export_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "format": config.export.format.value,
        "quantization": config.export.quantization,
        "source_path": merged_model_path,
        "export_path": str(export_path),
    }

    metadata_path = run_folder / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(export_metadata, f, indent=2)

    click.echo("[export] Export complete.")
    click.echo(f"[export] Exported model saved to: {export_path}")
    logger.info(f"Export metadata saved to {metadata_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def benchmark(config_file: str):
    """Benchmark the model using LM Studio, Ollama, or Transformers."""
    setup_logging("benchmark")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = get_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[benchmark] Run folder: {run_folder}")

    # Get exported model path
    export_metadata_path = run_folder / "export_metadata.json"
    if export_metadata_path.exists():
        with open(export_metadata_path) as f:
            export_metadata = json.load(f)
        model_path = export_metadata.get("export_path")
    else:
        model_path = str(run_folder / "export")

    click.echo(f"[benchmark] Running benchmark with {config.benchmark.backend.value} backend...")
    click.echo(f"[benchmark] Model: {model_path}")
    click.echo(f"[benchmark] Max tokens: {config.benchmark.max_tokens}")
    click.echo(f"[benchmark] Temperature: {config.benchmark.temperature}")

    # Create benchmark runner
    runner = BenchmarkRunner(
        config=config.benchmark,
        output_dir=run_folder / "benchmark",
    )

    # Check backend availability
    if not runner.is_backend_available():
        click.echo(f"[benchmark] Warning: {config.benchmark.backend.value} is not available.")
        click.echo("[benchmark] Make sure the service is running.")
        # Save error result
        error_result = {
            "status": "error",
            "error": f"{config.benchmark.backend.value} is not running",
            "timestamp": get_timestamp(),
        }
        error_path = run_folder / "benchmark" / "benchmark_error.json"
        with open(error_path, "w") as f:
            json.dump(error_result, f, indent=2)
        return

    # Load prompts
    prompts = []
    if config.benchmark.prompt_file:
        prompts = runner.load_prompts(config.benchmark.prompt_file)
        click.echo(f"[benchmark] Loaded {len(prompts)} prompts from {config.benchmark.prompt_file}")

    if not prompts:
        # Default prompts for testing
        prompts = [
            "What is machine learning?",
            "Explain the concept of neural networks.",
            "What are the benefits of fine-tuning models?",
            "Describe how LoRA fine-tuning works.",
            "What is retrieval-augmented generation?",
        ]
        click.echo(f"[benchmark] Using {len(prompts)} default prompts")

    # Run benchmark
    click.echo(f"[benchmark] Running {config.benchmark.num_runs} runs per prompt...")
    results = runner.run(prompts=prompts, model_path=model_path)

    logger.info(f"Benchmark results: {results}")

    # Generate benchmark report
    reporter = BenchmarkReporter(output_dir=run_folder / "benchmark")
    report_path = reporter.generate_report(results, output_name="benchmark_report")

    # Save summary
    summary = {
        "status": results.get("status", "unknown"),
        "backend": results.get("backend"),
        "total_runs": results.get("total_runs", 0),
        "total_tokens": results.get("total_tokens", 0),
        "average_tokens_per_second": results.get("average_tokens_per_second", 0),
        "report_path": str(report_path),
        "timestamp": get_timestamp(),
    }

    summary_path = run_folder / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    click.echo("[benchmark] Benchmark complete.")
    click.echo(f"[benchmark] Total runs: {results.get('total_runs', 0)}")
    click.echo(f"[benchmark] Average tokens/sec: {results.get('average_tokens_per_second', 0):.2f}")
    click.echo(f"[benchmark] Report saved to: {report_path}")
    logger.info(f"Benchmark summary saved to {summary_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def report(config_file: str):
    """Generate training and benchmark report."""
    setup_logging("report")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = get_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo(f"[report] Run folder: {run_folder}")

    # Load all metadata
    metadata_files = {
        "build": run_folder / "build_metadata.json",
        "train": run_folder / "train_metadata.json",
        "merge": run_folder / "merge_metadata.json",
        "export": run_folder / "export_metadata.json",
    }

    metadata = {}
    for name, path in metadata_files.items():
        if path.exists():
            with open(path) as f:
                metadata[name] = json.load(f)
        else:
            metadata[name] = {"status": "not_run", "timestamp": None}

    # Load benchmark results
    benchmark_summary_path = run_folder / "benchmark_summary.json"
    if benchmark_summary_path.exists():
        with open(benchmark_summary_path) as f:
            benchmark_data = json.load(f)
    else:
        benchmark_data = {"status": "not_run"}

    # Generate markdown report
    click.echo("[report] Generating markdown report...")

    report_lines = [
        "# FlatTune Report",
        "",
        f"**Run Name**: {config.name}",
        f"**Timestamp**: {get_timestamp()}",
        "",
        "## Git Info",
        "",
        f"- Branch: {metadata.get('build', {}).get('git', {}).get('branch', 'unknown')}",
        f"- Commit: {metadata.get('build', {}).get('git', {}).get('commit', 'unknown')}",
        f"- Dirty: {metadata.get('build', {}).get('git', {}).get('is_dirty', 'unknown')}",
        "",
        "## Build",
        "",
        f"- Status: {metadata.get('build', {}).get('status', 'unknown')}",
        f"- Dataset Hash: {metadata.get('build', {}).get('dataset_hash', 'unknown')}",
        "",
        "## Training",
        "",
        f"- Status: {metadata.get('train', {}).get('status', 'unknown')}",
        f"- Backend: {metadata.get('train', {}).get('backend', 'unknown')}",
        f"- Metrics: {json.dumps(metadata.get('train', {}).get('metrics', {}), indent=2)}",
        "",
        "## Merge",
        "",
        f"- Status: {metadata.get('merge', {}).get('status', 'unknown')}",
        f"- Merged Path: {metadata.get('merge', {}).get('merged_path', 'unknown')}",
        "",
        "## Export",
        "",
        f"- Status: {metadata.get('export', {}).get('status', 'unknown')}",
        f"- Format: {metadata.get('export', {}).get('format', 'unknown')}",
        f"- Export Path: {metadata.get('export', {}).get('export_path', 'unknown')}",
        "",
        "## Benchmark",
        "",
        f"- Status: {benchmark_data.get('status', 'unknown')}",
        f"- Backend: {benchmark_data.get('backend', 'unknown')}",
        f"- Total Runs: {benchmark_data.get('total_runs', 0)}",
        f"- Total Tokens: {benchmark_data.get('total_tokens', 0)}",
        f"- Average Tokens/sec: {benchmark_data.get('average_tokens_per_second', 0):.2f}",
        "",
    ]

    report_path = run_folder / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Save consolidated metrics.json
    metrics = {
        "build": metadata.get("build", {}),
        "train": {
            "status": metadata.get("train", {}).get("status"),
            "metrics": metadata.get("train", {}).get("metrics", {}),
        },
        "merge": metadata.get("merge", {}),
        "export": metadata.get("export", {}),
        "benchmark": benchmark_data,
    }

    metrics_path = run_folder / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save benchmark.json
    benchmark_json = {
        "summary": benchmark_data,
        "config": config.benchmark.to_dict(),
    }

    benchmark_json_path = run_folder / "benchmark.json"
    with open(benchmark_json_path, "w") as f:
        json.dump(benchmark_json, f, indent=2)

    # Generate loss.png if training metrics available
    train_metadata = metadata.get("train", {})
    if train_metadata.get("metrics"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Extract loss values if available
            metrics_data = train_metadata.get("metrics", {})
            if "loss_history" in metrics_data:
                losses = metrics_data["loss_history"]
                steps = list(range(len(losses)))

                plt.figure(figsize=(10, 6))
                plt.plot(steps, losses, label="Training Loss")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title(f"Training Loss - {config.name}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                loss_png_path = run_folder / "loss.png"
                plt.savefig(loss_png_path, dpi=150, bbox_inches="tight")
                plt.close()

                logger.info(f"Loss curve saved to {loss_png_path}")
        except Exception as e:
            logger.warning(f"Could not generate loss.png: {e}")

    click.echo("[report] Report complete.")
    click.echo(f"[report] Report saved to: {report_path}")
    click.echo(f"[report] Metrics saved to: {metrics_path}")
    click.echo(f"[report] Benchmark data saved to: {benchmark_json_path}")
    logger.info("Report generation complete")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def run(config_file: str):
    """Run the full pipeline: extract -> generate -> train -> merge -> export -> benchmark -> report."""
    setup_logging("run")
    logger.info(f"Loading config from {config_file}")

    config = FlatTuneConfig.from_yaml(config_file)
    run_folder = create_run_folder(config)

    logger.info(f"Run folder: {run_folder}")
    click.echo("[run] Full pipeline starting...")
    click.echo(f"[run] Run folder: {run_folder}")
    click.echo(f"[run] Config: {config_file}")

    # Set seed for reproducibility
    set_seed(config.train.seed)

    # Step 1: Build
    click.echo("\n" + "=" * 60)
    click.echo("[run:build] Extracting data and generating dataset...")
    click.echo("=" * 60)

    flatseek_cfg = config.flatseek
    provider = create_provider(
        path=flatseek_cfg.path or ".",
        mode=flatseek_cfg.mode,
        source=flatseek_cfg.source,
        query=flatseek_cfg.query,
        encryption_key=flatseek_cfg.encryption_key,
    )

    stats = provider.stats()
    logger.info(f"Index stats: {stats}")

    dataset_builder = DatasetBuilder(
        config=config.dataset,
        output_dir=run_folder / "dataset",
    )

    query = flatseek_cfg.query or "*"
    documents = provider.stream(query)

    dataset_paths = dataset_builder.build(
        documents=documents,
        dataset_name=config.name,
    )

    dataset_hash = compute_dataset_hash(run_folder / "dataset")

    build_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "git": get_git_info(),
        "environment": get_environment_info(),
        "dataset_hash": dataset_hash,
        "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
        "flatseek_stats": stats,
    }

    with open(run_folder / "build_metadata.json", "w") as f:
        json.dump(build_metadata, f, indent=2)

    click.echo("[run:build] Dataset generation complete.")

    # Step 2: Train
    click.echo("\n" + "=" * 60)
    click.echo("[run:train] Training model...")
    click.echo("=" * 60)

    train_dataset = dataset_paths.get("train", run_folder / "dataset" / f"{config.name}_train.jsonl")

    with ReproducibilityContext(seed=config.train.seed) as ctx:
        trainer = TrainerFactory.create(
            backend=config.train.backend,
            model_path=config.model.path,
            dataset_path=str(train_dataset),
            output_dir=run_folder / "checkpoints",
            config=config.train,
        )

        metrics = trainer.train()

    train_metadata = {
        "status": "completed",
        "timestamp": get_timestamp(),
        "backend": config.train.backend.value,
        "metrics": metrics,
        "config": config.train.to_dict(),
        "reproducibility": ctx.get_info(),
    }

    with open(run_folder / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    checkpoints = trainer.get_checkpoints()
    checkpoints_info = {
        "total_checkpoints": len(checkpoints),
        "checkpoints": [str(cp) for cp in checkpoints],
        "latest": str(trainer.get_latest_checkpoint()) if checkpoints else None,
    }

    with open(run_folder / "checkpoints_info.json", "w") as f:
        json.dump(checkpoints_info, f, indent=2)

    click.echo("[run:train] Training complete.")

    # Step 3: Merge
    click.echo("\n" + "=" * 60)
    click.echo("[run:merge] Merging adapter...")
    click.echo("=" * 60)

    latest_checkpoint = checkpoints_info.get("latest")

    merged_path = trainer.merge()

    merge_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "base_model": config.model.path,
        "adapter_checkpoint": latest_checkpoint,
        "merged_path": str(merged_path),
    }

    with open(run_folder / "merge_metadata.json", "w") as f:
        json.dump(merge_metadata, f, indent=2)

    click.echo("[run:merge] Merge complete.")

    # Step 4: Export
    click.echo("\n" + "=" * 60)
    click.echo("[run:export] Exporting model...")
    click.echo("=" * 60)

    exporter = ModelExporter(
        model_path=Path(merged_path),
        output_dir=run_folder / "export",
        format=ExportFormat(config.export.format.value),
    )

    export_path = exporter.export(quantization=config.export.quantization)

    export_metadata = {
        "status": "success",
        "timestamp": get_timestamp(),
        "format": config.export.format.value,
        "quantization": config.export.quantization,
        "source_path": str(merged_path),
        "export_path": str(export_path),
    }

    with open(run_folder / "export_metadata.json", "w") as f:
        json.dump(export_metadata, f, indent=2)

    click.echo("[run:export] Export complete.")

    # Step 5: Benchmark
    click.echo("\n" + "=" * 60)
    click.echo("[run:benchmark] Running benchmark...")
    click.echo("=" * 60)

    runner = BenchmarkRunner(
        config=config.benchmark,
        output_dir=run_folder / "benchmark",
    )

    prompts = []
    if config.benchmark.prompt_file:
        prompts = runner.load_prompts(config.benchmark.prompt_file)

    if not prompts:
        prompts = [
            "What is machine learning?",
            "Explain the concept of neural networks.",
            "What are the benefits of fine-tuning models?",
            "Describe how LoRA fine-tuning works.",
            "What is retrieval-augmented generation?",
        ]

    results = runner.run(prompts=prompts, model_path=str(export_path))

    reporter = BenchmarkReporter(output_dir=run_folder / "benchmark")
    report_path = reporter.generate_report(results, output_name="benchmark_report")

    summary = {
        "status": results.get("status", "unknown"),
        "backend": results.get("backend"),
        "total_runs": results.get("total_runs", 0),
        "total_tokens": results.get("total_tokens", 0),
        "average_tokens_per_second": results.get("average_tokens_per_second", 0),
        "report_path": str(report_path),
        "timestamp": get_timestamp(),
    }

    with open(run_folder / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo("[run:benchmark] Benchmark complete.")

    # Step 6: Report
    click.echo("\n" + "=" * 60)
    click.echo("[run:report] Generating report...")
    click.echo("=" * 60)

    metadata_files = {
        "build": run_folder / "build_metadata.json",
        "train": run_folder / "train_metadata.json",
        "merge": run_folder / "merge_metadata.json",
        "export": run_folder / "export_metadata.json",
    }

    metadata = {}
    for name, path in metadata_files.items():
        if path.exists():
            with open(path) as f:
                metadata[name] = json.load(f)
        else:
            metadata[name] = {"status": "not_run", "timestamp": None}

    report_lines = [
        "# FlatTune Report",
        "",
        f"**Run Name**: {config.name}",
        f"**Timestamp**: {get_timestamp()}",
        "",
        "## Git Info",
        "",
        f"- Branch: {metadata.get('build', {}).get('git', {}).get('branch', 'unknown')}",
        f"- Commit: {metadata.get('build', {}).get('git', {}).get('commit', 'unknown')}",
        "",
        "## Build",
        "",
        f"- Status: {metadata.get('build', {}).get('status', 'unknown')}",
        f"- Dataset Hash: {metadata.get('build', {}).get('dataset_hash', 'unknown')}",
        "",
        "## Training",
        "",
        f"- Status: {metadata.get('train', {}).get('status', 'unknown')}",
        f"- Backend: {metadata.get('train', {}).get('backend', 'unknown')}",
        f"- Metrics: {json.dumps(metadata.get('train', {}).get('metrics', {}), indent=2)}",
        "",
        "## Merge",
        "",
        f"- Status: {metadata.get('merge', {}).get('status', 'unknown')}",
        "",
        "## Export",
        "",
        f"- Status: {metadata.get('export', {}).get('status', 'unknown')}",
        f"- Format: {metadata.get('export', {}).get('format', 'unknown')}",
        "",
        "## Benchmark",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Total Runs: {summary.get('total_runs', 0)}",
        f"- Average Tokens/sec: {summary.get('average_tokens_per_second', 0):.2f}",
        "",
    ]

    report_path = run_folder / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    metrics = {
        "build": metadata.get("build", {}),
        "train": {
            "status": metadata.get("train", {}).get("status"),
            "metrics": metadata.get("train", {}).get("metrics", {}),
        },
        "merge": metadata.get("merge", {}),
        "export": metadata.get("export", {}),
        "benchmark": summary,
    }

    with open(run_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(run_folder / "benchmark.json", "w") as f:
        json.dump({"summary": summary, "config": config.benchmark.to_dict()}, f, indent=2)

    click.echo("[run:report] Report complete.")

    # Final summary
    click.echo("\n" + "=" * 60)
    click.echo("[run] Full pipeline complete!")
    click.echo("=" * 60)
    click.echo(f"[run] Results saved to: {run_folder}")
    click.echo(f"[run] Report: {report_path}")
    click.echo(f"[run] Metrics: {run_folder / 'metrics.json'}")
    click.echo(f"[run] Checkpoints: {len(checkpoints)} saved")
    click.echo(f"[run] Average tokens/sec: {summary.get('average_tokens_per_second', 0):.2f}")

    logger.info(f"Full pipeline complete. Results in {run_folder}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
