"""Teach CLI commands - Knowledge & Skill Compiler."""

from pathlib import Path

import click

from flattune.teach.pipeline import PipelineConfig, TeachPipeline
from flattune.teach.registry import GeneratorRegistry, ParserRegistry, TeacherRegistry
from flattune.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
def teach():
    """Teach commands - Compile knowledge into fine-tuning datasets."""
    pass


@teach.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--output", "-o", default="knowledge_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default=None, help="Teacher model (openai/anthropic/ollama). Omit for distill mode.")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
@click.option("--sample-types", "-s", multiple=True, default=["qa", "summarization", "extraction", "reasoning"], help="Sample types to generate")
@click.option("--max-samples", default=None, type=int, help="Maximum samples to generate")
@click.option("--min-quality", default=0.5, type=float, help="Minimum quality score")
@click.option("--distill", is_flag=True, help="Distill mode: generate from templates without LLM teacher")
@click.option("--system-prompt", default=None, help="System prompt to prepend to each sample")
def knowledge(sources, output, teacher, model, sample_types, max_samples, min_quality, distill, system_prompt):
    """Teach knowledge from documents.

    SOURCES: Source files or directories to process (Markdown, TXT, PDF, HTML, CSV, JSON, FlatSeek Index)

    Example:
        flattune teach knowledge ./docs/*.md -o dataset.jsonl

    Distill mode (no LLM needed):
        flattune teach knowledge ./docs/*.md --distill -o dataset.jsonl
    """
    setup_logging("teach")
    logger.info(f"Teaching knowledge from {len(sources)} sources (distill={distill})")

    # Create pipeline config
    config = PipelineConfig(
        teacher=teacher,
        teacher_model=model,
        sample_types=list(sample_types),
        max_samples=max_samples,
        min_quality_score=min_quality,
        system_prompt=system_prompt,
    )

    # Create pipeline
    pipeline = TeachPipeline(config, distill_mode=distill)

    # Run pipeline
    click.echo(f"[teach] Processing {len(sources)} sources...")
    click.echo(f"[teach] Teacher: {teacher}/{model}")
    click.echo(f"[teach] Sample types: {', '.join(sample_types)}")

    output_path = Path(output)
    count = 0

    try:
        for _sample in pipeline.run(list(sources), output_path):
            count += 1
            if count % 100 == 0:
                click.echo(f"[teach] Generated {count} samples...")

        click.echo(f"[teach] Complete! Generated {count} samples.")
        click.echo(f"[teach] Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"[teach] Error: {e}")
        logger.error(f"Teach knowledge failed: {e}")
        raise


@teach.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--output", "-o", default="software_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default=None, help="Teacher model. Omit for distill mode.")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
@click.option("--sample-types", "-s", multiple=True, default=["tool_call", "intent", "slot_fill", "validation"], help="Sample types")
@click.option("--distill", is_flag=True, help="Distill mode: generate from templates without LLM teacher")
@click.option("--system-prompt", default=None, help="System prompt to prepend to each sample")
@click.option("--max-samples", default=None, type=int, help="Maximum number of samples to generate")
def software(sources, output, teacher, model, sample_types, distill, system_prompt, max_samples):
    """Teach software APIs and tools.

    SOURCES: OpenAPI, Swagger, GraphQL, MCP, CLI docs, SDK docs

    Example:
        flattune teach software openapi.json --distill -o api_dataset.jsonl
    """
    setup_logging("teach")
    logger.info(f"Teaching software from {len(sources)} sources (distill={distill})")

    config = PipelineConfig(
        teacher=teacher,
        teacher_model=model,
        sample_types=list(sample_types),
        system_prompt=system_prompt,
        max_samples=max_samples,
    )

    pipeline = TeachPipeline(config, distill_mode=distill)

    click.echo(f"[teach] Processing {len(sources)} API/spec files...")
    click.echo(f"[teach] Sample types: {', '.join(sample_types)}")

    output_path = Path(output)
    count = 0

    try:
        for _sample in pipeline.run(list(sources), output_path):
            count += 1
            if count % 100 == 0:
                click.echo(f"[teach] Generated {count} samples...")

        click.echo(f"[teach] Complete! Generated {count} samples.")
        click.echo(f"[teach] Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"[teach] Error: {e}")
        logger.error(f"Teach software failed: {e}")
        raise


@teach.command()
@click.argument("schema", type=click.Path(exists=True))
@click.option("--output", "-o", default="database_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default=None, help="Teacher model. Omit for distill mode.")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
@click.option("--distill", is_flag=True, help="Distill mode: generate from templates without LLM teacher")
def database(schema, output, teacher, model, distill):
    """Teach database schemas.

    SCHEMA: Database schema file (PostgreSQL, MySQL, SQLite, SQLAlchemy)

    Example:
        flattune teach database schema.sql --distill -o sql_dataset.jsonl
    """
    setup_logging("teach")

    config = PipelineConfig(
        teacher=teacher,
        teacher_model=model,
        sample_types=["nl_to_sql", "sql"],
    )

    pipeline = TeachPipeline(config, distill_mode=distill)

    click.echo(f"[teach] Processing database schema: {schema} (distill={distill})")
    click.echo("[teach] Sample types: nl_to_sql, sql")

    output_path = Path(output)
    count = 0

    try:
        for _sample in pipeline.run([schema], output_path):
            count += 1
            if count % 100 == 0:
                click.echo(f"[teach] Generated {count} samples...")

        click.echo(f"[teach] Complete! Generated {count} samples.")
        click.echo(f"[teach] Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"[teach] Error: {e}")
        logger.error(f"Teach database failed: {e}")
        raise


@teach.command()
@click.argument("spec", type=click.Path(exists=True))
@click.option("--output", "-o", default="openapi_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default=None, help="Teacher model. Omit for distill mode.")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
@click.option("--distill", is_flag=True, help="Distill mode: generate from templates without LLM teacher")
def openapi(spec, output, teacher, model, distill):
    """Teach OpenAPI/Swagger specifications.

    SPEC: OpenAPI or Swagger JSON/YAML file

    Example:
        flattune teach openapi api.json --distill -o api_dataset.jsonl
    """
    setup_logging("teach")

    config = PipelineConfig(
        teacher=teacher,
        teacher_model=model,
        sample_types=["tool_call", "intent", "slot_fill", "validation", "error_recovery", "multi_turn"],
    )

    pipeline = TeachPipeline(config, distill_mode=distill)

    click.echo(f"[teach] Processing OpenAPI spec: {spec} (distill={distill})")
    click.echo("[teach] Generating diverse API interaction examples...")

    output_path = Path(output)
    count = 0

    try:
        for _sample in pipeline.run([spec], output_path):
            count += 1
            if count % 100 == 0:
                click.echo(f"[teach] Generated {count} samples...")

        click.echo(f"[teach] Complete! Generated {count} samples.")
        click.echo(f"[teach] Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"[teach] Error: {e}")
        logger.error(f"Teach openapi failed: {e}")
        raise


@teach.command()
@click.argument("server", type=click.Path(exists=True))
@click.option("--output", "-o", default="mcp_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default=None, help="Teacher model. Omit for distill mode.")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
@click.option("--distill", is_flag=True, help="Distill mode: generate from templates without LLM teacher")
def mcp(server, output, teacher, model, distill):
    """Teach MCP (Model Context Protocol) servers.

    SERVER: MCP server JSON definition file

    Example:
        flattune teach mcp server.json --distill -o mcp_dataset.jsonl
    """
    setup_logging("teach")

    config = PipelineConfig(
        teacher=teacher,
        teacher_model=model,
        sample_types=["tool_call", "intent", "slot_fill", "workflow"],
    )

    pipeline = TeachPipeline(config, distill_mode=distill)

    click.echo(f"[teach] Processing MCP server: {server} (distill={distill})")
    click.echo("[teach] Generating tool workflow examples...")

    output_path = Path(output)
    count = 0

    try:
        for _sample in pipeline.run([server], output_path):
            count += 1
            if count % 100 == 0:
                click.echo(f"[teach] Generated {count} samples...")

        click.echo(f"[teach] Complete! Generated {count} samples.")
        click.echo(f"[teach] Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"[teach] Error: {e}")
        logger.error(f"Teach mcp failed: {e}")
        raise


@teach.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", "-o", default="cli_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default="openai", help="Teacher model")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
def cli(source, output, teacher, model):
    """Teach CLI commands.

    SOURCE: Executable, --help output, man page, or Markdown docs

    Example:
        flattune teach cli ffmpeg --help -o cli_dataset.jsonl
    """
    setup_logging("teach")

    click.echo("[teach] CLI teaching not yet implemented")
    click.echo("[teach] Coming soon: Support for --help, man pages, and CLI docs")


@teach.command()
@click.argument("package", required=True)
@click.option("--output", "-o", default="python_dataset.jsonl", help="Output file path")
@click.option("--teacher", "-t", default="openai", help="Teacher model")
@click.option("--model", "-m", default="gpt-4o-mini", help="Teacher model name")
def python(package, output, teacher, model):
    """Teach Python library/API.

    PACKAGE: Python package name or path

    Example:
        flattune teach python pandas -o pandas_dataset.jsonl
    """
    setup_logging("teach")

    click.echo("[teach] Python library teaching not yet implemented")
    click.echo("[teach] Coming soon: Support for package introspection and docstring parsing")


@teach.command()
def list_parsers():
    """List all available parsers."""
    parsers = ParserRegistry.list_plugins()
    click.echo("[teach] Available parsers:")
    for p in parsers:
        click.echo(f"  - {p}")


@teach.command()
def list_teachers():
    """List all available teacher models."""
    teachers = TeacherRegistry.list_plugins()
    local = TeacherRegistry.list_local_teachers()
    TeacherRegistry.list_remote_teachers()

    click.echo("[teach] Available teachers:")
    for t in teachers:
        kind = "local" if t in local else "remote"
        click.echo(f"  - {t} ({kind})")


@teach.command()
def list_generators():
    """List all available generators."""
    generators = GeneratorRegistry.list_plugins()
    click.echo("[teach] Available generators:")
    for g in generators:
        click.echo(f"  - {g}")
