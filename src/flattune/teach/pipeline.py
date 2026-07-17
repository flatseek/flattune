"""Unified Teach Pipeline.

Orchestrates the complete teaching workflow:
Source → Parser → Knowledge Graph → Teacher → Generator → Quality → Export
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flattune.teach.knowledge_graph import KnowledgeGraph, KnowledgeNode
from flattune.teach.quality.deduplication import Deduplicator
from flattune.teach.quality.scoring import QualityScorer
from flattune.teach.registry import (
    BaseGenerator,
    BaseParser,
    BaseTeacher,
    GeneratorRegistry,
    ParserRegistry,
    SourceType,
    TeacherRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the teach pipeline."""
    # Parser settings
    parser: str = "auto"  # "auto" or specific parser name
    source_type: SourceType | None = None

    # Teacher settings
    teacher: str = "openai"
    teacher_model: str = "gpt-4o-mini"
    teacher_temperature: float = 0.7
    teacher_max_tokens: int = 2048

    # Generator settings
    generator: str = "auto"
    sample_types: list[str] = field(default_factory=list)  # Empty = all supported

    # Quality settings
    enable_deduplication: bool = True
    enable_quality_scoring: bool = True
    min_quality_score: float = 0.5

    # Output settings
    output_format: str = "jsonl"  # jsonl, sharegpt, alpaca
    output_dir: Path | None = None
    # System prompt for RAG inference — generates a single .prompt file
    # containing a formatted multiline prompt (not injected into jsonl samples).
    # Usage: loaded at inference time and POST-inserted with the user query.
    system_prompt: str | None = None

    # Batch settings
    batch_size: int = 10
    max_samples: int | None = None


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    sources_processed: int = 0
    nodes_created: int = 0
    samples_generated: int = 0
    samples_filtered: int = 0
    samples_exported: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else None
            ),
            "sources_processed": self.sources_processed,
            "nodes_created": self.nodes_created,
            "samples_generated": self.samples_generated,
            "samples_filtered": self.samples_filtered,
            "samples_exported": self.samples_exported,
            "errors": self.errors,
        }


class TeachPipeline:
    """Unified pipeline for compiling knowledge into training datasets.

    The pipeline follows these stages:
    1. Parse - Read and normalize source data
    2. Graph - Build knowledge graph representation
    3. Generate - Create training samples using generators
    4. Quality - Evaluate and filter samples
    5. Export - Write to target format
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        teacher: BaseTeacher | None = None,
        generator: BaseGenerator | None = None,
        distill_mode: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
            teacher: Optional pre-configured teacher (overrides config).
            generator: Optional pre-configured generator (overrides config).
            distill_mode: If True, use template/rule-based generation without LLM teacher.
        """
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        self.distill_mode = distill_mode

        # Initialize teacher (only if not distill mode)
        if teacher:
            self.teacher = teacher
        elif distill_mode:
            self.teacher = None
            logger.info("Distill mode enabled - using template-based generation without LLM")
        else:
            self.teacher = self._create_teacher()

        # Initialize generator
        if generator:
            self.generator = generator
        else:
            self.generator = self._create_generator()

        # Quality components
        self.deduplicator = Deduplicator() if self.config.enable_deduplication else None
        self.quality_scorer = QualityScorer() if self.config.enable_quality_scoring else None

    def _create_teacher(self) -> BaseTeacher:
        """Create teacher from config."""
        teacher_config = self.config.teacher

        if teacher_config == "auto":
            # Try to detect best available teacher
            if TeacherRegistry.get("openai"):
                teacher_config = "openai"
            elif TeacherRegistry.get("anthropic"):
                teacher_config = "anthropic"
            elif TeacherRegistry.get("ollama"):
                teacher_config = "ollama"
            else:
                raise ValueError("No teacher available. Install openai, anthropic, or ollama.")

        try:
            teacher = TeacherRegistry.create(
                teacher_config,
                model=self.config.teacher_model,
                temperature=self.config.teacher_temperature,
                max_tokens=self.config.teacher_max_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to create teacher '{teacher_config}': {e}. Using template-based generation.")
            teacher = None

        return teacher

    def _create_generator(self) -> BaseGenerator:
        """Create generator from config."""
        gen_config = self.config.generator

        if gen_config == "auto":
            # Use default generator based on sample types
            if not self.config.sample_types:
                gen_config = "multi"  # Multi-generator for diverse outputs
            else:
                gen_config = self.config.sample_types[0]

        return GeneratorRegistry.create(gen_config, teacher=self.teacher)

    def run(
        self,
        sources: list[str | Path],
        output_path: Path | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the complete pipeline.

        Args:
            sources: List of source paths/URLs to process.
            output_path: Optional path for output file.

        Yields:
            Generated sample dictionaries.
        """
        output_file = None
        prompt_path = None

        if output_path:
            output_file = open(output_path, "w")
            # Derive .prompt path from output file path
            prompt_path = Path(output_path).with_suffix(".prompt")

        # Auto-generate system prompt from source if available
        if self.config.system_prompt is None and sources:
            self.config.system_prompt = self._generate_system_prompt(sources)

        # Write system prompt file for RAG usage (single bulk .prompt text file)
        if self.config.system_prompt and prompt_path:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(self.config.system_prompt)
            logger.info(f"System prompt written to {prompt_path}")

        try:
            for source in sources:
                yield from self._process_source(source, output_file)

        finally:
            if output_file:
                output_file.close()

        self.stats.end_time = datetime.now()

    def _process_source(
        self,
        source: str | Path,
        output_file: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Process a single source.

        Args:
            source: Source to process.
            output_file: Optional output file handle.

        Yields:
            Generated samples.
        """
        source_str = str(source)
        logger.info(f"Processing source: {source_str}")

        # Find appropriate parser
        parser = self._get_parser(source_str)
        if parser is None:
            logger.error(f"No parser found for: {source_str}")
            self.stats.errors.append(f"No parser for: {source_str}")
            return

        # Parse source into knowledge graph
        graph = KnowledgeGraph()
        graph.metadata["source"] = source_str
        graph.metadata["parser"] = parser.__class__.__name__

        for parse_result in parser.parse(source_str):
            for node in self._parse_result_to_nodes(parse_result):
                graph.add_node(node)

        self.stats.sources_processed += 1
        self.stats.nodes_created += len(graph)

        logger.info(f"Created {len(graph)} nodes from {source_str}")

        # Generate samples from graph
        for sample in self._generate_from_graph(graph):
            # Quality filtering
            if self._passes_quality(sample):
                self.stats.samples_exported += 1

                output_record = self._format_output(sample)
                if output_file:
                    output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

                yield output_record
            else:
                self.stats.samples_filtered += 1

            # Check max samples limit
            if self.config.max_samples and self.stats.samples_exported >= self.config.max_samples:
                logger.info(f"Reached max_samples limit: {self.config.max_samples}")
                return

    def _get_parser(self, source: str) -> BaseParser | None:
        """Get appropriate parser for source.

        Args:
            source: Source path/URL.

        Returns:
            Parser instance or None.
        """
        parser_name = self.config.parser

        if parser_name == "auto":
            parser_class = ParserRegistry.get_for_source(source)
            if parser_class:
                return parser_class()
            return None

        parser_class = ParserRegistry.get(parser_name)
        if parser_class:
            return parser_class()

        return None

    def _parse_result_to_nodes(self, parse_result) -> Iterator[KnowledgeNode]:
        """Convert parse result to knowledge nodes.

        Args:
            parse_result: Result from parser.

        Yields:
            KnowledgeNode objects.
        """
        from flattune.teach.knowledge_graph import KnowledgeNode as KN
        from flattune.teach.knowledge_graph import NodeType

        # Create main content node
        node = KN(
            node_type=NodeType.DOCUMENT,
            content=parse_result.content,
            source=parse_result.source,
            metadata=parse_result.metadata,
        )

        # Add entities as separate nodes
        for entity in parse_result.entities:
            entity_node = KN(
                node_type=NodeType.CONCEPT,
                content=entity.get("content", str(entity)),
                source=parse_result.source,
                metadata=entity,
            )
            yield entity_node

        yield node

    def _generate_from_graph(self, graph: KnowledgeGraph) -> Iterator[dict[str, Any]]:
        """Generate samples from knowledge graph.

        Args:
            graph: Knowledge graph to generate from.

        Yields:
            Sample dictionaries.
        """
        sample_types = self.config.sample_types or ["qa", "summarization", "extraction"]

        for node in graph.iterate():
            for sample_type in sample_types:
                try:
                    for sample in self.generator.generate(node, sample_type=sample_type):
                        self.stats.samples_generated += 1

                        yield {
                            "messages": sample.conversation,
                            "sample_type": sample.sample_type,
                            "source": sample.source,
                            "quality_score": sample.quality_score,
                            "metadata": sample.metadata,
                        }
                except Exception as e:
                    logger.warning(f"Generation error for {node.id}: {e}")
                    self.stats.errors.append(f"Generation error: {e}")

    def _generate_system_prompt(self, sources: list[str | Path]) -> str:
        """Auto-generate a system prompt from source files (e.g., OpenAPI spec).

        Generates a detailed multiline prompt with per-endpoint parameter info,
        grouped by resource. Optimized for RAG: model can map user queries like
        "get active trackers" -> correct endpoint + params.

        Args:
            sources: Source file paths.

        Returns:
            Multiline system prompt string.
        """
        if not sources:
            return ""

        source_str = str(sources[0])
        path = Path(source_str)

        if path.suffix.lower() not in (".json", ".yaml", ".yml"):
            return ""

        try:
            import json as _json
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                spec = _json.loads(content)
            else:
                try:
                    import yaml
                    spec = yaml.safe_load(content)
                except Exception:
                    return ""
        except Exception:
            return ""

        if not isinstance(spec, dict):
            return ""

        openapi_version = str(spec.get("openapi", ""))
        if not openapi_version.startswith("3."):
            return ""

        info = spec.get("info", {})
        title = info.get("title", "API")
        version = info.get("version", "")
        servers = spec.get("servers", [])
        base_url = servers[0].get("url", "") if servers else ""

        lines = [
            f"You are an expert API assistant for {title} (v{version}).",
            f"Base URL: {base_url}",
            "",
            "Your task: map user requests to the correct API call. "
            "Extract parameters from the user's query. "
            "Respond with ONLY the API call in this format:",
            "  METHOD /path",
            "  Header: Authorization: Bearer <token>",
            "  Body/Query params as needed.",
            "",
            "Available endpoints:",
        ]

        # Group endpoints by first path segment (resource)
        paths = spec.get("paths", {})
        resources: dict[str, list] = {}

        for path_str, path_item in sorted(paths.items()):
            # Determine resource group
            segments = [s for s in path_str.strip("/").split("/") if s]
            resource = segments[0] if segments else "misc"
            if resource not in resources:
                resources[resource] = []
            resources[resource].append((path_str, path_item))

        # Emit grouped endpoints
        for resource, endpoints in sorted(resources.items()):
            lines.append(f"\n=== {resource.upper()} ===")
            for path_str, path_item in endpoints:
                for method, operation in sorted(path_item.items()):
                    if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
                        continue
                    if not isinstance(operation, dict):
                        continue

                    summary = operation.get("summary", "")
                    desc_short = ""
                    if summary:
                        desc_short = f" — {summary}"
                    lines.append(f"  {method.upper():6} {path_str}{desc_short}")

                    # Parameters
                    for p in operation.get("parameters", []):
                        pname = p.get("name", "")
                        pin = p.get("in", "query")
                        ptype = p.get("schema", {}).get("type", "string") if p.get("schema") else "string"
                        preq = "required" if p.get("required") else "optional"
                        pdesc = p.get("description", "")
                        if pdesc:
                            lines.append(f"    {pin}: {pname} ({ptype}, {preq}) — {pdesc}")
                        else:
                            lines.append(f"    {pin}: {pname} ({ptype}, {preq})")

                    # Request body
                    rb = operation.get("requestBody", {})
                    if rb:
                        content = rb.get("content", {})
                        for _mt, media in content.items():
                            schema = media.get("schema", {})
                            props = schema.get("properties", {})
                            if props:
                                required = schema.get("required", [])
                                for prop_name, prop in props.items():
                                    ptype = prop.get("type", "any")
                                    preq = "required" if prop_name in required else "optional"
                                    pdesc = prop.get("description", "")
                                    if pdesc:
                                        lines.append(f"    body: {prop_name} ({ptype}, {preq}) — {pdesc}")
                                    else:
                                        lines.append(f"    body: {prop_name} ({ptype}, {preq})")
                            elif schema.get("$ref"):
                                lines.append(f"    body: {schema.get('$ref')} (object)")

        lines.append("")
        lines.append("Guidelines:")
        lines.append("- Match the user's intent to the closest endpoint.")
        lines.append("- Extract all relevant parameters from the user's request.")
        lines.append("- Use query params for filters, path params for IDs.")
        lines.append("- For POST/PUT, include a reasonable request body.")
        lines.append("- Always include Authorization header for protected endpoints.")

        return "\n".join(lines)

    def _format_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format sample for output.

        System prompt is NOT injected per-sample when using bulk mode
        (written once to a separate file instead).

        Args:
            sample: Sample dictionary from generation.

        Returns:
            Formatted output record.
        """
        return sample

    def _passes_quality(self, sample: dict[str, Any]) -> bool:
        """Check if sample passes quality filters.

        Args:
            sample: Sample to check.

        Returns:
            True if sample passes quality checks.
        """
        if not self.config.enable_quality_scoring:
            return True

        score = sample.get("quality_score", 1.0)

        if score < self.config.min_quality_score:
            return False

        return True

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics.

        Returns:
            Pipeline statistics.
        """
        return self.stats


# ============================================================================
# Convenience Functions
# ============================================================================


def teach_knowledge(
    sources: list[str],
    output_path: str,
    **kwargs,
) -> PipelineStats:
    """Teach knowledge from documents.

    Args:
        sources: Source paths to process.
        output_path: Output file path.
        **kwargs: Additional pipeline config options.

    Returns:
        Pipeline statistics.
    """
    config = PipelineConfig(
        sample_types=["qa", "summarization", "extraction", "comparison", "reasoning"],
        output_dir=Path(output_path).parent,
        **kwargs,
    )

    pipeline = TeachPipeline(config)

    # Run pipeline
    list(pipeline.run(sources, Path(output_path)))

    return pipeline.get_stats()


def teach_software(
    sources: list[str],
    output_path: str,
    sample_types: list[str] | None = None,
    **kwargs,
) -> PipelineStats:
    """Teach software APIs and tools.

    Args:
        sources: Source paths (OpenAPI, MCP, etc).
        output_path: Output file path.
        sample_types: Types of samples to generate.
        **kwargs: Additional pipeline config options.

    Returns:
        Pipeline statistics.
    """
    if sample_types is None:
        sample_types = ["tool_call", "intent", "slot_fill", "validation", "error_recovery"]

    config = PipelineConfig(
        sample_types=sample_types,
        output_dir=Path(output_path).parent,
        **kwargs,
    )

    pipeline = TeachPipeline(config)

    # Run pipeline
    list(pipeline.run(sources, Path(output_path)))

    return pipeline.get_stats()


def teach_database(
    schema_path: str,
    output_path: str,
    **kwargs,
) -> PipelineStats:
    """Teach database schemas.

    Args:
        schema_path: Path to database schema file.
        output_path: Output file path.
        **kwargs: Additional pipeline config options.

    Returns:
        Pipeline statistics.
    """
    config = PipelineConfig(
        sample_types=["nl_to_sql", "sql_to_result"],
        output_dir=Path(output_path).parent,
        **kwargs,
    )

    pipeline = TeachPipeline(config)

    # Run pipeline
    list(pipeline.run([schema_path], Path(output_path)))

    return pipeline.get_stats()
