"""Build pipeline orchestrator for the build system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

from flattune.build.registry import (
    DatasetTypeRegistry,
    SourceDetectionResult,
)
from flattune.build.source import SourceDetector
from flattune.build.analyzer import ContentAnalyzer, AnalysisResult
from flattune.build.planner import BuildPlanner, BuildPlan, TypeSelection
from flattune.build.approval import InteractiveApproval, ApprovalResult
from flattune.build.exporter import BuildExporter
from flattune.dataset.generators import get_generator
from flattune.dataset.split import DatasetSplitter
from flattune.teach.registry import SourceType

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    sources_detected: int = 0
    documents_processed: int = 0
    samples_generated: int = 0
    samples_filtered: int = 0
    types_generated: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BuildPipeline:
    """Main orchestrator for the build pipeline.

    Pipeline stages:
    1. Source Detection
    2. Content Analysis
    3. Build Planning
    4. User Approval
    5. Dataset Generation
    6. Validation
    7. Export
    """

    def __init__(
        self,
        source: Union[str, Path],
        config: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        interactive: bool = True,
        yes_flag: bool = False,
    ):
        """Initialize build pipeline.

        Args:
            source: Source path or URL.
            config: Dataset configuration.
            output_dir: Output directory.
            interactive: Enable interactive mode.
            yes_flag: Skip interactive approval.
        """
        self.source = str(source)
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("build")
        self.interactive = interactive and not yes_flag

        # Components
        self.source_detector = SourceDetector()
        self.analyzer = ContentAnalyzer()
        self.planner = BuildPlanner()
        self.approval = InteractiveApproval(self.planner)
        self.exporter = BuildExporter()

        # Splitter for train/val/test
        self.splitter = DatasetSplitter()

        # State
        self.detection_result: Optional[SourceDetectionResult] = None
        self.analysis_result: Optional[AnalysisResult] = None
        self.build_plan: Optional[BuildPlan] = None
        self.approval_result: Optional[ApprovalResult] = None
        self.stats = PipelineStats()

        # Read sample content lazily
        self._sample_content: Optional[str] = None

    def run(
        self,
        documents: Iterator[dict[str, Any]],
        user_requested_types: Optional[list[str]] = None,
        force_types: Optional[list[str]] = None,
    ) -> dict[str, Path]:
        """Run the complete pipeline.

        Args:
            documents: Iterator of documents to process.
            user_requested_types: Types specified in config.
            force_types: Types specified via CLI --types.

        Returns:
            Dictionary mapping output paths.
        """
        try:
            # Stage 1: Source Detection
            self._stage_source_detection()

            # Stage 2: Content Analysis
            self._stage_content_analysis()

            # Get document count (we need to consume the iterator once)
            # For now, estimate from analysis
            doc_count = self._estimate_document_count()

            # Stage 3: Build Planning
            self._stage_build_planning(doc_count, user_requested_types, force_types)

            # Stage 4: User Approval
            self._stage_user_approval()

            if not self.approval_result or not self.approval_result.approved:
                raise RuntimeError("Build cancelled by user")

            # Use modified plan if user changed selection
            final_plan = self.approval_result.modified_plan or self.build_plan
            selected_types = self.approval_result.selected_types

            # Stage 5-7: Generate, Validate, Export
            return self._generate_and_export(documents, final_plan, selected_types)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats.errors.append(str(e))
            raise

    def _stage_source_detection(self) -> None:
        """Stage 1: Detect source type."""
        logger.info(f"Detecting source: {self.source}")
        self.detection_result = self.source_detector.detect(self.source)
        logger.info(f"Detected source type: {self.detection_result.source_type.value}")
        self.stats.sources_detected = 1

    def _stage_content_analysis(self) -> None:
        """Stage 2: Analyze content."""
        sample = self._read_sample_content()
        logger.info("Analyzing content...")
        self.analysis_result = self.analyzer.analyze(self.detection_result, sample)
        logger.info(f"Analysis complete: {len(self.analysis_result.suggested_types)} types suggested")

    def _stage_build_planning(
        self,
        doc_count: int,
        user_requested_types: Optional[list[str]] = None,
        force_types: Optional[list[str]] = None,
    ) -> None:
        """Stage 3: Create build plan."""
        logger.info("Creating build plan...")
        self.build_plan = self.planner.create_plan(
            self.detection_result,
            self.analysis_result,
            doc_count,
            user_requested_types=user_requested_types,
            force_types=force_types,
        )
        logger.info(f"Plan created with {len(self.build_plan.selected_types)} types")

    def _stage_user_approval(self) -> None:
        """Stage 4: Request user approval."""
        if self.build_plan is None:
            raise RuntimeError("No build plan available")

        logger.info("Requesting user approval...")
        self.approval_result = self.approval.request_approval(
            self.build_plan,
            force_interactive=self.interactive,
        )

        if self.approval_result.approved:
            logger.info(f"User approved {len(self.approval_result.selected_types)} types")
        else:
            logger.info("User cancelled build")

    def _generate_and_export(
        self,
        documents: Iterator[dict[str, Any]],
        plan: BuildPlan,
        selected_types: list[str],
    ) -> dict[str, Path]:
        """Stages 5-7: Generate samples, validate, and export.

        Args:
            documents: Document iterator.
            plan: Approved build plan.
            selected_types: User-selected type names.

        Returns:
            Dictionary of output paths.
        """
        logger.info("Generating datasets...")

        # Initialize generators for selected types
        generators: dict[str, Any] = {}
        for type_name in selected_types:
            dt = DatasetTypeRegistry.get(type_name)
            if dt:
                try:
                    generators[type_name] = get_generator(dt.generator_name)
                except ValueError:
                    logger.warning(f"Generator not found: {dt.generator_name}")
                    self.stats.errors.append(f"Generator not found: {dt.generator_name}")

        # Generate samples by type
        samples_by_type: dict[str, list[dict]] = {t: [] for t in selected_types}

        for doc in documents:
            self.stats.documents_processed += 1

            for type_name in selected_types:
                gen = generators.get(type_name)
                if gen is None:
                    continue

                try:
                    # Determine instruction template
                    dt = DatasetTypeRegistry.get(type_name)
                    instruction = dt.instruction_template if dt else ""

                    # Generate with instruction
                    generated = gen.generate(doc, self.config, instruction=instruction)
                    samples_by_type[type_name].extend(generated)
                    self.stats.samples_generated += len(generated)
                except Exception as e:
                    logger.debug(f"Generation error for {type_name}: {e}")
                    self.stats.errors.append(f"Error generating {type_name}: {e}")

        # Update types generated
        self.stats.types_generated = [
            t for t in selected_types if samples_by_type.get(t)
        ]

        # Export to structured output
        logger.info("Exporting datasets...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = self.exporter.export(
            samples_by_type=samples_by_type,
            output_dir=self.output_dir,
            plan=plan,
        )

        logger.info(f"Build complete: {self.stats.samples_generated} samples generated")
        return output_paths

    def _read_sample_content(self) -> str:
        """Read sample content for analysis."""
        if self._sample_content is not None:
            return self._sample_content

        self._sample_content = ""

        try:
            path = Path(self.source)
            if path.exists() and path.is_file():
                # Check if it's a text file we can read
                if path.suffix.lower() in [".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv"]:
                    with open(path, "r", encoding="utf-8") as f:
                        self._sample_content = f.read(4096)
        except Exception as e:
            logger.debug(f"Could not read sample content: {e}")

        return self._sample_content

    def _estimate_document_count(self) -> int:
        """Estimate document count from analysis."""
        if self.analysis_result is None:
            return 1

        # Rough estimate based on content length
        chars = self.analysis_result.content_statistics.get("total_characters", 0)
        if chars > 10000:
            return 10
        elif chars > 1000:
            return 5
        elif chars > 100:
            return 2
        return 1

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self.stats
