"""Dataset builder for generating training data from FlatSeek indexes."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flattune.config import DatasetConfig, DatasetType, FlatseekConfig
from flattune.dataset.generators import get_generator
from flattune.dataset.split import DatasetSplitter
from flattune.flatseek.auto import create_provider
from flattune.flatseek.provider import FlatseekProvider
from flattune.utils.logging import get_logger

logger = get_logger(__name__)


# Legacy type to new type mapping for backward compatibility
LEGACY_TYPE_MAP = {
    DatasetType.INSTRUCTION: ["procedure", "howto"],
    DatasetType.CHAT: ["chat", "dialogue"],
    DatasetType.QA: ["context_qa", "direct_qa"],
    DatasetType.CLASSIFICATION: ["topic_classify", "entity_classify"],
    DatasetType.SUMMARY: ["executive_summary", "concise_summary"],
    DatasetType.EXTRACTION: ["key_value_extract"],
    DatasetType.RAG: ["context_qa", "citation_qa"],
}


class DatasetBuilder:
    """Builds training datasets from FlatSeek indexes.

    Supports multiple dataset types:
    - instruction_tuning: Standard instruction-response pairs
    - chat: Multi-turn conversation format
    - qa: Question-answer format
    - classification: Text classification
    - summarization: Long text -> summary
    - extraction: Information extraction
    - rag_distillation: Generate QA from context
    - synthetic_qa: LLM-generated questions
    - conversation: Multi-turn dialogue
    - evaluation: Benchmark format

    Now supports the new intelligent build pipeline with:
    - Source auto-detection
    - Automatic dataset type suggestion
    - Interactive approval
    - --yes flag for non-interactive mode
    """

    # Mapping from dataset type to generator names
    DATASET_TYPE_GENERATORS = {
        DatasetType.INSTRUCTION: ["custom"],
        DatasetType.CHAT: ["conversation"],
        DatasetType.QA: ["qa"],
        DatasetType.CLASSIFICATION: ["classification"],
        DatasetType.SUMMARY: ["summary"],
        DatasetType.EXTRACTION: ["extraction"],
        DatasetType.RAG: ["qa", "extraction"],
    }

    def __init__(
        self,
        flatseek_config: FlatseekConfig | None = None,
        dataset_config: DatasetConfig | None = None,
        output_dir: str | Path | None = None,
        interactive: bool = True,
        yes_flag: bool = False,
    ):
        """Initialize the dataset builder.

        Args:
            flatseek_config: Configuration for FlatSeek index access.
            dataset_config: Configuration for dataset generation.
            output_dir: Directory to save generated datasets.
            interactive: Enable interactive mode for build planning.
            yes_flag: Skip interactive approval (non-interactive mode).
        """
        self.flatseek_config = flatseek_config or FlatseekConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive and not yes_flag
        self.yes_flag = yes_flag

        self.provider: FlatseekProvider | None = None

        # Initialize generators based on config
        self.generators = self._initialize_generators()

        # New build pipeline components (lazy init)
        self._build_pipeline = None
        self._build_plan = None

    def _initialize_generators(self) -> list:
        """Initialize generators based on dataset config.

        Returns:
            List of initialized generator instances.
        """
        generators = []

        # Get generators from config or infer from dataset type
        gen_names = self.dataset_config.generators
        if not gen_names:
            gen_names = self.DATASET_TYPE_GENERATORS.get(self.dataset_config.type, ["custom"])

        for gen_name in gen_names:
            gen_type = self._get_generator_type(gen_name)
            try:
                generator = get_generator(gen_type)
                generators.append(generator)
            except ValueError:
                logger.warning(f"Unknown generator: {gen_name}, skipping")

        return generators

    def _get_generator_type(self, name: str) -> str:
        """Map generator name to actual generator type.

        Args:
            name: Generator name from config.

        Returns:
            Generator type string for get_generator().
        """
        # Direct mappings
        type_map = {
            "qa": "qa",
            "summary": "summary",
            "classification": "classification",
            "comparison": "comparison",
            "conversation": "conversation",
            "extraction": "extraction",
            "custom": "custom",
            "safety": "safety",
            "facts": "facts",
            "glossary": "glossary",
            "concept": "concept",
            "procedure": "procedure",
        }
        return type_map.get(name.lower(), "custom")

    def connect(self) -> DatasetBuilder:
        """Connect to the FlatSeek index.

        Returns:
            Self for method chaining.
        """
        if self.flatseek_config.path:
            self.provider = create_provider(
                path=self.flatseek_config.path,
                mode=self.flatseek_config.mode,
                source=self.flatseek_config.source,
                query=self.flatseek_config.query,
                encryption_key=self.flatseek_config.encryption_key,
            )
        return self

    def build(
        self,
        documents: Iterator[dict[str, Any]] | None = None,
        dataset_name: str = "dataset",
        query: str | None = None,
        source_path: str | None = None,
        user_requested_types: list[str] | None = None,
    ) -> dict[str, Path]:
        """Build a dataset from documents or FlatSeek index.

        Args:
            documents: Optional iterator of documents. If not provided,
                     queries the FlatSeek index.
            dataset_name: Name for the dataset.
            query: Query string for FlatSeek export (used if documents not provided).
            source_path: Optional source path for build planning.
            user_requested_types: Optional explicit types to generate.

        Returns:
            Dictionary with paths to all generated files:
            - dataset.jsonl: All samples in JSONL format
            - train.jsonl: Training samples
            - val.jsonl: Validation samples
            - test.jsonl: Test samples (if test_split > 0)
            - metadata.json: Dataset metadata
            - statistics.json: Dataset statistics
        """
        logger.info(f"Building {self.dataset_config.type.value} dataset: {dataset_name}")

        # Check if we should use the new build pipeline
        if source_path and not self.yes_flag:
            try:
                return self._build_with_pipeline(
                    documents=documents,
                    dataset_name=dataset_name,
                    source_path=source_path,
                    user_requested_types=user_requested_types,
                )
            except Exception as e:
                logger.warning(f"Build pipeline failed: {e}, falling back to legacy mode")
                # Fall through to legacy build

        # Legacy build mode
        return self._build_legacy(
            documents=documents,
            dataset_name=dataset_name,
            query=query,
        )

    def _build_with_pipeline(
        self,
        documents: Iterator[dict[str, Any]],
        dataset_name: str,
        source_path: str,
        user_requested_types: list[str] | None = None,
    ) -> dict[str, Path]:
        """Build using the new intelligent pipeline.

        Args:
            documents: Document iterator.
            dataset_name: Name for dataset.
            source_path: Source path for analysis.
            user_requested_types: Optional explicit types.

        Returns:
            Dictionary of output paths.
        """
        from flattune.build.pipeline import BuildPipeline

        # Create build pipeline
        pipeline = BuildPipeline(
            source=source_path,
            config=self.dataset_config,
            output_dir=self.output_dir,
            interactive=self.interactive,
            yes_flag=self.yes_flag,
        )

        # Run pipeline
        output_paths = pipeline.run(
            documents=documents,
            user_requested_types=user_requested_types or self.dataset_config.generators,
        )

        self._build_plan = pipeline.build_plan
        self._build_pipeline = pipeline

        # Convert paths to Path objects
        result_paths = {}
        for key, path in output_paths.items():
            result_paths[key] = Path(path)

        return result_paths

    def _build_legacy(
        self,
        documents: Iterator[dict[str, Any]] | None,
        dataset_name: str,
        query: str | None,
    ) -> dict[str, Path]:
        """Legacy build mode for backward compatibility.

        Args:
            documents: Document iterator.
            dataset_name: Name for dataset.
            query: Query string.

        Returns:
            Dictionary of output paths.
        """
        logger.info(f"Building {self.dataset_config.type.value} dataset: {dataset_name}")

        # Get documents
        if documents is None:
            documents = self._fetch_documents(query)

        # Generate samples
        samples = list(self._generate_samples(documents))
        logger.info(f"Generated {len(samples)} samples")

        if not samples:
            logger.warning("No samples generated, creating empty dataset")
            samples = []

        # Split into train/val/test
        splitter = DatasetSplitter(
            train_ratio=1.0 - self.dataset_config.val_split - self.dataset_config.test_split,
            val_ratio=self.dataset_config.val_split,
            test_ratio=self.dataset_config.test_split,
            seed=42,
        )

        if self.dataset_config.test_split > 0:
            train_samples, val_samples, test_samples = splitter.split_three(samples)
        else:
            train_samples, val_samples = splitter.split(samples)
            test_samples = []

        # Save datasets
        paths = self._save_datasets(samples, train_samples, val_samples, test_samples, dataset_name)

        # Generate and save statistics
        stats = self._generate_statistics(samples, train_samples, val_samples, test_samples)
        stats_path = self.output_dir / f"{dataset_name}_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        paths["statistics"] = stats_path

        logger.info(f"Dataset saved to {self.output_dir}")
        return paths

    def _fetch_documents(self, query: str | None = None) -> Iterator[dict[str, Any]]:
        """Fetch documents from FlatSeek index.

        Args:
            query: Query string. Uses config default if not provided.

        Yields:
            Documents from the FlatSeek index.
        """
        if self.provider is None:
            self.connect()

        if self.provider is None:
            raise RuntimeError("No FlatSeek provider available and no documents provided")

        export_query = query or self.flatseek_config.query or "*"
        yield from self.provider.stream(export_query)

    def _generate_samples(
        self,
        documents: Iterator[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Generate training samples from documents.

        Args:
            documents: Iterator of documents from FlatSeek.

        Yields:
            Generated training samples.
        """
        sample_count = 0
        max_samples = self.dataset_config.max_samples

        for doc in documents:
            # Apply query filters if configured
            if not self._passes_filters(doc):
                continue

            # Generate samples using configured generators
            if self.generators:
                for generator in self.generators:
                    generated = generator.generate(doc, self.dataset_config)
                    for sample in generated:
                        # Validate sample
                        if self._validate_sample(sample):
                            yield sample
                            sample_count += 1
                            if max_samples and sample_count >= max_samples:
                                return
            else:
                # Use default generation
                sample = self._generate_default_sample(doc)
                if sample and self._validate_sample(sample):
                    yield sample
                    sample_count += 1
                    if max_samples and sample_count >= max_samples:
                        return

    def _passes_filters(self, doc: dict[str, Any]) -> bool:
        """Check if document passes configured filters.

        Args:
            doc: Document to check.

        Returns:
            True if document passes all filters.
        """
        # Check if document has any non-empty values
        # For KV data (athletes), check if at least one value exists
        has_content = False
        for key, value in doc.items():
            if key.startswith("_"):
                continue
            if value is not None and str(value).strip():
                has_content = True
                break

        if not has_content:
            return False

        # Length filter (only for text-heavy documents)
        content = self._extract_content(doc)
        if content and len(content) < self.dataset_config.min_length:
            # For KV docs with short values, don't reject if they have content
            if len(doc) > 1:  # Has multiple fields = likely KV data
                return True
            return False
        if content and len(content) > self.dataset_config.max_length:
            # Documents can still be truncated, so we don't reject here
            pass

        return True

    def _validate_sample(self, sample: dict[str, Any]) -> bool:
        """Validate that a sample has required fields.

        Args:
            sample: Sample to validate.

        Returns:
            True if sample is valid.
        """
        required_fields = ["instruction", "input", "output"]
        for field in required_fields:
            if field not in sample:
                return False

        # Check minimum content
        if not sample.get("input"):
            return False

        return True

    def _generate_default_sample(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a default sample from a document.

        Args:
            doc: Source document.

        Returns:
            Generated sample or None if invalid.
        """
        content = self._extract_content(doc)
        if not content or len(content) < self.dataset_config.min_length:
            return None

        if len(content) > self.dataset_config.max_length:
            content = content[: self.dataset_config.max_length]

        instruction = self._get_instruction_for_type(self.dataset_config.type)

        return {
            "instruction": instruction,
            "input": content,
            "output": "",
            "metadata": {
                "source": doc.get("_source", "unknown"),
                "type": self.dataset_config.type.value,
            },
        }

    def _get_instruction_for_type(self, dataset_type: DatasetType) -> str:
        """Get default instruction for dataset type.

        Args:
            dataset_type: Type of dataset.

        Returns:
            Instruction string.
        """
        instructions = {
            DatasetType.INSTRUCTION: "Process the following text according to the task.",
            DatasetType.CHAT: "Respond to the user's message.",
            DatasetType.QA: "Answer the following question based on the provided text.",
            DatasetType.CLASSIFICATION: "Classify the following text into a category.",
            DatasetType.SUMMARY: "Summarize the following text concisely.",
            DatasetType.EXTRACTION: "Extract key information from the following text.",
            DatasetType.RAG: "Answer the question based on the provided context.",
        }
        return instructions.get(dataset_type, "Process the following text.")

    def _extract_content(self, doc: dict[str, Any]) -> str:
        """Extract text content from a document.

        Args:
            doc: Source document.

        Returns:
            Extracted text content.
        """
        # Try common text fields in order
        for field in ["text", "content", "body", "description", "abstract"]:
            if field in doc and doc[field]:
                return str(doc[field])

        # Fall back to joining all string fields
        parts = []
        for key, value in doc.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str) and len(value) > 10:
                parts.append(value)

        return " ".join(parts) if parts else ""

    def _save_datasets(
        self,
        all_samples: list[dict[str, Any]],
        train_samples: list[dict[str, Any]],
        val_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]],
        dataset_name: str,
    ) -> dict[str, Path]:
        """Save datasets to files.

        Args:
            all_samples: All generated samples.
            train_samples: Training samples.
            val_samples: Validation samples.
            test_samples: Test samples.
            dataset_name: Name for the dataset.

        Returns:
            Dictionary mapping file types to paths.
        """
        paths = {}

        # Save all samples as JSONL
        all_path = self.output_dir / f"{dataset_name}.jsonl"
        with open(all_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        paths["dataset"] = all_path

        # Save train set
        train_path = self.output_dir / f"{dataset_name}_train.jsonl"
        with open(train_path, "w") as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        paths["train"] = train_path

        # Save validation set
        val_path = self.output_dir / f"{dataset_name}_val.jsonl"
        with open(val_path, "w") as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        paths["val"] = val_path

        # Save test set if present
        if test_samples:
            test_path = self.output_dir / f"{dataset_name}_test.jsonl"
            with open(test_path, "w") as f:
                for sample in test_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            paths["test"] = test_path

        # Save metadata
        metadata = self._generate_metadata(
            all_samples, train_samples, val_samples, test_samples, dataset_name
        )
        metadata_path = self.output_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        paths["metadata"] = metadata_path

        return paths

    def _generate_metadata(
        self,
        all_samples: list[dict[str, Any]],
        train_samples: list[dict[str, Any]],
        val_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]],
        dataset_name: str,
    ) -> dict[str, Any]:
        """Generate dataset metadata.

        Args:
            all_samples: All generated samples.
            train_samples: Training samples.
            val_samples: Validation samples.
            test_samples: Test samples.
            dataset_name: Name for the dataset.

        Returns:
            Metadata dictionary.
        """
        test_split = len(test_samples) / len(all_samples) if all_samples else 0

        return {
            "name": dataset_name,
            "type": self.dataset_config.type.value,
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "train_split": self.dataset_config.train_split,
            "val_split": self.dataset_config.val_split,
            "test_split": test_split,
            "max_samples": self.dataset_config.max_samples,
            "generators": [g.__class__.__name__ for g in self.generators],
            "min_length": self.dataset_config.min_length,
            "max_length": self.dataset_config.max_length,
        }

    def _generate_statistics(
        self,
        all_samples: list[dict[str, Any]],
        train_samples: list[dict[str, Any]],
        val_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate dataset statistics.

        Args:
            all_samples: All generated samples.
            train_samples: Training samples.
            val_samples: Validation samples.
            test_samples: Test samples.

        Returns:
            Statistics dictionary.
        """
        # Analyze samples
        input_lengths = []
        output_lengths = []
        sample_types: Counter = Counter()
        metadata_fields: Counter = Counter()

        for sample in all_samples:
            input_lengths.append(len(sample.get("input", "")))
            output_lengths.append(len(sample.get("output", "")))

            # Count types
            if "metadata" in sample and isinstance(sample["metadata"], dict):
                gen_type = sample["metadata"].get("generator", "unknown")
                sample_types[gen_type] += 1

                # Count metadata fields
                for key in sample["metadata"].keys():
                    metadata_fields[key] += 1

        # Calculate percentiles
        input_lengths.sort()
        output_lengths.sort()

        def percentile(sorted_list: list[int], p: float) -> int:
            if not sorted_list:
                return 0
            idx = int(len(sorted_list) * p)
            return sorted_list[min(idx, len(sorted_list) - 1)]

        stats = {
            "total_samples": len(all_samples),
            "split_sizes": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "input_length": {
                "min": min(input_lengths) if input_lengths else 0,
                "max": max(input_lengths) if input_lengths else 0,
                "mean": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
                "p50": percentile(input_lengths, 0.5),
                "p95": percentile(input_lengths, 0.95),
                "p99": percentile(input_lengths, 0.99),
            },
            "output_length": {
                "min": min(output_lengths) if output_lengths else 0,
                "max": max(output_lengths) if output_lengths else 0,
                "mean": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
                "p50": percentile(output_lengths, 0.5),
                "p95": percentile(output_lengths, 0.95),
                "p99": percentile(output_lengths, 0.99),
            },
            "sample_types": dict(sample_types),
            "metadata_fields": dict(metadata_fields),
        }

        return stats

    def get_build_plan(self) -> Any | None:
        """Get the build plan from the last build run.

        Returns:
            BuildPlan if available, None otherwise.
        """
        return self._build_plan


def build_dataset(
    flatseek_config: FlatseekConfig,
    dataset_config: DatasetConfig,
    output_dir: str | Path | None = None,
    dataset_name: str = "dataset",
    query: str | None = None,
) -> dict[str, Path]:
    """Convenience function to build a dataset.

    Args:
        flatseek_config: Configuration for FlatSeek index access.
        dataset_config: Configuration for dataset generation.
        output_dir: Directory to save generated datasets.
        dataset_name: Name for the dataset.
        query: Query string for FlatSeek export.

    Returns:
        Dictionary with paths to all generated files.
    """
    builder = DatasetBuilder(
        flatseek_config=flatseek_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
    )
    return builder.build(dataset_name=dataset_name, query=query)
