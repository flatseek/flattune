"""Base class for dataset generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratorOutput:
    """Standard output format for generators."""
    instruction: str
    input: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }


class BaseGenerator(ABC):
    """Abstract base class for dataset generators.

    Each generator produces samples in the instruction tuning format:
    - instruction: The task instruction
    - input: The input text/prompt
    - output: The expected output
    - metadata: Additional information about the sample

    Generators can be registered using the @register_generator decorator.
    """

    name: str = "base"
    supported_types: list[str] = field(default_factory=list)

    @abstractmethod
    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training samples from a document.

        Args:
            document: A document from the FlatSeek index or other source.
            config: Dataset configuration.
            instruction: Optional instruction template to use.

        Returns:
            List of generated samples, each containing:
            - instruction: The task instruction
            - input: The input text
            - output: The expected output
            - metadata: Additional sample metadata
        """
        pass

    def supports(self, dataset_type: str) -> bool:
        """Check if generator supports a dataset type.

        Args:
            dataset_type: Name of the dataset type.

        Returns:
            True if supported.
        """
        return dataset_type in self.supported_types or not self.supported_types

    def _filter_by_length(
        self,
        text: str,
        min_length: int,
        max_length: int,
    ) -> str:
        """Filter text by length constraints.

        Args:
            text: Text to filter.
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.

        Returns:
            Filtered text or empty string if outside bounds.
        """
        if len(text) < min_length:
            return ""
        if len(text) > max_length:
            text = text[:max_length]
        return text

    def _extract_field(
        self,
        document: dict[str, Any],
        fields: list[str],
    ) -> str:
        """Extract text from document using field priority.

        Args:
            document: Source document.
            fields: List of field names to try in order.

        Returns:
            First non-empty field value or empty string.
        """
        for field in fields:
            if field in document and document[field]:
                value = document[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (list, dict)):
                    # Skip complex types for now
                    continue
        return ""

    def _build_sample(
        self,
        instruction: str,
        input_text: str,
        output: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standardized sample dictionary.

        Args:
            instruction: Task instruction.
            input_text: Input text.
            output: Expected output.
            metadata: Additional metadata.

        Returns:
            Sample dictionary.
        """
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "metadata": metadata or {},
        }
