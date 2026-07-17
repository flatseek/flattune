"""Comparison dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class ComparisonGenerator(BaseGenerator):
    """Generates comparison training samples from documents.

    Creates input/output pairs for comparing or contrasting items,
    entities, or concepts found in documents.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate comparison samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing comparison samples if valid content exists.
        """
        # Extract content from document
        content = self._extract_field(document, ["text", "content", "body", "description"])
        if not content:
            return []

        # Filter by length
        content = self._filter_by_length(content, config.min_length, config.max_length)
        if not content:
            return []

        samples = []

        # Comparison prompt
        samples.append({
            "instruction": "Compare and contrast the items described in the following text.",
            "input": content,
            "output": "",  # Model learns to generate comparisons
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "comparison",
                "type": "comparison",
            },
        })

        # Similarity/difference analysis
        samples.append({
            "instruction": "Identify similarities and differences in the following text.",
            "input": content,
            "output": "",
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "comparison",
                "type": "similarity_analysis",
            },
        })

        return samples
