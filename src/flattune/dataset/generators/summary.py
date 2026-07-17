"""Summarization dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class SummaryGenerator(BaseGenerator):
    """Generates summarization training samples from documents.

    Creates input/output pairs where the input is a longer text
    and the output is (or would be) its summary.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate summarization samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing one summarization sample if valid content exists.
        """
        # Extract content from document
        content = self._extract_field(document, ["text", "content", "body", "description", "abstract"])
        if not content:
            return []

        # Need sufficient content for summarization
        if len(content) < config.min_length * 2:
            return []

        # Filter by max length
        if len(content) > config.max_length:
            content = content[:config.max_length]

        samples = []

        # Create summarization prompt
        samples.append({
            "instruction": "Summarize the following text concisely.",
            "input": content,
            "output": "",  # Model learns to generate summaries
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "summary",
                "type": "summary",
                "original_length": len(content),
            },
        })

        # Also add a key-points extraction variant
        samples.append({
            "instruction": "Extract the key points from the following text.",
            "input": content,
            "output": "",  # Model learns to extract key points
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "summary",
                "type": "key_points",
                "original_length": len(content),
            },
        })

        return samples
