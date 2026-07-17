"""Safety dataset generator for content filtering and safety training."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class SafetyGenerator(BaseGenerator):
    """Generates safety and content filtering training samples.

    Creates input/output pairs for training models to identify
    potentially unsafe or inappropriate content.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate safety training samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing safety-related samples if valid content exists.
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

        # Content safety classification
        samples.append(
            {
                "instruction": "Is the following content safe and appropriate? Answer yes or no.",
                "input": content,
                "output": "yes",  # Assuming FlatSeek content is curated/safe
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "safety",
                    "type": "safety_classification",
                },
            }
        )

        # Topic classification for sensitive content detection
        samples.append(
            {
                "instruction": "Does the following text contain sensitive topics? If yes, identify them.",
                "input": content,
                "output": "none",  # Placeholder for sensitive topic detection
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "safety",
                    "type": "sensitive_topic_detection",
                },
            }
        )

        return samples
