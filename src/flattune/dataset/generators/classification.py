"""Classification dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class ClassificationGenerator(BaseGenerator):
    """Generates text classification training samples.

    Creates input/output pairs where the input is text to classify
    and the output is the predicted class/category.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate classification samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing one classification sample if valid content exists.
        """
        # Extract content and category from document
        content = self._extract_field(document, ["text", "content", "body", "description", "title"])
        if not content:
            return []

        # Filter by length
        content = self._filter_by_length(content, config.min_length, config.max_length)
        if not content:
            return []

        # Extract or infer category
        category = self._extract_category(document)

        samples = []

        if category:
            # Direct classification sample
            samples.append({
                "instruction": "Classify the following text into a category.",
                "input": content,
                "output": category,
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "classification",
                    "type": "classification",
                    "category": category,
                },
            })

        # Multi-label classification variant
        samples.append({
            "instruction": "What topics does the following text cover?",
            "input": content,
            "output": "",  # Model learns topic detection
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "classification",
                "type": "topic_detection",
            },
        })

        return samples

    def _extract_category(self, document: dict[str, Any]) -> str:
        """Extract category from document."""
        # Check for explicit category field
        for field in ["category", "categories", "class", "type", "label"]:
            if field in document and document[field]:
                return str(document[field])

        # Check for tags
        if "tags" in document and document["tags"]:
            tags = document["tags"]
            if isinstance(tags, list) and tags:
                return tags[0]
            return str(tags)

        return ""
