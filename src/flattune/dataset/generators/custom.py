"""Custom dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class CustomGenerator(BaseGenerator):
    """Generates custom training samples based on document structure.

    This is a flexible generator that creates samples based on
    the actual fields present in each document, making it adaptable
    to various dataset schemas without explicit configuration.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate custom samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing custom samples based on document structure.
        """
        if not document:
            return []

        samples = []

        # Get all string fields that could be used for training
        text_fields = []
        for key, value in document.items():
            if key.startswith("_"):
                continue  # Skip metadata fields
            if isinstance(value, str) and len(value) > config.min_length:
                text_fields.append((key, value))

        if not text_fields:
            return []

        # Sort by length (longest first)
        text_fields.sort(key=lambda x: len(x[1]), reverse=True)

        # Create sample using the longest field
        field_name, content = text_fields[0]

        # Truncate if needed
        if len(content) > config.max_length:
            content = content[:config.max_length]

        # Generic instruction based on field name
        instruction = self._generate_instruction(field_name, document)

        samples.append({
            "instruction": instruction,
            "input": content,
            "output": "",
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "custom",
                "field": field_name,
                "type": "custom",
            },
        })

        # Also create a sample using second longest field if available
        if len(text_fields) > 1:
            field_name2, content2 = text_fields[1]
            if len(content2) > config.min_length:
                if len(content2) > config.max_length:
                    content2 = content2[:config.max_length]

                samples.append({
                    "instruction": f"Process the following {field_name2}.",
                    "input": content2,
                    "output": "",
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "custom",
                        "field": field_name2,
                        "type": "custom",
                    },
                })

        return samples

    def _generate_instruction(self, field_name: str, document: dict[str, Any]) -> str:
        """Generate an instruction based on field name."""
        field_lower = field_name.lower()

        if any(word in field_lower for word in ["title", "name", "heading"]):
            return "Process the following title."
        elif any(word in field_lower for word in ["desc", "summary", "abstract"]):
            return "Summarize the following description."
        elif any(word in field_lower for word in ["text", "content", "body"]):
            return "Process the following text."
        elif any(word in field_lower for word in ["question", "query"]):
            return "Answer the following question."
        elif any(word in field_lower for word in ["answer", "response"]):
            return "Evaluate the following response."
        elif any(word in field_lower for word in ["comment", "review", "feedback"]):
            return "Analyze the following comment."
        else:
            return f"Process the following {field_name}."
