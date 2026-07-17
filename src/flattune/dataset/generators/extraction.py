"""Information extraction dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class ExtractionGenerator(BaseGenerator):
    """Generates information extraction training samples.

    Creates input/output pairs for extracting structured information
    from unstructured text (entities, relationships, facts).
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate extraction samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing extraction samples if valid content exists.
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

        # Named entity extraction
        samples.append(
            {
                "instruction": "Extract all named entities (people, places, organizations) from the following text.",
                "input": content,
                "output": "",  # Model learns entity extraction
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "extraction",
                    "type": "ner",
                },
            }
        )

        # Key facts extraction
        samples.append(
            {
                "instruction": "Extract key facts and information from the following text.",
                "input": content,
                "output": "",
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "extraction",
                    "type": "fact_extraction",
                },
            }
        )

        # Structured data extraction
        samples.append(
            {
                "instruction": "Extract structured information as key-value pairs from the following text.",
                "input": content,
                "output": "",
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "extraction",
                    "type": "structured_extraction",
                },
            }
        )

        return samples
