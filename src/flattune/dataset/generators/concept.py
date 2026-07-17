"""Concept explanation dataset generator."""

from __future__ import annotations

from typing import Any, Optional

from flattune.dataset.generators.base import BaseGenerator


class ConceptGenerator(BaseGenerator):
    """Generates concept explanation samples from documents.

    Creates training samples that teach models to explain concepts
    in detail, based on provided context.

    Supported types: concept
    """

    name = "concept"
    supported_types = ["concept"]

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Generate concept explanation samples.

        Args:
            document: Source document.
            config: Dataset configuration.
            instruction: Optional instruction template.

        Returns:
            List of concept explanation samples.
        """
        samples = []
        base_instruction = instruction or "Explain this concept in detail."

        # Extract content
        content = self._extract_field(document, ["text", "content", "body", "description"])
        title = self._extract_field(document, ["title", "name", "subject"])

        if not content:
            return []

        # Filter by length (need substantial content for concepts)
        content = self._filter_by_length(content, 100, config.max_length)
        if not content:
            return []

        # Generate concept explanation sample
        samples.append({
            "instruction": base_instruction,
            "input": f"Concept: {title or content[:100]}\n\nContext: {content[:800]}",
            "output": content[:400] if len(content) > 400 else content,
            "metadata": {
                "source": document.get("_source", "unknown"),
                "generator": "concept",
                "type": "concept",
            },
        })

        return samples
