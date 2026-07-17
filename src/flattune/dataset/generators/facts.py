"""Facts extraction dataset generator."""

from __future__ import annotations

import re
from typing import Any, Optional

from flattune.dataset.generators.base import BaseGenerator


class FactsGenerator(BaseGenerator):
    """Extracts factual statements from documents.

    Generates samples that teach models to extract factual information
    including numbers, dates, definitions, and concrete details.

    Supported types: facts, definition
    """

    name = "facts"
    supported_types = ["facts", "definition"]

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Extract factual statements from a document.

        Args:
            document: Source document.
            config: Dataset configuration.
            instruction: Optional instruction template.

        Returns:
            List of fact extraction samples.
        """
        samples = []
        base_instruction = instruction or "Extract factual statements from the following content."

        # Extract content
        content = self._extract_field(document, ["text", "content", "body", "description", "abstract"])
        title = self._extract_field(document, ["title", "name", "subject"])

        if not content:
            return []

        # Filter by length
        content = self._filter_by_length(content, 50, config.max_length)
        if not content:
            return []

        # Extract numbers with context (facts)
        number_patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:million|billion|trillion)", "quantity_millions"),
            (r"(\d+(?:\.\d+)?)\s*(?:percent|%)", "percentage"),
            (r"(\d{4})", "year"),
            (r" magnitude (?:of|:|-|\s)*(\d+(?:\.\d+)?)", "magnitude"),
        ]

        for pattern, fact_type in number_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3 per type
                context_match = re.search(
                    rf".{{0,50}}{re.escape(match)}.{{0,50}}",
                    content,
                    re.IGNORECASE
                )
                if context_match:
                    context = context_match.group(0).strip()
                    samples.append({
                        "instruction": base_instruction,
                        "input": f"Text: {content[:500]}",
                        "output": context,
                        "metadata": {
                            "source": document.get("_source", "unknown"),
                            "generator": "facts",
                            "type": fact_type,
                            "extracted_value": match,
                        },
                    })

        # Extract definitions (definition type)
        definition_patterns = [
            (r"is defined as\s+([^\.]+)", "is_defined_as"),
            (r"means\s+([^\.]+)", "means"),
            (r"refers to\s+([^\.]+)", "refers_to"),
            (r"is a\s+([^\.]+)", "is_a"),
        ]

        for pattern, def_type in definition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 per type
                definition = match.strip()
                if len(definition) > 10 and len(definition) < 200:
                    samples.append({
                        "instruction": "Provide a definition for the term used in this context.",
                        "input": f"Text: {content[:500]}",
                        "output": definition,
                        "metadata": {
                            "source": document.get("_source", "unknown"),
                            "generator": "facts",
                            "type": "definition",
                            "definition_type": def_type,
                        },
                    })

        # If no specific patterns found, create a general fact extraction sample
        if not samples and len(content) > 100:
            samples.append({
                "instruction": base_instruction,
                "input": f"Text: {content[:500]}",
                "output": content[:200],
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "facts",
                    "type": "general",
                },
            })

        return samples
