"""Glossary extraction dataset generator."""

from __future__ import annotations

import re
from typing import Any, Optional

from flattune.dataset.generators.base import BaseGenerator


class GlossaryGenerator(BaseGenerator):
    """Extracts glossary terms and definitions from documents.

    Generates samples that teach models to identify and define terms,
    particularly from structured or semi-structured content.

    Supported types: glossary
    """

    name = "glossary"
    supported_types = ["glossary"]

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Extract glossary terms and definitions.

        Args:
            document: Source document.
            config: Dataset configuration.
            instruction: Optional instruction template.

        Returns:
            List of glossary samples.
        """
        samples = []
        base_instruction = instruction or "Define the following terms."

        # Extract content
        content = self._extract_field(document, ["text", "content", "body", "description"])
        title = self._extract_field(document, ["title", "name", "subject"])

        if not content:
            return []

        # Filter by length
        content = self._filter_by_length(content, 100, config.max_length)
        if not content:
            return []

        # Extract bold/italic terms (markdown formatting)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', content)
        italic_terms = re.findall(r'\*([^*]+)\*', content)
        all_terms = bold_terms + italic_terms

        for term in all_terms[:5]:  # Limit to 5 terms
            term = term.strip()
            if len(term) < 3 or len(term) > 50:
                continue

            # Find context around term
            term_pattern = re.escape(term)
            matches = list(re.finditer(rf'.{{0,30}}{re.escape(term)}.{{0,100}}', content, re.IGNORECASE))
            for match in matches[:2]:
                context = match.group(0).strip()
                # Remove markdown formatting
                context_clean = re.sub(r'\*+', '', context)

                samples.append({
                    "instruction": base_instruction,
                    "input": f"Term: {term}\nContext: {context_clean}",
                    "output": context_clean,
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "glossary",
                        "type": "glossary",
                        "term": term,
                    },
                })

        # Extract definition-like patterns
        definition_patterns = [
            (r"([A-Z][a-z]+(?:\s+[a-z]+)*)\s*[:\-]\s*([^\.\n]+)", "colon_definition"),
            (r"\"([^\"]+)\"\s*means\s*([^\.\n]+)", "means_definition"),
        ]

        for pattern, def_type in definition_patterns:
            matches = re.findall(pattern, content)
            for term, definition in matches[:3]:
                term = term.strip()
                definition = definition.strip()
                if len(term) > 2 and len(definition) > 5:
                    samples.append({
                        "instruction": base_instruction,
                        "input": f"Term: {term}",
                        "output": definition,
                        "metadata": {
                            "source": document.get("_source", "unknown"),
                            "generator": "glossary",
                            "type": "glossary",
                            "term": term,
                            "definition_type": def_type,
                        },
                    })

        return samples
