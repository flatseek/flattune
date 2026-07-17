"""Question-Answer dataset generator."""

from __future__ import annotations

import json
from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class QAGenerator(BaseGenerator):
    """Generates question-answer pairs from documents.

    This generator creates QA training samples where the input is a question
    derived from the document content and the output is the answer.

    It works generically with any key-value document by:
    1. Extracting all available fields from the document
    2. Generating QA pairs for each field
    3. Creating composite questions about the document

    Supported types: context_qa, direct_qa, yes_no, citation_qa
    """

    name = "qa"
    supported_types = ["context_qa", "direct_qa", "yes_no", "citation_qa", "qa"]

    # Fields to exclude from QA generation
    EXCLUDE_FIELDS = {"_id", "_source", "id", "index"}

    # Fields that should use specific question patterns
    FIELD_PATTERNS = {
        "name": ("What is the name of {value}?", "Tell me the name."),
        "title": ("What is the title?", "What is this titled?"),
        "description": ("Describe {value}.", "What is the description?"),
        "body": ("What does the body say?", "What is the content?"),
        "content": ("What is the content?", "What does this say?"),
        "text": ("What text is provided?", "Read the following text."),
        "summary": ("What is the summary?", "Summarize this."),
        "sport": ("What sport does {value} play?", "What sport is this?"),
        "team": ("Which team is {value} from?", "What team?"),
        "country": ("Which country is {value} from?", "What country?"),
        "city": ("Which city is this in?", "Where is this located?"),
        "year": ("What year was this?", "When did this happen?"),
        "date": ("What date is this?", "When is this?"),
        "age": ("How old is {value}?", "What is the age?"),
        "height": ("What is the height?", "How tall is this?"),
        "weight": ("What is the weight?", "How much does this weigh?"),
        "medal": ("What medal was won?", "Did they win a medal?"),
        "event": ("What event is this?", "What happened?"),
        "sex": ("What is the gender?", "Male or female?"),
        "season": ("What season was this?", "Summer or Winter?"),
        "games": ("What games is this from?", "Which Olympics?"),
        "noc": ("What is the NOC code?", "What organization?"),
    }

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate QA samples from a document.

        Args:
            document: Source document from FlatSeek or other source.
            config: Dataset configuration.
            instruction: Optional instruction template to use.

        Returns:
            List containing QA samples.
        """
        samples = []

        # Use provided instruction or default
        base_instruction = instruction or "Answer the question based on the provided information."

        # Get all fields from document
        fields = self._extract_all_fields(document)

        if not fields:
            return []

        # Generate composite context for context_qa
        context = self._build_context(fields)

        # Generate QA for each field
        for field_name, field_value in fields.items():
            if field_value is None or str(field_value).strip() == "":
                continue

            value_str = str(field_value)[:200]  # Limit value length

            # Get question pattern for this field
            question = self._get_question_for_field(field_name, value_str)

            # Add direct QA sample
            samples.append({
                "instruction": base_instruction,
                "input": f"Question: {question}\nContext: {context}",
                "output": value_str,
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "qa",
                    "type": "direct_qa",
                    "field": field_name,
                },
            })

            # Add yes/no for medal/gender fields
            if field_name in ("medal", "sex", "gender"):
                yes_no_question = "Did this athlete win a medal?"
                if field_value and str(field_value).lower() not in ("na", "none", "none", ""):
                    answer = "Yes"
                else:
                    answer = "No"
                samples.append({
                    "instruction": "Answer yes or no.",
                    "input": f"Question: {yes_no_question}",
                    "output": answer,
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "qa",
                        "type": "yes_no",
                        "field": field_name,
                    },
                })

        # Add composite context QA
        if len(fields) > 1:
            # Get name or title for composite
            name = fields.get("name") or fields.get("title") or fields.get("event") or "this"
            composite_question = f"Tell me about {name}."

            samples.append({
                "instruction": base_instruction,
                "input": f"Question: {composite_question}\nContext: {context}",
                "output": str(name) if name else context[:200],
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "qa",
                    "type": "context_qa",
                },
            })

        # Limit samples per document
        return samples[:10]

    def _extract_all_fields(self, document: dict) -> dict:
        """Extract all relevant fields from document.

        Args:
            document: Source document.

        Returns:
            Dictionary of field name -> value for all relevant fields.
        """
        fields = {}
        for key, value in document.items():
            if key in self.EXCLUDE_FIELDS:
                continue
            if key.startswith("_"):
                continue
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                # Skip complex types, convert to string
                value = json.dumps(value) if len(str(value)) < 100 else str(value)[:100]
            else:
                value = str(value)
            if value.strip():
                fields[key] = value
        return fields

    def _build_context(self, fields: dict) -> str:
        """Build a context string from fields.

        Args:
            fields: Dictionary of fields.

        Returns:
            Context string.
        """
        parts = []
        for key, value in list(fields.items())[:10]:  # Limit to 10 fields
            parts.append(f"{key}: {value}")
        return " | ".join(parts)

    def _get_question_for_field(self, field_name: str, value: str) -> str:
        """Get appropriate question for a field.

        Args:
            field_name: Name of the field.
            value: Value of the field.

        Returns:
            Question string.
        """
        # Check for specific pattern
        if field_name in self.FIELD_PATTERNS:
            question_templates = self.FIELD_PATTERNS[field_name]
            if isinstance(question_templates, tuple):
                question = question_templates[0]
            else:
                question = question_templates
            return question.replace("{value}", value[:30])

        # Generic fallback
        return f"What is the {field_name}?"

    def _extract_field(self, doc: dict, field_names: list) -> str:
        """Extract field value from document, trying multiple field names."""
        for name in field_names:
            if name in doc:
                val = doc[name]
                if val and str(val).strip():
                    return str(val)
        return None
