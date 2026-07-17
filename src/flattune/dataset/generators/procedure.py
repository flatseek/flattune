"""Procedure/instruction dataset generator."""

from __future__ import annotations

import re
from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class ProcedureGenerator(BaseGenerator):
    """Generates procedural instruction samples from documents.

    Creates training samples that teach models to follow or generate
    step-by-step instructions and procedures.

    Supported types: procedure, howto
    """

    name = "procedure"
    supported_types = ["procedure", "howto"]

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate procedure/instruction samples.

        Args:
            document: Source document.
            config: Dataset configuration.
            instruction: Optional instruction template.

        Returns:
            List of procedure samples.
        """
        samples = []

        # Use provided instruction or select based on type
        base_instruction = instruction or "Explain the steps to accomplish this task."

        # Extract content
        content = self._extract_field(document, ["text", "content", "body", "description"])
        title = self._extract_field(document, ["title", "name", "subject"])

        if not content:
            return []

        # Filter by length
        content = self._filter_by_length(content, 100, config.max_length)
        if not content:
            return []

        # Extract numbered steps
        numbered_steps = re.findall(
            r"(?:\d+\.?\s+(?:step\s+)?(?:(?:\d+|[a-z])\.?\s+)?)([^\n]+)", content, re.IGNORECASE
        )

        # Extract bullet points
        bullet_points = re.findall(r"^\s*[-*+]\s+([^\n]+)", content, re.MULTILINE)

        # Combine steps
        all_steps = numbered_steps + bullet_points

        if all_steps:
            # Create procedure sample
            steps_text = "\n".join(
                f"{i + 1}. {step.strip()}" for i, step in enumerate(all_steps[:10])
            )

            samples.append(
                {
                    "instruction": base_instruction,
                    "input": f"Task: {title or content[:100]}\n\n{steps_text}",
                    "output": content[:300] if len(content) > 300 else content,
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "procedure",
                        "type": "procedure",
                        "step_count": len(all_steps[:10]),
                    },
                }
            )

            # How-to sample
            samples.append(
                {
                    "instruction": "How do I accomplish this task?",
                    "input": f"Task: {title or content[:100]}",
                    "output": steps_text if all_steps else content[:200],
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "procedure",
                        "type": "howto",
                    },
                }
            )
        else:
            # No clear steps, create general instruction sample
            samples.append(
                {
                    "instruction": base_instruction,
                    "input": f"Task: {title or content[:100]}",
                    "output": content[:300] if len(content) > 300 else content,
                    "metadata": {
                        "source": document.get("_source", "unknown"),
                        "generator": "procedure",
                        "type": "procedure",
                    },
                }
            )

        return samples
