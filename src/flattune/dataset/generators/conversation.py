"""Conversation dataset generator."""

from typing import Any

from flattune.dataset.generators.base import BaseGenerator


class ConversationGenerator(BaseGenerator):
    """Generates multi-turn conversation training samples.

    Creates dialogue-style training data for chat-based fine-tuning.
    """

    def generate(
        self,
        document: dict[str, Any],
        config: Any,
    ) -> list[dict[str, Any]]:
        """Generate conversation samples from a document.

        Args:
            document: Source document from FlatSeek.
            config: Dataset configuration.

        Returns:
            List containing conversation samples if valid content exists.
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

        # Single-turn question answering as conversation
        samples.append(
            {
                "instruction": "",
                "input": f"User: Can you explain the following?\n{content}\n\nAssistant:",
                "output": "",  # Model learns to respond
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "conversation",
                    "type": "qa_dialogue",
                },
            }
        )

        # Context-based问答
        samples.append(
            {
                "instruction": "",
                "input": f"Context: {content}\n\nUser: Based on the above, what should I know?\n\nAssistant:",
                "output": "",
                "metadata": {
                    "source": document.get("_source", "unknown"),
                    "generator": "conversation",
                    "type": "contextual_qa",
                },
            }
        )

        return samples
