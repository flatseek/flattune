"""Anthropic teacher implementation."""

import logging
import os
from collections.abc import Iterator
from typing import Any

from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import register_teacher
from flattune.teach.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)


@register_teacher("anthropic")
class AnthropicTeacher(BaseTeacher):
    """Teacher using Anthropic Claude API."""

    name = "anthropic"
    supports_local = False
    supports_remote = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ):
        """Initialize Anthropic teacher.

        Args:
            model: Model name (e.g., claude-sonnet-4-20250514).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            api_key: Anthropic API key.
        """
        super().__init__(model, temperature, max_tokens, api_key)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("Anthropic package not installed. Install with: pip install anthropic")
                raise

        return self._client

    def generate(
        self,
        knowledge: str | KnowledgeNode,
        sample_type: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a training sample using Claude.

        Args:
            knowledge: Knowledge to generate from.
            sample_type: Type of sample to generate.
            **kwargs: Additional parameters.

        Returns:
            Generated conversation.
        """
        if isinstance(knowledge, KnowledgeNode):
            knowledge_text = knowledge.get_context()
        else:
            knowledge_text = str(knowledge)

        prompt = self._build_prompt(knowledge_text, sample_type)
        system_prompt = self.get_system_prompt(sample_type)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            return self._parse_response(content, sample_type, knowledge_text)

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return self._error_response(str(e), sample_type)

    def generate_batch(
        self,
        knowledge_iter: Iterator[KnowledgeNode],
        sample_type: str,
        **kwargs,
    ) -> Iterator[dict[str, Any]]:
        """Generate multiple samples."""
        for node in knowledge_iter:
            try:
                result = self.generate(node, sample_type, **kwargs)
                yield result
            except Exception as e:
                logger.warning(f"Failed to generate from node {node.id}: {e}")
                continue

    def _build_prompt(self, knowledge: str, sample_type: str) -> str:
        """Build generation prompt."""
        base_prompts = {
            "qa": f"""Based on the following knowledge, generate a Q&A pair.

Knowledge:
{knowledge}

Generate a question that tests understanding of this knowledge, followed by a comprehensive answer.
Format your response as:
Question: <your question>
Answer: <your answer>""",
            "summarization": f"""Summarize the following knowledge concisely.

Knowledge:
{knowledge}

Provide a clear, concise summary that captures the key points.""",
            "tool_call": f"""Based on the following API/tool documentation, generate example conversations
showing how a user would request actions and how the tool should be called.

API Documentation:
{knowledge}

Generate 2-3 diverse examples of user requests and appropriate tool calls.""",
            "nl_to_sql": f"""Based on the following database schema, generate natural language
to SQL query pairs.

Schema:
{knowledge}

Generate 2-3 examples of user questions and corresponding SQL queries.""",
        }

        return base_prompts.get(
            sample_type,
            f"Based on the following knowledge, generate a {sample_type} sample.\n\nKnowledge:\n{knowledge}",
        )

    def _parse_response(
        self,
        content: str,
        sample_type: str,
        knowledge: str,
    ) -> dict[str, Any]:
        """Parse Claude response into conversation format."""
        return {
            "messages": [
                {"role": "user", "content": f"Generate a {sample_type} response"},
                {"role": "assistant", "content": content},
            ],
            "sample_type": sample_type,
            "source": knowledge[:100],
            "quality_score": 0.8,
        }

    def _error_response(self, error: str, sample_type: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "messages": [
                {"role": "user", "content": "Generate response"},
                {"role": "assistant", "content": f"Error: {error}"},
            ],
            "sample_type": sample_type,
            "quality_score": 0.0,
            "error": error,
        }
