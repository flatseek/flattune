"""OpenAI teacher implementation."""

import logging
import os
from collections.abc import Iterator
from typing import Any

from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import register_teacher
from flattune.teach.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)


@register_teacher("openai", alias="openai-compatible")
class OpenAITeacher(BaseTeacher):
    """Teacher using OpenAI API or compatible endpoints."""

    name = "openai"
    supports_local = False
    supports_remote = True

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize OpenAI teacher.

        Args:
            model: Model name (e.g., gpt-4o-mini, gpt-4o).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            api_key: OpenAI API key.
            base_url: Custom API base URL (for compatible endpoints).
        """
        super().__init__(model, temperature, max_tokens, api_key, base_url)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                raise

        return self._client

    def generate(
        self,
        knowledge: str | KnowledgeNode,
        sample_type: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a training sample using OpenAI.

        Args:
            knowledge: Knowledge to generate from.
            sample_type: Type of sample to generate.
            **kwargs: Additional parameters.

        Returns:
            Generated conversation.
        """
        # Convert knowledge node to string if needed
        if isinstance(knowledge, KnowledgeNode):
            knowledge_text = knowledge.get_context()
        else:
            knowledge_text = str(knowledge)

        # Build prompt based on sample type
        prompt = self._build_prompt(knowledge_text, sample_type)
        system_prompt = self.get_system_prompt(sample_type)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            return self._parse_response(content, sample_type, knowledge_text)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._error_response(str(e), sample_type)

    def generate_batch(
        self,
        knowledge_iter: Iterator[KnowledgeNode],
        sample_type: str,
        **kwargs,
    ) -> Iterator[dict[str, Any]]:
        """Generate multiple samples.

        Args:
            knowledge_iter: Iterator of knowledge nodes.
            sample_type: Type of sample.
            **kwargs: Additional parameters.

        Yields:
            Generated conversations.
        """
        for node in knowledge_iter:
            try:
                result = self.generate(node, sample_type, **kwargs)
                yield result
            except Exception as e:
                logger.warning(f"Failed to generate from node {node.id}: {e}")
                continue

    def _build_prompt(self, knowledge: str, sample_type: str) -> str:
        """Build generation prompt.

        Args:
            knowledge: Knowledge content.
            sample_type: Type of sample.

        Returns:
            Formatted prompt.
        """
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

            "extraction": f"""Extract key information from the following knowledge.

Knowledge:
{knowledge}

List the most important facts, entities, and relationships.""",

            "tool_call": f"""Based on the following API/tool documentation, generate example conversations
showing how a user would request actions and how the tool should be called.

API Documentation:
{knowledge}

Generate 2-3 diverse examples of user requests and appropriate tool calls.""",

            "intent": f"""Based on the following tool/API, generate diverse ways users might express
the same intent.

Tool:
{knowledge}

Generate 5 different ways users might request this (formal, casual, with typos, etc).""",

            "nl_to_sql": f"""Based on the following database schema, generate natural language
to SQL query pairs.

Schema:
{knowledge}

Generate 2-3 examples of user questions and corresponding SQL queries.""",
        }

        return base_prompts.get(
            sample_type,
            f"Based on the following knowledge, generate a {sample_type} sample.\n\nKnowledge:\n{knowledge}\n\nGenerate the sample."
        )

    def _parse_response(
        self,
        content: str,
        sample_type: str,
        knowledge: str,
    ) -> dict[str, Any]:
        """Parse teacher response into conversation format.

        Args:
            content: Raw response content.
            sample_type: Type of sample.
            knowledge: Original knowledge text.

        Returns:
            Formatted conversation.
        """
        # Parse based on sample type
        if sample_type == "qa":
            return self._parse_qa(content, knowledge)
        elif sample_type in ("tool_call", "intent", "slot_fill"):
            return self._parse_tool_call(content, knowledge)
        elif sample_type == "nl_to_sql":
            return self._parse_sql(content, knowledge)
        else:
            return self._parse_generic(content, knowledge, sample_type)

    def _parse_qa(self, content: str, knowledge: str) -> dict[str, Any]:
        """Parse Q&A response."""
        lines = content.split("\n")
        question = ""
        answer = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("Question:"):
                question = line[9:].strip()
                current_section = "question"
            elif line.startswith("Answer:"):
                answer = line[7:].strip()
                current_section = "answer"
            elif current_section == "question":
                question += " " + line
            elif current_section == "answer":
                answer += " " + line

        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "sample_type": "qa",
            "source": knowledge[:100],
            "quality_score": 0.8,
        }

    def _parse_tool_call(self, content: str, knowledge: str) -> dict[str, Any]:
        """Parse tool call response."""
        # Simple parsing - extract user requests and tool calls
        return {
            "messages": [
                {"role": "user", "content": "Show me how to use this tool"},
                {"role": "assistant", "content": content},
            ],
            "sample_type": "tool_call",
            "source": knowledge[:100],
            "quality_score": 0.8,
        }

    def _parse_sql(self, content: str, knowledge: str) -> dict[str, Any]:
        """Parse SQL generation response."""
        return {
            "messages": [
                {"role": "user", "content": "Convert to SQL"},
                {"role": "assistant", "content": content},
            ],
            "sample_type": "nl_to_sql",
            "source": knowledge[:100],
            "quality_score": 0.8,
        }

    def _parse_generic(
        self,
        content: str,
        knowledge: str,
        sample_type: str,
    ) -> dict[str, Any]:
        """Parse generic response."""
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
