"""Base teacher class for generating training samples."""

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from flattune.teach.knowledge_graph import KnowledgeNode


class BaseTeacher:
    """Abstract base class for teacher models.

    Teachers are responsible for generating high-quality conversations
    from knowledge nodes using LLM inference.
    """

    name: str = "base"
    supports_local: bool = False
    supports_remote: bool = False

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize teacher.

        Args:
            model: Model name/identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            api_key: API key for remote inference.
            base_url: Base URL for API (for custom endpoints).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def generate(
        self,
        knowledge: str | KnowledgeNode,
        sample_type: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a training sample from knowledge.

        Args:
            knowledge: Knowledge to generate from.
            sample_type: Type of sample (qa, summarization, tool_call, etc).
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary containing generated conversation.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        knowledge_iter: Iterator[KnowledgeNode],
        sample_type: str,
        **kwargs,
    ) -> Iterator[dict[str, Any]]:
        """Generate multiple training samples.

        Args:
            knowledge_iter: Iterator of knowledge nodes.
            sample_type: Type of sample to generate.
            **kwargs: Additional generation parameters.

        Yields:
            Generated conversation dictionaries.
        """
        pass

    def supports(self, sample_type: str) -> bool:
        """Check if teacher supports a sample type.

        Args:
            sample_type: Type of sample to check.

        Returns:
            True if supported.
        """
        return True

    def get_system_prompt(self, sample_type: str) -> str:
        """Get system prompt for a sample type.

        Args:
            sample_type: Type of sample.

        Returns:
            System prompt string.
        """
        prompts = {
            "qa": "You are an expert educator who creates clear, accurate Q&A pairs. "
                  "Generate a question and comprehensive answer based on the provided knowledge. "
                  "The answer should be informative yet concise.",
            "summarization": "You are an expert at creating concise summaries. "
                             "Generate a summary of the provided content.",
            "extraction": "You are a data extraction expert. "
                          "Extract key information from the provided content.",
            "comparison": "You are an expert analyst. "
                         "Compare and contrast the presented concepts.",
            "reasoning": "You are an expert reasoning assistant. "
                         "Provide step-by-step reasoning based on the content.",
            "tool_call": "You are an expert API documentation writer. "
                         "Generate tool calling examples based on the provided API knowledge.",
            "intent": "You are an expert at understanding user intent. "
                      "Generate varied ways users might request the same action.",
            "slot_fill": "You are an expert at parameter extraction. "
                         "Generate examples of user requests with implied parameters.",
            "nl_to_sql": "You are an expert SQL developer. "
                         "Generate natural language to SQL queries.",
        }

        return prompts.get(
            sample_type,
            f"You are an expert educator. Generate educational content for: {sample_type}"
        )
