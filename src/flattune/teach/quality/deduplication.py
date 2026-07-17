"""Deduplication for generated samples."""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Deduplicator:
    """Deduplicates generated samples using various strategies."""

    def __init__(
        self,
        enable_exact: bool = True,
        enable_semantic: bool = True,
        similarity_threshold: float = 0.95,
    ):
        """Initialize deduplicator.

        Args:
            enable_exact: Enable exact match deduplication.
            enable_semantic: Enable semantic deduplication.
            similarity_threshold: Threshold for semantic similarity.
        """
        self.enable_exact = enable_exact
        self.enable_semantic = enable_semantic
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: set[str] = set()
        self._seen_texts: list[str] = []

    def is_duplicate(self, sample: dict[str, Any]) -> bool:
        """Check if sample is a duplicate.

        Args:
            sample: Sample to check.

        Returns:
            True if sample is a duplicate.
        """
        # Extract text content for comparison
        messages = sample.get("messages", [])
        text = self._extract_text(messages)

        # Exact deduplication
        if self.enable_exact:
            if self._is_exact_duplicate(text):
                return True

        # Semantic deduplication
        if self.enable_semantic:
            if self._is_semantic_duplicate(text):
                return True

        return False

    def _extract_text(self, messages: list[dict[str, str]]) -> str:
        """Extract text from messages.

        Args:
            messages: List of message dictionaries.

        Returns:
            Concatenated text content.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _is_exact_duplicate(self, text: str) -> bool:
        """Check for exact duplicate.

        Args:
            text: Text to check.

        Returns:
            True if exact duplicate.
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self._seen_hashes:
            logger.debug("Exact duplicate detected")
            return True

        self._seen_hashes.add(text_hash)
        return False

    def _is_semantic_duplicate(self, text: str) -> bool:
        """Check for semantic duplicate.

        Args:
            text: Text to check.

        Returns:
            True if semantically duplicate.
        """
        # Simple implementation: check for very similar texts
        # In production, use embeddings for better similarity detection
        text_lower = text.lower().strip()

        for seen in self._seen_texts:
            seen_lower = seen.lower().strip()
            similarity = self._jaccard_similarity(text_lower, seen_lower)

            if similarity >= self.similarity_threshold:
                logger.debug(f"Semantic duplicate detected (similarity: {similarity:.2f})")
                return True

        self._seen_texts.append(text)
        return False

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0-1).
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def reset(self) -> None:
        """Reset deduplication state."""
        self._seen_hashes.clear()
        self._seen_texts.clear()
