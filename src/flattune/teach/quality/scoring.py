"""Quality scoring for generated samples."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores generated samples based on various quality metrics."""

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 4096,
        require_complete_sentences: bool = True,
    ):
        """Initialize quality scorer.

        Args:
            min_length: Minimum acceptable text length.
            max_length: Maximum acceptable text length.
            require_complete_sentences: Require complete sentences.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_complete_sentences = require_complete_sentences

    def score(self, sample: dict[str, Any]) -> float:
        """Score a generated sample.

        Args:
            sample: Sample to score.

        Returns:
            Quality score (0-1).
        """
        messages = sample.get("messages", [])
        if not messages:
            return 0.0

        # Calculate scores for each metric
        length_score = self._score_length(messages)
        diversity_score = self._score_diversity(messages)
        completeness_score = self._score_completeness(messages)
        format_score = self._score_format(messages)

        # Weighted average
        final_score = (
            length_score * 0.2 +
            diversity_score * 0.2 +
            completeness_score * 0.3 +
            format_score * 0.3
        )

        return min(1.0, max(0.0, final_score))

    def _score_length(self, messages: list[dict[str, str]]) -> float:
        """Score based on response length appropriateness.

        Args:
            messages: Message list.

        Returns:
            Length score (0-1).
        """
        total_length = sum(len(msg.get("content", "")) for msg in messages)

        # Ideal range check
        if total_length < self.min_length:
            return total_length / self.min_length
        if total_length > self.max_length:
            return max(0, 1 - (total_length - self.max_length) / 1000)

        return 1.0

    def _score_diversity(self, messages: list[dict[str, str]]) -> float:
        """Score based on content diversity.

        Args:
            messages: Message list.

        Returns:
            Diversity score (0-1).
        """
        if len(messages) < 2:
            return 1.0

        # Check for repeated content
        contents = [msg.get("content", "") for msg in messages]
        unique_contents = set(contents)

        # Penalize excessive repetition
        diversity = len(unique_contents) / len(contents)

        # Check for repeated phrases
        all_words = set()
        for content in contents:
            words = set(content.lower().split())
            all_words.update(words)

        # Penalize very short or repetitive content
        avg_words_per_msg = sum(len(c.split()) for c in contents) / len(contents)

        if avg_words_per_msg < 5:
            return 0.3

        return diversity

    def _score_completeness(self, messages: list[dict[str, str]]) -> float:
        """Score based on completeness of responses.

        Args:
            messages: Message list.

        Returns:
            Completeness score (0-1).
        """
        if not messages:
            return 0.0

        # Check if assistant provided responses
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            return 0.0

        # Check for incomplete indicators
        incomplete_indicators = ["...", "to be continued", "[incomplete]", "###"]
        incomplete_count = 0

        for msg in assistant_msgs:
            content = msg.get("content", "").lower()
            for indicator in incomplete_indicators:
                if indicator.lower() in content:
                    incomplete_count += 1
                    break

        completeness = 1 - (incomplete_count / len(assistant_msgs))

        # Require complete sentences
        if self.require_complete_sentences:
            for msg in assistant_msgs:
                content = msg.get("content", "").strip()
                if content and not content.endswith(('.', '!', '?', '"', "'")):
                    # Allow some flexibility
                    pass

        return completeness

    def _score_format(self, messages: list[dict[str, str]]) -> float:
        """Score based on format correctness.

        Args:
            messages: Message list.

        Returns:
            Format score (0-1).
        """
        if not messages:
            return 0.0

        # Check for valid roles
        valid_roles = {"system", "user", "assistant", "tool"}
        invalid_count = 0

        for msg in messages:
            role = msg.get("role", "")
            if role not in valid_roles:
                invalid_count += 1

            # Check for content
            if "content" not in msg:
                invalid_count += 1

        format_score = 1 - (invalid_count / (len(messages) * 2))

        return max(0.0, format_score)

    def get_scores(self, sample: dict[str, Any]) -> dict[str, float]:
        """Get detailed scores for a sample.

        Args:
            sample: Sample to score.

        Returns:
            Dictionary with individual metric scores.
        """
        messages = sample.get("messages", [])

        return {
            "length": self._score_length(messages),
            "diversity": self._score_diversity(messages),
            "completeness": self._score_completeness(messages),
            "format": self._score_format(messages),
            "overall": self.score(sample),
        }
