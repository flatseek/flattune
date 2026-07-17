"""Hallucination detection for generated samples."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detects potential hallucinations in generated samples."""

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        enable_self_verification: bool = True,
    ):
        """Initialize hallucination detector.

        Args:
            confidence_threshold: Threshold for hallucination detection.
            enable_self_verification: Use self-verification prompts.
        """
        self.confidence_threshold = confidence_threshold
        self.enable_self_verification = enable_self_verification

        # Hallucination indicator patterns
        self._uncertainty_patterns = [
            r"\bI think\b",
            r"\bI believe\b",
            r"\bmaybe\b",
            r"\bperhaps\b",
            r"\bmight be\b",
            r"\bcould be\b",
            r"\bnot sure\b",
            r"\bI don't know\b",
            r"\bunsure\b",
            r"\buncertain\b",
        ]

        self._fabrication_patterns = [
            r"\baccording to\b(?!\s+(me|my|the\s+\w+\s+I))",
            r"\bresearch shows\b",
            r"\bstudies indicate\b",
            r"\bit is known\b",
            r"\beveryone knows\b",
            r"\bthe fact is\b",
            r"\bdefinitely\b",
            r"\bcertainly\b",
            r"\babsolutely\b",
        ]

    def detect(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Detect hallucinations in a sample.

        Args:
            sample: Sample to check.

        Returns:
            Dictionary with hallucination detection results.
        """
        messages = sample.get("messages", [])
        if not messages:
            return {"is_hallucination": False, "confidence": 0.0, "warnings": []}

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        warnings = []
        confidence = 1.0

        for msg in assistant_msgs:
            content = msg.get("content", "")

            # Check for uncertainty markers
            for pattern in self._uncertainty_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    warnings.append(f"Contains hedge language: '{pattern}'")
                    confidence *= 0.9

            # Check for unsubstantiated claims
            for pattern in self._fabrication_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    warnings.append(f"Contains assertion without citation: '{matches[0]}'")
                    confidence *= 0.85

            # Check for specific hallucination types
            warnings.extend(self._check_specific_hallucinations(content))

        is_hallucination = confidence < self.confidence_threshold

        return {
            "is_hallucination": is_hallucination,
            "confidence": confidence,
            "warnings": warnings,
        }

    def _check_specific_hallucinations(self, content: str) -> list[str]:
        """Check for specific types of hallucinations.

        Args:
            content: Text content.

        Returns:
            List of warning messages.
        """
        warnings = []

        # Check for very long numbers that might be fabricated
        numbers = re.findall(r"\b\d{6,}\b", content)
        for num in numbers:
            # Large round numbers are suspicious
            if num.endswith("000"):
                warnings.append(f"Suspicious large number: {num}")

        # Check for specific entities that might be hallucinated
        # This is a simplified check - production would use NER
        # e.g., "Dr. Smith" without clear context

        return warnings

    def is_safe(self, sample: dict[str, Any]) -> bool:
        """Check if sample is safe (no hallucinations detected).

        Args:
            sample: Sample to check.

        Returns:
            True if sample passes hallucination checks.
        """
        result = self.detect(sample)
        return not result["is_hallucination"]
