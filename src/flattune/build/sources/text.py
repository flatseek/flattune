"""Text source detector."""

from pathlib import Path
from typing import Any

from flattune.build.registry import (
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
)
from flattune.teach.registry import SourceType


@register_source("text")
class TextSourceDetector(BaseSourceDetector):
    """Detector for plain text files."""

    source_type = SourceType.TEXT
    priority = 50  # Lower priority than specific detectors

    def can_handle(self, source: str | Path) -> bool:
        """Check if source is a text file."""
        source_str = str(source)
        # Handle .txt files or files without extension
        if source_str.lower().endswith(".txt"):
            return True
        # Files without extension that are not recognized by other detectors
        path = Path(source_str)
        if "." not in path.name:
            try:
                with open(path, encoding="utf-8") as f:
                    f.read(100)
                return True
            except (UnicodeDecodeError, PermissionError):
                return False
        return False

    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect plain text file."""
        source_str = str(source)
        path = Path(source_str)

        sample_content = ""
        metadata: dict[str, Any] = {"filename": path.name}

        try:
            with open(path, encoding="utf-8") as f:
                sample_content = f.read(1024)
                # Try to detect encoding
                try:
                    sample_content.encode("utf-8")
                    metadata["encoding"] = "utf-8"
                except UnicodeEncodeError:
                    metadata["encoding"] = "unknown"
        except Exception:
            pass

        return SourceDetectionResult(
            source_type=self.source_type,
            confidence=0.7 if sample_content else 0.5,
            detected_format="text",
            sample_content=sample_content[:500],
            metadata=metadata,
        )
