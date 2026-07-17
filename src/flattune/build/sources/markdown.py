"""Markdown source detector."""

from pathlib import Path
from typing import Any

from flattune.build.registry import (
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
)
from flattune.teach.registry import SourceType


@register_source("markdown")
class MarkdownSourceDetector(BaseSourceDetector):
    """Detector for Markdown files."""

    source_type = SourceType.MARKDOWN
    priority = 10

    def can_handle(self, source: str | Path) -> bool:
        """Check if source is a Markdown file."""
        return str(source).lower().endswith((".md", ".markdown"))

    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect Markdown file and read sample content."""
        source_str = str(source)
        path = Path(source_str)

        sample_content = ""
        metadata: dict[str, Any] = {"filename": path.name}

        try:
            with open(path, encoding="utf-8") as f:
                sample_content = f.read(1024)
                # Extract frontmatter if present
                if sample_content.startswith("---"):
                    parts = sample_content.split("---", 2)
                    if len(parts) >= 3:
                        metadata["has_frontmatter"] = True
                        sample_content = parts[2].strip()
        except Exception:
            pass

        return SourceDetectionResult(
            source_type=self.source_type,
            confidence=1.0 if sample_content else 0.8,
            detected_format="markdown",
            sample_content=sample_content[:500],
            metadata=metadata,
        )
