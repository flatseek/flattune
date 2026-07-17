"""JSONL (JSON Lines) source detector."""

import json
from pathlib import Path
from typing import Any

from flattune.build.registry import (
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
)
from flattune.teach.registry import SourceType


@register_source("jsonl")
class JSONLSourceDetector(BaseSourceDetector):
    """Detector for JSONL (JSON Lines) files."""

    source_type = SourceType.JSONL
    priority = 10

    def can_handle(self, source: str | Path) -> bool:
        """Check if source is a JSONL file."""
        return str(source).lower().endswith(".jsonl")

    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect JSONL file and analyze structure."""
        source_str = str(source)
        path = Path(source_str)

        sample_content = ""
        metadata: dict[str, Any] = {"filename": path.name}

        try:
            with open(path, encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:  # Read first 10 lines for sampling
                        break
                    lines.append(line.strip())

                sample_content = "\n".join(lines)[:1024]
                metadata["sample_lines"] = len(lines)

                # Analyze structure from first valid JSON line
                for line in lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                metadata["keys"] = list(data.keys())[:20]
                                break
                        except json.JSONDecodeError:
                            pass

        except Exception:
            pass

        return SourceDetectionResult(
            source_type=self.source_type,
            confidence=1.0 if sample_content else 0.8,
            detected_format="jsonl",
            sample_content=sample_content,
            metadata=metadata,
        )
