"""JSON source detector."""

import json
from pathlib import Path
from typing import Any

from flattune.build.registry import (
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
)
from flattune.teach.registry import SourceType


@register_source("json")
class JSONSourceDetector(BaseSourceDetector):
    """Detector for JSON files."""

    source_type = SourceType.JSON
    priority = 10

    def can_handle(self, source: str | Path) -> bool:
        """Check if source is a JSON file."""
        return str(source).lower().endswith(".json") and not str(source).lower().endswith(".jsonl")

    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect JSON file and analyze structure."""
        source_str = str(source)
        path = Path(source_str)

        sample_content = ""
        metadata: dict[str, Any] = {"filename": path.name}

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read(2048)
                sample_content = content[:1024]

                # Parse to get structure info
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        metadata["keys"] = list(data.keys())[:20]
                        metadata["type"] = "object"
                    elif isinstance(data, list):
                        metadata["length"] = len(data)
                        metadata["type"] = "array"
                        if data:
                            metadata["item_keys"] = (
                                list(data[0].keys())
                                if isinstance(data[0], dict)
                                else type(data[0]).__name__
                            )
                except json.JSONDecodeError:
                    metadata["parse_error"] = True

        except Exception:
            pass

        return SourceDetectionResult(
            source_type=self.source_type,
            confidence=1.0 if sample_content else 0.8,
            detected_format="json",
            sample_content=sample_content,
            metadata=metadata,
        )
