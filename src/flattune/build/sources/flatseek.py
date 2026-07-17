"""FlatSeek source detector."""

from pathlib import Path
from typing import Any

from flattune.build.registry import (
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
)
from flattune.teach.registry import SourceType


@register_source("flatseek")
class FlatseekSourceDetector(BaseSourceDetector):
    """Detector for FlatSeek index files."""

    source_type = SourceType.FLATSEEK
    priority = 10

    def can_handle(self, source: str | Path) -> bool:
        """Check if source is a FlatSeek file or directory."""
        source_str = str(source)
        return (
            source_str.lower().endswith(".fsk")
            or source_str.lower().endswith(".flatseek")
            or Path(source_str).is_dir()
        )

    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect FlatSeek source."""
        source_str = str(source)
        path = Path(source_str)

        metadata: dict[str, Any] = {"filename": path.name if path.suffix else "directory"}

        # Determine if it's a file or directory
        if path.is_file():
            metadata["storage_type"] = "packed_archive"
            try:
                metadata["file_size"] = path.stat().st_size
            except Exception:
                pass
        else:
            metadata["storage_type"] = "directory_index"

        return SourceDetectionResult(
            source_type=self.source_type,
            confidence=0.9,
            detected_format="flatseek",
            sample_content=None,  # Binary or managed by FlatSeek library
            metadata=metadata,
        )
