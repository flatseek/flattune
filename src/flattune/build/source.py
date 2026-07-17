"""Source detection for the build pipeline."""

from pathlib import Path
from typing import Iterator, Union

from flattune.build.registry import SourceDetectionResult, SourceRegistry
from flattune.teach.registry import SourceType


class SourceDetector:
    """Detects source type from file path or URL.

    Detection order:
    1. Check registered SourceDetectors via SourceRegistry
    2. Check file extension
    3. Sample content for heuristics
    """

    # Extension-based fallback map
    EXTENSION_MAP = {
        ".md": SourceType.MARKDOWN,
        ".markdown": SourceType.MARKDOWN,
        ".txt": SourceType.TEXT,
        ".pdf": SourceType.PDF,
        ".html": SourceType.HTML,
        ".htm": SourceType.HTML,
        ".csv": SourceType.CSV,
        ".json": SourceType.JSON,
        ".jsonl": SourceType.JSONL,
        ".yaml": SourceType.YAML,
        ".yml": SourceType.YAML,
        ".fsk": SourceType.FLATSEEK,
        ".flatseek": SourceType.FLATSEEK,
        ".openapi": SourceType.OPENAPI,
        ".swagger": SourceType.SWAGGER,
        ".xml": SourceType.HTML,  # XML treated as HTML-like
    }

    def detect(self, source: Union[str, Path]) -> SourceDetectionResult:
        """Detect source type from path or URL.

        Args:
            source: Path or URL to detect.

        Returns:
            SourceDetectionResult with detected type and confidence.
        """
        source_str = str(source)

        # 1. Try registered detectors first
        detector_cls = SourceRegistry.get_for_source(source_str)
        if detector_cls:
            detector = detector_cls()
            return detector.detect(source_str)

        # 2. URL-based detection
        if source_str.startswith("http"):
            if "openapi" in source_str.lower():
                return SourceDetectionResult(source_type=SourceType.OPENAPI, confidence=0.9)
            if "swagger" in source_str.lower():
                return SourceDetectionResult(source_type=SourceType.SWAGGER, confidence=0.9)
            if "github" in source_str.lower() or "gitlab" in source_str.lower():
                # Try to detect file type from URL
                if source_str.endswith(".json"):
                    return SourceDetectionResult(source_type=SourceType.JSON, confidence=0.7)
                if source_str.endswith(".yaml") or source_str.endswith(".yml"):
                    return SourceDetectionResult(source_type=SourceType.YAML, confidence=0.7)

        # 3. Extension-based fallback
        ext = Path(source_str).suffix.lower()
        if ext in self.EXTENSION_MAP:
            return SourceDetectionResult(
                source_type=self.EXTENSION_MAP[ext],
                confidence=0.8,
                detected_format=ext.lstrip("."),
            )

        # 4. Try to read and detect as text
        path = Path(source_str)
        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(512)
                    # Heuristics for text content
                    if content:
                        # Contains mostly printable characters
                        printable_ratio = sum(c.isprintable() or c in '\n\r\t' for c in content) / len(content)
                        if printable_ratio > 0.9:
                            return SourceDetectionResult(
                                source_type=SourceType.TEXT,
                                confidence=0.5,
                                detected_format="text",
                                sample_content=content[:200],
                            )
            except (UnicodeDecodeError, PermissionError, OSError):
                pass

        # Default to text with low confidence
        return SourceDetectionResult(
            source_type=SourceType.TEXT,
            confidence=0.3,
            detected_format="unknown",
        )

    def detect_multiple(self, sources: list[Union[str, Path]]) -> list[SourceDetectionResult]:
        """Detect multiple sources.

        Args:
            sources: List of paths or URLs to detect.

        Returns:
            List of SourceDetectionResult for each source.
        """
        return [self.detect(s) for s in sources]

    def detect_from_directory(self, directory: Union[str, Path]) -> list[SourceDetectionResult]:
        """Detect all files in a directory.

        Args:
            directory: Directory path.

        Returns:
            List of SourceDetectionResult for each detected file.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return [self.detect(directory)]

        results = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                # Skip hidden files
                if file_path.name.startswith("."):
                    continue
                results.append(self.detect(file_path))

        return results
