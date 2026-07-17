"""JSON parser."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.registry import SourceType, register_parser


@register_parser("json")
class JSONParser(BaseParser):
    """Parser for JSON files."""

    source_type = SourceType.JSON

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a JSON file."""
        path = str(source).lower()
        return path.endswith(".json") and not path.endswith(".jsonl")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse JSON file into knowledge fragments.

        Yields:
            ParseResult for each top-level item or array element.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="",
                metadata={"error": str(e)},
            )
            return

        metadata = {
            "filename": path.name,
            "type": type(data).__name__,
        }

        # Handle different JSON structures
        if isinstance(data, dict):
            if "records" in data:
                # Array of records
                for i, record in enumerate(data["records"]):
                    yield self._record_to_result(source_str, record, i, metadata)
            elif "items" in data:
                for i, item in enumerate(data["items"]):
                    yield self._record_to_result(source_str, item, i, metadata)
            else:
                # Single object
                yield self._record_to_result(source_str, data, 0, metadata)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                yield self._record_to_result(source_str, item, i, metadata)

        else:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content=str(data),
                metadata=metadata,
            )

    def _record_to_result(
        self,
        source: str,
        record: Any,
        index: int,
        base_metadata: dict,
    ) -> ParseResult:
        """Convert a record to ParseResult."""
        metadata = dict(base_metadata)
        metadata["index"] = index

        if isinstance(record, dict):
            # Extract meaningful fields
            content_parts = []

            for key, value in record.items():
                if key in ("description", "content", "text", "summary"):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, str) and len(value) < 500:
                    content_parts.append(f"{key}: {value}")

            content = "\n".join(content_parts) if content_parts else json.dumps(record, indent=2)

            # Extract entities
            entities = []
            for key, value in record.items():
                if isinstance(value, str) and len(value) < 100:
                    entities.append(
                        {
                            "type": key,
                            "content": value,
                        }
                    )

            return ParseResult(
                source=source,
                source_type=self.source_type,
                content=content,
                metadata=metadata,
                entities=entities,
            )

        return ParseResult(
            source=source,
            source_type=self.source_type,
            content=str(record),
            metadata=metadata,
        )


@register_parser("jsonl")
class JSONLParser(BaseParser):
    """Parser for JSONL files."""

    source_type = SourceType.JSONL

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a JSONL file."""
        return str(source).lower().endswith(".jsonl")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse JSONL file line by line.

        Yields:
            ParseResult for each JSON object.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            metadata = {"line": i + 1}
                            content = (
                                json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
                            )

                            yield ParseResult(
                                source=source_str,
                                source_type=self.source_type,
                                content=content,
                                metadata=metadata,
                            )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="",
                metadata={"error": str(e)},
            )
