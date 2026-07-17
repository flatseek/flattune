"""FlatSeek index parser."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.registry import SourceType, register_parser


@register_parser("flatseek")
class FlatseekParser(BaseParser):
    """Parser for FlatSeek index files."""

    source_type = SourceType.FLATSEEK

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a FlatSeek index file."""
        return str(source).endswith(".fsk")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse FlatSeek index into knowledge fragments.

        Yields:
            ParseResult for each indexed record.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            # FlatSeek .fsk is typically a JSON array or NDJSON
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()

            # Try parsing as JSON array first
            if content.startswith("["):
                records = json.loads(content)
            else:
                # Try NDJSON (newline-delimited JSON)
                records = []
                for line in content.split("\n"):
                    if line.strip():
                        records.append(json.loads(line))

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
            "records": len(records),
        }

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            record_metadata = dict(metadata)
            record_metadata["index"] = i

            # Extract content
            content = self._extract_content(record)
            if not content:
                continue

            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content=content,
                metadata=record_metadata,
                entities=self._extract_entities(record),
            )

    def _extract_content(self, record: dict) -> str:
        """Extract meaningful content from record."""
        # Try common content fields
        for field in ("content", "text", "body", "description", "title"):
            if field in record and record[field]:
                return str(record[field])

        # Fall back to all string fields
        parts = []
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 10:
                parts.append(f"{key}: {value}")

        return "\n".join(parts) if parts else json.dumps(record)

    def _extract_entities(self, record: dict) -> list[dict[str, Any]]:
        """Extract entities from record."""
        entities = []

        for key, value in record.items():
            if isinstance(value, str) and len(value) < 200:
                entities.append(
                    {
                        "type": key,
                        "content": value,
                    }
                )
            elif isinstance(value, (int, float, bool)):
                entities.append(
                    {
                        "type": key,
                        "content": str(value),
                    }
                )

        return entities
