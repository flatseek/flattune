"""Text parser."""

from collections.abc import Iterator
from pathlib import Path

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.registry import SourceType, register_parser


@register_parser("text")
class TextParser(BaseParser):
    """Parser for plain text files."""

    source_type = SourceType.TEXT

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a text file."""
        path = str(source).lower()
        return path.endswith(".txt") or path.endswith(".text")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse text file into knowledge fragments.

        Yields:
            ParseResult for paragraphs.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
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
            "lines": len(content.split("\n")),
            "chars": len(content),
        }

        # Split by double newlines (paragraphs)
        paragraphs = content.split("\n\n")

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                para_metadata = dict(metadata)
                para_metadata["paragraph"] = i

                yield ParseResult(
                    source=source_str,
                    source_type=self.source_type,
                    content=para,
                    metadata=para_metadata,
                )


@register_parser("csv")
class CSVParser(BaseParser):
    """Parser for CSV files."""

    source_type = SourceType.CSV

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a CSV file."""
        return str(source).lower().endswith(".csv")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse CSV file into knowledge fragments.

        Yields:
            ParseResult for each row.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="",
                metadata={"error": str(e)},
            )
            return

        if not lines:
            return

        # Parse header
        header = [h.strip().strip('"') for h in lines[0].split(",")]
        metadata = {
            "filename": path.name,
            "columns": header,
            "rows": len(lines) - 1,
        }

        # Parse rows
        for i, line in enumerate(lines[1:], start=1):
            values = [v.strip().strip('"') for v in line.split(",")]
            if len(values) != len(header):
                continue

            row_data = dict(zip(header, values, strict=False))

            # Format as structured text
            content = "\n".join(f"{k}: {v}" for k, v in row_data.items() if v)

            row_metadata = dict(metadata)
            row_metadata["row"] = i

            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content=content,
                metadata=row_metadata,
                entities=[
                    {"type": k, "content": v}
                    for k, v in row_data.items() if v and len(v) < 100
                ],
            )
