"""Markdown parser."""

import re
from pathlib import Path
from typing import Iterator

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.registry import SourceType, register_parser


@register_parser("markdown")
class MarkdownParser(BaseParser):
    """Parser for Markdown documents."""

    source_type = SourceType.MARKDOWN

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is a Markdown file."""
        path = str(source).lower()
        return path.endswith(".md") or path.endswith(".markdown")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse Markdown file into knowledge fragments.

        Yields:
            ParseResult for each section/paragraph.
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="",
                metadata={"error": str(e)},
            )
            return

        # Extract metadata
        metadata = {
            "filename": path.name,
            "title": self._extract_title(content),
        }

        # Split into sections
        sections = self._split_into_sections(content)

        for i, (title, section_content) in enumerate(sections):
            if section_content.strip():
                section_metadata = dict(metadata)
                section_metadata["section"] = title
                section_metadata["section_index"] = i

                yield ParseResult(
                    source=source_str,
                    source_type=self.source_type,
                    content=section_content.strip(),
                    metadata=section_metadata,
                )

    def _extract_title(self, content: str) -> str:
        """Extract title from Markdown content."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _split_into_sections(self, content: str) -> list[tuple[str, str]]:
        """Split Markdown into sections by headers."""
        sections = []
        current_title = ""
        current_content = []

        lines = content.split("\n")
        for line in lines:
            # Check for headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))

                # Start new section
                level = len(header_match.group(1))
                current_title = header_match.group(2).strip()
                current_content = [line]
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections
