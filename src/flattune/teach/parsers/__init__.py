"""Parser implementations for various source types."""

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.parsers.flatseek import FlatseekParser
from flattune.teach.parsers.json_parser import JSONParser
from flattune.teach.parsers.markdown import MarkdownParser
from flattune.teach.parsers.openapi import OpenAPIParser
from flattune.teach.parsers.text import TextParser

__all__ = [
    "BaseParser",
    "ParseResult",
    "MarkdownParser",
    "JSONParser",
    "TextParser",
    "FlatseekParser",
    "OpenAPIParser",
]
