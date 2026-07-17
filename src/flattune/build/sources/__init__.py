"""Source detectors for the build pipeline.

This module provides source type detection for various file formats.
"""

from flattune.build.sources.markdown import MarkdownSourceDetector
from flattune.build.sources.json import JSONSourceDetector
from flattune.build.sources.jsonl import JSONLSourceDetector
from flattune.build.sources.flatseek import FlatseekSourceDetector
from flattune.build.sources.text import TextSourceDetector

__all__ = [
    "MarkdownSourceDetector",
    "JSONSourceDetector",
    "JSONLSourceDetector",
    "FlatseekSourceDetector",
    "TextSourceDetector",
]
