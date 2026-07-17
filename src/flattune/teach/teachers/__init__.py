"""Teacher model implementations."""

from flattune.teach.teachers.anthropic import AnthropicTeacher
from flattune.teach.teachers.base import BaseTeacher
from flattune.teach.teachers.ollama import OllamaTeacher
from flattune.teach.teachers.openai import OpenAITeacher

__all__ = [
    "BaseTeacher",
    "OpenAITeacher",
    "AnthropicTeacher",
    "OllamaTeacher",
]
