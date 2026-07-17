"""FlatTune Teach Framework - Knowledge & Skill Compiler.

A modular framework for compiling knowledge, software capabilities, APIs,
workflows, databases, SDKs, and documentation into high-quality fine-tuning datasets.
"""

# Import submodules to trigger plugin registration
from flattune.teach import parsers  # noqa: F401
from flattune.teach import generators  # noqa: F401
from flattune.teach import teachers  # noqa: F401

from flattune.teach.registry import ParserRegistry, TeacherRegistry, GeneratorRegistry
from flattune.teach.knowledge_graph import KnowledgeGraph, KnowledgeNode, NodeType, EdgeType
from flattune.teach.pipeline import TeachPipeline, PipelineConfig, teach_knowledge, teach_software, teach_database

__all__ = [
    # Registries
    "ParserRegistry",
    "TeacherRegistry",
    "GeneratorRegistry",
    # Knowledge Graph
    "KnowledgeGraph",
    "KnowledgeNode",
    "NodeType",
    "EdgeType",
    # Pipeline
    "TeachPipeline",
    "PipelineConfig",
    # Convenience functions
    "teach_knowledge",
    "teach_software",
    "teach_database",
]
