"""FlatTune Teach Framework - Knowledge & Skill Compiler.

A modular framework for compiling knowledge, software capabilities, APIs,
workflows, databases, SDKs, and documentation into high-quality fine-tuning datasets.
"""

# Import submodules to trigger plugin registration
from flattune.teach import (
    generators,  # noqa: F401
    parsers,  # noqa: F401
    teachers,  # noqa: F401
)
from flattune.teach.knowledge_graph import EdgeType, KnowledgeGraph, KnowledgeNode, NodeType
from flattune.teach.pipeline import (
    PipelineConfig,
    TeachPipeline,
    teach_database,
    teach_knowledge,
    teach_software,
)
from flattune.teach.registry import GeneratorRegistry, ParserRegistry, TeacherRegistry

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
