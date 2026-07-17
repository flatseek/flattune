"""Dataset generators for various task types.

This module provides a plugin-based architecture for dataset generators.
Generators are registered using the @register_generator decorator.
"""

from flattune.dataset.generators.base import BaseGenerator

# Import generators to trigger registration
from flattune.dataset.generators import (
    qa,
    summary,
    classification,
    comparison,
    conversation,
    extraction,
    custom,
    safety,
    facts,
    glossary,
    concept,
    procedure,
)

# Import individual generator classes for backward compatibility
from flattune.dataset.generators.qa import QAGenerator
from flattune.dataset.generators.summary import SummaryGenerator
from flattune.dataset.generators.classification import ClassificationGenerator
from flattune.dataset.generators.comparison import ComparisonGenerator
from flattune.dataset.generators.conversation import ConversationGenerator
from flattune.dataset.generators.extraction import ExtractionGenerator
from flattune.dataset.generators.custom import CustomGenerator
from flattune.dataset.generators.safety import SafetyGenerator
from flattune.dataset.generators.facts import FactsGenerator
from flattune.dataset.generators.glossary import GlossaryGenerator
from flattune.dataset.generators.concept import ConceptGenerator
from flattune.dataset.generators.procedure import ProcedureGenerator

# Legacy dictionary registry for backward compatibility
_LEGACY_GENERATORS = {
    "qa": QAGenerator,
    "summary": SummaryGenerator,
    "classification": ClassificationGenerator,
    "comparison": ComparisonGenerator,
    "conversation": ConversationGenerator,
    "extraction": ExtractionGenerator,
    "custom": CustomGenerator,
    "safety": SafetyGenerator,
    "facts": FactsGenerator,
    "glossary": GlossaryGenerator,
    "concept": ConceptGenerator,
    "procedure": ProcedureGenerator,
}


def get_generator(name: str) -> BaseGenerator:
    """Get a generator by name.

    Args:
        name: Name of the generator.

    Returns:
        Instance of the requested generator.

    Raises:
        ValueError: If generator name is not found.
    """
    # Prefer legacy (dataset) generators over teach generators
    # Teach generators have a different interface (knowledge graph based)
    # while dataset generators work with flat documents
    if name in _LEGACY_GENERATORS:
        return _LEGACY_GENERATORS[name]()

    # Only fall back to teach registry if not found in legacy
    try:
        from flattune.teach.registry import GeneratorRegistry
        cls = GeneratorRegistry.get(name)
        if cls is not None:
            return cls()
    except ImportError:
        pass

    raise ValueError(
        f"Unknown generator: {name}. "
        f"Available: {list(_LEGACY_GENERATORS.keys())}"
    )


def list_generators() -> list[str]:
    """List all available generator names."""
    # Combine new and legacy registries
    try:
        from flattune.teach.registry import GeneratorRegistry
        new_generators = set(GeneratorRegistry.list_plugins())
    except ImportError:
        new_generators = set()

    legacy = set(_LEGACY_GENERATORS.keys())
    return sorted(new_generators | legacy)


__all__ = [
    "BaseGenerator",
    "QAGenerator",
    "SummaryGenerator",
    "ClassificationGenerator",
    "ComparisonGenerator",
    "ConversationGenerator",
    "ExtractionGenerator",
    "CustomGenerator",
    "SafetyGenerator",
    "FactsGenerator",
    "GlossaryGenerator",
    "ConceptGenerator",
    "ProcedureGenerator",
    "get_generator",
    "list_generators",
]
