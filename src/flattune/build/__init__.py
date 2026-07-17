"""Build pipeline for FlatTune.

This module provides an intelligent dataset planning system that:
- Auto-detects source type and suggests dataset types
- Presents a build plan before generation
- Supports interactive approval or --yes for automation
- Is fully extensible via registries for third-party plugins
"""

from flattune.build.registry import (
    DatasetCategory,
    DatasetType,
    CategoryRegistry,
    DatasetTypeRegistry,
    SourceRegistry,
    BaseSourceDetector,
    SourceDetectionResult,
    register_source,
    register_category,
    register_dataset_type,
)
from flattune.build.source import SourceDetector
from flattune.build.analyzer import ContentAnalyzer, AnalysisResult
from flattune.build.planner import BuildPlanner, BuildPlan, TypeSelection
from flattune.build.approval import InteractiveApproval, ApprovalResult
from flattune.build.pipeline import BuildPipeline
from flattune.build.exporter import BuildExporter

__all__ = [
    # Registry classes
    "CategoryRegistry",
    "DatasetTypeRegistry",
    "SourceRegistry",
    "DatasetCategory",
    "DatasetType",
    "BaseSourceDetector",
    "SourceDetectionResult",
    # Decorators
    "register_source",
    "register_category",
    "register_dataset_type",
    # Core classes
    "SourceDetector",
    "ContentAnalyzer",
    "AnalysisResult",
    "BuildPlanner",
    "BuildPlan",
    "TypeSelection",
    "InteractiveApproval",
    "ApprovalResult",
    "BuildPipeline",
    "BuildExporter",
]
