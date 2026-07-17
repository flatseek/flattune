"""Build pipeline for FlatTune.

This module provides an intelligent dataset planning system that:
- Auto-detects source type and suggests dataset types
- Presents a build plan before generation
- Supports interactive approval or --yes for automation
- Is fully extensible via registries for third-party plugins
"""

from flattune.build.analyzer import AnalysisResult, ContentAnalyzer
from flattune.build.approval import ApprovalResult, InteractiveApproval
from flattune.build.exporter import BuildExporter
from flattune.build.pipeline import BuildPipeline
from flattune.build.planner import BuildPlan, BuildPlanner, TypeSelection
from flattune.build.registry import (
    BaseSourceDetector,
    CategoryRegistry,
    DatasetCategory,
    DatasetType,
    DatasetTypeRegistry,
    SourceDetectionResult,
    SourceRegistry,
    register_category,
    register_dataset_type,
    register_source,
)
from flattune.build.source import SourceDetector

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
