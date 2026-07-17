"""Build planning for the build pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from flattune.build.analyzer import AnalysisResult
from flattune.build.registry import (
    DatasetTypeRegistry,
    SourceDetectionResult,
)
from flattune.teach.registry import SourceType


@dataclass
class TypeSelection:
    """A selected dataset type with planning metadata."""
    type_name: str
    confidence: float
    estimated_samples: int
    generator_name: str
    user_selected: bool = False


@dataclass
class BuildPlan:
    """Complete build plan for user approval."""
    source: str
    source_type: SourceType
    selected_types: list[TypeSelection] = field(default_factory=list)
    total_estimated_samples: int = 0
    output_structure: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_display_dict(self) -> dict[str, Any]:
        """Format for interactive display."""
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "types": [
                {
                    "name": t.type_name,
                    "confidence": f"{t.confidence:.0%}",
                    "confidence_raw": t.confidence,
                    "estimated": t.estimated_samples,
                    "generator": t.generator_name,
                    "selected": t.user_selected,
                }
                for t in self.selected_types
            ],
            "total_samples": self.total_estimated_samples,
            "output_structure": self.output_structure,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_display_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildPlan:
        """Create BuildPlan from dictionary."""
        return cls(
            source=data["source"],
            source_type=SourceType(data["source_type"]),
            selected_types=[
                TypeSelection(
                    type_name=t["name"],
                    confidence=t["confidence_raw"],
                    estimated_samples=t["estimated"],
                    generator_name=t["generator"],
                    user_selected=t["selected"],
                )
                for t in data.get("types", [])
            ],
            total_estimated_samples=data.get("total_samples", 0),
            output_structure=data.get("output_structure", {}),
        )


class BuildPlanner:
    """Creates build plans from analysis results."""

    def __init__(self, min_confidence_threshold: float = 0.4):
        """Initialize build planner.

        Args:
            min_confidence_threshold: Minimum confidence to auto-select a type.
        """
        self.min_confidence_threshold = min_confidence_threshold

    def create_plan(
        self,
        detection_result: SourceDetectionResult,
        analysis_result: AnalysisResult,
        document_count: int,
        user_requested_types: list[str] | None = None,
        force_types: list[str] | None = None,
    ) -> BuildPlan:
        """Create a build plan from analysis.

        Args:
            detection_result: Source detection result.
            analysis_result: Content analysis result.
            document_count: Total documents in source.
            user_requested_types: Optional explicit types from config.
            force_types: Optional types to force include (--types CLI flag).
        """
        selected_types: list[TypeSelection] = []

        # If types specified via CLI --types, use those exclusively
        if force_types:
            for type_name in force_types:
                dt = DatasetTypeRegistry.get(type_name)
                if dt:
                    selected_types.append(TypeSelection(
                        type_name=type_name,
                        confidence=1.0,
                        estimated_samples=int(document_count * dt.estimated_samples_per_doc),
                        generator_name=dt.generator_name,
                        user_selected=True,
                    ))
        # If user specified types via config, use those
        elif user_requested_types:
            for type_name in user_requested_types:
                dt = DatasetTypeRegistry.get(type_name)
                if dt:
                    selected_types.append(TypeSelection(
                        type_name=type_name,
                        confidence=1.0,
                        estimated_samples=int(document_count * dt.estimated_samples_per_doc),
                        generator_name=dt.generator_name,
                        user_selected=True,
                    ))
                else:
                    # Try to find generator for unknown type
                    from flattune.dataset.generators import get_generator
                    try:
                        get_generator(type_name)
                        selected_types.append(TypeSelection(
                            type_name=type_name,
                            confidence=1.0,
                            estimated_samples=int(document_count * 2.0),
                            generator_name=type_name,
                            user_selected=True,
                        ))
                    except ValueError:
                        pass
        else:
            # Auto-select based on analysis confidence
            for type_name, confidence in analysis_result.suggested_types:
                if confidence >= self.min_confidence_threshold:
                    dt = DatasetTypeRegistry.get(type_name)
                    if dt:
                        selected_types.append(TypeSelection(
                            type_name=type_name,
                            confidence=confidence,
                            estimated_samples=int(document_count * dt.estimated_samples_per_doc),
                            generator_name=dt.generator_name,
                            user_selected=False,
                        ))

        # Build output structure by category
        output_structure: dict[str, list[str]] = {}
        for sel in selected_types:
            dt = DatasetTypeRegistry.get(sel.type_name)
            if dt:
                if dt.category not in output_structure:
                    output_structure[dt.category] = []
                if sel.type_name not in output_structure[dt.category]:
                    output_structure[dt.category].append(sel.type_name)

        # Calculate average confidence
        avg_confidence = (
            sum(s.confidence for s in selected_types) / len(selected_types)
            if selected_types else 0.0
        )

        return BuildPlan(
            source=str(detection_result),
            source_type=detection_result.source_type,
            selected_types=selected_types,
            total_estimated_samples=sum(s.estimated_samples for s in selected_types),
            output_structure=output_structure,
            metadata={
                "document_count": document_count,
                "analysis_confidence": avg_confidence,
                "source_format": detection_result.detected_format,
            },
        )

    def modify_plan(
        self,
        plan: BuildPlan,
        selected_type_names: list[str],
    ) -> BuildPlan:
        """Create a modified plan with user-selected types.

        Args:
            plan: Original build plan.
            selected_type_names: User-selected type names.

        Returns:
            Modified build plan.
        """
        selected_types = []
        for type_name in selected_type_names:
            # Find in original or create new
            original = next((t for t in plan.selected_types if t.type_name == type_name), None)
            if original:
                selected_types.append(original)
            else:
                # Create new from registry
                dt = DatasetTypeRegistry.get(type_name)
                if dt:
                    selected_types.append(TypeSelection(
                        type_name=type_name,
                        confidence=0.5,  # Lower since not analyzed
                        estimated_samples=plan.metadata.get("document_count", 0) * dt.estimated_samples_per_doc,
                        generator_name=dt.generator_name,
                        user_selected=True,
                    ))

        # Rebuild output structure
        output_structure: dict[str, list[str]] = {}
        for sel in selected_types:
            dt = DatasetTypeRegistry.get(sel.type_name)
            if dt:
                if dt.category not in output_structure:
                    output_structure[dt.category] = []
                if sel.type_name not in output_structure[dt.category]:
                    output_structure[dt.category].append(sel.type_name)

        return BuildPlan(
            source=plan.source,
            source_type=plan.source_type,
            selected_types=selected_types,
            total_estimated_samples=sum(s.estimated_samples for s in selected_types),
            output_structure=output_structure,
            metadata=plan.metadata,
        )
