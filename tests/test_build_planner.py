"""Tests for build planning."""

from __future__ import annotations

import pytest

from flattune.build.planner import BuildPlanner, BuildPlan, TypeSelection
from flattune.build.registry import SourceDetectionResult, DatasetTypeRegistry
from flattune.build.analyzer import AnalysisResult
from flattune.teach.registry import SourceType


class TestBuildPlanner:
    """Test BuildPlanner class."""

    def test_create_plan_auto_selection(self):
        """Test auto-selection based on analysis."""
        planner = BuildPlanner(min_confidence_threshold=0.4)

        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
            detected_format="markdown",
        )

        analysis = AnalysisResult(
            source_type=SourceType.MARKDOWN,
            content_summary="Markdown with code",
            suggested_categories=[("knowledge", 0.7), ("qa", 0.6)],
            suggested_types=[
                ("facts", 0.7),
                ("context_qa", 0.6),
                ("glossary", 0.5),
            ],
            content_statistics={"total_characters": 500},
        )

        plan = planner.create_plan(
            detection_result=detection,
            analysis_result=analysis,
            document_count=10,
        )

        assert isinstance(plan, BuildPlan)
        assert plan.source_type == SourceType.MARKDOWN
        assert len(plan.selected_types) >= 2  # At least facts and context_qa
        assert plan.total_estimated_samples > 0

    def test_create_plan_with_force_types(self):
        """Test creating plan with forced types (--types flag)."""
        planner = BuildPlanner()

        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        analysis = AnalysisResult(
            source_type=SourceType.MARKDOWN,
            content_summary="Test",
            suggested_categories=[],
            suggested_types=[],
            content_statistics={"total_characters": 100},
        )

        plan = planner.create_plan(
            detection_result=detection,
            analysis_result=analysis,
            document_count=5,
            force_types=["facts", "context_qa"],
        )

        type_names = [t.type_name for t in plan.selected_types]
        assert "facts" in type_names
        assert "context_qa" in type_names
        # All should be user selected
        assert all(t.user_selected for t in plan.selected_types)

    def test_create_plan_with_user_requested_types(self):
        """Test creating plan with user-requested types from config."""
        planner = BuildPlanner()

        detection = SourceDetectionResult(
            source_type=SourceType.JSON,
            confidence=0.8,
        )

        analysis = AnalysisResult(
            source_type=SourceType.JSON,
            content_summary="JSON data",
            suggested_categories=[],
            suggested_types=[],
            content_statistics={"total_characters": 200},
        )

        plan = planner.create_plan(
            detection_result=detection,
            analysis_result=analysis,
            document_count=3,
            user_requested_types=["context_qa", "definition"],
        )

        type_names = [t.type_name for t in plan.selected_types]
        assert "context_qa" in type_names
        assert "definition" in type_names

    def test_create_plan_output_structure(self):
        """Test output structure is built correctly."""
        planner = BuildPlanner()

        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        analysis = AnalysisResult(
            source_type=SourceType.MARKDOWN,
            content_summary="Test",
            suggested_categories=[("knowledge", 0.7), ("qa", 0.6)],
            suggested_types=[
                ("facts", 0.7),
                ("context_qa", 0.6),
            ],
            content_statistics={"total_characters": 500},
        )

        plan = planner.create_plan(
            detection_result=detection,
            analysis_result=analysis,
            document_count=10,
        )

        assert "knowledge" in plan.output_structure
        assert "qa" in plan.output_structure
        assert "facts" in plan.output_structure["knowledge"]
        assert "context_qa" in plan.output_structure["qa"]

    def test_modify_plan(self):
        """Test modifying an existing plan."""
        planner = BuildPlanner(min_confidence_threshold=0.4)

        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        analysis = AnalysisResult(
            source_type=SourceType.MARKDOWN,
            content_summary="Test",
            suggested_categories=[],
            suggested_types=[
                ("facts", 0.7),
                ("context_qa", 0.6),
                ("glossary", 0.5),
            ],
            content_statistics={"total_characters": 500},
        )

        original_plan = planner.create_plan(
            detection_result=detection,
            analysis_result=analysis,
            document_count=10,
        )

        # User deselects some types
        modified_plan = planner.modify_plan(
            original_plan,
            selected_type_names=["facts"],  # Only keep facts
        )

        type_names = [t.type_name for t in modified_plan.selected_types]
        assert type_names == ["facts"]
        assert modified_plan.total_estimated_samples < original_plan.total_estimated_samples


class TestBuildPlan:
    """Test BuildPlan dataclass."""

    def test_to_display_dict(self):
        """Test converting plan to display dictionary."""
        plan = BuildPlan(
            source="/path/to/docs",
            source_type=SourceType.MARKDOWN,
            selected_types=[
                TypeSelection(
                    type_name="facts",
                    confidence=0.7,
                    estimated_samples=30,
                    generator_name="facts",
                    user_selected=True,
                ),
                TypeSelection(
                    type_name="context_qa",
                    confidence=0.6,
                    estimated_samples=40,
                    generator_name="qa",
                    user_selected=False,
                ),
            ],
            total_estimated_samples=70,
            output_structure={"knowledge": ["facts"], "qa": ["context_qa"]},
        )

        display = plan.to_display_dict()

        assert display["source"] == "/path/to/docs"
        assert display["source_type"] == "markdown"
        assert len(display["types"]) == 2
        assert display["types"][0]["name"] == "facts"
        assert display["types"][0]["confidence"] == "70%"
        assert display["total_samples"] == 70

    def test_to_json(self):
        """Test serializing plan to JSON."""
        plan = BuildPlan(
            source="/path",
            source_type=SourceType.MARKDOWN,
            selected_types=[
                TypeSelection(
                    type_name="facts",
                    confidence=0.7,
                    estimated_samples=30,
                    generator_name="facts",
                ),
            ],
        )

        json_str = plan.to_json()

        assert '"source_type": "markdown"' in json_str
        assert '"name": "facts"' in json_str

    def test_from_dict(self):
        """Test creating plan from dictionary."""
        data = {
            "source": "/path/to/docs",
            "source_type": "markdown",
            "types": [
                {
                    "name": "facts",
                    "confidence_raw": 0.7,
                    "estimated": 30,
                    "generator": "facts",
                    "selected": True,
                },
            ],
            "total_samples": 30,
            "output_structure": {"knowledge": ["facts"]},
        }

        plan = BuildPlan.from_dict(data)

        assert plan.source_type == SourceType.MARKDOWN
        assert len(plan.selected_types) == 1
        assert plan.selected_types[0].type_name == "facts"
        assert plan.selected_types[0].confidence == 0.7


class TestTypeSelection:
    """Test TypeSelection dataclass."""

    def test_creation(self):
        """Test creating a type selection."""
        selection = TypeSelection(
            type_name="facts",
            confidence=0.8,
            estimated_samples=25,
            generator_name="facts",
            user_selected=True,
        )

        assert selection.type_name == "facts"
        assert selection.confidence == 0.8
        assert selection.estimated_samples == 25
        assert selection.generator_name == "facts"
        assert selection.user_selected is True

    def test_default_user_selected(self):
        """Test default user_selected is False."""
        selection = TypeSelection(
            type_name="facts",
            confidence=0.8,
            estimated_samples=25,
            generator_name="facts",
        )

        assert selection.user_selected is False
