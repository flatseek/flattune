"""Content analysis for the build pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from flattune.build.registry import (
    SourceDetectionResult,
    SourceType,
    CategoryRegistry,
    DatasetTypeRegistry,
)
from flattune.build.sources import (
    MarkdownSourceDetector,
    JSONSourceDetector,
    JSONLSourceDetector,
    FlatseekSourceDetector,
    TextSourceDetector,
)


@dataclass
class AnalysisResult:
    """Result from analyzing source content."""
    source_type: SourceType
    content_summary: str  # Brief description of content
    suggested_categories: list[tuple[str, float]]  # (category_name, confidence)
    suggested_types: list[tuple[str, float]]  # (type_name, confidence)
    content_statistics: dict[str, Any] = field(default_factory=dict)
    # Statistics fields:
    #   - total_documents: int
    #   - total_characters: int
    #   - avg_document_length: float
    #   - has_structured_fields: bool
    #   - field_names: list[str]
    #   - detected_language: str
    #   - has_code_blocks: bool
    #   - has_questions: bool


class ContentAnalyzer:
    """Analyzes source content to suggest dataset types."""

    def __init__(self):
        """Initialize content analyzer."""
        # Import source detectors to trigger registration
        self._source_detectors = {
            SourceType.MARKDOWN: MarkdownSourceDetector(),
            SourceType.JSON: JSONSourceDetector(),
            SourceType.JSONL: JSONLSourceDetector(),
            SourceType.FLATSEEK: FlatseekSourceDetector(),
            SourceType.TEXT: TextSourceDetector(),
        }

    def analyze(
        self,
        detection_result: SourceDetectionResult,
        content: str,
    ) -> AnalysisResult:
        """Analyze content sample and return suggestions.

        Args:
            detection_result: Source detection result.
            content: Content sample for analysis.

        Returns:
            AnalysisResult with suggested categories and types.
        """
        if not content:
            content = ""

        # Calculate content statistics
        stats = self._calculate_statistics(content, detection_result)

        # Score categories and types based on content
        categories = self._score_categories(content, detection_result, stats)
        types = self._score_types(content, detection_result, stats)

        # Generate summary
        summary = self._generate_summary(detection_result.source_type, stats)

        return AnalysisResult(
            source_type=detection_result.source_type,
            content_summary=summary,
            suggested_categories=categories,
            suggested_types=types,
            content_statistics=stats,
        )

    def _calculate_statistics(
        self,
        content: str,
        detection_result: SourceDetectionResult,
    ) -> dict[str, Any]:
        """Calculate content statistics."""
        stats: dict[str, Any] = {
            "total_characters": len(content),
            "has_structured_fields": False,
            "field_names": [],
            "detected_language": "unknown",
            "has_code_blocks": False,
            "has_questions": False,
            "has_numbers": False,
            "has_definitions": False,
            "has_list_items": False,
            "avg_sentence_length": 0,
        }

        if not content:
            return stats

        # Detect code blocks
        stats["has_code_blocks"] = bool(re.search(r"```[\s\S]*?```", content))

        # Detect questions
        question_pattern = r"(?:^|\s)(?:what|who|where|when|why|how|is|are|can|do|does)\s+\w+\??"
        stats["has_questions"] = bool(re.search(question_pattern, content, re.IGNORECASE | re.MULTILINE))

        # Detect numbers (potential facts)
        number_count = len(re.findall(r"\d+(?:\.\d+)?", content))
        stats["has_numbers"] = number_count > 5

        # Detect definitions
        definition_patterns = [
            r"is defined as",
            r"means",
            r"refers to",
            r"is a",
            r"are called",
        ]
        stats["has_definitions"] = any(
            re.search(pattern, content, re.IGNORECASE) for pattern in definition_patterns
        )

        # Detect list items
        list_patterns = [
            r"^\s*[-*+]\s+\w+",  # Bullet points
            r"^\s*\d+\.\s+\w+",  # Numbered lists
        ]
        stats["has_list_items"] = any(
            re.search(pattern, content, re.MULTILINE) for pattern in list_patterns
        )

        # Detect structured fields (JSON-like)
        field_pattern = r"\"[\w\s]+\"\s*:"
        field_matches = re.findall(field_pattern, content)
        if len(field_matches) > 2:
            stats["has_structured_fields"] = True
            stats["field_names"] = [f.strip('":') for f in field_matches[:20]]

        # Calculate average sentence length
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            stats["avg_sentence_length"] = sum(len(s) for s in sentences) / len(sentences)

        return stats

    def _score_categories(
        self,
        content: str,
        detection_result: SourceDetectionResult,
        stats: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """Score dataset categories based on content analysis."""
        scores: list[tuple[str, float]] = []

        # Knowledge category - factual content
        knowledge_score = 0.0
        if stats["has_numbers"]:
            knowledge_score += 0.3
        if stats["has_definitions"]:
            knowledge_score += 0.3
        if stats["has_structured_fields"]:
            knowledge_score += 0.2
        if detection_result.source_type in [SourceType.MARKDOWN, SourceType.TEXT]:
            knowledge_score += 0.2
        if knowledge_score > 0:
            scores.append(("knowledge", min(knowledge_score, 0.95)))

        # QA category - content with questions or context
        qa_score = 0.0
        if stats["has_questions"]:
            qa_score += 0.4
        if stats["total_characters"] > 200:
            qa_score += 0.3
        if stats["has_numbers"]:
            qa_score += 0.2
        if qa_score > 0:
            scores.append(("qa", min(qa_score, 0.95)))

        # Conversation category - dialogue-like content
        conversation_score = 0.0
        dialogue_indicators = [
            r"\w+:\s*\"",
            r"said\s+\w+",
            r"asked\s+\w+",
            r"replied",
        ]
        if any(re.search(p, content, re.IGNORECASE) for p in dialogue_indicators):
            conversation_score += 0.5
        if conversation_score > 0:
            scores.append(("conversation", min(conversation_score, 0.8)))

        # Instruction category - procedural content
        instruction_score = 0.0
        if stats["has_list_items"]:
            instruction_score += 0.3
        if any(kw in content.lower() for kw in ["step", "how to", "tutorial", "instructions"]):
            instruction_score += 0.4
        instruction_verbs = ["install", "configure", "setup", "run", "execute", "build", "create"]
        if sum(1 for v in instruction_verbs if v in content.lower()) >= 2:
            instruction_score += 0.3
        if instruction_score > 0:
            scores.append(("instruction", min(instruction_score, 0.9)))

        # Classification category - content with categories/labels
        classification_score = 0.0
        if stats["has_structured_fields"]:
            classification_score += 0.3
        class_indicators = ["category", "type", "class", "label", "tag", "topic"]
        if sum(1 for ind in class_indicators if ind in content.lower()) >= 2:
            classification_score += 0.4
        if classification_score > 0:
            scores.append(("classification", min(classification_score, 0.8)))

        # Summary category - longer content
        summary_score = 0.0
        if stats["total_characters"] > 500:
            summary_score += 0.3
        if stats["avg_sentence_length"] > 50:
            summary_score += 0.3
        if summary_score > 0:
            scores.append(("summary", min(summary_score, 0.85)))

        # Extraction category - structured content
        extraction_score = 0.0
        if stats["has_structured_fields"]:
            extraction_score += 0.5
        if stats["field_names"]:
            extraction_score += 0.3
        if extraction_score > 0:
            scores.append(("extraction", min(extraction_score, 0.9)))

        # RAG category - context-rich content
        rag_score = 0.0
        if stats["total_characters"] > 300:
            rag_score += 0.3
        if stats["has_numbers"] and stats["has_definitions"]:
            rag_score += 0.3
        if rag_score > 0:
            scores.append(("rag", min(rag_score, 0.8)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _score_types(
        self,
        content: str,
        detection_result: SourceDetectionResult,
        stats: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """Score dataset types based on content analysis."""
        scores: list[tuple[str, float]] = []

        # Get all registered types and filter by source compatibility
        all_types = DatasetTypeRegistry.list_all_types()

        for dt in all_types:
            if dt is None:
                continue

            # Check if source type is supported
            if dt.supported_sources:
                source_value = detection_result.source_type.value
                if source_value not in dt.supported_sources and "any" not in dt.supported_sources:
                    continue

            # Check minimum content length
            if stats["total_characters"] < dt.min_content_length:
                continue

            # Score based on category scoring
            category_score = 0.0
            for cat_name, cat_score in self._score_categories(content, detection_result, stats):
                if cat_name == dt.category:
                    category_score = cat_score
                    break

            # Type-specific adjustments
            type_score = category_score

            # Facts type
            if dt.name == "facts":
                if stats["has_numbers"]:
                    type_score = max(type_score, 0.7)
                if stats["has_definitions"]:
                    type_score = max(type_score, 0.6)

            # Glossary type
            elif dt.name == "glossary":
                term_indicators = len(re.findall(r'\*\*[^*]+\*\*', content))
                if term_indicators > 3:
                    type_score = max(type_score, 0.7)
                elif stats["has_definitions"]:
                    type_score = max(type_score, 0.6)

            # Context QA type
            elif dt.name == "context_qa":
                if stats["total_characters"] > 200:
                    type_score = max(type_score, 0.7)

            # Executive summary type
            elif dt.name == "executive_summary":
                if stats["total_characters"] > 500:
                    type_score = max(type_score, 0.6)

            # Topic classify
            elif dt.name == "topic_classify":
                if stats["has_structured_fields"]:
                    type_score = max(type_score, 0.6)

            if type_score >= 0.3:  # Minimum threshold
                scores.append((dt.name, type_score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:10]  # Top 10 types

    def _generate_summary(self, source_type: SourceType, stats: dict[str, Any]) -> str:
        """Generate a brief summary of the content."""
        parts = []

        # Source type description
        type_descriptions = {
            SourceType.MARKDOWN: "Markdown document",
            SourceType.TEXT: "Plain text",
            SourceType.JSON: "JSON data",
            SourceType.JSONL: "JSON Lines data",
            SourceType.CSV: "CSV data",
            SourceType.PDF: "PDF document",
            SourceType.HTML: "HTML document",
            SourceType.FLATSEEK: "FlatSeek index",
            SourceType.OPENAPI: "OpenAPI specification",
        }
        parts.append(type_descriptions.get(source_type, "Document"))

        # Content characteristics
        if stats["total_characters"] > 1000:
            parts.append("substantial")
        elif stats["total_characters"] > 200:
            parts.append("moderate length")
        else:
            parts.append("brief")

        if stats["has_structured_fields"]:
            parts.append("structured")
        if stats["has_code_blocks"]:
            parts.append("contains code")
        if stats["has_questions"]:
            parts.append("contains questions")
        if stats["has_definitions"]:
            parts.append("contains definitions")
        if stats["has_list_items"]:
            parts.append("contains lists")

        return " ".join(parts)
