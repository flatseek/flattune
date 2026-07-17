"""QA Generator - generates question-answer pairs from knowledge."""

import logging
from collections.abc import Iterator

from flattune.teach.generators.base import GeneratedSample
from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import BaseGenerator, register_generator

logger = logging.getLogger(__name__)


@register_generator("qa")
@register_generator("question_answer")
class QAGenerator(BaseGenerator):
    """Generates Q&A pairs from knowledge.

    Creates diverse question types:
    - Factual questions
    - Explanatory questions
    - Comparative questions
    - Procedural questions
    """

    name = "qa"
    supported_types = ["qa", "question_answer"]

    def generate(
        self,
        knowledge: str | KnowledgeNode,
        **kwargs,
    ) -> Iterator[GeneratedSample]:
        """Generate Q&A samples from knowledge.

        Args:
            knowledge: Knowledge to generate from.
            **kwargs: Additional parameters.

        Yields:
            GeneratedSample objects.
        """
        if isinstance(knowledge, KnowledgeNode):
            context = knowledge.get_context()
            source = knowledge.source or ""
        else:
            context = str(knowledge)
            source = ""

        # Try teacher first, fall back to templates
        if self.teacher:
            try:
                yield from self._generate_with_teacher(context, source)
                return  # Success - don't fall back
            except Exception as e:
                logger.warning(f"Teacher generation failed: {e}")

        # Template-based generation fallback
        yield from self._generate_templates(context, source)

    def _generate_with_teacher(self, context: str, source: str) -> Iterator[GeneratedSample]:
        """Generate using teacher model."""
        if not self.teacher:
            yield from self._generate_templates(context, source)
            return

        try:
            result = self.teacher.generate(context, "qa")

            # Check for errors or low quality
            if result.get("error") or result.get("quality_score", 1.0) < 0.5:
                yield from self._generate_templates(context, source)
                return

            messages = result.get("messages", [])
            if messages:
                yield GeneratedSample(
                    conversation=messages,
                    sample_type="qa",
                    source=source,
                    quality_score=result.get("quality_score", 0.8),
                )
            else:
                # Teacher returned no messages, fall back to templates
                yield from self._generate_templates(context, source)
        except Exception as e:
            logger.warning(f"Teacher generation failed: {e}")
            yield from self._generate_templates(context, source)

    def _generate_templates(self, context: str, source: str) -> Iterator[GeneratedSample]:
        """Generate using templates (fallback when no teacher)."""
        # Extract key sentences for question generation
        sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 20]

        if not sentences:
            sentences = [context[:500]]  # Use first 500 chars if no clear sentences

        for _i, sentence in enumerate(sentences[:3]):  # Max 3 Q&A pairs per source
            # Generate factual question
            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"What is described in this text: {sentence[:200]}?"},
                    {"role": "assistant", "content": sentence},
                ],
                sample_type="qa",
                source=source,
                quality_score=0.7,
                metadata={"template": "factual"},
            )

            # Generate explanation question
            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"Explain this statement: {sentence[:200]}"},
                    {"role": "assistant", "content": sentence},
                ],
                sample_type="qa",
                source=source,
                quality_score=0.7,
                metadata={"template": "explanation"},
            )
