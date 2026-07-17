"""Tool Call Generator - generates tool calling examples from API/software knowledge."""

import json
import logging
import re
from collections.abc import Iterator

from flattune.teach.generators.base import BaseGenerator, GeneratedSample
from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import register_generator

logger = logging.getLogger(__name__)


@register_generator("tool_call")
@register_generator("intent")
@register_generator("slot_fill")
class ToolCallGenerator(BaseGenerator):
    """Generates tool calling, intent recognition, and slot filling examples.

    Used for teaching models how to:
    - Recognize user intent from natural language
    - Extract parameters from user requests
    - Format tool calls correctly
    """

    name = "tool_call"
    supported_types = ["tool_call", "intent", "slot_fill", "validation", "error_recovery"]

    def generate(
        self,
        knowledge: str | KnowledgeNode,
        **kwargs,
    ) -> Iterator[GeneratedSample]:
        """Generate tool call samples from API/tool knowledge.

        Args:
            knowledge: Knowledge containing API or tool definitions.
            **kwargs: Additional parameters.

        Yields:
            GeneratedSample objects.
        """
        if isinstance(knowledge, KnowledgeNode):
            context = knowledge.get_context()
            metadata = knowledge.metadata
            source = knowledge.source or ""
            node_type = metadata.get("type", "")
        else:
            context = str(knowledge)
            metadata = {}
            source = ""
            node_type = ""

        # Skip non-endpoint nodes (like 'servers' info)
        if node_type not in ("", "endpoint", "tool", "function") and not metadata.get(
            "operation_id"
        ):
            return

        # Extract tool information from various metadata fields
        if metadata.get("path"):
            # Extract clean endpoint name from path
            path = metadata.get("path", "")
            method = metadata.get("method", "").upper()
            # Get last 2-3 path segments as name
            parts = [
                p for p in path.strip("/").split("/") if p and p not in ("api", "v1", "v2", "v3")
            ]
            endpoint_name = (
                "_".join(parts[-2:]) if len(parts) > 1 else (parts[-1] if parts else "api")
            )
            tool_name = f"{method}_{endpoint_name}" if method else endpoint_name
        else:
            tool_name = (
                metadata.get("name")
                or metadata.get("operation_id")
                or metadata.get("tool")
                or metadata.get("function")
                or "unknown_tool"
            )
        # Clean up tool name (remove special chars except underscore)
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
        tool_name = re.sub(r"_+", "_", tool_name).strip("_")

        tool_description = metadata.get("description") or metadata.get("summary") or context[:500]
        parameters = metadata.get("parameters", [])

        # Generate different sample types
        yield from self._generate_intent_examples(tool_name, tool_description, parameters, source)
        yield from self._generate_slot_filling(tool_name, tool_description, parameters, source)
        yield from self._generate_formal_casual(tool_name, tool_description, parameters, source)
        yield from self._generate_error_recovery(tool_name, tool_description, parameters, source)

    def _generate_intent_examples(
        self,
        tool_name: str,
        description: str,
        parameters: list,
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate intent recognition examples."""
        # Various ways to invoke the same tool
        intents = [
            f"I need to {tool_name.replace('_', ' ')}",
            f"Can you {tool_name.replace('_', ' ')} for me?",
            f"Please {tool_name.replace('_', ' ')}",
            f"{tool_name.replace('_', ' ').title()}!",
            f"Could you help me {tool_name.replace('_', ' ')}?",
        ]

        # Build expected tool call
        tool_call = {
            "tool": tool_name,
            "parameters": {},
        }
        for p in parameters[:3]:  # Max 3 params shown
            tool_call["parameters"][p.get("name", "param")] = f"<{p.get('type', 'value')}>"

        for intent in intents:
            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": json.dumps(tool_call)},
                ],
                sample_type="intent",
                source=source,
                quality_score=0.8,
                metadata={"tool": tool_name},
            )

    def _generate_slot_filling(
        self,
        tool_name: str,
        description: str,
        parameters: list,
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate slot filling examples with partial information."""
        if not parameters:
            return

        # Prefer body and query params over header params (they're more meaningful for API calls)
        priority_order = ["body", "query", "path", "header"]
        sorted_params = sorted(
            parameters,
            key=lambda p: (
                priority_order.index(p.get("in", "header"))
                if p.get("in", "header") in priority_order
                else 99
            ),
        )

        for param in sorted_params[:2]:  # Use top 2 meaningful params
            param_name = param.get("name", "value")
            param_type = param.get("type", "string")
            param_in = param.get("in", "query")

            # Example with explicit parameter
            yield GeneratedSample(
                conversation=[
                    {
                        "role": "user",
                        "content": f"{tool_name.replace('_', ' ')} with {param_name}={param_type}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "tool": tool_name,
                                "parameters": {param_name: f"<{param_in}:{param_type}>"},
                            }
                        ),
                    },
                ],
                sample_type="slot_fill",
                source=source,
                quality_score=0.7,
                metadata={"parameter": param_name, "in": param_in},
            )

    def _generate_formal_casual(
        self,
        tool_name: str,
        description: str,
        parameters: list,
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate formal and casual variations."""
        formal_intent = f"I would like to request the execution of {tool_name}"
        casual_intent = f"hey, {tool_name.replace('_', ' ')} please"

        for intent, style in [(formal_intent, "formal"), (casual_intent, "casual")]:
            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": intent},
                    {
                        "role": "assistant",
                        "content": json.dumps({"tool": tool_name, "parameters": {}}),
                    },
                ],
                sample_type="tool_call",
                source=source,
                quality_score=0.8,
                metadata={"style": style, "tool": tool_name},
            )

    def _generate_error_recovery(
        self,
        tool_name: str,
        description: str,
        parameters: list,
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate error recovery examples."""
        if not parameters:
            return

        # Invalid parameter type
        yield GeneratedSample(
            conversation=[
                {
                    "role": "user",
                    "content": f"{tool_name} with {parameters[0].get('name', 'param')}=not_a_number",
                },
                {
                    "role": "assistant",
                    "content": "I need a valid number for that parameter. Please provide a numeric value.",
                },
            ],
            sample_type="error_recovery",
            source=source,
            quality_score=0.7,
            metadata={"error_type": "invalid_type", "tool": tool_name},
        )

        # Missing required parameter
        if any(p.get("required") for p in parameters):
            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"{tool_name}"},
                    {
                        "role": "assistant",
                        "content": f"I need more information. Which {parameters[0].get('name', 'value')} should I use?",
                    },
                ],
                sample_type="error_recovery",
                source=source,
                quality_score=0.7,
                metadata={"error_type": "missing_param", "tool": tool_name},
            )
