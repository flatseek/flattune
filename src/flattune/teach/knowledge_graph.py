"""Knowledge Graph - Intermediate representation for teaching.

This module provides a structured intermediate representation that is:
- Independent from the source format
- Independent from the dataset format
- Suitable for diverse generation strategies
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator, Optional


class NodeType(Enum):
    """Types of knowledge nodes."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    API_ENDPOINT = "api_endpoint"
    API_PARAMETER = "api_parameter"
    API_SCHEMA = "api_schema"
    DATABASE_TABLE = "database_table"
    DATABASE_COLUMN = "database_column"
    DATABASE_RELATIONSHIP = "database_relationship"
    TOOL = "tool"
    TOOL_PARAMETER = "tool_parameter"
    WORKFLOW_STEP = "workflow_step"
    CODE_FUNCTION = "code_function"
    CODE_CLASS = "code_class"
    CLI_COMMAND = "cli_command"
    CONCEPT = "concept"
    FACT = "fact"
    PROCEDURE = "procedure"
    EXAMPLE = "example"


class EdgeType(Enum):
    """Types of relationships between nodes."""
    CONTAINS = "contains"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    DEPENDS_ON = "depends_on"
    INHERITS = "inherits"
    RETURNS = "returns"
    HAS_PARAMETER = "has_parameter"
    HAS_RETURN_TYPE = "has_return_type"
    HAS_EXAMPLE = "has_example"
    RELATED_TO = "related_to"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph.

    Represents a unit of knowledge that can be used to generate training samples.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.DOCUMENT
    content: str = ""
    summary: Optional[str] = None
    source: Optional[str] = None
    source_line: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    edges: list[tuple[str, EdgeType, str]] = field(default_factory=list)  # (target_id, edge_type, label)
    created_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        """String representation of the node."""
        preview = self.content[:100].replace("\n", " ")
        if len(self.content) > 100:
            preview += "..."
        return f"KnowledgeNode({self.node_type.value}: {preview})"

    def get_context(self, max_length: int = 2048) -> str:
        """Get formatted context for generation.

        Args:
            max_length: Maximum context length.

        Returns:
            Formatted context string.
        """
        parts = []

        if self.summary:
            parts.append(f"Summary: {self.summary}")

        if self.metadata.get("name"):
            parts.append(f"Name: {self.metadata['name']}")

        if self.metadata.get("type"):
            parts.append(f"Type: {self.metadata['type']}")

        parts.append(f"Content: {self.content}")

        # Add relevant properties
        for key in ["parameters", "returns", "examples", "description"]:
            if key in self.metadata:
                parts.append(f"{key.title()}: {self.metadata[key]}")

        context = "\n\n".join(parts)

        if len(context) > max_length:
            context = context[:max_length] + "..."

        return context

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "summary": self.summary,
            "source": self.source,
            "source_line": self.source_line,
            "metadata": self.metadata,
            "properties": self.properties,
            "edges": [(t, e.value, l) for t, e, l in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeNode:
        """Create from dictionary representation."""
        edges = [
            (t, EdgeType(e), l) for t, e, l in data.get("edges", [])
        ]
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            summary=data.get("summary"),
            source=data.get("source"),
            source_line=data.get("source_line"),
            metadata=data.get("metadata", {}),
            properties=data.get("properties", {}),
            edges=edges,
        )


@dataclass
class KnowledgeGraph:
    """A collection of knowledge nodes with relationships.

    This is the intermediate representation used by the pipeline:
    Source → Parser → Knowledge Graph → Teacher → Generator → Dataset
    """
    nodes: dict[str, KnowledgeNode] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        """Add a node to the graph.

        Args:
            node: Node to add.

        Returns:
            The added node.
        """
        self.nodes[node.id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, label: str = "") -> None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of edge.
            label: Optional edge label.
        """
        if source_id in self.nodes and target_id in self.nodes:
            self.nodes[source_id].edges.append((target_id, edge_type, label))

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID.

        Args:
            node_id: Node ID to find.

        Returns:
            Node or None if not found.
        """
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[KnowledgeNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type to filter by.

        Returns:
            List of matching nodes.
        """
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_related(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[KnowledgeNode]:
        """Get nodes related to a given node.

        Args:
            node_id: Node to find relations for.
            edge_type: Optional edge type filter.

        Returns:
            List of related nodes.
        """
        node = self.nodes.get(node_id)
        if not node:
            return []

        related = []
        for target_id, e_type, _ in node.edges:
            if edge_type is None or e_type == edge_type:
                target = self.nodes.get(target_id)
                if target:
                    related.append(target)

        return related

    def iterate(self, node_type: Optional[NodeType] = None) -> Iterator[KnowledgeNode]:
        """Iterate over nodes.

        Args:
            node_type: Optional type filter.

        Yields:
            KnowledgeNode objects.
        """
        for node in self.nodes.values():
            if node_type is None or node.node_type == node_type:
                yield node

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def stats(self) -> dict[str, Any]:
        """Get statistics about the graph.

        Returns:
            Dictionary with graph statistics.
        """
        type_counts: dict[str, int] = {}
        for node in self.nodes.values():
            type_name = node.node_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "by_type": type_counts,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        """Create from dictionary representation."""
        graph = cls(metadata=data.get("metadata", {}))
        for node_dict in data.get("nodes", {}).values():
            node = KnowledgeNode.from_dict(node_dict)
            graph.add_node(node)
        return graph


# ============================================================================
# Builder Helper
# ============================================================================


class KnowledgeGraphBuilder:
    """Builder for constructing knowledge graphs from parsed content."""

    def __init__(self, source: Optional[str] = None):
        """Initialize the builder.

        Args:
            source: Optional source identifier.
        """
        self.graph = KnowledgeGraph()
        self._source = source

    def add_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KnowledgeNode:
        """Add a document node.

        Args:
            content: Document content.
            metadata: Optional metadata.

        Returns:
            Created node.
        """
        node = KnowledgeNode(
            node_type=NodeType.DOCUMENT,
            content=content,
            source=self._source,
            metadata=metadata or {},
        )
        return self.graph.add_node(node)

    def add_section(
        self,
        content: str,
        parent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KnowledgeNode:
        """Add a section node.

        Args:
            content: Section content.
            parent_id: Optional parent document/section ID.
            metadata: Optional metadata.

        Returns:
            Created node.
        """
        node = KnowledgeNode(
            node_type=NodeType.SECTION,
            content=content,
            source=self._source,
            metadata=metadata or {},
        )
        self.graph.add_node(node)

        if parent_id:
            self.graph.add_edge(parent_id, node.id, EdgeType.CONTAINS)

        return node

    def add_api_endpoint(
        self,
        method: str,
        path: str,
        summary: str,
        parameters: Optional[list[dict]] = None,
        request_body: Optional[dict] = None,
        responses: Optional[dict] = None,
    ) -> KnowledgeNode:
        """Add an API endpoint node.

        Args:
            method: HTTP method.
            path: API path.
            summary: Endpoint summary.
            parameters: List of parameters.
            request_body: Request body schema.
            responses: Response schemas.

        Returns:
            Created node.
        """
        metadata = {
            "method": method,
            "path": path,
            "summary": summary,
            "parameters": parameters or [],
            "request_body": request_body,
            "responses": responses or {},
        }

        content = f"{method.upper()} {path}\n\n{summary}"

        if parameters:
            content += "\n\nParameters:\n"
            for p in parameters:
                content += f"- {p.get('name')}: {p.get('type')} ({p.get('in', 'query')})\n"

        if request_body:
            content += f"\nRequest Body: {request_body.get('description', '')}\n"

        node = KnowledgeNode(
            node_type=NodeType.API_ENDPOINT,
            content=content,
            summary=summary,
            source=self._source,
            metadata=metadata,
        )
        return self.graph.add_node(node)

    def add_tool(
        self,
        name: str,
        description: str,
        parameters: Optional[list[dict]] = None,
        examples: Optional[list[str]] = None,
    ) -> KnowledgeNode:
        """Add a tool/node function node.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: List of parameters.
            examples: Usage examples.

        Returns:
            Created node.
        """
        metadata = {
            "name": name,
            "description": description,
            "parameters": parameters or [],
            "examples": examples or [],
        }

        content = f"Tool: {name}\n\n{description}"

        if parameters:
            content += "\n\nParameters:\n"
            for p in parameters:
                content += f"- {p.get('name')}: {p.get('type')} - {p.get('description', '')}\n"

        if examples:
            content += "\n\nExamples:\n"
            for ex in examples:
                content += f"- {ex}\n"

        node = KnowledgeNode(
            node_type=NodeType.TOOL,
            content=content,
            summary=description,
            source=self._source,
            metadata=metadata,
        )
        return self.graph.add_node(node)

    def add_database_schema(
        self,
        tables: list[dict[str, Any]],
        relationships: Optional[list[dict[str, str]]] = None,
    ) -> KnowledgeNode:
        """Add a database schema node.

        Args:
            tables: List of table definitions.
            relationships: List of table relationships.

        Returns:
            Created node.
        """
        schema_node = KnowledgeNode(
            node_type=NodeType.DATABASE_TABLE,
            content=f"Database Schema with {len(tables)} tables",
            source=self._source,
            metadata={"tables": tables, "relationships": relationships or []},
        )
        self.graph.add_node(schema_node)

        for table in tables:
            table_node = KnowledgeNode(
                node_type=NodeType.DATABASE_TABLE,
                content=f"Table: {table['name']}\n\n" + "\n".join(
                    f"  {col['name']}: {col['type']}" for col in table.get("columns", [])
                ),
                source=self._source,
                metadata=table,
            )
            self.graph.add_node(table_node)
            self.graph.add_edge(schema_node.id, table_node.id, EdgeType.CONTAINS)

        return schema_node

    def build(self) -> KnowledgeGraph:
        """Build and return the knowledge graph.

        Returns:
            The constructed knowledge graph.
        """
        return self.graph
