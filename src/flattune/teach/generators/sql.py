"""SQL Generator - generates natural language to SQL examples from database schemas."""

import logging
import re
from collections.abc import Iterator

from flattune.teach.generators.base import BaseGenerator, GeneratedSample
from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import register_generator

logger = logging.getLogger(__name__)


@register_generator("nl_to_sql")
@register_generator("sql")
class SQLGenerator(BaseGenerator):
    """Generates natural language to SQL query examples from database schemas.

    Creates training samples for:
    - Simple SELECT queries
    - WHERE clause queries
    - JOIN queries
    - Aggregation queries
    - Complex multi-table queries
    """

    name = "sql"
    supported_types = ["nl_to_sql", "sql", "sql_to_result"]

    def generate(
        self,
        knowledge: str | KnowledgeNode,
        **kwargs,
    ) -> Iterator[GeneratedSample]:
        """Generate SQL examples from database schema knowledge.

        Args:
            knowledge: Knowledge containing database schema.
            **kwargs: Additional parameters.

        Yields:
            GeneratedSample objects.
        """
        if isinstance(knowledge, KnowledgeNode):
            context = knowledge.get_context()
            metadata = knowledge.metadata
            source = knowledge.source or ""
        else:
            context = str(knowledge)
            metadata = {}
            source = ""

        # Extract schema information
        tables = metadata.get("tables", [])
        metadata.get("relationships", [])

        if not tables:
            # Try to parse from content
            tables = self._parse_tables_from_text(context)

        if tables:
            yield from self._generate_simple_queries(tables, source)
            yield from self._generate_filter_queries(tables, source)
            yield from self._generate_aggregation_queries(tables, source)
            if len(tables) > 1:
                yield from self._generate_join_queries(tables, source)

    def _parse_tables_from_text(self, text: str) -> list[dict]:
        """Parse table definitions from text content."""
        tables = []

        # Look for table patterns like "CREATE TABLE" or "Table: name"
        create_pattern = r"(?:CREATE TABLE|Create Table|Table:)\s+(\w+)"
        matches = re.findall(create_pattern, text, re.IGNORECASE)

        for table_name in matches:
            tables.append({
                "name": table_name,
                "columns": [],
            })

        return tables

    def _generate_simple_queries(
        self,
        tables: list[dict],
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate simple SELECT queries."""
        for table in tables[:3]:  # Max 3 tables
            table_name = table.get("name", "users")

            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"Show me all records from {table_name}"},
                    {"role": "assistant", "content": f"SELECT * FROM {table_name};"},
                ],
                sample_type="nl_to_sql",
                source=source,
                quality_score=0.8,
                metadata={"table": table_name, "query_type": "select_all"},
            )

            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"How many rows are in {table_name}?"},
                    {"role": "assistant", "content": f"SELECT COUNT(*) FROM {table_name};"},
                ],
                sample_type="nl_to_sql",
                source=source,
                quality_score=0.8,
                metadata={"table": table_name, "query_type": "count"},
            )

    def _generate_filter_queries(
        self,
        tables: list[dict],
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate queries with WHERE clauses."""
        for table in tables[:2]:
            table_name = table.get("name", "users")
            columns = table.get("columns", [])

            if not columns:
                # Generic column names
                yield GeneratedSample(
                    conversation=[
                        {"role": "user", "content": f"Find all {table_name} with id greater than 10"},
                        {"role": "assistant", "content": f"SELECT * FROM {table_name} WHERE id > 10;"},
                    ],
                    sample_type="nl_to_sql",
                    source=source,
                    quality_score=0.7,
                    metadata={"table": table_name, "query_type": "where_comparison"},
                )
            else:
                # Use actual column names
                for col in columns[:2]:
                    col_name = col.get("name", "id")
                    col.get("type", "integer")

                    yield GeneratedSample(
                        conversation=[
                            {
                                "role": "user",
                                "content": f"Get all {table_name} where {col_name} is valid",
                            },
                            {
                                "role": "assistant",
                                "content": f"SELECT * FROM {table_name} WHERE {col_name} IS NOT NULL;",
                            },
                        ],
                        sample_type="nl_to_sql",
                        source=source,
                        quality_score=0.7,
                        metadata={"table": table_name, "column": col_name, "query_type": "where_not_null"},
                    )

    def _generate_aggregation_queries(
        self,
        tables: list[dict],
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate aggregation queries."""
        for table in tables[:2]:
            table_name = table.get("name", "users")

            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"What is the average value in {table_name}?"},
                    {"role": "assistant", "content": f"SELECT AVG(*) FROM {table_name};"},
                ],
                sample_type="nl_to_sql",
                source=source,
                quality_score=0.6,  # Lower quality - generic
                metadata={"table": table_name, "query_type": "avg"},
            )

            yield GeneratedSample(
                conversation=[
                    {"role": "user", "content": f"Show me {table_name} grouped by type"},
                    {"role": "assistant", "content": f"SELECT type, COUNT(*) FROM {table_name} GROUP BY type;"},
                ],
                sample_type="nl_to_sql",
                source=source,
                quality_score=0.7,
                metadata={"table": table_name, "query_type": "group_by"},
            )

    def _generate_join_queries(
        self,
        tables: list[dict],
        source: str,
    ) -> Iterator[GeneratedSample]:
        """Generate JOIN queries."""
        if len(tables) < 2:
            return

        table1 = tables[0].get("name", "users")
        table2 = tables[1].get("name", "orders")

        yield GeneratedSample(
            conversation=[
                {"role": "user", "content": f"Show all {table1} with their {table2}"},
                {
                    "role": "assistant",
                    "content": f"SELECT * FROM {table1} LEFT JOIN {table2} ON {table1}.id = {table2}.{table1}_id;",
                },
            ],
            sample_type="nl_to_sql",
            source=source,
            quality_score=0.6,  # Lower - generic join
            metadata={"tables": [table1, table2], "query_type": "left_join"},
        )
