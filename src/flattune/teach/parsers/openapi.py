"""OpenAPI/Swagger parser."""

import json
from collections.abc import Iterator
from pathlib import Path

from flattune.teach.parsers.base import BaseParser, ParseResult
from flattune.teach.registry import SourceType, register_parser


@register_parser("openapi")
class OpenAPIParser(BaseParser):
    """Parser for OpenAPI/Swagger specifications."""

    source_type = SourceType.OPENAPI
    priority = 10  # Higher priority than JSON parser

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is an OpenAPI file."""
        path = str(source).lower()
        return path.endswith(".json") or path.endswith(".yaml") or path.endswith(".yml")

    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse OpenAPI spec into knowledge fragments.

        Yields:
            ParseResult for paths (endpoints).
        """
        source_str = str(source)
        path = Path(source_str)

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            # Try YAML first, then JSON
            try:
                import yaml
                spec = yaml.safe_load(content)
            except Exception:
                spec = json.loads(content)

        except Exception as e:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="",
                metadata={"error": str(e)},
            )
            return

        if not isinstance(spec, dict):
            return

        # Extract metadata
        metadata = {
            "filename": path.name,
            "title": spec.get("info", {}).get("title", ""),
            "version": spec.get("info", {}).get("version", ""),
        }

        # Yield server info
        servers = spec.get("servers", [])
        if servers:
            yield ParseResult(
                source=source_str,
                source_type=self.source_type,
                content="API Servers:\n" + "\n".join(s.get("url", "") for s in servers),
                metadata=dict(metadata, **{"type": "servers"}),
            )

        # Yield each endpoint
        paths = spec.get("paths", {})
        for path_str, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
                    continue
                if not isinstance(operation, dict):
                    continue

                yield self._operation_to_result(source_str, path_str, method.upper(), operation, metadata)

    def _operation_to_result(
        self,
        source: str,
        path: str,
        method: str,
        operation: dict,
        base_metadata: dict,
    ) -> ParseResult:
        """Convert an operation to ParseResult."""
        op_id = operation.get("operationId", f"{method}_{path}".replace("/", "_"))
        summary = operation.get("summary", "")
        description = operation.get("description", "")

        metadata = dict(base_metadata)
        # Extract all parameters including request body
        all_params = []

        # Operation-level parameters (path, query, header)
        for p in operation.get("parameters", []):
            all_params.append({
                "name": p.get("name"),
                "in": p.get("in", "query"),
                "type": p.get("schema", {}).get("type", "string") if p.get("schema") else "string",
                "description": p.get("description", ""),
                "required": p.get("required", False),
            })

        # Request body parameters
        request_body = operation.get("requestBody")
        if request_body:
            content = request_body.get("content", {})
            for media_type, media in content.items():
                schema = media.get("schema", {})
                properties = schema.get("properties", {})
                for prop_name, prop in properties.items():
                    all_params.append({
                        "name": prop_name,
                        "in": "body",
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", ""),
                        "required": prop_name in schema.get("required", []),
                    })

        metadata.update({
            "type": "endpoint",
            "operation_id": op_id,
            "method": method,
            "path": path,
            "parameters": all_params,
        })

        # Build content
        content_parts = [
            f"Endpoint: {method} {path}",
            f"Operation ID: {op_id}",
        ]

        if summary:
            content_parts.append(f"Summary: {summary}")
        if description:
            content_parts.append(f"Description: {description}")

        # Parameters
        params = operation.get("parameters", [])
        if params:
            content_parts.append("\nParameters:")
            for p in params:
                param_str = f"- {p.get('name')} ({p.get('in', 'query')}): {p.get('type', 'string')}"
                if p.get('required'):
                    param_str += " [required]"
                if p.get('description'):
                    param_str += f" - {p.get('description')}"
                content_parts.append(param_str)

        # Request body
        request_body = operation.get("requestBody")
        if request_body:
            content_parts.append("\nRequest Body:")
            content_parts.append(f"  {request_body.get('description', 'No description')}")
            content = request_body.get("content", {})
            for media_type, media in content.items():
                content_parts.append(f"  Content-Type: {media_type}")
                schema = media.get("schema", {})
                if schema.get("properties"):
                    content_parts.append("  Properties:")
                    for prop_name, prop in schema["properties"].items():
                        content_parts.append(f"    - {prop_name}: {prop.get('type', 'any')}")

        # Responses
        responses = operation.get("responses", {})
        if responses:
            content_parts.append("\nResponses:")
            for code, resp in responses.items():
                resp_desc = resp.get("description", "")
                content_parts.append(f"  {code}: {resp_desc}")

        content = "\n".join(content_parts)

        # Extract entities
        entities = [
            {"type": "method", "content": method},
            {"type": "path", "content": path},
            {"type": "operation_id", "content": op_id},
        ]

        return ParseResult(
            source=source,
            source_type=self.source_type,
            content=content,
            metadata=metadata,
            entities=entities,
        )


@register_parser("swagger", alias="openapi")
class SwaggerParser(OpenAPIParser):
    """Parser for Swagger/OpenAPI 2.0 specs."""
    pass
