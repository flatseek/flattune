"""Pytest fixtures for FlatTune tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample markdown content for testing."""
    return """# Python asyncio

## Overview

Python's `asyncio` module provides tools for concurrent programming.

## Coroutines

The `@async def` syntax declares a coroutine function.

`await` suspends execution until the coroutine completes.

## Example

```python
async def fetch_data():
    result = await api.call()
    return result
```

## Setup Steps

1. Install Python 3.10+
2. Create virtual environment
3. Install dependencies
"""


@pytest.fixture
def sample_json_content() -> str:
    """Sample JSON content for testing."""
    return json.dumps([
        {"id": 1, "name": "Python", "category": "language", "votes": 150},
        {"id": 2, "name": "JavaScript", "category": "language", "votes": 120},
        {"id": 3, "name": "Go", "category": "language", "votes": 95},
    ])


@pytest.fixture
def sample_jsonl_content() -> str:
    """Sample JSONL content for testing."""
    return json.dumps({"question": "What is Python?", "answer": "A programming language."}) + "\n" + \
           json.dumps({"question": "What is asyncio?", "answer": "A concurrency library."}) + "\n"


@pytest.fixture
def sample_openapi_content() -> str:
    """Sample OpenAPI spec for testing."""
    return json.dumps({
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/invoices": {
                "post": {
                    "operationId": "createInvoice",
                    "summary": "Create an invoice",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "customer": {"type": "string"},
                                        "amount": {"type": "number"},
                                    },
                                    "required": ["customer", "amount"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Created"},
                        "400": {"description": "Bad Request"},
                    }
                }
            }
        }
    })


@pytest.fixture
def sample_sql_schema() -> str:
    """Sample SQL schema for testing."""
    return """
    CREATE TABLE customers (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        amount DECIMAL(10, 2) NOT NULL,
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """


@pytest.fixture
def sample_faq_content() -> str:
    """Sample FAQ content for testing."""
    return json.dumps([
        {
            "question": "How do I reset my password?",
            "answer": "Click 'Forgot Password' on the login page and follow the instructions.",
            "category": "account"
        },
        {
            "question": "What are the pricing plans?",
            "answer": "We offer Free, Pro ($29/month), and Enterprise plans.",
            "category": "billing"
        },
    ])


@pytest.fixture
def markdown_file(temp_dir: Path, sample_markdown_content: str) -> Path:
    """Create a sample markdown file."""
    f = temp_dir / "test.md"
    f.write_text(sample_markdown_content)
    return f


@pytest.fixture
def json_file(temp_dir: Path, sample_json_content: str) -> Path:
    """Create a sample JSON file."""
    f = temp_dir / "test.json"
    f.write_text(sample_json_content)
    return f


@pytest.fixture
def jsonl_file(temp_dir: Path, sample_jsonl_content: str) -> Path:
    """Create a sample JSONL file."""
    f = temp_dir / "test.jsonl"
    f.write_text(sample_jsonl_content)
    return f


@pytest.fixture
def openapi_file(temp_dir: Path, sample_openapi_content: str) -> Path:
    """Create a sample OpenAPI file."""
    f = temp_dir / "api.json"
    f.write_text(sample_openapi_content)
    return f


@pytest.fixture
def sql_file(temp_dir: Path, sample_sql_schema: str) -> Path:
    """Create a sample SQL file."""
    f = temp_dir / "schema.sql"
    f.write_text(sample_sql_schema)
    return f


@pytest.fixture
def faq_file(temp_dir: Path, sample_faq_content: str) -> Path:
    """Create a sample FAQ file."""
    f = temp_dir / "faq.json"
    f.write_text(sample_faq_content)
    return f


@pytest.fixture
def mock_flatseek_provider() -> MagicMock:
    """Create a mock FlatSeek provider."""
    mock = MagicMock()
    mock.search.return_value = [
        {"id": 1, "title": "Python Tutorial", "body": "Learn Python..."},
        {"id": 2, "title": "JavaScript Tutorial", "body": "Learn JavaScript..."},
    ]
    mock.export.return_value = [
        {"id": 1, "title": "Python Tutorial", "body": "Learn Python..."},
        {"id": 2, "title": "JavaScript Tutorial", "body": "Learn JavaScript..."},
    ]
    mock.stats.return_value = {"total": 2, "indexed_at": "2024-01-01"}
    mock.columns.return_value = ["id", "title", "body"]
    return mock


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock configuration object."""
    from flattune.config import FlatTuneConfig, DatasetConfig, TrainConfig
    config = MagicMock(spec=FlatTuneConfig)
    config.name = "test-project"
    config.output_dir = "outputs/test"
    config.dataset = DatasetConfig()
    config.train = TrainConfig()
    return config
