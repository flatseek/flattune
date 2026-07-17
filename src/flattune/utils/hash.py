"""Hashing utilities for FlatTune."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def compute_hash(
    data: Any,
    algorithm: str = "sha256",
) -> str:
    """Compute hash of data.

    Args:
        data: Data to hash (string, bytes, or dict/list).
        algorithm: Hash algorithm to use.

    Returns:
        Hexadecimal hash string.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True).encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compute_file_hash(
    path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """Compute hash of a file.

    Args:
        path: Path to file.
        algorithm: Hash algorithm to use.
        chunk_size: Size of chunks to read.

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_dataset_hash(dataset_path: Path) -> str:
    """Compute hash of a dataset directory.

    Args:
        dataset_path: Path to dataset directory.

    Returns:
        Hexadecimal hash string.
    """
    if dataset_path.is_file():
        return compute_file_hash(dataset_path)

    # Hash all files in directory
    all_hashes = []
    for file_path in sorted(dataset_path.rglob("*")):
        if file_path.is_file():
            file_hash = compute_file_hash(file_path)
            all_hashes.append(f"{file_path.name}:{file_hash}")

    combined = "|".join(all_hashes)
    return compute_hash(combined)


def verify_hash(
    data: Any,
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """Verify data matches expected hash.

    Args:
        data: Data to verify.
        expected_hash: Expected hash value.
        algorithm: Hash algorithm to use.

    Returns:
        True if hash matches.
    """
    actual_hash = compute_hash(data, algorithm)
    return actual_hash == expected_hash
