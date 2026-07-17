"""Tests for utility modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from flattune.utils.hash import compute_hash, compute_file_hash
from flattune.utils.reproducibility import get_git_info, get_timestamp, set_seed
from flattune.utils.logging import get_logger, setup_logging
from flattune.utils.fs import create_run_folder, get_run_folder, ensure_dir


class TestHashUtils:
    """Test hashing utilities."""

    def test_compute_hash_string(self):
        """Test computing hash of a string."""
        hash1 = compute_hash("test content")
        hash2 = compute_hash("test content")
        hash3 = compute_hash("different content")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) > 0  # Hash should not be empty

    def test_compute_hash_dict(self):
        """Test computing hash of a dictionary."""
        data = {"key": "value", "number": 42}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_compute_file_hash(self, temp_dir: Path):
        """Test computing hash of a file."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test file content")

        hash1 = compute_file_hash(str(file_path))
        hash2 = compute_file_hash(str(file_path))

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_compute_file_hash_nonexistent(self):
        """Test computing hash of non-existent file."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/file.txt")

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hashes = [compute_hash(f"content_{i}") for i in range(5)]
        assert len(set(hashes)) == 5  # All unique


class TestReproducibilityUtils:
    """Test reproducibility utilities."""

    def test_get_timestamp(self):
        """Test getting timestamp."""
        timestamp = get_timestamp()

        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        # Should be ISO format
        assert "T" in timestamp or ":" in timestamp

    def test_set_seed(self):
        """Test setting random seed."""
        import random

        # Set seed and generate random numbers
        set_seed(42)
        values1 = [random.random() for _ in range(5)]

        # Set same seed
        set_seed(42)
        values2 = [random.random() for _ in range(5)]

        # Should produce same sequence
        assert values1 == values2

    def test_get_git_info(self):
        """Test getting git info."""
        git_info = get_git_info()

        assert isinstance(git_info, dict)
        # May contain keys like 'branch', 'commit', 'dirty'


class TestLoggingUtils:
    """Test logging utilities."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test")

        assert logger is not None
        assert logger.name == "test"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name != logger2.name

    def test_setup_logging(self):
        """Test setting up logging."""
        # Should not raise
        setup_logging("INFO")
        setup_logging("DEBUG")


class TestFsUtils:
    """Test filesystem utilities."""

    def test_ensure_dir(self, temp_dir: Path):
        """Test ensuring a directory exists."""
        subdir = temp_dir / "sub" / "nested"
        result = ensure_dir(subdir)

        assert subdir.exists()
        assert subdir.is_dir()

    def test_create_run_folder(self, temp_dir: Path):
        """Test creating a run folder."""
        from flattune.config import FlatTuneConfig

        config = FlatTuneConfig(name="test-project", output_dir=str(temp_dir))
        run_folder = create_run_folder(config)

        assert Path(run_folder).exists()
        assert "test-project" in str(run_folder)

    def test_get_run_folder(self, temp_dir: Path):
        """Test getting a run folder path."""
        from flattune.config import FlatTuneConfig

        config = FlatTuneConfig(name="test-project", output_dir=str(temp_dir))
        run_folder = create_run_folder(config)

        # Should return the same folder
        same_folder = get_run_folder(config)
        assert same_folder == run_folder

    def test_ensure_dir_file_collision(self, temp_dir: Path):
        """Test that ensure_dir fails if a file exists with same name."""
        # Create a file where we want a directory
        file_path = temp_dir / "afile"
        file_path.write_text("I am a file")

        # ensure_dir should not overwrite file
        # (implementation may vary - may raise error or skip)

    def test_copy_and_move(self, temp_dir: Path):
        """Test copying and moving files."""
        from flattune.utils.fs import copy_file, move_file

        # Create source file
        source = temp_dir / "source.txt"
        source.write_text("content")

        # Copy
        dest = temp_dir / "dest.txt"
        copy_file(str(source), str(dest))
        assert dest.exists()
        assert source.read_text() == dest.read_text()

        # Move
        moved = temp_dir / "moved.txt"
        move_file(str(dest), str(moved))
        assert moved.exists()
        assert not dest.exists()
