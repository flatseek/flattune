"""Reproducibility utilities for FlatTune."""

import random
import subprocess
from datetime import datetime
from typing import Any

import numpy as np


def get_git_info() -> dict[str, Any]:
    """Get git repository information.

    Returns:
        Dictionary containing git branch, commit, and status.
    """
    info = {
        "branch": "unknown",
        "commit": "unknown",
        "is_dirty": False,
    }

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Get current commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:8]  # Short hash

        # Check if repo is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["is_dirty"] = bool(result.stdout.strip())

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return info


def get_timestamp() -> str:
    """Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string.
    """
    return datetime.now().isoformat()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Seed value to set.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Try PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_environment_info() -> dict[str, Any]:
    """Get environment information for reproducibility.

    Returns:
        Dictionary containing environment details.
    """
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # Try to get package versions
    packages = ["torch", "transformers", "unsloth", "flatseek"]
    for package in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            info[package] = version
        except (ImportError, NotImplementedError):
            info[package] = "not installed"

    return info


class ReproducibilityContext:
    """Context for reproducible runs."""

    def __init__(
        self,
        seed: int,
        include_env: bool = True,
    ):
        """Initialize reproducibility context.

        Args:
            seed: Random seed.
            include_env: Whether to include environment info.
        """
        self.seed = seed
        self.include_env = include_env
        self._original_seeds = {}

    def __enter__(self):
        """Enter context and set seeds."""
        # Save original seeds
        try:
            import torch

            self._original_seeds["torch"] = torch.initial_seed()
        except ImportError:
            pass

        # Set new seeds
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore seeds if needed."""
        return False  # Don't suppress exceptions

    def get_info(self) -> dict[str, Any]:
        """Get reproducibility information dictionary.

        Returns:
            Dictionary with seed, git, timestamp, and environment info.
        """
        info = {
            "seed": self.seed,
            "git": get_git_info(),
            "timestamp": get_timestamp(),
        }

        if self.include_env:
            info["environment"] = get_environment_info()

        return info
