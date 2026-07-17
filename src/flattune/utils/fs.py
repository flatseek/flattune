"""Filesystem utilities for FlatTune."""

import shutil
from pathlib import Path

from flattune.config import FlatTuneConfig


def create_run_folder(config: FlatTuneConfig) -> Path:
    """Create a new run folder with timestamp.

    Args:
        config: FlatTune configuration.

    Returns:
        Path to created run folder.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.name}_{timestamp}"
    run_dir = Path(config.output_dir) / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "dataset").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "merged").mkdir(exist_ok=True)
    (run_dir / "export").mkdir(exist_ok=True)
    (run_dir / "benchmark").mkdir(exist_ok=True)

    # Save config to run folder
    config.to_yaml(str(run_dir / "config.yaml"))

    return run_dir


def get_run_folder(config: FlatTuneConfig) -> Path:
    """Get the most recent run folder for a config.

    Args:
        config: FlatTune configuration.

    Returns:
        Path to the most recent run folder.
    """
    output_dir = Path(config.output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Find run directories matching the config name
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(config.name)]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for {config.name} in {output_dir}")

    # Return most recently modified
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    shutil.copy2(src, dst)


def move_file(src: Path, dst: Path) -> None:
    """Move a file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    shutil.move(src, dst)


def clean_directory(path: Path) -> None:
    """Remove all contents of a directory.

    Args:
        path: Directory path to clean.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
