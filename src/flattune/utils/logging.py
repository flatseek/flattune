"""Logging utilities for FlatTune."""

import logging
import sys
from pathlib import Path

# Global logger configuration
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    stage: str | None = None,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Set up logging configuration.

    Args:
        stage: Optional stage name (build, train, etc.) for file logging.
        level: Logging level.
        log_file: Optional file path for logging output.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if stage specified
    if stage:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{stage}.log")
        handlers.append(file_handler)

    logging.basicConfig(
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        level=level,
        handlers=handlers,
    )

    # Set specific library log levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("unsloth").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for staged logging."""

    def __init__(
        self,
        logger: logging.Logger,
        stage: str,
        message: str | None = None,
    ):
        """Initialize log context.

        Args:
            logger: Logger instance.
            stage: Stage name for indentation.
            message: Optional message to log on entry.
        """
        self.logger = logger
        self.stage = stage
        self.message = message

    def __enter__(self):
        """Enter context."""
        if self.message:
            self.logger.info(f"[{self.stage}] {self.message}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type:
            self.logger.error(f"[{self.stage}] Error: {exc_val}")
        return False
