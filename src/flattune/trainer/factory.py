"""Trainer factory for creating appropriate trainer instances."""

import logging
from pathlib import Path
from typing import Any

from flattune.config import TrainBackend
from flattune.trainer.base import TrainerBase
from flattune.trainer.transformers_trainer import TransformersTrainer
from flattune.trainer.unsloth_trainer import UnslothTrainer

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory for creating trainer instances based on backend selection.

    Supports automatic fallback:
    - If Unsloth is requested but not available, falls back to Transformers
    - Transformers trainer works on CPU/MPS without CUDA
    """

    _trainers = {
        TrainBackend.UNSLOTH: UnslothTrainer,
        TrainBackend.TRANSFORMERS: TransformersTrainer,
        TrainBackend.AXOLOTL: TransformersTrainer,  # Fallback
        TrainBackend.MLX_LM: TransformersTrainer,  # Fallback
    }

    @classmethod
    def create(
        cls,
        backend: TrainBackend,
        model_path: str,
        dataset_path: str | None = None,
        output_dir: Path | None = None,
        config: Any | None = None,
        flatseek_provider: Any | None = None,
        flatseek_query: str | None = None,
    ) -> TrainerBase:
        """Create a trainer instance for the specified backend.

        Args:
            backend: Training backend to use.
            model_path: Path to the base model.
            dataset_path: Path to the training dataset.
            output_dir: Directory for outputs.
            config: Training configuration.
            flatseek_provider: FlatSeek provider for streaming (optional).
            flatseek_query: Query to stream from FlatSeek (optional).

        Returns:
            Trainer instance for the specified backend.

        Raises:
            ValueError: If the backend is not supported.
        """
        trainer_class = cls._trainers.get(backend)

        if trainer_class is None:
            available = [b.value for b in cls._trainers.keys()]
            raise ValueError(
                f"Unsupported training backend: {backend.value}. Available backends: {available}"
            )

        # Check if we need to fallback to transformers
        if trainer_class == UnslothTrainer:
            try:
                import unsloth  # noqa
            except ImportError:
                logger.warning(
                    "Unsloth not available, falling back to Transformers trainer. "
                    "Install unsloth for GPU-accelerated training."
                )
                trainer_class = TransformersTrainer

        # Create trainer instance
        return trainer_class(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
            flatseek_provider=flatseek_provider,
            flatseek_query=flatseek_query,
        )

    @classmethod
    def register(cls, backend: TrainBackend, trainer_class: type[TrainerBase]) -> None:
        """Register a new trainer backend.

        Args:
            backend: Backend identifier.
            trainer_class: Trainer class to register.
        """
        if not issubclass(trainer_class, TrainerBase):
            raise TypeError(f"{trainer_class} must inherit from TrainerBase")
        cls._trainers[backend] = trainer_class

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all available training backends.

        Returns:
            List of backend names.
        """
        return [b.value for b in cls._trainers.keys()]
