"""Dataset splitting utilities."""

import random
from collections import Counter
from enum import Enum
from typing import Any


class SplitType(Enum):
    """Type of dataset split."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetSplitter:
    """Splits datasets into train/validation/test splits.

    Supports random splitting and stratified splitting based on labels.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratify_field: str | None = None,
    ):
        """Initialize the dataset splitter.

        Args:
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
            test_ratio: Ratio of data for testing.
            seed: Random seed for reproducibility.
            stratify_field: Optional field name to use for stratified splitting.
                           If provided, samples are split proportionally by label.

        Raises:
            ValueError: If ratios don't sum to 1.0 or stratify field is invalid.
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        if stratify_field is not None and stratify_field.strip() == "":
            stratify_field = None

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify_field = stratify_field

    def split(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split samples into train and validation sets.

        Args:
            samples: List of samples to split.

        Returns:
            Tuple of (train_samples, val_samples).
        """
        if not samples:
            return [], []

        # Use stratified split if field is specified
        if self.stratify_field:
            return self._stratified_split_two(samples)

        return self._random_split_two(samples)

    def split_three(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split samples into train, validation, and test sets.

        Args:
            samples: List of samples to split.

        Returns:
            Tuple of (train_samples, val_samples, test_samples).
        """
        if not samples:
            return [], [], []

        # Use stratified split if field is specified
        if self.stratify_field:
            return self._stratified_split_three(samples)

        return self._random_split_three(samples)

    def _random_split_two(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform random split into two sets."""
        random.seed(self.seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * self.train_ratio)

        train_samples = shuffled[:train_end]
        val_samples = shuffled[train_end:]

        return train_samples, val_samples

    def _random_split_three(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform random split into three sets."""
        random.seed(self.seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_samples = shuffled[:train_end]
        val_samples = shuffled[train_end:val_end]
        test_samples = shuffled[val_end:]

        return train_samples, val_samples, test_samples

    def _stratified_split_two(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform stratified split into two sets based on a label field."""
        # Group samples by label
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            label = self._get_label(sample)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)

        train_samples = []
        val_samples = []

        random.seed(self.seed)

        for label, group in label_groups.items():
            shuffled = group.copy()
            random.shuffle(shuffled)

            total = len(shuffled)
            train_end = int(total * self.train_ratio)

            train_samples.extend(shuffled[:train_end])
            val_samples.extend(shuffled[train_end:])

        # Shuffle final results
        random.shuffle(train_samples)
        random.shuffle(val_samples)

        return train_samples, val_samples

    def _stratified_split_three(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform stratified split into three sets based on a label field."""
        # Group samples by label
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            label = self._get_label(sample)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        random.seed(self.seed)

        for label, group in label_groups.items():
            shuffled = group.copy()
            random.shuffle(shuffled)

            total = len(shuffled)
            train_end = int(total * self.train_ratio)
            val_end = train_end + int(total * self.val_ratio)

            train_samples.extend(shuffled[:train_end])
            val_samples.extend(shuffled[train_end:val_end])
            test_samples.extend(shuffled[val_end:])

        # Shuffle final results
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        return train_samples, val_samples, test_samples

    def _get_label(self, sample: dict[str, Any]) -> str:
        """Extract label from sample for stratification."""
        # Check metadata first
        if "metadata" in sample and isinstance(sample["metadata"], dict):
            metadata = sample["metadata"]
            if self.stratify_field in metadata:
                return str(metadata[self.stratify_field])

        # Check output field (for classification)
        if self.stratify_field == "output" and "output" in sample:
            return str(sample["output"])

        # Check top-level field
        if self.stratify_field in sample:
            return str(sample[self.stratify_field])

        return "unknown"

    def get_split_statistics(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get statistics about sample distribution.

        Args:
            samples: List of samples to analyze.

        Returns:
            Dictionary with distribution statistics.
        """
        if not samples:
            return {"total": 0, "by_label": {}}

        label_counts: Counter = Counter()
        for sample in samples:
            label = self._get_label(sample)
            label_counts[label] += 1

        return {
            "total": len(samples),
            "unique_labels": len(label_counts),
            "by_label": dict(label_counts),
            "label_distribution": {
                label: count / len(samples) for label, count in label_counts.items()
            },
        }
