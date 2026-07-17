"""Build exporter for structured output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flattune.build.planner import BuildPlan
from flattune.build.registry import DatasetTypeRegistry

logger = logging.getLogger(__name__)


class BuildExporter:
    """Exports generated samples to the target directory structure."""

    def export(
        self,
        samples_by_type: dict[str, list[dict]],
        output_dir: Path,
        plan: BuildPlan,
    ) -> dict[str, Path]:
        """Export samples to structured output directories.

        Args:
            samples_by_type: Dictionary mapping type name to samples.
            output_dir: Root output directory.
            plan: Build plan with output structure.

        Returns:
            Dictionary mapping category/type to output path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # Group by category and save
        for type_name, samples in samples_by_type.items():
            if not samples:
                continue

            # Get category from registry
            dt = DatasetTypeRegistry.get(type_name)
            category = dt.category if dt else "uncategorized"

            # Create category directory
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)

            # Save type file
            output_path = category_dir / f"{type_name}.jsonl"
            self._write_jsonl(samples, output_path)
            paths[f"{category}/{type_name}"] = output_path
            logger.info(f"Wrote {len(samples)} samples to {output_path}")

        # Save metadata
        metadata = self._generate_metadata(plan, samples_by_type)
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        paths["metadata"] = metadata_path

        # Save build report
        report = self._generate_report(plan, samples_by_type)
        report_path = output_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        paths["report"] = report_path

        return paths

    def _write_jsonl(self, samples: list[dict], output_path: Path) -> None:
        """Write samples to JSONL file.

        Args:
            samples: List of sample dictionaries.
            output_path: Output file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                # Ensure metadata is present
                if "metadata" not in sample:
                    sample["metadata"] = {}
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _generate_metadata(self, plan: BuildPlan, samples_by_type: dict[str, list[dict]]) -> dict[str, Any]:
        """Generate metadata.json content.

        Args:
            plan: Build plan.
            samples_by_type: Generated samples.

        Returns:
            Metadata dictionary.
        """
        total = sum(len(s) for s in samples_by_type.values())

        return {
            "plan": plan.to_display_dict(),
            "generated": {
                "total_samples": total,
                "by_type": {t: len(s) for t, s in samples_by_type.items() if s},
            },
            "timestamp": self._get_timestamp(),
        }

    def _generate_report(self, plan: BuildPlan, samples_by_type: dict[str, list[dict]]) -> dict[str, Any]:
        """Generate report.json content.

        Args:
            plan: Build plan.
            samples_by_type: Generated samples.

        Returns:
            Report dictionary.
        """
        return {
            "source": plan.source,
            "source_type": plan.source_type.value,
            "selected_types": [t.type_name for t in plan.selected_types],
            "total_samples": sum(len(s) for s in samples_by_type.values()),
            "samples_by_type": {t: len(s) for t, s in samples_by_type.items() if s},
            "output_structure": plan.output_structure,
            "metadata": plan.metadata,
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
