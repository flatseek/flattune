"""Benchmark reporter for generating markdown reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkReporter:
    """Generates markdown reports from benchmark results."""

    def __init__(
        self,
        output_dir: Path | None = None,
    ):
        """Initialize the benchmark reporter.

        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir or "benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: dict[str, Any],
        output_name: str = "benchmark_report",
    ) -> dict[str, Path]:
        """Generate reports from benchmark results.

        Creates three output files:
        - report.md: Human-readable markdown report
        - metrics.json: Machine-readable metrics summary
        - benchmark.json: Full benchmark data

        Args:
            results: Benchmark results dictionary.
            output_name: Base name for the output files.

        Returns:
            Dictionary mapping file type to Path.
        """
        output_paths = {}

        # Generate markdown report
        report_path = self.output_dir / f"{output_name}.md"
        report_content = self._generate_markdown_report(results)
        report_path.write_text(report_content)
        output_paths["report.md"] = report_path
        logger.info(f"Report saved to: {report_path}")

        # Generate metrics JSON
        metrics_path = self.output_dir / f"{output_name}_metrics.json"
        metrics = self._extract_metrics(results)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        output_paths["metrics.json"] = metrics_path
        logger.info(f"Metrics saved to: {metrics_path}")

        # Generate full benchmark JSON
        benchmark_path = self.output_dir / f"{output_name}_benchmark.json"
        benchmark_path.write_text(json.dumps(results, indent=2))
        output_paths["benchmark.json"] = benchmark_path
        logger.info(f"Benchmark data saved to: {benchmark_path}")

        return output_paths

    def _generate_markdown_report(self, results: dict[str, Any]) -> str:
        """Generate a markdown report from benchmark results.

        Args:
            results: Benchmark results dictionary.

        Returns:
            Markdown string for the report.
        """
        lines = [
            "# Benchmark Report",
            "",
            f"**Status**: {results.get('status', 'unknown')}",
            f"**Backend**: {results.get('backend', 'unknown')}",
            f"**Date**: {self._get_timestamp()}",
            "",
            "## Summary",
            "",
        ]

        # Extract summary metrics
        backend = results.get("backend", "unknown")
        categories = results.get("categories_run", [])

        # Handle different result structures
        if "results" in results:
            # New structure with nested results
            benchmark_results = results.get("results", {})
            categories_data = benchmark_results.get("categories", {})

            # Summary table
            lines.extend(
                [
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            # Extract key metrics
            if "tokens_per_sec" in categories_data:
                tps_data = categories_data["tokens_per_sec"]
                tps = tps_data.get("average_tokens_per_second", 0)
                lines.append(f"| Avg Tokens/sec | {tps:.2f} |")

            if "latency" in categories_data:
                lat_data = categories_data["latency"]
                lat = lat_data.get("average_latency_seconds", 0)
                lines.append(f"| Avg Latency | {lat:.3f}s |")

            lines.extend(
                [
                    f"| Backend | {backend} |",
                    f"| Categories Run | {', '.join(categories)} |",
                    "",
                ]
            )

            # Detailed category results
            lines.extend(
                [
                    "## Detailed Results",
                    "",
                ]
            )

            for category in categories:
                if category in categories_data:
                    lines.extend(self._format_category_results(category, categories_data[category]))

        else:
            # Legacy structure
            lines.extend(
                [
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Total Runs | {results.get('total_runs', 0)} |",
                    f"| Total Tokens | {results.get('total_tokens', 0)} |",
                    f"| Total Time | {results.get('total_time', 0):.2f}s |",
                    f"| Wall Time | {results.get('wall_time', 0):.2f}s |",
                    f"| Avg Tokens/sec | {results.get('average_tokens_per_second', 0):.2f} |",
                    "",
                ]
            )

            # Individual results
            individual = results.get("individual_results", [])
            if individual:
                lines.extend(
                    [
                        "## Individual Results",
                        "",
                        "| Run | Prompt | Tokens | Time (s) | Tokens/sec |",
                        "|-----|--------|--------|----------|-----------|",
                    ]
                )

                for i, r in enumerate(individual[:50]):
                    lines.append(
                        f"| {i} | {r.get('prompt_index', 0)} | "
                        f"{r.get('tokens', 0)} | {r.get('elapsed', 0):.2f} | "
                        f"{r.get('tokens_per_second', 0):.2f} |"
                    )

                if len(individual) > 50:
                    lines.append(f"\n*... and {len(individual) - 50} more results*")

        # Add configuration section
        lines.extend(
            [
                "",
                "## Configuration",
                "",
                f"- Temperature: {results.get('temperature', 0)}",
                f"- Max Tokens: {results.get('max_tokens', 0)}",
                f"- Num Prompts: {results.get('num_prompts', 0)}",
                f"- Num Runs per Prompt: {results.get('num_runs', 0)}",
                "",
            ]
        )

        return "\n".join(lines)

    def _format_category_results(self, category: str, data: dict[str, Any]) -> list[str]:
        """Format results for a single category.

        Args:
            category: Category name.
            data: Category results data.

        Returns:
            List of formatted markdown lines.
        """
        lines = [
            f"### {self._format_category_name(category)}",
            "",
        ]

        if category == "latency":
            lines.extend(
                [
                    f"- Average Latency: {data.get('average_latency_seconds', 0):.3f}s",
                    f"- Number of Runs: {len(data.get('results', []))}",
                ]
            )

        elif category == "tokens_per_sec":
            lines.extend(
                [
                    f"- Average Tokens/sec: {data.get('average_tokens_per_second', 0):.2f}",
                    f"- Total Tokens: {data.get('total_tokens', 0)}",
                    f"- Total Time: {data.get('total_time', 0):.2f}s",
                ]
            )

        elif category in [
            "prompt_quality",
            "instruction_following",
            "domain_accuracy",
            "hallucination_tests",
            "regression_tests",
        ]:
            num_prompts = data.get("num_prompts", 0)
            successful = data.get("successful_generations", 0)
            lines.extend(
                [
                    f"- Prompts: {num_prompts}",
                    f"- Successful Generations: {successful}",
                ]
            )

            # Show sample results
            results_list = data.get("results", [])
            if results_list:
                lines.append("")
                lines.append("**Sample Results:**")
                for _i, r in enumerate(results_list[:3]):
                    prompt = r.get("prompt", "")[:50]
                    response = r.get("response", "")[:100]
                    lines.append(f'- Prompt: "{prompt}..."')
                    lines.append(f'  Response: "{response}..."')

        lines.append("")
        return lines

    def _format_category_name(self, category: str) -> str:
        """Format a category name for display.

        Args:
            category: Raw category name.

        Returns:
            Formatted category name.
        """
        return category.replace("_", " ").title()

    def _extract_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Extract key metrics from results.

        Args:
            results: Benchmark results dictionary.

        Returns:
            Dictionary of key metrics.
        """
        metrics = {
            "status": results.get("status", "unknown"),
            "backend": results.get("backend", "unknown"),
            "timestamp": self._get_timestamp(),
            "categories": {},
        }

        # Handle new structure
        if "results" in results:
            benchmark_results = results.get("results", {})
            categories_data = benchmark_results.get("categories", {})

            for category, data in categories_data.items():
                if category == "latency":
                    metrics["categories"]["latency"] = {
                        "average_seconds": data.get("average_latency_seconds", 0),
                    }
                elif category == "tokens_per_sec":
                    metrics["categories"]["tokens_per_sec"] = {
                        "average": data.get("average_tokens_per_second", 0),
                        "total_tokens": data.get("total_tokens", 0),
                        "total_time": data.get("total_time", 0),
                    }
                else:
                    metrics["categories"][category] = {
                        "num_prompts": data.get("num_prompts", 0),
                        "successful_generations": data.get("successful_generations", 0),
                    }

            # Also include summary if available
            if "summary" in benchmark_results:
                metrics["summary"] = benchmark_results["summary"]

        else:
            # Legacy structure
            metrics["total_runs"] = results.get("total_runs", 0)
            metrics["total_tokens"] = results.get("total_tokens", 0)
            metrics["total_time"] = results.get("total_time", 0)
            metrics["average_tokens_per_second"] = results.get("average_tokens_per_second", 0)

        return metrics

    def generate_comparison(
        self,
        results_before: dict[str, Any],
        results_after: dict[str, Any],
        output_name: str = "benchmark_comparison",
    ) -> dict[str, Path]:
        """Generate a comparison report between two benchmark runs.

        Creates three output files:
        - report.md: Human-readable comparison report
        - metrics.json: Machine-readable comparison metrics
        - benchmark.json: Full comparison data

        Args:
            results_before: Benchmark results from before training.
            results_after: Benchmark results from after training.
            output_name: Base name for the output files.

        Returns:
            Dictionary mapping file type to Path.
        """
        output_paths = {}

        # Generate markdown report
        report_path = self.output_dir / f"{output_name}.md"
        report_content = self._generate_markdown_comparison(results_before, results_after)
        report_path.write_text(report_content)
        output_paths["report.md"] = report_path
        logger.info(f"Comparison report saved to: {report_path}")

        # Generate metrics JSON
        metrics_path = self.output_dir / f"{output_name}_metrics.json"
        metrics = self._extract_comparison_metrics(results_before, results_after)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        output_paths["metrics.json"] = metrics_path
        logger.info(f"Comparison metrics saved to: {metrics_path}")

        # Generate full benchmark JSON
        benchmark_path = self.output_dir / f"{output_name}_benchmark.json"
        comparison_data = {
            "before": results_before,
            "after": results_after,
            "timestamp": self._get_timestamp(),
        }
        benchmark_path.write_text(json.dumps(comparison_data, indent=2))
        output_paths["benchmark.json"] = benchmark_path
        logger.info(f"Comparison data saved to: {benchmark_path}")

        return output_paths

    def _generate_markdown_comparison(
        self,
        results_before: dict[str, Any],
        results_after: dict[str, Any],
    ) -> str:
        """Generate a markdown comparison report.

        Args:
            results_before: Benchmark results from before training.
            results_after: Benchmark results from after training.

        Returns:
            Markdown string for the comparison report.
        """
        lines = [
            "# Benchmark Comparison Report",
            "",
            f"**Date**: {self._get_timestamp()}",
            "",
            "## Performance Comparison",
            "",
            "| Metric | Before | After | Change |",
            "|--------|--------|-------|--------|",
        ]

        # Extract key metrics for comparison
        before_tps = self._extract_tokens_per_second(results_before)
        after_tps = self._extract_tokens_per_second(results_after)
        tps_improvement = ((after_tps - before_tps) / before_tps * 100) if before_tps > 0 else 0

        before_lat = self._extract_latency(results_before)
        after_lat = self._extract_latency(results_after)
        lat_improvement = ((before_lat - after_lat) / before_lat * 100) if before_lat > 0 else 0

        lines.extend(
            [
                f"| Tokens/sec | {before_tps:.2f} | {after_tps:.2f} | {tps_improvement:+.1f}% |",
                f"| Latency (s) | {before_lat:.3f} | {after_lat:.3f} | {lat_improvement:+.1f}% |",
            ]
        )

        # Category-specific comparisons
        if "results" in results_before and "results" in results_after:
            before_cats = results_before.get("results", {}).get("categories", {})
            after_cats = results_after.get("results", {}).get("categories", {})

            lines.extend(
                [
                    "",
                    "## Category Comparisons",
                    "",
                ]
            )

            for category in before_cats.keys():
                if category in after_cats:
                    lines.extend(
                        self._format_category_comparison(
                            category, before_cats[category], after_cats[category]
                        )
                    )

        # Interpretation
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
            ]
        )

        if tps_improvement > 5 and lat_improvement > 0:
            lines.append(
                "The fine-tuned model shows **improved** performance in both speed and latency."
            )
        elif tps_improvement > 5:
            lines.append("The fine-tuned model shows **improved** throughput but latency may vary.")
        elif tps_improvement < -5:
            lines.append("The fine-tuned model shows **degraded** throughput performance.")
        else:
            lines.append("Performance is **similar** between base and fine-tuned models.")

        lines.extend(
            [
                "",
                "*Note: Quality of outputs should be evaluated separately from speed metrics.*",
                "",
            ]
        )

        return "\n".join(lines)

    def _format_category_comparison(
        self,
        category: str,
        before_data: dict[str, Any],
        after_data: dict[str, Any],
    ) -> list[str]:
        """Format comparison results for a category.

        Args:
            category: Category name.
            before_data: Before results.
            after_data: After results.

        Returns:
            List of formatted markdown lines.
        """
        lines = [
            f"### {self._format_category_name(category)}",
            "",
            f"- Before: {self._summarize_category_data(before_data)}",
            f"- After: {self._summarize_category_data(after_data)}",
            "",
        ]
        return lines

    def _summarize_category_data(self, data: dict[str, Any]) -> str:
        """Summarize category data for display.

        Args:
            data: Category data.

        Returns:
            Summary string.
        """
        if "average_tokens_per_second" in data:
            return f"{data['average_tokens_per_second']:.2f} tokens/sec"
        elif "average_latency_seconds" in data:
            return f"{data['average_latency_seconds']:.3f}s latency"
        elif "successful_generations" in data:
            return f"{data['successful_generations']}/{data['num_prompts']} successful"
        return "N/A"

    def _extract_tokens_per_second(self, results: dict[str, Any]) -> float:
        """Extract tokens per second from results.

        Args:
            results: Benchmark results.

        Returns:
            Average tokens per second.
        """
        if "results" in results:
            categories = results.get("results", {}).get("categories", {})
            tps_data = categories.get("tokens_per_sec", {})
            return tps_data.get("average_tokens_per_second", 0)
        return results.get("average_tokens_per_second", 0)

    def _extract_latency(self, results: dict[str, Any]) -> float:
        """Extract latency from results.

        Args:
            results: Benchmark results.

        Returns:
            Average latency in seconds.
        """
        if "results" in results:
            categories = results.get("results", {}).get("categories", {})
            lat_data = categories.get("latency", {})
            return lat_data.get("average_latency_seconds", 0)
        return 0

    def _extract_comparison_metrics(
        self,
        results_before: dict[str, Any],
        results_after: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract comparison metrics.

        Args:
            results_before: Benchmark results from before training.
            results_after: Benchmark results from after training.

        Returns:
            Dictionary of comparison metrics.
        """
        before_tps = self._extract_tokens_per_second(results_before)
        after_tps = self._extract_tokens_per_second(results_after)
        before_lat = self._extract_latency(results_before)
        after_lat = self._extract_latency(results_after)

        metrics = {
            "timestamp": self._get_timestamp(),
            "before": {
                "tokens_per_second": before_tps,
                "latency_seconds": before_lat,
            },
            "after": {
                "tokens_per_second": after_tps,
                "latency_seconds": after_lat,
            },
            "change": {
                "tokens_per_second_percent": ((after_tps - before_tps) / before_tps * 100)
                if before_tps > 0
                else 0,
                "latency_percent": ((before_lat - after_lat) / before_lat * 100)
                if before_lat > 0
                else 0,
            },
        }

        return metrics

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_summary_table(
        self,
        results_list: list[dict[str, Any]],
        output_name: str = "benchmark_summary",
    ) -> Path:
        """Generate a summary table from multiple benchmark runs.

        Args:
            results_list: List of benchmark results.
            output_name: Base name for the output file.

        Returns:
            Path to the generated report.
        """
        report_path = self.output_dir / f"{output_name}.md"

        lines = [
            "# Benchmark Summary Table",
            "",
            f"**Date**: {self._get_timestamp()}",
            "",
            "| Run | Backend | Tokens/sec | Latency | Status |",
            "|------|---------|------------|---------|--------|",
        ]

        for i, results in enumerate(results_list):
            backend = results.get("backend", "unknown")
            tps = self._extract_tokens_per_second(results)
            lat = self._extract_latency(results)
            status = results.get("status", "unknown")

            lines.append(f"| {i + 1} | {backend} | {tps:.2f} | {lat:.3f}s | {status} |")

        report_path.write_text("\n".join(lines))
        logger.info(f"Summary table saved to: {report_path}")

        return report_path
