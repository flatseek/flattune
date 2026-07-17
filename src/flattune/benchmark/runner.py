"""Benchmark runner for evaluating models."""

import json
import time
from pathlib import Path
from typing import Any

from flattune.config import BenchmarkBackend, BenchmarkConfig
from flattune.lmstudio.client import LMStudioClient
from flattune.ollama.client import OllamaClient
from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runs benchmarks against trained models.

    Supports multiple backends:
    - LM Studio: Local GGUF models via LM Studio API
    - Ollama: Local models via Ollama API
    - Transformers: Native HuggingFace Transformers (future)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        output_dir: Path | None = None,
    ):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
            output_dir: Directory to save benchmark results.
        """
        self.config = config
        self.output_dir = Path(output_dir or "benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._transformers_client = None

    def _create_client(self):
        """Create the benchmark client based on backend."""
        if self.config.backend == BenchmarkBackend.LMSTUDIO:
            self._client = LMStudioClient()
        elif self.config.backend == BenchmarkBackend.OLLAMA:
            self._client = OllamaClient()
        elif self.config.backend == BenchmarkBackend.TRANSFORMERS:
            self._client = self._create_transformers_client()
        else:
            raise ValueError(f"Unsupported benchmark backend: {self.config.backend}")

    def _create_transformers_client(self):
        """Create a transformers-based client for benchmarking.

        Returns:
            A client-like object for Transformers benchmarking.
        """
        try:
            from transformers import pipeline

            return TransformersClient(
                model_name=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except ImportError:
            raise RuntimeError("Transformers is not installed")

    @property
    def client(self):
        """Get the benchmark client, creating it if necessary."""
        if self._client is None:
            self._create_client()
        return self._client

    def is_backend_available(self) -> bool:
        """Check if the benchmark backend is available.

        Returns:
            True if the backend service is running.
        """
        return self.client.is_available()

    def run(
        self,
        prompts: list[str],
        model_path: str | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run benchmark with the given prompts.

        Args:
            prompts: List of prompt texts to benchmark.
            model_path: Optional path to model (for loading into backend).
            categories: Optional list of benchmark categories to run.

        Returns:
            Dictionary containing benchmark results.
        """
        categories = categories or [
            "latency",
            "tokens_per_sec",
            "prompt_quality",
            "instruction_following",
            "domain_accuracy",
            "hallucination_tests",
            "regression_tests",
        ]

        logger.info(f"Running benchmark with {len(prompts)} prompts")
        logger.info(f"Backend: {self.config.backend.value}")
        logger.info(f"Categories: {categories}")

        # Check backend availability
        if not self.is_backend_available():
            logger.warning(f"{self.config.backend.value} is not available")
            return {
                "status": "error",
                "error": f"{self.config.backend.value} is not running",
                "prompts": len(prompts),
            }

        # Import model if path provided
        if model_path and hasattr(self.client, "import_model"):
            logger.info(f"Importing model from {model_path}")
            success = self.client.import_model(model_path)
            if not success:
                logger.warning("Failed to import model, continuing anyway")

        # Run benchmark
        start_time = time.time()

        # Use the benchmark method with categories
        if hasattr(self.client, "benchmark"):
            results = self.client.benchmark(
                categories=categories,
                prompts=prompts,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                num_runs=self.config.num_runs,
            )
        else:
            # Fallback to basic benchmark
            results = self.client.benchmark(
                prompts=prompts,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                num_runs=self.config.num_runs,
            )

        elapsed = time.time() - start_time

        # Compile benchmark report
        report = {
            "status": "success",
            "backend": self.config.backend.value,
            "categories_run": categories,
            "num_prompts": len(prompts),
            "num_runs": self.config.num_runs,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "wall_time": elapsed,
            "results": results,
        }

        # Save results
        self._save_results(report)

        return report

    def run_before_after(
        self,
        prompts: list[str],
        model_path_before: str | None = None,
        model_path_after: str | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run benchmarks before and after fine-tuning for comparison.

        Args:
            prompts: List of prompt texts to benchmark.
            model_path_before: Optional path to model before fine-tuning.
            model_path_after: Optional path to model after fine-tuning.
            categories: Optional list of benchmark categories.

        Returns:
            Dictionary containing before/after comparison results.
        """
        logger.info("Running before/after benchmark comparison")
        logger.info(f"Model before: {model_path_before}")
        logger.info(f"Model after: {model_path_after}")

        results_before = None
        results_after = None

        # Run benchmark on "before" model
        if model_path_before:
            logger.info("Benchmarking model BEFORE fine-tuning...")
            results_before = self.run(
                prompts=prompts,
                model_path=model_path_before,
                categories=categories,
            )

        # Run benchmark on "after" model
        if model_path_after:
            logger.info("Benchmarking model AFTER fine-tuning...")
            results_after = self.run(
                prompts=prompts,
                model_path=model_path_after,
                categories=categories,
            )

        # Compile comparison report
        comparison = {
            "status": "success",
            "backend": self.config.backend.value,
            "categories_run": categories or [],
            "before": results_before,
            "after": results_after,
            "comparison": self._compare_results(results_before, results_after),
        }

        # Save comparison results
        self._save_comparison_results(comparison)

        return comparison

    def _compare_results(
        self,
        results_before: dict[str, Any] | None,
        results_after: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compare before and after benchmark results.

        Args:
            results_before: Results from before fine-tuning.
            results_after: Results from after fine-tuning.

        Returns:
            Dictionary containing comparison metrics.
        """
        comparison = {
            "has_before": results_before is not None and results_before.get("status") == "success",
            "has_after": results_after is not None and results_after.get("status") == "success",
            "category_comparisons": {},
        }

        if not comparison["has_before"] or not comparison["has_after"]:
            return comparison

        # Extract results dictionaries
        before_data = results_before.get("results", {})
        after_data = results_after.get("results", {})

        # Compare each category
        for category in [
            "latency",
            "tokens_per_sec",
            "prompt_quality",
            "instruction_following",
            "domain_accuracy",
            "hallucination_tests",
            "regression_tests",
        ]:
            before_cat = before_data.get("categories", {}).get(category, {})
            after_cat = after_data.get("categories", {}).get(category, {})

            comparison["category_comparisons"][category] = {
                "before": before_cat,
                "after": after_cat,
                "change": self._calculate_change(before_cat, after_cat, category),
            }

        return comparison

    def _calculate_change(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        category: str,
    ) -> dict[str, Any]:
        """Calculate the change between before and after results.

        Args:
            before: Before results for a category.
            after: After results for a category.
            category: The benchmark category.

        Returns:
            Dictionary with change metrics.
        """
        change = {"category": category}

        if category == "latency":
            before_val = before.get("average_latency_seconds", 0)
            after_val = after.get("average_latency_seconds", 0)
            if before_val > 0:
                change["latency_change_percent"] = ((after_val - before_val) / before_val) * 100
            change["before"] = before_val
            change["after"] = after_val

        elif category == "tokens_per_sec":
            before_tps = before.get("average_tokens_per_second", 0)
            after_tps = after.get("average_tokens_per_second", 0)
            if before_tps > 0:
                change["tps_change_percent"] = ((after_tps - before_tps) / before_tps) * 100
            change["before"] = before_tps
            change["after"] = after_tps

        else:
            # For qualitative categories, just mark if both have results
            change["compared"] = bool(before and after)

        return change

    def run_single(
        self,
        prompt: str,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        """Run a single prompt for quick testing.

        Args:
            prompt: Single prompt text.
            model_path: Optional path to model.

        Returns:
            Dictionary containing generation result.
        """
        if not self.is_backend_available():
            return {
                "status": "error",
                "error": f"{self.config.backend.value} is not running",
            }

        # Import model if path provided
        if model_path and hasattr(self.client, "import_model"):
            self.client.import_model(model_path)

        result = self.client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return {
            "status": "success",
            "prompt": prompt,
            "generation": result.get("text", ""),
            "elapsed": result.get("elapsed", 0),
            "tokens": result.get("usage", {}).get("total_tokens", 0) or result.get("eval_count", 0),
            "tokens_per_second": (
                result.get("usage", {}).get("total_tokens", 0) / result.get("elapsed", 1)
                if result.get("elapsed", 0) > 0
                else 0
            ),
        }

    def _save_results(self, results: dict[str, Any]) -> Path:
        """Save benchmark results to JSON file.

        Args:
            results: Benchmark results dictionary.

        Returns:
            Path to saved results file.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.config.backend.value}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to: {filepath}")
        return filepath

    def _save_comparison_results(self, results: dict[str, Any]) -> Path:
        """Save comparison benchmark results to JSON file.

        Args:
            results: Comparison results dictionary.

        Returns:
            Path to saved results file.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_comparison_{self.config.backend.value}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comparison results saved to: {filepath}")
        return filepath

    def load_prompts(self, prompt_file: str) -> list[str]:
        """Load prompts from a file.

        Args:
            prompt_file: Path to file containing prompts (one per line or JSON).

        Returns:
            List of prompt strings.
        """
        path = Path(prompt_file)

        if not path.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            return []

        content = path.read_text().strip()

        # Try JSON first (array of prompts)
        if content.startswith("["):
            try:
                prompts = json.loads(content)
                if isinstance(prompts, list):
                    return prompts
            except json.JSONDecodeError:
                pass

        # Fall back to one prompt per line
        prompts = [line.strip() for line in content.split("\n") if line.strip()]
        return prompts

    def get_supported_backends(self) -> list[BenchmarkBackend]:
        """Get list of supported benchmark backends.

        Returns:
            List of supported BenchmarkBackend values.
        """
        return list(BenchmarkBackend)


class TransformersClient:
    """Client for running benchmarks using HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str = "gpt2",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize the Transformers client.

        Args:
            model_name: Name or path of the model to use.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._pipeline = None

    def _get_pipeline(self):
        """Get or create the text generation pipeline.

        Returns:
            HuggingFace pipeline for text generation.
        """
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return self._pipeline

    def is_available(self) -> bool:
        """Check if Transformers is available.

        Returns:
            True if Transformers can be loaded.
        """
        try:
            self._get_pipeline()
            return True
        except Exception:
            return False

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> dict[str, Any]:
        """Generate text using Transformers.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dictionary containing generated text and metrics.
        """
        import time

        start_time = time.time()

        try:
            pipeline = self._get_pipeline()
            output = pipeline(prompt, max_new_tokens=max_tokens, temperature=temperature)
            elapsed = time.time() - start_time

            text = output[0]["generated_text"] if output else ""

            # Estimate token count (rough approximation)
            tokens = len(text.split())

            return {
                "text": text,
                "elapsed": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "text": "",
                "error": str(e),
                "elapsed": elapsed,
                "tokens": 0,
            }

    def benchmark(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """Run benchmark using Transformers.

        Args:
            prompts: List of prompt texts.
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
            num_runs: Number of times to run each prompt.

        Returns:
            Dictionary containing benchmark results.
        """
        results = []
        total_tokens = 0
        total_time = 0

        for i, prompt in enumerate(prompts):
            for run in range(num_runs):
                result = self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if "error" not in result:
                    tokens = result.get("tokens", 0)
                    elapsed = result.get("elapsed", 0)
                    total_tokens += tokens
                    total_time += elapsed
                    results.append(
                        {
                            "prompt_index": i,
                            "run": run,
                            "tokens": tokens,
                            "elapsed": elapsed,
                            "tokens_per_second": result.get("tokens_per_second", 0),
                        }
                    )

        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return {
            "total_runs": len(results),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "individual_results": results,
        }
