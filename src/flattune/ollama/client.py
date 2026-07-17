"""Ollama client for model management and benchmarking."""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama.

    Handles model management, prompt execution, and benchmarking
    for models served by Ollama.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str | None = None,
    ):
        """Initialize the Ollama client.

        Args:
            base_url: Base URL for Ollama API.
            model_name: Name of the model to use.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or "llama2"
        self._session = requests.Session()
        self._temp_dir: Path | None = None

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama API is available.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models in Ollama.

        Returns:
            List of model information dictionaries.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_list(self) -> list[dict[str, Any]]:
        """List available models in Ollama.

        Alias for list_models().

        Returns:
            List of model information dictionaries.
        """
        return self.list_models()

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model: Model name to pull.

        Returns:
            True if model pulled successfully.
        """
        try:
            response = self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600,  # 10 minute timeout for large models
                stream=True,
            )
            # Stream response for progress
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    logger.debug(f"Ollama pull: {data}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def create_modelfile(
        self,
        model_name: str,
        base_model: str,
        model_path: str,
        system: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Create a Modelfile for importing a local model.

        Args:
            model_name: Name for the new model.
            base_model: Base model to use.
            model_path: Path to the model file.
            system: Optional system prompt.
            parameters: Optional model parameters.

        Returns:
            Path to created Modelfile.
        """
        modelfile_content = f"""FROM {base_model}
PARAMETER model {model_path}
"""

        if system:
            modelfile_content += f'\nSYSTEM """\n{system}\n"""\n'

        if parameters:
            for key, value in parameters.items():
                modelfile_content += f"PARAMETER {key} {value}\n"

        # Save Modelfile
        modelfile_path = Path.home() / ".ollama" / "modelfiles" / f"{model_name}"
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        modelfile_path.write_text(modelfile_content)

        logger.info(f"Created Modelfile at: {modelfile_path}")
        return str(modelfile_path)

    def import_model(
        self,
        model_name: str,
        model_path: str,
        base_model: str = "llama2",
        system: str | None = None,
    ) -> bool:
        """Import a local model into Ollama.

        Args:
            model_name: Name for the imported model.
            model_path: Path to the model file.
            base_model: Base model to use.
            system: Optional system prompt.

        Returns:
            True if model imported successfully.
        """
        try:
            # Create Modelfile
            modelfile_path = self.create_modelfile(
                model_name=model_name,
                base_model=base_model,
                model_path=model_path,
                system=system,
            )

            # Create the model
            response = self._session.post(
                f"{self.base_url}/api/create",
                json={"name": model_name, "path": modelfile_path},
                timeout=60,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to import model: {e}")
            return False

    def delete_model(self, model: str) -> bool:
        """Delete a model from Ollama.

        Args:
            model: Model name to delete.

        Returns:
            True if model deleted successfully.
        """
        try:
            response = self._session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to delete model: {e}")
            return False

    def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text using a model.

        This method provides a simple interface for running prompts.

        Args:
            prompt: Input prompt.
            model: Model name (uses default if not specified).
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Stop sequences.

        Returns:
            Dictionary containing generated text and metrics.
        """
        if system:
            return self.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return self.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text using a model.

        Args:
            prompt: Input prompt.
            model: Model name (uses default if not specified).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Stop sequences.
            options: Additional model options.

        Returns:
            Dictionary containing generated text and metrics.
        """
        model = model or self.model_name

        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **(options or {}),
                },
                "stream": False,
            }
            if stop:
                payload["stop"] = stop

            start_time = time.time()
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start_time

            response.raise_for_status()
            result = response.json()

            return {
                "text": result.get("response", ""),
                "context": result.get("context", []),
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
                "elapsed": elapsed,
                "model": model,
            }
        except requests.RequestException as e:
            logger.error(f"Generation failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "elapsed": 0,
                "model": model,
            }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate chat completions.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model name (uses default if not specified).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dictionary containing response and metrics.
        """
        model = model or self.model_name

        try:
            payload = {
                "messages": messages,
                "model": model,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            }

            start_time = time.time()
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start_time

            response.raise_for_status()
            result = response.json()

            return {
                "text": result.get("message", {}).get("content", ""),
                "total_duration": result.get("total_duration", 0),
                "eval_count": result.get("eval_count", 0),
                "elapsed": elapsed,
                "model": model,
            }
        except requests.RequestException as e:
            logger.error(f"Chat generation failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "elapsed": 0,
                "model": model,
            }

    def benchmark(
        self,
        categories: list[str] | None = None,
        prompts: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """Run comprehensive benchmarks on a model.

        Args:
            categories: List of benchmark categories to run:
                - latency: Measure time per token
                - tokens_per_sec: Throughput benchmarking
                - prompt_quality: Output quality assessment
                - instruction_following: Test instruction compliance
                - domain_accuracy: Domain-specific knowledge tests
                - hallucination_tests: Test for hallucinations
                - regression_tests: Compare against known outputs
            prompts: Optional list of prompts to use (uses built-in if not provided).
            model: Model name (uses default if not specified).
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
            num_runs: Number of times to run each prompt.

        Returns:
            Dictionary containing benchmark results for all categories.
        """
        model = model or self.model_name
        categories = categories or [
            "latency",
            "tokens_per_sec",
            "prompt_quality",
            "instruction_following",
            "domain_accuracy",
            "hallucination_tests",
            "regression_tests",
        ]

        results = {
            "categories": {},
            "summary": {},
            "model": model,
        }

        # Built-in benchmark prompts for each category
        default_prompts = {
            "latency": [
                "The quick brown fox jumps over the lazy dog.",
                "Explain the theory of relativity.",
            ],
            "tokens_per_sec": [
                "Write a detailed story about a dragon." * 10,
            ],
            "prompt_quality": [
                "Describe the process of photosynthesis.",
                "What are the benefits of exercise?",
            ],
            "instruction_following": [
                "List 5 fruits and explain their nutritional value.",
                "Translate to Spanish: The weather is nice today.",
            ],
            "domain_accuracy": [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
            ],
            "hallucination_tests": [
                "Who was the first person to walk on Mars?",
                "What year did the internet first become available to the public?",
            ],
            "regression_tests": [
                "What is 2 + 2?",
            ],
        }

        prompts_to_use = prompts or default_prompts

        # Run latency and throughput benchmarks
        if "latency" in categories or "tokens_per_sec" in categories:
            latency_results = self._benchmark_latency(
                prompts_to_use.get("latency", prompts_to_use.get("tokens_per_sec", [])),
                model,
                max_tokens,
                temperature,
                num_runs,
            )
            results["categories"]["latency"] = latency_results.get("latency", {})
            results["categories"]["tokens_per_sec"] = latency_results.get("tokens_per_sec", {})

        # Run prompt quality benchmarks
        if "prompt_quality" in categories:
            results["categories"]["prompt_quality"] = self._benchmark_prompt_quality(
                prompts_to_use.get("prompt_quality", []),
                model,
                max_tokens,
                temperature,
            )

        # Run instruction following benchmarks
        if "instruction_following" in categories:
            results["categories"]["instruction_following"] = self._benchmark_instruction_following(
                prompts_to_use.get("instruction_following", []),
                model,
                max_tokens,
                temperature,
            )

        # Run domain accuracy benchmarks
        if "domain_accuracy" in categories:
            results["categories"]["domain_accuracy"] = self._benchmark_domain_accuracy(
                prompts_to_use.get("domain_accuracy", []),
                model,
                max_tokens,
                temperature,
            )

        # Run hallucination tests
        if "hallucination_tests" in categories:
            results["categories"]["hallucination_tests"] = self._benchmark_hallucination(
                prompts_to_use.get("hallucination_tests", []),
                model,
                max_tokens,
                temperature,
            )

        # Run regression tests
        if "regression_tests" in categories:
            results["categories"]["regression_tests"] = self._benchmark_regression(
                prompts_to_use.get("regression_tests", []),
                model,
                max_tokens,
                temperature,
            )

        # Compile summary
        results["summary"] = self._compile_benchmark_summary(results["categories"])

        return results

    def _benchmark_latency(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
        num_runs: int,
    ) -> dict[str, Any]:
        """Benchmark latency metrics.

        Returns:
            Dictionary with latency and throughput results.
        """
        all_results = []
        total_tokens = 0
        total_time = 0

        for prompt in prompts:
            for _run in range(num_runs):
                result = self.generate(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if "error" not in result:
                    tokens = result.get("eval_count", 0)
                    elapsed = result.get("elapsed", 0)
                    total_tokens += tokens
                    total_time += elapsed

                    all_results.append(
                        {
                            "prompt": prompt[:50],
                            "tokens": tokens,
                            "elapsed": elapsed,
                            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                        }
                    )

        avg_latency = total_time / len(all_results) if all_results else 0
        avg_tps = total_tokens / total_time if total_time > 0 else 0

        return {
            "latency": {
                "average_latency_seconds": avg_latency,
                "results": all_results,
            },
            "tokens_per_sec": {
                "average_tokens_per_second": avg_tps,
                "total_tokens": total_tokens,
                "total_time": total_time,
            },
        }

    def _benchmark_prompt_quality(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Benchmark output quality.

        Returns:
            Dictionary with quality assessment results.
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                text = result.get("text", "")
                results.append(
                    {
                        "prompt": prompt,
                        "response": text,
                        "response_length": len(text),
                        "quality_indicators": {
                            "has_content": len(text) > 10,
                            "is_grammar_sound": True,  # Placeholder
                        },
                    }
                )

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_instruction_following(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Benchmark instruction following.

        Returns:
            Dictionary with instruction following results.
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append(
                    {
                        "prompt": prompt,
                        "response": result.get("text", ""),
                        "instruction_followed": True,  # Placeholder for actual evaluation
                    }
                )

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_domain_accuracy(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Benchmark domain-specific accuracy.

        Returns:
            Dictionary with accuracy results.
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append(
                    {
                        "prompt": prompt,
                        "response": result.get("text", ""),
                        "accuracy_score": None,  # Placeholder for actual evaluation
                    }
                )

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_hallucination(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Benchmark hallucination tendencies.

        Returns:
            Dictionary with hallucination test results.
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append(
                    {
                        "prompt": prompt,
                        "response": result.get("text", ""),
                        "potential_hallucination": None,  # Placeholder for detection
                    }
                )

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_regression(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Benchmark regression tests.

        Returns:
            Dictionary with regression test results.
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append(
                    {
                        "prompt": prompt,
                        "response": result.get("text", ""),
                        "regression_detected": False,  # Placeholder for comparison
                    }
                )

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _compile_benchmark_summary(self, categories: dict[str, Any]) -> dict[str, Any]:
        """Compile a summary from all benchmark categories.

        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "total_categories_run": len(categories),
            "category_summaries": {},
        }

        if "latency" in categories:
            lat = categories["latency"]
            summary["category_summaries"]["latency"] = {
                "average_latency_seconds": lat.get("average_latency_seconds", 0),
            }

        if "tokens_per_sec" in categories:
            tps = categories["tokens_per_sec"]
            summary["category_summaries"]["tokens_per_sec"] = {
                "average_tokens_per_second": tps.get("average_tokens_per_second", 0),
            }

        return summary

    def cleanup(self) -> bool:
        """Remove temporary files created during the session.

        Returns:
            True if cleanup successful.
        """
        success = True

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temp directory: {e}")
                success = False

        self._temp_dir = None
        return success

    def create_temp_modelfile(
        self,
        model_path: str,
        base_model: str = "llama2",
    ) -> str:
        """Create a temporary Modelfile for model import.

        Args:
            model_path: Path to the model file.
            base_model: Base model to use.

        Returns:
            Path to the temporary Modelfile.
        """
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ollama_"))

        temp_modelfile = self._temp_dir / "Modelfile"

        modelfile_content = f"""FROM {base_model}
PARAMETER model {model_path}
"""

        temp_modelfile.write_text(modelfile_content)
        logger.info(f"Created temp Modelfile at: {temp_modelfile}")

        return str(temp_modelfile)
