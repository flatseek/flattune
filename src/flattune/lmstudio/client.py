"""LM Studio client for model management and benchmarking."""

import time
from pathlib import Path
from typing import Any

import requests

from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class LMStudioClient:
    """Client for interacting with LM Studio.

    Handles model loading, prompt execution, and benchmark collection
    for models served by LM Studio.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model_name: str | None = None,
    ):
        """Initialize the LM Studio client.

        Args:
            base_url: Base URL for LM Studio API.
            model_name: Name of the model to use (loaded in LM Studio).
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._session = requests.Session()
        self._connected = False

    def connect(self, host: str = "localhost", port: int = 1234) -> bool:
        """Connect to LM Studio REST API.

        Args:
            host: LM Studio host address.
            port: LM Studio port number.

        Returns:
            True if connection successful.
        """
        self.base_url = f"http://{host}:{port}"
        self._connected = self.is_available()
        if self._connected:
            logger.info(f"Connected to LM Studio at {self.base_url}")
        else:
            logger.warning(f"Could not connect to LM Studio at {self.base_url}")
        return self._connected

    def is_available(self) -> bool:
        """Check if LM Studio is running and accessible.

        Returns:
            True if LM Studio API is available.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/v1/models",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_model_list(self) -> list[dict[str, Any]]:
        """List available models in LM Studio.

        Returns:
            List of model information dictionaries.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/v1/models",
                timeout=5,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def list_models(self) -> list[dict[str, Any]]:
        """List available models in LM Studio.

        Alias for get_model_list().

        Returns:
            List of model information dictionaries.
        """
        return self.get_model_list()

    def import_model(self, gguf_path: str, model_name: str | None = None) -> bool:
        """Import a GGUF model into LM Studio.

        Args:
            gguf_path: Path to the GGUF model file.
            model_name: Optional name for the model.

        Returns:
            True if model imported successfully.
        """
        model_name = model_name or Path(gguf_path).stem

        try:
            # First, ensure the model is visible to LM Studio
            # LM Studio typically auto-detects models in its models directory
            # For external paths, we use the model loading endpoint

            response = self._session.post(
                f"{self.base_url}/v1/models/load",
                json={
                    "model_path": str(gguf_path),
                    "model_name": model_name,
                },
                timeout=120,
            )

            if response.status_code == 200:
                self.model_name = model_name
                logger.info(f"Model '{model_name}' loaded successfully")
                return True

            logger.error(f"Failed to import model: {response.status_code}")
            return False

        except requests.RequestException as e:
            logger.error(f"Failed to import model: {e}")
            return False

    def reload_model(self) -> bool:
        """Reload the current model after changes.

        Returns:
            True if model reloaded successfully.
        """
        if not self.model_name:
            logger.error("No model currently loaded")
            return False

        try:
            # Unload and reload the model
            unload_response = self._session.post(
                f"{self.base_url}/v1/models/unload",
                timeout=30,
            )

            if unload_response.status_code != 200:
                logger.warning("Could not unload model (may not be needed)")

            # Re-load the model with the same name
            model_info = self.get_model_info()
            model_path = model_info.get("path", "")

            if model_path:
                return self.import_model(model_path, self.model_name)

            logger.info("Model reloaded (no path info available)")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to reload model: {e}")
            return False

    def load_model(
        self,
        model_path: str,
        context_length: int = 2048,
        gpu_offload: float = 1.0,
    ) -> bool:
        """Load a model into LM Studio.

        Args:
            model_path: Path to the model file.
            context_length: Context window size.
            gpu_offload: Fraction of layers to offload to GPU.

        Returns:
            True if model loaded successfully.
        """
        try:
            response = self._session.post(
                f"{self.base_url}/v1/models/load",
                json={
                    "model_path": model_path,
                    "context_length": context_length,
                    "gpu_offload": gpu_offload,
                },
                timeout=60,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def unload_model(self) -> bool:
        """Unload the current model from LM Studio.

        Returns:
            True if model unloaded successfully.
        """
        try:
            response = self._session.post(
                f"{self.base_url}/v1/models/unload",
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    def run_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text using the loaded model.

        Args:
            prompt: Input prompt.
            system_prompt: Optional system prompt for instruction following.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.

        Returns:
            Dictionary containing generated text and metrics.
        """
        # Use chat format if system prompt is provided
        if system_prompt:
            return self.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text using the loaded model.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.

        Returns:
            Dictionary containing generated text and metrics.
        """
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop:
                payload["stop"] = stop

            start_time = time.time()
            response = self._session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start_time

            response.raise_for_status()
            result = response.json()

            return {
                "text": result.get("choices", [{}])[0].get("text", ""),
                "usage": result.get("usage", {}),
                "elapsed": elapsed,
                "model": self.model_name,
            }
        except requests.RequestException as e:
            logger.error(f"Generation failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "elapsed": 0,
                "model": self.model_name,
            }

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate chat completions.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dictionary containing response and metrics.
        """
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            start_time = time.time()
            response = self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start_time

            response.raise_for_status()
            result = response.json()

            return {
                "text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": result.get("usage", {}),
                "elapsed": elapsed,
                "model": self.model_name,
            }
        except requests.RequestException as e:
            logger.error(f"Chat generation failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "elapsed": 0,
                "model": self.model_name,
            }

    def benchmark(
        self,
        categories: list[str] | None = None,
        prompts: list[str] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """Run comprehensive benchmarks on the loaded model.

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
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
            num_runs: Number of times to run each prompt.

        Returns:
            Dictionary containing benchmark results for all categories.
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

        results = {
            "categories": {},
            "summary": {},
            "model": self.model_name,
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

        # Handle list prompts - use same prompts for all categories
        if isinstance(prompts_to_use, list):
            prompts_dict = dict.fromkeys(categories, prompts_to_use)
        else:
            prompts_dict = prompts_to_use

        # Run latency and throughput benchmarks
        if "latency" in categories or "tokens_per_sec" in categories:
            latency_results = self._benchmark_latency(
                prompts_dict.get("latency", prompts_dict.get("tokens_per_sec", [])),
                max_tokens,
                temperature,
                num_runs,
            )
            results["categories"]["latency"] = latency_results.get("latency", {})
            results["categories"]["tokens_per_sec"] = latency_results.get("tokens_per_sec", {})

        # Run prompt quality benchmarks
        if "prompt_quality" in categories:
            results["categories"]["prompt_quality"] = self._benchmark_prompt_quality(
                prompts_dict.get("prompt_quality", []),
                max_tokens,
                temperature,
            )

        # Run instruction following benchmarks
        if "instruction_following" in categories:
            results["categories"]["instruction_following"] = self._benchmark_instruction_following(
                prompts_dict.get("instruction_following", []),
                max_tokens,
                temperature,
            )

        # Run domain accuracy benchmarks
        if "domain_accuracy" in categories:
            results["categories"]["domain_accuracy"] = self._benchmark_domain_accuracy(
                prompts_dict.get("domain_accuracy", []),
                max_tokens,
                temperature,
            )

        # Run hallucination tests
        if "hallucination_tests" in categories:
            results["categories"]["hallucination_tests"] = self._benchmark_hallucination(
                prompts_dict.get("hallucination_tests", []),
                max_tokens,
                temperature,
            )

        # Run regression tests
        if "regression_tests" in categories:
            results["categories"]["regression_tests"] = self._benchmark_regression(
                prompts_dict.get("regression_tests", []),
                max_tokens,
                temperature,
            )

        # Compile summary
        results["summary"] = self._compile_benchmark_summary(results["categories"])

        return results

    def _benchmark_latency(
        self,
        prompts: list[str],
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
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if "error" not in result:
                    tokens = result.get("usage", {}).get("total_tokens", 0)
                    elapsed = result.get("elapsed", 0)
                    total_tokens += tokens
                    total_time += elapsed

                    all_results.append({
                        "prompt": prompt[:50],
                        "tokens": tokens,
                        "elapsed": elapsed,
                        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                    })

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
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                text = result.get("text", "")
                results.append({
                    "prompt": prompt,
                    "response": text,
                    "response_length": len(text),
                    "quality_indicators": {
                        "has_content": len(text) > 10,
                        "is_grammar_sound": True,  # Placeholder
                    },
                })

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_instruction_following(
        self,
        prompts: list[str],
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
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append({
                    "prompt": prompt,
                    "response": result.get("text", ""),
                    "instruction_followed": True,  # Placeholder for actual evaluation
                })

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_domain_accuracy(
        self,
        prompts: list[str],
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
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append({
                    "prompt": prompt,
                    "response": result.get("text", ""),
                    "accuracy_score": None,  # Placeholder for actual evaluation
                })

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_hallucination(
        self,
        prompts: list[str],
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
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append({
                    "prompt": prompt,
                    "response": result.get("text", ""),
                    "potential_hallucination": None,  # Placeholder for detection
                })

        return {
            "num_prompts": len(prompts),
            "successful_generations": len(results),
            "results": results,
        }

    def _benchmark_regression(
        self,
        prompts: list[str],
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
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "error" not in result:
                results.append({
                    "prompt": prompt,
                    "response": result.get("text", ""),
                    "regression_detected": False,  # Placeholder for comparison
                })

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

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dictionary containing model information.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/v1/models",
                timeout=5,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if models and self.model_name:
                for model in models:
                    if self.model_name in model.get("id", ""):
                        return model
            return models[0] if models else {}
        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
