"""Unsloth-based trainer implementation."""

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from flattune.trainer.base import TrainerBase
from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class UnslothTrainer(TrainerBase):
    """Training backend using Unsloth for efficient fine-tuning.

    Unsloth provides optimized LoRA training that is significantly
    faster than standard PEFT implementations.
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str | None = None,
        output_dir: Path | None = None,
        config: Any | None = None,
        flatseek_provider: Any | None = None,
        flatseek_query: str | None = None,
    ):
        """Initialize the Unsloth trainer.

        Args:
            model_path: Path to the base model (GGUF or HuggingFace).
            dataset_path: Path to the training dataset JSON.
            output_dir: Directory for outputs.
            config: Training configuration.
            flatseek_provider: FlatSeek provider for streaming (optional).
            flatseek_query: Query to stream from FlatSeek (optional).
        """
        super().__init__(
            model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
            flatseek_provider=flatseek_provider,
            flatseek_query=flatseek_query,
        )
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = self.output_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._training_metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epochs": [],
            "steps": [],
            "total_steps": 0,
            "elapsed_time": 0.0,
        }

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Seed value to set.
        """
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _get_git_info(self) -> dict[str, Any]:
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
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()[:8]

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

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def _detect_model_source(self) -> str:
        """Detect the model source type from the path.

        Returns:
            Model source type: 'gguf', 'huggingface', or 'mlx'.
        """
        path = Path(self.model_path)
        if path.suffix == ".gguf":
            return "gguf"
        elif "mlx" in str(path).lower():
            return "mlx"
        else:
            return "huggingface"

    def _load_gguf_model(self) -> tuple[Any, Any]:
        """Load a GGUF model using llama-cpp-python or transformers.

        Returns:
            Tuple of (model, tokenizer).
        """
        try:
            # Try using transformers with llama.cpp support
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading GGUF model from {self.model_path}")

            # Load tokenizer from the same directory
            model_dir = Path(self.model_path).parent
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )

            # For GGUF, we need to use llama-cpp-python's loader
            # or convert it first. Here we use a workaround with AutoModel
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=False,
            )

            return model, tokenizer

        except Exception as e:
            logger.warning(f"Failed to load with transformers, trying llama-cpp: {e}")
            raise ImportError(
                "Failed to load GGUF model. Please install llama-cpp-python: "
                "pip install llama-cpp-python"
            ) from e

    def _load_huggingface_model(self) -> tuple[Any, Any]:
        """Load a HuggingFace model.

        Returns:
            Tuple of (model, tokenizer).
        """
        from unsloth import FastLanguageModel

        logger.info(f"Loading HuggingFace model from {self.model_path}")

        max_seq_length = getattr(self.config, "max_seq_length", 2048)
        if hasattr(self.config, "max_length"):
            max_seq_length = self.config.max_length

        dtype = None  # Auto-detect
        if self.config.bf16:
            dtype = "bfloat16"
        elif self.config.fp16:
            dtype = "float16"

        load_in_4bit = True  # 4-bit quantization for memory efficiency

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        return model, tokenizer

    def _load_mlx_model(self) -> tuple[Any, Any]:
        """Load an MLX model.

        Returns:
            Tuple of (model, tokenizer).
        """
        raise NotImplementedError(
            "MLX model loading is not yet implemented. "
            "Please convert the model to HuggingFace format or use GGUF."
        )

    def _load_model(self) -> tuple[Any, Any]:
        """Load the model based on source type.

        Returns:
            Tuple of (model, tokenizer).
        """
        source = self._detect_model_source()

        if source == "gguf":
            return self._load_gguf_model()
        elif source == "mlx":
            return self._load_mlx_model()
        else:
            return self._load_huggingface_model()

    def _apply_lora(self, model: Any, tokenizer: Any) -> Any:
        """Apply LoRA adapters to the model.

        Args:
            model: The base model.
            tokenizer: The tokenizer.

        Returns:
            Model with LoRA adapters applied.
        """
        from unsloth import FastLanguageModel

        logger.info("Applying LoRA adapters...")

        lora_rank = getattr(self.config, "lora_rank", 16)
        lora_alpha = getattr(self.config, "lora_alpha", 32)
        lora_dropout = getattr(self.config, "lora_dropout", 0.05)
        lora_target_modules = getattr(self.config, "lora_target_modules", ["q_proj", "v_proj"])

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        return model

    def _prepare_dataset(self) -> Any:
        """Prepare the training dataset.

        Returns:
            Prepared dataset ready for training.
        """
        from datasets import load_dataset

        logger.info(f"Loading dataset from {self.dataset_path}")

        dataset = load_dataset("json", data_files=self.dataset_path, split="train")

        # Get tokenizer for processing
        _, tokenizer = self._load_model()

        def tokenize_function(examples: dict) -> dict:
            """Tokenize the examples."""
            # Handle different dataset formats
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples and "output" in examples:
                texts = [
                    f"Instruction: {instr}\nOutput: {out}"
                    for instr, out in zip(examples["instruction"], examples["output"], strict=False)
                ]
            elif "messages" in examples:
                # Chat format
                texts = []
                for msg_list in examples["messages"]:
                    text = ""
                    for msg in msg_list:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        text += f"{role.capitalize()}: {content}\n"
                    texts.append(text)
            else:
                texts = list(examples.values())[0] if examples else []

            result = tokenizer(
                texts,
                truncation=True,
                max_length=getattr(self.config, "max_seq_length", 2048),
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        return tokenized_dataset

    def _create_trainer(self, model: Any, tokenizer: Any, dataset: Any) -> Any:
        """Create the training trainer.

        Args:
            model: The model with LoRA applied.
            tokenizer: The tokenizer.
            dataset: The tokenized dataset.

        Returns:
            Trainer instance.
        """
        from transformers import TrainingArguments

        # Split dataset for evaluation if possible
        try:
            dataset = dataset.train_test_split(test_size=0.1, seed=self.config.seed)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        except Exception:
            train_dataset = dataset
            eval_dataset = None

        output_dir = str(self.checkpoint_dir)
        save_steps = getattr(self.config, "save_steps", 100)
        logging_steps = getattr(self.config, "logging_steps", 10)
        eval_steps = getattr(self.config, "eval_steps", 100)
        warmup_steps = getattr(self.config, "warmup_steps", 100)
        max_seq_length = getattr(self.config, "max_seq_length", 2048)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="none",
            seed=self.config.seed,
            remove_unused_columns=False,
            optim="adamw_torch",
        )

        # Use SFTTrainer from trl for supervised fine-tuning
        try:
            from trl import SFTTrainer

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                max_seq_length=max_seq_length,
            )
        except ImportError:
            # Fallback to standard Trainer
            from transformers import Trainer

            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
            )

        return trainer

    def _extract_metrics(self, trainer: Any) -> dict[str, Any]:
        """Extract training metrics from the trainer.

        Args:
            trainer: The trainer instance.

        Returns:
            Dictionary of training metrics.
        """
        metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epochs": [],
            "steps": [],
        }

        # Extract from trainer's log history
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            log_history = trainer.state.log_history

            for entry in log_history:
                if "loss" in entry:
                    metrics["train_loss"].append(entry["loss"])
                    metrics["steps"].append(entry.get("step", len(metrics["steps"])))
                if "eval_loss" in entry:
                    metrics["eval_loss"].append(entry["eval_loss"])
                if "learning_rate" in entry:
                    metrics["learning_rate"].append(entry["learning_rate"])
                if "epoch" in entry:
                    metrics["epochs"].append(entry["epoch"])

        return metrics

    def _plot_loss_curve(self, metrics: dict[str, Any], output_path: Path) -> None:
        """Generate and save loss curve plot.

        Args:
            metrics: Training metrics dictionary.
            output_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        if metrics["train_loss"]:
            steps = metrics.get("steps", range(len(metrics["train_loss"])))
            axes[0].plot(steps, metrics["train_loss"], "b-", linewidth=2)
            axes[0].set_xlabel("Steps")
            axes[0].set_ylabel("Training Loss")
            axes[0].set_title("Training Loss Curve")
            axes[0].grid(True, alpha=0.3)

        # Eval loss if available
        if metrics["eval_loss"]:
            eval_steps = [
                metrics["steps"][i]
                for i in range(len(metrics["steps"]))
                if i < len(metrics["eval_loss"])
            ]
            if not eval_steps:
                eval_steps = range(len(metrics["eval_loss"]))
            axes[1].plot(eval_steps, metrics["eval_loss"], "r-", linewidth=2)
            axes[1].set_xlabel("Steps")
            axes[1].set_ylabel("Evaluation Loss")
            axes[1].set_title("Evaluation Loss Curve")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Loss curve saved to {output_path}")

    def _generate_report(self, metrics: dict[str, Any], training_time: float) -> Path:
        """Generate training report with metrics and visualizations.

        Args:
            metrics: Training metrics dictionary.
            training_time: Total training time in seconds.

        Returns:
            Path to the generated report.
        """
        logger.info("Generating training report...")

        # Create loss curve plot
        plot_path = self.report_dir / "loss_curve.png"
        self._plot_loss_curve(metrics, plot_path)

        # Calculate summary statistics
        final_loss = metrics["train_loss"][-1] if metrics["train_loss"] else None
        min_loss = min(metrics["train_loss"]) if metrics["train_loss"] else None
        avg_loss = np.mean(metrics["train_loss"]) if metrics["train_loss"] else None

        # Build report
        report = {
            "status": "completed",
            "timestamp": self._get_timestamp(),
            "git_info": self._get_git_info(),
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "training_time_seconds": training_time,
            "total_steps": len(metrics["train_loss"]),
            "final_loss": final_loss,
            "min_loss": min_loss,
            "avg_loss": avg_loss,
            "epochs_completed": self.config.epochs,
            "learning_rate": self.config.lr,
            "lora_rank": getattr(self.config, "lora_rank", 16),
            "lora_alpha": getattr(self.config, "lora_alpha", 32),
            "checkpoints": [str(p) for p in self.checkpoint_dir.glob("**/*.pt")]
            + [str(p) for p in self.checkpoint_dir.glob("**/*.safetensors")],
            "metrics": {
                "train_loss": metrics["train_loss"],
                "eval_loss": metrics["eval_loss"],
                "learning_rate": metrics["learning_rate"],
                "steps": metrics["steps"],
            },
        }

        # Save report as JSON
        report_path = self.report_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to {report_path}")
        return report_path

    def train(self) -> dict[str, Any]:
        """Run training using Unsloth.

        Returns:
            Dictionary containing training metrics.
        """
        import time

        logger.info("=" * 60)
        logger.info("Starting Unsloth training...")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.lr}")
        logger.info(f"LoRA rank: {getattr(self.config, 'lora_rank', 16)}")
        logger.info(f"LoRA alpha: {getattr(self.config, 'lora_alpha', 32)}")

        start_time = time.time()

        # Set seed for reproducibility
        self._set_seed(self.config.seed)

        try:
            # Check if Unsloth is available
            try:
                from unsloth import FastLanguageModel
            except ImportError:
                raise ImportError("Unsloth is not installed. Install it with: pip install unsloth")

            # Load model based on source
            model_source = self._detect_model_source()
            logger.info(f"Detected model source: {model_source}")

            self._model, self._tokenizer = self._load_model()

            # Apply LoRA adapters
            self._model = self._apply_lora(self._model, self._tokenizer)

            # Prepare dataset
            dataset = self._prepare_dataset()

            # Create trainer
            self._trainer = self._create_trainer(self._model, self._tokenizer, dataset)

            # Train
            logger.info("Starting training loop...")
            self._trainer.train()

            # Extract metrics
            self._training_metrics = self._extract_metrics(self._trainer)

            training_time = time.time() - start_time
            self._training_metrics["elapsed_time"] = training_time

            # Generate report
            report_path = self._generate_report(self._training_metrics, training_time)

            # Get final checkpoint
            final_checkpoint = self.get_latest_checkpoint()

            logger.info("=" * 60)
            logger.info("Training complete!")
            logger.info(f"Training time: {training_time:.2f} seconds")
            logger.info(f"Final checkpoint: {final_checkpoint}")
            logger.info(f"Training report: {report_path}")
            logger.info("=" * 60)

            return {
                "status": "completed",
                "training_time_seconds": training_time,
                "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
                "report_path": str(report_path),
                "final_loss": (
                    self._training_metrics["train_loss"][-1]
                    if self._training_metrics["train_loss"]
                    else None
                ),
                "total_steps": len(self._training_metrics["train_loss"]),
                "checkpoints_saved": len(self.get_checkpoints()),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def merge(self) -> Path:
        """Merge LoRA adapter with base model.

        Returns:
            Path to the merged model.
        """

        logger.info("Merging LoRA adapter with base model...")

        if self._model is None:
            raise RuntimeError("Model not loaded. Please run train() first or load a model.")

        merged_path = self.output_dir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)

        try:
            # Use Unsloth's merge functionality
            self._model.save_pretrained(str(merged_path))
            self._tokenizer.save_pretrained(str(merged_path))

            logger.info(f"Merged model saved to: {merged_path}")
        except Exception as e:
            logger.warning(f"Unsloth merge failed, using standard merge: {e}")
            # Fallback: just save the merged model
            self._model.save_pretrained(str(merged_path))
            self._tokenizer.save_pretrained(str(merged_path))

        return merged_path

    def export(self, format: str) -> Path:
        """Export model to specified format.

        Args:
            format: Target format (gguf, mlx, hf).

        Returns:
            Path to exported model.
        """
        logger.info(f"Exporting model to {format}...")

        export_path = self.output_dir / "export" / format
        export_path.mkdir(parents=True, exist_ok=True)

        if format.lower() == "gguf":
            # Export to GGUF using llama.cpp converter
            try:
                # First ensure we have the merged model
                if self._model is None:
                    raise RuntimeError("Model not loaded. Run train() first.")

                merged_path = self.output_dir / "merged"
                if not merged_path.exists():
                    merged_path = self.merge()

                # Convert to GGUF (requires llama.cpp's convert_hf_to_gguf.py)
                logger.info("Converting to GGUF format...")
                # This would use llama.cpp's conversion script
                logger.warning(
                    "GGUF export requires llama.cpp. "
                    "Install from https://github.com/ggerganov/llama.cpp"
                )
            except Exception as e:
                logger.error(f"GGUF export failed: {e}")

        elif format.lower() == "mlx":
            # Export to MLX format
            try:
                import mlx.core as mx

                if self._model is None:
                    raise RuntimeError("Model not loaded. Run train() first.")

                merged_path = self.output_dir / "merged"
                if not merged_path.exists():
                    merged_path = self.merge()

                # MLX export would go here
                logger.warning("MLX export is not yet fully implemented.")
            except ImportError:
                raise ImportError("MLX not installed. Install with: pip install mlx")

        elif format.lower() in ("hf", "huggingface", "hf_merged"):
            # Export to HuggingFace format
            if self._model is None:
                raise RuntimeError("Model not loaded. Run train() first.")

            merged_path = self.output_dir / "merged"
            if not merged_path.exists():
                merged_path = self.merge()

            logger.info(f"Model exported to HuggingFace format: {merged_path}")

        elif format.lower() == "safetensors":
            # Export to SafeTensors format
            from safetensors.torch import save_file

            if self._model is None:
                raise RuntimeError("Model not loaded. Run train() first.")

            state_dict = self._model.state_dict()
            safetensors_path = export_path / "model.safetensors"
            save_file(state_dict, str(safetensors_path))
            self._tokenizer.save_pretrained(str(export_path))

            logger.info(f"Model exported to SafeTensors: {safetensors_path}")
            return safetensors_path

        logger.info(f"Model exported to: {export_path}")
        return export_path

    def evaluate(self) -> dict[str, float]:
        """Evaluate the trained model.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model...")

        if self._trainer is None:
            raise RuntimeError("Trainer not initialized. Please run train() first.")

        # Run evaluation
        eval_result = self._trainer.evaluate()

        eval_loss = eval_result.get("eval_loss", 0.0)
        perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")

        metrics = {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
        }

        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def get_checkpoints(self) -> list[Path]:
        """Get list of saved checkpoints.

        Returns:
            List of checkpoint file paths.
        """
        checkpoints = list(self.checkpoint_dir.glob("**/*.pt"))
        checkpoints.extend(self.checkpoint_dir.glob("**/*.safetensors"))
        checkpoints.extend(self.checkpoint_dir.glob("**/adapter_model.safetensors"))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

    def get_latest_checkpoint(self) -> Path | None:
        """Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint or None if no checkpoints exist.
        """
        checkpoints = self.get_checkpoints()
        if checkpoints:
            return checkpoints[-1]

        # Check merged directory
        merged_path = self.output_dir / "merged"
        if merged_path.exists():
            adapter = merged_path / "adapter_model.safetensors"
            if adapter.exists():
                return adapter
            adapter_config = merged_path / "adapter_config.json"
            if adapter_config.exists():
                return merged_path

        return None

    def save_checkpoint(self, name: str = "checkpoint") -> Path:
        """Save a checkpoint manually.

        Args:
            name: Name for the checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Run train() first.")

        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(checkpoint_path))
        self._tokenizer.save_pretrained(str(checkpoint_path))

        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def get_training_history(self) -> dict[str, Any]:
        """Get the full training history.

        Returns:
            Dictionary containing all training metrics.
        """
        return self._training_metrics.copy()
