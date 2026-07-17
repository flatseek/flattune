"""Transformers + PEFT trainer as fallback when Unsloth is not available."""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from flattune.trainer.base import TrainerBase

logger = logging.getLogger(__name__)


def is_gguf_file(path: str) -> bool:
    """Check if a file is a GGUF model."""
    path = os.path.expanduser(path)
    return path.endswith('.gguf') or path.endswith('.gguf.bin')


class StreamingDataset(Dataset):
    """Dataset that streams from an iterator (e.g., FlatSeek).

    No intermediate file needed - streams directly from FlatSeek query.
    """

    def __init__(
        self,
        data_iterator: Iterator[dict],
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.examples = []

        logger.info("Building dataset from stream...")
        for i, doc in enumerate(data_iterator):
            if max_samples and i >= max_samples:
                break

            # Format: instruction, input, output
            instruction = doc.get('instruction', '')
            input_text = doc.get('input', '')
            output = doc.get('output', '')

            # Build prompt
            if instruction:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"{input_text}\n\n{output}"

            self.examples.append(text)

            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {i + 1} examples")

        logger.info(f"Dataset built with {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze(),
        }


@dataclass
class TransformersTrainerConfig:
    """Configuration for transformers trainer."""
    epochs: int = 3
    lr: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 2
    max_seq_length: int = 2048
    max_samples: Optional[int] = None
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = False
    fp16: bool = False


class TransformersTrainer(TrainerBase):
    """Training using transformers + PEFT (fallback when Unsloth unavailable).

    Works on CPU/MPS for small models without CUDA.
    Supports direct streaming from FlatSeek (no intermediate file).
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[TransformersTrainerConfig] = None,
        flatseek_provider: Optional[Any] = None,
        flatseek_query: Optional[str] = None,
    ):
        """Initialize transformers trainer."""
        super().__init__(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=Path(output_dir) if output_dir else None,
            config=config,
            flatseek_provider=flatseek_provider,
            flatseek_query=flatseek_query,
        )
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _detect_device(self) -> str:
        """Detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load model and tokenizer."""
        model_path = os.path.expanduser(self.model_path)
        device = self._detect_device()
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {device}")

        if is_gguf_file(model_path):
            # GGUF files need special handling
            logger.info("GGUF file detected - using llama.cpp loader")
            return self._load_gguf_model(model_path, device)

        # Load HuggingFace model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            torch_dtype = torch.float32
            if device == "cuda":
                torch_dtype = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch_dtype,
                device_map='auto' if device == "cuda" else None,
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_gguf_model(self, model_path: str, device: str):
        """Load GGUF model using llama.cpp."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning("llama.cpp not available for GGUF loading")
            self._llama = None
            self.model = None
            self.tokenizer = None
            return

        self._llama = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            verbose=False,
        )
        logger.info("GGUF model loaded with llama.cpp")

        # For GGUF, we can't directly fine-tune - create placeholder
        self.model = None
        self.tokenizer = None

    def _apply_lora(self):
        """Apply LoRA adapters using PEFT."""
        if self.model is None:
            logger.warning("GGUF model - skipping LoRA (requires conversion)")
            return

        logger.info(f"Applying LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")

        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self) -> dict:
        """Train the model using streaming or file."""
        logger.info("=" * 60)
        logger.info("Starting Transformers training (fallback mode)...")
        logger.info("=" * 60)

        # Set seed
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Load model
        self._load_model()

        # Check if GGUF - can't directly fine-tune
        if self.model is None:
            return self._gguf_training_placeholder()

        self._apply_lora()

        # Create dataset from stream or file
        if self.flatseek_provider:
            dataset = StreamingDataset(
                data_iterator=self.flatseek_provider.stream(self.flatseek_query),
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                max_samples=self.config.max_samples,
            )
        else:
            dataset = StreamingDataset(
                data_iterator=self._load_jsonl(self.dataset_path),
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                max_samples=self.config.max_samples,
            )

        if len(dataset) == 0:
            logger.warning("No training samples available")
            return {"error": "No training samples"}

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=0,
            disable_tqdm=False,
            load_best_model_at_end=False,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info(f"Training on {len(dataset)} samples...")
        result = self.trainer.train()

        # Save checkpoint
        checkpoint_path = self.output_dir / "final"
        checkpoint_path.mkdir(exist_ok=True)
        self.trainer.save_model(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))

        logger.info(f"Training complete. Checkpoint saved to {checkpoint_path}")

        return {
            "final_loss": result.training_loss if hasattr(result, 'training_loss') else 0.0,
            "checkpoint_path": str(checkpoint_path),
            "training_time": "unknown",
            "loss_history": [],
        }

    def _gguf_training_placeholder(self) -> dict:
        """Create placeholder for GGUF training.

        GGUF models require conversion to HF format for fine-tuning.
        This creates a placeholder checkpoint with training info.
        """
        logger.warning("=" * 60)
        logger.warning("GGUF model detected - full fine-tuning requires:")
        logger.warning("1. Convert GGUF to HuggingFace format using llama.cpp")
        logger.warning("2. Or use Unsloth which supports GGUF directly")
        logger.warning("Creating placeholder checkpoint...")
        logger.warning("=" * 60)

        checkpoint_path = self.output_dir / "final"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save training info
        training_info = {
            "model_path": self.model_path,
            "is_gguf": True,
            "training_samples": self.config.max_samples or "all",
            "config": {
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "lora_rank": self.config.lora_alpha,
            },
            "note": "GGUF model - fine-tuning requires conversion to HF format",
            "next_steps": [
                "1. Use 'flattune export' to convert to GGUF",
                "2. Load in LM Studio or Ollama for inference",
                "3. Or convert GGUF to HF format for full training"
            ]
        }

        with open(checkpoint_path / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)

        return {
            "final_loss": 0.0,
            "checkpoint_path": str(checkpoint_path),
            "training_time": "N/A (GGUF placeholder)",
            "loss_history": [],
            "is_gguf_placeholder": True,
        }

    def _load_jsonl(self, path: str) -> Iterator[dict]:
        """Load JSONL file as iterator."""
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line.strip())

    def merge(self) -> Path:
        """Merge LoRA with base model."""
        if self.model is None:
            # Load model and adapter for merge
            logger.info("Loading model for merge...")
            self._load_model()
            # Load the adapter from checkpoint
            adapter_path = self.output_dir / "final" / "adapter_model.safetensors"
            if adapter_path.exists():
                from peft import PeftModel
                logger.info(f"Loading adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(self.output_dir / "final")
                )

        logger.info("Merging LoRA with base model...")
        merged_path = self.output_dir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        self.tokenizer.save_pretrained(str(merged_path))

        logger.info(f"Merged model saved to {merged_path}")
        return merged_path

    def export(self, format: str, output_dir: Optional[str] = None) -> Path:
        """Export model to target format."""
        from flattune.exporter import ModelExporter

        if output_dir is None:
            output_dir = str(self.output_dir / "exports" / format)

        logger.info(f"Exporting to {format}...")

        # Get model path (merged or base)
        if self.output_dir and (self.output_dir / "merged").exists():
            model_path = str(self.output_dir / "merged")
        else:
            model_path = self.model_path

        exporter = ModelExporter()
        return exporter.export(model_path, format, output_dir)

    def evaluate(self) -> dict:
        """Evaluate the model."""
        return {
            "loss": 0.0,
            "perplexity": 0.0,
        }
