"""Trainer configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Configuration for training.

    This is a higher-level training configuration that combines
    all settings needed for the trainer.
    """
    # Training parameters
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Model parameters
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "outputs"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_accumulation": self.gradient_accumulation,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
