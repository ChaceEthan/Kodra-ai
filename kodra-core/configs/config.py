import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class ModelConfig:
    context_length: int = 256
    embedding_dim: int = 256
    attention_heads: int = 4
    transformer_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = 96

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_epochs: int = 20
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 42
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    min_lr_ratio: float = 0.1
    checkpoint_interval_steps: int = 200
    validation_interval_steps: int = 200
    mixed_precision: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_cfg = ModelConfig(**data.get("model", {}))
        train_cfg = TrainingConfig(**data.get("training", {}))
        return Config(model=model_cfg, training=train_cfg)
    return Config()

def save_config(config: Config, config_path: str) -> None:
    data = {
        "model": asdict(config.model),
        "training": asdict(config.training),
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
