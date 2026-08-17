"""
Progressive Kodra GPT model size roadmap.

Only "kodra-tiny" is small enough to train on a laptop CPU and has ever
actually been trained in this repository. The larger configurations below
are architecture definitions for future GPU/cloud training runs — they are
NOT trained models. Do not present their parameter counts as evidence of a
trained checkpoint; always check for an actual checkpoint file before
claiming a model of that size exists.
"""
from dataclasses import dataclass
from typing import Dict

from .config import ModelConfig


@dataclass
class ModelSizeSpec:
    name: str
    display_name: str
    description: str
    config: ModelConfig
    trained: bool  # True only if this exact config ships with a real checkpoint in this repo


def _approx_param_count(cfg: ModelConfig) -> int:
    """Rough parameter estimate for documentation purposes only — the
    authoritative count always comes from `model.count_parameters()` on an
    instantiated KodraGPT."""
    tok_emb = cfg.vocab_size * cfg.embedding_dim
    pos_emb = cfg.context_length * cfg.embedding_dim
    attn = 4 * cfg.embedding_dim * cfg.embedding_dim + 4 * cfg.embedding_dim
    mlp = 2 * cfg.embedding_dim * (4 * cfg.embedding_dim) + 5 * cfg.embedding_dim
    ln = 4 * cfg.embedding_dim
    block = attn + mlp + ln
    return tok_emb + pos_emb + cfg.transformer_layers * block + 2 * cfg.embedding_dim


def validate_model_config(cfg: ModelConfig) -> None:
    """Raise ValueError if a model configuration is structurally invalid."""
    if cfg.embedding_dim % cfg.attention_heads != 0:
        raise ValueError(
            f"embedding_dim ({cfg.embedding_dim}) must be divisible by "
            f"attention_heads ({cfg.attention_heads})"
        )
    if cfg.context_length <= 0:
        raise ValueError("context_length must be positive")
    if cfg.transformer_layers <= 0:
        raise ValueError("transformer_layers must be positive")
    if cfg.attention_heads <= 0:
        raise ValueError("attention_heads must be positive")
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError("dropout must be in [0, 1)")


MODEL_SIZES: Dict[str, ModelSizeSpec] = {
    "kodra-tiny": ModelSizeSpec(
        name="kodra-tiny",
        display_name="Kodra Tiny",
        description="Phase 1 research/educational model. Trainable on a laptop CPU.",
        config=ModelConfig(
            context_length=256, embedding_dim=256, attention_heads=4,
            transformer_layers=6, dropout=0.1, vocab_size=96,
        ),
        trained=True,
    ),
    "kodra-small": ModelSizeSpec(
        name="kodra-small",
        display_name="Kodra Small",
        description="~20-30M parameters. Requires a single consumer/cloud GPU.",
        config=ModelConfig(
            context_length=512, embedding_dim=512, attention_heads=8,
            transformer_layers=8, dropout=0.1, vocab_size=8000,
        ),
        trained=False,
    ),
    "kodra-base": ModelSizeSpec(
        name="kodra-base",
        display_name="Kodra Base",
        description="~100-150M parameters. Requires multi-GPU or a cloud training run.",
        config=ModelConfig(
            context_length=1024, embedding_dim=768, attention_heads=12,
            transformer_layers=12, dropout=0.1, vocab_size=16000,
        ),
        trained=False,
    ),
    "kodra-medium": ModelSizeSpec(
        name="kodra-medium",
        display_name="Kodra Medium",
        description="~300-500M parameters. Requires substantial multi-GPU cloud infrastructure.",
        config=ModelConfig(
            context_length=2048, embedding_dim=1280, attention_heads=20,
            transformer_layers=24, dropout=0.1, vocab_size=32000,
        ),
        trained=False,
    ),
    # "kodra-large" (1B+ parameters) is intentionally NOT defined here.
    # Per project policy, 1B+ configurations are roadmap documentation only
    # and should not be added as a runnable config until sufficient GPU
    # infrastructure and licensed/quality dataset volume exist.
}

for _spec in MODEL_SIZES.values():
    validate_model_config(_spec.config)


def normalize_model_size_name(name: str) -> str:
    """Accepts either the short form ("tiny", as used by the
    KODRA_MODEL_SIZE environment variable) or the full form ("kodra-tiny")
    and returns the full form used as a MODEL_SIZES key."""
    name = (name or "").strip().lower()
    if name in MODEL_SIZES:
        return name
    prefixed = f"kodra-{name}"
    if prefixed in MODEL_SIZES:
        return prefixed
    raise KeyError(f"Unknown model size '{name}'. Available: {list(MODEL_SIZES.keys())}")


def get_model_size(name: str) -> ModelSizeSpec:
    return MODEL_SIZES[normalize_model_size_name(name)]
