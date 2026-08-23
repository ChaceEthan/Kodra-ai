"""
Fresh-training-mode safety helpers for Kodra GPT.

Centralizes the small set of rules that keep a "fresh" training run fully
isolated from any prior (legacy/diagnostic) tokenizer or checkpoint state -
FRESH_TRAINING is a hard safety switch that can only make a run MORE isolated
from prior state, never less. These rules back the notebook's CELL 06/08/17
so the behavior is unit-tested directly rather than only living as notebook
cell code nothing exercises automatically.
"""
from dataclasses import dataclass

DEFAULT_LEGACY_CHECKPOINT_DIR = "checkpoints"
DEFAULT_FRESH_CHECKPOINT_DIR = "checkpoints/fresh_training"
DEFAULT_FRESH_BPE_VOCAB_PATH = "tokenizer/vocab_bpe_fresh.json"
DEFAULT_LEGACY_BPE_VOCAB_PATH = "tokenizer/vocab_bpe.json"


@dataclass(frozen=True)
class FreshTrainingConfig:
    fresh_training: bool
    approved_dataset_dir: str
    fresh_tokenizer: bool
    use_old_checkpoint: bool
    checkpoint_dir: str
    legacy_checkpoint_dir: str
    tokenizer_path: str
    legacy_bpe_vocab_path: str


def resolve_fresh_training_config(
    fresh_training: bool,
    approved_dataset_dir: str,
    fresh_tokenizer: bool,
    use_old_checkpoint: bool,
    legacy_checkpoint_dir: str = DEFAULT_LEGACY_CHECKPOINT_DIR,
    fresh_checkpoint_dir: str = DEFAULT_FRESH_CHECKPOINT_DIR,
    fresh_bpe_vocab_path: str = DEFAULT_FRESH_BPE_VOCAB_PATH,
    legacy_bpe_vocab_path: str = DEFAULT_LEGACY_BPE_VOCAB_PATH,
) -> FreshTrainingConfig:
    """Applies the FRESH_TRAINING override: fresh_training=True forces
    fresh_tokenizer=True and use_old_checkpoint=False regardless of the values
    passed in, and routes checkpoints to fresh_checkpoint_dir instead of
    legacy_checkpoint_dir. approved_dataset_dir is always taken verbatim from
    the caller - this function never substitutes or hardcodes a dataset path
    of its own, so no external/private project directory can end up baked in
    here."""
    if fresh_training:
        fresh_tokenizer = True
        use_old_checkpoint = False
        checkpoint_dir = fresh_checkpoint_dir
    else:
        checkpoint_dir = legacy_checkpoint_dir if use_old_checkpoint else fresh_checkpoint_dir

    tokenizer_path = fresh_bpe_vocab_path if fresh_tokenizer else legacy_bpe_vocab_path

    return FreshTrainingConfig(
        fresh_training=fresh_training,
        approved_dataset_dir=approved_dataset_dir,
        fresh_tokenizer=fresh_tokenizer,
        use_old_checkpoint=use_old_checkpoint,
        checkpoint_dir=checkpoint_dir,
        legacy_checkpoint_dir=legacy_checkpoint_dir,
        tokenizer_path=tokenizer_path,
        legacy_bpe_vocab_path=legacy_bpe_vocab_path,
    )


def should_train_new_tokenizer(cfg: FreshTrainingConfig, tokenizer_path_exists: bool) -> bool:
    """True => the caller must print/do 'TRAINING NEW TOKENIZER FROM CLEAN DATASET'.
    False => the caller must print/do 'LOADING EXISTING CLEAN TOKENIZER'.
    Under fresh_tokenizer, this is always True, so a stale vocabulary trained on a
    different (e.g. tiny smoke-test) corpus can never be silently reused."""
    return cfg.fresh_tokenizer or not tokenizer_path_exists


def assert_checkpoint_reload_is_safe(cfg: FreshTrainingConfig) -> None:
    """Raises if a fresh run's resolved config would ever touch the legacy
    checkpoint lineage. This is the same guard CELL 17 runs immediately before
    reloading a checkpoint."""
    if cfg.fresh_training and cfg.use_old_checkpoint:
        raise AssertionError("FRESH_TRAINING forbids USE_OLD_CHECKPOINT.")
    if cfg.fresh_training and cfg.checkpoint_dir == cfg.legacy_checkpoint_dir:
        raise AssertionError(
            "FRESH_TRAINING must never point checkpoint_dir at the legacy checkpoint directory."
        )
