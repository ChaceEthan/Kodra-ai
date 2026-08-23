"""
Serious-training readiness gate for Kodra GPT.

This is distinct from the per-file quality filters in datasets/corpus_pipeline.py
(which decide whether an individual file enters the corpus at all). This module is
the final go/no-go check on the ASSEMBLED corpus + tokenizer, run immediately
before any serious (non-smoke) training loop is allowed to start. It exists so a
smoke-test-sized corpus - or contamination that somehow slipped past per-file
filtering - can never be silently treated as production-ready.

Kept intentionally dependency-light (no torch import) so it can be unit-tested
without a GPU/CPU-heavy model in the loop.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from datasets.corpus_pipeline import contains_serialized_metadata, contains_replacement_character

# Below this, a model cannot learn meaningful code syntax/structure - this is the
# hard minimum, not a target.
MIN_TOKENS_FOR_SERIOUS_TRAINING = 1_000_000
# The recommended range for a first meaningful Kodra Tiny experiment.
RECOMMENDED_MIN_TOKENS = 1_000_000
RECOMMENDED_MAX_TOKENS = 5_000_000


class _RoundTripTokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


@dataclass
class TrainingReadinessReport:
    passed: bool
    blockers: List[str] = field(default_factory=list)
    token_estimate: int = 0

    def summary(self) -> str:
        if self.passed:
            return f"READY: ~{self.token_estimate:,} estimated tokens, no blockers."
        lines = [f"NOT READY: ~{self.token_estimate:,} estimated tokens."]
        lines += [f"  - {b}" for b in self.blockers]
        return "\n".join(lines)


def check_serious_training_gate(
    train_text: str,
    val_text: str,
    token_estimate: int,
    tokenizer: Optional[_RoundTripTokenizer] = None,
    round_trip_sample: Optional[str] = None,
) -> TrainingReadinessReport:
    """Refuses (reports NOT READY for) serious training when any of the following
    hold, mirroring the required pre-flight checks for a real Kodra GPT run:

    - the corpus has fewer than MIN_TOKENS_FOR_SERIOUS_TRAINING estimated tokens
    - metadata contamination is present in the assembled train/val text
    - the Unicode replacement character (U+FFFD) is present in the assembled text
    - the validation split is empty
    - the tokenizer fails an encode/decode round-trip on a sample of the text

    This is a pure, deterministic check with no side effects - callers decide what
    to do with the report (e.g. the notebook's full-training cell raises on it)."""
    blockers: List[str] = []

    if token_estimate < MIN_TOKENS_FOR_SERIOUS_TRAINING:
        blockers.append(
            f"dataset has only ~{token_estimate:,} estimated tokens; serious training "
            f"requires at least {MIN_TOKENS_FOR_SERIOUS_TRAINING:,} "
            f"(recommended {RECOMMENDED_MIN_TOKENS:,}-{RECOMMENDED_MAX_TOKENS:,} for a first experiment)."
        )
    if contains_serialized_metadata(train_text) or contains_serialized_metadata(val_text):
        blockers.append("metadata contamination (a quoted dataset key like \"target\":) detected in the assembled text.")
    if contains_replacement_character(train_text) or contains_replacement_character(val_text):
        blockers.append("U+FFFD replacement character detected in the assembled text.")
    if not val_text.strip():
        blockers.append("validation split is empty.")
    if tokenizer is not None:
        sample = round_trip_sample if round_trip_sample is not None else train_text[:500]
        if sample and tokenizer.decode(tokenizer.encode(sample)) != sample:
            blockers.append("tokenizer round-trip failed on a sample of the training text.")

    return TrainingReadinessReport(passed=not blockers, blockers=blockers, token_estimate=token_estimate)
