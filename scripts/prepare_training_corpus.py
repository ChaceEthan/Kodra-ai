"""
Prepare a clean, deterministic Kodra GPT training corpus from a single
explicitly-approved local source directory.

Usage:
    python scripts/prepare_training_corpus.py --source <approved-directory> --output <manifest.json>

This script NEVER scans the internet, NEVER clones a repository, and NEVER
walks outside the exact directory passed via --source (see
datasets/corpus_pipeline.py's discover_source_files, which only recurses
under the given root). Only local trees you have the rights to train on -
repository-owned sample data or an explicitly user-approved directory -
should be pointed at it.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "kodra-core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from datasets.corpus_pipeline import (
    build_manifest, write_manifest, build_training_text,
    contains_serialized_metadata, contains_replacement_character,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a clean, validated manifest + LM training text for Kodra GPT "
                     "from one explicitly-approved local source directory."
    )
    parser.add_argument("--source", required=True, help="Approved local directory to scan (never scans outside it).")
    parser.add_argument("--output", required=True, help="Path to write the dataset manifest JSON to.")
    parser.add_argument("--license", default="unknown", help="License to record for this corpus.")
    parser.add_argument("--dataset-name", default="user-approved-corpus", help="Provenance/source label recorded in the manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deterministic train/val split.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of files held out for validation.")
    parser.add_argument(
        "--min-total-chars", type=int, default=50_000,
        help="Warn if the clean corpus has fewer characters than this "
             "(Kodra Tiny needs real volume to learn syntax).",
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        print(f"ERROR: --source directory does not exist: {source}")
        return 1

    manifest = build_manifest(
        source,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=0.0,
        license=args.license,
        source=args.dataset_name,
    )
    write_manifest(manifest, args.output)

    train_text = build_training_text(manifest, split="train")
    val_text = build_training_text(manifest, split="val")

    print("=" * 60)
    print(" KODRA GPT TRAINING CORPUS REPORT")
    print("=" * 60)
    print(f" Source directory:       {source}")
    print(f" Manifest written to:    {args.output}")
    print(f" Clean files kept:       {manifest.num_files}")
    print(f" Total characters:       {manifest.total_chars}")
    print(f" Approx. tokens:         {manifest.total_token_estimate} (byte-level approximation)")
    print(f" Languages:              {manifest.language_counts}")
    print(f" Train / val files:      {manifest.split_counts['train']} / {manifest.split_counts['val']}")
    print(f" Duplicates removed:     {manifest.num_duplicates_removed}")
    print(f" Secrets redacted:       {manifest.num_secrets_redacted}")
    print(f" Encoding rejected:      {manifest.num_encoding_rejected}")
    print(f" Lockfiles rejected:     {manifest.num_lockfiles_rejected}")
    print(f" Quality-rejected:       {manifest.num_filtered_out}  {manifest.filtered_reasons}")
    print("=" * 60)

    # Safety net: re-validate the ACTUAL concatenated training text (not just
    # individual files in isolation) before declaring the corpus usable.
    ok = True
    for split_name, text in (("train", train_text), ("val", val_text)):
        if contains_serialized_metadata(text):
            print(f"FAIL: {split_name} text contains serialized metadata after assembly.")
            ok = False
        if contains_replacement_character(text):
            print(f"FAIL: {split_name} text contains U+FFFD after assembly.")
            ok = False
    if ok:
        print("VALIDATION: PASS - no metadata contamination, no U+FFFD in assembled training text.")
    else:
        print("VALIDATION: FAIL - do not train on this corpus until the above is fixed.")

    if manifest.total_chars < args.min_total_chars:
        print(
            f"\nWARNING: corpus has only {manifest.total_chars} characters "
            f"(< --min-total-chars={args.min_total_chars}). This is a smoke-test-sized "
            f"corpus, not enough to teach a model real Python syntax - add more approved "
            f"source files before running a serious training job."
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
