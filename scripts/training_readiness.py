"""Honest, read-only Phase 3 training readiness report.

Usage: python scripts/training_readiness.py [--data PATH] [--tokenizer PATH]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "kodra-core")
sys.path.insert(0, CORE)

from configs.model_sizes import get_model_size, estimate_resources
from datasets.corpus_pipeline import DatasetManifest, build_training_text
from model.gpt_model import KodraGPT
from tokenizer.char_tokenizer import CharTokenizer
from training.readiness import check_serious_training_gate
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(CORE, "data", "processed"))
    p.add_argument("--tokenizer", default="")
    p.add_argument("--checkpoint-dir", default=os.path.join(CORE, "checkpoints"))
    p.add_argument("--model-size", default="tiny")
    args = p.parse_args()

    manifest = os.path.join(args.data, "manifest.json")
    dataset_ready = os.path.isfile(manifest)
    tokenizer_ready = bool(args.tokenizer and os.path.isfile(args.tokenizer))
    blockers = []
    train_text = ""
    val_text = ""
    token_estimate = 0
    if dataset_ready:
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                dataset_manifest = DatasetManifest(**json.load(f))
            train_text = build_training_text(dataset_manifest, split="train")
            val_text = build_training_text(dataset_manifest, split="val")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append(f"dataset manifest could not be assembled: {exc}")
    else:
        blockers.append("dataset manifest missing")

    tokenizer = None
    if tokenizer_ready:
        try:
            tokenizer = CharTokenizer()
            tokenizer.load(args.tokenizer)
            token_estimate = len(tokenizer.encode(train_text))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append(f"tokenizer could not be loaded: {exc}")
    else:
        blockers.append("compatible trained tokenizer missing")

    if tokenizer is not None and train_text and val_text:
        gate = check_serious_training_gate(
            train_text, val_text, token_estimate=token_estimate, tokenizer=tokenizer,
        )
        blockers.extend(gate.blockers)

    spec = get_model_size(args.model_size)
    if tokenizer is not None:
        spec.config.vocab_size = tokenizer.vocab_size
    model = KodraGPT(spec.config)
    params = model.count_parameters()
    resume = os.path.isfile(os.path.join(args.checkpoint_dir, "kodra_gpt_latest.pt"))
    gpu = torch.cuda.is_available()
    print("KodraGPT TRAINING READINESS")
    print(f"dataset_ready={dataset_ready}")
    print(f"dataset_manifest={manifest}")
    print(f"tokenizer_ready={tokenizer_ready}")
    print(f"model_configuration={spec.name}")
    print(f"parameter_count={params}")
    print(f"context_length={spec.config.context_length} vocab_size={spec.config.vocab_size}")
    print(f"device={'cuda' if gpu else 'cpu'} gpu_available={gpu}")
    r = estimate_resources(spec.config)
    print(f"estimated_training_vram_gb={r['training_vram_gb']:.2f} estimated_inference_vram_gb={r['inference_vram_gb']:.2f}")
    print(f"checkpoint_directory={args.checkpoint_dir} resume_checkpoint_available={resume}")
    print(f"assembled_train_tokens={token_estimate}")
    print("TRAINING READY" if not blockers else "BLOCKERS: " + "; ".join(blockers))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
