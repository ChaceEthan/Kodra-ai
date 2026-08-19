"""Honest, read-only Phase 3 training readiness report.

Usage: python scripts/training_readiness.py [--data PATH] [--tokenizer PATH]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "kodra-core")
sys.path.insert(0, CORE)

from configs.model_sizes import get_model_size, estimate_resources
from model.gpt_model import KodraGPT
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(CORE, "data", "processed"))
    p.add_argument("--tokenizer", default="")
    p.add_argument("--checkpoint-dir", default=os.path.join(CORE, "checkpoints"))
    p.add_argument("--model-size", default="tiny")
    args = p.parse_args()

    spec = get_model_size(args.model_size)
    model = KodraGPT(spec.config)
    params = model.count_parameters()
    manifest = os.path.join(args.data, "manifest.json")
    dataset_ready = os.path.isfile(manifest)
    tokenizer_ready = bool(args.tokenizer and os.path.isfile(args.tokenizer))
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
    blockers = []
    if not dataset_ready: blockers.append("dataset manifest missing")
    if not tokenizer_ready: blockers.append("compatible trained tokenizer missing")
    print("TRAINING READY" if not blockers else "BLOCKERS: " + "; ".join(blockers))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
