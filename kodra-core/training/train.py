import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import load_config
from tokenizer.char_tokenizer import CharTokenizer
from datasets.corpus_pipeline import DatasetManifest, build_training_text
from datasets.dataset import create_dataloader
from model.gpt_model import KodraGPT
from training.trainer import Trainer
from training.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train Kodra AI Causal GPT Model")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config file")
    parser.add_argument("--manifest", required=True, help="Path to a prepared corpus manifest.json")
    parser.add_argument("--tokenizer", default="checkpoints/kodra_char_tokenizer.json", help="Tokenizer artifact path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = DatasetManifest(**json.load(f))
    train_text = build_training_text(manifest, split="train")
    val_text = build_training_text(manifest, split="val")
    if not train_text.strip():
        raise ValueError("The manifest has no training text.")
    if not val_text.strip():
        raise ValueError("The manifest has no validation text; adjust the preparation split.")

    cfg = load_config(args.config)
    cfg.training.max_epochs = args.epochs
    set_seed(cfg.training.seed)
    device = get_device()

    tokenizer = CharTokenizer()
    if os.path.exists(args.tokenizer):
        tokenizer.load(args.tokenizer)
    else:
        tokenizer.train(train_text)
        tokenizer.save(args.tokenizer)
    cfg.model.vocab_size = tokenizer.vocab_size

    train_loader = create_dataloader(train_text, tokenizer, cfg.model.context_length, cfg.training.batch_size)
    val_loader = create_dataloader(
        val_text, tokenizer, cfg.model.context_length, cfg.training.batch_size, shuffle=False,
    )
    model = KodraGPT(cfg.model)

    print(f"Starting Kodra AI Training on {device}...")
    print(f"Model Parameters: {model.count_parameters():,}")

    manifest_id = f"{manifest.source}-seed{manifest.seed}-{manifest.created_at}"
    trainer = Trainer(
        model, cfg.training, train_loader, val_loader=val_loader, device=device,
        tokenizer_type=tokenizer.tokenizer_type, tokenizer_version=tokenizer.tokenizer_version,
        dataset_manifest_id=manifest_id,
    )

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(epoch, total_steps=args.epochs * len(train_loader))
        val_loss = trainer.evaluate()
        trainer.save_latest_and_best(args.checkpoint_dir, val_loss=val_loss)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | Val loss: {val_loss:.4f}")

    ckpt_path = os.path.join(args.checkpoint_dir, "kodra_gpt_latest.pt")
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
