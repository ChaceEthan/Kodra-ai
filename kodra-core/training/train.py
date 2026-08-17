import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import load_config
from tokenizer.char_tokenizer import CharTokenizer
from datasets.sample_code import SAMPLE_CODE_CORPUS
from datasets.dataset import create_dataloader
from model.gpt_model import KodraGPT
from training.trainer import Trainer
from training.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train Kodra AI Causal GPT Model")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = get_device()

    tokenizer = CharTokenizer()
    tokenizer.train(SAMPLE_CODE_CORPUS)
    cfg.model.vocab_size = tokenizer.vocab_size

    dataloader = create_dataloader(SAMPLE_CODE_CORPUS, tokenizer, cfg.model.context_length, cfg.training.batch_size)
    model = KodraGPT(cfg.model)

    print(f"Starting Kodra AI Training on {device}...")
    print(f"Model Parameters: {model.count_parameters():,}")

    trainer = Trainer(model, cfg.training, dataloader, device=device)

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f}")

    ckpt_path = os.path.join(args.checkpoint_dir, "kodra_gpt_latest.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
