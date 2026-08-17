import math
import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.gpt_model import KodraGPT
from tokenizer.char_tokenizer import CharTokenizer
from configs.config import load_config
from datasets.sample_code import SAMPLE_CODE_CORPUS
from datasets.dataset import create_dataloader
from inference.generator import CodeGenerator
from evaluation.code_eval import (
    run_code_completion_eval, run_syntax_eval, run_future_agent_eval_placeholders,
)

class ModelEvaluator:
    """
    Evaluator for Kodra AI models calculating validation loss and perplexity.
    """
    def __init__(self, model: KodraGPT, tokenizer: CharTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def evaluate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                if loss is not None:
                    total_loss += loss.item()
                    batches += 1

        avg_loss = total_loss / max(1, batches)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return {"val_loss": avg_loss, "perplexity": perplexity}

    def print_report(self, val_loader: DataLoader) -> None:
        metrics = self.evaluate(val_loader)
        print("==================================================")
        print("         KODRA AI EVALUATION REPORT              ")
        print("==================================================")
        print(f" Validation Loss: {metrics['val_loss']:.4f}")
        print(f" Perplexity:      {metrics['perplexity']:.2f}")
        print("==================================================")


def full_evaluation_report(model: KodraGPT, tokenizer: CharTokenizer, device: torch.device, val_loader: DataLoader) -> dict:
    """Runs every real, executed evaluation category and returns one combined
    report. No category in this report is ever a fabricated/hard-coded score."""
    lm_evaluator = ModelEvaluator(model, tokenizer, device)
    generator = CodeGenerator(model, tokenizer, device)

    return {
        "language_model": lm_evaluator.evaluate(val_loader),
        "code_completion": run_code_completion_eval(generator),
        "syntax": run_syntax_eval(generator),
        "future_agent_placeholders": run_future_agent_eval_placeholders(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kodra AI Model")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--full", action="store_true", help="Run the full evaluation suite (LM + code completion + syntax)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    tokenizer = CharTokenizer()
    tokenizer.train(SAMPLE_CODE_CORPUS)
    cfg.model.vocab_size = tokenizer.vocab_size

    dataloader = create_dataloader(SAMPLE_CODE_CORPUS, tokenizer, cfg.model.context_length, cfg.training.batch_size)
    model = KodraGPT(cfg.model)

    if args.full:
        report = full_evaluation_report(model, tokenizer, device, dataloader)
        print(json.dumps(report, indent=2, default=str))
    else:
        evaluator = ModelEvaluator(model, tokenizer, device)
        evaluator.print_report(dataloader)

if __name__ == "__main__":
    main()
