import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.gpt_model import KodraGPT
from tokenizer.char_tokenizer import CharTokenizer
from configs.config import load_config
from datasets.sample_code import SAMPLE_CODE_CORPUS
from inference.generator import CodeGenerator
from training.utils import get_device

def main():
    parser = argparse.ArgumentParser(description="Kodra AI - Autoregressive Code Generator")
    parser.add_argument("--prompt", type=str, default="def quicksort(arr):", help="Code prompt prefix")
    parser.add_argument("--max_tokens", type=int, default=64, help="Tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()

    tokenizer = CharTokenizer()
    tokenizer.train(SAMPLE_CODE_CORPUS)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = KodraGPT(cfg.model)
    generator = CodeGenerator(model, tokenizer, device)

    print("==================================================")
    print("      KODRA AI GENERATED CODE COMPLETION          ")
    print("==================================================")
    output = generator.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
    print(output)
    print("==================================================")

if __name__ == "__main__":
    main()
