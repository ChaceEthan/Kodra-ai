import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.sample_code import SAMPLE_CODE_CORPUS
from datasets.pipeline import CodeDataPipeline

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and train tokenizer for Kodra AI")
    parser.add_argument("--output_vocab", type=str, default="tokenizer/vocab.json", help="Path to save vocabulary")
    args = parser.parse_args()

    pipeline = CodeDataPipeline(SAMPLE_CODE_CORPUS)
    tokenizer = pipeline.run()
    pipeline.print_summary()

    tokenizer.save(args.output_vocab)
    print(f"Saved trained tokenizer vocabulary to {args.output_vocab}")

if __name__ == "__main__":
    main()
