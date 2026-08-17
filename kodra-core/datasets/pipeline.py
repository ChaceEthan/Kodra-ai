import os
from tokenizer.char_tokenizer import CharTokenizer

class CodeDataPipeline:
    """
    Code Dataset Processing Pipeline for Kodra AI.
    Handles data cleaning, normalization, vocabulary building, and batching.
    """
    def __init__(self, raw_corpus: str):
        self.raw_corpus = raw_corpus
        self.tokenizer = CharTokenizer()

    def run(self) -> CharTokenizer:
        self.tokenizer.train(self.raw_corpus)
        return self.tokenizer

    def get_stats(self) -> dict:
        tokens = self.tokenizer.encode(self.raw_corpus)
        return {
            "char_count": len(self.raw_corpus),
            "token_count": len(tokens),
            "vocab_size": self.tokenizer.vocab_size,
            "unique_chars": len(set(self.raw_corpus)),
        }

    def print_summary(self) -> None:
        stats = self.get_stats()
        print("==================================================")
        print("        Kodra AI - Dataset Statistics            ")
        print("==================================================")
        print(f" Raw Characters:   {stats['char_count']}")
        print(f" Token Count:      {stats['token_count']}")
        print(f" Vocabulary Size:  {stats['vocab_size']}")
        print("==================================================")
