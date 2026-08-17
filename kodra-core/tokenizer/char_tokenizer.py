import json
import os
from typing import List, Dict

class CharTokenizer:
    """
    Character-level tokenizer for Kodra AI code models (Phase 1).
    Maps string code sequences to integer token IDs losslessly.
    """
    tokenizer_type: str = "char"
    tokenizer_version: int = 1

    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0

    def train(self, text: str) -> None:
        unique_chars = sorted(list(set(text)))
        # Reserve 0 for <PAD>, 1 for <UNK>
        if "<PAD>" not in unique_chars:
            unique_chars = ["<PAD>", "<UNK>"] + unique_chars
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> List[int]:
        if not self.stoi:
            self.train(text)
        return [self.stoi.get(c, self.stoi.get("<UNK>", 1)) for c in text]

    def decode(self, tokens: List[int]) -> str:
        res = []
        for t in tokens:
            ch = self.itos.get(t, "")
            if ch not in ("<PAD>", "<UNK>"):
                res.append(ch)
        return "".join(res)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_type": self.tokenizer_type,
                "tokenizer_version": self.tokenizer_version,
                "vocab_size": self.vocab_size,
                "stoi": self.stoi,
                "itos": {str(k): v for k, v in self.itos.items()},
            }, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_type = data.get("tokenizer_type", "char")
        if file_type != self.tokenizer_type:
            raise ValueError(
                f"Tokenizer type mismatch: file was saved as '{file_type}' but "
                f"CharTokenizer expects '{self.tokenizer_type}'"
            )
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)
