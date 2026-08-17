import json
import os
from collections import Counter
from typing import Dict, List, Tuple

# Special tokens reserved at the start of every BPE vocabulary.
# Fixed order so tokenizer training is deterministic across runs.
SPECIAL_TOKENS: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class ByteLevelBPETokenizer:
    """
    Production-oriented Byte-Level BPE tokenizer for Kodra AI (Phase 2).

    Operates on raw UTF-8 bytes (0-255), so it can losslessly encode any text
    without an <UNK> fallback for individual characters, following the
    approach used by GPT-2 style tokenizers. Training is deterministic: pair
    merges are selected by (frequency desc, pair asc) so the same corpus and
    vocab_size always produce the same merge list.
    """
    tokenizer_type: str = "bpe"
    tokenizer_version: int = 1

    def __init__(self, vocab_size: int = 8000):
        self.target_vocab_size = vocab_size
        self.special_tokens = list(SPECIAL_TOKENS)
        self.merges: Dict[Tuple[int, int], int] = {}
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.vocab_size: int = 0
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        self.itos = {}
        for i, tok in enumerate(self.special_tokens):
            self.itos[i] = tok.encode("utf-8")
        offset = len(self.special_tokens)
        for b in range(256):
            self.itos[offset + b] = bytes([b])
        self.vocab_size = len(self.itos)

    def train(self, text: str, vocab_size: int = None) -> None:
        if vocab_size is not None:
            self.target_vocab_size = vocab_size
        self._build_base_vocab()
        self.merges = {}

        base_offset = len(self.special_tokens)
        tokens: List[int] = [base_offset + b for b in text.encode("utf-8")]

        num_merges = max(0, self.target_vocab_size - self.vocab_size)
        for _ in range(num_merges):
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if pair_counts[best_pair] < 2:
                break

            new_id = self.vocab_size
            self.merges[best_pair] = new_id
            self.itos[new_id] = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.vocab_size += 1
            tokens = self._merge_pair(tokens, best_pair, new_id)

        self.stoi = {}

    def _merge_pair(self, tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        merged = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def encode(self, text: str) -> List[int]:
        base_offset = len(self.special_tokens)
        tokens: List[int] = [base_offset + b for b in text.encode("utf-8")]
        for pair, new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            tokens = self._merge_pair(tokens, pair, new_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        special_ids = set(range(len(self.special_tokens)))
        raw = bytearray()
        for t in tokens:
            if t in special_ids:
                continue
            raw.extend(self.itos.get(t, b""))
        return raw.decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> int:
        return self.special_tokens.index(token) if token in self.special_tokens else -1

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "tokenizer_type": self.tokenizer_type,
            "tokenizer_version": self.tokenizer_version,
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            # merges stored as ["id_a", "id_b", "new_id"] triples, ordered by new_id
            "merges": [
                [a, b, new_id]
                for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
            "itos": {str(k): list(v) for k, v in self.itos.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_type = data.get("tokenizer_type", "bpe")
        if file_type != self.tokenizer_type:
            raise ValueError(
                f"Tokenizer type mismatch: file was saved as '{file_type}' but "
                f"ByteLevelBPETokenizer expects '{self.tokenizer_type}'"
            )
        self.special_tokens = data["special_tokens"]
        self.target_vocab_size = data["target_vocab_size"]
        self.vocab_size = data["vocab_size"]
        self.merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        self.itos = {int(k): bytes(v) for k, v in data["itos"].items()}
