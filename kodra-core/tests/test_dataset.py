import unittest
import os
import sys
import torch

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from tokenizer.char_tokenizer import CharTokenizer
from datasets.dataset import CodeDataset, create_dataloader

class TestDataset(unittest.TestCase):
    def test_dataset_shape(self):
        tok = CharTokenizer()
        sample = "def add(a, b):\n    return a + b\n" * 10
        tok.train(sample)
        dataset = CodeDataset(sample, tok, context_length=32)
        x, y = dataset[0]
        self.assertEqual(x.shape[0], 32)
        self.assertEqual(y.shape[0], 32)

    def test_causal_next_token_shift(self):
        """y must be x shifted forward by exactly one token, i.e.
        input_ids = tokens[:-1], target_ids = tokens[1:] - the defining
        property of causal language-model training."""
        tok = CharTokenizer()
        sample = "abcdefghijklmnopqrstuvwxyz" * 4
        tok.train(sample)
        tokens = tok.encode(sample)
        dataset = CodeDataset(sample, tok, context_length=16)
        x, y = dataset[0]

        self.assertEqual(x.tolist(), tokens[0:16])
        self.assertEqual(y.tolist(), tokens[1:17])
        # Every position in y is the token that immediately follows the
        # corresponding position in x within the same chunk.
        self.assertEqual(x.tolist()[1:], y.tolist()[:-1])

if __name__ == "__main__":
    unittest.main()
