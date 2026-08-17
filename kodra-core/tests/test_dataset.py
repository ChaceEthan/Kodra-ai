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

if __name__ == "__main__":
    unittest.main()
