import unittest
import os
import sys
import torch

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import ModelConfig
from model.gpt_model import KodraGPT, IsaacCodeGPT

class TestModel(unittest.TestCase):
    def test_kodragpt_forward(self):
        config = ModelConfig(
            context_length=64,
            embedding_dim=128,
            attention_heads=2,
            transformer_layers=2,
            vocab_size=50
        )
        model = KodraGPT(config)
        idx = torch.randint(0, 50, (2, 32))
        targets = torch.randint(0, 50, (2, 32))
        logits, loss = model(idx, targets)
        self.assertIsNotNone(logits)
        self.assertIsNotNone(loss)

    def test_legacy_alias(self):
        config = ModelConfig(context_length=64, embedding_dim=128, attention_heads=2, transformer_layers=2, vocab_size=50)
        legacy_model = IsaacCodeGPT(config)
        self.assertIsInstance(legacy_model, KodraGPT)

if __name__ == "__main__":
    unittest.main()
