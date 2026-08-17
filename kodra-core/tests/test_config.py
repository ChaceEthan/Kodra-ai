import unittest
import os
import sys

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import load_config, ModelConfig, TrainingConfig

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = load_config()
        self.assertEqual(cfg.model.context_length, 256)
        self.assertEqual(cfg.model.embedding_dim, 256)
        self.assertEqual(cfg.model.attention_heads, 4)
        self.assertEqual(cfg.model.transformer_layers, 6)
        self.assertEqual(cfg.training.batch_size, 16)

if __name__ == "__main__":
    unittest.main()
