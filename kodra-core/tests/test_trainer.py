import unittest
import os
import sys

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import ModelConfig, TrainingConfig
from tokenizer.char_tokenizer import CharTokenizer
from datasets.dataset import create_dataloader
from model.gpt_model import KodraGPT
from training.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def test_train_epoch(self):
        text = "def test_fn(): pass\n" * 5
        tok = CharTokenizer()
        tok.train(text)
        m_cfg = ModelConfig(context_length=16, embedding_dim=64, attention_heads=2, transformer_layers=2, vocab_size=tok.vocab_size)
        t_cfg = TrainingConfig(batch_size=2)
        loader = create_dataloader(text, tok, context_length=16, batch_size=2)
        model = KodraGPT(m_cfg)
        trainer = Trainer(model, t_cfg, loader)
        loss = trainer.train_epoch(1)
        self.assertGreater(loss, 0)

if __name__ == "__main__":
    unittest.main()
