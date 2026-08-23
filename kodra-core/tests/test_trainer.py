import unittest
import os
import sys
import tempfile

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

    def test_checkpoint_save_load_and_resume(self):
        text = "def test_fn(): pass\n" * 5
        tok = CharTokenizer()
        tok.train(text)
        m_cfg = ModelConfig(context_length=16, embedding_dim=64, attention_heads=2, transformer_layers=2, vocab_size=tok.vocab_size)
        t_cfg = TrainingConfig(batch_size=2)
        loader = create_dataloader(text, tok, context_length=16, batch_size=2)

        model = KodraGPT(m_cfg)
        trainer = Trainer(model, t_cfg, loader, tokenizer_type=tok.tokenizer_type)
        trainer.train_epoch(1)
        steps_before_save = trainer.step_count
        self.assertGreater(steps_before_save, 0)

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "kodra_gpt_latest.pt")
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            reloaded_model = KodraGPT(m_cfg)
            reloaded_trainer = Trainer(reloaded_model, t_cfg, loader, tokenizer_type=tok.tokenizer_type)
            reloaded_trainer.load_checkpoint(ckpt_path)

            # State (step count, weights) must match what was saved.
            self.assertEqual(reloaded_trainer.step_count, steps_before_save)
            for p1, p2 in zip(trainer.model.parameters(), reloaded_trainer.model.parameters()):
                self.assertTrue((p1 == p2).all())

            # Resuming training must continue advancing the step count.
            reloaded_trainer.train_epoch(2)
            self.assertGreater(reloaded_trainer.step_count, steps_before_save)

if __name__ == "__main__":
    unittest.main()
