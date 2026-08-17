import unittest
import os
import sys
import torch

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import ModelConfig
from tokenizer.char_tokenizer import CharTokenizer
from model.gpt_model import KodraGPT
from inference.generator import CodeGenerator

class TestGenerator(unittest.TestCase):
    def test_generate(self):
        text = "def hello(): print('hello')"
        tok = CharTokenizer()
        tok.train(text)
        m_cfg = ModelConfig(context_length=32, embedding_dim=64, attention_heads=2, transformer_layers=2, vocab_size=tok.vocab_size)
        model = KodraGPT(m_cfg)
        gen = CodeGenerator(model, tok, torch.device("cpu"))
        output = gen.generate("def", max_new_tokens=10)
        self.assertTrue(output.startswith("def"))

if __name__ == "__main__":
    unittest.main()
