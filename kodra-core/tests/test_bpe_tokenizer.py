import os
import sys
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from tokenizer.bpe_tokenizer import ByteLevelBPETokenizer
from tokenizer.char_tokenizer import CharTokenizer


class TestBPETokenizer(unittest.TestCase):
    CORPUS = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n" * 20

    def test_train_and_roundtrip(self):
        tok = ByteLevelBPETokenizer(vocab_size=300)
        tok.train(self.CORPUS)
        text = "def add(a, b):\n    return a + b"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        self.assertEqual(text, decoded)

    def test_vocab_grows_toward_target(self):
        tok = ByteLevelBPETokenizer(vocab_size=300)
        tok.train(self.CORPUS)
        # Base vocab (specials + 256 bytes) is 260; merges should grow it, though
        # small corpora may not have enough repeated pairs to hit the exact target.
        self.assertGreater(tok.vocab_size, 260)
        self.assertLessEqual(tok.vocab_size, 300)

    def test_unicode_roundtrip(self):
        tok = ByteLevelBPETokenizer(vocab_size=300)
        tok.train(self.CORPUS + "\n# unicode: café ☃\n")
        text = "café ☃"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_deterministic_training(self):
        tok1 = ByteLevelBPETokenizer(vocab_size=300)
        tok1.train(self.CORPUS)
        tok2 = ByteLevelBPETokenizer(vocab_size=300)
        tok2.train(self.CORPUS)
        self.assertEqual(tok1.merges, tok2.merges)

    def test_save_load_roundtrip(self):
        tok = ByteLevelBPETokenizer(vocab_size=300)
        tok.train(self.CORPUS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bpe.json")
            tok.save(path)
            loaded = ByteLevelBPETokenizer()
            loaded.load(path)
            text = "def add(a, b):\n    return a + b"
            self.assertEqual(loaded.encode(text), tok.encode(text))
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_type_mismatch_rejected_on_load(self):
        char_tok = CharTokenizer()
        char_tok.train("hello world")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "char.json")
            char_tok.save(path)
            bpe_tok = ByteLevelBPETokenizer()
            with self.assertRaises(ValueError):
                bpe_tok.load(path)

    def test_char_tokenizer_still_works_phase1_compat(self):
        tok = CharTokenizer()
        text = "def kodra(): return 1"
        tok.train(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.tokenizer_type, "char")


if __name__ == "__main__":
    unittest.main()
