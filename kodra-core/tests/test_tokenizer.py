import unittest
import os
import sys

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from tokenizer.char_tokenizer import CharTokenizer

class TestTokenizer(unittest.TestCase):
    def test_encode_decode(self):
        tok = CharTokenizer()
        code_str = "def kodra_ai(): return 42"
        tok.train(code_str)
        tokens = tok.encode(code_str)
        decoded = tok.decode(tokens)
        self.assertEqual(code_str, decoded)

if __name__ == "__main__":
    unittest.main()
