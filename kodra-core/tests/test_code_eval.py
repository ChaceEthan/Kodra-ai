import os
import sys
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import ModelConfig
from tokenizer.char_tokenizer import CharTokenizer
from model.gpt_model import KodraGPT
from inference.generator import CodeGenerator
from evaluation.code_eval import (
    run_code_completion_eval, run_syntax_eval, run_future_agent_eval_placeholders,
    python_parses, json_is_valid,
)


class TestCodeEval(unittest.TestCase):
    def setUp(self):
        text = "def add(a, b):\n    return a + b\n" * 10
        self.tok = CharTokenizer()
        self.tok.train(text)
        cfg = ModelConfig(context_length=32, embedding_dim=32, attention_heads=2, transformer_layers=2, vocab_size=self.tok.vocab_size)
        model = KodraGPT(cfg)
        self.generator = CodeGenerator(model, self.tok, device="cpu")

    def test_code_completion_eval_runs_and_returns_real_scores(self):
        report = run_code_completion_eval(self.generator, max_new_tokens=8)
        self.assertIn("pass_rate", report)
        self.assertEqual(report["total"], len(report["results"]))
        self.assertGreaterEqual(report["pass_rate"], 0.0)
        self.assertLessEqual(report["pass_rate"], 1.0)

    def test_syntax_eval_runs(self):
        report = run_syntax_eval(self.generator, num_python_samples=2, max_new_tokens=8)
        self.assertIn("python_parse_rate", report)
        self.assertEqual(len(report["python_results"]), 2)

    def test_future_agent_placeholders_are_explicitly_unimplemented(self):
        report = run_future_agent_eval_placeholders()
        for entry in report.values():
            self.assertEqual(entry["status"], "not_implemented")
            self.assertIsNone(entry["score"])

    def test_python_parses_detects_syntax_errors(self):
        self.assertTrue(python_parses("def f():\n    return 1\n"))
        self.assertFalse(python_parses("def f(:\n    return 1\n"))

    def test_json_is_valid(self):
        self.assertTrue(json_is_valid('{"a": 1}'))
        self.assertFalse(json_is_valid("{a: 1"))


if __name__ == "__main__":
    unittest.main()
