import os
import sys
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from training.readiness import check_serious_training_gate, MIN_TOKENS_FOR_SERIOUS_TRAINING
from tokenizer.char_tokenizer import CharTokenizer


CLEAN_LARGE_TRAIN_TEXT = ("def add(a, b):\n    return a + b\n\n" * 60_000)  # well over 1M chars
CLEAN_VAL_TEXT = "def sub(a, b):\n    return a - b\n"


class TestServeriousTrainingGate(unittest.TestCase):
    def test_current_733_char_dataset_fails_gate(self):
        # This is exactly the size of the bundled data/code smoke-test corpus.
        train_text = "def kodra_ai_binary_search(arr, target):\n    return -1\n"
        val_text = ""
        report = check_serious_training_gate(train_text, val_text, token_estimate=733)
        self.assertFalse(report.passed)
        self.assertTrue(any("estimated tokens" in b for b in report.blockers))

    def test_large_clean_corpus_with_working_tokenizer_passes(self):
        tok = CharTokenizer()
        tok.train(CLEAN_LARGE_TRAIN_TEXT + CLEAN_VAL_TEXT)
        report = check_serious_training_gate(
            CLEAN_LARGE_TRAIN_TEXT, CLEAN_VAL_TEXT,
            token_estimate=len(CLEAN_LARGE_TRAIN_TEXT),
            tokenizer=tok,
        )
        self.assertTrue(report.passed, msg=report.summary())
        self.assertEqual(report.blockers, [])

    def test_gate_blocks_on_metadata_contamination(self):
        contaminated = CLEAN_LARGE_TRAIN_TEXT + '\n{"prompt": "p", "target": "t"}\n'
        report = check_serious_training_gate(
            contaminated, CLEAN_VAL_TEXT, token_estimate=len(contaminated)
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("metadata contamination" in b for b in report.blockers))

    def test_gate_blocks_on_replacement_character(self):
        corrupted = CLEAN_LARGE_TRAIN_TEXT + "\n�\n"
        report = check_serious_training_gate(
            corrupted, CLEAN_VAL_TEXT, token_estimate=len(corrupted)
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("U+FFFD" in b for b in report.blockers))

    def test_gate_blocks_on_empty_validation_split(self):
        report = check_serious_training_gate(
            CLEAN_LARGE_TRAIN_TEXT, "", token_estimate=len(CLEAN_LARGE_TRAIN_TEXT)
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("validation split is empty" in b for b in report.blockers))

    def test_gate_blocks_on_tokenizer_round_trip_failure(self):
        class BrokenTokenizer:
            def encode(self, text):
                return [1, 2, 3]

            def decode(self, tokens):
                return "not the original text"

        report = check_serious_training_gate(
            CLEAN_LARGE_TRAIN_TEXT, CLEAN_VAL_TEXT,
            token_estimate=len(CLEAN_LARGE_TRAIN_TEXT),
            tokenizer=BrokenTokenizer(),
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("round-trip" in b for b in report.blockers))

    def test_min_tokens_constant_is_one_million(self):
        self.assertEqual(MIN_TOKENS_FOR_SERIOUS_TRAINING, 1_000_000)


if __name__ == "__main__":
    unittest.main()
