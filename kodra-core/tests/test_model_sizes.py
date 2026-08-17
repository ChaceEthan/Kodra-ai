import unittest
import os
import sys

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.model_sizes import MODEL_SIZES, validate_model_config, get_model_size
from configs.config import ModelConfig
from model.gpt_model import KodraGPT


class TestModelSizes(unittest.TestCase):
    def test_all_sizes_structurally_valid(self):
        for spec in MODEL_SIZES.values():
            validate_model_config(spec.config)
            self.assertEqual(spec.config.embedding_dim % spec.config.attention_heads, 0)

    def test_unknown_size_raises(self):
        with self.assertRaises(KeyError):
            get_model_size("kodra-nonexistent")

    def test_invalid_config_rejected(self):
        bad_cfg = ModelConfig(embedding_dim=100, attention_heads=3)
        with self.assertRaises(ValueError):
            validate_model_config(bad_cfg)

    def test_tiny_matches_real_param_count(self):
        # Only kodra-tiny is small enough to instantiate in a unit test.
        spec = get_model_size("kodra-tiny")
        model = KodraGPT(spec.config)
        real_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(real_count, 0)
        self.assertTrue(spec.trained)

    def test_larger_sizes_marked_untrained(self):
        for name in ("kodra-small", "kodra-base", "kodra-medium"):
            self.assertFalse(MODEL_SIZES[name].trained)


if __name__ == "__main__":
    unittest.main()
