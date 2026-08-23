import os
import sys
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from training.fresh_mode import (
    resolve_fresh_training_config, should_train_new_tokenizer,
    assert_checkpoint_reload_is_safe,
    DEFAULT_LEGACY_CHECKPOINT_DIR, DEFAULT_FRESH_CHECKPOINT_DIR,
    DEFAULT_FRESH_BPE_VOCAB_PATH, DEFAULT_LEGACY_BPE_VOCAB_PATH,
)


class TestFreshMode(unittest.TestCase):
    def test_fresh_training_forces_safe_flags_even_when_caller_tries_to_weaken_them(self):
        # Caller passes fresh_tokenizer=False and use_old_checkpoint=True - both must be
        # overridden because fresh_training=True is a hard safety switch.
        cfg = resolve_fresh_training_config(
            fresh_training=True,
            approved_dataset_dir="data/code",
            fresh_tokenizer=False,
            use_old_checkpoint=True,
        )
        self.assertTrue(cfg.fresh_tokenizer)
        self.assertFalse(cfg.use_old_checkpoint)
        self.assertEqual(cfg.checkpoint_dir, DEFAULT_FRESH_CHECKPOINT_DIR)
        self.assertNotEqual(cfg.checkpoint_dir, DEFAULT_LEGACY_CHECKPOINT_DIR)

    def test_fresh_training_never_reuses_legacy_tokenizer_path(self):
        cfg = resolve_fresh_training_config(
            fresh_training=True, approved_dataset_dir="data/code",
            fresh_tokenizer=True, use_old_checkpoint=False,
        )
        self.assertEqual(cfg.tokenizer_path, DEFAULT_FRESH_BPE_VOCAB_PATH)
        self.assertNotEqual(cfg.tokenizer_path, DEFAULT_LEGACY_BPE_VOCAB_PATH)

    def test_should_train_new_tokenizer_always_true_under_fresh_tokenizer(self):
        cfg = resolve_fresh_training_config(
            fresh_training=True, approved_dataset_dir="data/code",
            fresh_tokenizer=True, use_old_checkpoint=False,
        )
        # Even if a file happened to already exist at the fresh path (e.g. left over from
        # a previous fresh run), fresh_tokenizer=True must still force retraining - an old
        # tokenizer can never be silently reused.
        self.assertTrue(should_train_new_tokenizer(cfg, tokenizer_path_exists=True))
        self.assertTrue(should_train_new_tokenizer(cfg, tokenizer_path_exists=False))

    def test_should_load_existing_tokenizer_when_not_fresh_and_file_exists(self):
        cfg = resolve_fresh_training_config(
            fresh_training=False, approved_dataset_dir="data/code",
            fresh_tokenizer=False, use_old_checkpoint=True,
        )
        self.assertFalse(should_train_new_tokenizer(cfg, tokenizer_path_exists=True))
        self.assertTrue(should_train_new_tokenizer(cfg, tokenizer_path_exists=False))

    def test_fresh_training_blocks_legacy_checkpoint_reload(self):
        cfg = resolve_fresh_training_config(
            fresh_training=True, approved_dataset_dir="data/code",
            fresh_tokenizer=True, use_old_checkpoint=False,
        )
        # Safe by construction - must not raise.
        assert_checkpoint_reload_is_safe(cfg)

    def test_fresh_training_with_tampered_checkpoint_dir_is_rejected(self):
        # Simulates someone manually reassigning checkpoint_dir back to the legacy path
        # after resolve_fresh_training_config ran - the guard must still catch it.
        cfg = resolve_fresh_training_config(
            fresh_training=True, approved_dataset_dir="data/code",
            fresh_tokenizer=True, use_old_checkpoint=False,
        )
        import dataclasses
        tampered = dataclasses.replace(cfg, checkpoint_dir=DEFAULT_LEGACY_CHECKPOINT_DIR)
        with self.assertRaises(AssertionError):
            assert_checkpoint_reload_is_safe(tampered)

    def test_non_fresh_mode_with_use_old_checkpoint_is_allowed(self):
        cfg = resolve_fresh_training_config(
            fresh_training=False, approved_dataset_dir="data/code",
            fresh_tokenizer=False, use_old_checkpoint=True,
        )
        # Only fresh_training=True is a hard block; explicit non-fresh mode is allowed to
        # use the legacy lineage on purpose.
        assert_checkpoint_reload_is_safe(cfg)
        self.assertEqual(cfg.checkpoint_dir, DEFAULT_LEGACY_CHECKPOINT_DIR)

    def test_approved_dataset_dir_is_passed_through_verbatim_never_hardcoded(self):
        for path in ("data/code", "/some/user/approved/corpus", "C:/approved/corpus"):
            cfg = resolve_fresh_training_config(
                fresh_training=True, approved_dataset_dir=path,
                fresh_tokenizer=True, use_old_checkpoint=False,
            )
            self.assertEqual(cfg.approved_dataset_dir, path)


if __name__ == "__main__":
    unittest.main()
