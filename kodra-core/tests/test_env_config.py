import os
import sys
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

import configs.env as env_module
from configs.env import load_runtime_config, _split_csv, _parse_bool
from training.utils import get_device


class TestEnvConfig(unittest.TestCase):
    def setUp(self):
        # Snapshot and clear the relevant env vars so tests don't leak into
        # each other or pick up the real developer's environment.
        self._keys = [
            "KODRA_DEVICE", "KODRA_CHECKPOINT_DIR", "KODRA_DATA_DIR",
            "KODRA_ALLOWED_ORIGINS", "KODRA_REQUIRE_TOOL_APPROVAL",
            "KODRA_ENABLE_TERMINAL_TOOLS", "LOG_LEVEL", "KODRA_MODEL_SIZE",
            "KODRA_API_KEY",
        ]
        self._snapshot = {k: os.environ.get(k) for k in self._keys}
        for k in self._keys:
            os.environ.pop(k, None)
        env_module._ENV_LOADED = True  # prevent .env file loading during tests

    def tearDown(self):
        for k in self._keys:
            if self._snapshot[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self._snapshot[k]

    def test_defaults_are_safe_with_no_env_set(self):
        cfg = load_runtime_config(kodra_core_dir="/tmp/kodra-core")
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.allowed_origins, ["http://localhost:3000"])
        self.assertTrue(cfg.require_tool_approval)
        self.assertFalse(cfg.enable_terminal_tools)
        self.assertEqual(cfg.model_size, "tiny")
        self.assertEqual(cfg.api_key, "")

    def test_env_vars_override_defaults(self):
        os.environ["KODRA_DEVICE"] = "cpu"
        os.environ["KODRA_ALLOWED_ORIGINS"] = "http://a.com, http://b.com"
        os.environ["KODRA_REQUIRE_TOOL_APPROVAL"] = "false"
        os.environ["KODRA_ENABLE_TERMINAL_TOOLS"] = "true"
        os.environ["KODRA_MODEL_SIZE"] = "small"

        cfg = load_runtime_config(kodra_core_dir="/tmp/kodra-core")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.allowed_origins, ["http://a.com", "http://b.com"])
        self.assertFalse(cfg.require_tool_approval)
        self.assertTrue(cfg.enable_terminal_tools)
        self.assertEqual(cfg.model_size, "small")

    def test_split_csv(self):
        self.assertEqual(_split_csv("a, b,c"), ["a", "b", "c"])
        self.assertEqual(_split_csv(""), [])

    def test_parse_bool(self):
        self.assertTrue(_parse_bool("true", default=False))
        self.assertTrue(_parse_bool("1", default=False))
        self.assertFalse(_parse_bool("false", default=True))
        self.assertEqual(_parse_bool("", default=True), True)

    def test_dotenv_file_parsing_does_not_override_real_env(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = os.path.join(d, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("KODRA_DEVICE=cuda\n# comment\nKODRA_MODEL_SIZE=base\n")

            os.environ["KODRA_DEVICE"] = "cpu"  # already set - must not be overridden
            env_module._ENV_LOADED = False
            env_module.load_dotenv_once(env_path)

            self.assertEqual(os.environ["KODRA_DEVICE"], "cpu")
            self.assertEqual(os.environ["KODRA_MODEL_SIZE"], "base")

    def test_get_device_explicit_cpu(self):
        self.assertEqual(str(get_device("cpu")), "cpu")

    def test_get_device_auto_default(self):
        # Should not raise regardless of hardware.
        device = get_device("auto")
        self.assertIn(str(device), ("cpu", "cuda", "mps"))


if __name__ == "__main__":
    unittest.main()
