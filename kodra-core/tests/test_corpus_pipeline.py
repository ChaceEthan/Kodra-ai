import os
import sys
import shutil
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from datasets.corpus_pipeline import (
    build_manifest, write_manifest, discover_source_files,
    contains_secret, is_binary,
    contains_serialized_metadata, contains_replacement_character,
)


class TestCorpusPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp_dir, "node_modules"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, "src"), exist_ok=True)

        with open(os.path.join(self.tmp_dir, "src", "a.py"), "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n")
        with open(os.path.join(self.tmp_dir, "src", "b.py"), "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n")  # exact duplicate of a.py
        with open(os.path.join(self.tmp_dir, "src", "c.js"), "w", encoding="utf-8") as f:
            f.write("function mul(a, b) { return a * b; }\n")
        with open(os.path.join(self.tmp_dir, "src", "secret.py"), "w", encoding="utf-8") as f:
            f.write('AWS_KEY = "AKIAABCDEFGHIJKLMNOP"\n')
        with open(os.path.join(self.tmp_dir, "node_modules", "vendor.js"), "w", encoding="utf-8") as f:
            f.write("var vendored = true;\n")
        with open(os.path.join(self.tmp_dir, "src", "binary.py"), "wb") as f:
            f.write(b"\x00\x01\x02binarydata")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_discovery_excludes_vendor_and_binary(self):
        files = discover_source_files(self.tmp_dir)
        rels = [os.path.relpath(p, self.tmp_dir).replace(os.sep, "/") for p in files]
        self.assertIn("src/a.py", rels)
        self.assertNotIn("node_modules/vendor.js", rels)
        self.assertNotIn("src/binary.py", rels)

    def test_manifest_dedup_and_secret_filtering(self):
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertIn("src/a.py", rels)
        self.assertNotIn("src/b.py", rels)  # exact duplicate removed
        self.assertNotIn("src/secret.py", rels)  # secret redacted
        self.assertEqual(manifest.num_duplicates_removed, 1)
        self.assertEqual(manifest.num_secrets_redacted, 1)
        self.assertIn("python", manifest.language_counts)
        self.assertIn("javascript", manifest.language_counts)

    def test_manifest_deterministic_across_runs(self):
        m1 = build_manifest(self.tmp_dir, seed=7)
        m2 = build_manifest(self.tmp_dir, seed=7)
        self.assertEqual(
            [f["relative_path"] for f in m1.files],
            [f["relative_path"] for f in m2.files],
        )
        self.assertEqual(
            [f["split"] for f in m1.files],
            [f["split"] for f in m2.files],
        )

    def test_write_manifest_roundtrip(self):
        manifest = build_manifest(self.tmp_dir, seed=1)
        out_path = os.path.join(self.tmp_dir, "manifest.json")
        write_manifest(manifest, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_secret_detection(self):
        self.assertTrue(contains_secret('AWS_KEY = "AKIAABCDEFGHIJKLMNOP"'))
        self.assertFalse(contains_secret("def add(a, b): return a + b"))

    def test_detects_serialized_metadata(self):
        self.assertTrue(contains_serialized_metadata('{"prompt": "hi", "target": "there"}'))
        self.assertTrue(contains_serialized_metadata("{'category': 'debugging'}"))

    def test_does_not_flag_legitimate_target_variable(self):
        # `target` used as an ordinary Python variable followed by a block
        # colon must never be treated as leaked metadata - this is the exact
        # code shape that produced a false-positive "contamination" alert.
        code = (
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
        )
        self.assertFalse(contains_serialized_metadata(code))

    def test_detects_replacement_character(self):
        self.assertTrue(contains_replacement_character("def f():\n    return �\n"))
        self.assertFalse(contains_replacement_character("def f():\n    return 1\n"))

    def test_manifest_rejects_serialized_metadata_record(self):
        with open(os.path.join(self.tmp_dir, "src", "leaked.py"), "w", encoding="utf-8") as f:
            f.write('TRAIN_EXAMPLE = \'{"prompt": "write a function", "target": "def f(): pass", "category": "code_generation"}\'\n')
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/leaked.py", rels)

    def test_manifest_keeps_legitimate_code_with_target_variable(self):
        with open(os.path.join(self.tmp_dir, "src", "search.py"), "w", encoding="utf-8") as f:
            f.write(
                "def binary_search(arr, target):\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"
            )
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertIn("src/search.py", rels)

    def test_manifest_rejects_replacement_character(self):
        with open(os.path.join(self.tmp_dir, "src", "corrupt.py"), "w", encoding="utf-8") as f:
            f.write("def f():\n    return �\n    # padding to clear min length filter\n")
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/corrupt.py", rels)


if __name__ == "__main__":
    unittest.main()
