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


if __name__ == "__main__":
    unittest.main()
