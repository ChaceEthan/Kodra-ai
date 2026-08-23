import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "prepare_training_corpus.py")


class TestPrepareTrainingCorpusScript(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.tmp_dir, "approved_source")
        os.makedirs(self.source_dir, exist_ok=True)
        with open(os.path.join(self.source_dir, "a.py"), "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n" * 5)
        with open(os.path.join(self.source_dir, "b.md"), "w", encoding="utf-8") as f:
            f.write("# Docs\n\nThis explains the `add` function in some detail.\n")
        self.output_manifest = os.path.join(self.tmp_dir, "manifest.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, extra_args=None):
        args = [
            sys.executable, SCRIPT_PATH,
            "--source", self.source_dir,
            "--output", self.output_manifest,
        ] + (extra_args or [])
        return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)

    def test_builds_manifest_and_reports_stats(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertTrue(os.path.exists(self.output_manifest))

        with open(self.output_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["num_files"], 2)
        self.assertIn("python", manifest["language_counts"])
        self.assertIn("markdown", manifest["language_counts"])
        self.assertIn("VALIDATION: PASS", result.stdout)

    def test_rejects_source_directory_that_does_not_exist(self):
        result = self._run(["--source", os.path.join(self.tmp_dir, "does_not_exist")])
        self.assertNotEqual(result.returncode, 0)

    def test_warns_when_corpus_too_small(self):
        result = self._run(["--min-total-chars", "1000000"])
        self.assertIn("WARNING", result.stdout)

    def test_never_scans_outside_source_directory(self):
        # A sibling directory to source_dir must never appear in the manifest,
        # proving the script only walks the exact --source path given.
        sibling_dir = os.path.join(self.tmp_dir, "not_approved")
        os.makedirs(sibling_dir, exist_ok=True)
        with open(os.path.join(sibling_dir, "secret_plan.py"), "w", encoding="utf-8") as f:
            f.write("def top_secret():\n    return 'should never be scanned'\n")

        self._run()
        with open(self.output_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        rels = [rec["relative_path"] for rec in manifest["files"]]
        self.assertFalse(any("secret_plan" in r for r in rels))


if __name__ == "__main__":
    unittest.main()
