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
    looks_minified_or_generated, build_training_text,
    is_generated_lockfile,
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

    def test_write_manifest_supports_bare_filename(self):
        manifest = build_manifest(self.tmp_dir, seed=1)
        old_cwd = os.getcwd()
        try:
            os.chdir(self.tmp_dir)
            write_manifest(manifest, "manifest.json")
        finally:
            os.chdir(old_cwd)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "manifest.json")))

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

    def test_detects_minified_and_generated_files(self):
        minified = "var a=1;function f(x){return x*2}" * 60  # one long, dense line
        self.assertTrue(looks_minified_or_generated(minified))
        generated = "// AUTO-GENERATED FILE. DO NOT EDIT.\nfunction f() { return 1; }\n"
        self.assertTrue(looks_minified_or_generated(generated))
        normal = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
        self.assertFalse(looks_minified_or_generated(normal))

    def test_manifest_rejects_generated_banner_file(self):
        with open(os.path.join(self.tmp_dir, "src", "generated.js"), "w", encoding="utf-8") as f:
            f.write("// AUTO-GENERATED FILE. DO NOT EDIT.\nfunction f() { return 1 + 1; }\n")
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/generated.js", rels)
        self.assertIn("filter_not_minified_or_generated", manifest.filtered_reasons)

    def test_manifest_reports_filtered_reasons_breakdown(self):
        with open(os.path.join(self.tmp_dir, "src", "leaked.py"), "w", encoding="utf-8") as f:
            f.write('X = \'{"prompt": "p", "target": "t"}\'\n')
        manifest = build_manifest(self.tmp_dir, seed=42)
        self.assertEqual(
            manifest.num_filtered_out,
            sum(manifest.filtered_reasons.values()),
        )
        self.assertIn("filter_no_serialized_metadata", manifest.filtered_reasons)

    def test_build_training_text_excludes_manifest_metadata(self):
        manifest = build_manifest(self.tmp_dir, seed=42, license="MIT", source="unit-test-corpus")
        train_text = build_training_text(manifest, split="train")
        # The concatenated training text must be pure file content - no
        # manifest bookkeeping (license, source label, sha256, ...) leaking in.
        self.assertNotIn("MIT", train_text)
        self.assertNotIn("unit-test-corpus", train_text)
        self.assertIn("def add(a, b):", train_text)

    def test_build_training_text_rejects_changed_source(self):
        manifest = build_manifest(self.tmp_dir, seed=42, val_ratio=0.0, test_ratio=0.0)
        source_path = os.path.join(self.tmp_dir, "src", "a.py")
        with open(source_path, "a", encoding="utf-8") as f:
            f.write("\n# changed after manifest creation\n")
        with self.assertRaisesRegex(ValueError, "Manifest source changed"):
            build_training_text(manifest, split="train")

    # --- Generated lockfile exclusion ---------------------------------------
    def test_is_generated_lockfile_matches_known_basenames(self):
        self.assertTrue(is_generated_lockfile("/some/project/package-lock.json"))
        self.assertTrue(is_generated_lockfile("project\\yarn.lock"))
        self.assertTrue(is_generated_lockfile("pnpm-lock.yaml"))
        self.assertTrue(is_generated_lockfile("PACKAGE-LOCK.JSON"))  # case-insensitive

    def test_is_generated_lockfile_does_not_match_normal_config(self):
        # Only an exact basename match counts - a config file that merely
        # contains "lock" in its name or content must never be caught.
        self.assertFalse(is_generated_lockfile("package.json"))
        self.assertFalse(is_generated_lockfile("jsconfig.json"))
        self.assertFalse(is_generated_lockfile("tsconfig.json"))
        self.assertFalse(is_generated_lockfile("lockScreenConfig.json"))
        self.assertFalse(is_generated_lockfile("app-lock-settings.json"))

    def test_manifest_rejects_package_lock_json(self):
        with open(os.path.join(self.tmp_dir, "src", "package-lock.json"), "w", encoding="utf-8") as f:
            f.write('{\n  "name": "demo",\n  "lockfileVersion": 3,\n  "packages": {}\n}\n' * 5)
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/package-lock.json", rels)
        self.assertEqual(manifest.num_lockfiles_rejected, 1)

    def test_manifest_accepts_normal_package_json(self):
        with open(os.path.join(self.tmp_dir, "src", "package.json"), "w", encoding="utf-8") as f:
            f.write('{\n  "name": "demo",\n  "version": "1.0.0",\n  "scripts": {"start": "node index.js"}\n}\n')
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertIn("src/package.json", rels)
        self.assertEqual(manifest.num_lockfiles_rejected, 0)

    def test_manifest_accepts_useful_json_config_files(self):
        # tsconfig.json is a good example of a hand-authored, useful JSON
        # config file that must remain fully eligible for training.
        with open(os.path.join(self.tmp_dir, "src", "tsconfig.json"), "w", encoding="utf-8") as f:
            f.write('{\n  "compilerOptions": {\n    "target": "es2020",\n    "module": "esnext"\n  }\n}\n')
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/tsconfig.json", rels)  # rejected, but by the metadata filter...
        self.assertIn("filter_no_serialized_metadata", manifest.filtered_reasons)
        self.assertEqual(manifest.num_lockfiles_rejected, 0)  # ...never by the lockfile filter

    def test_jsconfig_json_target_key_is_a_documented_metadata_filter_false_positive(self):
        # jsconfig.json's "target": "..." is a TypeScript/JS compiler option
        # (ECMAScript target version), not dataset-record metadata. The
        # metadata filter is deliberately conservative about ANY quoted
        # "target"/"prompt"/"completion"/"category"/"metadata" key, so this
        # file is rejected as a known, acceptable trade-off - documented and
        # tested here rather than left as an unexplained surprise. It is
        # never touched by the lockfile filter, which is basename-only.
        with open(os.path.join(self.tmp_dir, "src", "jsconfig.json"), "w", encoding="utf-8") as f:
            f.write('{\n  "compilerOptions": {\n    "target": "ES2020"\n  }\n}\n')
        manifest = build_manifest(self.tmp_dir, seed=42)
        rels = [f["relative_path"] for f in manifest.files]
        self.assertNotIn("src/jsconfig.json", rels)
        self.assertIn("filter_no_serialized_metadata", manifest.filtered_reasons)
        self.assertEqual(manifest.num_lockfiles_rejected, 0)

    def test_generated_lockfiles_never_enter_lm_training_text(self):
        with open(os.path.join(self.tmp_dir, "src", "package-lock.json"), "w", encoding="utf-8") as f:
            f.write('{\n  "name": "unique-lockfile-marker-xyz",\n  "lockfileVersion": 3\n}\n' * 5)
        manifest = build_manifest(self.tmp_dir, seed=42, val_ratio=0.0, test_ratio=0.0)
        train_text = build_training_text(manifest, split="train")
        val_text = build_training_text(manifest, split="val")
        self.assertNotIn("unique-lockfile-marker-xyz", train_text)
        self.assertNotIn("unique-lockfile-marker-xyz", val_text)


if __name__ == "__main__":
    unittest.main()
