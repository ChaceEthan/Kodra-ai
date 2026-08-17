import os
import sys
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from agent.tools import KodraAgentTools, ToolStatus, resolve_within_root


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        with open(os.path.join(self.workspace, "hello.py"), "w", encoding="utf-8") as f:
            f.write("print('hello')\n")
        # Explicit flags so these tests don't depend on the real environment
        # or an ambient .env file.
        self.tools = KodraAgentTools(self.workspace, require_approval=True, enable_terminal_tools=False)

    def test_read_file(self):
        result = self.tools.read_file("hello.py")
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn("hello", result.data)

    def test_path_traversal_denied(self):
        result = self.tools.read_file("../../etc/passwd")
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_sibling_prefix_bypass_denied(self):
        # A sibling directory that string-prefixes the workspace root name
        # must NOT be treated as inside the workspace.
        sibling = self.workspace + "-secrets"
        os.makedirs(sibling, exist_ok=True)
        with open(os.path.join(sibling, "secret.txt"), "w", encoding="utf-8") as f:
            f.write("top secret")
        result = resolve_within_root(self.workspace, os.path.join("..", os.path.basename(sibling), "secret.txt"))
        self.assertIsNone(result)

    def test_create_file_requires_approval(self):
        result = self.tools.create_file("new.py", "x = 1")
        self.assertEqual(result.status, ToolStatus.REQUIRES_APPROVAL)
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "new.py")))

    def test_create_file_with_approval_writes(self):
        result = self.tools.create_file("new.py", "x = 1", approved=True)
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "new.py")))

    def test_require_approval_false_skips_gate_for_file_tools(self):
        tools = KodraAgentTools(self.workspace, require_approval=False, enable_terminal_tools=False)
        result = tools.create_file("auto.py", "x = 1")
        self.assertEqual(result.status, ToolStatus.OK)

    def test_run_terminal_disabled_by_default(self):
        result = self.tools.run_terminal("echo hi", approved=True)
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_run_terminal_requires_approval_even_when_require_approval_false(self):
        tools = KodraAgentTools(self.workspace, require_approval=False, enable_terminal_tools=True)
        result = tools.run_terminal("echo hi")
        self.assertEqual(result.status, ToolStatus.REQUIRES_APPROVAL)

    def test_run_terminal_denylist_blocks_destructive_command(self):
        tools = KodraAgentTools(self.workspace, require_approval=True, enable_terminal_tools=True)
        result = tools.run_terminal("rm -rf /", approved=True)
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_search_code_finds_match(self):
        result = self.tools.search_code(r"print\(")
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(len(result.data), 1)

    # --- apply_patch -------------------------------------------------------
    def test_apply_patch_dry_run_does_not_write(self):
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('hello world')\n"
        )
        result = self.tools.apply_patch(diff, dry_run=True)
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn("print('hello world')", result.data["preview"]["hello.py"])
        with open(os.path.join(self.workspace, "hello.py"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "print('hello')\n")

    def test_apply_patch_requires_approval(self):
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('hello world')\n"
        )
        result = self.tools.apply_patch(diff)
        self.assertEqual(result.status, ToolStatus.REQUIRES_APPROVAL)

    def test_apply_patch_with_approval_writes_and_returns_backup(self):
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('hello world')\n"
        )
        result = self.tools.apply_patch(diff, approved=True)
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.data["backups"]["hello.py"], "print('hello')\n")
        with open(os.path.join(self.workspace, "hello.py"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "print('hello world')\n")

    def test_apply_patch_reverts_from_backup(self):
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('hello world')\n"
        )
        result = self.tools.apply_patch(diff, approved=True)
        original = result.data["backups"]["hello.py"]
        revert = self.tools.edit_file("hello.py", original, approved=True)
        self.assertEqual(revert.status, ToolStatus.OK)
        with open(os.path.join(self.workspace, "hello.py"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "print('hello')\n")

    def test_apply_patch_rejects_context_mismatch(self):
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1 +1 @@\n"
            "-this line does not exist in the file\n"
            "+print('hacked')\n"
        )
        result = self.tools.apply_patch(diff, approved=True)
        self.assertEqual(result.status, ToolStatus.ERROR)

    def test_apply_patch_rejects_target_outside_workspace(self):
        diff = (
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        result = self.tools.apply_patch(diff, approved=True)
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_apply_patch_missing_target_file(self):
        diff = (
            "--- a/does_not_exist.py\n"
            "+++ b/does_not_exist.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        result = self.tools.apply_patch(diff, approved=True)
        self.assertEqual(result.status, ToolStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
