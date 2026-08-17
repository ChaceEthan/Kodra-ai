import os
import sys
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from agent.tools import KodraAgentTools, ToolStatus


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        with open(os.path.join(self.workspace, "hello.py"), "w", encoding="utf-8") as f:
            f.write("print('hello')\n")
        self.tools = KodraAgentTools(self.workspace)

    def test_read_file(self):
        result = self.tools.read_file("hello.py")
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn("hello", result.data)

    def test_path_traversal_denied(self):
        result = self.tools.read_file("../../etc/passwd")
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_create_file_requires_approval(self):
        result = self.tools.create_file("new.py", "x = 1")
        self.assertEqual(result.status, ToolStatus.REQUIRES_APPROVAL)
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "new.py")))

    def test_create_file_with_approval_writes(self):
        result = self.tools.create_file("new.py", "x = 1", approved=True)
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "new.py")))

    def test_run_terminal_denylist_blocks_destructive_command(self):
        result = self.tools.run_terminal("rm -rf /", approved=True)
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_search_code_finds_match(self):
        result = self.tools.search_code(r"print\(")
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(len(result.data), 1)


if __name__ == "__main__":
    unittest.main()
